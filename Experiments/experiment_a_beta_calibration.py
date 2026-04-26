"""Experiment A: Domain beta calibration (Adrian review).

Computes per-domain beta = 1/sigma^2(Delta_r) on held-out domain corpora
using the Llama 3.1 8B BPE tokenizer and the Wikipedia global rank table.

Δr = log2(r_global / r_local) is computed PER TOKEN OCCURRENCE (every token
in every document, weighted by frequency), not per unique token type. This
matches Adrian's spec and gives a frequency-weighted precision estimate
consistent with how the scoring primitive aggregates per-token deltas.

Adrian's prediction (from email):
    creative/general -- highest variance (lowest beta)
    legal / biomedical -- lowest variance (highest beta)

Per Experiment A step 6, the resulting per-domain beta values are then
used to re-score FRANK / TruthfulQA / HaluEval and the AUC delta vs beta=1
is reported.

Usage:
    cd Experiments
    python experiment_a_beta_calibration.py                   # default 500 docs/domain
    python experiment_a_beta_calibration.py --max-docs 1000
    python experiment_a_beta_calibration.py --domains news,biomedical
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

# HF hub has been returning 503/504 on HEAD checks today.  The tokenizer
# is already cached locally; we need datasets to stream from HF.
# Strategy: load the tokenizer in offline mode (skips the HEAD check),
# THEN unset offline so the datasets library can fetch corpora.

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

from shared.utils import (
    DOMAIN_CORPORA,
    REFERENCE_CORPUS,
    RankTable,
    get_tokenizer,
    load_corpus_texts,
    tokenize_text,
)

DOMAINS_DEFAULT = ["news", "biomedical", "legal", "code", "social_media"]
TOKENIZER_NAME = "llama-3.1-8b"

RANK_TABLE_DIR = EXP_DIR / "shared" / "rank_tables"
RESULTS_DIR = EXP_DIR / "exp02_beta_calibration" / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-occurrence Δr statistics
# ---------------------------------------------------------------------------


def compute_per_occurrence_stats(
    token_ids: list[int],
    global_rank_table: RankTable,
) -> dict:
    """Compute per-token-occurrence Δr statistics for a domain corpus.

    Δr[i] = log2(r_global(t) / r_local(t)) at every occurrence i.
    """
    if not token_ids:
        return _empty_stats()

    freq = Counter(token_ids)
    # Local rank: 1 = most frequent
    sorted_items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    local_rank = {tid: i + 1 for i, (tid, _) in enumerate(sorted_items)}

    n = len(token_ids)
    deltas = np.empty(n, dtype=np.float64)
    unseen_count = 0
    for i, tid in enumerate(token_ids):
        g = global_rank_table.get_rank(tid)  # vocab_size+1 for unseen
        if g > global_rank_table.vocab_size:
            unseen_count += 1
        l = local_rank[tid]
        deltas[i] = np.log2(g / max(l, 1))

    mean_d = float(np.mean(deltas))
    var_d = float(np.var(deltas))
    beta = 1.0 / var_d if var_d > 0 else float("inf")

    return {
        "n_occurrences": n,
        "n_unique": len(freq),
        "n_unseen_in_global": unseen_count,
        "frac_unseen": unseen_count / n if n > 0 else 0.0,
        "mean_delta_r": mean_d,
        "var_delta_r": var_d,
        "std_delta_r": float(np.std(deltas)),
        "beta": beta,
        "median_delta_r": float(np.median(deltas)),
        "percentiles": {
            "p5": float(np.percentile(deltas, 5)),
            "p25": float(np.percentile(deltas, 25)),
            "p50": float(np.percentile(deltas, 50)),
            "p75": float(np.percentile(deltas, 75)),
            "p95": float(np.percentile(deltas, 95)),
        },
        "histogram": {
            "bin_edges": np.histogram_bin_edges(deltas, bins=60).tolist(),
            "counts": np.histogram(deltas, bins=60)[0].tolist(),
        },
    }


def _empty_stats() -> dict:
    return {
        "n_occurrences": 0,
        "n_unique": 0,
        "n_unseen_in_global": 0,
        "frac_unseen": 0.0,
        "mean_delta_r": 0.0,
        "var_delta_r": 0.0,
        "std_delta_r": 0.0,
        "beta": float("inf"),
        "median_delta_r": 0.0,
        "percentiles": {"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0},
        "histogram": {"bin_edges": [], "counts": []},
    }


# ---------------------------------------------------------------------------
# Per-domain tokenization
# ---------------------------------------------------------------------------


def tokenize_domain(
    domain: str,
    tokenizer,
    tokenizer_name: str,
    max_docs: int,
) -> list[int]:
    """Stream domain corpus, tokenize each doc, return flat token list."""
    config = DOMAIN_CORPORA[domain]
    tokens: list[int] = []
    start = time.time()
    n_docs = 0
    try:
        for doc in load_corpus_texts(config, max_docs=max_docs):
            tokens.extend(tokenize_text(doc, tokenizer, tokenizer_name))
            n_docs += 1
    except Exception as e:
        print(f"    corpus stream error on '{domain}' after {n_docs} docs: {e}")
    dt = time.time() - start
    print(f"    tokenized {n_docs} docs ({len(tokens):,} tokens) in {dt:.1f}s")
    return tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def per_occurrence_stats_from_rank_table(
    local_rt: RankTable,
    global_rt: RankTable,
) -> dict:
    """Compute per-occurrence Δr stats directly from a cached domain rank table.

    Uses frequency weights from the rank table (no corpus retokenisation needed).
    For each unique token t with local rank l and frequency f, its Δr =
    log2(g_global(t) / l) contributes f times to the variance.
    """
    if not local_rt.token_to_rank:
        return _empty_stats()

    types = []
    weights = []
    deltas = []
    unseen = 0
    for token_id, l_rank in local_rt.token_to_rank.items():
        f = local_rt.token_to_freq.get(token_id, 0)
        if f <= 0:
            continue
        g_rank = global_rt.get_rank(token_id)
        if g_rank > global_rt.vocab_size:
            unseen += f
        types.append(token_id)
        weights.append(f)
        deltas.append(np.log2(g_rank / max(l_rank, 1)))

    if not deltas:
        return _empty_stats()

    deltas_arr = np.array(deltas)
    weights_arr = np.array(weights, dtype=np.float64)
    n_occ = int(weights_arr.sum())

    mean_d = float(np.average(deltas_arr, weights=weights_arr))
    var_d = float(np.average((deltas_arr - mean_d) ** 2, weights=weights_arr))
    beta = 1.0 / var_d if var_d > 0 else float("inf")

    # Frequency-weighted percentile estimates (sort by delta, then cumulative weight)
    order = np.argsort(deltas_arr)
    sorted_d = deltas_arr[order]
    sorted_w = weights_arr[order]
    cumw = np.cumsum(sorted_w)
    cumw /= cumw[-1]

    def _percentile(p: float) -> float:
        idx = int(np.searchsorted(cumw, p / 100.0))
        idx = min(idx, len(sorted_d) - 1)
        return float(sorted_d[idx])

    return {
        "n_occurrences": n_occ,
        "n_unique": len(deltas),
        "n_unseen_in_global": int(unseen),
        "frac_unseen": float(unseen / n_occ) if n_occ > 0 else 0.0,
        "mean_delta_r": mean_d,
        "var_delta_r": var_d,
        "std_delta_r": float(np.sqrt(var_d)),
        "beta": beta,
        "median_delta_r": _percentile(50),
        "percentiles": {
            "p5": _percentile(5),
            "p25": _percentile(25),
            "p50": _percentile(50),
            "p75": _percentile(75),
            "p95": _percentile(95),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment A (domain beta calibration)")
    parser.add_argument("--max-docs", type=int, default=500,
                        help="Max docs per domain (default 500); only used when fresh tokenisation is possible")
    parser.add_argument("--domains", default=",".join(DOMAINS_DEFAULT),
                        help="Comma-separated domains")
    parser.add_argument("--source", choices=["fresh", "cached"], default="cached",
                        help="'cached' uses pre-built domain rank tables (no corpus download); 'fresh' re-tokenises")
    parser.add_argument("--out", default=str(RESULTS_DIR / "beta_calibration_adrian_llama.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    # Global rank table (Wikipedia, Llama tokenizer)
    global_path = RANK_TABLE_DIR / f"wikipedia_full_{TOKENIZER_NAME}.json"
    if not global_path.exists():
        print(f"ERROR: global rank table not found at {global_path}")
        sys.exit(1)
    print(f"Loading global rank table: {global_path.name}")
    global_rt = RankTable.load(global_path)
    print(f"  vocab_size={global_rt.vocab_size:,}, total_tokens={global_rt.total_tokens:,}")

    results: dict[str, dict] = {}

    if args.source == "cached":
        print("Using cached per-domain rank tables (frequency-weighted variance).")
        for domain in domains:
            print(f"\n=== {domain} ===")
            cache_path = RANK_TABLE_DIR / f"{TOKENIZER_NAME}_{domain}.json"
            if not cache_path.exists():
                print(f"  no cached rank table at {cache_path}; skipping")
                continue
            print(f"  loading {cache_path.name}")
            local_rt = RankTable.load(cache_path)
            stats = per_occurrence_stats_from_rank_table(local_rt, global_rt)
            results[domain] = stats
            print(f"  n_occurrences = {stats['n_occurrences']:,}")
            print(f"  n_unique      = {stats['n_unique']:,}")
            print(f"  mean dr       = {stats['mean_delta_r']:+.4f}")
            print(f"  sigma^2(dr)   = {stats['var_delta_r']:.4f}")
            print(f"  beta = 1/s2   = {stats['beta']:.4f}")
            print(f"  frac unseen   = {stats['frac_unseen']:.2%}")
    else:
        print(f"Loading tokenizer: {TOKENIZER_NAME} (offline, from local cache)")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        tokenizer = get_tokenizer(TOKENIZER_NAME)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        try:
            import huggingface_hub.constants as hf_const
            import importlib
            importlib.reload(hf_const)
        except Exception:
            pass
        print("  tokenizer loaded; HF online for datasets")

        for domain in domains:
            print(f"\n=== {domain} ===")
            if domain not in DOMAIN_CORPORA:
                print(f"  unknown domain {domain!r}; skipping")
                continue

            token_ids = tokenize_domain(domain, tokenizer, TOKENIZER_NAME, args.max_docs)
            if not token_ids:
                print(f"  no tokens; skipping stats")
                continue

            stats = compute_per_occurrence_stats(token_ids, global_rt)
            results[domain] = stats

            print(f"  n_occurrences = {stats['n_occurrences']:,}")
            print(f"  n_unique      = {stats['n_unique']:,}")
            print(f"  mean dr       = {stats['mean_delta_r']:+.4f}")
            print(f"  sigma^2(dr)   = {stats['var_delta_r']:.4f}")
            print(f"  beta = 1/s2   = {stats['beta']:.4f}")
            print(f"  frac unseen   = {stats['frac_unseen']:.2%}")

    # Sort beta descending (highest precision first) for the printed summary
    print("\n" + "=" * 72)
    print(f"{'Domain':<15} {'n_occ':>10} {'n_uniq':>8} {'mean_dr':>10} {'sigma2':>8} {'beta':>8}")
    print("=" * 72)
    for domain, stats in sorted(results.items(), key=lambda kv: -kv[1]["beta"]):
        print(f"{domain:<15} "
              f"{stats['n_occurrences']:>10,} "
              f"{stats['n_unique']:>8,} "
              f"{stats['mean_delta_r']:>+10.4f} "
              f"{stats['var_delta_r']:>8.4f} "
              f"{stats['beta']:>8.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "tokenizer": TOKENIZER_NAME,
            "max_docs": args.max_docs,
            "per_occurrence": True,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
