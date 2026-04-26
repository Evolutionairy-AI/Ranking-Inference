"""
Step 1 (Exp02): Compute rank deviations Delta_r for each domain corpus.

For each domain, tokenize reference corpus text, compute local ranks,
and measure Delta_r = log2(r_global / r_local) against Wikipedia global baseline.
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import (
    DOMAIN_CORPORA,
    REFERENCE_CORPUS,
    get_tokenizer,
    tokenize_corpus,
    build_rank_table,
    compute_rank_deviations,
    RankTable,
)

EXP_DIR = Path(__file__).resolve().parent.parent
RANK_TABLE_DIR = PROJECT_ROOT / "shared" / "rank_tables"
DATA_DIR = EXP_DIR / "data"


def build_global_reference(tokenizer, tokenizer_name: str, max_docs: int = 50000) -> RankTable:
    """Build global rank table from Wikipedia reference corpus."""
    cache_path = RANK_TABLE_DIR / f"wikipedia_{tokenizer_name}.json"

    if cache_path.exists():
        print(f"  Loading cached Wikipedia rank table: {cache_path}")
        return RankTable.load(cache_path)

    print(f"  Building Wikipedia rank table ({max_docs} docs)...")
    token_ids = tokenize_corpus(
        REFERENCE_CORPUS, tokenizer, tokenizer_name, max_docs=max_docs
    )
    rt = build_rank_table(token_ids, tokenizer_name, "wikipedia")
    rt.save(cache_path)
    print(f"  Saved: {cache_path} ({rt.total_tokens:,} tokens, {rt.vocab_size:,} unique)")
    return rt


def compute_domain_deviations(
    domain: str,
    tokenizer,
    tokenizer_name: str,
    global_rank_table: RankTable,
    max_docs: int = 10000,
) -> dict:
    """Compute rank deviation statistics for one domain.

    Returns:
        dict with deviations array, sigma^2, beta, and descriptive stats
    """
    corpus_config = DOMAIN_CORPORA[domain]

    # Check for cached domain rank table
    cache_path = RANK_TABLE_DIR / f"{domain}_{tokenizer_name}.json"
    if cache_path.exists():
        print(f"    Loading cached {domain} rank table")
        local_rt = RankTable.load(cache_path)
        # Recompute deviations from cached table
        deviations = []
        for token_id, local_rank in local_rt.token_to_rank.items():
            global_rank = global_rank_table.get_rank(token_id)
            if global_rank > 0 and global_rank <= global_rank_table.vocab_size:
                delta_r = np.log2(global_rank / local_rank)
                deviations.append(delta_r)
        deviations = np.array(deviations)
    else:
        print(f"    Tokenizing {domain} corpus...")
        token_ids = tokenize_corpus(
            corpus_config, tokenizer, tokenizer_name, max_docs=max_docs
        )

        # Build local rank table
        local_rt = build_rank_table(token_ids, tokenizer_name, domain)
        local_rt.save(cache_path)

        # Compute rank deviations
        deviations = compute_rank_deviations(token_ids, global_rank_table)

    # Compute statistics
    sigma2 = float(np.var(deviations))
    beta = 1.0 / sigma2 if sigma2 > 0 else float("inf")

    return {
        "domain": domain,
        "tokenizer": tokenizer_name,
        "n_unique_tokens": len(deviations),
        "total_tokens": local_rt.total_tokens,
        "sigma2_delta_r": sigma2,
        "beta": beta,
        "mean_delta_r": float(np.mean(deviations)),
        "std_delta_r": float(np.std(deviations)),
        "median_delta_r": float(np.median(deviations)),
        "skew_delta_r": float(
            np.mean(((deviations - deviations.mean()) / deviations.std()) ** 3)
        ) if deviations.std() > 0 else 0.0,
        "percentiles": {
            "p5": float(np.percentile(deviations, 5)),
            "p25": float(np.percentile(deviations, 25)),
            "p50": float(np.percentile(deviations, 50)),
            "p75": float(np.percentile(deviations, 75)),
            "p95": float(np.percentile(deviations, 95)),
        },
        "deviations_histogram": {
            "bin_edges": np.histogram_bin_edges(deviations, bins=50).tolist(),
            "counts": np.histogram(deviations, bins=50)[0].tolist(),
        },
    }


def bootstrap_beta_ci(
    deviations: np.ndarray,
    n_iterations: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    """Bootstrap confidence interval for beta = 1/sigma^2."""
    betas = []
    n = len(deviations)
    for _ in range(n_iterations):
        sample = np.random.choice(deviations, size=n, replace=True)
        s2 = np.var(sample)
        if s2 > 0:
            betas.append(1.0 / s2)

    alpha = (1 - ci_level) / 2
    return {
        "mean": float(np.mean(betas)),
        "ci_lower": float(np.percentile(betas, 100 * alpha)),
        "ci_upper": float(np.percentile(betas, 100 * (1 - alpha))),
        "std": float(np.std(betas)),
    }


def main():
    RANK_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    domains = list(DOMAIN_CORPORA.keys())
    tokenizer_names = ["gpt-5.1"]  # Start with one; extend to all 3 later

    all_results = {}

    for tok_name in tokenizer_names:
        print(f"\n=== Tokenizer: {tok_name} ===")
        tokenizer = get_tokenizer(tok_name)

        # Build global reference
        global_rt = build_global_reference(tokenizer, tok_name)

        for domain in domains:
            print(f"\n  --- {domain} ---")
            try:
                result = compute_domain_deviations(
                    domain, tokenizer, tok_name, global_rt
                )
                key = f"{domain}_{tok_name}"
                all_results[key] = result

                print(f"    sigma^2 = {result['sigma2_delta_r']:.4f}")
                print(f"    beta   = {result['beta']:.4f}")
                print(f"    mean   = {result['mean_delta_r']:.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                print(f"    Skipping {domain}, continuing...")

    # Save results
    results_path = DATA_DIR / "rank_deviation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Domain':<15} {'sigma^2':>10} {'beta':>10} {'mean_dr':>10} {'n_tokens':>10}")
    print("=" * 70)
    for key, result in sorted(all_results.items()):
        print(
            f"{result['domain']:<15} "
            f"{result['sigma2_delta_r']:>10.4f} "
            f"{result['beta']:>10.4f} "
            f"{result['mean_delta_r']:>10.4f} "
            f"{result['total_tokens']:>10,}"
        )


if __name__ == "__main__":
    main()
