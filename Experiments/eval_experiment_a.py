"""Experiment A part 2: AUC delta from domain-matched beta.

Once `experiment_a_beta_calibration.py` has produced per-domain beta values
and Experiment B scoring has saved three-mode aggregates per example /
candidate / span, this script recomputes the posterior-weighted score at
each domain's beta and reports AUC vs the beta=1 baseline.

Math
----
For a per-token posterior anomaly score
    posterior(t) = -(log P_LLM(t) + beta * log G_RI(t))
and the saved aggregates
    log_delta_mean   = mean(log P_LLM - log G_RI)
    neg_log_g_mean   = mean(-log G_RI)
the posterior aggregate at any beta is
    posterior_mean(beta) = -log_delta_mean + (1 + beta) * neg_log_g_mean
For the all-tokens mode this uses `all_mean_log_delta` and `all_mean_neg_log_g`.
For the entity-level mode it uses `entity_mean_log_delta` and
`rank_only_mean_neg_log_g` (the entity-restricted -log G_RI).

Domain mapping
--------------
FRANK (summarisation) -> news beta
TruthfulQA            -> social_media beta (proxy for general/creative)
HaluEval qa           -> social_media beta (general)
HaluEval dialogue     -> social_media beta (conversational)
HaluEval summarization-> news beta

Usage:
    python eval_experiment_a.py --beta-source exp02_beta_calibration/data/beta_calibration_adrian_llama.json
    python eval_experiment_a.py --out results_experiment_a.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

from shared.utils import compute_roc_auc  # noqa: E402


# ---------------------------------------------------------------------------
# Posterior recomputation
# ---------------------------------------------------------------------------


def posterior_at_beta(mean_log_delta: float, mean_neg_log_g: float, beta: float) -> float:
    """posterior_mean(beta) = -mean_log_delta + (1 + beta) * mean_neg_log_g"""
    return -mean_log_delta + (1.0 + beta) * mean_neg_log_g


def safe_auc(labels: list[int], scores: list[float]) -> float:
    lbl = np.asarray(labels, dtype=int)
    scr = np.asarray(scores, dtype=float)
    if len(np.unique(lbl)) < 2 or len(scr) == 0:
        return float("nan")
    return float(compute_roc_auc(lbl, scr))


# ---------------------------------------------------------------------------
# FRANK
# ---------------------------------------------------------------------------


def frank_auc_at_betas(frank_path: Path, betas: list[float]) -> dict:
    """For FRANK, recompute entity-level posterior AUC at each beta.

    Posterior at beta = -entity_mean_log_delta + (1+beta) * rank_only_mean_neg_log_g_global
    """
    from exp06_frank.src.score_examples import load_scored_spans

    spans = load_scored_spans(frank_path)
    if not spans:
        return {"n_spans": 0}

    # Filter to spans with at least one entity token (otherwise entity score is 0)
    rows = []
    for s in spans:
        if s.entity_n_tokens == 0:
            continue
        rows.append({
            "is_error": int(s.is_error),
            "error_type": s.error_type,
            "entity_log_delta": s.entity_mean_log_delta,
            "entity_neg_log_g": s.rank_only_mean_neg_log_g_global,
        })

    if not rows:
        return {"n_spans": 0}

    by_beta = {}
    for beta in betas:
        scores = [posterior_at_beta(r["entity_log_delta"], r["entity_neg_log_g"], beta)
                  for r in rows]
        labels = [r["is_error"] for r in rows]
        by_beta[f"{beta:.4f}"] = safe_auc(labels, scores)

    return {
        "n_spans_with_entities": len(rows),
        "auc_by_beta": by_beta,
    }


# ---------------------------------------------------------------------------
# HaluEval
# ---------------------------------------------------------------------------


def halueval_auc_at_betas(data_dir: Path, model_name: str, betas: list[float]) -> dict:
    from exp04_halueval.src.score_examples import load_scored_examples

    files = sorted(Path(data_dir).glob(f"scored_{model_name}_*.jsonl"))
    if not files:
        return {"n_examples": 0}

    rows = []
    by_task = {}
    for f in files:
        examples = load_scored_examples(f)
        for ex in examples:
            tm = ex.three_mode or {}
            if tm.get("entity_n_tokens", 0) == 0:
                continue
            row = {
                "label": int(ex.label),
                "task": ex.task,
                "log_delta": tm.get("entity_mean_log_delta", 0.0),
                "neg_log_g": tm.get("rank_only_mean_neg_log_g", 0.0),
            }
            rows.append(row)
            by_task.setdefault(ex.task, []).append(row)

    def _auc_at(rows_, beta):
        if not rows_:
            return float("nan")
        scores = [posterior_at_beta(r["log_delta"], r["neg_log_g"], beta) for r in rows_]
        labels = [r["label"] for r in rows_]
        return safe_auc(labels, scores)

    return {
        "n_examples_with_entities": len(rows),
        "auc_by_beta_overall": {f"{b:.4f}": _auc_at(rows, b) for b in betas},
        "auc_by_beta_per_task": {
            task: {f"{b:.4f}": _auc_at(task_rows, b) for b in betas}
            for task, task_rows in by_task.items()
        },
    }


# ---------------------------------------------------------------------------
# TruthfulQA
# ---------------------------------------------------------------------------


def truthfulqa_auc_at_betas(scored_path: Path, betas: list[float]) -> dict:
    from exp05_truthfulqa.src.score_examples import load_scored_questions

    qs = load_scored_questions(scored_path)
    if not qs:
        return {"n_candidates": 0}

    rows = []
    for q in qs:
        for cand_idx, tm in enumerate(q.mc1_three_mode):
            if tm.get("entity_n_tokens", 0) == 0:
                continue
            rows.append({
                "label": 0 if cand_idx == q.mc1_correct_idx else 1,
                "log_delta": tm.get("entity_mean_log_delta", 0.0),
                "neg_log_g": tm.get("rank_only_mean_neg_log_g", 0.0),
            })

    def _auc_at(rows_, beta):
        if not rows_:
            return float("nan")
        scores = [posterior_at_beta(r["log_delta"], r["neg_log_g"], beta) for r in rows_]
        labels = [r["label"] for r in rows_]
        return safe_auc(labels, scores)

    return {
        "n_candidates_with_entities": len(rows),
        "auc_by_beta": {f"{b:.4f}": _auc_at(rows, b) for b in betas},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# Default benchmark -> domain mapping (used to pick a beta)
DEFAULT_BENCH_TO_DOMAIN = {
    "frank": "news",
    "truthfulqa": "social_media",  # general/creative proxy
    "halueval_qa": "social_media",
    "halueval_dialogue": "social_media",
    "halueval_summarization": "news",
}


def main():
    parser = argparse.ArgumentParser(description="Experiment A AUC evaluation at domain-matched beta")
    parser.add_argument("--beta-source",
                        default=str(EXP_DIR / "exp02_beta_calibration" / "data" / "beta_calibration_adrian_llama.json"))
    parser.add_argument("--model", default="llama-3.1-8b")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    beta_data = json.load(open(args.beta_source))
    domain_betas = {dom: stats["beta"] for dom, stats in beta_data["results"].items()}
    print(f"Per-domain beta values (from {Path(args.beta_source).name}):")
    for dom, b in sorted(domain_betas.items(), key=lambda kv: -kv[1]):
        print(f"  {dom:<15}  beta = {b:.4f}")

    # Always include beta=1 as a reference
    all_betas = sorted(set([1.0] + list(domain_betas.values())))

    results = {"betas_evaluated": all_betas, "domain_betas": domain_betas, "benchmarks": {}}

    frank_path = EXP_DIR / "exp06_frank" / "output" / f"scored_frank_{args.model}.jsonl"
    if frank_path.exists():
        results["benchmarks"]["frank"] = frank_auc_at_betas(frank_path, all_betas)

    halueval_dir = EXP_DIR / "exp04_halueval" / "data" / args.model
    if halueval_dir.exists():
        results["benchmarks"]["halueval"] = halueval_auc_at_betas(halueval_dir, args.model, all_betas)

    tqa_path = EXP_DIR / "exp05_truthfulqa" / "results" / args.model / f"scored_{args.model}.json"
    if tqa_path.exists():
        results["benchmarks"]["truthfulqa"] = truthfulqa_auc_at_betas(tqa_path, all_betas)

    print()
    print("=== AUC at each beta ===")
    for bench, data in results["benchmarks"].items():
        print(f"\n{bench}:")
        if "auc_by_beta" in data:
            for beta_str, auc in data["auc_by_beta"].items():
                print(f"  beta={beta_str:>8s}  AUC={auc:.4f}  n={data.get('n_spans_with_entities', data.get('n_candidates_with_entities', 0))}")
        elif "auc_by_beta_overall" in data:
            print("  overall:")
            for beta_str, auc in data["auc_by_beta_overall"].items():
                print(f"    beta={beta_str:>8s}  AUC={auc:.4f}")
            for task, by_b in data["auc_by_beta_per_task"].items():
                print(f"  task={task}:")
                for beta_str, auc in by_b.items():
                    print(f"    beta={beta_str:>8s}  AUC={auc:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
