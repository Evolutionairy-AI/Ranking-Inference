"""End-to-end entity-level FRANK scoring + AUC (Experiment D).

Usage:
    python Experiments/exp06_frank/run_entities.py \
        --model llama-3.1-8b --max-examples 500

Outputs:
    Experiments/exp06_frank/output/scored_frank_entities_<model>.jsonl
    Experiments/exp06_frank/results/entity_auc_<model>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp06_frank.src.load_dataset import load_frank
from exp06_frank.src.score_entities import (
    FRANKScoredEntity,
    load_scored_entities,
    score_dataset_entities,
)


def _find_rank_table(model_name: str) -> Path:
    rank_dir = PROJECT_ROOT / "shared" / "rank_tables"
    path = rank_dir / f"wikipedia_full_{model_name}.json"
    if path.exists():
        return path
    for candidate in rank_dir.glob("wikipedia_*.json"):
        stem = candidate.stem
        if model_name in stem or model_name.replace("-", "_") in stem:
            return candidate
    raise FileNotFoundError(
        f"No Wikipedia rank table for {model_name} in {rank_dir}"
    )


def _auc_or_nan(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    mask = ~np.isnan(y_score)
    if mask.sum() < 2:
        return float("nan")
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_entity_aucs(entities: list[FRANKScoredEntity]) -> dict:
    """Compute AUC of each feature for the fabricated/grounded classification.

    Two label definitions (strict, relaxed) and three entity cohorts
    (all, named-only, person-org-gpe).
    """
    results: dict = {"n_entities_total": len(entities)}

    cohorts = {
        "all": lambda e: True,
        "named": lambda e: e.is_named,
        "person_org_gpe": lambda e: e.entity_type in {"PERSON", "ORG", "GPE"},
    }
    features = [
        "mean_log_delta",
        "mean_log_delta_source",
        "mean_neg_log_g_global",
        "mean_neg_log_g_source",
        "mean_log_rank_global",
        "mean_rank_deviation",
    ]
    labels = ["fabricated_strict", "fabricated_relaxed"]

    for cohort_name, predicate in cohorts.items():
        cohort = [e for e in entities if predicate(e)]
        if not cohort:
            results[cohort_name] = {"n": 0}
            continue
        n = len(cohort)
        cohort_summary = {
            "n": n,
            "n_fabricated_strict": int(sum(1 for e in cohort if e.fabricated_strict)),
            "n_fabricated_relaxed": int(sum(1 for e in cohort if e.fabricated_relaxed)),
            "auc": {},
        }
        for label_name in labels:
            y_true = np.array([int(getattr(e, label_name)) for e in cohort])
            cohort_summary["auc"][label_name] = {}
            for feat in features:
                y_score = np.array([getattr(e, feat) for e in cohort])
                auc = _auc_or_nan(y_true, y_score)
                cohort_summary["auc"][label_name][feat] = auc

        # Entity-type breakdown (top 6 types by count)
        types = {}
        for e in cohort:
            types.setdefault(e.entity_type, []).append(e)
        type_breakdown = {}
        for etype, subset in sorted(types.items(), key=lambda kv: -len(kv[1]))[:8]:
            if len(subset) < 10:
                continue
            y_true = np.array([int(e.fabricated_strict) for e in subset])
            y_score = np.array([e.mean_log_delta for e in subset])
            type_breakdown[etype] = {
                "n": len(subset),
                "n_fabricated": int(y_true.sum()),
                "auc_mean_log_delta_strict": _auc_or_nan(y_true, y_score),
            }
        cohort_summary["by_entity_type"] = type_breakdown
        results[cohort_name] = cohort_summary

    return results


def run(
    model_name: str,
    max_examples: int | None,
    evaluate_only: bool,
    data_dir: Path,
    output_dir: Path,
    results_dir: Path,
) -> None:
    output_path = output_dir / f"scored_frank_entities_{model_name}.jsonl"

    if not evaluate_only:
        examples = load_frank(data_dir=data_dir, max_examples=max_examples)
        if not examples:
            print("No FRANK examples loaded.")
            return
        rank_table_path = _find_rank_table(model_name)
        print(f"Using rank table: {rank_table_path}")
        score_dataset_entities(
            examples=examples,
            model_name=model_name,
            global_rank_table_path=str(rank_table_path),
            output_dir=output_dir,
            checkpoint_every=25,
        )

    if not output_path.exists():
        print(f"No entity-level output at {output_path}.")
        return

    entities = load_scored_entities(output_path)
    print(f"Loaded {len(entities)} entity records.")
    aucs = compute_entity_aucs(entities)

    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"entity_auc_{model_name}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(aucs, f, indent=2)
    print(f"Wrote {results_path}")

    for cohort_name, summary in aucs.items():
        if cohort_name == "n_entities_total" or not isinstance(summary, dict):
            continue
        if summary.get("n", 0) == 0:
            continue
        print(f"\n[{cohort_name}] n={summary['n']}, "
              f"fabricated_strict={summary['n_fabricated_strict']}, "
              f"fabricated_relaxed={summary['n_fabricated_relaxed']}")
        for label, feat_aucs in summary["auc"].items():
            print(f"  label={label}")
            for feat, auc in feat_aucs.items():
                print(f"    {feat:35s} {auc:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-3.1-8b")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(__file__).resolve().parent
    data_dir = exp_dir / "data"
    output_dir = exp_dir / "output"
    results_dir = exp_dir / "results"

    run(
        model_name=args.model,
        max_examples=args.max_examples,
        evaluate_only=args.evaluate_only,
        data_dir=data_dir,
        output_dir=output_dir,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
