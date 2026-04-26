"""Experiment B evaluation: three-mode AUC comparison across benchmarks.

Computes AUC-ROC for each scoring mode (output-level, entity-level,
rank-only at entity positions) on FRANK, TruthfulQA, and HaluEval.

Usage:
    python eval_experiment_b.py                 # full run, prints the table
    python eval_experiment_b.py --model llama-3.1-8b
    python eval_experiment_b.py --out results_experiment_b.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

from shared.utils import compute_roc_auc  # noqa: E402


# ---------------------------------------------------------------------------
# AUC helper with direction handling
# ---------------------------------------------------------------------------


def safe_auc(labels: list[int], scores: list[float]) -> float:
    """Compute AUC-ROC, returning NaN if degenerate."""
    lbl = np.asarray(labels, dtype=int)
    scr = np.asarray(scores, dtype=float)
    if len(np.unique(lbl)) < 2 or len(scr) == 0:
        return float("nan")
    return float(compute_roc_auc(lbl, scr))


# ---------------------------------------------------------------------------
# FRANK
# ---------------------------------------------------------------------------


def frank_auc_table(frank_path: Path) -> dict:
    """AUC for FRANK, stratified overall + by error type.

    Score directions used for AUC:
      output_level   : mean_log_delta              (larger = more anomalous)
      entity_level   : entity_mean_log_delta       (larger = more anomalous)
      rank_only_rank : -rank_only_mean_rank_deviation
                        (log2(r_global/r_source_article); fabricated entities
                         have r_source = vocab+1, so Δr goes very negative.
                         Flip sign so larger = more anomalous.)
      rank_only_g    : rank_only_mean_neg_log_g_global (larger = rarer globally)
    """
    from exp06_frank.src.score_examples import load_scored_spans

    spans = load_scored_spans(frank_path)
    if not spans:
        return {"n_spans": 0}

    records = []
    for s in spans:
        records.append({
            "is_error": 1 if s.is_error else 0,
            "error_type": s.error_type,
            "tier": s.tier,
            "n_tokens": s.n_tokens,
            "entity_n_tokens": s.entity_n_tokens,
            "output_level": s.mean_log_delta,
            "entity_level": s.entity_mean_log_delta,
            "rank_only_log_rank": s.rank_only_mean_log_rank,
            "rank_only_neg_log_g": s.rank_only_mean_neg_log_g_global,
            "rank_only_rank_dev": -s.rank_only_mean_rank_deviation,  # flipped
        })

    def _auc_for(rows, key):
        # For entity/rank-only modes, exclude rows with 0 entity tokens
        # (the aggregate is meaningless there).  For output_level we use all.
        if key in ("entity_level", "rank_only_log_rank",
                   "rank_only_neg_log_g", "rank_only_rank_dev"):
            rows = [r for r in rows if r["entity_n_tokens"] > 0]
        if not rows:
            return {"auc": float("nan"), "n": 0}
        labels = [r["is_error"] for r in rows]
        scores = [r[key] for r in rows]
        return {"auc": safe_auc(labels, scores), "n": len(rows)}

    modes = ["output_level", "entity_level",
             "rank_only_log_rank", "rank_only_neg_log_g", "rank_only_rank_dev"]

    result: dict = {
        "n_spans": len(records),
        "overall": {m: _auc_for(records, m) for m in modes},
        "by_error_type": {},
        "by_tier": {},
    }

    # Tier / error-type strata
    # For error-type AUC we pair each error type against ALL controls
    controls = [r for r in records if r["error_type"] == "control"]
    error_types = sorted({r["error_type"] for r in records
                          if r["error_type"] != "control"})
    for et in error_types:
        ethit = [r for r in records if r["error_type"] == et]
        subset = ethit + controls
        result["by_error_type"][et] = {
            m: _auc_for(subset, m) for m in modes
        }
        result["by_error_type"][et]["n_errors"] = len(ethit)
        result["by_error_type"][et]["n_controls"] = len(controls)

    tiers = sorted({r["tier"] for r in records if r["tier"] != "control"})
    for t in tiers:
        thit = [r for r in records if r["tier"] == t]
        subset = thit + controls
        result["by_tier"][t] = {m: _auc_for(subset, m) for m in modes}

    return result


# ---------------------------------------------------------------------------
# HaluEval
# ---------------------------------------------------------------------------


def halueval_auc_table(data_dir: Path, model_name: str) -> dict:
    """AUC for HaluEval per-task and overall.

    Score directions:
      output_level   : three_mode.all_mean_log_delta
      entity_level   : three_mode.entity_mean_log_delta
      rank_only_rank : three_mode.rank_only_mean_log_rank
      rank_only_g    : three_mode.rank_only_mean_neg_log_g
    """
    from exp04_halueval.src.score_examples import load_scored_examples

    files = sorted(Path(data_dir).glob(f"scored_{model_name}_*.jsonl"))
    if not files:
        return {"n_examples": 0}

    all_rows: list[dict] = []
    by_task: dict[str, list[dict]] = {}
    for f in files:
        examples = load_scored_examples(f)
        for ex in examples:
            tm = ex.three_mode or {}
            row = {
                "label": int(ex.label),
                "task": ex.task,
                "n_entities": ex.n_entities,
                "output_level": tm.get("all_mean_log_delta", 0.0),
                "entity_level": tm.get("entity_mean_log_delta", 0.0),
                "rank_only_log_rank": tm.get("rank_only_mean_log_rank", 0.0),
                "rank_only_neg_log_g": tm.get("rank_only_mean_neg_log_g", 0.0),
                "entity_n_tokens": tm.get("entity_n_tokens", 0),
            }
            all_rows.append(row)
            by_task.setdefault(ex.task, []).append(row)

    modes = ["output_level", "entity_level",
             "rank_only_log_rank", "rank_only_neg_log_g"]

    def _auc_for(rows, key):
        if key in ("entity_level", "rank_only_log_rank", "rank_only_neg_log_g"):
            rows = [r for r in rows if r["entity_n_tokens"] > 0]
        if not rows:
            return {"auc": float("nan"), "n": 0}
        labels = [r["label"] for r in rows]
        scores = [r[key] for r in rows]
        return {"auc": safe_auc(labels, scores), "n": len(rows)}

    result: dict = {
        "n_examples": len(all_rows),
        "overall": {m: _auc_for(all_rows, m) for m in modes},
        "by_task": {
            task: {m: _auc_for(rows, m) for m in modes}
            for task, rows in by_task.items()
        },
    }
    return result


# ---------------------------------------------------------------------------
# TruthfulQA
# ---------------------------------------------------------------------------


def truthfulqa_auc_table(scored_path: Path) -> dict:
    """AUC for TruthfulQA MC1 -- does the mode separate correct from incorrect?

    Each question has several candidate answers with one correct.  For AUC
    we treat every candidate as a row (labels: 1 = incorrect/hallucinated,
    0 = correct).  Score direction: higher = more anomalous / less truthful.
    """
    from exp05_truthfulqa.src.score_examples import load_scored_questions

    qs = load_scored_questions(scored_path)
    if not qs:
        return {"n_candidates": 0}

    all_rows: list[dict] = []
    by_tier: dict[str, list[dict]] = {}
    by_category: dict[str, list[dict]] = {}

    for q in qs:
        if not q.mc1_three_mode:
            continue
        for cand_idx, tm in enumerate(q.mc1_three_mode):
            # MC1: one correct, rest incorrect
            is_hallucinated = 0 if cand_idx == q.mc1_correct_idx else 1
            row = {
                "label": is_hallucinated,
                "tier": q.tier,
                "category": q.category,
                "entity_n_tokens": tm.get("entity_n_tokens", 0),
                "output_level": tm.get("all_mean_log_delta", 0.0),
                "entity_level": tm.get("entity_mean_log_delta", 0.0),
                "rank_only_log_rank": tm.get("rank_only_mean_log_rank", 0.0),
                "rank_only_neg_log_g": tm.get("rank_only_mean_neg_log_g", 0.0),
            }
            all_rows.append(row)
            by_tier.setdefault(q.tier, []).append(row)
            by_category.setdefault(q.category, []).append(row)

    modes = ["output_level", "entity_level",
             "rank_only_log_rank", "rank_only_neg_log_g"]

    def _auc_for(rows, key):
        if key in ("entity_level", "rank_only_log_rank", "rank_only_neg_log_g"):
            rows = [r for r in rows if r["entity_n_tokens"] > 0]
        if not rows:
            return {"auc": float("nan"), "n": 0}
        labels = [r["label"] for r in rows]
        scores = [r[key] for r in rows]
        return {"auc": safe_auc(labels, scores), "n": len(rows)}

    result: dict = {
        "n_questions": len(qs),
        "n_candidates": len(all_rows),
        "overall": {m: _auc_for(all_rows, m) for m in modes},
        "by_tier": {
            tier: {m: _auc_for(rows, m) for m in modes}
            for tier, rows in by_tier.items()
        },
        "by_category": {
            cat: {m: _auc_for(rows, m) for m in modes}
            for cat, rows in by_category.items()
        },
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fmt(auc_entry: dict) -> str:
    auc = auc_entry.get("auc", float("nan"))
    n = auc_entry.get("n", 0)
    if math.isnan(auc):
        return f"  n/a  (n={n})"
    return f"{auc:5.3f}  (n={n})"


def print_report(results: dict, model_name: str) -> None:
    print(f"\n================== Experiment B AUC report [{model_name}] ==================\n")

    frank = results.get("frank", {})
    if frank.get("n_spans", 0) > 0:
        print(f"FRANK ({frank['n_spans']} spans)")
        ov = frank["overall"]
        print("  Overall:")
        print(f"    output_level      {_fmt(ov['output_level'])}")
        print(f"    entity_level      {_fmt(ov['entity_level'])}")
        print(f"    rank_only_logrank {_fmt(ov['rank_only_log_rank'])}")
        print(f"    rank_only_neg_logG{_fmt(ov['rank_only_neg_log_g'])}")
        print(f"    rank_only_rank_dev{_fmt(ov['rank_only_rank_dev'])}  (log2 r_g/r_src)")
        for et, modes in frank.get("by_error_type", {}).items():
            print(f"  Error type '{et}' (n_err={modes.get('n_errors', 0)}, n_ctrl={modes.get('n_controls', 0)}):")
            for m_key in ["output_level", "entity_level",
                          "rank_only_log_rank", "rank_only_rank_dev"]:
                if m_key in modes:
                    print(f"    {m_key:22s}{_fmt(modes[m_key])}")
    else:
        print("FRANK: no scored data yet.\n")

    halu = results.get("halueval", {})
    if halu.get("n_examples", 0) > 0:
        print(f"\nHaluEval ({halu['n_examples']} examples)")
        ov = halu["overall"]
        for m_key in ["output_level", "entity_level",
                      "rank_only_log_rank", "rank_only_neg_log_g"]:
            print(f"  overall  {m_key:22s}{_fmt(ov[m_key])}")
        for task, modes in halu.get("by_task", {}).items():
            for m_key in ["output_level", "entity_level",
                          "rank_only_log_rank", "rank_only_neg_log_g"]:
                print(f"  {task:14s} {m_key:22s}{_fmt(modes[m_key])}")
    else:
        print("\nHaluEval: no scored data yet.")

    tqa = results.get("truthfulqa", {})
    if tqa.get("n_candidates", 0) > 0:
        print(f"\nTruthfulQA ({tqa['n_candidates']} candidates from {tqa['n_questions']} questions)")
        ov = tqa["overall"]
        for m_key in ["output_level", "entity_level",
                      "rank_only_log_rank", "rank_only_neg_log_g"]:
            print(f"  overall  {m_key:22s}{_fmt(ov[m_key])}")
        for tier, modes in tqa.get("by_tier", {}).items():
            for m_key in ["output_level", "entity_level",
                          "rank_only_log_rank", "rank_only_neg_log_g"]:
                print(f"  tier={tier:10s} {m_key:22s}{_fmt(modes[m_key])}")
    else:
        print("\nTruthfulQA: no scored data yet.")

    print()


def main():
    parser = argparse.ArgumentParser(description="Experiment B AUC evaluation")
    parser.add_argument("--model", default="llama-3.1-8b",
                        help="Model whose scored data to load")
    parser.add_argument("--out", default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    results: dict = {}

    frank_path = EXP_DIR / "exp06_frank" / "output" / f"scored_frank_{args.model}.jsonl"
    if frank_path.exists():
        results["frank"] = frank_auc_table(frank_path)
    else:
        results["frank"] = {"n_spans": 0}

    halueval_dir = EXP_DIR / "exp04_halueval" / "data" / args.model
    if halueval_dir.exists():
        results["halueval"] = halueval_auc_table(halueval_dir, args.model)
    else:
        results["halueval"] = {"n_examples": 0}

    tqa_path = EXP_DIR / "exp05_truthfulqa" / "results" / args.model / f"scored_{args.model}.json"
    if tqa_path.exists():
        results["truthfulqa"] = truthfulqa_auc_table(tqa_path)
    else:
        results["truthfulqa"] = {"n_candidates": 0}

    print_report(results, args.model)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved full results to {args.out}")


if __name__ == "__main__":
    main()
