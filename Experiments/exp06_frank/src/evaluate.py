"""FRANK span-level evaluation per error type.

Evaluates whether RI delta scores can distinguish error spans from control
spans, with the core hypothesis: F1 should follow a gradient from high
(Tier 1 errors like OutE) to near-zero (Tier 2 errors like CorefE).
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import (
    compute_roc_auc,
    compute_f1_at_optimal_threshold,
    bootstrap_ci,
    compute_cohens_d,
)
from exp06_frank.src.load_dataset import TIER_SIGNAL_ORDER, ERROR_TYPE_TO_TIER


def _get_delta_key(baseline: str, log_space: bool = False) -> str:
    """Return the attribute name for the chosen baseline and space."""
    if log_space:
        if baseline == "source":
            return "mean_log_delta_source"
        return "mean_log_delta"
    if baseline == "source":
        return "mean_delta_source"
    return "mean_delta"


def _spans_to_arrays(spans, baseline="global", log_space=False):
    """Convert span list to (y_true, scores) arrays.

    Args:
        spans: list of FRANKScoredSpan or dicts with same fields
        baseline: "global" or "source"
        log_space: if True, use log-space delta instead of linear

    Returns:
        (y_true, scores) as numpy arrays
    """
    delta_key = _get_delta_key(baseline, log_space=log_space)
    y_true = []
    scores = []
    for s in spans:
        if isinstance(s, dict):
            y_true.append(1 if s["is_error"] else 0)
            scores.append(s[delta_key])
        else:
            y_true.append(1 if s.is_error else 0)
            scores.append(getattr(s, delta_key))
    return np.array(y_true), np.array(scores)


def compute_span_f1_by_error_type(spans, baseline="global", log_space=False):
    """Span-level F1 per FRANK error type.

    Uses 80/20 split: find optimal threshold on 20% (tuning set),
    evaluate F1 on the remaining 80% (test set).

    Args:
        spans: list of FRANKScoredSpan (or dicts with same fields)
        baseline: "global" uses mean_delta, "source" uses mean_delta_source
        log_space: if True, use log-space delta instead of linear

    Returns:
        dict[error_type -> {"f1", "precision", "recall", "threshold",
                            "n_error", "n_control", "tier", "auc",
                            "cohens_d", "auc_ci"}]
    """
    delta_key = _get_delta_key(baseline, log_space=log_space)

    # Group error spans by type, collect all control spans
    error_by_type = defaultdict(list)
    control_spans = []

    for s in spans:
        if isinstance(s, dict):
            is_err = s["is_error"]
            etype = s["error_type"]
            delta_val = s[delta_key]
        else:
            is_err = s.is_error
            etype = s.error_type
            delta_val = getattr(s, delta_key)

        if is_err:
            error_by_type[etype].append(delta_val)
        else:
            control_spans.append(delta_val)

    if not control_spans:
        return {}

    control_arr = np.array(control_spans)
    results = {}

    for etype, error_deltas in error_by_type.items():
        if not error_deltas:
            continue

        error_arr = np.array(error_deltas)
        n_error = len(error_arr)
        n_control = len(control_arr)

        # Build paired arrays: errors labelled 1, controls labelled 0
        y_true = np.concatenate([np.ones(n_error), np.zeros(n_control)])
        all_scores = np.concatenate([error_arr, control_arr])

        # 80/20 split for threshold tuning
        n_total = len(y_true)
        rng = np.random.default_rng(42)
        indices = rng.permutation(n_total)
        n_tune = max(1, int(0.2 * n_total))
        tune_idx = indices[:n_tune]
        test_idx = indices[n_tune:]

        y_tune = y_true[tune_idx]
        s_tune = all_scores[tune_idx]
        y_test = y_true[test_idx]
        s_test = all_scores[test_idx]

        # Find optimal threshold on tuning set
        if len(np.unique(y_tune)) < 2:
            # Degenerate tuning set, use full data
            f1, threshold = compute_f1_at_optimal_threshold(y_true, all_scores)
            y_eval = y_true
            s_eval = all_scores
        else:
            _, threshold = compute_f1_at_optimal_threshold(y_tune, s_tune)
            y_eval = y_test
            s_eval = s_test

        # Evaluate on test set
        y_pred = (s_eval >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_eval == 1)))
        fp = int(np.sum((y_pred == 1) & (y_eval == 0)))
        fn = int(np.sum((y_pred == 0) & (y_eval == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # AUC on full data
        auc = compute_roc_auc(y_true, all_scores)

        # Bootstrap CI for AUC
        auc_ci = bootstrap_ci(y_true, all_scores, compute_roc_auc,
                              n_resamples=500, confidence=0.95, seed=42)

        # Cohen's d effect size
        d = compute_cohens_d(error_arr, control_arr)

        tier = ERROR_TYPE_TO_TIER.get(etype, "unknown")

        results[etype] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "threshold": threshold,
            "n_error": n_error,
            "n_control": n_control,
            "tier": tier,
            "auc": auc,
            "auc_ci": auc_ci,
            "cohens_d": d,
        }

    return results


def compute_error_type_gradient(per_type_results: dict) -> dict:
    """Compute Spearman correlation between predicted tier order and actual F1.

    The core hypothesis: error types that are more "ontological" (Tier 1)
    should have higher F1 than "structural" ones (Tier 2), following:

        OutE > EntE > CircE > PredE > LinkE ~ CorefE

    Args:
        per_type_results: dict from compute_span_f1_by_error_type

    Returns:
        dict with keys: spearman_rho, p_value, predicted_order, actual_order,
        gradient_holds (bool), summary
    """
    from scipy.stats import spearmanr

    # Only include error types present in results AND in our ordering
    common_types = [et for et in TIER_SIGNAL_ORDER if et in per_type_results]

    if len(common_types) < 3:
        return {
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
            "predicted_order": [],
            "actual_order": [],
            "gradient_holds": False,
            "summary": f"Insufficient error types for gradient analysis ({len(common_types)} < 3)",
        }

    # Predicted ordering: by TIER_SIGNAL_ORDER (higher = stronger predicted signal)
    predicted_ranks = [TIER_SIGNAL_ORDER[et] for et in common_types]

    # Actual F1 values
    actual_f1 = [per_type_results[et]["f1"] for et in common_types]

    rho, p_value = spearmanr(predicted_ranks, actual_f1)

    # Sort by predicted signal strength (descending) for display
    sorted_by_predicted = sorted(common_types, key=lambda et: TIER_SIGNAL_ORDER[et], reverse=True)
    sorted_by_actual = sorted(common_types, key=lambda et: per_type_results[et]["f1"], reverse=True)

    # Gradient "holds" if rho > 0.5 and p < 0.2 (relaxed for small n)
    gradient_holds = rho > 0.5 and p_value < 0.2

    summary_lines = [
        f"Spearman rho = {rho:.3f} (p = {p_value:.3f})",
        f"Predicted order: {' > '.join(sorted_by_predicted)}",
        f"Actual F1 order: {' > '.join(sorted_by_actual)}",
    ]
    for et in sorted_by_predicted:
        r = per_type_results[et]
        summary_lines.append(
            f"  {et} ({r['tier']}): F1={r['f1']:.3f}, AUC={r['auc']:.3f}, d={r['cohens_d']:.3f}"
        )

    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "predicted_order": sorted_by_predicted,
        "actual_order": sorted_by_actual,
        "gradient_holds": gradient_holds,
        "summary": "\n".join(summary_lines),
    }


def evaluate_frank(scored_spans, model_name):
    """Full evaluation: per-error-type F1 for both baselines, gradient analysis.

    Evaluates in both linear space (legacy delta) and log space (Bayesian posterior).

    Args:
        scored_spans: list of FRANKScoredSpan or dicts
        model_name: model identifier for labelling

    Returns:
        Comprehensive results dict with linear and log-space metrics.
    """
    # Check if log-space fields are available
    sample = scored_spans[0] if scored_spans else None
    has_log = False
    if sample is not None:
        if isinstance(sample, dict):
            has_log = "mean_log_delta" in sample and sample["mean_log_delta"] != 0.0
        else:
            has_log = hasattr(sample, "mean_log_delta") and sample.mean_log_delta != 0.0

    # --- Linear space (legacy) ---
    global_results = compute_span_f1_by_error_type(scored_spans, baseline="global")
    source_results = compute_span_f1_by_error_type(scored_spans, baseline="source")
    global_gradient = compute_error_type_gradient(global_results)
    source_gradient = compute_error_type_gradient(source_results)

    y_true_g, scores_g = _spans_to_arrays(scored_spans, baseline="global")
    y_true_s, scores_s = _spans_to_arrays(scored_spans, baseline="source")

    overall_auc_global = compute_roc_auc(y_true_g, scores_g)
    overall_auc_source = compute_roc_auc(y_true_s, scores_s)
    overall_f1_global, _ = compute_f1_at_optimal_threshold(y_true_g, scores_g)
    overall_f1_source, _ = compute_f1_at_optimal_threshold(y_true_s, scores_s)

    n_error = int(np.sum(y_true_g == 1))
    n_control = int(np.sum(y_true_g == 0))

    results = {
        "model_name": model_name,
        "global_baseline": global_results,
        "source_baseline": source_results,
        "global_gradient": global_gradient,
        "source_gradient": source_gradient,
        "overall_auc_global": overall_auc_global,
        "overall_auc_source": overall_auc_source,
        "overall_f1_global": overall_f1_global,
        "overall_f1_source": overall_f1_source,
        "n_error_spans": n_error,
        "n_control_spans": n_control,
    }

    # --- Log space (Bayesian posterior) ---
    if has_log:
        log_global_results = compute_span_f1_by_error_type(scored_spans, baseline="global", log_space=True)
        log_source_results = compute_span_f1_by_error_type(scored_spans, baseline="source", log_space=True)
        log_global_gradient = compute_error_type_gradient(log_global_results)
        log_source_gradient = compute_error_type_gradient(log_source_results)

        y_true_lg, scores_lg = _spans_to_arrays(scored_spans, baseline="global", log_space=True)
        y_true_ls, scores_ls = _spans_to_arrays(scored_spans, baseline="source", log_space=True)

        log_auc_global = compute_roc_auc(y_true_lg, scores_lg)
        log_auc_source = compute_roc_auc(y_true_ls, scores_ls)
        log_f1_global, _ = compute_f1_at_optimal_threshold(y_true_lg, scores_lg)
        log_f1_source, _ = compute_f1_at_optimal_threshold(y_true_ls, scores_ls)

        results.update({
            "log_global_baseline": log_global_results,
            "log_source_baseline": log_source_results,
            "log_global_gradient": log_global_gradient,
            "log_source_gradient": log_source_gradient,
            "log_overall_auc_global": log_auc_global,
            "log_overall_auc_source": log_auc_source,
            "log_overall_f1_global": log_f1_global,
            "log_overall_f1_source": log_f1_source,
        })

    # Print summary
    print(f"\n{'='*60}")
    print(f"FRANK Evaluation Results — {model_name}")
    print(f"{'='*60}")
    print(f"Spans: {n_error} error, {n_control} control")
    print(f"\n--- LINEAR SPACE (legacy delta) ---")
    print(f"Overall AUC:  global={overall_auc_global:.3f}  source={overall_auc_source:.3f}")
    print(f"Overall F1:   global={overall_f1_global:.3f}  source={overall_f1_source:.3f}")
    print(f"Gradient: {global_gradient.get('summary', 'N/A')}")

    if has_log:
        print(f"\n--- LOG SPACE (Bayesian posterior) ---")
        print(f"Overall AUC:  global={log_auc_global:.3f}  source={log_auc_source:.3f}")
        print(f"Overall F1:   global={log_f1_global:.3f}  source={log_f1_source:.3f}")
        print(f"Gradient: {log_global_gradient.get('summary', 'N/A')}")

    print(f"{'='*60}\n")

    return results
