"""HaluEval evaluation with length bias controls.

Each scored example has a single response with a binary label (0=correct,
1=hallucinated) and RI aggregate scores. Standard binary classification
evaluation with two length bias controls.

Trivial baseline reference: classifying any response with >27 characters
as hallucinated achieves ~93.3% accuracy on HaluEval.
"""

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from shared.utils import (
    compute_roc_auc,
    compute_pr_auc,
    bootstrap_ci,
    compute_f1_at_optimal_threshold,
    compute_cohens_d,
)


def compute_halueval_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
) -> dict:
    """Compute AUC-ROC, AUC-PR, and F1@optimal with bootstrap CIs.

    Args:
        labels: binary array (0=correct, 1=hallucinated)
        scores: continuous RI aggregate scores (higher = more likely hallucinated)
        n_bootstrap: number of bootstrap resamples for CIs
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    roc_auc = compute_roc_auc(labels, scores)
    pr_auc = compute_pr_auc(labels, scores)
    f1, threshold = compute_f1_at_optimal_threshold(labels, scores)

    if len(np.unique(labels)) >= 2:
        roc_ci = bootstrap_ci(labels, scores, compute_roc_auc, n_resamples=n_bootstrap)
        pr_ci = bootstrap_ci(labels, scores, compute_pr_auc, n_resamples=n_bootstrap)
    else:
        roc_ci = (roc_auc, roc_auc, roc_auc)
        pr_ci = (pr_auc, pr_auc, pr_auc)

    correct_scores = scores[labels == 0]
    hallucinated_scores = scores[labels == 1]
    if len(correct_scores) > 0 and len(hallucinated_scores) > 0:
        d = compute_cohens_d(hallucinated_scores, correct_scores)
    else:
        d = 0.0

    return {
        "roc_auc": roc_auc,
        "roc_auc_ci": (roc_ci[0], roc_ci[2]),
        "pr_auc": pr_auc,
        "pr_auc_ci": (pr_ci[0], pr_ci[2]),
        "f1_optimal": f1,
        "f1_threshold": threshold,
        "cohens_d": d,
        "n_samples": len(labels),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
    }


def compute_length_matched_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    lengths: np.ndarray,
    n_bins: int = 10,
    seed: int = 42,
) -> dict:
    """Control 1: Length-matched binning to neutralise the length confound.

    Bins examples by text length, then within each bin samples equal numbers
    from each class. Computes AUC on the matched subset.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    lengths = np.asarray(lengths)
    rng = np.random.default_rng(seed)

    bin_edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1
    bin_indices = np.digitize(lengths, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    matched_labels = []
    matched_scores = []
    n_bins_used = 0

    for b in range(n_bins):
        mask = bin_indices == b
        bin_labels = labels[mask]
        bin_scores = scores[mask]

        n_pos = int((bin_labels == 1).sum())
        n_neg = int((bin_labels == 0).sum())

        if n_pos == 0 or n_neg == 0:
            continue

        n_bins_used += 1
        n_sample = min(n_pos, n_neg)

        pos_idx = np.where(bin_labels == 1)[0]
        neg_idx = np.where(bin_labels == 0)[0]

        sampled_pos = rng.choice(pos_idx, size=n_sample, replace=False)
        sampled_neg = rng.choice(neg_idx, size=n_sample, replace=False)

        matched_labels.extend(bin_labels[sampled_pos].tolist())
        matched_labels.extend(bin_labels[sampled_neg].tolist())
        matched_scores.extend(bin_scores[sampled_pos].tolist())
        matched_scores.extend(bin_scores[sampled_neg].tolist())

    if len(matched_labels) < 4:
        return {
            "matched_roc_auc": 0.5,
            "n_matched": len(matched_labels),
            "n_bins_used": n_bins_used,
        }

    matched_roc_auc = compute_roc_auc(
        np.array(matched_labels), np.array(matched_scores)
    )

    return {
        "matched_roc_auc": matched_roc_auc,
        "n_matched": len(matched_labels),
        "n_bins_used": n_bins_used,
    }


def compute_length_regression(
    labels: np.ndarray,
    scores: np.ndarray,
    lengths: np.ndarray,
) -> dict:
    """Control 2: Logistic regression with length covariate.

    Fits three models to isolate the RI signal from length:
      - length_only, ri_only, ri+length
    delta_auc = AUC(ri+length) - AUC(length_only)
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    lengths = np.asarray(lengths, dtype=float).reshape(-1, 1)

    def _standardise(x):
        std = x.std()
        if std == 0:
            return x - x.mean()
        return (x - x.mean()) / std

    scores_std = _standardise(scores)
    lengths_std = _standardise(lengths)

    lr_length = LogisticRegression(max_iter=1000, random_state=42)
    lr_length.fit(lengths_std, labels)
    pred_length = lr_length.predict_proba(lengths_std)[:, 1]
    auc_length = compute_roc_auc(labels, pred_length)

    lr_ri = LogisticRegression(max_iter=1000, random_state=42)
    lr_ri.fit(scores_std, labels)
    pred_ri = lr_ri.predict_proba(scores_std)[:, 1]
    auc_ri = compute_roc_auc(labels, pred_ri)

    X_both = np.hstack([scores_std, lengths_std])
    lr_both = LogisticRegression(max_iter=1000, random_state=42)
    lr_both.fit(X_both, labels)
    pred_both = lr_both.predict_proba(X_both)[:, 1]
    auc_both = compute_roc_auc(labels, pred_both)

    return {
        "auc_length_only": auc_length,
        "auc_ri_only": auc_ri,
        "auc_ri_plus_length": auc_both,
        "delta_auc": auc_both - auc_length,
        "ri_coefficient": float(lr_both.coef_[0][0]),
        "length_coefficient": float(lr_both.coef_[0][1]),
    }


def evaluate_by_task(
    scored_examples: list,
    aggregation_strategy: str = "entity_weighted_mean",
) -> dict:
    """Stratify evaluation by HaluEval task type.

    Each scored example has: label (0/1), scores (dict), text_length.

    Args:
        scored_examples: list of HaluEvalScoredExample (or dicts)
        aggregation_strategy: key from aggregate_all() output

    Returns:
        Dict keyed by task name + "all", each with metrics, length controls.
    """
    by_task: dict[str, list] = defaultdict(list)
    for ex in scored_examples:
        task = ex["task"] if isinstance(ex, dict) else ex.task
        by_task[task].append(ex)

    results = {}

    for task_name, task_examples in by_task.items():
        labels, scores, lengths = _extract_data(task_examples, aggregation_strategy)
        results[task_name] = _evaluate_single_split(labels, scores, lengths, len(task_examples))

    # Combined "all" evaluation
    all_labels, all_scores, all_lengths = _extract_data(scored_examples, aggregation_strategy)
    results["all"] = _evaluate_single_split(
        all_labels, all_scores, all_lengths, len(scored_examples)
    )

    return results


def _extract_data(
    examples: list,
    aggregation_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract labels, scores, and lengths from scored examples."""
    labels = []
    scores = []
    lengths = []

    for ex in examples:
        if isinstance(ex, dict):
            label = ex["label"]
            score_dict = ex["scores"]
            text_len = ex["text_length"]
        else:
            label = ex.label
            score_dict = ex.scores
            text_len = ex.text_length

        labels.append(label)
        scores.append(score_dict.get(aggregation_strategy, 0.0))
        lengths.append(text_len)

    return np.array(labels), np.array(scores), np.array(lengths)


def _evaluate_single_split(
    labels: np.ndarray,
    scores: np.ndarray,
    lengths: np.ndarray,
    n_examples: int,
) -> dict:
    """Run full evaluation on a single task split."""
    metrics = compute_halueval_metrics(labels, scores)
    length_matched = compute_length_matched_auc(labels, scores, lengths)
    length_regression = compute_length_regression(labels, scores, lengths)

    return {
        "metrics": metrics,
        "length_matched": length_matched,
        "length_regression": length_regression,
        "n_examples": n_examples,
    }
