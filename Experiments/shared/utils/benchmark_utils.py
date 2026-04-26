"""Benchmark evaluation utilities for ranking-inference experiments.

Provides standard classification metrics (ROC-AUC, PR-AUC, F1) plus
statistical utilities (bootstrap CI, paired bootstrap significance test,
Cohen's d effect size).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from typing import Callable


def compute_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC-ROC.

    Returns 0.5 if the input is degenerate (single class present), instead
    of raising a ValueError from sklearn.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def compute_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC-PR (average precision score).

    Returns 0.0 if the input is degenerate (single class present).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, scores))


def bootstrap_ci(
    y_true,
    scores,
    metric_fn: Callable,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Resamples (y_true, scores) with replacement n_resamples times and
    evaluates metric_fn on each resample. Degenerate resamples (where
    only a single class is present) are skipped.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    scores : array-like of shape (n,)
    metric_fn : callable(y_true, scores) -> float
    n_resamples : int
    confidence : float — e.g. 0.95 for a 95% CI
    seed : int

    Returns
    -------
    (lower, point_estimate, upper)
        point_estimate is metric_fn evaluated on the original (full) data.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point_estimate = metric_fn(y_true, scores)

    boot_stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        s_boot = scores[idx]
        if len(np.unique(y_boot)) < 2:
            # Skip degenerate resample
            continue
        boot_stats.append(metric_fn(y_boot, s_boot))

    boot_stats = np.array(boot_stats)
    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lower, float(point_estimate), upper


def paired_bootstrap_test(
    y_true,
    scores_a,
    scores_b,
    metric_fn: Callable,
    n_resamples: int = 1000,
    seed: int = 42,
) -> float:
    """Paired bootstrap significance test (Koehn 2004).

    For each resample, compute delta = metric_a - metric_b on a bootstrap
    sample. The two-sided p-value is the fraction of resamples where
    |delta_boot| >= |delta_observed|.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    scores_a, scores_b : array-like of shape (n,) — paired scores
    metric_fn : callable(y_true, scores) -> float
    n_resamples : int
    seed : int

    Returns
    -------
    float — two-sided p-value in [0, 1]
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    observed_diff = metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b)

    count_extreme = 0
    valid = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        diff_boot = (
            metric_fn(y_boot, scores_a[idx]) - metric_fn(y_boot, scores_b[idx])
        )
        # Koehn (2004): center the bootstrap distribution under the null.
        # Under H0 (true diff = 0), delta_boot - observed_diff approximates
        # the null distribution.  Count resamples where the centered absolute
        # deviation exceeds the observed absolute difference.
        if abs(diff_boot - observed_diff) >= abs(observed_diff):
            count_extreme += 1
        valid += 1

    if valid == 0:
        return 1.0  # Cannot determine significance
    return float(count_extreme / valid)


def compute_f1_at_optimal_threshold(
    y_true,
    scores,
    n_thresholds: int = 200,
) -> tuple[float, float]:
    """Sweep thresholds and return the best F1 score and its threshold.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    scores : array-like of shape (n,) — continuous scores in [0, 1]
    n_thresholds : int — number of evenly-spaced thresholds to evaluate

    Returns
    -------
    (best_f1, best_threshold) — both floats in [0, 1]
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    best_f1 = 0.0
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        # Skip degenerate predictions (all same class) to avoid ill-defined F1
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        current_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(t)

    return best_f1, best_threshold


def compute_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    d = (mean_a - mean_b) / pooled_std

    Uses ddof=1 (sample variance) for the pooled standard deviation.

    Parameters
    ----------
    a, b : array-like — the two groups to compare

    Returns
    -------
    float — Cohen's d (positive when mean_a > mean_b)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0.0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)
