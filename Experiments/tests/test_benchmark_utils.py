"""Tests for shared/utils/benchmark_utils.py — Benchmark Evaluation Utilities."""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils.benchmark_utils import (
    compute_roc_auc,
    compute_pr_auc,
    bootstrap_ci,
    paired_bootstrap_test,
    compute_f1_at_optimal_threshold,
    compute_cohens_d,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_predictions():
    """50 negatives from N(0.3, 0.15), 50 positives from N(0.7, 0.15). seed=42."""
    rng = np.random.default_rng(42)
    neg_scores = rng.normal(0.3, 0.15, 50)
    pos_scores = rng.normal(0.7, 0.15, 50)
    scores = np.concatenate([neg_scores, pos_scores])
    y_true = np.array([0] * 50 + [1] * 50)
    return y_true, scores


@pytest.fixture
def random_predictions():
    """100 uniform [0,1] scores with 50/50 labels. seed=42."""
    rng = np.random.default_rng(42)
    scores = rng.uniform(0.0, 1.0, 100)
    y_true = np.array([0] * 50 + [1] * 50)
    return y_true, scores


# ---------------------------------------------------------------------------
# compute_roc_auc
# ---------------------------------------------------------------------------

class TestComputeRocAuc:
    def test_good_classifier_above_threshold(self, binary_predictions):
        y_true, scores = binary_predictions
        auc = compute_roc_auc(y_true, scores)
        assert auc > 0.8, f"Expected AUC > 0.8 for good classifier, got {auc:.4f}"

    def test_random_classifier_near_half(self, random_predictions):
        y_true, scores = random_predictions
        auc = compute_roc_auc(y_true, scores)
        assert 0.3 <= auc <= 0.7, f"Expected AUC near 0.5 for random, got {auc:.4f}"

    def test_degenerate_single_class_returns_half(self):
        """Single-class input should return 0.5, not raise."""
        y_true = np.zeros(20)
        scores = np.random.rand(20)
        result = compute_roc_auc(y_true, scores)
        assert result == 0.5

    def test_returns_float(self, binary_predictions):
        y_true, scores = binary_predictions
        assert isinstance(compute_roc_auc(y_true, scores), float)


# ---------------------------------------------------------------------------
# compute_pr_auc
# ---------------------------------------------------------------------------

class TestComputePrAuc:
    def test_good_classifier_above_threshold(self, binary_predictions):
        y_true, scores = binary_predictions
        pr_auc = compute_pr_auc(y_true, scores)
        assert pr_auc > 0.7, f"Expected PR-AUC > 0.7 for good classifier, got {pr_auc:.4f}"

    def test_degenerate_single_class_returns_zero(self):
        """Single-class input should return 0.0, not raise."""
        y_true = np.zeros(20)
        scores = np.random.rand(20)
        result = compute_pr_auc(y_true, scores)
        assert result == 0.0

    def test_returns_float(self, binary_predictions):
        y_true, scores = binary_predictions
        assert isinstance(compute_pr_auc(y_true, scores), float)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCi:
    def test_returns_triple(self, binary_predictions):
        y_true, scores = binary_predictions
        result = bootstrap_ci(y_true, scores, compute_roc_auc)
        assert len(result) == 3, "bootstrap_ci must return (lower, point, upper)"

    def test_lower_le_point_le_upper(self, binary_predictions):
        y_true, scores = binary_predictions
        lower, point, upper = bootstrap_ci(y_true, scores, compute_roc_auc)
        assert lower <= point <= upper, (
            f"Expected lower <= point <= upper, got {lower:.4f} <= {point:.4f} <= {upper:.4f}"
        )

    def test_point_estimate_close_to_direct_auc(self, binary_predictions):
        y_true, scores = binary_predictions
        direct_auc = compute_roc_auc(y_true, scores)
        lower, point, upper = bootstrap_ci(y_true, scores, compute_roc_auc)
        assert abs(point - direct_auc) < 0.05, (
            f"Bootstrap point estimate {point:.4f} far from direct AUC {direct_auc:.4f}"
        )

    def test_ci_is_non_trivial(self, binary_predictions):
        """Lower and upper should differ (non-degenerate confidence interval)."""
        y_true, scores = binary_predictions
        lower, point, upper = bootstrap_ci(y_true, scores, compute_roc_auc)
        assert upper > lower, "CI should have positive width"


# ---------------------------------------------------------------------------
# paired_bootstrap_test
# ---------------------------------------------------------------------------

class TestPairedBootstrapTest:
    def test_different_methods_significant(self, binary_predictions, random_predictions):
        """Good classifier vs random should yield p < 0.05."""
        y_true_good, scores_good = binary_predictions
        # Use the same y_true for both (paired test requires same labels)
        # Construct a near-random score for comparison
        rng = np.random.default_rng(0)
        scores_random = rng.uniform(0.0, 1.0, len(y_true_good))
        p = paired_bootstrap_test(y_true_good, scores_good, scores_random, compute_roc_auc)
        assert p < 0.05, f"Expected p < 0.05 for clearly different classifiers, got {p:.4f}"

    def test_identical_methods_not_significant(self, binary_predictions):
        """Same scores vs same scores should yield p > 0.05."""
        y_true, scores = binary_predictions
        p = paired_bootstrap_test(y_true, scores, scores.copy(), compute_roc_auc)
        assert p > 0.05, f"Expected p > 0.05 for identical scores, got {p:.4f}"

    def test_returns_float_in_unit_interval(self, binary_predictions):
        y_true, scores = binary_predictions
        p = paired_bootstrap_test(y_true, scores, scores, compute_roc_auc)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# compute_f1_at_optimal_threshold
# ---------------------------------------------------------------------------

class TestComputeF1AtOptimalThreshold:
    def test_returns_pair(self, binary_predictions):
        y_true, scores = binary_predictions
        result = compute_f1_at_optimal_threshold(y_true, scores)
        assert len(result) == 2, "Must return (f1, threshold)"

    def test_both_values_in_unit_interval(self, binary_predictions):
        y_true, scores = binary_predictions
        f1, threshold = compute_f1_at_optimal_threshold(y_true, scores)
        assert 0.0 <= f1 <= 1.0, f"F1 {f1:.4f} out of [0,1]"
        assert 0.0 <= threshold <= 1.0, f"Threshold {threshold:.4f} out of [0,1]"

    def test_good_classifier_f1_above_threshold(self, binary_predictions):
        y_true, scores = binary_predictions
        f1, threshold = compute_f1_at_optimal_threshold(y_true, scores)
        assert f1 > 0.7, f"Expected F1 > 0.7 for good classifier, got {f1:.4f}"

    def test_threshold_sweep_used(self, binary_predictions):
        """n_thresholds parameter accepted without error."""
        y_true, scores = binary_predictions
        f1, threshold = compute_f1_at_optimal_threshold(y_true, scores, n_thresholds=50)
        assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# compute_cohens_d
# ---------------------------------------------------------------------------

class TestComputeCohensD:
    def test_well_separated_distributions(self):
        """N(0,1) vs N(2,1) should give |d| > 1.0."""
        rng = np.random.default_rng(42)
        a = rng.normal(0.0, 1.0, 200)
        b = rng.normal(2.0, 1.0, 200)
        d = compute_cohens_d(a, b)
        assert abs(d) > 1.0, f"Expected |d| > 1.0 for well-separated groups, got {d:.4f}"

    def test_same_distribution_small_effect(self):
        """Samples from same distribution should give |d| < 0.5."""
        rng = np.random.default_rng(42)
        a = rng.normal(0.0, 1.0, 200)
        b = rng.normal(0.0, 1.0, 200)
        d = compute_cohens_d(a, b)
        assert abs(d) < 0.5, f"Expected |d| < 0.5 for same distribution, got {d:.4f}"

    def test_direction_of_effect(self):
        """d should be positive when mean_a > mean_b."""
        rng = np.random.default_rng(99)
        a = rng.normal(3.0, 1.0, 100)
        b = rng.normal(1.0, 1.0, 100)
        d = compute_cohens_d(a, b)
        assert d > 0, f"Expected positive d when mean_a > mean_b, got {d:.4f}"

    def test_returns_float(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert isinstance(compute_cohens_d(a, b), float)
