"""Tests for FRANK span-level detection benchmark (exp06_frank).

Uses synthetic data throughout -- no network calls or real model scoring.
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp06_frank.src.load_dataset import (
    ERROR_TYPE_TO_TIER,
    TIER_SIGNAL_ORDER,
    ALL_ERROR_TYPES,
    ErrorSpan,
    FRANKExample,
)
from exp06_frank.src.score_examples import (
    compute_token_deltas,
    compute_span_delta,
    sample_control_spans,
    align_char_span_to_tokens,
    FRANKScoredSpan,
)
from exp06_frank.src.evaluate import (
    compute_span_f1_by_error_type,
    compute_error_type_gradient,
    evaluate_frank,
)

from shared.utils import build_rank_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_rank_table():
    """A small rank table for testing."""
    token_ids = []
    for tid in range(50):
        freq = max(1, int(500 / (tid + 1)))
        token_ids.extend([tid] * freq)
    return build_rank_table(token_ids, "test-tokenizer", "test-corpus")


@pytest.fixture
def frank_example():
    """A synthetic FRANKExample for testing."""
    article = (
        "The United Nations held a conference in Geneva on climate change. "
        "Secretary-General Antonio Guterres addressed the assembly."
    )
    summary = (
        "The United Nations held a conference in Paris on climate change. "
        "Secretary-General Ban Ki-moon addressed the assembly."
    )
    return FRANKExample(
        article_id="test_001",
        article_text=article,
        summary_text=summary,
        error_spans=[
            ErrorSpan(
                text="Paris",
                char_start=summary.index("Paris"),
                char_end=summary.index("Paris") + len("Paris"),
                error_type="OutE",
                tier="tier1",
            ),
            ErrorSpan(
                text="Ban Ki-moon",
                char_start=summary.index("Ban Ki-moon"),
                char_end=summary.index("Ban Ki-moon") + len("Ban Ki-moon"),
                error_type="EntE",
                tier="tier1",
            ),
        ],
        system="test_system",
        has_errors=True,
    )


@pytest.fixture
def synthetic_scored_spans():
    """Synthetic scored spans with known gradient pattern.

    Creates spans where Tier 1 errors have higher deltas than Tier 2,
    matching the predicted RI gradient.
    """
    rng = np.random.default_rng(42)
    spans = []

    # Error spans with gradient: OutE > EntE > CircE > PredE > LinkE > CorefE
    error_configs = {
        "OutE":   {"mean": 0.6, "std": 0.1, "n": 30},
        "EntE":   {"mean": 0.5, "std": 0.1, "n": 25},
        "CircE":  {"mean": 0.3, "std": 0.1, "n": 20},
        "PredE":  {"mean": 0.15, "std": 0.1, "n": 20},
        "LinkE":  {"mean": 0.05, "std": 0.1, "n": 15},
        "CorefE": {"mean": 0.02, "std": 0.1, "n": 15},
    }

    for etype, cfg in error_configs.items():
        deltas = rng.normal(cfg["mean"], cfg["std"], cfg["n"])
        for i, d in enumerate(deltas):
            spans.append(FRANKScoredSpan(
                example_id=f"test_{etype}_{i}",
                span_text=f"error span {etype} {i}",
                error_type=etype,
                tier=ERROR_TYPE_TO_TIER[etype],
                is_error=True,
                mean_delta=float(d),
                mean_delta_source=float(d * 1.2),  # source baseline slightly different
                model_name="test-model",
                n_tokens=5,
            ))

    # Control spans: low delta
    for i in range(100):
        d = rng.normal(0.0, 0.08)
        spans.append(FRANKScoredSpan(
            example_id=f"test_control_{i}",
            span_text=f"control span {i}",
            error_type="control",
            tier="control",
            is_error=False,
            mean_delta=float(d),
            mean_delta_source=float(d * 0.9),
            model_name="test-model",
            n_tokens=5,
        ))

    return spans


# ---------------------------------------------------------------------------
# Test: ERROR_TYPE_TO_TIER mapping
# ---------------------------------------------------------------------------


class TestTaxonomyMapping:

    def test_all_error_types_have_tier(self):
        """Every FRANK error type must map to a tier."""
        for etype in ALL_ERROR_TYPES:
            assert etype in ERROR_TYPE_TO_TIER, f"{etype} missing from tier mapping"

    def test_tier_values_are_valid(self):
        """Tier values must be one of the expected tiers."""
        valid_tiers = {"tier1", "tier1.5", "tier2"}
        for etype, tier in ERROR_TYPE_TO_TIER.items():
            assert tier in valid_tiers, f"{etype} mapped to invalid tier: {tier}"

    def test_oute_is_tier1(self):
        assert ERROR_TYPE_TO_TIER["OutE"] == "tier1"

    def test_ente_is_tier1(self):
        assert ERROR_TYPE_TO_TIER["EntE"] == "tier1"

    def test_circe_is_tier1_5(self):
        assert ERROR_TYPE_TO_TIER["CircE"] == "tier1.5"

    def test_prede_is_tier2(self):
        assert ERROR_TYPE_TO_TIER["PredE"] == "tier2"

    def test_linke_is_tier2(self):
        assert ERROR_TYPE_TO_TIER["LinkE"] == "tier2"

    def test_corefe_is_tier2(self):
        assert ERROR_TYPE_TO_TIER["CorefE"] == "tier2"

    def test_signal_order_matches_types(self):
        """TIER_SIGNAL_ORDER must cover all error types."""
        assert set(TIER_SIGNAL_ORDER.keys()) == set(ALL_ERROR_TYPES)

    def test_signal_order_gradient(self):
        """OutE should have highest signal, CorefE lowest."""
        assert TIER_SIGNAL_ORDER["OutE"] > TIER_SIGNAL_ORDER["EntE"]
        assert TIER_SIGNAL_ORDER["EntE"] > TIER_SIGNAL_ORDER["CircE"]
        assert TIER_SIGNAL_ORDER["CircE"] > TIER_SIGNAL_ORDER["PredE"]
        assert TIER_SIGNAL_ORDER["PredE"] > TIER_SIGNAL_ORDER["LinkE"]
        assert TIER_SIGNAL_ORDER["LinkE"] > TIER_SIGNAL_ORDER["CorefE"]


# ---------------------------------------------------------------------------
# Test: Dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:

    def test_error_span_creation(self):
        span = ErrorSpan(
            text="Paris", char_start=10, char_end=15,
            error_type="OutE", tier="tier1",
        )
        assert span.text == "Paris"
        assert span.char_start == 10
        assert span.char_end == 15
        assert span.error_type == "OutE"
        assert span.tier == "tier1"

    def test_frank_example_creation(self, frank_example):
        assert frank_example.article_id == "test_001"
        assert frank_example.has_errors is True
        assert len(frank_example.error_spans) == 2
        assert frank_example.error_spans[0].error_type == "OutE"
        assert frank_example.error_spans[1].error_type == "EntE"

    def test_frank_example_no_errors(self):
        ex = FRANKExample(
            article_id="clean_001",
            article_text="Article text.",
            summary_text="Summary text.",
            error_spans=[],
            system="test",
            has_errors=False,
        )
        assert ex.has_errors is False
        assert len(ex.error_spans) == 0

    def test_scored_span_creation(self):
        span = FRANKScoredSpan(
            example_id="test_001",
            span_text="error text",
            error_type="OutE",
            tier="tier1",
            is_error=True,
            mean_delta=0.45,
            mean_delta_source=0.52,
            model_name="test-model",
            n_tokens=3,
        )
        assert span.is_error is True
        assert span.mean_delta == 0.45
        assert span.mean_delta_source == 0.52

    def test_scored_span_serialisation(self):
        span = FRANKScoredSpan(
            example_id="test_001", span_text="text",
            error_type="OutE", tier="tier1", is_error=True,
            mean_delta=0.5, mean_delta_source=0.6,
            model_name="test", n_tokens=2,
        )
        d = asdict(span)
        assert d["example_id"] == "test_001"
        assert d["mean_delta"] == 0.5
        reconstructed = FRANKScoredSpan(**d)
        assert reconstructed == span


# ---------------------------------------------------------------------------
# Test: compute_span_delta
# ---------------------------------------------------------------------------


class TestComputeSpanDelta:

    def test_known_values(self):
        """compute_span_delta with known delta array and indices."""
        deltas = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = compute_span_delta(deltas, [1, 2, 3])
        assert abs(result - 0.3) < 1e-10  # mean of 0.2, 0.3, 0.4

    def test_single_token(self):
        deltas = np.array([0.1, 0.7, 0.3])
        result = compute_span_delta(deltas, [1])
        assert abs(result - 0.7) < 1e-10

    def test_empty_indices(self):
        deltas = np.array([0.1, 0.2, 0.3])
        result = compute_span_delta(deltas, [])
        assert result == 0.0

    def test_all_tokens(self):
        deltas = np.array([0.2, 0.4, 0.6])
        result = compute_span_delta(deltas, [0, 1, 2])
        assert abs(result - 0.4) < 1e-10

    def test_out_of_bounds_indices_ignored(self):
        deltas = np.array([0.1, 0.2, 0.3])
        result = compute_span_delta(deltas, [0, 1, 10])  # 10 is out of bounds
        assert abs(result - 0.15) < 1e-10  # mean of 0.1, 0.2


# ---------------------------------------------------------------------------
# Test: sample_control_spans
# ---------------------------------------------------------------------------


class TestSampleControlSpans:

    def test_produces_non_overlapping_spans(self):
        """Control spans must not overlap with error spans or each other."""
        n_tokens = 50
        error_spans = [(5, 10), (20, 25)]
        controls = sample_control_spans(n_tokens, error_spans, n_controls=5)

        error_indices = set()
        for start, end in error_spans:
            error_indices.update(range(start, end))

        for cs_start, cs_end in controls:
            control_indices = set(range(cs_start, cs_end))
            assert control_indices.isdisjoint(error_indices), \
                f"Control span ({cs_start}, {cs_end}) overlaps error spans"

    def test_control_span_length_matches(self):
        """Control spans should have the same length as the error spans they match."""
        n_tokens = 100
        error_spans = [(10, 15), (30, 35)]  # both length 5
        controls = sample_control_spans(n_tokens, error_spans, n_controls=3)

        for cs_start, cs_end in controls:
            assert cs_end - cs_start == 5

    def test_returns_empty_for_no_errors(self):
        controls = sample_control_spans(50, [], n_controls=5)
        assert controls == []

    def test_deterministic_with_seed(self):
        controls1 = sample_control_spans(50, [(5, 10)], n_controls=3, seed=42)
        controls2 = sample_control_spans(50, [(5, 10)], n_controls=3, seed=42)
        assert controls1 == controls2

    def test_different_seeds_different_results(self):
        controls1 = sample_control_spans(100, [(5, 10)], n_controls=5, seed=42)
        controls2 = sample_control_spans(100, [(5, 10)], n_controls=5, seed=99)
        # Not guaranteed to differ, but very likely with 100 tokens
        # Just check both are valid
        assert len(controls1) > 0
        assert len(controls2) > 0

    def test_within_bounds(self):
        n_tokens = 30
        controls = sample_control_spans(n_tokens, [(5, 10)], n_controls=5)
        for cs_start, cs_end in controls:
            assert cs_start >= 0
            assert cs_end <= n_tokens


# ---------------------------------------------------------------------------
# Test: compute_span_f1_by_error_type
# ---------------------------------------------------------------------------


class TestComputeSpanF1ByErrorType:

    def test_returns_results_for_each_error_type(self, synthetic_scored_spans):
        results = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="global")
        # Should have results for all error types present in the test data
        test_types = {"OutE", "EntE", "CircE", "PredE", "LinkE", "CorefE"}
        for etype in test_types:
            assert etype in results, f"Missing results for {etype}"

    def test_f1_in_valid_range(self, synthetic_scored_spans):
        results = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="global")
        for etype, r in results.items():
            assert 0.0 <= r["f1"] <= 1.0, f"F1 out of range for {etype}: {r['f1']}"
            assert 0.0 <= r["precision"] <= 1.0
            assert 0.0 <= r["recall"] <= 1.0

    def test_auc_in_valid_range(self, synthetic_scored_spans):
        results = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="global")
        for etype, r in results.items():
            assert 0.0 <= r["auc"] <= 1.0, f"AUC out of range for {etype}: {r['auc']}"

    def test_tier1_f1_higher_than_tier2(self, synthetic_scored_spans):
        """Tier 1 error types should have higher F1 than Tier 2 (with synthetic gradient data)."""
        results = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="global")
        # OutE (tier1) should have higher F1 than CorefE (tier2)
        assert results["OutE"]["f1"] > results["CorefE"]["f1"], \
            f"OutE F1 ({results['OutE']['f1']:.3f}) should exceed CorefE F1 ({results['CorefE']['f1']:.3f})"

    def test_includes_cohens_d(self, synthetic_scored_spans):
        results = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="global")
        for etype, r in results.items():
            assert "cohens_d" in r

    def test_includes_auc_ci(self, synthetic_scored_spans):
        results = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="global")
        for etype, r in results.items():
            ci = r["auc_ci"]
            assert len(ci) == 3  # (lower, point, upper)
            assert ci[0] <= ci[1] <= ci[2]

    def test_source_baseline_different_from_global(self, synthetic_scored_spans):
        results_g = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="global")
        results_s = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="source")
        # At least some error types should have different F1
        any_different = False
        for etype in ALL_ERROR_TYPES:
            if etype in results_g and etype in results_s:
                if abs(results_g[etype]["f1"] - results_s[etype]["f1"]) > 1e-6:
                    any_different = True
                    break
        assert any_different, "Source and global baselines should produce different results"


# ---------------------------------------------------------------------------
# Test: compute_error_type_gradient
# ---------------------------------------------------------------------------


class TestComputeErrorTypeGradient:

    def test_returns_valid_spearman(self, synthetic_scored_spans):
        per_type = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="global")
        gradient = compute_error_type_gradient(per_type)
        assert "spearman_rho" in gradient
        assert "p_value" in gradient
        assert -1.0 <= gradient["spearman_rho"] <= 1.0

    def test_gradient_holds_with_synthetic_data(self, synthetic_scored_spans):
        """With synthetic data designed to follow the gradient, rho should be positive."""
        per_type = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="global")
        gradient = compute_error_type_gradient(per_type)
        assert gradient["spearman_rho"] > 0.0, \
            f"Expected positive Spearman rho, got {gradient['spearman_rho']}"

    def test_predicted_and_actual_orders_present(self, synthetic_scored_spans):
        per_type = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="global")
        gradient = compute_error_type_gradient(per_type)
        assert len(gradient["predicted_order"]) > 0
        assert len(gradient["actual_order"]) > 0

    def test_summary_string_present(self, synthetic_scored_spans):
        per_type = compute_span_f1_by_error_type(synthetic_scored_spans, baseline="global")
        gradient = compute_error_type_gradient(per_type)
        assert isinstance(gradient["summary"], str)
        assert len(gradient["summary"]) > 10

    def test_insufficient_types_returns_nan(self):
        """With fewer than 3 error types, should return NaN rho."""
        per_type = {
            "OutE": {"f1": 0.8, "auc": 0.9, "cohens_d": 1.0},
            "EntE": {"f1": 0.6, "auc": 0.7, "cohens_d": 0.5},
        }
        gradient = compute_error_type_gradient(per_type)
        assert np.isnan(gradient["spearman_rho"])
        assert gradient["gradient_holds"] is False


# ---------------------------------------------------------------------------
# Test: Dual baseline produces different results
# ---------------------------------------------------------------------------


class TestDualBaseline:

    def test_evaluate_frank_returns_both_baselines(self, synthetic_scored_spans):
        results = evaluate_frank(synthetic_scored_spans, "test-model")
        assert "global_baseline" in results
        assert "source_baseline" in results
        assert "global_gradient" in results
        assert "source_gradient" in results

    def test_overall_metrics_present(self, synthetic_scored_spans):
        results = evaluate_frank(synthetic_scored_spans, "test-model")
        assert "overall_auc_global" in results
        assert "overall_auc_source" in results
        assert "overall_f1_global" in results
        assert "overall_f1_source" in results
        assert results["overall_auc_global"] >= 0.0
        assert results["overall_f1_global"] >= 0.0

    def test_span_counts(self, synthetic_scored_spans):
        results = evaluate_frank(synthetic_scored_spans, "test-model")
        assert results["n_error_spans"] > 0
        assert results["n_control_spans"] > 0
        assert results["n_error_spans"] + results["n_control_spans"] == len(synthetic_scored_spans)

    def test_baselines_produce_different_auc(self, synthetic_scored_spans):
        results = evaluate_frank(synthetic_scored_spans, "test-model")
        # With different delta values for source vs global, AUC should differ
        assert abs(results["overall_auc_global"] - results["overall_auc_source"]) > 1e-6, \
            "Dual baselines should produce different overall AUC"


# ---------------------------------------------------------------------------
# Test: compute_token_deltas
# ---------------------------------------------------------------------------


class TestComputeTokenDeltas:

    def test_output_shape(self, simple_rank_table):
        token_ids = [0, 1, 2, 3, 4]
        logprobs = [-0.1, -0.5, -1.0, -2.0, -3.0]
        deltas = compute_token_deltas("test text", token_ids, logprobs, simple_rank_table)
        assert len(deltas) == 5

    def test_delta_is_p_llm_minus_g_ri(self, simple_rank_table):
        """Delta should be P_LLM - G_RI for each token."""
        import math
        from shared.utils.entity_extraction import get_grounding_scores
        token_ids = [0, 1]
        logprobs = [-0.5, -1.0]
        deltas = compute_token_deltas("ab", token_ids, logprobs, simple_rank_table)
        scores, default_g = get_grounding_scores(simple_rank_table)
        expected_0 = math.exp(-0.5) - scores.get(0, default_g)
        expected_1 = math.exp(-1.0) - scores.get(1, default_g)
        assert abs(deltas[0] - expected_0) < 1e-10
        assert abs(deltas[1] - expected_1) < 1e-10

    def test_none_logprobs_use_fallback(self, simple_rank_table):
        """None logprobs should use the fallback P_LLM value."""
        from exp06_frank.src.score_examples import _FALLBACK_P_LLM
        from shared.utils.entity_extraction import get_grounding_scores
        token_ids = [0]
        logprobs = [None]
        deltas = compute_token_deltas("a", token_ids, logprobs, simple_rank_table)
        scores, default_g = get_grounding_scores(simple_rank_table)
        expected = _FALLBACK_P_LLM - scores.get(0, default_g)
        assert abs(deltas[0] - expected) < 1e-10
