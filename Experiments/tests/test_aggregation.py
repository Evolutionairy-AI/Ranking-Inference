"""Tests for sequence-level aggregation strategies (Task 3)."""

import pytest
from shared.utils.entity_extraction import EntityGapResult
from shared.utils.aggregation import (
    entity_weighted_mean_delta,
    max_entity_delta,
    proportion_above_threshold,
    aggregate_all,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_entity_results():
    """Two EntityGapResult objects for aggregation testing."""
    mit = EntityGapResult(
        text="MIT",
        entity_type="ORG",
        token_indices=[0, 1, 2],
        mean_delta=0.749,
        max_delta=0.799,
        mean_global_rank=50000.0,
        mean_p_llm=0.75,
        mean_g_ri=0.001,
    )
    dr_smith = EntityGapResult(
        text="Dr. Smith",
        entity_type="PERSON",
        token_indices=[5, 6, 7],
        mean_delta=0.04,
        max_delta=0.05,
        mean_global_rank=500.0,
        mean_p_llm=0.63,
        mean_g_ri=0.59,
    )
    return [mit, dr_smith]


# ---------------------------------------------------------------------------
# entity_weighted_mean_delta
# ---------------------------------------------------------------------------

class TestEntityWeightedMeanDelta:

    def test_mean_of_two_entities(self, two_entity_results):
        result = entity_weighted_mean_delta(two_entity_results)
        expected = (0.749 + 0.04) / 2  # 0.3945
        assert abs(result - expected) < 1e-9

    def test_empty_returns_zero(self):
        assert entity_weighted_mean_delta([]) == 0.0

    def test_single_entity(self, two_entity_results):
        result = entity_weighted_mean_delta([two_entity_results[0]])
        assert abs(result - 0.749) < 1e-9


# ---------------------------------------------------------------------------
# max_entity_delta
# ---------------------------------------------------------------------------

class TestMaxEntityDelta:

    def test_returns_highest_mean_delta(self, two_entity_results):
        result = max_entity_delta(two_entity_results)
        assert abs(result - 0.749) < 1e-9

    def test_empty_returns_zero(self):
        assert max_entity_delta([]) == 0.0

    def test_single_entity(self, two_entity_results):
        result = max_entity_delta([two_entity_results[1]])
        assert abs(result - 0.04) < 1e-9


# ---------------------------------------------------------------------------
# proportion_above_threshold
# ---------------------------------------------------------------------------

class TestProportionAboveThreshold:

    def test_threshold_0_1_half_above(self, two_entity_results):
        # MIT (0.749) above 0.1; Dr. Smith (0.04) not above 0.1 -> 0.5
        result = proportion_above_threshold(two_entity_results, threshold=0.1)
        assert abs(result - 0.5) < 1e-9

    def test_threshold_0_01_all_above(self, two_entity_results):
        # Both 0.749 and 0.04 are above 0.01 -> 1.0
        result = proportion_above_threshold(two_entity_results, threshold=0.01)
        assert abs(result - 1.0) < 1e-9

    def test_threshold_1_0_none_above(self, two_entity_results):
        # Neither 0.749 nor 0.04 exceeds 1.0 -> 0.0
        result = proportion_above_threshold(two_entity_results, threshold=1.0)
        assert abs(result - 0.0) < 1e-9

    def test_empty_returns_zero(self):
        assert proportion_above_threshold([], threshold=0.1) == 0.0


# ---------------------------------------------------------------------------
# aggregate_all
# ---------------------------------------------------------------------------

class TestAggregateAll:

    def test_returns_all_three_keys(self, two_entity_results):
        result = aggregate_all(two_entity_results)
        assert set(result.keys()) == {
            "entity_weighted_mean",
            "max_entity_delta",
            "proportion_above_threshold",
        }

    def test_values_match_individual_functions(self, two_entity_results):
        result = aggregate_all(two_entity_results, threshold=0.1)
        assert abs(result["entity_weighted_mean"] - 0.3945) < 1e-9
        assert abs(result["max_entity_delta"] - 0.749) < 1e-9
        assert abs(result["proportion_above_threshold"] - 0.5) < 1e-9

    def test_default_threshold_is_0_1(self, two_entity_results):
        result_default = aggregate_all(two_entity_results)
        result_explicit = aggregate_all(two_entity_results, threshold=0.1)
        assert result_default == result_explicit

    def test_empty_returns_all_zeros(self):
        result = aggregate_all([])
        assert result["entity_weighted_mean"] == 0.0
        assert result["max_entity_delta"] == 0.0
        assert result["proportion_above_threshold"] == 0.0
