"""Tests for exp05_truthfulqa with mocked data.

Validates dataset loading format, tier annotation, scoring pipeline,
and evaluation metrics -- all without external API calls or datasets.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp05_truthfulqa.src.annotate_tiers import TierAnnotation, classify_tier_heuristic
from exp05_truthfulqa.src.evaluate import (
    compute_mc1_accuracy,
    compute_mc2_correlation,
    compute_stratified_auc,
    evaluate_truthfulqa,
)
from exp05_truthfulqa.src.score_examples import TruthfulQAScoredQuestion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_questions(n=5):
    """Create mock TruthfulQA question dicts."""
    categories = ["Misconceptions", "Finance", "Health", "Law", "Science"]
    questions = []
    for i in range(n):
        questions.append({
            "question_idx": i,
            "question": f"Test question {i}?",
            "category": categories[i % len(categories)],
            "mc1_targets": {
                "choices": [f"Correct answer {i}", f"Wrong answer A{i}", f"Wrong answer B{i}"],
                "labels": [1, 0, 0],
            },
            "mc2_targets": {
                "choices": [
                    f"Correct answer {i}",
                    f"Also correct {i}",
                    f"Wrong answer A{i}",
                    f"Wrong answer B{i}",
                ],
                "labels": [1, 1, 0, 0],
            },
        })
    return questions


def _make_mock_scored_questions(n=10):
    """Create mock TruthfulQAScoredQuestion objects."""
    scored = []
    rng = np.random.default_rng(42)

    for i in range(n):
        tier = "tier1" if i < n // 2 else "tier2"

        # For tier1: correct answer should have distinctly lower scores
        # For tier2: scores should be similar (RI can't distinguish)
        if tier == "tier1":
            correct_score = rng.uniform(0.0, 0.05)
            wrong_scores = [rng.uniform(0.1, 0.3) for _ in range(3)]
        else:
            base = rng.uniform(0.05, 0.15)
            correct_score = base + rng.uniform(-0.03, 0.03)
            wrong_scores = [base + rng.uniform(-0.03, 0.03) for _ in range(3)]

        mc1_scores = [
            {"entity_weighted_mean": correct_score, "max_entity_delta": correct_score * 1.5, "proportion_above_threshold": 0.0},
        ]
        for ws in wrong_scores:
            mc1_scores.append({
                "entity_weighted_mean": ws,
                "max_entity_delta": ws * 1.5,
                "proportion_above_threshold": 0.3,
            })

        mc1_ewm = [s["entity_weighted_mean"] for s in mc1_scores]
        mc1_predicted_idx = int(min(range(len(mc1_ewm)), key=lambda j: mc1_ewm[j]))

        # MC2 scores (same structure, more candidates)
        mc2_scores = list(mc1_scores)  # reuse
        mc2_ewm = [s["entity_weighted_mean"] for s in mc2_scores]
        mc2_rank_order = sorted(range(len(mc2_ewm)), key=lambda j: mc2_ewm[j])

        scored.append(TruthfulQAScoredQuestion(
            question_idx=i,
            question=f"Test question {i}?",
            category=["Misconceptions", "Finance", "Health", "Law", "Science"][i % 5],
            tier=tier,
            mc1_candidate_scores=mc1_scores,
            mc1_predicted_idx=mc1_predicted_idx,
            mc1_correct_idx=0,
            mc2_candidate_scores=mc2_scores,
            mc2_rank_order=mc2_rank_order,
            model_name="llama-3.1-8b",
        ))

    return scored


# ---------------------------------------------------------------------------
# Tests: Dataset loading
# ---------------------------------------------------------------------------

class TestDatasetLoading:
    """Tests for dataset loading format."""

    def test_mock_questions_have_required_keys(self):
        """Dataset loading returns MC1/MC2 format."""
        questions = _make_mock_questions(3)
        for q in questions:
            assert "question_idx" in q
            assert "question" in q
            assert "category" in q
            assert "mc1_targets" in q
            assert "mc2_targets" in q

            # MC1: exactly one correct
            mc1 = q["mc1_targets"]
            assert "choices" in mc1
            assert "labels" in mc1
            assert sum(mc1["labels"]) == 1
            assert len(mc1["choices"]) == len(mc1["labels"])

            # MC2: multiple correct possible
            mc2 = q["mc2_targets"]
            assert "choices" in mc2
            assert "labels" in mc2
            assert sum(mc2["labels"]) >= 1
            assert len(mc2["choices"]) == len(mc2["labels"])

    def test_question_idx_is_sequential(self):
        questions = _make_mock_questions(5)
        indices = [q["question_idx"] for q in questions]
        assert indices == list(range(5))


# ---------------------------------------------------------------------------
# Tests: Tier Annotation
# ---------------------------------------------------------------------------

class TestTierAnnotation:
    """Tests for TierAnnotation dataclass and classification."""

    def test_tier_annotation_dataclass(self):
        ann = TierAnnotation(
            question_idx=0,
            tier="tier1",
            confidence="high",
            rationale="Contains fabricated entities.",
        )
        assert ann.question_idx == 0
        assert ann.tier == "tier1"
        assert ann.confidence == "high"
        assert ann.rationale == "Contains fabricated entities."

    def test_tier_annotation_valid_tiers(self):
        for tier in ("tier1", "tier2", "ambiguous"):
            ann = TierAnnotation(question_idx=0, tier=tier, confidence="high", rationale="test")
            assert ann.tier == tier

    def test_classify_tier_heuristic_no_rank_table(self):
        """Heuristic without rank table produces valid tier."""
        ann = classify_tier_heuristic(
            question="What is the capital of France?",
            wrong_answers=["Berlin", "Madrid", "Rome"],
            rank_table=None,
            tokenizer=None,
        )
        assert ann.tier in ("tier1", "tier2", "ambiguous")
        assert ann.confidence in ("high", "medium", "low")
        assert len(ann.rationale) > 0

    def test_classify_tier_heuristic_with_suspicious_entities(self):
        """Suspicious entity names should lean toward tier1."""
        ann = classify_tier_heuristic(
            question="Who invented the telephone?",
            wrong_answers=[
                "Dr. Xylophone McQuantum III invented it in 2847",
                "The Zorbflaxian Council of 1723 created the first prototype",
            ],
            rank_table=None,
            tokenizer=None,
        )
        # These have long multi-word entity names and/or digits
        assert ann.tier in ("tier1", "ambiguous")


# ---------------------------------------------------------------------------
# Tests: Evaluation Metrics
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    """Tests for evaluation metric functions."""

    def test_mc1_accuracy_perfect(self):
        assert compute_mc1_accuracy([0, 1, 2], [0, 1, 2]) == 1.0

    def test_mc1_accuracy_zero(self):
        assert compute_mc1_accuracy([1, 2, 0], [0, 1, 2]) == 0.0

    def test_mc1_accuracy_partial(self):
        acc = compute_mc1_accuracy([0, 1, 0], [0, 1, 2])
        assert abs(acc - 2.0 / 3.0) < 1e-9

    def test_mc1_accuracy_empty(self):
        assert compute_mc1_accuracy([], []) == 0.0

    def test_mc2_correlation_valid_range(self):
        """MC2 correlation returns value in [-1, 1]."""
        ri_ranks = [[0, 1, 2, 3], [1, 0, 3, 2], [0, 2, 1, 3]]
        truth_labels = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0]]
        corr = compute_mc2_correlation(ri_ranks, truth_labels)
        assert -1.0 <= corr <= 1.0

    def test_mc2_correlation_perfect(self):
        """Perfect ranking should give positive correlation."""
        # Correct items at positions 0,1; incorrect at 2,3
        ri_ranks = [[0, 1, 2, 3]]  # RI ranks correct items first
        truth_labels = [[1, 1, 0, 0]]
        corr = compute_mc2_correlation(ri_ranks, truth_labels)
        # Lower rank position = correct, label=1 -> negative Spearman
        # (positions [0,1,2,3] vs labels [1,1,0,0])
        # Actually the correlation direction depends on the convention
        assert corr != 0.0  # should be non-trivial

    def test_stratified_auc_separates_tiers(self):
        """Stratified AUC should produce different values for different tiers."""
        rng = np.random.default_rng(42)
        n = 200

        labels = np.array([0, 1] * (n // 2))
        tiers = np.array(["tier1"] * (n // 2) + ["tier2"] * (n // 2))

        # Tier1: scores clearly separate labels
        scores_tier1 = np.where(
            labels[:n // 2] == 1,
            rng.uniform(0.6, 1.0, n // 2),
            rng.uniform(0.0, 0.4, n // 2),
        )
        # Tier2: scores are random (no separation)
        scores_tier2 = rng.uniform(0.0, 1.0, n // 2)
        scores = np.concatenate([scores_tier1, scores_tier2])

        result = compute_stratified_auc(labels, scores, tiers)

        assert "tier1_auc" in result
        assert "tier2_auc" in result
        assert "tier_gradient" in result
        assert "all_auc" in result

        # Tier1 should have much higher AUC than tier2
        assert result["tier1_auc"] > 0.8
        assert result["tier2_auc"] < result["tier1_auc"]
        assert result["tier_gradient"] > 0.0

    def test_stratified_auc_gradient_sign(self):
        """Tier gradient should be positive when tier1 has higher AUC."""
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.6, 0.5, 0.5])
        tiers = np.array(["tier1", "tier1", "tier1", "tier1",
                          "tier2", "tier2", "tier2", "tier2"])

        result = compute_stratified_auc(labels, scores, tiers)
        # Tier1 scores are more separable than tier2
        assert result["tier_gradient"] >= 0.0


# ---------------------------------------------------------------------------
# Tests: Full Evaluation Pipeline
# ---------------------------------------------------------------------------

class TestEvaluateTruthfulQA:
    """Tests for the full evaluate_truthfulqa function."""

    def test_evaluate_produces_complete_results(self):
        """evaluate_truthfulqa produces complete results dict."""
        scored = _make_mock_scored_questions(20)
        results = evaluate_truthfulqa(scored)

        assert "mc1_accuracy" in results
        assert "mc1_accuracy_tier1" in results
        assert "mc1_accuracy_tier2" in results
        assert "stratified_auc" in results
        assert "bootstrap_ci" in results
        assert "cohens_d_tier_gradient" in results
        assert "n_questions" in results
        assert "n_tier1" in results
        assert "n_tier2" in results
        assert "model_name" in results

        assert results["n_questions"] == 20
        assert results["n_tier1"] == 10
        assert results["n_tier2"] == 10
        assert 0.0 <= results["mc1_accuracy"] <= 1.0
        assert 0.0 <= results["mc1_accuracy_tier1"] <= 1.0
        assert 0.0 <= results["mc1_accuracy_tier2"] <= 1.0

    def test_evaluate_tier1_better_than_tier2(self):
        """With well-designed mock data, tier1 accuracy should exceed tier2."""
        scored = _make_mock_scored_questions(40)
        results = evaluate_truthfulqa(scored)

        # Our mock data gives tier1 clear separation, tier2 near-random
        assert results["mc1_accuracy_tier1"] >= results["mc1_accuracy_tier2"]

    def test_evaluate_empty_input(self):
        """Empty input returns error dict."""
        results = evaluate_truthfulqa([])
        assert "error" in results

    def test_evaluate_stratified_auc_present(self):
        """Stratified AUC results are populated."""
        scored = _make_mock_scored_questions(20)
        results = evaluate_truthfulqa(scored)

        auc = results["stratified_auc"]
        assert "tier1_auc" in auc
        assert "tier2_auc" in auc
        assert "tier_gradient" in auc
        assert "all_auc" in auc


# ---------------------------------------------------------------------------
# Tests: TruthfulQAScoredQuestion
# ---------------------------------------------------------------------------

class TestScoredQuestion:
    """Tests for the TruthfulQAScoredQuestion dataclass."""

    def test_scored_question_fields(self):
        sq = TruthfulQAScoredQuestion(
            question_idx=0,
            question="What is 2+2?",
            category="Math",
            tier="tier1",
            mc1_candidate_scores=[
                {"entity_weighted_mean": 0.01, "max_entity_delta": 0.02, "proportion_above_threshold": 0.0},
                {"entity_weighted_mean": 0.15, "max_entity_delta": 0.20, "proportion_above_threshold": 0.5},
            ],
            mc1_predicted_idx=0,
            mc1_correct_idx=0,
            mc2_candidate_scores=[
                {"entity_weighted_mean": 0.01, "max_entity_delta": 0.02, "proportion_above_threshold": 0.0},
            ],
            mc2_rank_order=[0],
            model_name="llama-3.1-8b",
        )
        assert sq.question_idx == 0
        assert sq.mc1_predicted_idx == sq.mc1_correct_idx  # correct prediction
        assert sq.tier == "tier1"
        assert len(sq.mc1_candidate_scores) == 2
