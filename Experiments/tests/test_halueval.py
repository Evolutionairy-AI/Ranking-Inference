"""Tests for HaluEval binary detection experiment (exp04).

All tests use mocks for API calls and dataset loading to run quickly
without external dependencies.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp04_halueval.src.evaluate import (
    compute_halueval_metrics,
    compute_length_matched_auc,
    compute_length_regression,
    evaluate_by_task,
)
from exp04_halueval.src.score_examples import HaluEvalScoredExample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_binary_data():
    """Synthetic labels and scores for binary classification testing."""
    rng = np.random.default_rng(42)
    n = 200
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    scores = np.concatenate([
        rng.normal(0.3, 0.15, n // 2),
        rng.normal(0.7, 0.15, n // 2),
    ])
    scores = np.clip(scores, 0, 1)
    lengths = np.concatenate([
        rng.integers(10, 50, n // 2),
        rng.integers(30, 80, n // 2),
    ])
    return labels, scores, lengths


@pytest.fixture
def mock_halueval_raw_data():
    """Mock raw HaluEval data as returned by HuggingFace datasets."""
    qa_data = [
        {
            "knowledge": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "answer": "Paris is the capital of France.",
            "hallucination": "no",
        },
        {
            "knowledge": "The Earth orbits the Sun.",
            "question": "What does the Earth orbit?",
            "answer": "The Earth orbits the Moon.",
            "hallucination": "yes",
        },
    ]
    return qa_data


@pytest.fixture
def scored_examples():
    """Pre-built scored examples for evaluation testing."""
    examples = []
    tasks = ["qa", "dialogue", "summarization"]
    rng = np.random.default_rng(123)

    for task in tasks:
        for i in range(20):
            is_hallucinated = i % 2  # alternate labels
            if is_hallucinated:
                score_val = rng.uniform(0.3, 0.8)
            else:
                score_val = rng.uniform(0.0, 0.4)
            examples.append(HaluEvalScoredExample(
                example_id=f"{task}_{i}",
                task=task,
                label=is_hallucinated,
                scores={
                    "entity_weighted_mean": score_val,
                    "max_entity_delta": score_val + rng.uniform(0, 0.1),
                    "proportion_above_threshold": rng.uniform(0, 0.5),
                },
                text_length=rng.integers(10, 100),
                model_name="llama-3.1-8b",
                n_entities=rng.integers(1, 5),
            ))
    return examples


# ---------------------------------------------------------------------------
# Test: Dataset loading
# ---------------------------------------------------------------------------

class TestLoadDataset:
    """Tests for HaluEval dataset loading."""

    @patch("exp04_halueval.src.load_dataset.datasets.load_dataset")
    def test_load_halueval_split_qa(self, mock_load, mock_halueval_raw_data):
        """Loading QA split returns correct structure."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(mock_halueval_raw_data))
        mock_ds.__len__ = MagicMock(return_value=len(mock_halueval_raw_data))
        mock_ds.select = MagicMock(return_value=mock_ds)
        mock_load.return_value = mock_ds

        from exp04_halueval.src.load_dataset import load_halueval_split

        examples = load_halueval_split("qa", max_examples=2)

        assert len(examples) == 2
        ex = examples[0]
        assert ex["example_id"] == "qa_0"
        assert ex["task"] == "qa"
        assert "prompt" in ex
        assert "response_text" in ex
        assert "label" in ex
        assert ex["label"] == 0  # "no" hallucination
        assert examples[1]["label"] == 1  # "yes" hallucination

    @patch("exp04_halueval.src.load_dataset.datasets.load_dataset")
    def test_load_halueval_split_dialogue(self, mock_load):
        """Loading dialogue split maps fields correctly."""
        dialogue_data = [{
            "knowledge": "Dogs are pets.",
            "dialogue_history": "User: Do you have a pet?",
            "response": "Yes, I have a dog.",
            "hallucination": "no",
        }]
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(dialogue_data))
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_ds.select = MagicMock(return_value=mock_ds)
        mock_load.return_value = mock_ds

        from exp04_halueval.src.load_dataset import load_halueval_split

        examples = load_halueval_split("dialogue")
        assert len(examples) == 1
        assert examples[0]["task"] == "dialogue"
        assert examples[0]["label"] == 0
        assert "dog" in examples[0]["response_text"]

    @patch("exp04_halueval.src.load_dataset.datasets.load_dataset")
    def test_load_halueval_split_summarization(self, mock_load):
        """Loading summarization split maps fields correctly."""
        summ_data = [{
            "document": "The quick brown fox jumped over the lazy dog.",
            "summary": "A fox jumped over a dog.",
            "hallucination": "yes",
        }]
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(summ_data))
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_ds.select = MagicMock(return_value=mock_ds)
        mock_load.return_value = mock_ds

        from exp04_halueval.src.load_dataset import load_halueval_split

        examples = load_halueval_split("summarization")
        assert len(examples) == 1
        assert examples[0]["task"] == "summarization"
        assert examples[0]["label"] == 1

    def test_load_invalid_task(self):
        """Loading an invalid task raises ValueError."""
        from exp04_halueval.src.load_dataset import load_halueval_split

        with pytest.raises(ValueError, match="Unknown task"):
            load_halueval_split("invalid_task")


# ---------------------------------------------------------------------------
# Test: Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    """Tests for HaluEval scoring pipeline."""

    def test_scored_example_structure(self):
        """HaluEvalScoredExample has all required fields."""
        ex = HaluEvalScoredExample(
            example_id="qa_0",
            task="qa",
            label=1,
            scores={"entity_weighted_mean": 0.5, "max_entity_delta": 0.6,
                    "proportion_above_threshold": 0.3},
            text_length=25,
            model_name="llama-3.1-8b",
            n_entities=3,
        )
        assert ex.example_id == "qa_0"
        assert ex.label == 1
        assert ex.scores["entity_weighted_mean"] == 0.5
        assert ex.model_name == "llama-3.1-8b"

    @patch("exp04_halueval.src.score_examples.score_text_logprobs")
    @patch("exp04_halueval.src.score_examples.compute_entity_gaps")
    def test_score_example_produces_result(self, mock_gaps, mock_logprobs):
        """score_example returns a valid HaluEvalScoredExample."""
        from exp04_halueval.src.score_examples import score_example
        from shared.utils import EntityGapResult

        mock_scoring_result = MagicMock()
        mock_scoring_result.token_ids = [1, 2, 3, 4, 5]
        mock_scoring_result.logprobs = [-1.0, -2.0, -0.5, -1.5, -3.0]
        mock_logprobs.return_value = mock_scoring_result

        mock_gaps.return_value = [
            EntityGapResult(
                text="Paris",
                entity_type="GPE",
                token_indices=[0, 1],
                mean_delta=0.3,
                max_delta=0.5,
                mean_global_rank=100.0,
                mean_p_llm=0.4,
                mean_g_ri=0.1,
            )
        ]

        example = {
            "example_id": "qa_0",
            "task": "qa",
            "prompt": "What is the capital?",
            "response_text": "Paris is the capital.",
            "label": 0,
        }

        result = score_example(
            example=example,
            model_name="llama-3.1-8b",
            rank_table=MagicMock(),
            tokenizer=MagicMock(),
            tokenizer_name="llama-3.1-8b",
        )

        assert isinstance(result, HaluEvalScoredExample)
        assert result.example_id == "qa_0"
        assert result.label == 0
        assert "entity_weighted_mean" in result.scores
        assert result.text_length == 5
        assert result.n_entities == 1


# ---------------------------------------------------------------------------
# Test: Metrics computation
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for evaluation metrics."""

    def test_compute_halueval_metrics_returns_auc(self, synthetic_binary_data):
        labels, scores, _ = synthetic_binary_data
        result = compute_halueval_metrics(labels, scores, n_bootstrap=100)

        assert "roc_auc" in result
        assert 0.5 < result["roc_auc"] <= 1.0
        assert "roc_auc_ci" in result
        assert len(result["roc_auc_ci"]) == 2
        assert "pr_auc" in result
        assert "f1_optimal" in result
        assert "cohens_d" in result
        assert result["cohens_d"] > 0

    def test_compute_halueval_metrics_degenerate(self):
        labels = np.zeros(20)
        scores = np.random.rand(20)
        result = compute_halueval_metrics(labels, scores, n_bootstrap=50)
        assert result["roc_auc"] == 0.5

    def test_compute_length_matched_auc(self, synthetic_binary_data):
        labels, scores, lengths = synthetic_binary_data
        result = compute_length_matched_auc(labels, scores, lengths, n_bins=5)

        assert "matched_roc_auc" in result
        assert "n_matched" in result
        assert "n_bins_used" in result
        assert 0.0 <= result["matched_roc_auc"] <= 1.0
        assert result["n_matched"] > 0

    def test_compute_length_matched_auc_no_overlap(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        lengths = np.array([1, 2, 3, 100, 200, 300])
        result = compute_length_matched_auc(labels, scores, lengths, n_bins=6)
        assert "matched_roc_auc" in result

    def test_compute_length_regression(self, synthetic_binary_data):
        labels, scores, lengths = synthetic_binary_data
        result = compute_length_regression(labels, scores, lengths)

        assert "auc_length_only" in result
        assert "auc_ri_only" in result
        assert "auc_ri_plus_length" in result
        assert "delta_auc" in result
        assert result["delta_auc"] >= -0.05
        assert result["auc_ri_only"] > 0.6

    def test_compute_length_regression_zero_variance(self):
        labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.6, 0.15, 0.9])
        lengths = np.array([50, 50, 50, 50, 50, 50, 50, 50])
        result = compute_length_regression(labels, scores, lengths)
        assert "delta_auc" in result


# ---------------------------------------------------------------------------
# Test: Task-stratified evaluation
# ---------------------------------------------------------------------------

class TestEvaluateByTask:
    """Tests for task-stratified evaluation."""

    def test_evaluate_by_task_stratifies(self, scored_examples):
        results = evaluate_by_task(scored_examples)
        assert "qa" in results
        assert "dialogue" in results
        assert "summarization" in results
        assert "all" in results

    def test_evaluate_by_task_metrics_present(self, scored_examples):
        results = evaluate_by_task(scored_examples)
        for task_name in ["qa", "dialogue", "summarization", "all"]:
            task_result = results[task_name]
            assert "metrics" in task_result
            assert "length_matched" in task_result
            assert "length_regression" in task_result
            assert "n_examples" in task_result
            assert "roc_auc" in task_result["metrics"]

    def test_evaluate_by_task_correct_counts(self, scored_examples):
        results = evaluate_by_task(scored_examples)
        # 20 examples per task, each is one data point
        assert results["qa"]["metrics"]["n_samples"] == 20
        assert results["dialogue"]["metrics"]["n_samples"] == 20
        assert results["summarization"]["metrics"]["n_samples"] == 20
        assert results["all"]["metrics"]["n_samples"] == 60

    def test_evaluate_by_task_with_dicts(self):
        examples = [
            {
                "example_id": "qa_0",
                "task": "qa",
                "label": 0,
                "scores": {"entity_weighted_mean": 0.1,
                           "max_entity_delta": 0.2,
                           "proportion_above_threshold": 0.0},
                "text_length": 20,
                "model_name": "llama-3.1-8b",
                "n_entities": 1,
            },
            {
                "example_id": "qa_1",
                "task": "qa",
                "label": 1,
                "scores": {"entity_weighted_mean": 0.6,
                           "max_entity_delta": 0.7,
                           "proportion_above_threshold": 0.5},
                "text_length": 40,
                "model_name": "llama-3.1-8b",
                "n_entities": 2,
            },
        ]
        results = evaluate_by_task(examples)
        assert "qa" in results
        assert "all" in results
        assert results["qa"]["metrics"]["n_samples"] == 2

    def test_different_aggregation_strategies(self, scored_examples):
        results_ewm = evaluate_by_task(scored_examples, "entity_weighted_mean")
        results_max = evaluate_by_task(scored_examples, "max_entity_delta")
        results_pat = evaluate_by_task(scored_examples, "proportion_above_threshold")
        assert results_ewm["all"]["metrics"]["roc_auc"] > 0
        assert results_max["all"]["metrics"]["roc_auc"] > 0
        assert results_pat["all"]["metrics"]["roc_auc"] > 0
