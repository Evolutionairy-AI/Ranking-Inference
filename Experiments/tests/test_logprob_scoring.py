"""Tests for multi-model logprob scoring utility."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from shared.utils.logprob_scoring import (
    ModelConfig,
    ScoringResult,
    score_text_logprobs,
    get_model_config,
    SUPPORTED_MODELS,
    score_batch,
)


class TestModelConfig:
    def test_supported_models_exist(self):
        assert "gpt-5.1" in SUPPORTED_MODELS
        assert "claude-sonnet-4" in SUPPORTED_MODELS
        assert "llama-3.1-8b" in SUPPORTED_MODELS

    def test_get_model_config_valid(self):
        config = get_model_config("gpt-5.1")
        assert isinstance(config, ModelConfig)
        assert config.provider == "openai"
        assert config.tokenizer_name == "gpt-5.1"

    def test_get_model_config_ollama(self):
        config = get_model_config("llama-3.1-8b")
        assert config.provider == "ollama"

    def test_get_model_config_anthropic(self):
        config = get_model_config("claude-sonnet-4")
        assert config.provider == "anthropic"

    def test_get_model_config_invalid(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            get_model_config("nonexistent-model")


class TestScoringResult:
    def test_fields(self):
        result = ScoringResult(
            text="hello world",
            token_ids=[1, 2],
            tokens=["hello", " world"],
            logprobs=[-0.5, -1.2],
            model_name="gpt-5.1",
        )
        assert len(result.token_ids) == len(result.logprobs)
        assert result.model_name == "gpt-5.1"
        assert result.text == "hello world"

    def test_none_logprobs(self):
        result = ScoringResult(
            text="test",
            token_ids=[1],
            tokens=["test"],
            logprobs=[None],
            model_name="claude-sonnet-4",
        )
        assert result.logprobs[0] is None


class TestScoreTextDispatch:
    @patch("shared.utils.logprob_scoring._score_openai")
    def test_dispatches_to_openai(self, mock_score):
        mock_score.return_value = ScoringResult(
            text="test", token_ids=[1], tokens=["test"],
            logprobs=[-0.1], model_name="gpt-5.1",
        )
        result = score_text_logprobs("test", "gpt-5.1")
        mock_score.assert_called_once()
        assert result.model_name == "gpt-5.1"

    @patch("shared.utils.logprob_scoring._score_openai")
    def test_dispatches_ollama_to_openai(self, mock_score):
        mock_score.return_value = ScoringResult(
            text="test", token_ids=[1], tokens=["test"],
            logprobs=[-0.1], model_name="llama-3.1-8b",
        )
        result = score_text_logprobs("test", "llama-3.1-8b")
        mock_score.assert_called_once()
        assert result.model_name == "llama-3.1-8b"

    @patch("shared.utils.logprob_scoring._score_anthropic")
    def test_dispatches_to_anthropic(self, mock_score):
        mock_score.return_value = ScoringResult(
            text="test", token_ids=[1], tokens=["test"],
            logprobs=[None], model_name="claude-sonnet-4",
        )
        result = score_text_logprobs("test", "claude-sonnet-4")
        mock_score.assert_called_once()
        assert result.model_name == "claude-sonnet-4"

    def test_empty_text_returns_empty(self):
        result = score_text_logprobs("", "gpt-5.1")
        assert result.token_ids == []
        assert result.logprobs == []

    def test_whitespace_text_returns_empty(self):
        result = score_text_logprobs("   ", "gpt-5.1")
        assert result.token_ids == []


class TestScoreBatch:
    @patch("shared.utils.logprob_scoring.score_text_logprobs")
    def test_batch_scoring(self, mock_score):
        mock_score.return_value = ScoringResult(
            text="t", token_ids=[1], tokens=["t"],
            logprobs=[-0.1], model_name="gpt-5.1",
        )
        results = score_batch(["a", "b", "c"], "gpt-5.1", delay=0)
        assert len(results) == 3
        assert mock_score.call_count == 3
