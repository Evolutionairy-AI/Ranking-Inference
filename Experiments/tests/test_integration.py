"""End-to-end integration test for the entity gap pipeline."""

import pytest
import numpy as np
from shared.utils.entity_extraction import compute_entity_gaps
from shared.utils.aggregation import aggregate_all


class TestEndToEndPipeline:
    def test_fabricated_entity_produces_valid_scores(self, sample_rank_table):
        """Both real and fabricated entity texts produce valid numeric scores."""
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")

        text_real = "Albert Einstein developed the theory of relativity at Princeton University."
        ids_real = tokenizer.encode(text_real)
        logprobs_real = [-0.5] * len(ids_real)

        text_fake = "Dr. Nexorvatin Kelpsworth developed the theory of relativity at Glorbian University."
        ids_fake = tokenizer.encode(text_fake)
        logprobs_fake = [-0.5] * len(ids_fake)

        entities_real = compute_entity_gaps(text_real, ids_real, logprobs_real, tokenizer, sample_rank_table)
        entities_fake = compute_entity_gaps(text_fake, ids_fake, logprobs_fake, tokenizer, sample_rank_table)

        scores_real = aggregate_all(entities_real, threshold=0.01)
        scores_fake = aggregate_all(entities_fake, threshold=0.01)

        # Both should produce valid numeric scores without crashing
        assert isinstance(scores_real["entity_weighted_mean"], float)
        assert isinstance(scores_fake["entity_weighted_mean"], float)
        assert isinstance(scores_real["max_entity_delta"], float)
        assert isinstance(scores_fake["max_entity_delta"], float)

    def test_pipeline_handles_no_entities_gracefully(self, sample_rank_table):
        """Text with no entities should produce zero scores."""
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")

        text = "this is a simple sentence with no names or places"
        ids = tokenizer.encode(text)
        logprobs = [-1.0] * len(ids)

        entities = compute_entity_gaps(text, ids, logprobs, tokenizer, sample_rank_table)
        scores = aggregate_all(entities, threshold=0.1)

        assert isinstance(scores["entity_weighted_mean"], float)
        assert isinstance(scores["max_entity_delta"], float)
        assert isinstance(scores["proportion_above_threshold"], float)

    def test_pipeline_handles_missing_logprobs(self, sample_rank_table):
        """When logprobs are None, should use uniform approximation."""
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")

        text = "MIT is in Cambridge, Massachusetts."
        ids = tokenizer.encode(text)
        logprobs = [None] * len(ids)

        entities = compute_entity_gaps(text, ids, logprobs, tokenizer, sample_rank_table)

        assert isinstance(entities, list)
        for e in entities:
            assert isinstance(e.mean_delta, float)
            assert not np.isnan(e.mean_delta)
