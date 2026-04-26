"""Tests for entity extraction pipeline (Task 2)."""

import pytest
import tiktoken
import numpy as np

from shared.utils.entity_extraction import (
    EntitySpan,
    EntityGapResult,
    extract_entities,
    align_entities_to_tokens,
    compute_entity_gaps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# TestExtractEntities
# ---------------------------------------------------------------------------

class TestExtractEntities:

    def test_finds_entities_in_sample_text(self, sample_text_with_entities):
        entities = extract_entities(sample_text_with_entities)
        assert len(entities) > 0
        entity_texts = [e.text for e in entities]
        # Should find at least one of these known entities
        found_any = any(
            known in " ".join(entity_texts)
            for known in ["MIT", "Cambridge", "Massachusetts", "Nature", "CRISPR"]
        )
        assert found_any, f"Expected known entities, got: {entity_texts}"

    def test_entities_have_correct_fields(self, sample_text_with_entities):
        entities = extract_entities(sample_text_with_entities)
        for ent in entities:
            assert isinstance(ent, EntitySpan)
            assert isinstance(ent.text, str) and len(ent.text) > 0
            assert isinstance(ent.entity_type, str) and len(ent.entity_type) > 0
            assert isinstance(ent.char_start, int)
            assert isinstance(ent.char_end, int)
            assert ent.char_end > ent.char_start

    def test_handles_empty_string(self):
        entities = extract_entities("")
        assert entities == []


# ---------------------------------------------------------------------------
# TestAlignEntitiesToTokens
# ---------------------------------------------------------------------------

class TestAlignEntitiesToTokens:

    def test_produces_valid_token_indices(self, sample_text_with_entities, tokenizer):
        entities = extract_entities(sample_text_with_entities)
        token_ids = tokenizer.encode(sample_text_with_entities)
        aligned = align_entities_to_tokens(
            entities, sample_text_with_entities, token_ids, tokenizer
        )
        for entity_span, indices in aligned:
            assert isinstance(entity_span, EntitySpan)
            assert isinstance(indices, list)
            assert len(indices) > 0, f"Entity '{entity_span.text}' got no token indices"
            for idx in indices:
                assert 0 <= idx < len(token_ids), (
                    f"Index {idx} out of bounds for {len(token_ids)} tokens"
                )


# ---------------------------------------------------------------------------
# TestComputeEntityGaps
# ---------------------------------------------------------------------------

class TestComputeEntityGaps:

    def test_produces_entity_gap_results(
        self, sample_text_with_entities, sample_rank_table, tokenizer
    ):
        token_ids = tokenizer.encode(sample_text_with_entities)
        # Simulate logprobs (one per token)
        logprobs = [np.log(0.1)] * len(token_ids)
        results = compute_entity_gaps(
            sample_text_with_entities,
            token_ids,
            logprobs,
            tokenizer,
            sample_rank_table,
        )
        assert len(results) > 0
        for r in results:
            assert isinstance(r, EntityGapResult)
            assert isinstance(r.mean_delta, float)
            assert isinstance(r.max_delta, float)
            assert isinstance(r.mean_global_rank, float)

    def test_handles_no_entity_text(self, sample_rank_table, tokenizer):
        text = "the the the the the"
        token_ids = tokenizer.encode(text)
        logprobs = [np.log(0.5)] * len(token_ids)
        results = compute_entity_gaps(
            text, token_ids, logprobs, tokenizer, sample_rank_table
        )
        assert results == []

    def test_handles_none_logprobs(
        self, sample_text_with_entities, sample_rank_table, tokenizer
    ):
        token_ids = tokenizer.encode(sample_text_with_entities)
        # All logprobs are None
        logprobs = [None] * len(token_ids)
        results = compute_entity_gaps(
            sample_text_with_entities,
            token_ids,
            logprobs,
            tokenizer,
            sample_rank_table,
        )
        # Should still produce results (using fallback probability)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, EntityGapResult)
            assert r.mean_p_llm == pytest.approx(1.0 / 50000, rel=1e-3)
