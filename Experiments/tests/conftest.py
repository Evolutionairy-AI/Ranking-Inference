"""Shared fixtures for Tier 2 infrastructure tests."""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils import RankTable, build_rank_table


@pytest.fixture
def sample_rank_table():
    """A small rank table for testing (simulates Wikipedia baseline).

    Contains 100 tokens with Zipf-like frequency distribution.
    Token IDs 0-99, where token 0 is most frequent.
    """
    # Simulate Zipf-like frequencies: freq(r) ~ 1000/r
    token_ids = []
    for tid in range(100):
        freq = max(1, int(1000 / (tid + 1)))
        token_ids.extend([tid] * freq)

    return build_rank_table(token_ids, "test-tokenizer", "test-corpus")


@pytest.fixture
def sample_text_with_entities():
    """Sample text containing known named entities for NER testing."""
    return (
        "Dr. Helena Marchetti of the Nexorvatin Research Institute "
        "published a paper in Nature about CRISPR gene editing. "
        "The study was conducted at MIT in Cambridge, Massachusetts."
    )


@pytest.fixture
def sample_text_no_entities():
    """Sample text with no named entities."""
    return "The quick brown fox jumps over the lazy dog repeatedly."


@pytest.fixture
def sample_token_gaps():
    """Pre-computed token-level gap data for aggregation testing.

    Returns a list of dicts simulating compute_gap output with entity info.
    Tokens 0-4: entity "MIT" (tokens 0,1,2 are entity, 3,4 are not)
    Tokens 5-9: entity "Dr. Smith" (tokens 5,6,7 are entity, 8,9 are not)
    """
    return [
        # Entity: "MIT" -- high delta (anomalous)
        {"token": "M", "p_llm": 0.8, "g_ri": 0.001, "delta": 0.799,
         "entity": "MIT", "entity_type": "ORG", "is_entity": True},
        {"token": "I", "p_llm": 0.7, "g_ri": 0.002, "delta": 0.698,
         "entity": "MIT", "entity_type": "ORG", "is_entity": True},
        {"token": "T", "p_llm": 0.75, "g_ri": 0.001, "delta": 0.749,
         "entity": "MIT", "entity_type": "ORG", "is_entity": True},
        # Non-entity tokens -- low delta
        {"token": " is", "p_llm": 0.9, "g_ri": 0.85, "delta": 0.05,
         "entity": None, "entity_type": None, "is_entity": False},
        {"token": " a", "p_llm": 0.95, "g_ri": 0.9, "delta": 0.05,
         "entity": None, "entity_type": None, "is_entity": False},
        # Entity: "Dr. Smith" -- low delta (well-grounded)
        {"token": "Dr", "p_llm": 0.6, "g_ri": 0.55, "delta": 0.05,
         "entity": "Dr. Smith", "entity_type": "PERSON", "is_entity": True},
        {"token": ".", "p_llm": 0.8, "g_ri": 0.78, "delta": 0.02,
         "entity": "Dr. Smith", "entity_type": "PERSON", "is_entity": True},
        {"token": " Smith", "p_llm": 0.5, "g_ri": 0.45, "delta": 0.05,
         "entity": "Dr. Smith", "entity_type": "PERSON", "is_entity": True},
        # Non-entity tokens
        {"token": " said", "p_llm": 0.85, "g_ri": 0.82, "delta": 0.03,
         "entity": None, "entity_type": None, "is_entity": False},
        {"token": " that", "p_llm": 0.9, "g_ri": 0.88, "delta": 0.02,
         "entity": None, "entity_type": None, "is_entity": False},
    ]
