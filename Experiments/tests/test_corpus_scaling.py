"""Tests for streaming Wikipedia corpus processing (Task 6)."""

import pytest
import sys
from pathlib import Path
from collections import Counter
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils.rank_utils import build_rank_table
from shared.utils.corpus_utils import tokenize_text, get_tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_wikipedia_articles(n=10):
    """Generate synthetic article-like strings for offline testing."""
    sentences = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "Wikipedia is a free online encyclopedia that anyone can edit. " * 5,
        "Natural language processing is a subfield of artificial intelligence. " * 5,
        "Machine learning models can be trained on large corpora of text data. " * 5,
        "The Mandelbrot set is a famous fractal that exhibits self-similarity. " * 5,
        "Tokenization is the process of splitting text into smaller units. " * 5,
        "Rank frequency distributions follow a power law in natural language. " * 5,
        "Deep neural networks have achieved state-of-the-art performance. " * 5,
        "Information theory provides a framework for measuring entropy. " * 5,
        "Language models assign probabilities to sequences of tokens. " * 5,
    ]
    return sentences[:n]


# ---------------------------------------------------------------------------
# TestStreamWikipediaArticles
# ---------------------------------------------------------------------------

class TestStreamWikipediaArticles:
    """Tests for stream_wikipedia_articles()."""

    def test_yields_strings(self):
        """stream_wikipedia_articles yields string objects."""
        from shared.utils.corpus_scaling import stream_wikipedia_articles

        fake_rows = [{"text": t} for t in _fake_wikipedia_articles(10)]

        with patch("shared.utils.corpus_scaling.load_dataset") as mock_ld:
            mock_ld.return_value = iter(fake_rows)
            articles = list(stream_wikipedia_articles(max_articles=5))

        assert len(articles) > 0
        for art in articles:
            assert isinstance(art, str), f"Expected str, got {type(art)}"

    def test_respects_max_articles(self):
        """stream_wikipedia_articles stops after max_articles."""
        from shared.utils.corpus_scaling import stream_wikipedia_articles

        fake_rows = [{"text": t} for t in _fake_wikipedia_articles(10)]

        with patch("shared.utils.corpus_scaling.load_dataset") as mock_ld:
            mock_ld.return_value = iter(fake_rows)
            articles = list(stream_wikipedia_articles(max_articles=5))

        assert len(articles) == 5

    def test_filters_short_articles(self):
        """Articles shorter than min_length are skipped."""
        from shared.utils.corpus_scaling import stream_wikipedia_articles

        short = "Hi."
        long = "This is a long enough article. " * 10
        fake_rows = [{"text": short}, {"text": long}, {"text": short}, {"text": long}]

        with patch("shared.utils.corpus_scaling.load_dataset") as mock_ld:
            mock_ld.return_value = iter(fake_rows)
            articles = list(stream_wikipedia_articles(max_articles=None, min_length=50))

        for art in articles:
            assert len(art) >= 50

    def test_calls_load_dataset_with_streaming(self):
        """load_dataset is called with streaming=True."""
        from shared.utils.corpus_scaling import stream_wikipedia_articles

        fake_rows = [{"text": t} for t in _fake_wikipedia_articles(3)]

        with patch("shared.utils.corpus_scaling.load_dataset") as mock_ld:
            mock_ld.return_value = iter(fake_rows)
            list(stream_wikipedia_articles(max_articles=3))
            call_kwargs = mock_ld.call_args[1]
            assert call_kwargs.get("streaming") is True, "Must use streaming=True"


# ---------------------------------------------------------------------------
# TestBuildRankTableStreaming
# ---------------------------------------------------------------------------

class TestBuildRankTableStreaming:
    """Tests for build_rank_table_streaming()."""

    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer("gpt-5.1")

    @pytest.fixture
    def three_texts(self):
        return [
            "the cat sat on the mat",
            "the dog ran in the park the park",
            "cats and dogs are common pets",
        ]

    def test_returns_valid_rank_table(self, tokenizer, three_texts):
        """build_rank_table_streaming returns a RankTable with correct attributes."""
        from shared.utils.corpus_scaling import build_rank_table_streaming
        from shared.utils.rank_utils import RankTable

        rt = build_rank_table_streaming(
            iter(three_texts),
            tokenizer,
            tokenizer_name="gpt-5.1",
            corpus_name="test",
            progress=False,
        )

        assert isinstance(rt, RankTable)
        assert rt.vocab_size > 0
        assert rt.total_tokens > 0
        assert rt.tokenizer_name == "gpt-5.1"
        assert rt.corpus_name == "test"

    def test_ranks_are_one_indexed(self, tokenizer, three_texts):
        """All token ranks are >= 1."""
        from shared.utils.corpus_scaling import build_rank_table_streaming

        rt = build_rank_table_streaming(
            iter(three_texts),
            tokenizer,
            tokenizer_name="gpt-5.1",
            corpus_name="test",
            progress=False,
        )

        for token_id, rank in rt.token_to_rank.items():
            assert rank >= 1, f"Token {token_id} has rank {rank} (must be >= 1)"

    def test_streaming_matches_batch(self, tokenizer, three_texts):
        """Critical: streaming result MUST match build_rank_table on identical input.

        Compares vocab_size, total_tokens, and token_to_rank for every token.
        """
        from shared.utils.corpus_scaling import build_rank_table_streaming

        # Build batch result
        all_token_ids = []
        for text in three_texts:
            all_token_ids.extend(tokenize_text(text, tokenizer, "gpt-5.1"))
        batch_rt = build_rank_table(all_token_ids, "gpt-5.1", "test")

        # Build streaming result
        streaming_rt = build_rank_table_streaming(
            iter(three_texts),
            tokenizer,
            tokenizer_name="gpt-5.1",
            corpus_name="test",
            progress=False,
        )

        assert streaming_rt.vocab_size == batch_rt.vocab_size, (
            f"vocab_size mismatch: streaming={streaming_rt.vocab_size}, "
            f"batch={batch_rt.vocab_size}"
        )

        assert streaming_rt.total_tokens == batch_rt.total_tokens, (
            f"total_tokens mismatch: streaming={streaming_rt.total_tokens}, "
            f"batch={batch_rt.total_tokens}"
        )

        for token_id, expected_rank in batch_rt.token_to_rank.items():
            actual_rank = streaming_rt.token_to_rank.get(token_id)
            assert actual_rank is not None, (
                f"Token {token_id} missing from streaming result"
            )
            assert actual_rank == expected_rank, (
                f"Token {token_id}: streaming rank={actual_rank}, "
                f"batch rank={expected_rank}"
            )

        # Check no extra tokens in streaming result
        assert set(streaming_rt.token_to_rank.keys()) == set(batch_rt.token_to_rank.keys()), (
            "Token sets differ between streaming and batch"
        )

    def test_most_frequent_token_has_rank_1(self, tokenizer):
        """The most frequent token should be assigned rank 1."""
        from shared.utils.corpus_scaling import build_rank_table_streaming

        # "the" is the most common token in English — repeat it heavily
        texts = ["the the the the the other words here"] * 10

        rt = build_rank_table_streaming(
            iter(texts),
            tokenizer,
            tokenizer_name="gpt-5.1",
            corpus_name="test",
            progress=False,
        )

        # Find the token that occurs most (rank 1)
        rank_1_token = rt.rank_to_token[1]
        assert rt.token_to_freq[rank_1_token] == max(rt.token_to_freq.values())

    def test_empty_input_returns_empty_table(self, tokenizer):
        """Empty iterator produces a RankTable with zero tokens."""
        from shared.utils.corpus_scaling import build_rank_table_streaming
        from shared.utils.rank_utils import RankTable

        rt = build_rank_table_streaming(
            iter([]),
            tokenizer,
            tokenizer_name="gpt-5.1",
            corpus_name="test",
            progress=False,
        )

        assert isinstance(rt, RankTable)
        assert rt.total_tokens == 0
        assert rt.vocab_size == 0


# ---------------------------------------------------------------------------
# TestProcessFullWikipedia (smoke tests only — no actual download)
# ---------------------------------------------------------------------------

class TestProcessFullWikipedia:
    """Smoke tests for process_full_wikipedia (mocked dataset)."""

    def test_saves_to_disk_and_returns_rank_table(self, tmp_path):
        """process_full_wikipedia saves a file and returns a RankTable."""
        from shared.utils.corpus_scaling import process_full_wikipedia
        from shared.utils.rank_utils import RankTable

        tokenizer = get_tokenizer("gpt-5.1")
        output_path = tmp_path / "test_wiki.json"

        fake_rows = [{"text": t} for t in _fake_wikipedia_articles(10)]

        with patch("shared.utils.corpus_scaling.load_dataset") as mock_ld:
            mock_ld.return_value = iter(fake_rows)
            rt = process_full_wikipedia(
                tokenizer=tokenizer,
                tokenizer_name="gpt-5.1",
                output_path=output_path,
                max_articles=5,
                checkpoint_every=2,
            )

        assert isinstance(rt, RankTable)
        assert output_path.exists(), "output file was not created"

    def test_saved_file_is_loadable(self, tmp_path):
        """File saved by process_full_wikipedia can be loaded back with RankTable.load."""
        from shared.utils.corpus_scaling import process_full_wikipedia
        from shared.utils.rank_utils import RankTable

        tokenizer = get_tokenizer("gpt-5.1")
        output_path = tmp_path / "wiki_load_test.json"

        fake_rows = [{"text": t} for t in _fake_wikipedia_articles(6)]

        with patch("shared.utils.corpus_scaling.load_dataset") as mock_ld:
            mock_ld.return_value = iter(fake_rows)
            process_full_wikipedia(
                tokenizer=tokenizer,
                tokenizer_name="gpt-5.1",
                output_path=output_path,
                max_articles=6,
                checkpoint_every=3,
            )

        loaded = RankTable.load(output_path)
        assert loaded.vocab_size > 0
        assert loaded.total_tokens > 0
