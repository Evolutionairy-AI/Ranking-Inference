"""Smoke tests for ranking-inference core primitives.

These are fast, dependency-light checks that the package's public API is
importable and the math/data primitives behave as documented. They do not
exercise spaCy NER or transformers tokenizers.
"""

import numpy as np

from ranking_inference import (
    MandelbrotParams,
    RankTable,
    build_rank_table,
    mandelbrot_freq,
    mandelbrot_pmf,
)


def test_mandelbrot_params_dataclass():
    p = MandelbrotParams(
        C=1.0, q=2.7, s=1.05,
        log_likelihood=-100.0, n_tokens=1000, vocab_size=128000,
    )
    assert p.q == 2.7
    assert p.s == 1.05
    assert p.vocab_size == 128000


def test_mandelbrot_freq_decreases_with_rank():
    ranks = np.array([1, 10, 100, 1000])
    f = mandelbrot_freq(ranks, C=1.0, q=2.7, s=1.05)
    assert np.all(np.diff(f) < 0), "frequency must decrease monotonically with rank"


def test_mandelbrot_pmf_normalized():
    ranks = np.arange(1, 1001)
    pmf = mandelbrot_pmf(ranks, q=2.7, s=1.05)
    assert pmf.shape == ranks.shape
    assert np.isclose(pmf.sum(), 1.0), "pmf must sum to 1"
    assert np.all(pmf > 0), "pmf must be strictly positive"


def test_build_rank_table_assigns_ranks_by_frequency():
    # Token 5 appears most, then 1, then 7
    tokens = [5, 5, 5, 5, 1, 1, 1, 7, 7]
    rt = build_rank_table(tokens, tokenizer_name="dummy", corpus_name="test")
    assert rt.get_rank(5) == 1
    assert rt.get_rank(1) == 2
    assert rt.get_rank(7) == 3
    assert rt.total_tokens == len(tokens)


def test_rank_table_save_load_roundtrip(tmp_path):
    tokens = [5, 5, 5, 1, 1, 7]
    rt = build_rank_table(tokens, tokenizer_name="dummy", corpus_name="test")
    path = tmp_path / "table.json"
    rt.save(path)

    loaded = RankTable.load(path)
    assert loaded.get_rank(5) == rt.get_rank(5)
    assert loaded.get_rank(1) == rt.get_rank(1)
    assert loaded.tokenizer_name == "dummy"
    assert loaded.corpus_name == "test"
    assert loaded.total_tokens == rt.total_tokens
