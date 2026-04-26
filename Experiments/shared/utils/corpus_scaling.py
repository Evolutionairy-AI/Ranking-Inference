"""Streaming Wikipedia processing for building rank tables at ~4B token scale."""

import json
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional
from datasets import load_dataset
from tqdm import tqdm
from .rank_utils import RankTable
from .corpus_utils import tokenize_text
from .mandelbrot import fit_mandelbrot_mle


def stream_wikipedia_articles(
    subset: str = "20231101.en",
    max_articles: Optional[int] = None,
    min_length: int = 100,
) -> Iterator[str]:
    """Stream English Wikipedia articles from HuggingFace.

    Args:
        subset: Wikipedia dump version (e.g. "20231101.en").
        max_articles: Stop after this many qualifying articles. None = stream all.
        min_length: Skip articles with fewer than this many characters.

    Yields:
        Article text strings (len >= min_length).
    """
    ds = load_dataset(
        "wikimedia/wikipedia",
        subset,
        split="train",
        streaming=True,
    )

    count = 0
    for example in ds:
        text = example.get("text", "")
        if text and len(text) >= min_length:
            yield text
            count += 1
            if max_articles is not None and count >= max_articles:
                break


def build_rank_table_streaming(
    texts_iter: Iterator[str],
    tokenizer,
    tokenizer_name: str,
    corpus_name: str,
    progress: bool = True,
    total: Optional[int] = None,
) -> RankTable:
    """Build a RankTable from a streaming text iterator.

    Keeps only a Counter in memory (O(vocab_size), not O(corpus_size)).
    Produces identical results to build_rank_table() for the same input.

    Args:
        texts_iter: Iterator of raw text strings.
        tokenizer: Tokenizer instance (tiktoken or HuggingFace).
        tokenizer_name: Name string for the tokenizer.
        corpus_name: Name string for the corpus.
        progress: Show tqdm progress bar.
        total: Total expected items for tqdm (optional hint).

    Returns:
        RankTable with frequency-ordered ranks (rank 1 = most frequent).
    """
    freq_counter: Counter = Counter()
    total_tokens = 0

    iterable = texts_iter
    if progress:
        iterable = tqdm(iterable, desc=f"Streaming {corpus_name}", total=total)

    for text in iterable:
        token_ids = tokenize_text(text, tokenizer, tokenizer_name)
        freq_counter.update(token_ids)
        total_tokens += len(token_ids)

    if not freq_counter:
        # Empty corpus — return a zero-sized RankTable
        return RankTable(
            token_to_rank={},
            rank_to_token={},
            token_to_freq={},
            rank_to_freq=np.zeros(1, dtype=np.int64),
            tokenizer_name=tokenizer_name,
            corpus_name=corpus_name,
            total_tokens=0,
            vocab_size=0,
        )

    # Sort by frequency descending, then by token_id ascending for stability.
    # This mirrors exactly what build_rank_table() does.
    sorted_tokens = sorted(freq_counter.items(), key=lambda x: (-x[1], x[0]))

    token_to_rank: dict[int, int] = {}
    rank_to_token: dict[int, int] = {}
    token_to_freq: dict[int, int] = dict(freq_counter)

    for rank_0indexed, (token_id, _freq) in enumerate(sorted_tokens):
        rank = rank_0indexed + 1  # 1-indexed
        token_to_rank[token_id] = rank
        rank_to_token[rank] = token_id

    max_rank = len(sorted_tokens)
    rank_to_freq = np.zeros(max_rank + 1, dtype=np.int64)  # index 0 unused
    for rank, token_id in rank_to_token.items():
        rank_to_freq[rank] = freq_counter[token_id]

    return RankTable(
        token_to_rank=token_to_rank,
        rank_to_token=rank_to_token,
        token_to_freq=token_to_freq,
        rank_to_freq=rank_to_freq,
        tokenizer_name=tokenizer_name,
        corpus_name=corpus_name,
        total_tokens=total_tokens,
        vocab_size=max_rank,
    )


def process_full_wikipedia(
    tokenizer,
    tokenizer_name: str,
    output_path: Path,
    subset: str = "20231101.en",
    max_articles: Optional[int] = None,
    checkpoint_every: int = 100_000,
) -> RankTable:
    """Stream Wikipedia, build a rank table, fit Mandelbrot, and save to disk.

    Args:
        tokenizer: Tokenizer instance.
        tokenizer_name: Name string for the tokenizer.
        output_path: Path to save the resulting rank table JSON.
        subset: Wikipedia dump version.
        max_articles: Cap on number of articles (None = all ~6.7M).
        checkpoint_every: Print a progress message every N articles.

    Returns:
        The built RankTable.
    """
    output_path = Path(output_path)
    print(f"[corpus_scaling] Starting Wikipedia streaming: subset={subset}, "
          f"tokenizer={tokenizer_name}, max_articles={max_articles}")

    # We stream articles ourselves to support checkpoint logging
    freq_counter: Counter = Counter()
    total_tokens = 0
    article_count = 0

    ds = load_dataset(
        "wikimedia/wikipedia",
        subset,
        split="train",
        streaming=True,
    )

    for example in ds:
        text = example.get("text", "")
        if not text or len(text) < 100:
            continue

        token_ids = tokenize_text(text, tokenizer, tokenizer_name)
        freq_counter.update(token_ids)
        total_tokens += len(token_ids)
        article_count += 1

        if article_count % checkpoint_every == 0:
            print(
                f"[corpus_scaling] Processed {article_count:,} articles, "
                f"{total_tokens:,} tokens, vocab_size={len(freq_counter):,}"
            )

        if max_articles is not None and article_count >= max_articles:
            break

    print(f"[corpus_scaling] Done streaming: {article_count:,} articles, "
          f"{total_tokens:,} tokens, vocab_size={len(freq_counter):,}")

    # Build RankTable from accumulated counts
    if not freq_counter:
        rank_table = RankTable(
            token_to_rank={},
            rank_to_token={},
            token_to_freq={},
            rank_to_freq=np.zeros(1, dtype=np.int64),
            tokenizer_name=tokenizer_name,
            corpus_name=f"wikipedia_{subset}",
            total_tokens=0,
            vocab_size=0,
        )
    else:
        sorted_tokens = sorted(freq_counter.items(), key=lambda x: (-x[1], x[0]))
        token_to_rank: dict[int, int] = {}
        rank_to_token: dict[int, int] = {}
        token_to_freq: dict[int, int] = dict(freq_counter)

        for rank_0indexed, (token_id, _freq) in enumerate(sorted_tokens):
            rank = rank_0indexed + 1
            token_to_rank[token_id] = rank
            rank_to_token[rank] = token_id

        max_rank = len(sorted_tokens)
        rank_to_freq = np.zeros(max_rank + 1, dtype=np.int64)
        for rank, token_id in rank_to_token.items():
            rank_to_freq[rank] = freq_counter[token_id]

        rank_table = RankTable(
            token_to_rank=token_to_rank,
            rank_to_token=rank_to_token,
            token_to_freq=token_to_freq,
            rank_to_freq=rank_to_freq,
            tokenizer_name=tokenizer_name,
            corpus_name=f"wikipedia_{subset}",
            total_tokens=total_tokens,
            vocab_size=max_rank,
        )

    # Fit Mandelbrot and print parameters
    if rank_table.vocab_size > 0:
        try:
            ranks_arr = np.arange(1, rank_table.vocab_size + 1, dtype=np.float64)
            freqs_arr = rank_table.rank_to_freq[1:].astype(np.float64)
            params = fit_mandelbrot_mle(ranks_arr, freqs_arr)
            print(
                f"[corpus_scaling] Mandelbrot fit: "
                f"C={params.C:.4g}, q={params.q:.4f}, s={params.s:.4f}, "
                f"log_likelihood={params.log_likelihood:.4g}"
            )
        except Exception as exc:
            print(f"[corpus_scaling] Mandelbrot fit failed: {exc}")

    # Save rank table
    rank_table.save(output_path)
    print(f"[corpus_scaling] Rank table saved to {output_path}")

    return rank_table
