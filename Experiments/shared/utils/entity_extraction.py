"""Entity extraction utilities for Ranking Inference experiments.

This module provides NER-based entity extraction, token alignment,
and per-entity confidence-grounding gap computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EntitySpan:
    """A named-entity span identified by NER."""

    text: str
    entity_type: str
    char_start: int
    char_end: int


@dataclass
class EntityGapResult:
    """Aggregated gap statistics for a single named entity span."""

    text: str
    entity_type: str
    token_indices: list[int]
    mean_delta: float
    max_delta: float
    mean_global_rank: float
    mean_p_llm: float
    mean_g_ri: float
    # Log-space metrics (Bayesian posterior formulation)
    mean_log_delta: float = 0.0       # mean of log(P_LLM) - log(G_RI)
    mean_posterior: float = 0.0       # mean of -(logprob + beta * log(G_RI))


# ---------------------------------------------------------------------------
# SpaCy lazy loading
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    """Lazy-load a SpaCy model. Prefer en_core_web_trf, fall back to sm."""
    global _NLP
    if _NLP is not None:
        return _NLP
    import spacy

    for model_name in ("en_core_web_trf", "en_core_web_sm"):
        try:
            _NLP = spacy.load(model_name)
            return _NLP
        except OSError:
            continue
    raise RuntimeError(
        "No SpaCy English NER model found. Install en_core_web_sm or en_core_web_trf."
    )


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> list[EntitySpan]:
    """Extract named entities from *text* using SpaCy NER.

    Returns an empty list for empty or whitespace-only input.
    """
    if not text or not text.strip():
        return []
    nlp = _get_nlp()
    doc = nlp(text)
    return [
        EntitySpan(
            text=ent.text,
            entity_type=ent.label_,
            char_start=ent.start_char,
            char_end=ent.end_char,
        )
        for ent in doc.ents
    ]


# ---------------------------------------------------------------------------
# Token ↔ character alignment
# ---------------------------------------------------------------------------


def align_entities_to_tokens(
    entities: list[EntitySpan],
    text: str,
    token_ids: list[int],
    tokenizer,
) -> list[tuple[EntitySpan, list[int]]]:
    """Map each entity to the BPE token indices that overlap its char span.

    Builds a char→token map by decoding each token sequentially and tracking
    the character position in the original string.
    """
    # Build (token_index) -> (char_start, char_end) mapping
    token_char_spans: list[tuple[int, int]] = []
    char_pos = 0
    for idx, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        tok_len = len(decoded)
        # Some tokenizers may produce tokens whose decoded form doesn't
        # exactly match the original text (e.g. leading spaces encoded
        # differently).  We scan forward to find where this decoded chunk
        # aligns in the original text.
        start = text.find(decoded, char_pos)
        if start == -1:
            # Fallback: assign based on running position
            start = char_pos
        end = start + tok_len
        token_char_spans.append((start, end))
        char_pos = end

    results: list[tuple[EntitySpan, list[int]]] = []
    for ent in entities:
        indices: list[int] = []
        for tok_idx, (ts, te) in enumerate(token_char_spans):
            # Overlap check: entity [ent.char_start, ent.char_end) vs
            # token [ts, te)
            if ts < ent.char_end and te > ent.char_start:
                indices.append(tok_idx)
        results.append((ent, indices))
    return results


# ---------------------------------------------------------------------------
# Grounding scores (Mandelbrot-based G_RI)
# ---------------------------------------------------------------------------

_GROUNDING_CACHE: dict[int, tuple[dict[int, float], float]] = {}


def get_grounding_scores(
    rank_table,
) -> tuple[dict[int, float], float]:
    """Fit Mandelbrot to *rank_table* frequencies and return normalised G_RI.

    Returns ``(scores_dict, default_for_unseen)`` where *scores_dict* maps
    ``token_id -> G_RI`` and *default_for_unseen* is the score assigned to
    tokens absent from the rank table.

    Results are cached by ``id(rank_table)``.
    """
    cache_key = id(rank_table)
    if cache_key in _GROUNDING_CACHE:
        return _GROUNDING_CACHE[cache_key]

    from shared.utils import fit_mandelbrot_mle
    from shared.utils.mandelbrot import mandelbrot_freq

    # Build non-zero rank/freq arrays (rank_to_freq is 1-indexed, index 0 unused)
    max_rank = rank_table.vocab_size
    ranks = np.arange(1, max_rank + 1)
    freqs = rank_table.rank_to_freq[1: max_rank + 1].astype(np.float64)

    # Filter to non-zero
    nz_mask = freqs > 0
    ranks_nz = ranks[nz_mask]
    freqs_nz = freqs[nz_mask]

    if len(ranks_nz) == 0:
        # Degenerate: no data
        empty: dict[int, float] = {}
        _GROUNDING_CACHE[cache_key] = (empty, 0.0)
        return empty, 0.0

    params = fit_mandelbrot_mle(ranks_nz, freqs_nz)

    # Compute fitted frequencies for all ranks and normalise to probabilities
    fitted = mandelbrot_freq(ranks.astype(np.float64), params.C, params.q, params.s)
    total_fitted = fitted.sum()
    g_ri_array = fitted / total_fitted  # probability mass per rank

    scores: dict[int, float] = {}
    for token_id, rank in rank_table.token_to_rank.items():
        scores[token_id] = float(g_ri_array[rank - 1])  # rank 1 → index 0

    # Default for unseen: use the lowest-rank probability
    default = float(g_ri_array[-1]) if len(g_ri_array) > 0 else 0.0

    _GROUNDING_CACHE[cache_key] = (scores, default)
    return scores, default


# ---------------------------------------------------------------------------
# Main pipeline: compute_entity_gaps
# ---------------------------------------------------------------------------

_FALLBACK_P_LLM = 1.0 / 50000


def compute_entity_gaps(
    text: str,
    token_ids: list[int],
    logprobs: list[Optional[float]],
    tokenizer,
    rank_table,
    beta: float = 1.0,
    entities: Optional[list[EntitySpan]] = None,
) -> list[EntityGapResult]:
    """Compute per-entity confidence-grounding gaps.

    For each entity found by NER:
      delta(t) = exp(logprob_t) - G_RI(token_id_t)
    aggregated into an :class:`EntityGapResult`.

    Parameters
    ----------
    text : str
        The original text.
    token_ids : list[int]
        BPE token IDs produced by *tokenizer* for *text*.
    logprobs : list[float | None]
        Log-probabilities from the LLM, one per token.  ``None`` entries
        are replaced with ``log(1/50000)``.
    tokenizer
        A tokenizer with a ``.decode([token_id]) -> str`` method
        (e.g. a tiktoken ``Encoding``).
    rank_table
        A :class:`RankTable` instance.
    beta : float
        Scaling parameter (reserved for future use; currently unused).

    Returns
    -------
    list[EntityGapResult]
        One result per detected entity (entities with no aligned tokens
        are omitted).
    """
    if entities is None:
        entities = extract_entities(text)
    if not entities:
        return []

    aligned = align_entities_to_tokens(entities, text, token_ids, tokenizer)
    scores, default_g = get_grounding_scores(rank_table)

    _LOG_FLOOR = 1e-20  # avoid log(0)

    results: list[EntityGapResult] = []
    for entity_span, indices in aligned:
        if not indices:
            continue

        deltas: list[float] = []
        log_deltas: list[float] = []
        posteriors: list[float] = []
        p_llms: list[float] = []
        g_ris: list[float] = []
        global_ranks: list[float] = []

        for idx in indices:
            tid = token_ids[idx]
            lp = logprobs[idx]
            log_p = lp if lp is not None else math.log(_FALLBACK_P_LLM)
            p_llm = math.exp(log_p)
            g_ri = scores.get(tid, default_g)
            log_g = math.log(max(g_ri, _LOG_FLOOR))

            deltas.append(p_llm - g_ri)
            log_deltas.append(log_p - log_g)
            posteriors.append(-(log_p + beta * log_g))
            p_llms.append(p_llm)
            g_ris.append(g_ri)
            global_ranks.append(float(rank_table.get_rank(tid)))

        results.append(
            EntityGapResult(
                text=entity_span.text,
                entity_type=entity_span.entity_type,
                token_indices=indices,
                mean_delta=float(np.mean(deltas)),
                max_delta=float(np.max(deltas)),
                mean_global_rank=float(np.mean(global_ranks)),
                mean_p_llm=float(np.mean(p_llms)),
                mean_g_ri=float(np.mean(g_ris)),
                mean_log_delta=float(np.mean(log_deltas)),
                mean_posterior=float(np.mean(posteriors)),
            )
        )
    return results
