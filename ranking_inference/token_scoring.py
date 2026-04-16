"""Token-level scoring with three aggregation modes (Experiment B).

Modes
-----
(a) all-tokens:    aggregate log(P_LLM/G_RI) across every token in the output
(b) entity-level:  aggregate across tokens covered by NER-tagged entities
(c) rank-only:     aggregate signal at entity positions without using logprobs
                   (tests whether rank deviation alone detects fabricated entities)

The module is designed so the expensive parts (logprob unpacking, NER,
grounding-score lookup) run once per output, and the cheap aggregation can
run multiple times with different token index sets (e.g. per error / control
span in FRANK).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .entity_extraction import (
    extract_entities,
    align_entities_to_tokens,
    get_grounding_scores,
)

_FALLBACK_P_LLM = 1.0 / 50000
_LOG_FLOOR = 1e-20


@dataclass
class TokenScores:
    """Per-token scoring quantities for one output."""

    token_ids: list[int]
    logprobs: list[Optional[float]]
    log_p_llm: list[float]       # fallback-filled log P_LLM per token
    log_g_ri: list[float]        # log G_RI(token) (rank-derived Mandelbrot prob)
    global_ranks: list[int]      # rank of token in the global rank table
    log_deltas: list[float]      # log(P_LLM) - log(G_RI)
    linear_deltas: list[float]   # P_LLM - G_RI
    entity_positions: list[int]  # token indices covered by any NER entity
    entity_spans: list[dict]     # [{'text','type','indices'}]
    n_entities: int


def compute_token_scores(
    text: str,
    token_ids: list[int],
    logprobs: list[Optional[float]],
    tokenizer,
    rank_table,
) -> TokenScores:
    """Compute per-token quantities required for three-mode aggregation.

    Runs NER once, aligns entities to BPE token indices, and records for every
    token: the LLM logprob, rank-derived log G_RI, global rank, and both linear
    and log-space deltas.
    """
    scores, default_g = get_grounding_scores(rank_table)

    n = len(token_ids)
    log_p_llm: list[float] = [0.0] * n
    log_g_ri: list[float] = [0.0] * n
    global_ranks: list[int] = [0] * n
    log_deltas: list[float] = [0.0] * n
    linear_deltas: list[float] = [0.0] * n

    for i, tid in enumerate(token_ids):
        lp = logprobs[i] if i < len(logprobs) else None
        log_p = lp if lp is not None else math.log(_FALLBACK_P_LLM)
        p_llm = math.exp(log_p)
        g_ri = scores.get(tid, default_g)
        log_g = math.log(max(g_ri, _LOG_FLOOR))

        log_p_llm[i] = log_p
        log_g_ri[i] = log_g
        global_ranks[i] = rank_table.get_rank(tid)
        log_deltas[i] = log_p - log_g
        linear_deltas[i] = p_llm - g_ri

    entities = extract_entities(text)
    entity_positions: set[int] = set()
    entity_spans: list[dict] = []
    if entities and n > 0:
        aligned = align_entities_to_tokens(entities, text, token_ids, tokenizer)
        for span, indices in aligned:
            if indices:
                entity_positions.update(indices)
                entity_spans.append({
                    "text": span.text,
                    "type": span.entity_type,
                    "indices": indices,
                })

    return TokenScores(
        token_ids=token_ids,
        logprobs=logprobs,
        log_p_llm=log_p_llm,
        log_g_ri=log_g_ri,
        global_ranks=global_ranks,
        log_deltas=log_deltas,
        linear_deltas=linear_deltas,
        entity_positions=sorted(entity_positions),
        entity_spans=entity_spans,
        n_entities=len(entity_spans),
    )


_EMPTY_AGGREGATE = {
    "all_n_tokens": 0,
    "all_mean_log_delta": 0.0,
    "all_mean_linear_delta": 0.0,
    "all_mean_neg_log_g": 0.0,
    "entity_n_tokens": 0,
    "entity_mean_log_delta": 0.0,
    "entity_mean_linear_delta": 0.0,
    "rank_only_mean_log_rank": 0.0,
    "rank_only_mean_neg_log_g": 0.0,
}


def aggregate_three_modes(
    token_scores: TokenScores,
    token_indices: Optional[list[int]] = None,
) -> dict:
    """Aggregate the three scoring modes over a subset of token indices.

    Parameters
    ----------
    token_scores : TokenScores
        Per-token data produced by :func:`compute_token_scores`.
    token_indices : list[int] | None
        Restrict aggregation to these token indices. ``None`` uses all tokens
        in the output. Use the token indices of an error / control span for
        FRANK-style span aggregation.

    Returns
    -------
    dict with keys:
        all_n_tokens              number of tokens in scope
        all_mean_log_delta        mean log(P_LLM/G_RI) across ALL tokens in scope
        all_mean_linear_delta     mean (P_LLM - G_RI) across ALL tokens in scope
        all_mean_neg_log_g        mean -log(G_RI) across ALL tokens in scope

        entity_n_tokens           count of entity tokens within scope
        entity_mean_log_delta     mean log(P_LLM/G_RI) at entity tokens in scope
        entity_mean_linear_delta  mean (P_LLM - G_RI) at entity tokens in scope

        rank_only_mean_log_rank   mean log2(global_rank) at entity tokens in scope
        rank_only_mean_neg_log_g  mean -log(G_RI) at entity tokens in scope
    """
    if not token_scores.token_ids:
        return dict(_EMPTY_AGGREGATE)

    if token_indices is None:
        all_idx = list(range(len(token_scores.token_ids)))
    else:
        all_idx = [i for i in token_indices
                   if 0 <= i < len(token_scores.token_ids)]

    all_n = len(all_idx)
    if all_n == 0:
        return dict(_EMPTY_AGGREGATE)

    all_mean_log_delta = (
        sum(token_scores.log_deltas[i] for i in all_idx) / all_n
    )
    all_mean_linear_delta = (
        sum(token_scores.linear_deltas[i] for i in all_idx) / all_n
    )
    all_mean_neg_log_g = (
        sum(-token_scores.log_g_ri[i] for i in all_idx) / all_n
    )

    entity_set = set(token_scores.entity_positions)
    entity_idx = [i for i in all_idx if i in entity_set]
    entity_n = len(entity_idx)

    if entity_n > 0:
        entity_mean_log_delta = (
            sum(token_scores.log_deltas[i] for i in entity_idx) / entity_n
        )
        entity_mean_linear_delta = (
            sum(token_scores.linear_deltas[i] for i in entity_idx) / entity_n
        )
        rank_only_mean_log_rank = (
            sum(math.log2(max(token_scores.global_ranks[i], 1))
                for i in entity_idx) / entity_n
        )
        rank_only_mean_neg_log_g = (
            sum(-token_scores.log_g_ri[i] for i in entity_idx) / entity_n
        )
    else:
        entity_mean_log_delta = 0.0
        entity_mean_linear_delta = 0.0
        rank_only_mean_log_rank = 0.0
        rank_only_mean_neg_log_g = 0.0

    return {
        "all_n_tokens": all_n,
        "all_mean_log_delta": all_mean_log_delta,
        "all_mean_linear_delta": all_mean_linear_delta,
        "all_mean_neg_log_g": all_mean_neg_log_g,
        "entity_n_tokens": entity_n,
        "entity_mean_log_delta": entity_mean_log_delta,
        "entity_mean_linear_delta": entity_mean_linear_delta,
        "rank_only_mean_log_rank": rank_only_mean_log_rank,
        "rank_only_mean_neg_log_g": rank_only_mean_neg_log_g,
    }
