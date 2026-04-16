"""Ranking Inference: distributional grounding primitives for LLM outputs.

Core modules
------------
mandelbrot       : fit the Mandelbrot ranking distribution f(r) = C / (r+q)^s
rank_utils       : build and serialise per-token rank tables
token_scoring    : per-token log(P_LLM / G_RI) scoring with three aggregation modes
entity_extraction: span-level entity tagging via spaCy NER
aggregation      : sentence/document aggregation helpers

Quick start
-----------
>>> from ranking_inference import RankTable, compute_token_scores, aggregate_three_modes
>>> rt = RankTable.load("rank_tables/wikipedia_full_llama-3.1-8b.json")
>>> scores = compute_token_scores(text, token_ids, logprobs, tokenizer, rt)
>>> agg = aggregate_three_modes(scores)
>>> agg["all_mean_log_delta"], agg["entity_mean_log_delta"]
"""

from .mandelbrot import (
    MandelbrotParams,
    mandelbrot_freq,
    mandelbrot_log_freq,
    mandelbrot_pmf,
    fit_mandelbrot_mle,
    fit_mandelbrot_ols_loglog,
    goodness_of_fit,
    compare_distributions,
)
from .rank_utils import RankTable, build_rank_table
from .token_scoring import (
    TokenScores,
    compute_token_scores,
    aggregate_three_modes,
)
from .entity_extraction import (
    EntitySpan,
    EntityGapResult,
    extract_entities,
    align_entities_to_tokens,
    compute_entity_gaps,
)
from .aggregation import (
    entity_weighted_mean_delta,
    max_entity_delta,
    posterior_entity_weighted_mean,
    aggregate_all,
)

__version__ = "0.1.0"
__all__ = [
    "MandelbrotParams",
    "mandelbrot_freq",
    "mandelbrot_log_freq",
    "mandelbrot_pmf",
    "fit_mandelbrot_mle",
    "fit_mandelbrot_ols_loglog",
    "goodness_of_fit",
    "compare_distributions",
    "RankTable",
    "build_rank_table",
    "TokenScores",
    "compute_token_scores",
    "aggregate_three_modes",
    "EntitySpan",
    "EntityGapResult",
    "extract_entities",
    "align_entities_to_tokens",
    "compute_entity_gaps",
    "entity_weighted_mean_delta",
    "max_entity_delta",
    "posterior_entity_weighted_mean",
    "aggregate_all",
]
