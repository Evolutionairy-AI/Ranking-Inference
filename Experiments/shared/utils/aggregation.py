"""Sequence-level aggregation strategies for entity gap results.

These functions summarise a list of EntityGapResult objects — one per named
entity span — into scalar scores that describe the overall hallucination risk
for an entire generated sequence.
"""

from .entity_extraction import EntityGapResult


def entity_weighted_mean_delta(entity_results: list[EntityGapResult]) -> float:
    """Mean of each entity's mean_delta.

    Each entity contributes equally regardless of how many tokens it spans.

    Returns 0.0 if the list is empty.
    """
    if not entity_results:
        return 0.0
    return sum(e.mean_delta for e in entity_results) / len(entity_results)


def max_entity_delta(entity_results: list[EntityGapResult]) -> float:
    """Maximum per-entity mean_delta across all entities in the sequence.

    Useful as a worst-case / most-anomalous-entity signal.

    Returns 0.0 if the list is empty.
    """
    if not entity_results:
        return 0.0
    return max(e.mean_delta for e in entity_results)


def proportion_above_threshold(
    entity_results: list[EntityGapResult], threshold: float
) -> float:
    """Fraction of entities whose mean_delta strictly exceeds *threshold*.

    A value of 1.0 means every entity is anomalous; 0.0 means none are.

    Returns 0.0 if the list is empty.
    """
    if not entity_results:
        return 0.0
    count_above = sum(1 for e in entity_results if e.mean_delta > threshold)
    return count_above / len(entity_results)


def log_entity_weighted_mean(entity_results: list[EntityGapResult]) -> float:
    """Mean of each entity's mean_log_delta (log-space ratio).

    log_delta = log(P_LLM) - log(G_RI) per token, averaged over entity tokens,
    then averaged over entities. Higher = LLM more confident relative to corpus.
    """
    if not entity_results:
        return 0.0
    return sum(e.mean_log_delta for e in entity_results) / len(entity_results)


def log_max_entity_delta(entity_results: list[EntityGapResult]) -> float:
    """Maximum per-entity mean_log_delta across all entities."""
    if not entity_results:
        return 0.0
    return max(e.mean_log_delta for e in entity_results)


def posterior_entity_weighted_mean(entity_results: list[EntityGapResult]) -> float:
    """Mean of each entity's mean_posterior (Bayesian anomaly score).

    posterior = -(logprob + beta * log(G_RI)), higher = more anomalous.
    """
    if not entity_results:
        return 0.0
    return sum(e.mean_posterior for e in entity_results) / len(entity_results)


def aggregate_all(
    entity_results: list[EntityGapResult], threshold: float = 0.1
) -> dict[str, float]:
    """Compute all aggregation scores and return them as a dict.

    Keys (linear space — legacy)
    ----
    entity_weighted_mean      : mean of per-entity mean_delta values
    max_entity_delta          : highest per-entity mean_delta
    proportion_above_threshold: fraction of entities with mean_delta > threshold

    Keys (log space — Bayesian posterior)
    ----
    log_entity_weighted_mean      : mean of per-entity log(P_LLM/G_RI)
    log_max_entity_delta          : max of per-entity log(P_LLM/G_RI)
    posterior_entity_weighted_mean: mean of per-entity -(logprob + beta*log(G_RI))

    Returns a dict with all keys set to 0.0 when *entity_results* is empty.
    """
    return {
        "entity_weighted_mean": entity_weighted_mean_delta(entity_results),
        "max_entity_delta": max_entity_delta(entity_results),
        "proportion_above_threshold": proportion_above_threshold(
            entity_results, threshold
        ),
        "log_entity_weighted_mean": log_entity_weighted_mean(entity_results),
        "log_max_entity_delta": log_max_entity_delta(entity_results),
        "posterior_entity_weighted_mean": posterior_entity_weighted_mean(entity_results),
    }
