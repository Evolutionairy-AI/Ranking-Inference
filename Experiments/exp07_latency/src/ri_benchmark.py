"""RI verification pass timing benchmark.

Times compute_entity_gaps() in two modes:
- "full": includes SpaCy NER extraction
- "gap_only": uses pre-extracted EntitySpan objects from cached data

Also times setup costs (rank table load, grounding scores).
"""

import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp07_latency.src.timer import time_operation, TimingResult, system_info
from shared.utils import (
    RankTable,
    get_model_config,
    get_tokenizer,
    compute_entity_gaps,
    aggregate_all,
)
from shared.utils.entity_extraction import EntitySpan, get_grounding_scores

RANK_TABLE_PATH = PROJECT_ROOT / "shared" / "rank_tables" / "wikipedia_full_llama-3.1-8b.json"
MODEL_NAME = "llama-3.1-8b"


def _load_cached_data(cached_data_path: Path) -> list[dict]:
    """Load cached JSONL data produced by prepare_data.py."""
    examples = []
    with open(cached_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _group_by_bin(examples: list[dict]) -> dict[str, list[dict]]:
    """Group cached examples by their length_bin field."""
    bins: dict[str, list[dict]] = {}
    for ex in examples:
        b = ex["length_bin"]
        bins.setdefault(b, []).append(ex)
    return bins


def _reconstruct_entities(entity_dicts: list[dict]) -> list[EntitySpan]:
    """Reconstruct EntitySpan objects from cached entity dicts."""
    return [
        EntitySpan(
            text=d["text"],
            entity_type=d["entity_type"],
            char_start=d["char_start"],
            char_end=d["char_end"],
        )
        for d in entity_dicts
    ]


def run_ri_benchmark(
    cached_data_path: Path,
    n_runs: int = 100,
    n_warmup: int = 10,
) -> dict:
    """Run the RI verification timing benchmark.

    Parameters
    ----------
    cached_data_path : Path
        Path to cached_logprobs.jsonl from prepare_data.py.
    n_runs : int
        Number of timed repetitions per example.
    n_warmup : int
        Warmup runs before timing.

    Returns
    -------
    dict
        Results with setup timings, per-bin timings, and metadata.
    """
    # --- Setup timing: rank table load ---
    rank_table_timing = time_operation(
        "rank_table_load",
        RankTable.load,
        args=(RANK_TABLE_PATH,),
        n_runs=5,
        n_warmup=1,
    )
    rank_table = RankTable.load(RANK_TABLE_PATH)

    # --- Setup timing: grounding scores ---
    # get_grounding_scores caches by id(rank_table), so we force a fresh
    # computation by clearing the cache and timing the first call.
    from shared.utils.entity_extraction import _GROUNDING_CACHE
    _GROUNDING_CACHE.clear()

    grounding_timing = time_operation(
        "grounding_scores",
        get_grounding_scores,
        args=(rank_table,),
        n_runs=1,
        n_warmup=0,
    )
    # Ensure scores are cached for subsequent calls
    get_grounding_scores(rank_table)

    # --- Load data and tokenizer ---
    examples = _load_cached_data(cached_data_path)
    config = get_model_config(MODEL_NAME)
    tokenizer = get_tokenizer(config.tokenizer_name)

    binned = _group_by_bin(examples)

    # --- Per-bin benchmarks ---
    per_bin: dict[str, dict] = {}

    for bin_name, bin_examples in sorted(binned.items()):
        full_results = []
        gap_only_results = []
        aggregate_results = []

        for ex in bin_examples:
            text = ex["text"]
            token_ids = ex["token_ids"]
            logprobs = ex["logprobs"]
            n_tokens = ex["n_tokens"]
            entity_dicts = ex.get("entities", [])
            pre_extracted = _reconstruct_entities(entity_dicts)
            n_entities = len(pre_extracted)

            # --- Full mode (includes SpaCy NER) ---
            full_timing = time_operation(
                "compute_entity_gaps_full",
                compute_entity_gaps,
                kwargs={
                    "text": text,
                    "token_ids": token_ids,
                    "logprobs": logprobs,
                    "tokenizer": tokenizer,
                    "rank_table": rank_table,
                },
                n_runs=n_runs,
                n_warmup=n_warmup,
            )
            full_dict = full_timing.to_dict()
            full_dict["n_tokens"] = n_tokens
            full_dict["n_entities"] = n_entities
            full_results.append(full_dict)

            # --- Gap-only mode (pre-extracted entities) ---
            gap_timing = time_operation(
                "compute_entity_gaps_gap_only",
                compute_entity_gaps,
                kwargs={
                    "text": text,
                    "token_ids": token_ids,
                    "logprobs": logprobs,
                    "tokenizer": tokenizer,
                    "rank_table": rank_table,
                    "entities": pre_extracted,
                },
                n_runs=n_runs,
                n_warmup=n_warmup,
            )
            gap_dict = gap_timing.to_dict()
            gap_dict["n_tokens"] = n_tokens
            gap_dict["n_entities"] = n_entities
            gap_only_results.append(gap_dict)

            # --- Aggregate timing ---
            # Time aggregate_all on the entity gap results
            entity_gaps = compute_entity_gaps(
                text=text,
                token_ids=token_ids,
                logprobs=logprobs,
                tokenizer=tokenizer,
                rank_table=rank_table,
                entities=pre_extracted,
            )

            if entity_gaps:
                agg_timing = time_operation(
                    "aggregate_all",
                    aggregate_all,
                    args=(entity_gaps,),
                    n_runs=n_runs,
                    n_warmup=n_warmup,
                )
                agg_dict = agg_timing.to_dict()
                agg_dict["n_tokens"] = n_tokens
                agg_dict["n_entities"] = n_entities
                aggregate_results.append(agg_dict)

        per_bin[bin_name] = {
            "full": full_results,
            "gap_only": gap_only_results,
            "aggregate": aggregate_results,
        }

    return {
        "setup": {
            "rank_table_load": rank_table_timing.to_dict(),
            "grounding_scores": grounding_timing.to_dict(),
        },
        "per_bin": per_bin,
        "model": MODEL_NAME,
        "n_examples": len(examples),
        "n_runs_per_example": n_runs,
    }
