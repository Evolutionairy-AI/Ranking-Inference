"""Score HaluEval examples through the RI entity-gap pipeline (Experiment B).

Each example has one response text with a binary hallucination label.
We score the response through the LLM logprob pipeline, compute per-token
scores, and aggregate into three modes:

  (a) all-tokens          average log(P_LLM/G_RI) across every token
  (b) entity-level        average log(P_LLM/G_RI) at NER-tagged token positions
  (c) rank-only entities  average log2(global_rank) (or -log G_RI) at entity
                          positions, WITHOUT using logprobs

Supports JSONL checkpointing so long runs can be resumed.
"""

import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import (
    RankTable,
    get_tokenizer,
    compute_entity_gaps,
    aggregate_all,
    score_text_logprobs,
    get_model_config,
    compute_token_scores,
    aggregate_three_modes,
)

# Zero scores returned when scoring fails or yields no tokens
_ZERO_LEGACY_SCORES = {
    "entity_weighted_mean": 0.0,
    "max_entity_delta": 0.0,
    "proportion_above_threshold": 0.0,
    "log_entity_weighted_mean": 0.0,
    "log_max_entity_delta": 0.0,
    "posterior_entity_weighted_mean": 0.0,
}


@dataclass
class HaluEvalScoredExample:
    """Scored HaluEval example with legacy + three-mode RI scores."""

    example_id: str
    task: str
    label: int  # 1 = hallucinated, 0 = correct
    scores: dict  # legacy aggregate_all() output (entity-weighted mean etc.)
    text_length: int
    model_name: str
    n_entities: int
    # Three-mode aggregation (Experiment B)
    three_mode: dict = field(default_factory=dict)


def _score_single_text(
    text: str,
    model_name: str,
    rank_table,
    tokenizer,
    tokenizer_name: str,
    prompt: str,
    beta: float = 1.0,
) -> tuple[dict, dict, int, int]:
    """Score a single text through the full RI pipeline.

    Returns
    -------
    (legacy_scores, three_mode_scores, token_count, entity_count)
    """
    scoring_result = score_text_logprobs(text, model_name, prompt=prompt)
    token_count = len(scoring_result.token_ids)

    if token_count == 0:
        return (dict(_ZERO_LEGACY_SCORES), {}, 0, 0)

    # Legacy path (kept so existing evaluation code keeps working)
    entity_results = compute_entity_gaps(
        text=text,
        token_ids=scoring_result.token_ids,
        logprobs=scoring_result.logprobs,
        tokenizer=tokenizer,
        rank_table=rank_table,
        beta=beta,
    )
    entity_count = len(entity_results)
    if entity_count == 0:
        legacy_scores = dict(_ZERO_LEGACY_SCORES)
    else:
        legacy_scores = aggregate_all(entity_results)

    # Three-mode path (Experiment B)
    token_scores = compute_token_scores(
        text=text,
        token_ids=scoring_result.token_ids,
        logprobs=scoring_result.logprobs,
        tokenizer=tokenizer,
        rank_table=rank_table,
    )
    three_mode = aggregate_three_modes(token_scores)
    three_mode["n_entities"] = token_scores.n_entities

    return (legacy_scores, three_mode, token_count, entity_count)


def score_example(
    example: dict,
    model_name: str,
    rank_table,
    tokenizer,
    tokenizer_name: str,
    beta: float = 1.0,
) -> HaluEvalScoredExample:
    """Score one HaluEval example."""
    legacy, three_mode, text_len, n_ent = _score_single_text(
        text=example["response_text"],
        model_name=model_name,
        rank_table=rank_table,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        prompt=example["prompt"],
        beta=beta,
    )

    return HaluEvalScoredExample(
        example_id=example["example_id"],
        task=example["task"],
        label=example["label"],
        scores=legacy,
        text_length=text_len,
        model_name=model_name,
        n_entities=n_ent,
        three_mode=three_mode,
    )


def _load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load already-scored example IDs from a JSONL checkpoint file."""
    done_ids: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    done_ids.add(record["example_id"])
    return done_ids


def score_dataset(
    examples: list[dict],
    model_name: str,
    rank_table_path: str,
    output_dir: str | Path,
    beta: float = 1.0,
    checkpoint_every: int = 100,
) -> list[HaluEvalScoredExample]:
    """Score all examples with JSONL checkpointing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_table = RankTable.load(rank_table_path)
    config = get_model_config(model_name)
    tokenizer = get_tokenizer(config.tokenizer_name)

    tasks = set(ex["task"] for ex in examples)
    task_label = list(tasks)[0] if len(tasks) == 1 else "mixed"
    output_path = output_dir / f"scored_{model_name}_{task_label}.jsonl"

    done_ids = _load_checkpoint(output_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} examples already scored in {output_path}")

    remaining = [ex for ex in examples if ex["example_id"] not in done_ids]
    print(f"Scoring {len(remaining)} examples with {model_name} (task={task_label})")

    scored: list[HaluEvalScoredExample] = []
    buffer: list[dict] = []

    for i, example in enumerate(tqdm(remaining, desc=f"Scoring [{model_name}]")):
        try:
            result = score_example(
                example=example,
                model_name=model_name,
                rank_table=rank_table,
                tokenizer=tokenizer,
                tokenizer_name=config.tokenizer_name,
                beta=beta,
            )
            scored.append(result)
            buffer.append(asdict(result))

            if len(buffer) >= checkpoint_every:
                _flush_buffer(output_path, buffer)
                buffer.clear()

        except Exception as e:
            print(f"Error scoring {example['example_id']}: {e}")
            continue

    if buffer:
        _flush_buffer(output_path, buffer)

    print(f"Scored {len(scored)} examples. Output: {output_path}")
    return scored


def _flush_buffer(path: Path, buffer: list[dict]) -> None:
    """Append buffered records to the JSONL checkpoint file."""
    with open(path, "a") as f:
        for record in buffer:
            f.write(json.dumps(record) + "\n")


def load_scored_examples(path: str | Path) -> list[HaluEvalScoredExample]:
    """Load scored examples from a JSONL file."""
    path = Path(path)
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Backward compat: earlier runs did not populate three_mode
            if "three_mode" not in record:
                record["three_mode"] = {}
            results.append(HaluEvalScoredExample(**record))
    return results
