"""Score TruthfulQA candidates through the RI pipeline (Experiment B variant).

For each question, every MC1 and MC2 candidate is scored through:
  1. LLM logprob scoring (score_text_logprobs)
  2. Per-token scoring (compute_token_scores)
  3. Legacy entity-gap aggregation (aggregate_all) -- preserved for backward
     compatibility with existing evaluation code
  4. Three-mode aggregation (Experiment B): all-tokens, entity-level, rank-only

MC1/MC2 predictions continue to use the legacy entity_weighted_mean for
backward compatibility.  The three-mode aggregates are stored alongside so
Experiment B evaluation can compute AUC under each scoring mode.
"""

import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import (
    RankTable,
    get_tokenizer,
    tokenize_text,
    compute_entity_gaps,
    aggregate_all,
    score_text_logprobs,
    get_model_config,
    compute_token_scores,
    aggregate_three_modes,
)


_ZEROED_LEGACY = {
    "entity_weighted_mean": 0.0,
    "max_entity_delta": 0.0,
    "proportion_above_threshold": 0.0,
    "log_entity_weighted_mean": 0.0,
    "log_max_entity_delta": 0.0,
    "posterior_entity_weighted_mean": 0.0,
}


@dataclass
class TruthfulQAScoredQuestion:
    """Scored results for a single TruthfulQA question."""

    question_idx: int
    question: str
    category: str
    tier: str
    mc1_candidate_scores: list[dict]   # legacy aggregate_all() dicts
    mc1_predicted_idx: int             # argmin of entity_weighted_mean
    mc1_correct_idx: int
    mc2_candidate_scores: list[dict]
    mc2_rank_order: list[int]          # legacy ranking
    model_name: str
    # Experiment B three-mode aggregates, one dict per candidate
    mc1_three_mode: list[dict] = field(default_factory=list)
    mc2_three_mode: list[dict] = field(default_factory=list)


def score_candidate(
    text: str,
    model_name: str,
    rank_table,
    tokenizer,
    tokenizer_name: str,
    question: str,
    beta: float = 1.0,
) -> tuple[dict, dict]:
    """Score one candidate answer through the RI pipeline.

    Returns
    -------
    (legacy_aggregates, three_mode_aggregates)
    """
    prompt = f"Answer the following question:\n{question}\n\nAnswer:"

    scoring_result = score_text_logprobs(text, model_name, prompt=prompt)

    if not scoring_result.token_ids:
        return dict(_ZEROED_LEGACY), {}

    entity_results = compute_entity_gaps(
        text=text,
        token_ids=scoring_result.token_ids,
        logprobs=scoring_result.logprobs,
        tokenizer=tokenizer,
        rank_table=rank_table,
        beta=beta,
    )
    legacy = aggregate_all(entity_results) if entity_results else dict(_ZEROED_LEGACY)

    token_scores = compute_token_scores(
        text=text,
        token_ids=scoring_result.token_ids,
        logprobs=scoring_result.logprobs,
        tokenizer=tokenizer,
        rank_table=rank_table,
    )
    three_mode = aggregate_three_modes(token_scores)
    three_mode["n_entities"] = token_scores.n_entities

    return legacy, three_mode


def score_question(
    question_data: dict,
    model_name: str,
    rank_table,
    tokenizer,
    tokenizer_name: str,
    tier_annotations: dict,
    beta: float = 1.0,
) -> TruthfulQAScoredQuestion:
    """Score all MC1 and MC2 candidates for one question."""
    idx = question_data["question_idx"]
    question = question_data["question"]
    category = question_data["category"]

    tier = "unknown"
    if idx in tier_annotations:
        tier = tier_annotations[idx].tier

    mc1 = question_data["mc1_targets"]
    mc1_legacy_list: list[dict] = []
    mc1_three_list: list[dict] = []
    for choice in mc1["choices"]:
        legacy, three_mode = score_candidate(
            text=choice,
            model_name=model_name,
            rank_table=rank_table,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            question=question,
            beta=beta,
        )
        mc1_legacy_list.append(legacy)
        mc1_three_list.append(three_mode)

    mc1_ewm = [s["entity_weighted_mean"] for s in mc1_legacy_list]
    mc1_predicted_idx = int(min(range(len(mc1_ewm)), key=lambda i: mc1_ewm[i]))
    mc1_correct_idx = mc1["labels"].index(1)

    mc2 = question_data["mc2_targets"]
    mc2_legacy_list: list[dict] = []
    mc2_three_list: list[dict] = []
    for choice in mc2["choices"]:
        legacy, three_mode = score_candidate(
            text=choice,
            model_name=model_name,
            rank_table=rank_table,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            question=question,
            beta=beta,
        )
        mc2_legacy_list.append(legacy)
        mc2_three_list.append(three_mode)

    mc2_ewm = [s["entity_weighted_mean"] for s in mc2_legacy_list]
    mc2_rank_order = sorted(range(len(mc2_ewm)), key=lambda i: mc2_ewm[i])

    return TruthfulQAScoredQuestion(
        question_idx=idx,
        question=question,
        category=category,
        tier=tier,
        mc1_candidate_scores=mc1_legacy_list,
        mc1_predicted_idx=mc1_predicted_idx,
        mc1_correct_idx=mc1_correct_idx,
        mc2_candidate_scores=mc2_legacy_list,
        mc2_rank_order=mc2_rank_order,
        model_name=model_name,
        mc1_three_mode=mc1_three_list,
        mc2_three_mode=mc2_three_list,
    )


def score_dataset(
    questions: list[dict],
    model_name: str,
    rank_table_path: str | Path,
    output_dir: str | Path,
    tier_annotations: dict,
    beta: float = 1.0,
    checkpoint_every: int = 50,
) -> list[TruthfulQAScoredQuestion]:
    """Score all questions with checkpointing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"scored_{model_name}_checkpoint.json"
    final_path = output_dir / f"scored_{model_name}.json"

    rank_table = RankTable.load(Path(rank_table_path))
    config = get_model_config(model_name)
    tokenizer = get_tokenizer(config.tokenizer_name)
    tokenizer_name = config.tokenizer_name

    scored: list[TruthfulQAScoredQuestion] = []
    done_indices: set[int] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        for item in checkpoint_data:
            # Backward compat for checkpoints without three-mode fields
            item.setdefault("mc1_three_mode", [])
            item.setdefault("mc2_three_mode", [])
            sq = TruthfulQAScoredQuestion(**item)
            scored.append(sq)
            done_indices.add(sq.question_idx)
        print(f"Resumed from checkpoint: {len(scored)} questions already scored.")

    remaining = [q for q in questions if q["question_idx"] not in done_indices]
    print(f"Scoring {len(remaining)} remaining questions with {model_name}...")

    for i, q in enumerate(tqdm(remaining, desc=f"Scoring [{model_name}]")):
        try:
            sq = score_question(
                question_data=q,
                model_name=model_name,
                rank_table=rank_table,
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                tier_annotations=tier_annotations,
                beta=beta,
            )
            scored.append(sq)
        except Exception as e:
            print(f"  Error scoring question {q['question_idx']}: {e}")
            continue

        if (i + 1) % checkpoint_every == 0:
            _save_scored(scored, checkpoint_path)
            print(f"  Checkpoint saved at {len(scored)} questions.")

    _save_scored(scored, final_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Scoring complete: {len(scored)} questions saved to {final_path}")

    return scored


def _save_scored(
    scored: list[TruthfulQAScoredQuestion], path: Path
) -> None:
    """Save scored questions to JSON."""
    data = [asdict(sq) for sq in scored]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_scored_questions(path: Path) -> list[TruthfulQAScoredQuestion]:
    """Load scored questions from JSON."""
    with open(path) as f:
        data = json.load(f)
    for item in data:
        item.setdefault("mc1_three_mode", [])
        item.setdefault("mc2_three_mode", [])
    return [TruthfulQAScoredQuestion(**item) for item in data]
