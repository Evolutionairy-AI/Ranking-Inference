"""Tier annotation for TruthfulQA questions.

Each question is classified as:
- tier1: Wrong answers use distributionally anomalous vocabulary (fabricated
  entities, unusual word combinations). RI should detect these.
- tier2: Wrong answers use entirely normal vocabulary but assert false facts.
  RI should NOT detect these.
- ambiguous: Unclear mapping.
"""

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.entity_extraction import extract_entities
from shared.utils.corpus_utils import get_tokenizer, tokenize_text


@dataclass
class TierAnnotation:
    """Tier classification for a single TruthfulQA question."""

    question_idx: int
    tier: str  # "tier1", "tier2", "ambiguous"
    confidence: str  # "high", "medium", "low"
    rationale: str


def _compute_mean_rank(text: str, rank_table, tokenizer, tokenizer_name: str) -> float:
    """Compute mean rank of tokens in *text* using the rank table.

    Returns vocab_size + 1 if text produces no tokens.
    """
    token_ids = tokenize_text(text, tokenizer, tokenizer_name)
    if not token_ids:
        return float(rank_table.vocab_size + 1)
    ranks = [rank_table.get_rank(tid) for tid in token_ids]
    return float(np.mean(ranks))


def _compute_entity_rank_stats(
    text: str, rank_table, tokenizer, tokenizer_name: str
) -> dict:
    """Extract entities from text and compute rank statistics.

    Returns dict with:
      - entity_count: number of entities found
      - entities: list of (entity_text, mean_rank) tuples
      - mean_entity_rank: mean rank across all entity tokens
      - max_entity_rank: max mean rank among entities
      - has_unseen: whether any entity token is unseen in rank table
    """
    entities = extract_entities(text)
    if not entities:
        return {
            "entity_count": 0,
            "entities": [],
            "mean_entity_rank": 0.0,
            "max_entity_rank": 0.0,
            "has_unseen": False,
        }

    entity_stats = []
    all_ranks = []
    has_unseen = False
    unseen_rank = rank_table.vocab_size + 1

    for ent in entities:
        token_ids = tokenize_text(ent.text, tokenizer, tokenizer_name)
        if not token_ids:
            continue
        ent_ranks = [rank_table.get_rank(tid) for tid in token_ids]
        mean_rank = float(np.mean(ent_ranks))
        entity_stats.append((ent.text, mean_rank))
        all_ranks.extend(ent_ranks)
        if any(r >= unseen_rank for r in ent_ranks):
            has_unseen = True

    if not all_ranks:
        return {
            "entity_count": len(entities),
            "entities": entity_stats,
            "mean_entity_rank": 0.0,
            "max_entity_rank": 0.0,
            "has_unseen": False,
        }

    return {
        "entity_count": len(entities),
        "entities": entity_stats,
        "mean_entity_rank": float(np.mean(all_ranks)),
        "max_entity_rank": float(max(r for _, r in entity_stats)) if entity_stats else 0.0,
        "has_unseen": has_unseen,
    }


def classify_tier_heuristic(
    question: str,
    wrong_answers: list[str],
    rank_table=None,
    tokenizer=None,
    tokenizer_name: str = "llama-3.1-8b",
) -> TierAnnotation:
    """Heuristic tier classification using entity extraction + rank lookup.

    Tier 1 indicators (check wrong answers):
    - Contains entities not found in Wikipedia rank table (fabricated)
    - Has tokens with very high rank (rare/unusual vocabulary)
    - Contains unusual word collocations

    Tier 2 indicators:
    - All entities are common/real
    - Vocabulary is entirely normal/common
    - Error is purely factual (semantic, not distributional)

    When no rank_table/tokenizer provided, falls back to entity-only heuristic:
    entities with numbers or very long names suggest tier1.
    """
    question_idx = -1  # Caller should set this after

    # Fallback: entity-only heuristic (no rank table)
    if rank_table is None or tokenizer is None:
        all_entities = []
        for ans in wrong_answers:
            all_entities.extend(extract_entities(ans))

        if not all_entities:
            return TierAnnotation(
                question_idx=question_idx,
                tier="tier2",
                confidence="low",
                rationale="No entities found in wrong answers; default to tier2.",
            )

        # Rough heuristic: long entity names or entities with digits suggest fabrication
        has_suspicious = any(
            len(e.text.split()) > 4 or any(c.isdigit() for c in e.text)
            for e in all_entities
        )
        if has_suspicious:
            return TierAnnotation(
                question_idx=question_idx,
                tier="tier1",
                confidence="low",
                rationale="Wrong answers contain suspiciously long or numeric entities (no rank table).",
            )
        return TierAnnotation(
            question_idx=question_idx,
            tier="tier2",
            confidence="low",
            rationale="Entities appear normal; likely factual error (no rank table).",
        )

    # Full heuristic with rank table
    # Compute 90th percentile rank as threshold for "anomalous"
    rank_90th = int(rank_table.vocab_size * 0.9)

    combined_stats = {
        "entity_count": 0,
        "max_entity_rank": 0.0,
        "mean_entity_rank": 0.0,
        "has_unseen": False,
        "high_rank_entities": [],
    }

    for ans in wrong_answers:
        stats = _compute_entity_rank_stats(ans, rank_table, tokenizer, tokenizer_name)
        combined_stats["entity_count"] += stats["entity_count"]
        if stats["max_entity_rank"] > combined_stats["max_entity_rank"]:
            combined_stats["max_entity_rank"] = stats["max_entity_rank"]
        if stats["has_unseen"]:
            combined_stats["has_unseen"] = True
        # Track entities above the 90th percentile rank
        for ent_text, ent_rank in stats["entities"]:
            if ent_rank > rank_90th:
                combined_stats["high_rank_entities"].append((ent_text, ent_rank))

    # Decision logic
    if combined_stats["has_unseen"]:
        return TierAnnotation(
            question_idx=question_idx,
            tier="tier1",
            confidence="high",
            rationale=(
                f"Wrong answers contain unseen entity tokens "
                f"(not in Wikipedia rank table). "
                f"Entity count: {combined_stats['entity_count']}."
            ),
        )

    if combined_stats["high_rank_entities"]:
        n_high = len(combined_stats["high_rank_entities"])
        example = combined_stats["high_rank_entities"][0][0]
        return TierAnnotation(
            question_idx=question_idx,
            tier="tier1",
            confidence="medium",
            rationale=(
                f"{n_high} entities above 90th percentile rank "
                f"(threshold={rank_90th}). Example: '{example}'. "
                f"Max entity rank: {combined_stats['max_entity_rank']:.0f}."
            ),
        )

    if combined_stats["entity_count"] == 0:
        return TierAnnotation(
            question_idx=question_idx,
            tier="tier2",
            confidence="medium",
            rationale="No entities found in wrong answers; error is likely semantic/factual.",
        )

    return TierAnnotation(
        question_idx=question_idx,
        tier="tier2",
        confidence="high",
        rationale=(
            f"All {combined_stats['entity_count']} entities have normal ranks "
            f"(max={combined_stats['max_entity_rank']:.0f}, "
            f"threshold={rank_90th}). Error is factual, not distributional."
        ),
    )


def classify_tier_llm(
    question: str,
    wrong_answers: list[str],
    correct_answers: list[str],
) -> TierAnnotation:
    """LLM-assisted tier classification via Anthropic API.

    Sends structured prompt to Claude asking for tier classification with
    rationale.  Requires ANTHROPIC_API_KEY or API_KEYS/Claude_Key.txt.
    """
    import anthropic

    # Load API key
    key_path = PROJECT_ROOT / "API_KEYS" / "Claude_Key.txt"
    if key_path.exists():
        api_key = key_path.read_text().strip()
    else:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise FileNotFoundError(
                f"No API key found at {key_path} or ANTHROPIC_API_KEY env var."
            )

    client = anthropic.Anthropic(api_key=api_key)

    wrong_str = "\n".join(f"  - {a}" for a in wrong_answers[:5])
    correct_str = "\n".join(f"  - {a}" for a in correct_answers[:3])

    prompt = f"""You are classifying TruthfulQA questions for a Ranking Inference hallucination detection study.

QUESTION: {question}

CORRECT ANSWERS:
{correct_str}

WRONG ANSWERS:
{wrong_str}

Classify this question into one of three tiers:

TIER 1 (distributional anomaly): The wrong answers contain fabricated entities, unusual
word combinations, or vocabulary that would be statistically rare in Wikipedia. A system
that detects token-level distributional anomalies SHOULD catch these.

TIER 2 (normal vocabulary, false facts): The wrong answers use entirely common, real-world
vocabulary but assert false facts. A distributional anomaly detector should NOT catch these
because the words themselves are all normal.

AMBIGUOUS: The wrong answers have mixed characteristics or are unclear.

Respond in EXACTLY this JSON format (no other text):
{{"tier": "tier1" or "tier2" or "ambiguous", "confidence": "high" or "medium" or "low", "rationale": "one sentence explanation"}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.content[0].text.strip()

    # Parse JSON response
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            parsed = {
                "tier": "ambiguous",
                "confidence": "low",
                "rationale": f"Failed to parse LLM response: {response_text[:100]}",
            }

    tier = parsed.get("tier", "ambiguous")
    if tier not in ("tier1", "tier2", "ambiguous"):
        tier = "ambiguous"

    confidence = parsed.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    return TierAnnotation(
        question_idx=-1,  # Caller sets this
        tier=tier,
        confidence=confidence,
        rationale=parsed.get("rationale", "No rationale provided."),
    )


def annotate_all(
    questions: list[dict],
    output_path: Path,
    method: str = "heuristic",
    resume: bool = True,
    rank_table=None,
    tokenizer=None,
    tokenizer_name: str = "llama-3.1-8b",
) -> list[TierAnnotation]:
    """Annotate all questions. Save to tier_annotations.json.

    Args:
        questions: list of question dicts from load_truthfulqa()
        output_path: where to save the JSON annotations
        method: "heuristic" or "llm"
        resume: if True and output_path exists, load existing and skip done
        rank_table: RankTable instance (required for heuristic method)
        tokenizer: tokenizer instance (required for heuristic method)
        tokenizer_name: tokenizer name for tokenize_text()

    Returns:
        list of TierAnnotation objects
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    existing: dict[int, TierAnnotation] = {}
    if resume and output_path.exists():
        existing = load_tier_annotations(output_path)
        print(f"Resuming: {len(existing)} existing annotations loaded.")

    annotations: list[TierAnnotation] = []

    for q in questions:
        idx = q["question_idx"]

        if idx in existing:
            annotations.append(existing[idx])
            continue

        # Extract wrong answers from MC1 targets
        mc1 = q["mc1_targets"]
        wrong_answers = [
            choice for choice, label in zip(mc1["choices"], mc1["labels"])
            if label == 0
        ]
        correct_answers = [
            choice for choice, label in zip(mc1["choices"], mc1["labels"])
            if label == 1
        ]

        if method == "heuristic":
            ann = classify_tier_heuristic(
                question=q["question"],
                wrong_answers=wrong_answers,
                rank_table=rank_table,
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
            )
        elif method == "llm":
            ann = classify_tier_llm(
                question=q["question"],
                wrong_answers=wrong_answers,
                correct_answers=correct_answers,
            )
        else:
            raise ValueError(f"Unknown method: {method}. Expected 'heuristic' or 'llm'.")

        ann.question_idx = idx
        annotations.append(ann)

        # Periodic save
        if len(annotations) % 50 == 0:
            _save_annotations(annotations, output_path)
            print(f"  Saved checkpoint at {len(annotations)} annotations.")

    _save_annotations(annotations, output_path)
    print(f"Annotation complete: {len(annotations)} questions annotated.")

    # Print tier distribution
    tier_counts = {}
    for a in annotations:
        tier_counts[a.tier] = tier_counts.get(a.tier, 0) + 1
    print(f"Tier distribution: {tier_counts}")

    return annotations


def _save_annotations(annotations: list[TierAnnotation], output_path: Path) -> None:
    """Save annotations list to JSON."""
    data = [asdict(a) for a in annotations]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_tier_annotations(path: Path) -> dict[int, TierAnnotation]:
    """Load saved annotations as {question_idx: TierAnnotation}."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    result: dict[int, TierAnnotation] = {}
    for item in data:
        ann = TierAnnotation(**item)
        result[ann.question_idx] = ann
    return result
