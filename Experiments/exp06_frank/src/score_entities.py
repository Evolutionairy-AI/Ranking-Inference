"""Entity-level scoring for FRANK summaries (Experiment D).

Adrian's critique of Experiment B's "entity-level" row: Table 2 actually
scored *spans* with per-token features restricted to NER positions, but
the spans are still the classification target (is this sentence an
error?).  The prediction was about a different task: for each named
entity extracted from a summary, is that specific entity fabricated
(i.e. not present in the source article)?

This script implements the true entity-level task:

  1. For each FRANK example, call the LLM to score the summary (token
     IDs + logprobs).
  2. Run SpaCy NER on the summary.
  3. For each extracted entity:
       - Align to BPE tokens.
       - Label: fabricated iff the normalized entity text is NOT a
         substring of the normalized article_text.
       - Features: mean log-delta (global Wikipedia baseline), mean
         log-rank, mean -log G_RI, mean rank deviation between global
         and source-article rank tables.
  4. Emit one JSONL row per entity (not per span).
  5. Consumer computes AUC over entities.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import (
    RankTable,
    build_rank_table,
    get_tokenizer,
    tokenize_text,
    score_text_logprobs,
    get_model_config,
)
from shared.utils.entity_extraction import (
    EntitySpan,
    align_entities_to_tokens,
    extract_entities,
    get_grounding_scores,
    _FALLBACK_P_LLM,
)

_LOG_FLOOR = 1e-20

# Entity types treated as "named" (as opposed to numeric/date entities).
_NAMED_ENTITY_TYPES = {
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
}

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from",
    "by", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this", "that",
    "these", "those", "it", "its",
}


# ---------------------------------------------------------------------------
# Normalisation / grounding check
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, collapse non-alphanumeric to single spaces, trim."""
    t = text.lower()
    t = re.sub(r"[^0-9a-z]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _content_tokens(text: str) -> list[str]:
    """Split normalized text into content tokens (len >= 3, not stopword)."""
    return [
        w for w in _normalize(text).split()
        if len(w) >= 3 and w not in _STOPWORDS
    ]


def entity_is_fabricated_strict(entity_text: str, article_text: str) -> bool:
    """True iff the whole normalized entity string is NOT in the article."""
    norm_entity = _normalize(entity_text)
    if not norm_entity:
        return False
    norm_article = _normalize(article_text)
    return norm_entity not in norm_article


def entity_is_fabricated_relaxed(
    entity_text: str, article_text: str,
) -> bool:
    """True iff no content word of the entity appears in the article.

    This is a more forgiving label: "Dr. James Morton" is considered
    grounded if "Morton" appears in the article, even if the full
    string does not.
    """
    norm_article = _normalize(article_text)
    article_tokens = set(norm_article.split())
    content = _content_tokens(entity_text)
    if not content:
        # No testable content (e.g. a bare number like "42"); fall back
        # to strict check.
        return entity_is_fabricated_strict(entity_text, article_text)
    return not any(tok in article_tokens for tok in content)


# ---------------------------------------------------------------------------
# Source-article rank table
# ---------------------------------------------------------------------------


def _build_source_rank_table(article_text, tokenizer, tokenizer_name):
    token_ids = tokenize_text(article_text, tokenizer, tokenizer_name)
    if not token_ids:
        token_ids = [0]
    return build_rank_table(token_ids, tokenizer_name, "source_article")


# ---------------------------------------------------------------------------
# Scored record
# ---------------------------------------------------------------------------


@dataclass
class FRANKScoredEntity:
    """One scored named entity from a FRANK summary."""

    example_id: str
    entity_text: str
    entity_type: str
    n_tokens: int
    is_named: bool          # True iff entity_type in _NAMED_ENTITY_TYPES
    fabricated_strict: bool  # whole entity NOT in article
    fabricated_relaxed: bool  # no content word of entity in article
    # Features computed over the entity's BPE tokens in the summary
    mean_log_delta: float            # mean of (logP_LLM - log G_RI_global)
    mean_log_delta_source: float     # mean of (logP_LLM - log G_RI_source)
    mean_neg_log_g_global: float     # mean of -log G_RI_global  (rank-only)
    mean_neg_log_g_source: float     # mean of -log G_RI_source  (rank-only)
    mean_log_rank_global: float      # mean of log2(global_rank)
    mean_rank_deviation: float       # mean of log2(r_global / r_source)
    model_name: str


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


def score_entities_for_example(
    example,
    model_name: str,
    global_rank_table,
    tokenizer,
    tokenizer_name: str,
) -> list[FRANKScoredEntity]:
    """Score every named entity in one FRANK example.

    Returns [] if the LLM scoring fails or no entities are found.
    """
    summary = example.summary_text
    article = example.article_text

    prompt = f"Summarize the following article:\n\n{article[:2000]}"
    scoring_result = score_text_logprobs(summary, model_name, prompt=prompt)
    if not scoring_result.token_ids:
        return []

    token_ids = scoring_result.token_ids
    logprobs = scoring_result.logprobs
    n_tokens = len(token_ids)

    entities = extract_entities(summary)
    if not entities:
        return []

    source_rank_table = _build_source_rank_table(
        article, tokenizer, tokenizer_name,
    )
    scores_global, default_g_global = get_grounding_scores(global_rank_table)
    scores_source, default_g_source = get_grounding_scores(source_rank_table)

    aligned = align_entities_to_tokens(entities, summary, token_ids, tokenizer)

    out: list[FRANKScoredEntity] = []
    for ent, indices in aligned:
        if not indices:
            continue

        log_deltas: list[float] = []
        log_deltas_source: list[float] = []
        neg_log_g_globals: list[float] = []
        neg_log_g_sources: list[float] = []
        log_ranks_global: list[float] = []
        rank_deviations: list[float] = []

        for idx in indices:
            if idx >= n_tokens:
                continue
            tid = token_ids[idx]
            lp = logprobs[idx] if idx < len(logprobs) else None
            log_p = lp if lp is not None else math.log(_FALLBACK_P_LLM)

            g_global = scores_global.get(tid, default_g_global)
            g_source = scores_source.get(tid, default_g_source)
            log_g_global = math.log(max(g_global, _LOG_FLOOR))
            log_g_source = math.log(max(g_source, _LOG_FLOOR))

            log_deltas.append(log_p - log_g_global)
            log_deltas_source.append(log_p - log_g_source)
            neg_log_g_globals.append(-log_g_global)
            neg_log_g_sources.append(-log_g_source)

            r_global = max(global_rank_table.get_rank(tid), 1)
            r_source = max(source_rank_table.get_rank(tid), 1)
            log_ranks_global.append(math.log2(r_global))
            rank_deviations.append(math.log2(r_global / r_source))

        if not log_deltas:
            continue

        out.append(FRANKScoredEntity(
            example_id=example.article_id,
            entity_text=ent.text[:200],
            entity_type=ent.entity_type,
            n_tokens=len(log_deltas),
            is_named=ent.entity_type in _NAMED_ENTITY_TYPES,
            fabricated_strict=entity_is_fabricated_strict(ent.text, article),
            fabricated_relaxed=entity_is_fabricated_relaxed(ent.text, article),
            mean_log_delta=float(np.mean(log_deltas)),
            mean_log_delta_source=float(np.mean(log_deltas_source)),
            mean_neg_log_g_global=float(np.mean(neg_log_g_globals)),
            mean_neg_log_g_source=float(np.mean(neg_log_g_sources)),
            mean_log_rank_global=float(np.mean(log_ranks_global)),
            mean_rank_deviation=float(np.mean(rank_deviations)),
            model_name=model_name,
        ))

    return out


# ---------------------------------------------------------------------------
# Dataset-level scoring with checkpointing
# ---------------------------------------------------------------------------


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done.add(rec["example_id"])
    return done


def _append(path: Path, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def score_dataset_entities(
    examples,
    model_name: str,
    global_rank_table_path: str,
    output_dir: Path,
    checkpoint_every: int = 25,
) -> list[FRANKScoredEntity]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_rank_table = RankTable.load(global_rank_table_path)
    config = get_model_config(model_name)
    tokenizer = get_tokenizer(config.tokenizer_name)

    output_path = output_dir / f"scored_frank_entities_{model_name}.jsonl"

    done = _load_done_ids(output_path)
    if done:
        print(f"Resuming: {len(done)} example ids already scored.")
    remaining = [ex for ex in examples if ex.article_id not in done]
    print(f"Entity-scoring {len(remaining)} FRANK examples with {model_name}.")

    all_entities: list[FRANKScoredEntity] = []
    buffer: list[dict] = []

    for ex in tqdm(remaining, desc=f"Entities [{model_name}]"):
        try:
            rows = score_entities_for_example(
                example=ex, model_name=model_name,
                global_rank_table=global_rank_table,
                tokenizer=tokenizer, tokenizer_name=config.tokenizer_name,
            )
        except Exception as e:
            print(f"Error scoring example {ex.article_id}: {e}")
            continue

        all_entities.extend(rows)
        buffer.extend(asdict(r) for r in rows)

        if len(buffer) >= checkpoint_every:
            _append(output_path, buffer)
            buffer.clear()

    if buffer:
        _append(output_path, buffer)

    print(f"Scored {len(all_entities)} entities from {len(remaining)} examples.")
    print(f"Output: {output_path}")
    return all_entities


def load_scored_entities(path) -> list[FRANKScoredEntity]:
    path = Path(path)
    out: list[FRANKScoredEntity] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(FRANKScoredEntity(**rec))
    return out
