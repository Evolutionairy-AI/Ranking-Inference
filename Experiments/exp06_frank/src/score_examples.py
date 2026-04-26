"""Score FRANK summaries with dual baselines (global + source-article).

For each FRANK example, computes per-token RI deltas using two baselines:
1. Global Wikipedia rank table (standard RI grounding)
2. Source-article rank table (faithfulness-specific grounding)

Error spans are aligned from character offsets to token indices, and
control spans of matched length are sampled for paired comparison.

Experiment B additions (entity-level scoring):
  For every error / control span we also emit:
    - entity-filtered log-delta (global + source baselines)
    - rank-only signals at entity positions within the span (no logprobs):
        * mean log2(global_rank)
        * mean log2(r_global / r_source_article) -- the classic rank deviation
"""

import json
import math
import sys
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import Counter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import (
    RankTable,
    build_rank_table,
    get_tokenizer,
    tokenize_text,
    compute_entity_gaps,
    aggregate_all,
    score_text_logprobs,
    get_model_config,
    compute_token_scores,
    aggregate_three_modes,
)
from shared.utils.entity_extraction import _FALLBACK_P_LLM, get_grounding_scores

# ---------------------------------------------------------------------------
# Token-level delta computation (legacy path kept for backward compat)
# ---------------------------------------------------------------------------


_LOG_FLOOR = 1e-20  # avoid log(0)


def compute_token_deltas(text, token_ids, logprobs, rank_table):
    """Compute per-token deltas in both linear and log space (legacy API)."""
    scores, default_g = get_grounding_scores(rank_table)
    deltas = []
    log_deltas = []
    for i, tid in enumerate(token_ids):
        lp = logprobs[i] if i < len(logprobs) else None
        log_p = lp if lp is not None else math.log(_FALLBACK_P_LLM)
        p_llm = math.exp(log_p)
        g_ri = scores.get(tid, default_g)
        log_g = math.log(max(g_ri, _LOG_FLOOR))

        deltas.append(p_llm - g_ri)
        log_deltas.append(log_p - log_g)
    return np.array(deltas), np.array(log_deltas)


def compute_span_delta(token_deltas: np.ndarray, span_indices: list[int]) -> float:
    """Mean delta(t) over tokens in a span."""
    if not span_indices:
        return 0.0
    valid_indices = [i for i in span_indices if 0 <= i < len(token_deltas)]
    if not valid_indices:
        return 0.0
    return float(np.mean(token_deltas[valid_indices]))


# ---------------------------------------------------------------------------
# Span alignment
# ---------------------------------------------------------------------------


def align_char_span_to_tokens(char_start, char_end, text, token_ids, tokenizer):
    """Map a character span [char_start, char_end) to token indices."""
    token_char_spans = []
    char_pos = 0
    for idx, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        tok_len = len(decoded)
        start = text.find(decoded, char_pos)
        if start == -1:
            start = char_pos
        end = start + tok_len
        token_char_spans.append((start, end))
        char_pos = end

    indices = []
    for tok_idx, (ts, te) in enumerate(token_char_spans):
        if ts < char_end and te > char_start:
            indices.append(tok_idx)
    return indices


# ---------------------------------------------------------------------------
# Control span sampling
# ---------------------------------------------------------------------------


def sample_control_spans(n_tokens, error_token_spans, n_controls=10, seed=42):
    """Sample non-overlapping control spans of matched length."""
    rng = np.random.default_rng(seed)

    error_indices = set()
    for start, end in error_token_spans:
        for i in range(start, min(end, n_tokens)):
            error_indices.add(i)

    non_error_indices = sorted(set(range(n_tokens)) - error_indices)

    controls = []
    for start, end in error_token_spans:
        span_len = end - start
        if span_len <= 0:
            continue

        candidates = []
        for ni in non_error_indices:
            if ni + span_len <= n_tokens:
                span_range = set(range(ni, ni + span_len))
                if span_range.isdisjoint(error_indices):
                    candidates.append(ni)

        if not candidates:
            continue

        selected = set()
        attempts = 0
        max_attempts = n_controls * 10
        while len(selected) < n_controls and attempts < max_attempts:
            idx = rng.choice(candidates)
            proposed = set(range(idx, idx + span_len))
            overlap = False
            for s in selected:
                existing = set(range(s, s + span_len))
                if not proposed.isdisjoint(existing):
                    overlap = True
                    break
            if not overlap:
                selected.add(idx)
            attempts += 1

        for s in selected:
            controls.append((s, s + span_len))

    return controls


# ---------------------------------------------------------------------------
# Source-article rank table
# ---------------------------------------------------------------------------


def build_source_article_rank_table(article_text, tokenizer, tokenizer_name):
    """Build a local rank table from source article tokens."""
    token_ids = tokenize_text(article_text, tokenizer, tokenizer_name)
    if not token_ids:
        token_ids = [0]
    return build_rank_table(token_ids, tokenizer_name, "source_article")


# ---------------------------------------------------------------------------
# Rank-deviation helper (Experiment B rank-only signal)
# ---------------------------------------------------------------------------


def span_rank_deviation_at_entities(
    global_ranks: list[int],
    source_ranks: list[int],
    entity_positions: set[int],
    span_indices: list[int],
) -> tuple[float, int]:
    """Mean log2(r_global / r_source) at entity tokens within a span.

    Returns (mean_delta_r, n_entity_tokens_in_span).  ``0.0`` and ``0`` when
    no entity tokens fall inside the span.  ``r_source = vocab_size + 1`` for
    tokens absent from the source article (giving large negative Δr, which is
    the out-of-article signal).
    """
    scope = [i for i in span_indices
             if 0 <= i < len(global_ranks) and i in entity_positions]
    if not scope:
        return 0.0, 0
    deltas = []
    for i in scope:
        g = max(global_ranks[i], 1)
        l = max(source_ranks[i], 1)
        deltas.append(math.log2(g / l))
    return float(sum(deltas) / len(deltas)), len(scope)


# ---------------------------------------------------------------------------
# Scored span dataclass
# ---------------------------------------------------------------------------


@dataclass
class FRANKScoredSpan:
    """A scored span from a FRANK example (error or control).

    Carries legacy per-span deltas and Experiment B entity-level / rank-only
    signals (with ``=0.0`` defaults so older JSONL files load cleanly).
    """

    example_id: str
    span_text: str
    error_type: str
    tier: str
    is_error: bool
    mean_delta: float              # legacy linear delta (global)
    mean_delta_source: float       # legacy linear delta (source)
    model_name: str
    n_tokens: int
    mean_log_delta: float = 0.0         # legacy log-delta (global)
    mean_log_delta_source: float = 0.0  # legacy log-delta (source)
    # Experiment B: entity-filtered aggregates within this span
    entity_n_tokens: int = 0
    entity_mean_log_delta: float = 0.0         # entity tokens, global G_RI
    entity_mean_log_delta_source: float = 0.0  # entity tokens, source G_RI
    entity_mean_linear_delta: float = 0.0      # entity tokens, global, linear
    # Experiment B: rank-only signals at entity positions (no logprobs)
    rank_only_mean_log_rank: float = 0.0        # log2(global_rank)
    rank_only_mean_neg_log_g_global: float = 0.0
    rank_only_mean_neg_log_g_source: float = 0.0
    rank_only_mean_rank_deviation: float = 0.0  # log2(r_global / r_source)


# ---------------------------------------------------------------------------
# Per-example scoring
# ---------------------------------------------------------------------------


def _three_mode_entry(
    token_scores,
    span_indices: list[int],
) -> dict:
    """Aggregate-three-modes bundled with n_entity_tokens for this span."""
    return aggregate_three_modes(token_scores, token_indices=span_indices)


def score_example(example, model_name, global_rank_table, tokenizer, tokenizer_name, beta=1.0):
    """Score one FRANK example with dual baselines + Experiment B additions."""
    summary = example.summary_text
    article = example.article_text

    # Step 1: Get logprobs from the LLM
    prompt = f"Summarize the following article:\n\n{article[:2000]}"
    scoring_result = score_text_logprobs(summary, model_name, prompt=prompt)

    if not scoring_result.token_ids:
        return []

    token_ids = scoring_result.token_ids
    logprobs = scoring_result.logprobs
    n_tokens = len(token_ids)

    # Step 2: Build source-article rank table
    source_rank_table = build_source_article_rank_table(article, tokenizer, tokenizer_name)

    # Step 3 & 4: Per-token scoring with both rank tables (Experiment B)
    token_scores_global = compute_token_scores(
        text=summary, token_ids=token_ids, logprobs=logprobs,
        tokenizer=tokenizer, rank_table=global_rank_table,
    )
    token_scores_source = compute_token_scores(
        text=summary, token_ids=token_ids, logprobs=logprobs,
        tokenizer=tokenizer, rank_table=source_rank_table,
    )
    entity_positions: set[int] = set(token_scores_global.entity_positions)

    # Legacy per-token delta arrays (for existing callers that read the old fields)
    global_deltas, global_log_deltas = compute_token_deltas(
        summary, token_ids, logprobs, global_rank_table
    )
    source_deltas, source_log_deltas = compute_token_deltas(
        summary, token_ids, logprobs, source_rank_table
    )

    scored_spans = []
    error_token_spans = []

    for error_span in example.error_spans:
        span_indices = align_char_span_to_tokens(
            error_span.char_start, error_span.char_end,
            summary, token_ids, tokenizer,
        )

        if not span_indices:
            continue

        span_start = min(span_indices)
        span_end = max(span_indices) + 1
        error_token_spans.append((span_start, span_end))

        # Legacy
        mean_delta_global = compute_span_delta(global_deltas, span_indices)
        mean_delta_source = compute_span_delta(source_deltas, span_indices)
        mean_log_delta_global = compute_span_delta(global_log_deltas, span_indices)
        mean_log_delta_source = compute_span_delta(source_log_deltas, span_indices)

        # Experiment B: three-mode aggregates within the span
        tm_global = _three_mode_entry(token_scores_global, span_indices)
        tm_source = _three_mode_entry(token_scores_source, span_indices)
        rank_dev, n_ent = span_rank_deviation_at_entities(
            token_scores_global.global_ranks,
            token_scores_source.global_ranks,
            entity_positions,
            span_indices,
        )

        scored_spans.append(FRANKScoredSpan(
            example_id=example.article_id,
            span_text=error_span.text[:200],
            error_type=error_span.error_type,
            tier=error_span.tier,
            is_error=True,
            mean_delta=mean_delta_global,
            mean_delta_source=mean_delta_source,
            model_name=model_name,
            n_tokens=len(span_indices),
            mean_log_delta=mean_log_delta_global,
            mean_log_delta_source=mean_log_delta_source,
            entity_n_tokens=n_ent,
            entity_mean_log_delta=tm_global["entity_mean_log_delta"],
            entity_mean_log_delta_source=tm_source["entity_mean_log_delta"],
            entity_mean_linear_delta=tm_global["entity_mean_linear_delta"],
            rank_only_mean_log_rank=tm_global["rank_only_mean_log_rank"],
            rank_only_mean_neg_log_g_global=tm_global["rank_only_mean_neg_log_g"],
            rank_only_mean_neg_log_g_source=tm_source["rank_only_mean_neg_log_g"],
            rank_only_mean_rank_deviation=rank_dev,
        ))

    if error_token_spans:
        control_spans = sample_control_spans(
            n_tokens, error_token_spans, n_controls=5,
            seed=hash(example.article_id) % (2**31),
        )

        for cs_start, cs_end in control_spans:
            control_indices = list(range(cs_start, min(cs_end, n_tokens)))

            mean_delta_global = compute_span_delta(global_deltas, control_indices)
            mean_delta_source = compute_span_delta(source_deltas, control_indices)
            mean_log_delta_global = compute_span_delta(global_log_deltas, control_indices)
            mean_log_delta_source = compute_span_delta(source_log_deltas, control_indices)

            tm_global = _three_mode_entry(token_scores_global, control_indices)
            tm_source = _three_mode_entry(token_scores_source, control_indices)
            rank_dev, n_ent = span_rank_deviation_at_entities(
                token_scores_global.global_ranks,
                token_scores_source.global_ranks,
                entity_positions,
                control_indices,
            )

            control_text_tokens = [tokenizer.decode([token_ids[i]]) for i in control_indices]
            control_text = "".join(control_text_tokens)

            scored_spans.append(FRANKScoredSpan(
                example_id=example.article_id,
                span_text=control_text[:200],
                error_type="control",
                tier="control",
                is_error=False,
                mean_delta=mean_delta_global,
                mean_delta_source=mean_delta_source,
                model_name=model_name,
                n_tokens=len(control_indices),
                mean_log_delta=mean_log_delta_global,
                mean_log_delta_source=mean_log_delta_source,
                entity_n_tokens=n_ent,
                entity_mean_log_delta=tm_global["entity_mean_log_delta"],
                entity_mean_log_delta_source=tm_source["entity_mean_log_delta"],
                entity_mean_linear_delta=tm_global["entity_mean_linear_delta"],
                rank_only_mean_log_rank=tm_global["rank_only_mean_log_rank"],
                rank_only_mean_neg_log_g_global=tm_global["rank_only_mean_neg_log_g"],
                rank_only_mean_neg_log_g_source=tm_source["rank_only_mean_neg_log_g"],
                rank_only_mean_rank_deviation=rank_dev,
            ))

    if not example.has_errors:
        all_indices = list(range(n_tokens))
        mean_delta_global = compute_span_delta(global_deltas, all_indices)
        mean_delta_source = compute_span_delta(source_deltas, all_indices)
        mean_log_delta_global = compute_span_delta(global_log_deltas, all_indices)
        mean_log_delta_source = compute_span_delta(source_log_deltas, all_indices)

        tm_global = _three_mode_entry(token_scores_global, all_indices)
        tm_source = _three_mode_entry(token_scores_source, all_indices)
        rank_dev, n_ent = span_rank_deviation_at_entities(
            token_scores_global.global_ranks,
            token_scores_source.global_ranks,
            entity_positions,
            all_indices,
        )

        scored_spans.append(FRANKScoredSpan(
            example_id=example.article_id,
            span_text=summary[:200],
            error_type="control",
            tier="control",
            is_error=False,
            mean_delta=mean_delta_global,
            mean_delta_source=mean_delta_source,
            model_name=model_name,
            n_tokens=n_tokens,
            mean_log_delta=mean_log_delta_global,
            mean_log_delta_source=mean_log_delta_source,
            entity_n_tokens=n_ent,
            entity_mean_log_delta=tm_global["entity_mean_log_delta"],
            entity_mean_log_delta_source=tm_source["entity_mean_log_delta"],
            entity_mean_linear_delta=tm_global["entity_mean_linear_delta"],
            rank_only_mean_log_rank=tm_global["rank_only_mean_log_rank"],
            rank_only_mean_neg_log_g_global=tm_global["rank_only_mean_neg_log_g"],
            rank_only_mean_neg_log_g_source=tm_source["rank_only_mean_neg_log_g"],
            rank_only_mean_rank_deviation=rank_dev,
        ))

    return scored_spans


# ---------------------------------------------------------------------------
# Dataset-level scoring with checkpointing
# ---------------------------------------------------------------------------


def _flush_buffer(path: Path, buffer: list[dict]) -> None:
    """Append buffered records to the JSONL checkpoint file."""
    with open(path, "a", encoding="utf-8") as f:
        for record in buffer:
            f.write(json.dumps(record) + "\n")


def _load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load already-scored example IDs from a JSONL checkpoint file."""
    done_ids: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    done_ids.add(record["example_id"])
    return done_ids


def score_dataset(
    examples,
    model_name,
    global_rank_table_path,
    output_dir,
    beta=1.0,
    checkpoint_every=50,
):
    """Score all FRANK examples with checkpointing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_rank_table = RankTable.load(global_rank_table_path)
    config = get_model_config(model_name)
    tokenizer = get_tokenizer(config.tokenizer_name)

    output_path = output_dir / f"scored_frank_{model_name}.jsonl"

    done_ids = _load_checkpoint(output_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} examples already scored in {output_path}")

    remaining = [ex for ex in examples if ex.article_id not in done_ids]
    print(f"Scoring {len(remaining)} FRANK examples with {model_name}")

    all_spans: list[FRANKScoredSpan] = []
    buffer: list[dict] = []

    for i, example in enumerate(tqdm(remaining, desc=f"Scoring [{model_name}]")):
        try:
            spans = score_example(
                example=example,
                model_name=model_name,
                global_rank_table=global_rank_table,
                tokenizer=tokenizer,
                tokenizer_name=config.tokenizer_name,
                beta=beta,
            )
            all_spans.extend(spans)
            for span in spans:
                buffer.append(asdict(span))

            if len(buffer) >= checkpoint_every:
                _flush_buffer(output_path, buffer)
                buffer.clear()

        except Exception as e:
            print(f"Error scoring {example.article_id}: {e}")
            continue

    if buffer:
        _flush_buffer(output_path, buffer)

    print(f"Scored {len(all_spans)} spans from {len(remaining)} examples. Output: {output_path}")
    return all_spans


def load_scored_spans(path) -> list[FRANKScoredSpan]:
    """Load scored spans from a JSONL file. Handles old records missing Experiment B fields."""
    path = Path(path)
    results = []
    defaults = {
        "entity_n_tokens": 0,
        "entity_mean_log_delta": 0.0,
        "entity_mean_log_delta_source": 0.0,
        "entity_mean_linear_delta": 0.0,
        "rank_only_mean_log_rank": 0.0,
        "rank_only_mean_neg_log_g_global": 0.0,
        "rank_only_mean_neg_log_g_source": 0.0,
        "rank_only_mean_rank_deviation": 0.0,
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for k, v in defaults.items():
                record.setdefault(k, v)
            results.append(FRANKScoredSpan(**record))
    return results
