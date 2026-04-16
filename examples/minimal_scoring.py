"""Minimal end-to-end example: score a short text with the three aggregation modes.

Requires:
    pip install ranking-inference[ner,tokenizers]
    python -m spacy download en_core_web_sm

Usage:
    python examples/minimal_scoring.py
"""

from pathlib import Path

from transformers import AutoTokenizer

from ranking_inference import (
    RankTable,
    compute_token_scores,
    aggregate_three_modes,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RANK_TABLE = REPO_ROOT / "rank_tables" / "wikipedia_full_llama-3.1-8b.json"

# Any output from a Llama 3.1 8B compatible model.  For this demo we supply
# synthetic uniform logprobs because the point is to show the aggregation,
# not logprob provenance.
TEXT = (
    "The 1925 eruption of Mount Vesuvius devastated Pompeii, "
    "killing Marcus Agrippa and his entire family."
)


def main() -> None:
    print(f"Loading rank table: {RANK_TABLE.name}")
    rt = RankTable.load(RANK_TABLE)
    print(f"  vocab_size={rt.vocab_size:,}, total_tokens={rt.total_tokens:,}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    token_ids = tokenizer.encode(TEXT, add_special_tokens=False)
    n = len(token_ids)

    # Synthetic flat logprobs ~ -log(vocab); real usage passes actual model logprobs
    import math
    logprobs = [-math.log(rt.vocab_size)] * n

    scores = compute_token_scores(
        text=TEXT,
        token_ids=token_ids,
        logprobs=logprobs,
        tokenizer=tokenizer,
        rank_table=rt,
    )

    agg = aggregate_three_modes(scores)
    print()
    print("Three-mode aggregates:")
    for k in [
        "all_mean_log_delta",        # output-level (every token)
        "entity_mean_log_delta",      # entity-level (NER-tagged positions only)
        "rank_only_mean_log_rank",    # rank-only (no logprob dependency)
        "rank_only_mean_neg_log_g",
        "entity_n_tokens",
    ]:
        if k in agg:
            print(f"  {k:32s}  {agg[k]}")


if __name__ == "__main__":
    main()
