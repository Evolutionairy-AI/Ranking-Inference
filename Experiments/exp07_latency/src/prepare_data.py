"""Prepare cached logprob data for latency benchmarking.

Samples ~500 examples across length bins from HaluEval and FRANK,
re-scores them through Ollama, and caches token_ids + logprobs
for the timing benchmark.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import (
    score_text_logprobs,
    get_model_config,
    get_tokenizer,
    tokenize_text,
    extract_entities,
)

LENGTH_BINS = [
    (0, 50),
    (50, 100),
    (100, 250),
    (250, 500),
    (500, 1000),
    (1000, 999999),
]
EXAMPLES_PER_BIN = 80
MODEL_NAME = "llama-3.1-8b"


def load_example_texts() -> list[dict]:
    """Load texts from HaluEval and FRANK datasets."""
    texts = []

    try:
        from exp04_halueval.src.load_dataset import load_halueval_split

        for task in ["dialogue", "qa", "summarization"]:
            examples = load_halueval_split(task, max_examples=500)
            for ex in examples:
                texts.append({
                    "source": f"halueval_{task}",
                    "text": ex["response_text"],
                    "label": ex["label"],
                })
    except Exception as e:
        print(f"Warning: Could not load HaluEval: {e}")

    try:
        from exp06_frank.src.load_dataset import load_frank

        frank_examples = load_frank(max_examples=500)
        for ex in frank_examples:
            texts.append({
                "source": "frank",
                "text": ex.summary_text,
                "label": 1 if ex.has_errors else 0,
            })
    except Exception as e:
        print(f"Warning: Could not load FRANK: {e}")

    return texts


def bin_by_length(texts: list[dict], model_name: str) -> dict[str, list[dict]]:
    """Tokenize texts and sort into length bins."""
    config = get_model_config(model_name)
    tokenizer = get_tokenizer(config.tokenizer_name)

    binned = {f"{lo}-{hi}": [] for lo, hi in LENGTH_BINS}
    for item in texts:
        token_ids = tokenize_text(item["text"], tokenizer, config.tokenizer_name)
        n_tokens = len(token_ids)
        for lo, hi in LENGTH_BINS:
            if lo <= n_tokens < hi:
                item["n_tokens"] = n_tokens
                binned[f"{lo}-{hi}"].append(item)
                break

    return binned


def prepare_cached_data(output_path: Path, model_name: str = MODEL_NAME):
    """Sample examples across length bins, score through Ollama, cache results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading example texts...")
    texts = load_example_texts()
    print(f"Loaded {len(texts)} texts")

    print("Binning by token length...")
    binned = bin_by_length(texts, model_name)
    for bin_name, items in binned.items():
        print(f"  {bin_name}: {len(items)} examples")

    print(f"\nScoring through {model_name} (this takes ~30 minutes)...")
    cached = []
    config = get_model_config(model_name)
    tokenizer = get_tokenizer(config.tokenizer_name)

    for bin_name, items in binned.items():
        sample = items[:EXAMPLES_PER_BIN]
        print(f"  Scoring {len(sample)} examples for bin {bin_name}...")

        for item in sample:
            try:
                result = score_text_logprobs(
                    item["text"], model_name, prompt="Continue:"
                )
                if not result.token_ids:
                    continue

                entities = extract_entities(item["text"])
                entity_dicts = [
                    {
                        "text": e.text,
                        "entity_type": e.entity_type,
                        "char_start": e.char_start,
                        "char_end": e.char_end,
                    }
                    for e in entities
                ]

                cached.append({
                    "source": item["source"],
                    "text": item["text"],
                    "label": item["label"],
                    "token_ids": result.token_ids,
                    "logprobs": result.logprobs,
                    "n_tokens": len(result.token_ids),
                    "length_bin": bin_name,
                    "entities": entity_dicts,
                })
            except Exception as e:
                print(f"    Error: {e}")
                continue

    with open(output_path, "w", encoding="utf-8") as f:
        for item in cached:
            f.write(json.dumps(item) + "\n")

    print(f"\nCached {len(cached)} examples to {output_path}")
    return cached


if __name__ == "__main__":
    output = PROJECT_ROOT / "exp07_latency" / "data" / "cached_logprobs.jsonl"
    prepare_cached_data(output)
