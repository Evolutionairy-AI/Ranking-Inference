"""Runner script for streaming Wikipedia corpus processing.

Processes one or all tokenizers, builds Mandelbrot Ranking Distribution rank
tables at full Wikipedia scale (~4B tokens), and saves results to disk.

Usage examples:
    python run_corpus_scaling.py --tokenizer gpt-5.1
    python run_corpus_scaling.py --tokenizer all --max-articles 100000
    python run_corpus_scaling.py --tokenizer llama-3.1-8b --subset 20231101.en
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.utils.corpus_utils import get_tokenizer
from shared.utils.corpus_scaling import process_full_wikipedia

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKENIZER_NAMES = ["gpt-5.1", "claude-sonnet-4", "llama-3.1-8b"]

OUTPUT_DIR = Path("shared/rank_tables")
METADATA_PATH = Path("shared/corpus_metadata.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def output_path_for(tokenizer_name: str) -> Path:
    """Return the output path for a given tokenizer's Wikipedia rank table."""
    safe_name = tokenizer_name.replace("/", "_").replace(" ", "_")
    return OUTPUT_DIR / f"wikipedia_full_{safe_name}.json"


def run_tokenizer(
    tokenizer_name: str,
    subset: str,
    max_articles: int | None,
) -> dict:
    """Process Wikipedia for a single tokenizer. Returns metadata dict."""
    print(f"\n{'='*60}")
    print(f"Processing tokenizer: {tokenizer_name}")
    print(f"{'='*60}")

    tokenizer = get_tokenizer(tokenizer_name)
    out_path = output_path_for(tokenizer_name)

    start = time.time()
    rank_table = process_full_wikipedia(
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        output_path=out_path,
        subset=subset,
        max_articles=max_articles,
    )
    elapsed = time.time() - start

    metadata = {
        "tokenizer_name": tokenizer_name,
        "subset": subset,
        "max_articles": max_articles,
        "total_tokens": rank_table.total_tokens,
        "vocab_size": rank_table.vocab_size,
        "output_path": str(out_path),
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"\n[runner] Finished {tokenizer_name} in {elapsed:.1f}s")
    print(f"  total_tokens : {rank_table.total_tokens:,}")
    print(f"  vocab_size   : {rank_table.vocab_size:,}")
    print(f"  saved to     : {out_path}")

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Mandelbrot Ranking Distribution rank tables from Wikipedia."
    )
    parser.add_argument(
        "--tokenizer",
        choices=TOKENIZER_NAMES + ["all"],
        required=True,
        help="Tokenizer to use, or 'all' to process all three.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N articles (default: all articles).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="20231101.en",
        help="Wikipedia dump subset (default: 20231101.en).",
    )
    args = parser.parse_args()

    # Determine which tokenizers to process
    if args.tokenizer == "all":
        names_to_process = TOKENIZER_NAMES
    else:
        names_to_process = [args.tokenizer]

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load or initialise metadata
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            corpus_metadata = json.load(f)
    else:
        corpus_metadata = {}

    # Process each tokenizer
    for name in names_to_process:
        try:
            meta = run_tokenizer(
                tokenizer_name=name,
                subset=args.subset,
                max_articles=args.max_articles,
            )
            corpus_metadata[name] = meta
        except Exception as exc:
            print(f"[runner] ERROR processing {name}: {exc}")
            corpus_metadata[name] = {"error": str(exc)}

    # Save metadata
    with open(METADATA_PATH, "w") as f:
        json.dump(corpus_metadata, f, indent=2)
    print(f"\n[runner] Metadata saved to {METADATA_PATH}")

    print("\n[runner] All done.")


if __name__ == "__main__":
    main()
