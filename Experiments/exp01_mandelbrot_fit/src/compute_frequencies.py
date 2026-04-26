"""
Step 2: Compute token frequencies and ranks from LLM outputs.

Tokenizes all generated outputs using each model's tokenizer,
counts token frequencies, and builds rank tables.
"""

import json
import sys
from pathlib import Path
from collections import Counter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import get_tokenizer, tokenize_text, build_rank_table

EXP_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = EXP_DIR / "data" / "outputs"
RANK_TABLE_DIR = PROJECT_ROOT / "shared" / "rank_tables"


def compute_frequencies_for_model(model_name: str) -> dict:
    """Compute token frequency distribution from a model's outputs.

    Returns dict with:
        - freq_table: {token_id: count}
        - rank_table: RankTable object
        - by_domain: {domain: {token_id: count}}
        - total_tokens: int
    """
    output_file = OUTPUT_DIR / model_name / "all_outputs.json"
    if not output_file.exists():
        raise FileNotFoundError(f"No outputs found for {model_name}. Run generate_outputs.py first.")

    with open(output_file) as f:
        data = json.load(f)

    # Skip models with no successful outputs
    successful = [o for o in data["outputs"] if o.get("text") and "error" not in o]
    if not successful:
        raise FileNotFoundError(f"No successful outputs for {model_name} (all {len(data['outputs'])} failed).")

    tokenizer = get_tokenizer(model_name)

    all_token_ids = []
    domain_token_ids = {}

    for output in tqdm(data["outputs"], desc=f"Tokenizing {model_name}"):
        if "error" in output or not output.get("text"):
            continue

        domain = output["domain"]
        tokens = tokenize_text(output["text"], tokenizer, model_name)
        all_token_ids.extend(tokens)

        if domain not in domain_token_ids:
            domain_token_ids[domain] = []
        domain_token_ids[domain].extend(tokens)

    # Build rank tables
    global_rank_table = build_rank_table(all_token_ids, model_name, f"{model_name}_all")
    domain_rank_tables = {}
    for domain, tokens in domain_token_ids.items():
        domain_rank_tables[domain] = build_rank_table(tokens, model_name, f"{model_name}_{domain}")

    return {
        "global_rank_table": global_rank_table,
        "domain_rank_tables": domain_rank_tables,
        "total_tokens": len(all_token_ids),
        "domain_token_counts": {d: len(t) for d, t in domain_token_ids.items()},
    }


def main():
    RANK_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    models = ["gpt-5.1", "claude-sonnet-4-6", "llama-3.1-8b", "gemini-2.5-pro", "mistral-large", "qwen-2.5-7b"]
    summary = {}

    for model_name in models:
        print(f"\n=== Processing {model_name} ===")

        try:
            result = compute_frequencies_for_model(model_name)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        # Save global rank table
        table_path = RANK_TABLE_DIR / f"{model_name}_global.json"
        result["global_rank_table"].save(table_path)
        print(f"  Global rank table saved: {table_path}")
        print(f"  Total tokens: {result['total_tokens']:,}")
        print(f"  Unique tokens (vocab): {result['global_rank_table'].vocab_size:,}")

        # Save per-domain rank tables
        for domain, rt in result["domain_rank_tables"].items():
            domain_path = RANK_TABLE_DIR / f"{model_name}_{domain}.json"
            rt.save(domain_path)
            print(f"  {domain}: {result['domain_token_counts'][domain]:,} tokens, {rt.vocab_size:,} unique")

        summary[model_name] = {
            "total_tokens": result["total_tokens"],
            "vocab_size": result["global_rank_table"].vocab_size,
            "domain_counts": result["domain_token_counts"],
        }

    # Save summary
    summary_path = EXP_DIR / "data" / "frequency_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
