"""TruthfulQA Taxonomy Validation -- End-to-end runner.

Usage:
    python exp05_truthfulqa/run.py --annotate-tiers [--method heuristic|llm] [--max-examples 20]
    python exp05_truthfulqa/run.py --model llama-3.1-8b [--max-examples 20]
    python exp05_truthfulqa/run.py --model all --evaluate
    python exp05_truthfulqa/run.py --evaluate-only  (skip scoring, just evaluate existing results)
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp05_truthfulqa.src.load_dataset import load_truthfulqa
from exp05_truthfulqa.src.annotate_tiers import (
    annotate_all,
    load_tier_annotations,
)
from exp05_truthfulqa.src.score_examples import score_dataset, load_scored_questions
from exp05_truthfulqa.src.evaluate import evaluate_truthfulqa
from exp05_truthfulqa.src.visualize import (
    plot_tier_gradient,
    plot_stratified_roc,
    plot_category_breakdown,
)


MODELS = ["gpt-5.1", "claude-sonnet-4", "llama-3.1-8b"]
EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
RANK_TABLE_DIR = PROJECT_ROOT / "shared" / "rank_tables"


def run_annotation(args):
    """Run tier annotation on the dataset."""
    print("=" * 60)
    print("PHASE 1: Tier Annotation")
    print("=" * 60)

    questions = load_truthfulqa(max_examples=args.max_examples)
    print(f"Loaded {len(questions)} questions.")

    annotation_path = DATA_DIR / "tier_annotations.json"

    # Load rank table and tokenizer for heuristic method
    rank_table = None
    tokenizer = None
    tokenizer_name = "llama-3.1-8b"

    if args.method == "heuristic":
        rt_path = RANK_TABLE_DIR / "wikipedia_full_llama-3.1-8b.json"
        if rt_path.exists():
            from shared.utils import RankTable, get_tokenizer
            rank_table = RankTable.load(rt_path)
            tokenizer = get_tokenizer(tokenizer_name)
            print(f"Loaded rank table from {rt_path}")
        else:
            print(f"Warning: Rank table not found at {rt_path}. "
                  "Using entity-only heuristic (no rank lookup).")

    annotations = annotate_all(
        questions=questions,
        output_path=annotation_path,
        method=args.method,
        resume=True,
        rank_table=rank_table,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
    )

    print(f"Annotations saved to {annotation_path}")
    return annotations


def run_scoring(args):
    """Run RI scoring on the dataset."""
    print("=" * 60)
    print("PHASE 2: RI Scoring")
    print("=" * 60)

    questions = load_truthfulqa(max_examples=args.max_examples)
    print(f"Loaded {len(questions)} questions.")

    # Load tier annotations
    annotation_path = DATA_DIR / "tier_annotations.json"
    if not annotation_path.exists():
        print("No tier annotations found. Running heuristic annotation first...")
        args_copy = argparse.Namespace(**vars(args))
        args_copy.method = "heuristic"
        run_annotation(args_copy)

    tier_annotations = load_tier_annotations(annotation_path)
    print(f"Loaded {len(tier_annotations)} tier annotations.")

    # Determine which models to score
    if args.model == "all":
        models_to_score = MODELS
    else:
        models_to_score = [args.model]

    all_scored = {}
    for model_name in models_to_score:
        print(f"\n--- Scoring with {model_name} ---")
        rt_path = RANK_TABLE_DIR / f"wikipedia_full_{model_name}.json"
        if not rt_path.exists():
            print(f"  Warning: Rank table not found at {rt_path}. Skipping.")
            continue

        output_dir = RESULTS_DIR / model_name
        scored = score_dataset(
            questions=questions,
            model_name=model_name,
            rank_table_path=rt_path,
            output_dir=output_dir,
            tier_annotations=tier_annotations,
            beta=1.0,
            checkpoint_every=50,
        )
        all_scored[model_name] = scored

    return all_scored


def run_evaluation(args):
    """Run evaluation on existing scored results."""
    print("=" * 60)
    print("PHASE 3: Evaluation")
    print("=" * 60)

    all_results = {}

    for model_name in MODELS:
        scored_path = RESULTS_DIR / model_name / f"scored_{model_name}.json"
        if not scored_path.exists():
            print(f"  No scored results for {model_name}. Skipping.")
            continue

        scored = load_scored_questions(scored_path)
        print(f"  Loaded {len(scored)} scored questions for {model_name}.")

        results = evaluate_truthfulqa(scored)
        all_results[model_name] = results

        # Print summary
        print(f"\n  === {model_name} Results ===")
        print(f"  MC1 Accuracy: {results['mc1_accuracy']:.3f}")
        print(f"    Tier 1: {results['mc1_accuracy_tier1']:.3f}")
        print(f"    Tier 2: {results['mc1_accuracy_tier2']:.3f}")
        print(f"  Stratified AUC:")
        for k, v in results["stratified_auc"].items():
            print(f"    {k}: {v:.3f}")
        print(f"  Cohen's d (tier gradient): {results['cohens_d_tier_gradient']:.3f}")
        print(f"  Bootstrap CIs:")
        for tier, ci in results["bootstrap_ci"].items():
            print(f"    {tier}: [{ci[0]:.3f}, {ci[1]:.3f}, {ci[2]:.3f}]")

    if not all_results:
        print("No results to evaluate. Run scoring first.")
        return {}

    # Save combined results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        # Convert numpy types for JSON serialization
        serializable = {}
        for model, res in all_results.items():
            s = {}
            for k, v in res.items():
                if isinstance(v, dict):
                    s[k] = {
                        kk: (list(vv) if isinstance(vv, tuple) else vv)
                        for kk, vv in v.items()
                    }
                else:
                    s[k] = v
            serializable[model] = s
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate figures
    print("\n--- Generating Figures ---")
    figures_dir = RESULTS_DIR / "figures"

    plot_tier_gradient(all_results, figures_dir)

    # Per-model ROC and category breakdown
    for model_name in all_results:
        scored_path = RESULTS_DIR / model_name / f"scored_{model_name}.json"
        scored = load_scored_questions(scored_path)
        plot_stratified_roc(scored, figures_dir)
        plot_category_breakdown(scored, figures_dir)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="TruthfulQA Taxonomy Validation Experiment"
    )
    parser.add_argument(
        "--annotate-tiers",
        action="store_true",
        help="Run tier annotation phase",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="heuristic",
        choices=["heuristic", "llm"],
        help="Tier annotation method (default: heuristic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to score with (or 'all')",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after scoring",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation on existing results",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples (for testing)",
    )

    args = parser.parse_args()

    # Create dirs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.evaluate_only:
        run_evaluation(args)
        return

    if args.annotate_tiers:
        run_annotation(args)

    if args.model:
        run_scoring(args)

    if args.evaluate or args.evaluate_only:
        run_evaluation(args)

    if not args.annotate_tiers and not args.model and not args.evaluate:
        parser.print_help()


if __name__ == "__main__":
    main()
