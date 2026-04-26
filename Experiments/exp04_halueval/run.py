"""HaluEval Binary Detection -- End-to-end runner.

Usage:
    python exp04_halueval/run.py --model llama-3.1-8b --task qa --max-examples 10
    python exp04_halueval/run.py --model all --task all
    python exp04_halueval/run.py --evaluate-only
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import SUPPORTED_MODELS

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
RANK_TABLE_DIR = PROJECT_ROOT / "shared" / "rank_tables"

ALL_MODELS = list(SUPPORTED_MODELS.keys())
ALL_TASKS = ["qa", "dialogue", "summarization"]
AGGREGATION_STRATEGIES = [
    "entity_weighted_mean",
    "max_entity_delta",
    "proportion_above_threshold",
]


def run_scoring(
    model_name: str,
    task: str,
    max_examples: int | None,
    beta: float,
) -> None:
    """Load dataset and score examples through the RI pipeline."""
    from exp04_halueval.src.load_dataset import load_halueval_split
    from exp04_halueval.src.score_examples import score_dataset

    print(f"\n{'=' * 60}")
    print(f"SCORING: model={model_name}, task={task}")
    print(f"{'=' * 60}")

    examples = load_halueval_split(task, max_examples=max_examples)
    print(f"Loaded {len(examples)} examples for task '{task}'")

    rank_table_path = RANK_TABLE_DIR / f"wikipedia_full_{model_name}.json"
    if not rank_table_path.exists():
        print(f"WARNING: Rank table not found at {rank_table_path}")
        print("Falling back to any available rank table...")
        available = list(RANK_TABLE_DIR.glob("wikipedia_full_*.json"))
        if not available:
            print("ERROR: No rank tables found. Run corpus scaling first.")
            return
        rank_table_path = available[0]
        print(f"Using: {rank_table_path}")

    output_dir = DATA_DIR / model_name
    score_dataset(
        examples=examples,
        model_name=model_name,
        rank_table_path=str(rank_table_path),
        output_dir=str(output_dir),
        beta=beta,
    )


def run_evaluation() -> None:
    """Evaluate all scored results and generate visualisations."""
    from exp04_halueval.src.score_examples import load_scored_examples
    from exp04_halueval.src.evaluate import evaluate_by_task
    from exp04_halueval.src.visualize import (
        plot_roc_curves,
        plot_entity_delta_distributions,
        generate_results_table,
    )

    print(f"\n{'=' * 60}")
    print("EVALUATION")
    print(f"{'=' * 60}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find all scored JSONL files
    scored_files = list(DATA_DIR.glob("*/scored_*.jsonl"))
    if not scored_files:
        print("No scored files found. Run scoring first.")
        return

    print(f"Found {len(scored_files)} scored file(s)")

    all_results = {}
    roc_data = {}

    # Group files by model
    model_files: dict[str, list[Path]] = {}
    for f in scored_files:
        model_name = f.parent.name
        model_files.setdefault(model_name, []).append(f)

    for model_name, files in model_files.items():
        print(f"\nEvaluating model: {model_name}")

        # Load all scored examples for this model
        all_examples = []
        for f in files:
            examples = load_scored_examples(f)
            all_examples.extend(examples)
            print(f"  Loaded {len(examples)} from {f.name}")

        if not all_examples:
            continue

        # Evaluate with each aggregation strategy
        model_results = {}
        for strategy in AGGREGATION_STRATEGIES:
            print(f"  Strategy: {strategy}")
            task_results = evaluate_by_task(all_examples, aggregation_strategy=strategy)
            model_results[strategy] = task_results

            # Print summary
            if "all" in task_results:
                m = task_results["all"]["metrics"]
                lr = task_results["all"]["length_regression"]
                print(f"    AUC-ROC: {m['roc_auc']:.3f} "
                      f"CI: [{m['roc_auc_ci'][0]:.3f}, {m['roc_auc_ci'][1]:.3f}]")
                print(f"    AUC-PR:  {m['pr_auc']:.3f}")
                print(f"    F1@opt:  {m['f1_optimal']:.3f} (t={m['f1_threshold']:.3f})")
                print(f"    Cohen's d: {m['cohens_d']:.3f}")
                print(f"    Length-only AUC: {lr['auc_length_only']:.3f}, "
                      f"RI+Length AUC: {lr['auc_ri_plus_length']:.3f}, "
                      f"Delta: {lr['delta_auc']:.3f}")

        all_results[model_name] = model_results

        # Prepare ROC curve data
        roc_data[model_name] = _prepare_roc_data(all_examples)

        # Distribution plots per model
        plot_entity_delta_distributions(all_examples, RESULTS_DIR)

    # ROC curves
    if roc_data:
        from exp04_halueval.src.visualize import plot_roc_curves
        plot_roc_curves(roc_data, RESULTS_DIR)

    # Generate results tables for each strategy
    for strategy in AGGREGATION_STRATEGIES:
        combined_results = {}
        for model_name, model_results in all_results.items():
            if strategy in model_results:
                for task_name, task_data in model_results[strategy].items():
                    key = f"{model_name}/{task_name}"
                    combined_results[key] = task_data

        if combined_results:
            strategy_dir = RESULTS_DIR / strategy
            from exp04_halueval.src.visualize import generate_results_table
            generate_results_table(combined_results, strategy_dir)

    print(f"\nResults saved to {RESULTS_DIR}")


def _prepare_roc_data(scored_examples: list) -> dict:
    """Prepare data for ROC curve plotting."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from exp04_halueval.src.evaluate import _extract_data

    roc_data = {}
    for strategy in AGGREGATION_STRATEGIES:
        labels, scores, lengths = _extract_data(scored_examples, strategy)

        # Fit length-only model for baseline ROC
        lengths_std = lengths.reshape(-1, 1)
        if lengths_std.std() > 0:
            lengths_std = (lengths_std - lengths_std.mean()) / lengths_std.std()
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(lengths_std, labels)
        length_scores = lr.predict_proba(lengths_std)[:, 1]
        from shared.utils import compute_roc_auc
        length_auc = compute_roc_auc(labels, length_scores)
        ri_auc = compute_roc_auc(labels, scores)

        roc_data[strategy] = {
            "labels": labels,
            "scores": scores,
            "length_scores": length_scores,
            "auc": ri_auc,
            "length_auc": length_auc,
        }

    return roc_data


def main():
    parser = argparse.ArgumentParser(
        description="HaluEval Binary Detection Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score 10 QA examples with llama
  python exp04_halueval/run.py --model llama-3.1-8b --task qa --max-examples 10

  # Score all tasks with all models
  python exp04_halueval/run.py --model all --task all

  # Only evaluate existing scored data
  python exp04_halueval/run.py --evaluate-only

  # Resume interrupted scoring
  python exp04_halueval/run.py --model llama-3.1-8b --task qa --resume
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b",
        help=f"Model to use for scoring. Options: {ALL_MODELS + ['all']} (default: llama-3.1-8b)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="qa",
        help=f"HaluEval task. Options: {ALL_TASKS + ['all']} (default: qa)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum examples per task (default: all)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta parameter for entity gap computation (default: 1.0)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoints (automatic if checkpoint exists)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip scoring, only evaluate existing results",
    )

    args = parser.parse_args()

    # Determine models and tasks
    models = ALL_MODELS if args.model == "all" else [args.model]
    tasks = ALL_TASKS if args.task == "all" else [args.task]

    # Validate
    for m in models:
        if m not in ALL_MODELS:
            print(f"ERROR: Unknown model '{m}'. Options: {ALL_MODELS}")
            sys.exit(1)
    for t in tasks:
        if t not in ALL_TASKS:
            print(f"ERROR: Unknown task '{t}'. Options: {ALL_TASKS}")
            sys.exit(1)

    # Scoring phase
    if not args.evaluate_only:
        for model in models:
            for task in tasks:
                run_scoring(model, task, args.max_examples, args.beta)

    # Evaluation phase
    run_evaluation()

    print(f"\n{'=' * 60}")
    print("EXPERIMENT 04 (HaluEval) COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
