"""FRANK Span-Level Detection -- End-to-end runner.

Runs the full FRANK benchmark pipeline: load dataset, score summaries
with dual baselines (global Wikipedia + source-article rank tables),
evaluate span-level detection per error type, and generate visualisations.

Usage:
    python exp06_frank/run.py --model llama-3.1-8b --max-examples 10
    python exp06_frank/run.py --model all
    python exp06_frank/run.py --evaluate-only
    python exp06_frank/run.py --model gpt-5.1 --beta 1.5
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import SUPPORTED_MODELS

from exp06_frank.src.load_dataset import load_frank
from exp06_frank.src.score_examples import score_dataset, load_scored_spans
from exp06_frank.src.evaluate import evaluate_frank
from exp06_frank.src.visualize import (
    plot_error_type_gradient,
    plot_dual_baseline_comparison,
    plot_span_delta_distributions,
)


def _find_rank_table(model_name: str) -> Path:
    """Locate the global Wikipedia rank table for a model."""
    rank_dir = PROJECT_ROOT / "shared" / "rank_tables"
    pattern = f"wikipedia_full_{model_name}.json"
    path = rank_dir / pattern
    if path.exists():
        return path
    # Try alternative names
    for candidate in rank_dir.glob("wikipedia_*.json"):
        if model_name.replace("-", "_") in candidate.stem or model_name in candidate.stem:
            return candidate
    raise FileNotFoundError(
        f"No rank table found for {model_name} in {rank_dir}. "
        f"Available: {[p.name for p in rank_dir.glob('*.json')]}"
    )


def run_scoring(model_name: str, examples, output_dir: Path, beta: float):
    """Run scoring for a single model."""
    rank_table_path = _find_rank_table(model_name)
    print(f"Using rank table: {rank_table_path}")

    scored_spans = score_dataset(
        examples=examples,
        model_name=model_name,
        global_rank_table_path=str(rank_table_path),
        output_dir=output_dir,
        beta=beta,
        checkpoint_every=50,
    )
    return scored_spans


def run_evaluation(model_name: str, output_dir: Path, results_dir: Path):
    """Run evaluation and visualisation for a single model."""
    scored_path = output_dir / f"scored_frank_{model_name}.jsonl"
    if not scored_path.exists():
        print(f"No scored data found at {scored_path}. Run scoring first.")
        return None

    scored_spans = load_scored_spans(scored_path)
    if not scored_spans:
        print(f"No scored spans found in {scored_path}.")
        return None

    # Evaluate
    results = evaluate_frank(scored_spans, model_name)

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"frank_results_{model_name}.json"

    # Convert non-serialisable values for JSON
    serialisable = _make_serialisable(results)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Saved results to {results_path}")

    # Visualise
    figures_dir = results_dir / "figures"
    plot_error_type_gradient(results, figures_dir)
    plot_dual_baseline_comparison(results, figures_dir)
    plot_span_delta_distributions(scored_spans, figures_dir)

    return results


def _make_serialisable(obj):
    """Recursively convert numpy types and tuples for JSON serialisation."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="FRANK Span-Level Detection Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp06_frank/run.py --model llama-3.1-8b --max-examples 10
  python exp06_frank/run.py --model all
  python exp06_frank/run.py --evaluate-only --model gpt-5.1
        """,
    )
    parser.add_argument(
        "--model", type=str, default="llama-3.1-8b",
        help="Model name (gpt-5.1, claude-sonnet-4, llama-3.1-8b, or 'all')",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Maximum number of FRANK examples to load",
    )
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="Scaling parameter for RI computation",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Skip scoring, only run evaluation on existing scored data",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory for FRANK dataset cache",
    )

    args = parser.parse_args()

    exp_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else exp_dir / "data"
    output_dir = exp_dir / "output"
    results_dir = exp_dir / "results"

    # Determine models to run
    if args.model == "all":
        models = list(SUPPORTED_MODELS.keys())
    else:
        if args.model not in SUPPORTED_MODELS:
            print(f"Unknown model: {args.model}. Supported: {list(SUPPORTED_MODELS.keys())}")
            sys.exit(1)
        models = [args.model]

    if not args.evaluate_only:
        # Load dataset
        print(f"Loading FRANK dataset (max_examples={args.max_examples})...")
        examples = load_frank(data_dir=data_dir, max_examples=args.max_examples)

        if not examples:
            print("No FRANK examples loaded. Check dataset availability.")
            sys.exit(1)

        # Score each model
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"Scoring with {model_name}")
            print(f"{'='*60}")
            try:
                run_scoring(model_name, examples, output_dir, args.beta)
            except Exception as e:
                print(f"Error scoring with {model_name}: {e}")
                continue

    # Evaluate each model
    all_results = {}
    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        results = run_evaluation(model_name, output_dir, results_dir)
        if results:
            all_results[model_name] = results

    # Cross-model summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Cross-Model Summary")
        print(f"{'='*60}")
        for model_name, res in all_results.items():
            gradient = res.get("global_gradient", {})
            print(f"  {model_name}: "
                  f"AUC={res['overall_auc_global']:.3f}, "
                  f"F1={res['overall_f1_global']:.3f}, "
                  f"rho={gradient.get('spearman_rho', float('nan')):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
