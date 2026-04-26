"""Phase 2 Orchestrator — Run all three core benchmarks.

Usage:
    python run_phase2.py --model all --quick       # validation run (10 examples each)
    python run_phase2.py --model all --full         # full benchmark run
    python run_phase2.py --evaluate-only            # re-evaluate from saved scores
    python run_phase2.py --model llama-3.1-8b       # single model, all benchmarks
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS = ["gpt-5.1", "claude-sonnet-4", "llama-3.1-8b"]
RANK_TABLE_DIR = PROJECT_ROOT / "shared" / "rank_tables"


def run_halueval(model: str, max_examples: int | None, evaluate_only: bool, beta: float):
    """Run HaluEval benchmark for one model."""
    from exp04_halueval.src.load_dataset import load_all_tasks
    from exp04_halueval.src.score_examples import score_dataset
    from exp04_halueval.src.evaluate import evaluate_by_task

    output_dir = PROJECT_ROOT / "exp04_halueval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not evaluate_only:
        rank_table_path = RANK_TABLE_DIR / f"wikipedia_full_{model}.json"
        if not rank_table_path.exists():
            # Fallback for claude which uses different naming
            rank_table_path = RANK_TABLE_DIR / f"wikipedia_full_claude-sonnet-4.json"

        examples = load_all_tasks(max_per_task=max_examples)
        print(f"\n{'='*60}")
        print(f"HaluEval: Scoring {len(examples)} examples with {model}")
        print(f"{'='*60}")
        score_dataset(examples, model, rank_table_path, output_dir, beta=beta)

    # Evaluate
    results = {}
    for strategy in ["entity_weighted_mean", "max_entity_delta", "proportion_above_threshold"]:
        task_results = evaluate_by_task(_load_scored_examples("halueval", model, output_dir), strategy)
        results[strategy] = task_results

    return results


def run_truthfulqa(model: str, max_examples: int | None, evaluate_only: bool, beta: float):
    """Run TruthfulQA benchmark for one model."""
    from exp05_truthfulqa.src.load_dataset import load_truthfulqa
    from exp05_truthfulqa.src.annotate_tiers import annotate_all, load_tier_annotations
    from exp05_truthfulqa.src.score_examples import score_dataset
    from exp05_truthfulqa.src.evaluate import evaluate_truthfulqa

    data_dir = PROJECT_ROOT / "exp05_truthfulqa" / "data"
    output_dir = PROJECT_ROOT / "exp05_truthfulqa" / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = data_dir / "tier_annotations.json"

    if not evaluate_only:
        questions = load_truthfulqa(max_examples=max_examples)

        # Ensure tier annotations exist
        if not annotations_path.exists():
            print(f"\n{'='*60}")
            print(f"TruthfulQA: Annotating {len(questions)} questions for tier classification")
            print(f"{'='*60}")
            annotate_all(questions, annotations_path, method="heuristic")

        tier_annotations = load_tier_annotations(annotations_path)
        rank_table_path = RANK_TABLE_DIR / f"wikipedia_full_{model}.json"
        if not rank_table_path.exists():
            rank_table_path = RANK_TABLE_DIR / f"wikipedia_full_claude-sonnet-4.json"

        print(f"\n{'='*60}")
        print(f"TruthfulQA: Scoring {len(questions)} questions with {model}")
        print(f"{'='*60}")
        score_dataset(questions, model, rank_table_path, output_dir, tier_annotations, beta=beta)

    # Evaluate
    scored = _load_scored_questions("truthfulqa", model, output_dir)
    results = evaluate_truthfulqa(scored)
    return results


def run_frank(model: str, max_examples: int | None, evaluate_only: bool, beta: float):
    """Run FRANK benchmark for one model."""
    from exp06_frank.src.load_dataset import load_frank
    from exp06_frank.src.score_examples import score_dataset
    from exp06_frank.src.evaluate import evaluate_frank

    output_dir = PROJECT_ROOT / "exp06_frank" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not evaluate_only:
        examples = load_frank(max_examples=max_examples)
        rank_table_path = RANK_TABLE_DIR / f"wikipedia_full_{model}.json"
        if not rank_table_path.exists():
            rank_table_path = RANK_TABLE_DIR / f"wikipedia_full_claude-sonnet-4.json"

        print(f"\n{'='*60}")
        print(f"FRANK: Scoring {len(examples)} examples with {model}")
        print(f"{'='*60}")
        score_dataset(examples, model, rank_table_path, output_dir, beta=beta)

    # Evaluate — check both output/ and results/ for scored data
    scored = _load_scored_spans("frank", model, output_dir)
    if not scored:
        alt_dir = PROJECT_ROOT / "exp06_frank" / "output"
        scored = _load_scored_spans("frank", model, alt_dir)
    results = evaluate_frank(scored, model)
    return results


def _load_scored_examples(benchmark, model, output_dir):
    """Load scored examples from JSONL checkpoint files."""
    from exp04_halueval.src.score_examples import HaluEvalScoredExample
    results = []
    for pattern in [f"scored_{model}_*.jsonl", f"scored_{model}.jsonl"]:
        for path in output_dir.glob(pattern):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        results.append(HaluEvalScoredExample(**data))
    return results


def _load_scored_questions(benchmark, model, output_dir):
    """Load scored TruthfulQA questions from JSON or JSONL."""
    from exp05_truthfulqa.src.score_examples import TruthfulQAScoredQuestion

    # Try model subdirectory first (where the scorer writes), then output_dir
    search_dirs = [output_dir / model, output_dir]
    results = []

    for search_dir in search_dirs:
        if results:
            break
        # Try JSON array format first (what the scorer actually produces)
        json_path = search_dir / f"scored_{model}.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            for item in data:
                results.append(TruthfulQAScoredQuestion(**item))
            break
        # Fall back to JSONL format
        jsonl_path = search_dir / f"scored_{model}.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(TruthfulQAScoredQuestion(**json.loads(line)))
            break

    return results


def _load_scored_spans(benchmark, model, output_dir):
    """Load scored FRANK spans from checkpoint."""
    from exp06_frank.src.score_examples import FRANKScoredSpan
    results = []
    # Try both naming patterns
    for pattern in [f"scored_frank_{model}.jsonl", f"scored_{model}.jsonl"]:
        path = output_dir / pattern
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        results.append(FRANKScoredSpan(**data))
    return results


def print_cross_benchmark_summary(all_results: dict):
    """Print cross-benchmark summary table."""
    print(f"\n{'='*80}")
    print("PHASE 2: CROSS-BENCHMARK SUMMARY")
    print(f"{'='*80}")

    header = f"{'Benchmark':<25} {'Model':<18} {'Strategy':<25} {'AUC-ROC':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))

    for model, benchmarks in all_results.items():
        for bench_name, results in benchmarks.items():
            if bench_name == "halueval" and isinstance(results, dict):
                for strategy, task_results in results.items():
                    if isinstance(task_results, dict):
                        for task, task_data in task_results.items():
                            if isinstance(task_data, dict):
                                # Metrics are nested under "metrics" key
                                m = task_data.get("metrics", task_data)
                                auc = m.get("roc_auc", m.get("auc_roc", "—"))
                                f1 = m.get("f1_optimal", m.get("f1", "—"))
                                auc_str = f"{auc:.3f}" if isinstance(auc, float) else str(auc)
                                f1_str = f"{f1:.3f}" if isinstance(f1, float) else str(f1)
                                print(f"HaluEval {task:<16} {model:<18} {strategy:<25} {auc_str:>8} {f1_str:>8}")

            elif bench_name == "truthfulqa" and isinstance(results, dict):
                mc1 = results.get("mc1_accuracy", "—")
                mc1_str = f"{mc1:.3f}" if isinstance(mc1, float) else str(mc1)
                print(f"{'TruthfulQA (MC1)':<25} {model:<18} {'entity_weighted_mean':<25} {mc1_str:>8} {'—':>8}")

                stratified = results.get("stratified_auc", {})
                for tier in ["tier1", "tier2"]:
                    auc = stratified.get(f"{tier}_auc", stratified.get(tier, "—"))
                    auc_str = f"{auc:.3f}" if isinstance(auc, float) else str(auc)
                    print(f"{'TruthfulQA (' + tier + ')':<25} {model:<18} {'entity_weighted_mean':<25} {auc_str:>8} {'—':>8}")

                gradient = stratified.get("tier_gradient", "—")
                grad_str = f"{gradient:+.3f}" if isinstance(gradient, float) else str(gradient)
                print(f"{'TruthfulQA (gradient)':<25} {model:<18} {'tier1 - tier2':<25} {grad_str:>8} {'—':>8}")

            elif bench_name == "frank" and isinstance(results, dict):
                for baseline_type in ["global", "source"]:
                    per_type = results.get(f"{baseline_type}_per_type", results.get("per_type", {}))
                    if isinstance(per_type, dict):
                        for error_type, metrics in per_type.items():
                            if isinstance(metrics, dict):
                                f1 = metrics.get("f1", "—")
                                auc = metrics.get("auc", "—")
                                f1_str = f"{f1:.3f}" if isinstance(f1, float) else str(f1)
                                auc_str = f"{auc:.3f}" if isinstance(auc, float) else str(auc)
                                print(f"FRANK {error_type} ({baseline_type}){'':<7} {model:<18} {'span-level':<25} {auc_str:>8} {f1_str:>8}")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Core Benchmarks Orchestrator")
    parser.add_argument("--model", default="all", help="Model name or 'all'")
    parser.add_argument("--quick", action="store_true", help="Quick validation (10 examples each)")
    parser.add_argument("--full", action="store_true", help="Full benchmark run")
    parser.add_argument("--evaluate-only", action="store_true", help="Re-evaluate from saved scores")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter")
    parser.add_argument("--benchmark", default="all",
                        help="Which benchmark(s): halueval, truthfulqa, frank, or all")
    args = parser.parse_args()

    models = MODELS if args.model == "all" else [args.model]
    max_examples = 10 if args.quick else None

    all_results = {}
    start_time = time.time()

    for model in models:
        print(f"\n{'#'*60}")
        print(f"# Model: {model}")
        print(f"{'#'*60}")

        all_results[model] = {}

        if args.benchmark in ("all", "halueval"):
            try:
                all_results[model]["halueval"] = run_halueval(
                    model, max_examples, args.evaluate_only, args.beta)
            except Exception as e:
                print(f"HaluEval failed for {model}: {e}")
                all_results[model]["halueval"] = {"error": str(e)}

        if args.benchmark in ("all", "truthfulqa"):
            try:
                all_results[model]["truthfulqa"] = run_truthfulqa(
                    model, max_examples, args.evaluate_only, args.beta)
            except Exception as e:
                print(f"TruthfulQA failed for {model}: {e}")
                all_results[model]["truthfulqa"] = {"error": str(e)}

        if args.benchmark in ("all", "frank"):
            try:
                all_results[model]["frank"] = run_frank(
                    model, max_examples, args.evaluate_only, args.beta)
            except Exception as e:
                print(f"FRANK failed for {model}: {e}")
                all_results[model]["frank"] = {"error": str(e)}

    elapsed = time.time() - start_time
    print_cross_benchmark_summary(all_results)
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Save results
    results_path = PROJECT_ROOT / "phase2_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
