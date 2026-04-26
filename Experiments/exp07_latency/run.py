"""Orchestrator for the latency benchmarking experiment (exp07).

Usage:
    python exp07_latency/run.py                  # full benchmark
    python exp07_latency/run.py --ri-only        # skip baselines
    python exp07_latency/run.py --analytical     # force analytical fallback
    python exp07_latency/run.py --prepare-only   # just prepare cached data
    python exp07_latency/run.py --skip-prepare   # use existing cache
    python exp07_latency/run.py --n-runs 50      # fewer repetitions
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp07_latency.src.timer import system_info

CACHED_DATA_PATH = PROJECT_ROOT / "exp07_latency" / "data" / "cached_logprobs.jsonl"
RESULTS_DIR = PROJECT_ROOT / "exp07_latency" / "results"
FIGURES_DIR = PROJECT_ROOT / "exp07_latency" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latency benchmarking for RI verification vs baselines",
    )
    parser.add_argument(
        "--ri-only",
        action="store_true",
        help="Skip baseline benchmarks, run RI timing only",
    )
    parser.add_argument(
        "--analytical",
        action="store_true",
        help="Force analytical fallback for baselines (skip empirical attempts)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare cached data, do not run benchmarks",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Use existing cached data without re-preparing",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Number of timed repetitions per example (default: 100)",
    )
    return parser.parse_args()


def ensure_cached_data(skip_prepare: bool) -> bool:
    """Check for cached data, prepare if needed.

    Returns True if cached data is available.
    """
    if CACHED_DATA_PATH.exists():
        n_lines = sum(1 for _ in open(CACHED_DATA_PATH, encoding="utf-8"))
        print(f"Cached data found: {CACHED_DATA_PATH} ({n_lines} examples)")
        return True

    if skip_prepare:
        print(f"ERROR: --skip-prepare specified but {CACHED_DATA_PATH} not found")
        return False

    print("Preparing cached data (requires Ollama)...")
    from exp07_latency.src.prepare_data import prepare_cached_data
    prepare_cached_data(CACHED_DATA_PATH)
    return CACHED_DATA_PATH.exists()


def main():
    args = parse_args()

    print("=" * 60)
    print("Exp07: Latency Benchmarking")
    print("=" * 60)
    start_time = time.time()

    # Step 1: Ensure cached data
    print("\n--- Step 1: Check/Prepare Cached Data ---")
    if not ensure_cached_data(args.skip_prepare):
        print("Cannot proceed without cached data. Exiting.")
        sys.exit(1)

    if args.prepare_only:
        print("--prepare-only: stopping after data preparation.")
        return

    # Step 2: System info
    print("\n--- Step 2: System Information ---")
    sys_info = system_info()
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    # Step 3: RI benchmark
    print(f"\n--- Step 3: RI Benchmark ({args.n_runs} runs/example) ---")
    from exp07_latency.src.ri_benchmark import run_ri_benchmark
    ri_results = run_ri_benchmark(
        CACHED_DATA_PATH,
        n_runs=args.n_runs,
        n_warmup=max(args.n_runs // 10, 5),
    )
    print(f"  RI benchmark complete: {ri_results['n_examples']} examples")

    # Step 4: Baseline benchmark
    baseline_results = {}
    if not args.ri_only:
        print("\n--- Step 4: Baseline Benchmarks ---")
        if args.analytical:
            print("  Forcing analytical fallback...")
            from exp07_latency.src.baselines import analytical_baseline_timing
            baseline_results = analytical_baseline_timing(CACHED_DATA_PATH)
        else:
            from exp07_latency.src.baselines import run_baseline_benchmarks
            baseline_results = run_baseline_benchmarks(CACHED_DATA_PATH)
        print("  Baseline benchmarks complete")
    else:
        print("\n--- Step 4: Skipped (--ri-only) ---")

    # Step 5: Generate charts
    print("\n--- Step 5: Generate Charts ---")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from exp07_latency.src.charts import (
        plot_pareto_frontier,
        plot_latency_vs_length,
        generate_summary_table,
    )

    plot_pareto_frontier(
        ri_results, baseline_results,
        FIGURES_DIR / "pareto_frontier.png",
    )
    plot_latency_vs_length(
        ri_results, baseline_results,
        FIGURES_DIR / "latency_vs_length.png",
    )

    summary_table = generate_summary_table(ri_results, baseline_results)
    print("\n" + summary_table)

    # Step 5b: FLOPs estimation
    from exp07_latency.src.flops import estimate_flops
    flops_100 = estimate_flops(n_tokens=100)
    # Remove non-serializable markdown_table from JSON
    flops_for_json = {k: v for k, v in flops_100.items() if k != "markdown_table"}

    # Step 6: Save results
    print("\n--- Step 6: Save Results ---")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = {
        "timestamp": timestamp,
        "system_info": sys_info,
        "ri_results": ri_results,
        "baseline_results": baseline_results,
        "flops_estimate": flops_for_json,
        "args": {
            "n_runs": args.n_runs,
            "ri_only": args.ri_only,
            "analytical": args.analytical,
        },
    }

    results_path = RESULTS_DIR / f"latency_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")

    # Also save latest
    latest_path = RESULTS_DIR / "latency_results_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Latest results saved to {latest_path}")

    # Save summary table
    table_path = RESULTS_DIR / "summary_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(summary_table)
    print(f"  Summary table saved to {table_path}")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
