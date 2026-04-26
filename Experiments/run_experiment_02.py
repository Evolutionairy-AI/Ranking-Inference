"""
Master runner for Experiment 02: Beta Calibration Curves by Domain.

Runs all steps sequentially:
  1. Compute rank deviations per domain (requires reference corpus)
  2. Generate plots

Usage:
  # Full pipeline:
  python run_experiment_02.py

  # Individual steps:
  python run_experiment_02.py --step deviations
  python run_experiment_02.py --step plot
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load API keys from API_KEYS/ directory
from shared.utils.api_keys import load_api_keys
load_api_keys()


def run_step(step: str):
    if step == "deviations":
        print("\n" + "=" * 60)
        print("STEP 1: Computing rank deviations per domain")
        print("=" * 60)
        from exp02_beta_calibration.src.compute_rank_deviations import main
        main()

    elif step == "plot":
        print("\n" + "=" * 60)
        print("STEP 2: Generating plots")
        print("=" * 60)
        from exp02_beta_calibration.src.plot_results import main
        main()

    else:
        print(f"Unknown step: {step}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 02")
    parser.add_argument(
        "--step",
        choices=["deviations", "plot", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    steps = ["deviations", "plot"] if args.step == "all" else [args.step]

    for step in steps:
        run_step(step)

    print("\n" + "=" * 60)
    print("EXPERIMENT 02 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
