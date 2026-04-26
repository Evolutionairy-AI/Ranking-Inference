"""
Master runner for Experiment 03: Confidence-Grounding Gap Signal Analysis.

Steps:
  1. Generate outputs with logprobs (factual vs hallucination-inducing)
  2. Compute confidence-grounding gap delta(t) per token
  3. Statistical analysis and plots

Usage:
  python run_experiment_03.py
  python run_experiment_03.py --step generate
  python run_experiment_03.py --step gap
  python run_experiment_03.py --step analyze
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shared.utils.api_keys import load_api_keys
load_api_keys()


def run_step(step: str):
    if step == "generate":
        print("\n" + "=" * 60)
        print("STEP 1: Generating outputs with logprobs")
        print("=" * 60)
        from exp03_gap_signal.src.generate_with_logprobs import main
        main()

    elif step == "gap":
        print("\n" + "=" * 60)
        print("STEP 2: Computing confidence-grounding gap")
        print("=" * 60)
        from exp03_gap_signal.src.compute_gap import main
        main()

    elif step == "analyze":
        print("\n" + "=" * 60)
        print("STEP 3: Statistical analysis and plots")
        print("=" * 60)
        from exp03_gap_signal.src.analyze_and_plot import main
        main()

    else:
        print(f"Unknown step: {step}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 03")
    parser.add_argument(
        "--step",
        choices=["generate", "gap", "analyze", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    steps = ["generate", "gap", "analyze"] if args.step == "all" else [args.step]

    for step in steps:
        run_step(step)

    print("\n" + "=" * 60)
    print("EXPERIMENT 03 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
