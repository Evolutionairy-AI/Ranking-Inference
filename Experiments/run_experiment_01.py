"""
Master runner for Experiment 01: Mandelbrot Ranking Distribution Fit Analysis.

Runs all steps sequentially:
  1. Generate LLM outputs (calls APIs)
  2. Compute token frequencies and ranks
  3. Fit Mandelbrot distribution and evaluate
  4. Generate plots

Usage:
  # Full pipeline:
  python run_experiment_01.py

  # Individual steps:
  python run_experiment_01.py --step generate
  python run_experiment_01.py --step frequencies
  python run_experiment_01.py --step fit
  python run_experiment_01.py --step plot
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Load API keys from API_KEYS/ directory
from shared.utils.api_keys import load_api_keys
load_api_keys()


def run_step(step: str):
    if step == "generate":
        print("\n" + "=" * 60)
        print("STEP 1: Generating LLM outputs")
        print("=" * 60)
        from exp01_mandelbrot_fit.src.generate_outputs import main
        main()

    elif step == "frequencies":
        print("\n" + "=" * 60)
        print("STEP 2: Computing token frequencies and ranks")
        print("=" * 60)
        from exp01_mandelbrot_fit.src.compute_frequencies import main
        main()

    elif step == "fit":
        print("\n" + "=" * 60)
        print("STEP 3: Fitting Mandelbrot distribution")
        print("=" * 60)
        from exp01_mandelbrot_fit.src.fit_mandelbrot import main
        main()

    elif step == "plot":
        print("\n" + "=" * 60)
        print("STEP 4: Generating plots")
        print("=" * 60)
        from exp01_mandelbrot_fit.src.plot_results import main
        main()

    else:
        print(f"Unknown step: {step}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 01")
    parser.add_argument(
        "--step",
        choices=["generate", "frequencies", "fit", "plot", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    steps = ["generate", "frequencies", "fit", "plot"] if args.step == "all" else [args.step]

    for step in steps:
        run_step(step)

    print("\n" + "=" * 60)
    print("EXPERIMENT 01 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
