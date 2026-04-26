"""
Step 3: Fit Mandelbrot distribution to observed rank-frequency data.

Fits f(r) = C/(r+q)^s via MLE and OLS, compares against pure Zipf,
and computes goodness-of-fit metrics with bootstrap CIs.
"""

import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import (
    RankTable,
    fit_mandelbrot_mle,
    fit_mandelbrot_ols_loglog,
    goodness_of_fit,
    compare_distributions,
)

EXP_DIR = Path(__file__).resolve().parent.parent
RANK_TABLE_DIR = PROJECT_ROOT / "shared" / "rank_tables"
RESULTS_DIR = EXP_DIR / "results"


def fit_and_evaluate(rank_table: RankTable, label: str) -> dict:
    """Fit Mandelbrot distribution and evaluate goodness of fit.

    Args:
        rank_table: precomputed rank table
        label: identifier for this fit (e.g., "gpt-5.1_global")

    Returns:
        dict with fit parameters, GoF metrics, and model comparison
    """
    # Extract rank-frequency arrays (1-indexed)
    max_rank = rank_table.vocab_size
    ranks = np.arange(1, max_rank + 1)
    frequencies = rank_table.rank_to_freq[1:max_rank + 1].copy()

    # Filter to non-zero frequencies
    mask = frequencies > 0
    ranks_nz = ranks[mask]
    freq_nz = frequencies[mask]

    print(f"  {label}: {len(ranks_nz)} non-zero ranks, {freq_nz.sum():,} total tokens")

    # Fit via MLE
    print("  Fitting MLE...")
    mle_params = fit_mandelbrot_mle(ranks_nz, freq_nz)
    mle_gof = goodness_of_fit(ranks_nz, freq_nz, mle_params)

    # Fit via OLS on log-log
    print("  Fitting OLS...")
    ols_params = fit_mandelbrot_ols_loglog(ranks_nz, freq_nz)
    ols_gof = goodness_of_fit(ranks_nz, freq_nz, ols_params)

    # Compare Mandelbrot vs Zipf vs alternatives
    print("  Comparing distributions...")
    comparison = compare_distributions(ranks_nz, freq_nz)

    # Bootstrap CIs for MLE parameters
    print("  Bootstrap CIs (100 iterations)...")
    bootstrap_params = bootstrap_ci(ranks_nz, freq_nz, n_iterations=100)

    return {
        "label": label,
        "n_ranks": len(ranks_nz),
        "total_tokens": int(freq_nz.sum()),
        "mle": {
            "C": mle_params.C,
            "q": mle_params.q,
            "s": mle_params.s,
            "log_likelihood": mle_params.log_likelihood,
            "gof": mle_gof,
        },
        "ols": {
            "C": ols_params.C,
            "q": ols_params.q,
            "s": ols_params.s,
            "r_squared_loglog": ols_params.log_likelihood,
            "gof": ols_gof,
        },
        "model_comparison": comparison,
        "bootstrap_ci": bootstrap_params,
    }


def bootstrap_ci(
    ranks: np.ndarray,
    frequencies: np.ndarray,
    n_iterations: int = 100,
    ci_level: float = 0.95,
) -> dict:
    """Compute bootstrap confidence intervals for Mandelbrot parameters.

    Resamples the frequency data and refits to get CI on q and s.
    """
    total_count = frequencies.sum()
    probs = frequencies / total_count

    q_samples = []
    s_samples = []

    for _ in tqdm(range(n_iterations), desc="  Bootstrap", leave=False):
        # Resample token counts from multinomial
        resampled_freq = np.random.multinomial(total_count, probs)
        mask = resampled_freq > 0
        if mask.sum() < 10:
            continue
        try:
            params = fit_mandelbrot_mle(ranks[mask], resampled_freq[mask])
            q_samples.append(params.q)
            s_samples.append(params.s)
        except Exception:
            continue

    if len(q_samples) < 10:
        return {"error": "too few successful bootstrap iterations"}

    alpha = (1 - ci_level) / 2
    return {
        "q": {
            "mean": float(np.mean(q_samples)),
            "ci_lower": float(np.percentile(q_samples, 100 * alpha)),
            "ci_upper": float(np.percentile(q_samples, 100 * (1 - alpha))),
            "std": float(np.std(q_samples)),
        },
        "s": {
            "mean": float(np.mean(s_samples)),
            "ci_lower": float(np.percentile(s_samples, 100 * alpha)),
            "ci_upper": float(np.percentile(s_samples, 100 * (1 - alpha))),
            "std": float(np.std(s_samples)),
        },
        "n_successful": len(q_samples),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = ["gpt-5.1", "claude-sonnet-4-6", "llama-3.1-8b", "gemini-2.5-pro", "mistral-large", "qwen-2.5-7b"]
    domains = ["news", "biomedical", "legal", "code", "social_media"]

    all_results = {}

    for model_name in models:
        print(f"\n=== {model_name} ===")

        # Global fit
        global_path = RANK_TABLE_DIR / f"{model_name}_global.json"
        if global_path.exists():
            rt = RankTable.load(global_path)
            if rt.vocab_size == 0:
                print(f"  Skipping {model_name} (no data)")
                continue
            result = fit_and_evaluate(rt, f"{model_name}_global")
            all_results[f"{model_name}_global"] = result
        else:
            print(f"  Skipping {model_name} (no rank table)")
            continue

        # Per-domain fits
        for domain in domains:
            domain_path = RANK_TABLE_DIR / f"{model_name}_{domain}.json"
            if domain_path.exists():
                rt = RankTable.load(domain_path)
                if rt.vocab_size == 0:
                    continue
                result = fit_and_evaluate(rt, f"{model_name}_{domain}")
                all_results[f"{model_name}_{domain}"] = result

    # Save all results
    results_path = RESULTS_DIR / "mandelbrot_fit_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Label':<30} {'Method':<6} {'q':>8} {'s':>8} {'R2':>8} {'KS':>8} {'AIC_d':>10}")
    print("=" * 90)
    for label, result in all_results.items():
        mle = result["mle"]
        comp = result["model_comparison"]
        aic_delta = comp["zipf"]["aic"] - comp["mandelbrot"]["aic"]
        print(
            f"{label:<30} {'MLE':<6} "
            f"{mle['q']:>8.3f} {mle['s']:>8.3f} "
            f"{mle['gof']['r_squared']:>8.4f} "
            f"{mle['gof']['ks_statistic']:>8.4f} "
            f"{aic_delta:>10.1f}"
        )


if __name__ == "__main__":
    main()
