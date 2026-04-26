"""
Step 4: Generate publication-quality plots for Experiment 01.

Produces:
1. Rank-frequency curves (log-log) with Mandelbrot overlay
2. Residual analysis by region (head/body/tail)
3. Model comparison (Zipf vs Mandelbrot)
4. Cross-model comparison (GPT vs Claude vs Llama)
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import RankTable, mandelbrot_freq

EXP_DIR = Path(__file__).resolve().parent.parent
RANK_TABLE_DIR = PROJECT_ROOT / "shared" / "rank_tables"
RESULTS_DIR = EXP_DIR / "results"

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

MODEL_COLORS = {
    "gpt-5.1": "#10A37F",             # OpenAI green
    "claude-sonnet-4-6": "#D97706",   # Anthropic amber
    "llama-3.1-8b": "#7C3AED",        # Meta purple
    "gemini-2.5-pro": "#4285F4",      # Google blue
    "mistral-large": "#FF7000",       # Mistral orange
    "qwen-2.5-7b": "#C0392B",         # Qwen red
}

MODEL_LABELS = {
    "gpt-5.1": "GPT-5.1",
    "claude-sonnet-4-6": "Claude 4.6 Sonnet",
    "llama-3.1-8b": "Llama 3.1 8B",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "mistral-large": "Mistral Large",
    "qwen-2.5-7b": "Qwen 2.5 7B",
}


def load_results():
    with open(RESULTS_DIR / "mandelbrot_fit_results.json") as f:
        return json.load(f)


def plot_rank_frequency_curves(results: dict):
    """Plot 1: Log-log rank-frequency curves with Mandelbrot theoretical overlay."""
    models = ["gpt-5.1", "claude-sonnet-4-6", "llama-3.1-8b", "gemini-2.5-pro", "mistral-large", "qwen-2.5-7b"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (ax, model_name) in enumerate(zip(axes, models)):
        key = f"{model_name}_global"
        if key not in results:
            ax.set_title(f"{MODEL_LABELS[model_name]} (no data)")
            continue

        result = results[key]
        rt_path = RANK_TABLE_DIR / f"{model_name}_global.json"
        rt = RankTable.load(rt_path)

        # Empirical data
        max_rank = rt.vocab_size
        ranks = np.arange(1, max_rank + 1)
        freq = rt.rank_to_freq[1:max_rank + 1].astype(float)
        mask = freq > 0
        ranks_nz = ranks[mask]
        freq_nz = freq[mask]

        # Plot empirical
        ax.scatter(
            ranks_nz, freq_nz,
            s=1, alpha=0.3, color=MODEL_COLORS[model_name],
            label="Observed", rasterized=True,
        )

        # Plot Mandelbrot fit
        mle = result["mle"]
        theoretical = mandelbrot_freq(ranks_nz, mle["C"], mle["q"], mle["s"])
        ax.plot(
            ranks_nz, theoretical,
            color="black", linewidth=1.5, linestyle="--",
            label=f'Mandelbrot (q={mle["q"]:.2f}, s={mle["s"]:.2f})',
        )

        # Plot pure Zipf for comparison
        comp = result["model_comparison"]
        s_zipf = comp["zipf"]["params"]["s"]
        C_zipf = freq_nz[0] * 1.0  # approximate normalization
        zipf_theoretical = C_zipf / ranks_nz ** s_zipf
        ax.plot(
            ranks_nz, zipf_theoretical,
            color="red", linewidth=1, linestyle=":",
            alpha=0.6,
            label=f"Zipf (s={s_zipf:.2f})",
        )

        # Region markers
        ax.axvline(10, color="gray", alpha=0.3, linestyle="-")
        ax.axvline(50000, color="gray", alpha=0.3, linestyle="-")
        ax.text(5, ax.get_ylim()[1] * 0.5, "Head", fontsize=8, alpha=0.5, ha="center")

        ax.set_xscale("log")
        ax.set_yscale("log")
        if idx >= 3:
            ax.set_xlabel("Rank")
        if idx % 3 == 0:
            ax.set_ylabel("Frequency")
        ax.set_title(f"{MODEL_LABELS[model_name]}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

        # Annotate GoF
        r2 = mle["gof"]["r_squared"]
        ks = mle["gof"]["ks_statistic"]
        ax.text(
            0.03, 0.03,
            f"R²={r2:.4f}\nKS={ks:.4f}",
            transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.suptitle(
        "Rank-Frequency Distribution: LLM Outputs vs Mandelbrot Theoretical Curve",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "rank_frequency_curves.png")
    fig.savefig(RESULTS_DIR / "rank_frequency_curves.pdf")
    fig.savefig(RESULTS_DIR / "rank_frequency_curves.eps", format="eps")
    print("Saved: rank_frequency_curves.png/pdf")
    plt.close()


def plot_residuals_by_region(results: dict):
    """Plot 2: Residual analysis by region (head/body/tail).

    Tests Section 9.5 prediction: head and tail deviate, body fits well.
    """
    models = ["gpt-5.1", "claude-sonnet-4-6", "llama-3.1-8b", "gemini-2.5-pro", "mistral-large", "qwen-2.5-7b"]
    regions = ["head", "body", "tail"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, model_name in zip(axes, models):
        key = f"{model_name}_global"
        if key not in results:
            continue

        gof = results[key]["mle"]["gof"]
        residuals = gof["residuals_by_region"]

        means = []
        stds = []
        labels = []

        for region in regions:
            if region in residuals:
                means.append(residuals[region]["mean_residual"])
                stds.append(residuals[region]["std_residual"])
                labels.append(f"{region}\n(n={residuals[region]['n_tokens']})")
            else:
                means.append(0)
                stds.append(0)
                labels.append(f"{region}\n(n=0)")

        x = np.arange(len(regions))
        bars = ax.bar(
            x, means, yerr=stds,
            color=[MODEL_COLORS[model_name]] * 3,
            alpha=0.7, capsize=5, edgecolor="black", linewidth=0.5,
        )

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(MODEL_LABELS[model_name])
        if ax == axes[0]:
            ax.set_ylabel("Mean Log Residual (obs - predicted)")
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Mandelbrot Fit Residuals by Rank Region\n"
        "(Section 9.5 predicts head & tail deviations, good body fit)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "residuals_by_region.png")
    fig.savefig(RESULTS_DIR / "residuals_by_region.pdf")
    fig.savefig(RESULTS_DIR / "residuals_by_region.eps", format="eps")
    print("Saved: residuals_by_region.png/pdf")
    plt.close()


def plot_model_comparison_aic(results: dict):
    """Plot 3: AIC comparison Mandelbrot vs Zipf across all fits."""
    labels = []
    aic_deltas = []  # Zipf AIC - Mandelbrot AIC (positive = Mandelbrot wins)
    colors = []

    for key, result in sorted(results.items()):
        comp = result["model_comparison"]
        delta = comp["zipf"]["aic"] - comp["mandelbrot"]["aic"]
        labels.append(key.replace("_", "\n"))
        aic_deltas.append(delta)

        for model_name, color in MODEL_COLORS.items():
            if model_name in key:
                colors.append(color)
                break
        else:
            colors.append("gray")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    ax.barh(x, aic_deltas, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AIC(Zipf) - AIC(Mandelbrot)\n(positive = Mandelbrot is better fit)")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Model Selection: Mandelbrot vs Pure Zipf", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison_aic.png")
    fig.savefig(RESULTS_DIR / "model_comparison_aic.pdf")
    fig.savefig(RESULTS_DIR / "model_comparison_aic.eps", format="eps")
    print("Saved: model_comparison_aic.png/pdf")
    plt.close()


def plot_cross_model_overlay(results: dict):
    """Plot 4: All 3 models on one log-log plot for direct comparison."""
    models = ["gpt-5.1", "claude-sonnet-4-6", "llama-3.1-8b", "gemini-2.5-pro", "mistral-large", "qwen-2.5-7b"]

    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name in models:
        key = f"{model_name}_global"
        if key not in results:
            continue

        rt_path = RANK_TABLE_DIR / f"{model_name}_global.json"
        if not rt_path.exists():
            continue
        rt = RankTable.load(rt_path)

        max_rank = rt.vocab_size
        ranks = np.arange(1, max_rank + 1)
        freq = rt.rank_to_freq[1:max_rank + 1].astype(float)
        mask = freq > 0

        # Normalize to relative frequency for cross-model comparison
        rel_freq = freq[mask] / freq[mask].sum()

        ax.plot(
            ranks[mask], rel_freq,
            linewidth=1.2, alpha=0.7,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )

    # Add theoretical Mandelbrot (use average params)
    all_q = []
    all_s = []
    for model_name in models:
        key = f"{model_name}_global"
        if key in results:
            all_q.append(results[key]["mle"]["q"])
            all_s.append(results[key]["mle"]["s"])

    if all_q:
        avg_q = np.mean(all_q)
        avg_s = np.mean(all_s)
        r_range = np.logspace(0, 5, 1000)
        theoretical = 1.0 / (r_range + avg_q) ** avg_s
        theoretical /= theoretical.sum()
        ax.plot(
            r_range, theoretical,
            "k--", linewidth=2, alpha=0.8,
            label=f"Mandelbrot (avg q={avg_q:.2f}, s={avg_s:.2f})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative Frequency")
    ax.set_title("Cross-Model Rank-Frequency Comparison", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "cross_model_overlay.png")
    fig.savefig(RESULTS_DIR / "cross_model_overlay.pdf")
    fig.savefig(RESULTS_DIR / "cross_model_overlay.eps", format="eps")
    print("Saved: cross_model_overlay.png/pdf")
    plt.close()


def plot_domain_parameter_variation(results: dict):
    """Plot 5: How Mandelbrot parameters (q, s) vary across domains.

    This bridges to Experiment 02 by showing domain-dependent structure.
    """
    models = ["gpt-5.1", "claude-sonnet-4-6", "llama-3.1-8b", "gemini-2.5-pro", "mistral-large", "qwen-2.5-7b"]
    domains = ["news", "biomedical", "legal", "code", "social_media"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name in models:
        qs = []
        ss = []
        valid_domains = []
        for domain in domains:
            key = f"{model_name}_{domain}"
            if key in results:
                qs.append(results[key]["mle"]["q"])
                ss.append(results[key]["mle"]["s"])
                valid_domains.append(domain)

        if valid_domains:
            x = np.arange(len(valid_domains))
            axes[0].plot(x, qs, "o-", color=MODEL_COLORS[model_name],
                         label=MODEL_LABELS[model_name], markersize=6)
            axes[1].plot(x, ss, "o-", color=MODEL_COLORS[model_name],
                         label=MODEL_LABELS[model_name], markersize=6)

    for ax, param_name in zip(axes, ["q (shift)", "s (exponent)"]):
        ax.set_xticks(np.arange(len(domains)))
        ax.set_xticklabels([d.replace("_", "\n") for d in domains], fontsize=9)
        ax.set_ylabel(param_name)
        ax.set_title(f"Mandelbrot {param_name} by Domain")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Domain Variation of Mandelbrot Parameters", fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "domain_parameter_variation.png")
    fig.savefig(RESULTS_DIR / "domain_parameter_variation.pdf")
    fig.savefig(RESULTS_DIR / "domain_parameter_variation.eps", format="eps")
    print("Saved: domain_parameter_variation.png/pdf")
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    plot_rank_frequency_curves(results)
    plot_residuals_by_region(results)
    plot_model_comparison_aic(results)
    plot_cross_model_overlay(results)
    plot_domain_parameter_variation(results)

    print(f"\nAll plots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
