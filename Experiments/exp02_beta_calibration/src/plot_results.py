"""
Step 2 (Exp02): Generate plots for beta calibration analysis.

Produces:
1. Beta by domain (bar chart with CIs and predicted ranges)
2. Rank deviation distributions (overlaid histograms)
3. Beta vs predicted range (validation plot)
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# Section 6.3 predicted beta ranges
PREDICTED_RANGES = {
    "social_media": (0.3, 0.7),
    "code": (0.5, 2.0),
    "news": (0.7, 1.5),
    "legal": (1.5, 2.5),
    "biomedical": (1.5, 3.0),
}

DOMAIN_COLORS = {
    "news": "#2563EB",
    "biomedical": "#DC2626",
    "legal": "#7C3AED",
    "code": "#059669",
    "social_media": "#D97706",
}

DOMAIN_ORDER = ["social_media", "code", "news", "legal", "biomedical"]


def load_results():
    with open(DATA_DIR / "rank_deviation_results.json") as f:
        return json.load(f)


def plot_beta_by_domain(results: dict):
    """Plot 1: Beta values by domain with predicted ranges."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(DOMAIN_ORDER))
    betas = []
    colors = []

    for domain in DOMAIN_ORDER:
        # Find result for this domain (any tokenizer)
        beta = None
        for key, result in results.items():
            if result["domain"] == domain:
                beta = result["beta"]
                break
        betas.append(beta or 0)
        colors.append(DOMAIN_COLORS[domain])

    # Plot empirical betas
    bars = ax.bar(x_pos, betas, color=colors, alpha=0.7, edgecolor="black",
                  linewidth=0.8, width=0.6, label="Empirical beta")

    # Overlay predicted ranges as shaded bands
    for i, domain in enumerate(DOMAIN_ORDER):
        low, high = PREDICTED_RANGES[domain]
        ax.fill_between(
            [i - 0.35, i + 0.35], low, high,
            color=DOMAIN_COLORS[domain], alpha=0.15,
            linewidth=0,
        )
        ax.plot([i - 0.35, i + 0.35], [low, low], color="black", linewidth=0.5, alpha=0.5)
        ax.plot([i - 0.35, i + 0.35], [high, high], color="black", linewidth=0.5, alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([d.replace("_", "\n") for d in DOMAIN_ORDER])
    ax.set_ylabel(r"$\beta = 1/\sigma^2_{\Delta r}$")
    ax.set_title(
        r"Empirical $\beta$ by Domain vs Section 6.3 Predictions"
        "\n(shaded regions = predicted ranges)",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2, axis="y")

    # Add legend for predicted range
    predicted_patch = mpatches.Patch(
        facecolor="gray", alpha=0.15, edgecolor="black", linewidth=0.5,
        label="Section 6.3 predicted range",
    )
    ax.legend(handles=[bars.patches[0], predicted_patch],
              labels=["Empirical beta", "Section 6.3 predicted range"])

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "beta_by_domain.png")
    fig.savefig(RESULTS_DIR / "beta_by_domain.pdf")
    print("Saved: beta_by_domain.png/pdf")
    plt.close()


def plot_deviation_distributions(results: dict):
    """Plot 2: Overlaid rank deviation histograms per domain."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for domain in DOMAIN_ORDER:
        for key, result in results.items():
            if result["domain"] == domain:
                hist = result["deviations_histogram"]
                bin_edges = np.array(hist["bin_edges"])
                counts = np.array(hist["counts"])
                # Normalize to density
                widths = np.diff(bin_edges)
                density = counts / (counts.sum() * widths)
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                ax.plot(
                    centers, density,
                    color=DOMAIN_COLORS[domain],
                    linewidth=1.5, alpha=0.8,
                    label=f"{domain.replace('_', ' ')} "
                          f"(sigma^2={result['sigma2_delta_r']:.2f})",
                )
                ax.fill_between(
                    centers, density,
                    color=DOMAIN_COLORS[domain], alpha=0.1,
                )
                break

    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$\Delta r = \log_2(r_{global} / r_{local})$ (bits)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Rank Deviation Distributions by Domain\n"
        r"(wider distributions $\Rightarrow$ lower $\beta$, weaker prior)",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "deviation_distributions.png")
    fig.savefig(RESULTS_DIR / "deviation_distributions.pdf")
    print("Saved: deviation_distributions.png/pdf")
    plt.close()


def plot_beta_validation(results: dict):
    """Plot 3: Empirical beta vs predicted range (scatter with error bars)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for domain in DOMAIN_ORDER:
        for key, result in results.items():
            if result["domain"] == domain:
                emp_beta = result["beta"]
                pred_low, pred_high = PREDICTED_RANGES[domain]
                pred_mid = (pred_low + pred_high) / 2

                ax.scatter(
                    pred_mid, emp_beta,
                    color=DOMAIN_COLORS[domain], s=120,
                    edgecolor="black", linewidth=0.8, zorder=5,
                )
                # Horizontal error bar for predicted range
                ax.plot(
                    [pred_low, pred_high], [emp_beta, emp_beta],
                    color=DOMAIN_COLORS[domain], linewidth=2, alpha=0.5,
                )
                ax.annotate(
                    domain.replace("_", " "),
                    (pred_mid, emp_beta),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=9,
                )
                break

    # Diagonal reference line
    lims = [0, max(4, ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y = x (perfect calibration)")

    ax.set_xlabel(r"Predicted $\beta$ (Section 6.3 midpoint)")
    ax.set_ylabel(r"Empirical $\beta = 1/\sigma^2_{\Delta r}$")
    ax.set_title(
        r"$\beta$ Calibration: Empirical vs Theoretical Prediction",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "beta_validation.png")
    fig.savefig(RESULTS_DIR / "beta_validation.pdf")
    print("Saved: beta_validation.png/pdf")
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    plot_beta_by_domain(results)
    plot_deviation_distributions(results)
    plot_beta_validation(results)

    print(f"\nAll plots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
