"""
Step 3 (Exp03): Statistical analysis and visualization of the gap signal.

Tests:
1. Is the mean gap higher for hallucination-inducing outputs?
2. Are the gap distributions separable? (KS test, AUC-ROC)
3. How does beta affect the signal?
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = Path(__file__).resolve().parent.parent
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

CONDITION_COLORS = {
    "factual": "#059669",       # Green
    "hallucination": "#DC2626", # Red
    "synthetic_base": "#2563EB",# Blue
}

CONDITION_LABELS = {
    "factual": "Factual",
    "hallucination": "Hallucination-inducing",
    "synthetic_base": "Synthetic base",
}


def load_gap_results(beta: float) -> dict:
    path = RESULTS_DIR / f"gap_results_beta_{beta}.json"
    with open(path) as f:
        return json.load(f)


def statistical_tests(factual_gaps: np.ndarray, hallucination_gaps: np.ndarray) -> dict:
    """Run statistical tests comparing factual vs hallucination gap distributions."""
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(
        factual_gaps, hallucination_gaps, alternative="less"
    )

    # KS test
    ks_stat, ks_pvalue = stats.ks_2samp(factual_gaps, hallucination_gaps)

    # Cohen's d (effect size)
    pooled_std = np.sqrt(
        (np.std(factual_gaps) ** 2 + np.std(hallucination_gaps) ** 2) / 2
    )
    cohens_d = (np.mean(hallucination_gaps) - np.mean(factual_gaps)) / pooled_std if pooled_std > 0 else 0

    # AUC-ROC: can mean_gap discriminate factual from hallucinated?
    labels = np.concatenate([
        np.zeros(len(factual_gaps)),    # 0 = factual
        np.ones(len(hallucination_gaps))  # 1 = hallucination
    ])
    scores = np.concatenate([factual_gaps, hallucination_gaps])
    auroc = roc_auc_score(labels, scores)

    return {
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(u_pvalue),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "cohens_d": float(cohens_d),
        "auroc": float(auroc),
        "factual_mean": float(np.mean(factual_gaps)),
        "factual_std": float(np.std(factual_gaps)),
        "hallucination_mean": float(np.mean(hallucination_gaps)),
        "hallucination_std": float(np.std(hallucination_gaps)),
        "n_factual": len(factual_gaps),
        "n_hallucination": len(hallucination_gaps),
    }


def plot_gap_distributions(results: dict, beta: float):
    """Plot 1: Overlaid gap distributions for factual vs hallucination."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of mean_gap per output
    for condition in ["factual", "hallucination"]:
        if condition not in results:
            continue
        mean_gaps = [r["mean_gap"] for r in results[condition]]
        axes[0].hist(
            mean_gaps, bins=25, alpha=0.5, density=True,
            color=CONDITION_COLORS[condition],
            label=f"{CONDITION_LABELS[condition]} (n={len(mean_gaps)})",
            edgecolor="black", linewidth=0.5,
        )

    axes[0].set_xlabel("Mean gap delta(t) per output")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Distribution of Mean Gap (beta={beta})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    axes[0].axvline(0, color="black", linewidth=0.5, linestyle="--")

    # Right: histogram of max_gap per output
    for condition in ["factual", "hallucination"]:
        if condition not in results:
            continue
        max_gaps = [r["max_gap"] for r in results[condition]]
        axes[1].hist(
            max_gaps, bins=25, alpha=0.5, density=True,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
            edgecolor="black", linewidth=0.5,
        )

    axes[1].set_xlabel("Max gap delta(t) per output")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Distribution of Max Gap (beta={beta})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(
        "Confidence-Grounding Gap: Factual vs Hallucination-Inducing Outputs",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"gap_distributions_beta_{beta}.png")
    fig.savefig(RESULTS_DIR / f"gap_distributions_beta_{beta}.pdf")
    print(f"Saved: gap_distributions_beta_{beta}.png/pdf")
    plt.close()


def plot_roc_curve(results: dict, beta: float):
    """Plot 2: ROC curve for gap-based hallucination discrimination."""
    factual_gaps = np.array([r["mean_gap"] for r in results.get("factual", [])])
    halluc_gaps = np.array([r["mean_gap"] for r in results.get("hallucination", [])])

    if len(factual_gaps) == 0 or len(halluc_gaps) == 0:
        return

    labels = np.concatenate([np.zeros(len(factual_gaps)), np.ones(len(halluc_gaps))])
    scores = np.concatenate([factual_gaps, halluc_gaps])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#DC2626", linewidth=2, label=f"Mean gap (AUC={auroc:.3f})")

    # Also try max_gap
    factual_max = np.array([r["max_gap"] for r in results["factual"]])
    halluc_max = np.array([r["max_gap"] for r in results["hallucination"]])
    scores_max = np.concatenate([factual_max, halluc_max])
    fpr_max, tpr_max, _ = roc_curve(labels, scores_max)
    auroc_max = roc_auc_score(labels, scores_max)
    ax.plot(fpr_max, tpr_max, color="#2563EB", linewidth=2,
            label=f"Max gap (AUC={auroc_max:.3f})")

    # Try pct_positive_gap
    factual_pct = np.array([r["pct_positive_gap"] for r in results["factual"]])
    halluc_pct = np.array([r["pct_positive_gap"] for r in results["hallucination"]])
    scores_pct = np.concatenate([factual_pct, halluc_pct])
    fpr_pct, tpr_pct, _ = roc_curve(labels, scores_pct)
    auroc_pct = roc_auc_score(labels, scores_pct)
    ax.plot(fpr_pct, tpr_pct, color="#D97706", linewidth=2,
            label=f"% positive gap (AUC={auroc_pct:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.500)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC: Gap Signal for Hallucination Detection (beta={beta})", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"roc_curve_beta_{beta}.png")
    fig.savefig(RESULTS_DIR / f"roc_curve_beta_{beta}.pdf")
    print(f"Saved: roc_curve_beta_{beta}.png/pdf")
    plt.close()

    return {"mean_gap_auroc": auroc, "max_gap_auroc": auroc_max, "pct_pos_auroc": auroc_pct}


def plot_beta_sensitivity(beta_values: list):
    """Plot 3: How AUC-ROC varies with beta."""
    aurocs_mean = []
    aurocs_max = []
    aurocs_pct = []
    cohens_ds = []

    for beta in beta_values:
        results = load_gap_results(beta)
        factual_gaps = np.array([r["mean_gap"] for r in results.get("factual", [])])
        halluc_gaps = np.array([r["mean_gap"] for r in results.get("hallucination", [])])

        if len(factual_gaps) == 0 or len(halluc_gaps) == 0:
            continue

        labels = np.concatenate([np.zeros(len(factual_gaps)), np.ones(len(halluc_gaps))])

        # Mean gap AUROC
        scores = np.concatenate([factual_gaps, halluc_gaps])
        aurocs_mean.append(roc_auc_score(labels, scores))

        # Max gap AUROC
        scores_max = np.concatenate([
            [r["max_gap"] for r in results["factual"]],
            [r["max_gap"] for r in results["hallucination"]]
        ])
        aurocs_max.append(roc_auc_score(labels, scores_max))

        # Pct positive AUROC
        scores_pct = np.concatenate([
            [r["pct_positive_gap"] for r in results["factual"]],
            [r["pct_positive_gap"] for r in results["hallucination"]]
        ])
        aurocs_pct.append(roc_auc_score(labels, scores_pct))

        # Cohen's d
        pooled = np.sqrt((np.std(factual_gaps)**2 + np.std(halluc_gaps)**2) / 2)
        cohens_ds.append((np.mean(halluc_gaps) - np.mean(factual_gaps)) / pooled if pooled > 0 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AUC vs beta
    axes[0].plot(beta_values[:len(aurocs_mean)], aurocs_mean, "o-", color="#DC2626",
                 label="Mean gap", linewidth=2, markersize=8)
    axes[0].plot(beta_values[:len(aurocs_max)], aurocs_max, "s-", color="#2563EB",
                 label="Max gap", linewidth=2, markersize=8)
    axes[0].plot(beta_values[:len(aurocs_pct)], aurocs_pct, "^-", color="#D97706",
                 label="% positive gap", linewidth=2, markersize=8)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    axes[0].set_xlabel("Beta")
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("Discrimination Power vs Beta")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    axes[0].set_ylim(0.3, 1.0)

    # Cohen's d vs beta
    axes[1].plot(beta_values[:len(cohens_ds)], cohens_ds, "o-", color="#7C3AED",
                 linewidth=2, markersize=8)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].axhline(0.2, color="green", linestyle=":", alpha=0.3, label="Small effect")
    axes[1].axhline(0.5, color="orange", linestyle=":", alpha=0.3, label="Medium effect")
    axes[1].axhline(0.8, color="red", linestyle=":", alpha=0.3, label="Large effect")
    axes[1].set_xlabel("Beta")
    axes[1].set_ylabel("Cohen's d")
    axes[1].set_title("Effect Size vs Beta")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    fig.suptitle("Beta Sensitivity Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "beta_sensitivity.png")
    fig.savefig(RESULTS_DIR / "beta_sensitivity.pdf")
    print("Saved: beta_sensitivity.png/pdf")
    plt.close()


def main():
    beta_values = [0.5, 1.0, 1.5, 2.0]
    all_stats = {}

    for beta in beta_values:
        print(f"\n=== Beta = {beta} ===")
        results = load_gap_results(beta)

        factual_gaps = np.array([r["mean_gap"] for r in results.get("factual", [])])
        halluc_gaps = np.array([r["mean_gap"] for r in results.get("hallucination", [])])

        if len(factual_gaps) == 0 or len(halluc_gaps) == 0:
            print("  Insufficient data, skipping.")
            continue

        # Statistical tests
        test_results = statistical_tests(factual_gaps, halluc_gaps)
        all_stats[f"beta_{beta}"] = test_results

        print(f"  Factual:       mean={test_results['factual_mean']:.6f} +/- {test_results['factual_std']:.6f}")
        print(f"  Hallucination: mean={test_results['hallucination_mean']:.6f} +/- {test_results['hallucination_std']:.6f}")
        print(f"  Cohen's d:     {test_results['cohens_d']:.4f}")
        print(f"  AUC-ROC:       {test_results['auroc']:.4f}")
        print(f"  Mann-Whitney:  p={test_results['mann_whitney_p']:.6f}")
        print(f"  KS test:       stat={test_results['ks_statistic']:.4f}, p={test_results['ks_pvalue']:.6f}")

        # Plots
        plot_gap_distributions(results, beta)
        aurocs = plot_roc_curve(results, beta)
        if aurocs:
            all_stats[f"beta_{beta}"].update(aurocs)

    # Beta sensitivity plot
    plot_beta_sensitivity(beta_values)

    # Save all statistics
    stats_path = RESULTS_DIR / "statistical_analysis.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStatistical analysis saved to {stats_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Beta':<8} {'AUC-ROC':>10} {'Cohen_d':>10} {'MWU_p':>12} {'KS_stat':>10}")
    print("=" * 80)
    for key, s in sorted(all_stats.items()):
        print(f"{key:<8} {s['auroc']:>10.4f} {s['cohens_d']:>10.4f} "
              f"{s['mann_whitney_p']:>12.6f} {s['ks_statistic']:>10.4f}")


if __name__ == "__main__":
    main()
