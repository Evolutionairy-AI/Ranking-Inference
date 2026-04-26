"""Visualizations for conviction analysis and ROUGE comparison.

Generates figures for:
1. Reliability diagram (conviction vs accuracy)
2. Cost-savings / triage curve
3. ROUGE vs RI correlation scatter
4. Pareto frontier: cost vs accuracy
5. Cascade pipeline simulation
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent.parent / "results"
CONV_FILE = OUT_DIR / "conviction_analysis.json"
ROUGE_FILE = OUT_DIR / "rouge_comparison.json"
FRANK_SCORED = ROOT / "exp06_frank" / "output" / "scored_frank_llama-3.1-8b.jsonl"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def load_data():
    with open(CONV_FILE) as f:
        conv = json.load(f)
    with open(ROUGE_FILE) as f:
        rouge = json.load(f)
    return conv, rouge


def load_frank_spans():
    data = []
    with open(FRANK_SCORED) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def fig1_reliability_diagram(conv_data):
    """Reliability diagram: conviction bins vs accuracy for all 4 scores."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Reliability Diagrams: Conviction vs Accuracy (FRANK)", fontsize=14)

    titles = {
        "mean_delta_global": "Linear Global Prior",
        "mean_delta_source": "Linear Source Prior",
        "mean_log_delta_global": "Log-space Global Prior",
        "mean_log_delta_source": "Log-space Source Prior",
    }

    for ax, (name, title) in zip(axes.flat, titles.items()):
        bins = conv_data["frank"][name]["conviction"]["bins"]
        # Filter bins with enough samples
        bins = [b for b in bins if b["n_samples"] >= 5]

        if not bins:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_title(title)
            continue

        x = [b["conviction_mean"] for b in bins]
        y = [b["accuracy"] for b in bins]
        sizes = [max(20, min(300, b["n_samples"] / 10)) for b in bins]

        ax.scatter(x, y, s=sizes, alpha=0.7, c="steelblue", edgecolors="navy", zorder=3)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")

        overall_acc = conv_data["frank"][name]["conviction"]["overall_accuracy"]
        ax.axhline(y=overall_acc, color="red", linestyle=":", alpha=0.7,
                   label=f"Overall: {overall_acc:.1%}")

        # Annotate sample counts
        for xi, yi, b in zip(x, y, bins):
            ax.annotate(f"n={b['n_samples']}", (xi, yi),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=7, alpha=0.7)

        ece = conv_data["frank"][name]["ece"]["ece"]
        j = conv_data["frank"][name]["conviction"]["best_youdens_j"]
        ax.set_title(f"{title}\nECE={ece:.4f}, J={j:.4f}")
        ax.set_xlabel("Conviction (|score - threshold|)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_reliability_diagrams.png", bbox_inches="tight")
    plt.close()
    print("Saved fig1_reliability_diagrams.png")


def fig2_cost_savings(conv_data):
    """Triage curves: % resolved vs accuracy at threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Triage Performance: Cost Savings vs Accuracy", fontsize=14)

    for ax, space, names in [
        (axes[0], "Linear", ["mean_delta_global", "mean_delta_source"]),
        (axes[1], "Log-space", ["mean_log_delta_global", "mean_log_delta_source"]),
    ]:
        for name, color, label in zip(names,
                                       ["steelblue", "coral"],
                                       ["Global Prior", "Source Prior"]):
            cs = conv_data["frank"][name]["cost_savings"]
            if not cs:
                continue
            pct = [c["pct_resolved"] for c in cs]
            acc = [c["accuracy_on_resolved"] for c in cs]
            ax.plot(pct, acc, "-o", markersize=2, color=color, label=label, alpha=0.8)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("% of outputs resolved by RI (remainder deferred)")
        ax.set_ylabel("Accuracy on resolved outputs")
        ax.set_title(f"{space} Scores")
        ax.set_xlim(0, 105)
        ax.set_ylim(0.4, 0.75)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_cost_savings.png", bbox_inches="tight")
    plt.close()
    print("Saved fig2_cost_savings.png")


def fig3_rouge_vs_ri(rouge_data):
    """Scatter: ROUGE-2 vs RI scores at article level."""
    articles = rouge_data["per_article"]

    rouge2 = [a["rouge2_f1"] for a in articles]
    ri_source_log = [a["mean_log_ri_source"] for a in articles]
    ri_global_log = [a["mean_log_ri_global"] for a in articles]
    error_rate = [a["error_rate"] for a in articles]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("ROUGE-2 vs RI Scores (FRANK, Article Level)", fontsize=14)

    # Panel 1: ROUGE-2 vs RI source (log)
    ax = axes[0]
    sc = ax.scatter(rouge2, ri_source_log, c=error_rate, cmap="RdYlGn_r",
                    alpha=0.6, s=20, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Error Rate")
    corr = rouge_data["correlations"]["ROUGE-2_vs_RI_source_log"]
    ax.set_xlabel("ROUGE-2 F1")
    ax.set_ylabel("RI Source (log-space)")
    ax.set_title(f"Source RI vs ROUGE-2\nr={corr['pearson_r']:.3f}, p={corr['pearson_p']:.1e}")
    ax.grid(True, alpha=0.3)

    # Panel 2: ROUGE-2 vs RI global (log)
    ax = axes[1]
    sc = ax.scatter(rouge2, ri_global_log, c=error_rate, cmap="RdYlGn_r",
                    alpha=0.6, s=20, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Error Rate")
    corr = rouge_data["correlations"]["ROUGE-2_vs_RI_global_log"]
    ax.set_xlabel("ROUGE-2 F1")
    ax.set_ylabel("RI Global (log-space)")
    ax.set_title(f"Global RI vs ROUGE-2\nr={corr['pearson_r']:.3f}, p={corr['pearson_p']:.1e}")
    ax.grid(True, alpha=0.3)

    # Panel 3: Both RI vs error rate
    ax = axes[2]
    ax.scatter(ri_source_log, error_rate, alpha=0.4, s=15, label="Source RI", c="coral")
    ax.scatter(ri_global_log, error_rate, alpha=0.4, s=15, label="Global RI", c="steelblue")
    corr_src = rouge_data["correlations"]["RI_source_log_vs_error_rate"]
    corr_glb = rouge_data["correlations"]["RI_global_log_vs_error_rate"]
    ax.set_xlabel("RI Score (log-space)")
    ax.set_ylabel("Error Rate")
    ax.set_title(f"RI vs Error Rate\nSource r={corr_src['pearson_r']:.3f}, Global r={corr_glb['pearson_r']:.3f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_rouge_vs_ri.png", bbox_inches="tight")
    plt.close()
    print("Saved fig3_rouge_vs_ri.png")


def fig4_pareto_frontier():
    """Pareto frontier: latency vs AUC-ROC across methods."""
    # Published baselines + our results
    methods = [
        # (name, latency_ms, auc_roc, color, marker)
        ("RI gap-only", 0.139, 0.516, "forestgreen", "D"),
        ("RI source-log", 0.139, 0.585, "limegreen", "D"),
        ("RI + NER", 10.5, 0.516, "green", "s"),
        ("ROUGE-2 (article-level)", 0.5, 0.845, "goldenrod", "^"),
        ("ROUGE-1 (article-level)", 0.5, 0.718, "orange", "^"),
        ("SelfCheckGPT (N=5)", 20603, 0.75, "firebrick", "o"),
        ("Semantic Entropy (k=5)", 21003, 0.75, "darkred", "o"),
        ("FActScore", 5000, 0.85, "purple", "p"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, lat, auc, color, marker in methods:
        ax.scatter(lat, auc, s=120, c=color, marker=marker, zorder=3,
                  edgecolors="black", linewidths=0.5)
        # Offset labels to avoid overlap
        offset = (8, 5) if lat < 100 else (-10, 8)
        ax.annotate(name, (lat, auc), textcoords="offset points",
                   xytext=offset, fontsize=8, alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Pareto Frontier: Latency vs Detection Quality\n"
                 "RI provides fast triage; ROUGE/specialist methods provide accuracy")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.05, 50000)
    ax.set_ylim(0.4, 0.95)

    # Shade the triage zone
    ax.axvspan(0.05, 15, alpha=0.08, color="green", label="RI triage zone (<15ms)")
    ax.axvspan(15, 50000, alpha=0.05, color="red", label="Expensive verification")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_pareto_frontier.png", bbox_inches="tight")
    plt.close()
    print("Saved fig4_pareto_frontier.png")


def fig5_cascade_simulation(conv_data):
    """Simulate a cascade: RI handles high-conviction cases,
    expensive method handles the rest. Show overall accuracy + cost."""

    # Use log-space source as best RI variant
    cs = conv_data["frank"]["mean_log_delta_source"]["cost_savings"]
    if not cs:
        print("No cost savings data for cascade simulation")
        return

    # Assume expensive method has AUC=0.85 (FActScore-like)
    expensive_acc = 0.85

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cascade Pipeline: RI Triage + Expensive Verification", fontsize=14)

    pct_ri = []
    cascade_acc = []
    cascade_cost_pct = []

    for c in cs:
        ri_pct = c["pct_resolved"] / 100
        ri_acc = c["accuracy_on_resolved"]
        deferred_pct = 1 - ri_pct

        # Cascade accuracy = weighted average
        total_acc = ri_pct * ri_acc + deferred_pct * expensive_acc
        # Cost = RI is ~0.001% of expensive method cost, so cost ≈ deferred_pct
        cost_pct = deferred_pct * 100  # % of expensive calls needed

        pct_ri.append(c["pct_resolved"])
        cascade_acc.append(total_acc)
        cascade_cost_pct.append(cost_pct)

    # Panel 1: Cascade accuracy vs % handled by RI
    ax1.plot(pct_ri, cascade_acc, "-", color="steelblue", linewidth=2)
    ax1.axhline(y=expensive_acc, color="red", linestyle="--",
               label=f"Expensive-only: {expensive_acc:.0%}", alpha=0.7)
    ax1.set_xlabel("% of outputs triaged by RI")
    ax1.set_ylabel("Cascade accuracy")
    ax1.set_title("Overall Accuracy with Cascade")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 0.90)

    # Panel 2: Cost reduction
    ax2.plot(pct_ri, cascade_cost_pct, "-", color="coral", linewidth=2)
    ax2.set_xlabel("% of outputs triaged by RI")
    ax2.set_ylabel("% of expensive method calls needed")
    ax2.set_title("Cost Reduction")
    ax2.grid(True, alpha=0.3)

    # Annotate a sweet spot
    # Find point where cascade_acc > 0.80 with minimum cost
    for i, (pct, acc, cost) in enumerate(zip(pct_ri, cascade_acc, cascade_cost_pct)):
        if acc >= 0.78 and pct >= 20:
            ax1.annotate(f"{pct:.0f}% by RI\nacc={acc:.1%}",
                        (pct, acc), textcoords="offset points",
                        xytext=(10, -15), fontsize=8,
                        arrowprops=dict(arrowstyle="->", alpha=0.5))
            ax2.annotate(f"Save {100-cost:.0f}% cost",
                        (pct, cost), textcoords="offset points",
                        xytext=(10, 10), fontsize=8,
                        arrowprops=dict(arrowstyle="->", alpha=0.5))
            break

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_cascade_pipeline.png", bbox_inches="tight")
    plt.close()
    print("Saved fig5_cascade_pipeline.png")


def fig6_auc_comparison(rouge_data):
    """Bar chart: AUC-ROC for high-error article detection across methods."""
    auc = rouge_data["auc_comparison"]

    names = list(auc.keys())
    values = [auc[n] for n in names]

    # Clean up names
    clean_names = [n.replace("_", " ").replace("F1", "(F1)") for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["goldenrod" if "ROUGE" in n else "steelblue" if "global" in n else "coral"
              for n in names]

    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(clean_names, fontsize=9)
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Article-Level Error Detection: ROUGE vs RI\n"
                 "(FRANK dataset, binary: error_rate > median)")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlim(0.45, 0.92)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig6_auc_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved fig6_auc_comparison.png")


def main():
    conv_data, rouge_data = load_data()

    fig1_reliability_diagram(conv_data)
    fig2_cost_savings(conv_data)
    fig3_rouge_vs_ri(rouge_data)
    fig4_pareto_frontier()
    fig5_cascade_simulation(conv_data)
    fig6_auc_comparison(rouge_data)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
