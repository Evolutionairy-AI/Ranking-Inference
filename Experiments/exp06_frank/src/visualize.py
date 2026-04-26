"""FRANK visualization: error-type gradient, dual baseline comparison, distributions."""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp06_frank.src.load_dataset import TIER_SIGNAL_ORDER, ERROR_TYPE_TO_TIER

# Colour palette: Tier 1 = red tones, Tier 1.5 = orange, Tier 2 = blue tones
TIER_COLOURS = {
    "tier1": "#d62728",
    "tier1.5": "#ff7f0e",
    "tier2": "#1f77b4",
    "control": "#7f7f7f",
}

ERROR_TYPE_COLOURS = {
    "OutE": "#d62728",
    "EntE": "#e45756",
    "CircE": "#ff7f0e",
    "PredE": "#4c78a8",
    "LinkE": "#1f77b4",
    "CorefE": "#72b7d6",
}

# Canonical ordering (strongest predicted signal first)
ORDERED_TYPES = sorted(TIER_SIGNAL_ORDER.keys(), key=lambda k: TIER_SIGNAL_ORDER[k], reverse=True)


def plot_error_type_gradient(results: dict, output_dir: Path):
    """THE key figure: F1 (y) vs FRANK error types ordered by predicted RI signal (x).

    Bar plot with error types sorted from Tier 1 (left) to Tier 2 (right).
    Two bar series: global baseline and source-article baseline.
    Overlay predicted performance (dotted line connecting expected values).

    Args:
        results: output from evaluate_frank()
        output_dir: directory to save the figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    global_results = results.get("global_baseline", {})
    source_results = results.get("source_baseline", {})

    # Only include types present in results
    types_present = [et for et in ORDERED_TYPES if et in global_results or et in source_results]
    if not types_present:
        print("No error types to plot.")
        return

    n = len(types_present)
    x = np.arange(n)
    width = 0.35

    # Extract F1 values
    f1_global = [global_results.get(et, {}).get("f1", 0.0) for et in types_present]
    f1_source = [source_results.get(et, {}).get("f1", 0.0) for et in types_present]

    # Predicted trend line (normalised signal order)
    max_signal = max(TIER_SIGNAL_ORDER[et] for et in types_present)
    predicted_f1 = [TIER_SIGNAL_ORDER[et] / max_signal * max(max(f1_global), max(f1_source), 0.5)
                    for et in types_present]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar colours by tier
    bar_colours = [ERROR_TYPE_COLOURS.get(et, "#999999") for et in types_present]

    bars_g = ax.bar(x - width/2, f1_global, width, label="Global (Wikipedia)",
                    color=bar_colours, alpha=0.85, edgecolor="black", linewidth=0.5)
    bars_s = ax.bar(x + width/2, f1_source, width, label="Source-article",
                    color=bar_colours, alpha=0.45, edgecolor="black", linewidth=0.5,
                    hatch="//")

    # Predicted trend overlay
    ax.plot(x, predicted_f1, "k--", marker="D", markersize=5, linewidth=1.5,
            label="Predicted gradient", alpha=0.6)

    # Tier labels on top
    tier_labels = [ERROR_TYPE_TO_TIER.get(et, "") for et in types_present]
    for i, (et, tier) in enumerate(zip(types_present, tier_labels)):
        ax.annotate(tier, (i, max(f1_global[i], f1_source[i]) + 0.02),
                    ha="center", fontsize=8, color="gray")

    ax.set_xlabel("FRANK Error Type (ordered by predicted RI signal strength)", fontsize=11)
    ax.set_ylabel("Span-Level F1", fontsize=11)
    ax.set_title(f"Error-Type Gradient: F1 vs RI Taxonomy — {results.get('model_name', 'unknown')}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(types_present, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Add Spearman rho annotation
    gradient = results.get("global_gradient", {})
    rho = gradient.get("spearman_rho", float("nan"))
    p_val = gradient.get("p_value", float("nan"))
    if not np.isnan(rho):
        ax.text(0.02, 0.95, f"Spearman rho = {rho:.3f} (p = {p_val:.3f})",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig_path = output_dir / f"error_type_gradient_{results.get('model_name', 'model')}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved gradient plot to {fig_path}")


def plot_dual_baseline_comparison(results: dict, output_dir: Path):
    """Side-by-side comparison: global vs source-article baseline metrics per error type.

    Shows AUC and Cohen's d for both baselines across error types.

    Args:
        results: output from evaluate_frank()
        output_dir: directory to save the figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    global_results = results.get("global_baseline", {})
    source_results = results.get("source_baseline", {})

    types_present = [et for et in ORDERED_TYPES if et in global_results or et in source_results]
    if not types_present:
        return

    n = len(types_present)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: AUC comparison
    auc_global = [global_results.get(et, {}).get("auc", 0.5) for et in types_present]
    auc_source = [source_results.get(et, {}).get("auc", 0.5) for et in types_present]

    axes[0].bar(x - width/2, auc_global, width, label="Global", color="#2ca02c", alpha=0.7)
    axes[0].bar(x + width/2, auc_source, width, label="Source", color="#9467bd", alpha=0.7)
    axes[0].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("AUC by Error Type and Baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(types_present)
    axes[0].set_ylim(0.3, 1.0)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Cohen's d comparison
    d_global = [global_results.get(et, {}).get("cohens_d", 0.0) for et in types_present]
    d_source = [source_results.get(et, {}).get("cohens_d", 0.0) for et in types_present]

    axes[1].bar(x - width/2, d_global, width, label="Global", color="#2ca02c", alpha=0.7)
    axes[1].bar(x + width/2, d_source, width, label="Source", color="#9467bd", alpha=0.7)
    axes[1].axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_ylabel("Cohen's d")
    axes[1].set_title("Effect Size by Error Type and Baseline")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(types_present)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Dual Baseline Comparison — {results.get('model_name', 'unknown')}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = output_dir / f"dual_baseline_{results.get('model_name', 'model')}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dual baseline plot to {fig_path}")


def plot_span_delta_distributions(scored_spans: list, output_dir: Path):
    """Distribution of span deltas for error vs control, faceted by error type.

    Each facet shows the histogram of mean_delta for error spans of one type
    overlaid with the control span distribution.

    Args:
        scored_spans: list of FRANKScoredSpan or dicts
        output_dir: directory to save the figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group deltas by error type
    error_deltas = defaultdict(list)
    control_deltas = []

    for s in scored_spans:
        if isinstance(s, dict):
            is_err = s["is_error"]
            etype = s["error_type"]
            delta = s["mean_delta"]
        else:
            is_err = s.is_error
            etype = s.error_type
            delta = s.mean_delta

        if is_err:
            error_deltas[etype].append(delta)
        else:
            control_deltas.append(delta)

    types_present = [et for et in ORDERED_TYPES if et in error_deltas]
    if not types_present:
        return

    n_types = len(types_present)
    n_cols = min(3, n_types)
    n_rows = (n_types + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_types == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    bins = np.linspace(-0.5, 1.0, 40)

    for i, etype in enumerate(types_present):
        ax = axes[i]
        colour = ERROR_TYPE_COLOURS.get(etype, "#999999")

        ax.hist(control_deltas, bins=bins, alpha=0.4, color="gray",
                density=True, label="Control")
        ax.hist(error_deltas[etype], bins=bins, alpha=0.6, color=colour,
                density=True, label=f"{etype} errors")

        ax.set_title(f"{etype} ({ERROR_TYPE_TO_TIER.get(etype, '?')})", fontsize=11)
        ax.set_xlabel("Mean span delta")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for j in range(n_types, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Span Delta Distributions: Error vs Control by Type",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    model_name = ""
    if scored_spans:
        s = scored_spans[0]
        model_name = s.get("model_name", "") if isinstance(s, dict) else getattr(s, "model_name", "")

    fig_path = output_dir / f"delta_distributions_{model_name or 'model'}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved delta distribution plot to {fig_path}")
