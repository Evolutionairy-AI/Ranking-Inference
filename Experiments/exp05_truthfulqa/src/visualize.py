"""TruthfulQA visualization.

Produces the key figures for taxonomy validation:
1. Tier gradient bar plot (THE primary figure)
2. Stratified ROC curves
3. Category breakdown heatmap
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import compute_roc_auc

# Consistent styling
COLORS = {
    "tier1": "#2196F3",
    "tier2": "#FF5722",
    "ambiguous": "#9E9E9E",
    "all": "#4CAF50",
}
MODEL_MARKERS = {"gpt-5.1": "o", "claude-sonnet-4": "s", "llama-3.1-8b": "^"}


def plot_tier_gradient(results: dict, output_dir: Path) -> Path:
    """THE key figure: bar plot of AUC for tier1 vs tier2, per model.

    Error bars from bootstrap CIs.  The gap between bars = taxonomy validation.

    Args:
        results: dict mapping model_name -> evaluate_truthfulqa() output
        output_dir: directory to save figures

    Returns:
        Path to saved figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(results.keys())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(3 + 2.5 * n_models, 5))

    bar_width = 0.25
    x = np.arange(n_models)

    for i, tier in enumerate(["tier1", "tier2", "all"]):
        aucs = []
        ci_lows = []
        ci_highs = []

        for model in models:
            r = results[model]
            auc_key = f"{tier}_auc" if tier != "all" else "all_auc"
            auc_val = r["stratified_auc"].get(auc_key, 0.5)
            aucs.append(auc_val)

            # Bootstrap CI
            ci = r.get("bootstrap_ci", {}).get(tier, (0.5, 0.5, 0.5))
            ci_lows.append(auc_val - ci[0])
            ci_highs.append(ci[2] - auc_val)

        bars = ax.bar(
            x + i * bar_width,
            aucs,
            bar_width,
            label=tier.replace("all", "Overall"),
            color=COLORS[tier],
            alpha=0.85,
            yerr=[ci_lows, ci_highs],
            capsize=4,
        )

    # Add tier gradient annotations
    for j, model in enumerate(models):
        r = results[model]
        gradient = r["stratified_auc"].get("tier_gradient", 0.0)
        max_auc = max(
            r["stratified_auc"].get("tier1_auc", 0.5),
            r["stratified_auc"].get("tier2_auc", 0.5),
        )
        ax.annotate(
            f"\u0394={gradient:.3f}",
            xy=(j + bar_width, max_auc + 0.03),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title(
        "Taxonomy Validation: Tier 1 vs Tier 2 AUC\n"
        "(Mandelbrot Ranking Distribution signal)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "tier_gradient.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Tier gradient figure saved to {fig_path}")
    return fig_path


def plot_stratified_roc(
    scored_questions: list,
    output_dir: Path,
    aggregation_strategy: str = "entity_weighted_mean",
) -> Path:
    """Separate ROC curves for tier1 vs tier2, overlaid.

    Args:
        scored_questions: list of TruthfulQAScoredQuestion
        output_dir: directory to save figures
        aggregation_strategy: which aggregation score to use

    Returns:
        Path to saved figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Extract candidate-level data by tier
    tier_data: dict[str, tuple[list, list]] = {
        "tier1": ([], []),
        "tier2": ([], []),
    }

    for sq in scored_questions:
        if sq.tier not in tier_data:
            continue
        correct_idx = sq.mc1_correct_idx
        for i, cand_score in enumerate(sq.mc1_candidate_scores):
            is_incorrect = 1 if i != correct_idx else 0
            tier_data[sq.tier][0].append(is_incorrect)
            tier_data[sq.tier][1].append(cand_score[aggregation_strategy])

    for tier_name in ("tier1", "tier2"):
        labels_list, scores_list = tier_data[tier_name]
        if len(labels_list) < 2 or len(set(labels_list)) < 2:
            continue
        labels_arr = np.array(labels_list)
        scores_arr = np.array(scores_list)

        fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
        auc_val = compute_roc_auc(labels_arr, scores_arr)

        ax.plot(
            fpr, tpr,
            color=COLORS[tier_name],
            linewidth=2,
            label=f"{tier_name} (AUC={auc_val:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "Stratified ROC: Tier 1 vs Tier 2\n"
        "(Mandelbrot Ranking Distribution hallucination detection)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    model_name = scored_questions[0].model_name if scored_questions else "unknown"
    fig_path = output_dir / f"stratified_roc_{model_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Stratified ROC figure saved to {fig_path}")
    return fig_path


def plot_category_breakdown(
    scored_questions: list,
    output_dir: Path,
    aggregation_strategy: str = "entity_weighted_mean",
) -> Path:
    """AUC across 38 categories, sorted.

    Args:
        scored_questions: list of TruthfulQAScoredQuestion
        output_dir: directory to save figures
        aggregation_strategy: which aggregation score to use

    Returns:
        Path to saved figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by category
    category_data: dict[str, tuple[list, list, list]] = {}
    for sq in scored_questions:
        cat = sq.category
        if cat not in category_data:
            category_data[cat] = ([], [], [])
        correct_idx = sq.mc1_correct_idx
        for i, cand_score in enumerate(sq.mc1_candidate_scores):
            is_incorrect = 1 if i != correct_idx else 0
            category_data[cat][0].append(is_incorrect)
            category_data[cat][1].append(cand_score[aggregation_strategy])
            category_data[cat][2].append(sq.tier)

    # Compute AUC per category
    cat_aucs = []
    cat_tiers = []
    for cat, (labels_list, scores_list, tiers_list) in category_data.items():
        labels_arr = np.array(labels_list)
        scores_arr = np.array(scores_list)
        if len(labels_arr) < 4 or len(np.unique(labels_arr)) < 2:
            continue
        auc_val = compute_roc_auc(labels_arr, scores_arr)
        # Determine dominant tier for this category
        tier_counts = {}
        for t in tiers_list:
            tier_counts[t] = tier_counts.get(t, 0) + 1
        dominant_tier = max(tier_counts, key=tier_counts.get)
        cat_aucs.append((cat, auc_val, dominant_tier))

    if not cat_aucs:
        print("No categories with enough data for AUC computation.")
        fig_path = output_dir / "category_breakdown.png"
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=14)
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        return fig_path

    # Sort by AUC descending
    cat_aucs.sort(key=lambda x: x[1], reverse=True)
    categories = [c[0] for c in cat_aucs]
    aucs = [c[1] for c in cat_aucs]
    dominant_tiers = [c[2] for c in cat_aucs]
    bar_colors = [COLORS.get(t, COLORS["ambiguous"]) for t in dominant_tiers]

    fig, ax = plt.subplots(figsize=(12, max(6, len(categories) * 0.35)))

    y_pos = np.arange(len(categories))
    ax.barh(y_pos, aucs, color=bar_colors, alpha=0.8, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=8)
    ax.set_xlabel("AUC-ROC", fontsize=12)
    ax.set_title(
        "RI Detection AUC by TruthfulQA Category\n"
        "(Blue = Tier 1 dominant, Orange = Tier 2 dominant)",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(0.0, 1.05)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    model_name = scored_questions[0].model_name if scored_questions else "unknown"
    fig_path = output_dir / f"category_breakdown_{model_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Category breakdown figure saved to {fig_path}")
    return fig_path
