"""HaluEval visualization.

Generates ROC curves, entity delta distribution plots, and results tables
for the HaluEval binary detection benchmark.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve

from shared.utils import compute_roc_auc


def plot_roc_curves(results: dict, output_dir: Path) -> None:
    """Plot ROC curves per model x aggregation strategy.

    Includes a length-only logistic regression baseline as a dashed line
    for comparison.

    Args:
        results: nested dict with structure
            {model: {strategy: {"labels": array, "scores": array,
                                "length_scores": array, "auc": float,
                                "length_auc": float}}}
        output_dir: directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = ["entity_weighted_mean", "max_entity_delta", "proportion_above_threshold"]
    strategy_labels = {
        "entity_weighted_mean": "Entity Weighted Mean",
        "max_entity_delta": "Max Entity Delta",
        "proportion_above_threshold": "Proportion Above Threshold",
    }

    for model_name, model_results in results.items():
        fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 5))
        if len(strategies) == 1:
            axes = [axes]

        for ax, strategy in zip(axes, strategies):
            if strategy not in model_results:
                ax.set_title(f"{strategy_labels.get(strategy, strategy)}\n(no data)")
                continue

            data = model_results[strategy]
            labels = np.asarray(data["labels"])
            scores = np.asarray(data["scores"])

            # RI ROC curve
            fpr, tpr, _ = roc_curve(labels, scores)
            auc_val = data.get("auc", compute_roc_auc(labels, scores))
            ax.plot(fpr, tpr, label=f"RI ({auc_val:.3f})", linewidth=2)

            # Length-only baseline
            if "length_scores" in data:
                length_scores = np.asarray(data["length_scores"])
                fpr_l, tpr_l, _ = roc_curve(labels, length_scores)
                length_auc = data.get("length_auc", compute_roc_auc(labels, length_scores))
                ax.plot(
                    fpr_l, tpr_l,
                    label=f"Length only ({length_auc:.3f})",
                    linewidth=1.5, linestyle="--", color="gray",
                )

            # Diagonal reference
            ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{strategy_labels.get(strategy, strategy)}")
            ax.legend(loc="lower right", fontsize=9)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

        fig.suptitle(f"HaluEval ROC — {model_name}", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(
            output_dir / f"roc_curves_{model_name}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    print(f"ROC curve plots saved to {output_dir}")


def plot_entity_delta_distributions(
    scored_examples: list,
    output_dir: Path,
) -> None:
    """Violin plots of per-entity mean_delta for hallucinated vs correct.

    Compares the distribution of RI aggregate scores between the two
    classes, stratified by task type.

    Args:
        scored_examples: list of HaluEvalScoredExample (or dicts)
        output_dir: directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data for violin plots
    plot_data = []
    for ex in scored_examples:
        if isinstance(ex, dict):
            task = ex["task"]
            label_int = ex["label"]
            score_val = ex["scores"].get("entity_weighted_mean", 0.0)
        else:
            task = ex.task
            label_int = ex.label
            score_val = ex.scores.get("entity_weighted_mean", 0.0)

        plot_data.append({
            "task": task,
            "label": "Hallucinated" if label_int == 1 else "Correct",
            "entity_weighted_mean": score_val,
        })

    if not plot_data:
        print("No data for violin plots.")
        return

    # Convert to arrays for seaborn
    tasks = [d["task"] for d in plot_data]
    labels = [d["label"] for d in plot_data]
    values = [d["entity_weighted_mean"] for d in plot_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    unique_tasks = sorted(set(tasks))

    # Build positions for grouped violins
    positions = []
    tick_positions = []
    tick_labels = []
    colors = {"Correct": "#4CAF50", "Hallucinated": "#F44336"}

    for i, task in enumerate(unique_tasks):
        for j, label in enumerate(["Correct", "Hallucinated"]):
            task_label_values = [
                v for t, l, v in zip(tasks, labels, values)
                if t == task and l == label
            ]
            if task_label_values:
                pos = i * 3 + j
                parts = ax.violinplot(
                    task_label_values, positions=[pos],
                    showmeans=True, showmedians=True,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(colors[label])
                    pc.set_alpha(0.7)

        tick_positions.append(i * 3 + 0.5)
        tick_labels.append(task.capitalize())

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Entity Weighted Mean Delta")
    ax.set_title("RI Score Distributions: Correct vs Hallucinated")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", alpha=0.7, label="Correct"),
        Patch(facecolor="#F44336", alpha=0.7, label="Hallucinated"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    fig.savefig(
        output_dir / "entity_delta_distributions.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Distribution plot saved to {output_dir / 'entity_delta_distributions.png'}")


def generate_results_table(results: dict, output_dir: Path) -> None:
    """Save a markdown table with all metrics.

    Args:
        results: output from evaluate_by_task(), keyed by task name
        output_dir: directory to save the table
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# HaluEval Binary Detection Results",
        "",
        "## Trivial Baseline Reference",
        "",
        "- Rule: response length > 27 characters -> hallucinated",
        "- Reported accuracy: ~93.3%",
        "- This highlights the severe length confound in HaluEval",
        "",
        "## RI Signal Results",
        "",
        "| Task | N | AUC-ROC | AUC-ROC 95% CI | AUC-PR | F1@opt | Cohen's d |"
        " Matched AUC | Length-only AUC | RI+Length AUC | Delta AUC |",
        "|------|---|---------|----------------|--------|--------|-----------|"
        "-------------|-----------------|---------------|-----------|",
    ]

    # Order: individual tasks first, then "all"
    task_order = [t for t in sorted(results.keys()) if t != "all"]
    if "all" in results:
        task_order.append("all")

    for task_name in task_order:
        task_result = results[task_name]
        m = task_result["metrics"]
        lm = task_result["length_matched"]
        lr = task_result["length_regression"]

        ci_str = f"[{m['roc_auc_ci'][0]:.3f}, {m['roc_auc_ci'][1]:.3f}]"
        line = (
            f"| {task_name} | {m['n_samples']} | {m['roc_auc']:.3f} | {ci_str} "
            f"| {m['pr_auc']:.3f} | {m['f1_optimal']:.3f} | {m['cohens_d']:.3f} "
            f"| {lm['matched_roc_auc']:.3f} | {lr['auc_length_only']:.3f} "
            f"| {lr['auc_ri_plus_length']:.3f} | {lr['delta_auc']:.3f} |"
        )
        lines.append(line)

    lines.extend([
        "",
        "## Length Bias Analysis",
        "",
        "| Task | RI Coeff | Length Coeff | AUC(length) | AUC(RI) | AUC(RI+length) |",
        "|------|----------|-------------|-------------|---------|----------------|",
    ])

    for task_name in task_order:
        lr = results[task_name]["length_regression"]
        line = (
            f"| {task_name} | {lr['ri_coefficient']:.4f} "
            f"| {lr['length_coefficient']:.4f} "
            f"| {lr['auc_length_only']:.3f} | {lr['auc_ri_only']:.3f} "
            f"| {lr['auc_ri_plus_length']:.3f} |"
        )
        lines.append(line)

    lines.append("")

    table_path = output_dir / "metrics_table.md"
    table_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Results table saved to {table_path}")

    # Also save as JSON for programmatic access
    json_path = output_dir / "metrics.json"

    def _make_serialisable(obj):
        """Convert numpy types to Python natives for JSON serialisation."""
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [_make_serialisable(x) for x in obj]
            return tuple(converted) if isinstance(obj, tuple) else converted
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w") as f:
        json.dump(_make_serialisable(results), f, indent=2)
    print(f"Metrics JSON saved to {json_path}")
