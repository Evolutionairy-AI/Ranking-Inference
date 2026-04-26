"""Publication-quality charts for latency benchmarking results.

Generates Pareto frontier and latency-vs-length plots comparing
RI verification against Semantic Entropy and SelfCheckGPT baselines.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp07_latency.src.flops import estimate_flops

# Phase 2 AUC-ROC results for RI
RI_AUCS = {
    "FRANK": 0.585,
    "TruthfulQA": 0.573,
    "HaluEval": 0.593,
}

# Published baselines
BASELINE_AUC = 0.75


def _get_ri_median_ms(ri_results: dict, mode: str = "gap_only") -> float:
    """Extract overall median latency in ms from RI results across all bins."""
    all_medians = []
    for bin_name, bin_data in ri_results.get("per_bin", {}).items():
        for entry in bin_data.get(mode, []):
            all_medians.append(entry["median_ms"])
    if all_medians:
        return float(np.median(all_medians))
    return 0.1  # fallback


def _get_se_timings_ms(baseline_results: dict) -> dict[str, float]:
    """Extract SE timings per k from baseline results."""
    timings = {}
    se = baseline_results.get("semantic_entropy", {})
    for k_label in ["k=2", "k=5", "k=10"]:
        k_data = se.get(k_label, {})
        per_bin = k_data.get("per_bin", {})
        if per_bin:
            all_totals = [v["total_ms"] for v in per_bin.values()]
            timings[k_label] = float(np.median(all_totals))
        else:
            # Analytical fallback: estimate from forward pass
            timings[k_label] = 500.0 * int(k_label.split("=")[1])
    return timings


def _get_scgpt_timings_ms(baseline_results: dict) -> dict[str, float]:
    """Extract SelfCheckGPT timings per N from baseline results."""
    timings = {}
    scgpt = baseline_results.get("selfcheckgpt", {})
    for n_label in ["N=2", "N=5", "N=10"]:
        n_data = scgpt.get(n_label, {})
        per_bin = n_data.get("per_bin", {})
        if per_bin:
            all_totals = [v["total_ms"] for v in per_bin.values()]
            timings[n_label] = float(np.median(all_totals))
        else:
            timings[n_label] = 500.0 * int(n_label.split("=")[1])
    return timings


def plot_pareto_frontier(
    ri_results: dict,
    baseline_results: dict,
    output_path: Path,
) -> None:
    """Plot Pareto frontier: verification latency (ms) vs AUC-ROC.

    X-axis: verification latency in milliseconds (log scale)
    Y-axis: AUC-ROC

    Shows RI at Phase 2 AUCs, SE at k=2/5/10, SelfCheckGPT at N=2/5/10.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # RI points - use gap_only median for each benchmark
    ri_ms = _get_ri_median_ms(ri_results, "gap_only")
    ri_full_ms = _get_ri_median_ms(ri_results, "full")

    for label, auc in RI_AUCS.items():
        ax.scatter(ri_ms, auc, marker="D", s=80, color="#2196F3", zorder=5)
        ax.annotate(
            f"RI ({label})",
            (ri_ms, auc),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=8,
            color="#2196F3",
        )

    # SE points — stagger labels downward
    se_timings = _get_se_timings_ms(baseline_results)
    se_auc = baseline_results.get("semantic_entropy", {}).get("k=5", {}).get(
        "auc_roc", BASELINE_AUC
    )
    for i, (k_label, ms) in enumerate(se_timings.items()):
        ax.scatter(ms, se_auc, marker="s", s=80, color="#F44336", zorder=5)
        ax.annotate(
            f"SE ({k_label})",
            (ms, se_auc),
            textcoords="offset points",
            xytext=(10, -14 * i),
            fontsize=7.5,
            color="#F44336",
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3),
        )

    # SelfCheckGPT points — stagger labels upward
    scgpt_timings = _get_scgpt_timings_ms(baseline_results)
    scgpt_auc = baseline_results.get("selfcheckgpt", {}).get("N=5", {}).get(
        "auc_roc", BASELINE_AUC
    )
    for i, (n_label, ms) in enumerate(scgpt_timings.items()):
        ax.scatter(ms, scgpt_auc, marker="^", s=80, color="#FF9800", zorder=5)
        ax.annotate(
            f"SC-GPT ({n_label})",
            (ms, scgpt_auc),
            textcoords="offset points",
            xytext=(10, 14 * (i + 1)),
            fontsize=7.5,
            color="#FF9800",
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Verification Latency (ms)", fontsize=11)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title("Pareto Frontier: Latency vs Detection Quality", fontsize=13)
    ax.grid(True, alpha=0.3, which="both")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#2196F3",
               markersize=8, label="RI (gap-only)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#F44336",
               markersize=8, label="Semantic Entropy"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#FF9800",
               markersize=8, label="SelfCheckGPT"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Caption
    fig.text(
        0.5, 0.01,
        "Note: RI latency measured empirically; baseline AUCs from published results; "
        "baseline latencies projected analytically from single forward pass.",
        ha="center", fontsize=7, style="italic", color="gray",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pareto frontier to {output_path}")


def plot_latency_vs_length(
    ri_results: dict,
    baseline_results: dict,
    output_path: Path,
) -> None:
    """Plot latency vs sequence length.

    X-axis: sequence length (tokens)
    Y-axis: verification time in microseconds (log scale, single axis)

    Shows RI full and gap-only as separate lines, plus SE k=5 and
    SelfCheckGPT N=5 projected lines.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Collect RI data points per bin
    bin_midpoints = []
    full_medians_us = []
    gap_medians_us = []

    per_bin = ri_results.get("per_bin", {})
    for bin_name in sorted(per_bin.keys(), key=lambda x: int(x.split("-")[0])):
        bin_data = per_bin[bin_name]
        lo, hi = bin_name.split("-")
        mid = (int(lo) + min(int(hi), 2000)) / 2
        bin_midpoints.append(mid)

        full_entries = bin_data.get("full", [])
        gap_entries = bin_data.get("gap_only", [])

        if full_entries:
            full_medians_us.append(float(np.median([e["median_us"] for e in full_entries])))
        if gap_entries:
            gap_medians_us.append(float(np.median([e["median_us"] for e in gap_entries])))

    # Plot RI lines
    if bin_midpoints and full_medians_us:
        ax.plot(
            bin_midpoints[:len(full_medians_us)], full_medians_us,
            "o-", color="#2196F3", label="RI (full, incl. NER)", linewidth=2,
        )
    if bin_midpoints and gap_medians_us:
        ax.plot(
            bin_midpoints[:len(gap_medians_us)], gap_medians_us,
            "D--", color="#1565C0", label="RI (gap-only)", linewidth=2,
        )

    # Project baseline lines across same token range
    if bin_midpoints:
        x_range = np.array(bin_midpoints)

        # Forward pass per bin
        fp_per_bin = baseline_results.get("forward_pass", {})

        # SE k=5
        se_k5 = baseline_results.get("semantic_entropy", {}).get("k=5", {}).get("per_bin", {})
        se_us = []
        for bin_name in sorted(per_bin.keys(), key=lambda x: int(x.split("-")[0])):
            if bin_name in se_k5:
                se_us.append(se_k5[bin_name]["total_ms"] * 1000)  # ms -> us
            elif fp_per_bin:
                # Fallback: estimate
                fp_ms = list(fp_per_bin.values())[0].get("median_ms", 500)
                se_us.append(5 * fp_ms * 1000)
            else:
                se_us.append(2_500_000)  # 2.5s default
        if se_us:
            ax.plot(
                x_range[:len(se_us)], se_us,
                "s:", color="#F44336", label="SE (k=5, projected)", linewidth=1.5,
            )

        # SelfCheckGPT N=5
        scgpt_n5 = baseline_results.get("selfcheckgpt", {}).get("N=5", {}).get("per_bin", {})
        scgpt_us = []
        for bin_name in sorted(per_bin.keys(), key=lambda x: int(x.split("-")[0])):
            if bin_name in scgpt_n5:
                scgpt_us.append(scgpt_n5[bin_name]["total_ms"] * 1000)
            elif fp_per_bin:
                fp_ms = list(fp_per_bin.values())[0].get("median_ms", 500)
                scgpt_us.append(5 * fp_ms * 1000)
            else:
                scgpt_us.append(2_500_000)
        if scgpt_us:
            ax.plot(
                x_range[:len(scgpt_us)], scgpt_us,
                "^:", color="#FF9800", label="SelfCheckGPT (N=5, projected)", linewidth=1.5,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length (tokens)", fontsize=11)
    ax.set_ylabel("Verification Time (\u00b5s)", fontsize=11)
    ax.set_title("Latency vs Sequence Length", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved latency vs length to {output_path}")


def generate_summary_table(ri_results: dict, baseline_results: dict) -> str:
    """Generate a markdown summary table of latency results.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    lines = []
    lines.append("# Latency Benchmark Summary")
    lines.append("")
    lines.append("## RI Verification Latency")
    lines.append("")
    lines.append("| Length Bin | Mode | Median (us) | P5 (us) | P95 (us) | n |")
    lines.append("|-----------|------|-------------|---------|----------|---|")

    per_bin = ri_results.get("per_bin", {})
    for bin_name in sorted(per_bin.keys(), key=lambda x: int(x.split("-")[0])):
        bin_data = per_bin[bin_name]
        for mode in ["full", "gap_only"]:
            entries = bin_data.get(mode, [])
            if entries:
                median = float(np.median([e["median_us"] for e in entries]))
                p5 = float(np.median([e["p5_ns"] for e in entries])) / 1000
                p95 = float(np.median([e["p95_ns"] for e in entries])) / 1000
                lines.append(
                    f"| {bin_name} | {mode} | {median:.1f} | {p5:.1f} | {p95:.1f} | {len(entries)} |"
                )

    lines.append("")
    lines.append("## Baseline Projections")
    lines.append("")
    lines.append("| Method | Config | Median Latency (ms) | AUC-ROC | Source |")
    lines.append("|--------|--------|---------------------|---------|--------|")

    # RI summary row
    ri_gap_ms = _get_ri_median_ms(ri_results, "gap_only")
    ri_full_ms = _get_ri_median_ms(ri_results, "full")
    avg_auc = float(np.mean(list(RI_AUCS.values())))
    lines.append(f"| RI | gap-only | {ri_gap_ms:.3f} | {avg_auc:.3f} | Phase 2 |")
    lines.append(f"| RI | full | {ri_full_ms:.3f} | {avg_auc:.3f} | Phase 2 |")

    # SE
    se_timings = _get_se_timings_ms(baseline_results)
    se_auc = BASELINE_AUC
    for k_label, ms in se_timings.items():
        lines.append(f"| SE | {k_label} | {ms:.1f} | ~{se_auc} | Published |")

    # SelfCheckGPT
    scgpt_timings = _get_scgpt_timings_ms(baseline_results)
    for n_label, ms in scgpt_timings.items():
        lines.append(f"| SelfCheckGPT | {n_label} | {ms:.1f} | ~{BASELINE_AUC} | Published |")

    lines.append("")
    lines.append("## Setup Costs")
    lines.append("")
    setup = ri_results.get("setup", {})
    rt_load = setup.get("rank_table_load", {})
    gs = setup.get("grounding_scores", {})
    lines.append(f"- Rank table load: {rt_load.get('median_ms', 'N/A')} ms")
    lines.append(f"- Grounding scores: {gs.get('median_ms', 'N/A')} ms")
    lines.append("")
    lines.append(
        "*Note: RI latency measured empirically; baseline AUCs from published results; "
        "baseline latencies projected analytically from single forward pass measurement.*"
    )

    # FLOPs comparison
    # Use median token count from RI results
    all_token_counts = []
    for bin_name in per_bin:
        lo, hi = bin_name.split("-")
        mid = (int(lo) + min(int(hi), 2000)) / 2
        n_entries = len(per_bin[bin_name].get("full", []))
        all_token_counts.extend([mid] * n_entries)
    median_tokens = int(np.median(all_token_counts)) if all_token_counts else 100

    flops = estimate_flops(n_tokens=median_tokens)

    lines.append("")
    lines.append("## FLOPs Comparison")
    lines.append("")
    lines.append(f"*Estimated at median sequence length of {median_tokens} tokens (Llama 3.1-8B, 8B params)*")
    lines.append("")
    lines.append(flops["markdown_table"])

    return "\n".join(lines)
