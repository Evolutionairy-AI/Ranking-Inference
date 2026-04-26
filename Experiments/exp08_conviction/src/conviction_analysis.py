"""Conviction-based analysis of RI hallucination detection.

Answers the core question: When RI is confident in its prediction,
how accurate is it? If we can identify a conviction threshold where
accuracy is reliably high, RI can serve as a cheap triage filter
before expensive SOTA methods.

Analyses:
1. Binned accuracy by conviction level (reliability diagram)
2. Expected Calibration Error (ECE)
3. Cost savings: % of outputs resolved at each confidence threshold
4. AUC-ROC at high-confidence subsets
5. Source-baseline vs ROUGE comparison (FRANK summarization)
"""

import json
import sys
from pathlib import Path

import numpy as np
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
FRANK_SCORED = ROOT / "exp06_frank" / "output" / "scored_frank_llama-3.1-8b.jsonl"
HALUEVAL_SCORED = ROOT / "exp04_halueval" / "results" / "scored_llama-3.1-8b_mixed.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def load_frank_data():
    """Load FRANK scored spans with both baselines."""
    data = []
    with open(FRANK_SCORED) as f:
        for line in f:
            row = json.loads(line)
            data.append(row)
    return data


def load_halueval_data():
    """Load HaluEval scored examples."""
    data = []
    with open(HALUEVAL_SCORED) as f:
        for line in f:
            row = json.loads(line)
            data.append(row)
    return data


# ── Conviction score ───────────────────────────────────────────────────
# The RI signal (mean_delta) ranges roughly [0, 1].
# Higher mean_delta = model more confident relative to corpus prior.
# For binary classification: we pick a threshold and predict is_error
# if mean_delta > threshold (or < threshold, depending on signal direction).
#
# "Conviction" = |score - threshold| = how far the score is from the
# decision boundary. Higher conviction = RI is more certain.


def compute_conviction_bins(scores, labels, n_bins=10, score_name="mean_delta"):
    """Bin predictions by conviction and compute accuracy per bin.

    Returns a list of dicts with bin info.
    """
    scores = np.array(scores)
    labels = np.array(labels)

    # Find optimal threshold via Youden's J
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_j, best_thr = -1, 0.5
    for thr in thresholds:
        preds = (scores > thr).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j, best_thr = j, thr

    # Also check inverted signal (lower score = hallucination)
    best_j_inv, best_thr_inv = -1, 0.5
    for thr in thresholds:
        preds = (scores < thr).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j_inv:
            best_j_inv, best_thr_inv = j, thr

    inverted = best_j_inv > best_j
    if inverted:
        best_thr = best_thr_inv
        preds_all = (scores < best_thr).astype(int)
        conviction = np.abs(scores - best_thr)
        # For inverted: lower score = predict hallucination, conviction = distance from threshold
    else:
        preds_all = (scores > best_thr).astype(int)
        conviction = np.abs(scores - best_thr)

    # Bin by conviction
    bin_edges = np.linspace(0, conviction.max() + 1e-9, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conviction >= lo) & (conviction < hi)
        n = mask.sum()
        if n == 0:
            continue
        correct = (preds_all[mask] == labels[mask]).sum()
        acc = correct / n
        bins.append({
            "bin_idx": i,
            "conviction_lo": float(lo),
            "conviction_hi": float(hi),
            "conviction_mean": float(conviction[mask].mean()),
            "n_samples": int(n),
            "n_correct": int(correct),
            "accuracy": float(acc),
            "pct_of_total": float(n / len(scores) * 100),
        })

    return {
        "score_name": score_name,
        "optimal_threshold": float(best_thr),
        "inverted": inverted,
        "best_youdens_j": float(max(best_j, best_j_inv)),
        "overall_accuracy": float((preds_all == labels).mean()),
        "bins": bins,
    }


def compute_ece(scores, labels, threshold, inverted=False, n_bins=10):
    """Expected Calibration Error.

    We treat |score - threshold| as a proxy for confidence,
    normalized to [0, 1].
    """
    scores = np.array(scores)
    labels = np.array(labels)

    if inverted:
        preds = (scores < threshold).astype(int)
    else:
        preds = (scores > threshold).astype(int)

    conviction = np.abs(scores - threshold)
    max_conv = conviction.max() if conviction.max() > 0 else 1
    confidence = 0.5 + 0.5 * (conviction / max_conv)  # map to [0.5, 1.0]

    bin_edges = np.linspace(0.5, 1.0 + 1e-9, n_bins + 1)
    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi)
        n = mask.sum()
        if n == 0:
            continue
        acc = (preds[mask] == labels[mask]).mean()
        conf = confidence[mask].mean()
        ece += (n / len(scores)) * abs(acc - conf)
        bin_details.append({
            "confidence_lo": float(lo),
            "confidence_hi": float(hi),
            "mean_confidence": float(conf),
            "accuracy": float(acc),
            "n_samples": int(n),
            "gap": float(abs(acc - conf)),
        })

    return {"ece": float(ece), "bins": bin_details}


def compute_cost_savings(scores, labels, threshold, inverted=False):
    """At various conviction thresholds, what % of outputs can RI resolve
    (i.e., make a confident call) and what is the accuracy on those?

    This models the triage scenario: RI handles the "easy" cases,
    expensive methods only handle uncertain ones.
    """
    scores = np.array(scores)
    labels = np.array(labels)

    if inverted:
        preds = (scores < threshold).astype(int)
    else:
        preds = (scores > threshold).astype(int)

    conviction = np.abs(scores - threshold)

    results = []
    for min_conviction in np.arange(0.0, conviction.max(), 0.02):
        mask = conviction >= min_conviction
        n_resolved = mask.sum()
        if n_resolved == 0:
            continue
        acc = (preds[mask] == labels[mask]).mean()
        results.append({
            "min_conviction": float(min_conviction),
            "pct_resolved": float(n_resolved / len(scores) * 100),
            "accuracy_on_resolved": float(acc),
            "n_resolved": int(n_resolved),
            "n_deferred": int(len(scores) - n_resolved),
        })

    return results


def compute_auc_at_conviction(scores, labels, threshold, inverted=False,
                               conviction_cutoffs=[0.05, 0.10, 0.15, 0.20]):
    """ROC-AUC computed only on high-conviction subsets."""
    from sklearn.metrics import roc_auc_score

    scores = np.array(scores)
    labels = np.array(labels)
    conviction = np.abs(scores - threshold)

    results = []
    for cutoff in conviction_cutoffs:
        mask = conviction >= cutoff
        n = mask.sum()
        if n < 10 or len(np.unique(labels[mask])) < 2:
            continue
        try:
            if inverted:
                auc = roc_auc_score(labels[mask], -scores[mask])
            else:
                auc = roc_auc_score(labels[mask], scores[mask])
        except ValueError:
            continue
        acc = ((scores[mask] > threshold).astype(int) == labels[mask]).mean() if not inverted else \
              ((scores[mask] < threshold).astype(int) == labels[mask]).mean()
        results.append({
            "min_conviction": float(cutoff),
            "auc_roc": float(auc),
            "accuracy": float(acc),
            "n_samples": int(n),
            "pct_of_total": float(n / len(scores) * 100),
        })

    return results


def analyze_frank():
    """Full conviction analysis on FRANK dataset."""
    data = load_frank_data()

    scores_global = [d["mean_delta"] for d in data]
    scores_source = [d["mean_delta_source"] for d in data]
    scores_log_global = [d["mean_log_delta"] for d in data]
    scores_log_source = [d["mean_log_delta_source"] for d in data]
    labels = [int(d["is_error"]) for d in data]

    results = {}

    # ── Per-score conviction analysis ──
    for name, scores in [
        ("mean_delta_global", scores_global),
        ("mean_delta_source", scores_source),
        ("mean_log_delta_global", scores_log_global),
        ("mean_log_delta_source", scores_log_source),
    ]:
        conv = compute_conviction_bins(scores, labels, n_bins=10, score_name=name)
        ece = compute_ece(scores, labels, conv["optimal_threshold"], conv["inverted"])
        cost = compute_cost_savings(scores, labels, conv["optimal_threshold"], conv["inverted"])
        try:
            auc_conv = compute_auc_at_conviction(
                scores, labels, conv["optimal_threshold"], conv["inverted"],
                conviction_cutoffs=[0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]
            )
        except ImportError:
            auc_conv = []

        results[name] = {
            "conviction": conv,
            "ece": ece,
            "cost_savings": cost,
            "auc_at_conviction": auc_conv,
        }

    # ── Per error-type breakdown ──
    by_error_type = defaultdict(list)
    for d in data:
        by_error_type[d["error_type"]].append(d)

    error_type_conviction = {}
    for etype, rows in by_error_type.items():
        if etype == "control":
            continue
        # For per-type analysis, include controls + this error type
        subset = [d for d in data if d["error_type"] == etype or d["error_type"] == "control"]
        s = [d["mean_delta"] for d in subset]
        l = [int(d["is_error"]) for d in subset]
        if len(set(l)) < 2:
            continue
        conv = compute_conviction_bins(s, l, n_bins=5, score_name=f"mean_delta_{etype}")
        error_type_conviction[etype] = {
            "conviction": conv,
            "n_errors": sum(1 for d in rows),
            "tier": rows[0]["tier"],
        }

    results["per_error_type"] = error_type_conviction

    # ── Source vs Global gap analysis ──
    # The key insight: source baseline should approximate lexical overlap
    gap = []
    for d in data:
        gap.append({
            "global": d["mean_delta"],
            "source": d["mean_delta_source"],
            "gap": d["mean_delta"] - d["mean_delta_source"],
            "log_global": d["mean_log_delta"],
            "log_source": d["mean_log_delta_source"],
            "log_gap": d["mean_log_delta"] - d["mean_log_delta_source"],
            "is_error": d["is_error"],
        })

    gap_arr = np.array([g["gap"] for g in gap])
    log_gap_arr = np.array([g["log_gap"] for g in gap])
    gap_labels = np.array([int(g["is_error"]) for g in gap])

    results["source_vs_global"] = {
        "mean_gap_errors": float(gap_arr[gap_labels == 1].mean()),
        "mean_gap_controls": float(gap_arr[gap_labels == 0].mean()),
        "mean_log_gap_errors": float(log_gap_arr[gap_labels == 1].mean()),
        "mean_log_gap_controls": float(log_gap_arr[gap_labels == 0].mean()),
        "correlation_global_source": float(np.corrcoef(
            [g["global"] for g in gap], [g["source"] for g in gap]
        )[0, 1]),
        "correlation_log_global_source": float(np.corrcoef(
            [g["log_global"] for g in gap], [g["log_source"] for g in gap]
        )[0, 1]),
    }

    return results


def analyze_halueval():
    """Conviction analysis on HaluEval (per-task and combined)."""
    data = load_halueval_data()

    results = {}
    for task in ["qa", "dialogue", "summarization", "all"]:
        if task == "all":
            subset = data
        else:
            subset = [d for d in data if d["task"] == task]

        scores = [d["scores"]["entity_weighted_mean"] for d in subset]
        labels = [d["label"] for d in subset]

        if len(set(labels)) < 2 or len(scores) < 5:
            continue

        conv = compute_conviction_bins(scores, labels, n_bins=5, score_name=f"halueval_{task}")
        ece = compute_ece(scores, labels, conv["optimal_threshold"], conv["inverted"])
        cost = compute_cost_savings(scores, labels, conv["optimal_threshold"], conv["inverted"])

        results[task] = {
            "conviction": conv,
            "ece": ece,
            "cost_savings": cost,
        }

    return results


def format_summary(frank_results, halueval_results):
    """Generate a human-readable summary of key findings."""
    lines = []
    lines.append("=" * 70)
    lines.append("CONVICTION ANALYSIS SUMMARY")
    lines.append("=" * 70)

    # ── FRANK results ──
    lines.append("\n── FRANK Dataset (6,356 spans) ──\n")
    for name in ["mean_delta_global", "mean_delta_source",
                  "mean_log_delta_global", "mean_log_delta_source"]:
        r = frank_results[name]
        conv = r["conviction"]
        lines.append(f"Score: {name}")
        lines.append(f"  Optimal threshold: {conv['optimal_threshold']:.4f}")
        lines.append(f"  Signal inverted: {conv['inverted']}")
        lines.append(f"  Youden's J: {conv['best_youdens_j']:.4f}")
        lines.append(f"  Overall accuracy: {conv['overall_accuracy']:.1%}")
        lines.append(f"  ECE: {r['ece']['ece']:.4f}")
        lines.append("")

        # Show high-conviction bins
        high_conv_bins = [b for b in conv["bins"] if b["accuracy"] >= 0.60]
        if high_conv_bins:
            best = max(high_conv_bins, key=lambda b: b["accuracy"])
            lines.append(f"  Best bin: conviction [{best['conviction_lo']:.3f}, {best['conviction_hi']:.3f}]")
            lines.append(f"    Accuracy: {best['accuracy']:.1%} on {best['n_samples']} samples ({best['pct_of_total']:.1f}% of data)")
        lines.append("")

        # AUC at conviction
        if r["auc_at_conviction"]:
            lines.append("  AUC-ROC at conviction thresholds:")
            for a in r["auc_at_conviction"]:
                lines.append(f"    conviction >= {a['min_conviction']:.2f}: "
                           f"AUC={a['auc_roc']:.3f}, acc={a['accuracy']:.1%}, "
                           f"n={a['n_samples']} ({a['pct_of_total']:.1f}%)")
        lines.append("")

        # Cost savings sweet spots
        if r["cost_savings"]:
            lines.append("  Cost savings (triage performance):")
            for cs in r["cost_savings"]:
                if cs["accuracy_on_resolved"] >= 0.70 and cs["pct_resolved"] >= 10:
                    lines.append(f"    conviction >= {cs['min_conviction']:.2f}: "
                               f"resolve {cs['pct_resolved']:.1f}% at {cs['accuracy_on_resolved']:.1%} accuracy "
                               f"(defer {cs['n_deferred']} to expensive methods)")
        lines.append("-" * 50)

    # ── Source vs Global ──
    svg = frank_results["source_vs_global"]
    lines.append("\n── Source vs Global Baseline ──")
    lines.append(f"  Correlation (linear): {svg['correlation_global_source']:.4f}")
    lines.append(f"  Correlation (log):    {svg['correlation_log_global_source']:.4f}")
    lines.append(f"  Mean gap (errors):    {svg['mean_gap_errors']:.4f}")
    lines.append(f"  Mean gap (controls):  {svg['mean_gap_controls']:.4f}")
    lines.append(f"  Mean log gap (errors):   {svg['mean_log_gap_errors']:.4f}")
    lines.append(f"  Mean log gap (controls): {svg['mean_log_gap_controls']:.4f}")

    # ── Per error type ──
    lines.append("\n── Per Error Type (FRANK, mean_delta_global) ──")
    for etype, info in sorted(frank_results["per_error_type"].items()):
        conv = info["conviction"]
        lines.append(f"  {etype} ({info['tier']}, n={info['n_errors']}): "
                    f"acc={conv['overall_accuracy']:.1%}, "
                    f"J={conv['best_youdens_j']:.3f}")

    # ── HaluEval ──
    if halueval_results:
        lines.append("\n── HaluEval ──")
        for task, r in halueval_results.items():
            conv = r["conviction"]
            lines.append(f"  {task}: acc={conv['overall_accuracy']:.1%}, "
                        f"J={conv['best_youdens_j']:.3f}, "
                        f"ECE={r['ece']['ece']:.4f}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and analyzing FRANK data...")
    frank_results = analyze_frank()

    print("Loading and analyzing HaluEval data...")
    halueval_results = analyze_halueval()

    # Save full results
    full_results = {
        "frank": frank_results,
        "halueval": halueval_results,
    }

    with open(OUT_DIR / "conviction_analysis.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    # Print and save summary
    summary = format_summary(frank_results, halueval_results)
    print(summary)

    with open(OUT_DIR / "conviction_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
