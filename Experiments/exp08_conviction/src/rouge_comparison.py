"""Compare source-baseline RI scores against ROUGE on FRANK dataset.

Ibrahim's insight: when the Mandelbrot prior is built from the source
article (not global Wikipedia), the RI signal degenerates into a lexical
overlap measure. If true, source-RI should correlate with ROUGE, and
ROUGE becomes the fair baseline for source-based experiments.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FRANK_DATA = ROOT / "exp06_frank" / "data" / "frank_sentence_annotations.json"
FRANK_SCORED = ROOT / "exp06_frank" / "output" / "scored_frank_llama-3.1-8b.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def load_frank_source():
    """Load FRANK annotations with source articles."""
    with open(FRANK_DATA, encoding="utf-8") as f:
        data = json.load(f)
    by_hash = {d["hash"]: d for d in data}
    return by_hash


def load_scored():
    """Load scored spans."""
    data = []
    with open(FRANK_SCORED) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_rouge_n(reference, hypothesis, n=1):
    """Simple ROUGE-N implementation (no external deps)."""
    def ngrams(text, n):
        tokens = text.lower().split()
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    ref_ngrams = ngrams(reference, n)
    hyp_ngrams = ngrams(hypothesis, n)

    if not ref_ngrams or not hyp_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_counts = defaultdict(int)
    for ng in ref_ngrams:
        ref_counts[ng] += 1

    hyp_counts = defaultdict(int)
    for ng in hyp_ngrams:
        hyp_counts[ng] += 1

    overlap = 0
    for ng, count in hyp_counts.items():
        overlap += min(count, ref_counts.get(ng, 0))

    precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0
    recall = overlap / len(ref_ngrams) if ref_ngrams else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rouge_l(reference, hypothesis):
    """ROUGE-L via longest common subsequence."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs = dp[m][n]
    precision = lcs / n if n > 0 else 0
    recall = lcs / m if m > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FRANK source data...")
    source_data = load_frank_source()
    print(f"  {len(source_data)} articles loaded")

    print("Loading scored spans...")
    scored = load_scored()
    print(f"  {len(scored)} spans loaded")

    # Group scored spans by example_id (= article hash)
    by_example = defaultdict(list)
    for s in scored:
        by_example[s["example_id"]].append(s)

    # For each article, compute ROUGE between article and summary
    # Then correlate with source-baseline RI scores
    results = []
    missing = 0

    for example_id, spans in by_example.items():
        if example_id not in source_data:
            missing += 1
            continue

        src = source_data[example_id]
        article = src.get("article", "")
        summary = src.get("summary", "")

        if not article or not summary:
            continue

        # ROUGE scores (summary against source article)
        rouge1 = compute_rouge_n(article, summary, n=1)
        rouge2 = compute_rouge_n(article, summary, n=2)
        rougel = compute_rouge_l(article, summary)

        # RI scores for this article's spans
        error_spans = [s for s in spans if s["is_error"]]
        control_spans = [s for s in spans if not s["is_error"]]

        mean_ri_source = np.mean([s["mean_delta_source"] for s in spans])
        mean_ri_global = np.mean([s["mean_delta"] for s in spans])
        mean_log_ri_source = np.mean([s["mean_log_delta_source"] for s in spans])
        mean_log_ri_global = np.mean([s["mean_log_delta"] for s in spans])

        # Per-article error rate
        error_rate = len(error_spans) / len(spans) if spans else 0

        results.append({
            "example_id": example_id,
            "n_spans": len(spans),
            "n_errors": len(error_spans),
            "n_controls": len(control_spans),
            "error_rate": error_rate,
            "rouge1_f1": rouge1["f1"],
            "rouge1_precision": rouge1["precision"],
            "rouge1_recall": rouge1["recall"],
            "rouge2_f1": rouge2["f1"],
            "rougel_f1": rougel["f1"],
            "mean_ri_source": float(mean_ri_source),
            "mean_ri_global": float(mean_ri_global),
            "mean_log_ri_source": float(mean_log_ri_source),
            "mean_log_ri_global": float(mean_log_ri_global),
        })

    print(f"  {len(results)} articles matched, {missing} missing")

    # ── Correlation analysis ──
    rouge1_f1 = np.array([r["rouge1_f1"] for r in results])
    rouge2_f1 = np.array([r["rouge2_f1"] for r in results])
    rougel_f1 = np.array([r["rougel_f1"] for r in results])
    ri_source = np.array([r["mean_ri_source"] for r in results])
    ri_global = np.array([r["mean_ri_global"] for r in results])
    log_ri_source = np.array([r["mean_log_ri_source"] for r in results])
    log_ri_global = np.array([r["mean_log_ri_global"] for r in results])
    error_rate = np.array([r["error_rate"] for r in results])

    from scipy.stats import pearsonr, spearmanr

    correlations = {}
    for rouge_name, rouge_vals in [("ROUGE-1", rouge1_f1), ("ROUGE-2", rouge2_f1), ("ROUGE-L", rougel_f1)]:
        for ri_name, ri_vals in [
            ("RI_source_linear", ri_source),
            ("RI_global_linear", ri_global),
            ("RI_source_log", log_ri_source),
            ("RI_global_log", log_ri_global),
        ]:
            pr, pp = pearsonr(rouge_vals, ri_vals)
            sr, sp = spearmanr(rouge_vals, ri_vals)
            correlations[f"{rouge_name}_vs_{ri_name}"] = {
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_rho": float(sr),
                "spearman_p": float(sp),
            }

    # ── ROUGE as error rate predictor ──
    for rouge_name, rouge_vals in [("ROUGE-1", rouge1_f1), ("ROUGE-2", rouge2_f1), ("ROUGE-L", rougel_f1)]:
        pr, pp = pearsonr(rouge_vals, error_rate)
        sr, sp = spearmanr(rouge_vals, error_rate)
        correlations[f"{rouge_name}_vs_error_rate"] = {
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp),
        }

    for ri_name, ri_vals in [
        ("RI_source_linear", ri_source),
        ("RI_global_linear", ri_global),
        ("RI_source_log", log_ri_source),
        ("RI_global_log", log_ri_global),
    ]:
        pr, pp = pearsonr(ri_vals, error_rate)
        sr, sp = spearmanr(ri_vals, error_rate)
        correlations[f"{ri_name}_vs_error_rate"] = {
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp),
        }

    # ── ROC-AUC: ROUGE vs RI for predicting high-error articles ──
    from sklearn.metrics import roc_auc_score
    # Binary: articles with error_rate > median are "hallucinated"
    median_err = np.median(error_rate)
    binary_labels = (error_rate > median_err).astype(int)

    auc_comparison = {}
    if len(np.unique(binary_labels)) == 2:
        for name, vals in [
            ("ROUGE-1_F1", -rouge1_f1),  # negative because higher ROUGE = less error
            ("ROUGE-2_F1", -rouge2_f1),
            ("ROUGE-L_F1", -rougel_f1),
            ("RI_source_linear", ri_source),
            ("RI_global_linear", ri_global),
            ("RI_source_log", -log_ri_source),  # negative because lower log = more error (inverted)
            ("RI_global_log", -log_ri_global),
        ]:
            try:
                auc = roc_auc_score(binary_labels, vals)
                auc_comparison[name] = float(auc)
            except ValueError:
                pass

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("ROUGE vs RI CORRELATION ANALYSIS")
    print("=" * 60)

    print(f"\nDataset: FRANK ({len(results)} articles)")
    print(f"Median error rate: {median_err:.3f}")

    print("\n── RI-Source vs ROUGE correlations ──")
    print("(If source-RI ≈ lexical overlap, these should be high)")
    for key, val in sorted(correlations.items()):
        if "source" in key.lower() and "error_rate" not in key:
            print(f"  {key}:")
            print(f"    Pearson r = {val['pearson_r']:.4f} (p={val['pearson_p']:.2e})")
            print(f"    Spearman ρ = {val['spearman_rho']:.4f} (p={val['spearman_p']:.2e})")

    print("\n── RI-Global vs ROUGE correlations ──")
    print("(Global-RI should be less correlated with ROUGE)")
    for key, val in sorted(correlations.items()):
        if "global" in key.lower() and "error_rate" not in key:
            print(f"  {key}:")
            print(f"    Pearson r = {val['pearson_r']:.4f} (p={val['pearson_p']:.2e})")
            print(f"    Spearman ρ = {val['spearman_rho']:.4f} (p={val['spearman_p']:.2e})")

    print("\n── Error rate prediction (article-level) ──")
    for key, val in sorted(correlations.items()):
        if "error_rate" in key:
            print(f"  {key}:")
            print(f"    Pearson r = {val['pearson_r']:.4f} (p={val['pearson_p']:.2e})")

    print("\n── ROC-AUC for high-error article detection ──")
    for name, auc in sorted(auc_comparison.items(), key=lambda x: -x[1]):
        print(f"  {name}: AUC = {auc:.4f}")

    # ── Save results ──
    output = {
        "n_articles": len(results),
        "median_error_rate": float(median_err),
        "correlations": correlations,
        "auc_comparison": auc_comparison,
        "per_article": results,
        "rouge_stats": {
            "rouge1_f1_mean": float(rouge1_f1.mean()),
            "rouge1_f1_std": float(rouge1_f1.std()),
            "rouge2_f1_mean": float(rouge2_f1.mean()),
            "rouge2_f1_std": float(rouge2_f1.std()),
            "rougel_f1_mean": float(rougel_f1.mean()),
            "rougel_f1_std": float(rougel_f1.std()),
        },
    }

    with open(OUT_DIR / "rouge_comparison.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUT_DIR / 'rouge_comparison.json'}")


if __name__ == "__main__":
    main()
