# Conviction Analysis & ROUGE Comparison Report

**Date**: 2026-04-07  
**Dataset**: FRANK (6,356 spans, 499 articles), HaluEval (30 examples)  
**Model**: Llama 3.1-8B

---

## Executive Summary

This analysis responds to three questions Ibrahim raised about the existing experimental results:

1. **When RI is confident, how often is it right?** The data shows RI's conviction does not reliably gate accuracy. Even at high conviction thresholds, span-level accuracy remains in the 55-67% range, never reaching the 90%+ needed for autonomous replacement of SOTA methods.

2. **Does source-based RI reduce to lexical overlap (ROUGE)?** Yes, partially. Source-RI (log-space) correlates significantly with ROUGE-2 (Spearman rho = 0.349, p < 1e-15). At article level, source-RI achieves AUC = 0.713 for error detection vs ROUGE-2's 0.845. Global-RI does not correlate with ROUGE (rho = 0.090), confirming it measures distributional grounding, not lexical overlap.

3. **Can RI serve as a Pareto-optimal triage filter?** The cascade simulation shows that routing 28% of outputs through RI (saving 28% expensive calls) maintains 78% overall accuracy vs the 85% expensive-only baseline. The value proposition is speed (0.14ms vs 20,000ms) not accuracy.

---

## 1. Conviction Analysis

### 1.1 Span-Level Results (FRANK)

| Score Variant | Optimal Threshold | Youden's J | Overall Acc | ECE |
|---|---|---|---|---|
| Linear Global | 0.8526 | 0.094 | 56.6% | 0.077 |
| Linear Source | 0.8558 | 0.092 | 56.8% | 0.073 |
| Log Global | 7.930 (inv.) | 0.102 | 57.0% | 0.064 |
| **Log Source** | **5.434 (inv.)** | **0.142** | **57.4%** | **0.057** |

**Key finding**: Log-space source baseline is the strongest variant (J = 0.142), but all variants produce near-chance accuracy at the span level. Conviction bins show no reliable "high-accuracy zone" -- accuracy fluctuates between 45-67% regardless of conviction level, and bins with high accuracy contain too few samples (< 10) to be statistically meaningful.

### 1.2 AUC-ROC at High Conviction (Log Source)

| Min Conviction | AUC-ROC | Accuracy | n Samples | % of Data |
|---|---|---|---|---|
| 0.01 | 0.585 | 57.4% | 6,242 | 98.2% |
| 0.05 | 0.587 | 58.0% | 5,803 | 91.3% |
| 0.10 | 0.588 | 58.5% | 5,278 | 83.0% |
| 0.20 | 0.591 | 58.6% | 4,277 | 67.3% |
| 0.30 | 0.596 | 58.6% | 3,268 | 51.4% |

**Interpretation**: Higher conviction marginally improves AUC (0.585 -> 0.596) while shedding nearly half the data. There is no conviction threshold where accuracy jumps to 90%+ on a meaningful subset.

### 1.3 Cost Savings (Triage Performance)

The log-space source variant reaches 65% accuracy when resolving only 8% of outputs, and 67% accuracy on less than 2% of outputs. These are insufficient for the "confident triage" scenario Ibrahim envisioned.

**Ibrahim's hypothesis**: "If conviction > 0.87, RI is 90% correct" -- **Not supported by the data**. The signal is too weak at the span level for this.

### 1.4 Per-Error-Type Analysis

When evaluating each error type against controls separately:

| Error Type | Tier | n Errors | Accuracy | Youden's J |
|---|---|---|---|---|
| CorefE | tier2 | 266 | 87.3% | 0.099 |
| LinkE | tier2 | 80 | 78.6% | 0.146 |
| OutE | tier1 | 960 | 72.1% | 0.084 |
| CircE | tier1.5 | 318 | 71.8% | 0.109 |
| RelE | tier1.5 | 274 | 71.0% | 0.108 |
| EntE | tier1 | 974 | 66.8% | 0.140 |

Note: High accuracy for tier2 errors is driven by class imbalance (3,484 controls vs small error sets), not discriminative power.

---

## 2. Source Baseline vs ROUGE Comparison

### 2.1 Correlation Analysis (Article Level, n=499)

| Comparison | Pearson r | p-value | Interpretation |
|---|---|---|---|
| RI_source_log vs ROUGE-2 | 0.291 | 3.6e-11 | **Moderate** -- confirms partial overlap |
| RI_source_log vs ROUGE-L | 0.219 | 7.7e-07 | Moderate |
| RI_source_log vs ROUGE-1 | 0.156 | 4.6e-04 | Weak |
| RI_global_log vs ROUGE-2 | 0.078 | 0.082 | **Not significant** -- different signal |
| RI_source_linear vs ROUGE-2 | -0.001 | 0.980 | No correlation (linear space) |

**Finding**: Ibrahim's insight is confirmed in log-space only. Source-RI in log-space partially reduces to a lexical overlap measure (r = 0.29 with ROUGE-2). Global-RI does not (r = 0.08, n.s.). Linear-space scores show no correlation with any ROUGE variant.

### 2.2 Article-Level Error Detection (AUC-ROC)

| Method | AUC-ROC | Category |
|---|---|---|
| ROUGE-2 F1 | **0.845** | Lexical overlap |
| ROUGE-L F1 | 0.781 | Lexical overlap |
| ROUGE-1 F1 | 0.718 | Lexical overlap |
| **RI source log** | **0.713** | RI (source prior) |
| RI global log | 0.564 | RI (global prior) |
| RI source linear | 0.536 | RI (source prior) |
| RI global linear | 0.536 | RI (global prior) |

**Interpretation**: Source-RI in log-space (AUC = 0.713) performs comparably to ROUGE-1 (AUC = 0.718) at the article level. This validates that source-based RI has degenerated into an approximate lexical overlap check. ROUGE-2 (AUC = 0.845) remains the proper baseline for source-based summarization evaluation.

### 2.3 Source vs Global Gap

| Metric | Value |
|---|---|
| Linear correlation (source vs global) | 0.9997 |
| Log correlation (source vs global) | 0.4944 |

In linear space, source and global baselines are nearly identical (r = 0.9997). The data leakage Ibrahim identified only manifests in log-space, where the source prior captures token-level frequency differences between the source article and the output.

---

## 3. Pareto Frontier / Triage Framing

### 3.1 The Honest Positioning

RI is not competitive with SOTA methods on raw detection accuracy:
- RI best case: AUC = 0.585 (span-level), 0.713 (article-level, source-log)
- SelfCheckGPT: AUC ~ 0.75, but costs 20,000ms
- FActScore: AUC ~ 0.85, but costs 5,000ms
- ROUGE-2: AUC = 0.845, costs ~0.5ms

RI's advantage is pure speed: 0.14ms per span (gap computation only).

### 3.2 Cascade Pipeline Simulation

Assuming an expensive method with 85% accuracy:
- **Route 10% through RI**: Overall accuracy drops from 85% to 84.3%, save 10% of expensive calls
- **Route 28% through RI**: Overall accuracy = 78.4%, save 28% of expensive calls
- **Route 50% through RI**: Overall accuracy = 71%, save 50% of expensive calls

The cascade shows a roughly linear accuracy-cost tradeoff with no "sweet spot" where RI handles a large fraction at high accuracy.

### 3.3 What This Means for the Paper

The triage framing works **only if speed is the constraint, not accuracy**. In production scenarios where:
- Volume is extreme (millions of outputs/day)
- Latency budget is <10ms
- Approximate triage is acceptable (flag for human review rather than auto-reject)

RI can serve as a first-pass filter. But the paper should not claim it replaces SOTA methods -- it provides a different point on the Pareto frontier.

---

## 4. Recommendations for Next Steps

1. **Do not claim conviction-gated accuracy** -- the data doesn't support it. Instead, frame RI as a distributional primitive that provides weak-but-fast signal.

2. **Separate source-based and global-based results** in the paper. Source-based RI = approximate ROUGE (use ROUGE as baseline). Global-based RI = genuine distributional grounding (use SelfCheckGPT/SE as baseline).

3. **Reframe the contribution**: RI isn't a hallucination detector -- it's a distributional anomaly signal that runs 150,000x faster than sampling-based methods. The theoretical contribution (Mandelbrot ranking distribution) is stronger than the empirical detection performance.

4. **Explore the tool/agentic hallucination direction** Ibrahim mentioned -- fewer competing baselines, stronger case for RI's distributional approach.

---

## Files

- `results/conviction_analysis.json` -- Full conviction binning results
- `results/conviction_summary.txt` -- Text summary
- `results/rouge_comparison.json` -- ROUGE vs RI correlation data
- `results/fig1_reliability_diagrams.png` -- Conviction vs accuracy bins
- `results/fig2_cost_savings.png` -- Triage accuracy curves
- `results/fig3_rouge_vs_ri.png` -- ROUGE-2 vs RI scatter plots
- `results/fig4_pareto_frontier.png` -- Latency vs AUC Pareto chart
- `results/fig5_cascade_pipeline.png` -- Cascade simulation
- `results/fig6_auc_comparison.png` -- AUC bar chart comparison
