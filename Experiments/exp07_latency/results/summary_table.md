# Latency Benchmark Summary

## RI Verification Latency

| Length Bin | Mode | Median (us) | P5 (us) | P95 (us) | n |
|-----------|------|-------------|---------|----------|---|
| 0-50 | full | 5072.1 | 4833.8 | 6316.9 | 80 |
| 0-50 | gap_only | 54.2 | 52.9 | 66.4 | 80 |
| 50-100 | full | 11377.0 | 10913.4 | 13587.7 | 80 |
| 50-100 | gap_only | 158.7 | 148.6 | 204.4 | 80 |
| 100-250 | full | 19673.5 | 18639.0 | 23463.1 | 53 |
| 100-250 | gap_only | 317.2 | 286.2 | 457.3 | 53 |

## Baseline Projections

| Method | Config | Median Latency (ms) | AUC-ROC | Source |
|--------|--------|---------------------|---------|--------|
| RI | gap-only | 0.139 | 0.584 | Phase 2 |
| RI | full | 10.506 | 0.584 | Phase 2 |
| SE | k=2 | 8251.5 | ~0.75 | Published |
| SE | k=5 | 21003.7 | ~0.75 | Published |
| SE | k=10 | 43257.4 | ~0.75 | Published |
| SelfCheckGPT | N=2 | 8241.5 | ~0.75 | Published |
| SelfCheckGPT | N=5 | 20603.7 | ~0.75 | Published |
| SelfCheckGPT | N=10 | 41207.4 | ~0.75 | Published |

## Setup Costs

- Rank table load: 157.2464 ms
- Grounding scores: 642.9186 ms

*Note: RI latency measured empirically; baseline AUCs from published results; baseline latencies projected analytically from single forward pass measurement.*

## FLOPs Comparison

*Estimated at median sequence length of 75 tokens (Llama 3.1-8B, 8B params)*

| Method | Extra Forward Passes | FLOPs/example | Notes |
|--------|---------------------|---------------|-------|
| RI (gap-only) | 0 | 300 | Hash lookup + log subtraction per token |
| RI (full, incl. NER) | 0 | 150.3K | Gap computation + SpaCy NER (CPU-bound) |
| SE (k=2) | 2 | 2.5T | 2 forward passes + 1 NLI pairs |
| SE (k=5) | 5 | 7.0T | 5 forward passes + 10 NLI pairs |
| SE (k=10) | 10 | 16.6T | 10 forward passes + 45 NLI pairs |
| SC-GPT (N=2) | 2 | 2.4T | 2 forward passes + consistency scoring |
| SC-GPT (N=5) | 5 | 6.1T | 5 forward passes + consistency scoring |
| SC-GPT (N=10) | 10 | 12.2T | 10 forward passes + consistency scoring |