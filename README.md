# Ranking Inference

  [![arXiv](https://img.shields.io/badge/arXiv-2604.25634-b31b1b.svg)](https://arxiv.org/abs/2604.25634)

  Distributional grounding primitives for large language model outputs, built on

This repository accompanies the paper

> **The Surprising Universality of LLM Outputs: A Real-Time Verification Primitive**
> Alex Bogdan and Adrian de Valois-Franklin (Evolutionairy AI), 2026.

It provides the core scoring utilities, the precomputed Wikipedia rank table,
and the experiment scripts and configurations needed to reproduce the paper's
results.

## What is it?

Modern LLMs produce token distributions that follow the Mandelbrot law very
tightly over the body of the rank spectrum. Deviations of an output's local
token rank `r_local` from the global reference rank `r_global` (computed on a
reference corpus, typically Wikipedia) are an information-theoretic anomaly
signal:

```
Δr(t) = log2( r_global(t) / r_local(t) )
posterior(t) ∝ log P_LLM(t) + β · log G_RI(t)
```

where G_RI is the fitted Mandelbrot PMF and β = 1 / σ²(Δr) is the measured
precision of the global-vs-local rank agreement on the domain of interest
(β is a **measurement**, not a hyperparameter).

## Install

```bash
pip install ranking-inference          # core primitives
pip install ranking-inference[ner]     # + spaCy NER for entity-level scoring
pip install ranking-inference[tokenizers]  # + HF tokenizers
python -m spacy download en_core_web_sm
```

## Quick start

```python
from ranking_inference import (
    RankTable, compute_token_scores, aggregate_three_modes,
)
from transformers import AutoTokenizer

rt = RankTable.load("rank_tables/wikipedia_full_llama-3.1-8b.json")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

text = "Marcus Agrippa died in the 1925 eruption of Mount Vesuvius."
token_ids = tok.encode(text, add_special_tokens=False)
logprobs = ...   # length-matched per-token log P_LLM from your model

scores = compute_token_scores(text, token_ids, logprobs, tok, rt)
agg    = aggregate_three_modes(scores)

agg["all_mean_log_delta"]        # output-level (all tokens)
agg["entity_mean_log_delta"]     # entity-level (NER positions only)
agg["rank_only_mean_log_rank"]   # rank-only (no logprobs required)
```

See `examples/minimal_scoring.py` for a runnable end-to-end example.

## What's in the repo

```
ranking_inference/
  mandelbrot.py        C / (r+q)^s fitting (MLE + OLS log-log), AIC/BIC
  rank_utils.py        RankTable: build, serialise, rank deviation helpers
  token_scoring.py     per-token log(P_LLM / G_RI), three aggregation modes
  entity_extraction.py spaCy-NER alignment to subword tokens
  aggregation.py       sentence/document aggregation helpers

rank_tables/
  wikipedia_full_llama-3.1-8b.json   the reference corpus rank table

examples/
  minimal_scoring.py   minimal end-to-end scoring example

tests/
  test_smoke.py        smoke tests for the core primitives

Experiments/
  exp01_mandelbrot_fit/   six-model rank-frequency convergence (Section 3)
  exp02_beta_calibration/ domain-level β estimation (Section 5.2.3)
  exp03_gap_signal/       early gap-signal validation
  exp04_halueval/         HaluEval scoring (Section 5.2)
  exp05_truthfulqa/       TruthfulQA scoring (Section 5.2)
  exp06_frank/            FRANK scoring + entity-level (Sections 5.2, 5.2.2)
  exp07_latency/          CPU latency benchmarking (Section 5.4)
  exp08_conviction/       conviction analysis + ROUGE comparison
  shared/                 shared utilities (corpus tools, tokenizers)
```

## Three aggregation modes

The scoring primitive deliberately emits three aggregates so downstream work
can choose the appropriate privacy / black-box trade-off:

1. **Output-level** — mean log(P_LLM / G_RI) over every token. Requires
   logprobs. Most information, least black-box.
2. **Entity-level** — same as above but restricted to NER-tagged token
   positions. Filters the signal to the tokens most likely to carry factual
   risk.
3. **Rank-only at entities** — just log2(r_global / r_local) at entity
   positions. **Requires no logprobs at all.** Fully black-box, works for
   any API regardless of logprob exposure.

## Reproducing paper results

The `Experiments/` directory contains the source code, configurations, and
orchestration scripts for every experiment reported in the paper. Each
experiment has its own `src/`, `config/`, and `run.py` (or `run_experiment_*.py`)
entry point.

Two categories of inputs are **not** redistributed in this repository:

- **Third-party benchmark datasets** (FRANK, TruthfulQA, HaluEval). Each
  experiment's `run.py` will download or expect these under their own license.
- **Proprietary model outputs** from closed APIs (GPT-5.1, Claude 4.6 Sonnet,
  Gemini 2.5 Pro, Mistral Large). The prompts used to generate them are in
  `Experiments/exp01_mandelbrot_fit/data/prompts/`, and the fitting and
  scoring scripts will regenerate outputs from those prompts using API keys
  you supply.

Set up API keys by creating a directory `Experiments/API_KEYS/` containing
files named like `OpenAI_RI.key.txt`, `Claude_Key.txt`, etc. (one secret per
file). The loader at `Experiments/shared/utils/api_keys.py` will pick them up
as environment variables.

## Tests

```bash
pip install ranking-inference[dev]
pytest tests/
```

## Citation

```bibtex
@article{bogdan2026universality,
    title         = {The Surprising Universality of LLM Outputs:
                     A Real-Time Verification Primitive},
    author        = {Bogdan, Alex and de Valois-Franklin, Adrian},
    year          = {2026},
    eprint        = {2604.25634},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL},
  }
```

## License

MIT. See LICENSE.
