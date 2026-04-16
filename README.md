# Ranking Inference

Distributional grounding primitives for large language model outputs, built on
the **Mandelbrot Ranking Distribution** f(r) = C / (r + q)^s.

This repository accompanies the paper
*Universal Distributional Convergence: A Mandelbrot-Grounded Validation Primitive for LLM Outputs*
and provides the core utilities needed to reproduce its results or apply the
primitive to new settings.

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
  minimal_scoring.py
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

The `Experiments/` directory is kept in the authors' internal repo and is
available on request. Everything needed to run the primitive and reproduce
the rank-only numbers is in this public release.

## Citation

```bibtex
@article{ranking_inference_2026,
  title  = {Universal Distributional Convergence: A Mandelbrot-Grounded
            Validation Primitive for LLM Outputs},
  author = {Wallace AI},
  year   = {2026},
  note   = {arXiv preprint, forthcoming},
}
```

## License

MIT. See LICENSE.
