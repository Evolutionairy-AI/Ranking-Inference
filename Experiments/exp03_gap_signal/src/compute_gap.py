"""
Step 2 (Exp03): Compute the confidence-grounding gap delta(t) per token.

For each generated token:
  P_LLM(t)  = exp(logprob) from model output
  G_RI(t)   = P_RI(t) / Z_RI, where P_RI(t) = C/(r+q)^s from Mandelbrot
  delta(t)  = P_LLM(t) - G_RI(t)

High delta means the model is more confident in a token than the
distributional baseline warrants -- the core hallucination signal.
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import (
    RankTable,
    mandelbrot_freq,
    fit_mandelbrot_mle,
    get_tokenizer,
    tokenize_text,
)

EXP_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = EXP_DIR / "data" / "outputs"
RESULTS_DIR = EXP_DIR / "results"
RANK_TABLE_DIR = PROJECT_ROOT / "shared" / "rank_tables"


def load_rank_table() -> RankTable:
    """Load the Wikipedia global rank table."""
    path = RANK_TABLE_DIR / "wikipedia_gpt-5.1.json"
    if not path.exists():
        raise FileNotFoundError(
            "Wikipedia rank table not found. Run Experiment 02 first."
        )
    return RankTable.load(path)


def compute_grounding_scores(rank_table: RankTable) -> dict:
    """Precompute normalized Mandelbrot grounding scores G_RI(t) for all tokens.

    G_RI(t) = P_RI(t) / Z_RI where P_RI(t) = C/(r+q)^s
    Normalized to [0, 1] over the output vocabulary.
    """
    # Fit Mandelbrot to the reference corpus
    max_rank = rank_table.vocab_size
    ranks = np.arange(1, max_rank + 1)
    freq = rank_table.rank_to_freq[1:max_rank + 1].copy()
    mask = freq > 0
    ranks_nz = ranks[mask]
    freq_nz = freq[mask]

    params = fit_mandelbrot_mle(ranks_nz, freq_nz)
    print(f"Mandelbrot fit: C={params.C:.2f}, q={params.q:.2f}, s={params.s:.2f}")

    # Compute P_RI for all ranks
    all_ranks = np.arange(1, max_rank + 2)  # +1 for unseen tokens
    p_ri = mandelbrot_freq(all_ranks, params.C, params.q, params.s)
    z_ri = p_ri.sum()
    g_ri = p_ri / z_ri  # Normalized grounding scores

    # Build lookup: token_id -> G_RI score
    grounding_scores = {}
    for token_id, rank in rank_table.token_to_rank.items():
        if rank <= max_rank:
            grounding_scores[token_id] = float(g_ri[rank - 1])

    # Default score for unseen tokens (lowest grounding)
    default_score = float(g_ri[-1])

    return grounding_scores, default_score, params


def compute_gap_for_output(
    output: dict,
    tokenizer,
    tokenizer_name: str,
    grounding_scores: dict,
    default_grounding: float,
    beta: float = 1.0,
) -> dict:
    """Compute confidence-grounding gap for every token in an output.

    Returns dict with per-token gap values and aggregated metrics.
    """
    text = output.get("text", "")
    if not text:
        return None

    tokens_data = output.get("tokens", [])
    has_logprobs = output.get("has_logprobs", False)

    # Tokenize the text to get token IDs
    token_ids = tokenize_text(text, tokenizer, tokenizer_name)

    gaps = []
    token_details = []

    # If we have logprobs, use them for P_LLM
    if has_logprobs and len(tokens_data) > 0:
        # Align logprob tokens with tokenizer token IDs
        # Ollama tokens may not 1:1 match tiktoken, so we use logprobs directly
        for i, td in enumerate(tokens_data):
            logprob = td.get("logprob")
            if logprob is None:
                continue

            p_llm = np.exp(logprob)  # Convert logprob to probability

            # Find the token ID for grounding lookup
            # Tokenize the individual token text
            tok_ids = tokenize_text(td["token"], tokenizer, tokenizer_name)
            if tok_ids:
                tid = tok_ids[0]
                g_ri = grounding_scores.get(tid, default_grounding)
            else:
                g_ri = default_grounding

            # Confidence-grounding gap (Section 5)
            delta = p_llm - g_ri

            # Bayesian posterior ratio (Section 6)
            # P_posterior proportional to P_LLM * P_RI^beta
            log_posterior = logprob + beta * np.log(max(g_ri, 1e-20))

            gaps.append(delta)
            token_details.append({
                "token": td["token"],
                "p_llm": float(p_llm),
                "g_ri": float(g_ri),
                "delta": float(delta),
                "logprob": float(logprob),
                "log_posterior": float(log_posterior),
            })
    else:
        # No logprobs: use uniform P_LLM approximation (1/V)
        # This is a weaker signal but still tests the grounding component
        approx_p_llm = 1.0 / 50000  # approximate vocab size

        for tid in token_ids:
            g_ri = grounding_scores.get(tid, default_grounding)
            delta = approx_p_llm - g_ri
            gaps.append(delta)
            token_details.append({
                "token_id": tid,
                "p_llm": float(approx_p_llm),
                "g_ri": float(g_ri),
                "delta": float(delta),
            })

    if not gaps:
        return None

    gaps_arr = np.array(gaps)

    return {
        "condition": output["condition"],
        "prompt_index": output["prompt_index"],
        "n_tokens": len(gaps),
        "has_logprobs": has_logprobs,
        # Aggregate gap metrics
        "mean_gap": float(np.mean(gaps_arr)),
        "median_gap": float(np.median(gaps_arr)),
        "max_gap": float(np.max(gaps_arr)),
        "std_gap": float(np.std(gaps_arr)),
        "pct_positive_gap": float(np.mean(gaps_arr > 0)),
        "mean_abs_gap": float(np.mean(np.abs(gaps_arr))),
        # Distribution summary
        "gap_percentiles": {
            "p5": float(np.percentile(gaps_arr, 5)),
            "p25": float(np.percentile(gaps_arr, 25)),
            "p50": float(np.percentile(gaps_arr, 50)),
            "p75": float(np.percentile(gaps_arr, 75)),
            "p95": float(np.percentile(gaps_arr, 95)),
        },
        # Per-token details (for downstream analysis)
        "token_details": token_details,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load rank table and compute grounding scores
    print("Loading Wikipedia rank table...")
    rank_table = load_rank_table()
    print(f"  {rank_table.vocab_size:,} tokens, {rank_table.total_tokens:,} total")

    print("Computing Mandelbrot grounding scores...")
    grounding_scores, default_grounding, mandelbrot_params = compute_grounding_scores(rank_table)
    print(f"  {len(grounding_scores):,} tokens with grounding scores")

    # Load outputs
    combined_path = OUTPUT_DIR / "all_outputs.json"
    with open(combined_path) as f:
        all_outputs = json.load(f)

    tokenizer = get_tokenizer("gpt-5.1")

    # Compute gaps for all outputs at multiple beta values
    beta_values = [0.5, 1.0, 1.5, 2.0]

    for beta in beta_values:
        print(f"\n=== Beta = {beta} ===")
        all_gap_results = {}

        for condition, outputs in all_outputs.items():
            print(f"  {condition}...")
            gap_results = []

            for output in tqdm(outputs, desc=f"  {condition}", leave=False):
                if output.get("error"):
                    continue
                result = compute_gap_for_output(
                    output, tokenizer, "gpt-5.1",
                    grounding_scores, default_grounding, beta=beta,
                )
                if result:
                    gap_results.append(result)

            all_gap_results[condition] = gap_results

            if gap_results:
                mean_gaps = [r["mean_gap"] for r in gap_results]
                print(f"    n={len(gap_results)}, "
                      f"mean_gap={np.mean(mean_gaps):.6f}, "
                      f"std={np.std(mean_gaps):.6f}")

        # Save results for this beta
        results_path = RESULTS_DIR / f"gap_results_beta_{beta}.json"
        # Save without token_details to keep file size manageable
        summary_results = {}
        for condition, results in all_gap_results.items():
            summary_results[condition] = [
                {k: v for k, v in r.items() if k != "token_details"}
                for r in results
            ]
        with open(results_path, "w") as f:
            json.dump(summary_results, f, indent=2)

        # Save full details for beta=1.0 only
        if beta == 1.0:
            full_path = RESULTS_DIR / "gap_results_full_beta_1.0.json"
            with open(full_path, "w") as f:
                json.dump(all_gap_results, f, indent=2)

    print("\nAll gap results saved.")


if __name__ == "__main__":
    main()
