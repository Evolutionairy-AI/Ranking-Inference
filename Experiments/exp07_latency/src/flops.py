"""FLOPs estimation for RI vs baseline methods.

All estimates are analytical — no measurement required.
Based on Llama 3.1-8B architecture (8 billion parameters).
"""

MODEL_PARAMS = 8e9  # Llama 3.1-8B

# RI gap-only: hash lookup + log subtraction per token
# ~4 FLOPs per token (2 lookups + 1 subtract + 1 compare)
RI_GAP_FLOPS_PER_TOKEN = 4

# SpaCy NER is CPU-bound; we report it separately as "not GPU FLOPs"
# Approximate from profiling: ~500 FLOPs per character for NER
SPACY_NER_FLOPS_PER_CHAR = 500


def estimate_flops(
    n_tokens: int = 100,
    model_params: float = MODEL_PARAMS,
    avg_chars_per_token: float = 4.0,
) -> dict:
    """Estimate FLOPs for each verification method at a given sequence length.

    Parameters
    ----------
    n_tokens : int
        Number of tokens in the sequence being verified.
    model_params : float
        Model parameter count (default: 8B for Llama 3.1-8B).
    avg_chars_per_token : float
        Average characters per token for NER estimation.

    Returns
    -------
    dict
        Per-method FLOPs estimates with metadata and a markdown table.
    """
    n_chars = int(n_tokens * avg_chars_per_token)

    # Cost of one forward pass: ~2 * params * tokens
    forward_pass_flops = 2 * model_params * n_tokens

    # RI gap-only: trivial arithmetic per token
    ri_gap_flops = RI_GAP_FLOPS_PER_TOKEN * n_tokens

    # RI full: gap computation + SpaCy NER
    ri_full_flops = ri_gap_flops + SPACY_NER_FLOPS_PER_CHAR * n_chars

    methods = {
        "ri_gap_only": {
            "label": "RI (gap-only)",
            "flops": ri_gap_flops,
            "extra_forward_passes": 0,
            "notes": "Hash lookup + log subtraction per token",
        },
        "ri_full": {
            "label": "RI (full, incl. NER)",
            "flops": ri_full_flops,
            "extra_forward_passes": 0,
            "notes": "Gap computation + SpaCy NER (CPU-bound)",
        },
    }

    # SE and SelfCheckGPT at various k/N
    for k in [2, 5, 10]:
        nli_pairs = k * (k - 1) // 2
        # NLI model is ~400M params, ~2 * 400M * 128 tokens per pair
        nli_flops = nli_pairs * 2 * 400e6 * 128
        methods[f"se_k{k}"] = {
            "label": f"SE (k={k})",
            "flops": k * forward_pass_flops + nli_flops,
            "extra_forward_passes": k,
            "notes": f"{k} forward passes + {nli_pairs} NLI pairs",
        }

    for n in [2, 5, 10]:
        # BERTScore uses ~110M param model, ~2 * 110M * n_tokens per sample
        consistency_flops = n * 2 * 110e6 * n_tokens
        methods[f"scgpt_n{n}"] = {
            "label": f"SC-GPT (N={n})",
            "flops": n * forward_pass_flops + consistency_flops,
            "extra_forward_passes": n,
            "notes": f"{n} forward passes + consistency scoring",
        }

    # Generate markdown table
    lines = [
        "| Method | Extra Forward Passes | FLOPs/example | Notes |",
        "|--------|---------------------|---------------|-------|",
    ]
    for key in ["ri_gap_only", "ri_full", "se_k2", "se_k5", "se_k10", "scgpt_n2", "scgpt_n5", "scgpt_n10"]:
        m = methods[key]
        flops_str = _format_flops(m["flops"])
        lines.append(f"| {m['label']} | {m['extra_forward_passes']} | {flops_str} | {m['notes']} |")

    methods["markdown_table"] = "\n".join(lines)
    methods["n_tokens"] = n_tokens
    methods["model_params"] = model_params

    return methods


def _format_flops(flops: float) -> str:
    """Format FLOPs count into human-readable string."""
    if flops >= 1e12:
        return f"{flops / 1e12:.1f}T"
    elif flops >= 1e9:
        return f"{flops / 1e9:.1f}G"
    elif flops >= 1e6:
        return f"{flops / 1e6:.1f}M"
    elif flops >= 1e3:
        return f"{flops / 1e3:.1f}K"
    else:
        return str(int(flops))
