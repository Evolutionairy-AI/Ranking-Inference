"""
Experiment 3b: Token-level entity analysis of the confidence-grounding gap.

Instead of averaging delta(t) across all tokens (which mixes signal with noise),
this analysis:
1. Identifies "entity tokens" — capitalized multi-token sequences (names, terms)
2. Computes gap statistics specifically at entity tokens
3. Compares entity-level gap between factual and hallucination conditions

The hypothesis: fabricated entity names (e.g., "Nexorvatin", "Dr. Helena Marchetti")
should have LOWER grounding scores G_RI than real entity names (e.g., "DNA", "Newton"),
producing a distinctive gap signature at entity tokens specifically.

Key insight from Exp 3 results: the raw gap P_LLM - G_RI was higher for factual text
because the model is more confident on well-known topics. The refined signal should
focus on G_RI alone (grounding score) or the ratio P_LLM / G_RI at entity positions.
"""

import json
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from collections import defaultdict

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

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def load_grounding():
    """Load Wikipedia rank table and compute grounding scores."""
    rt = RankTable.load(RANK_TABLE_DIR / "wikipedia_gpt-5.1.json")

    max_rank = rt.vocab_size
    ranks = np.arange(1, max_rank + 1)
    freq = rt.rank_to_freq[1:max_rank + 1].copy()
    mask = freq > 0
    params = fit_mandelbrot_mle(ranks[mask], freq[mask])

    all_ranks = np.arange(1, max_rank + 2)
    p_ri = mandelbrot_freq(all_ranks, params.C, params.q, params.s)
    z_ri = p_ri.sum()
    g_ri = p_ri / z_ri

    grounding = {}
    for token_id, rank in rt.token_to_rank.items():
        if rank <= max_rank:
            grounding[token_id] = float(g_ri[rank - 1])

    default = float(g_ri[-1])
    return grounding, default, rt


def extract_entity_spans(tokens_data: list) -> list:
    """Identify entity-like token spans from logprob token sequence.

    Heuristic: sequences of tokens that start with a capital letter or
    contain unusual character patterns (likely proper nouns, technical terms).
    """
    entities = []
    current_entity = []
    current_start = None

    for i, td in enumerate(tokens_data):
        token = td.get("token", "")
        stripped = token.strip()

        # Is this token entity-like?
        is_entity_token = (
            stripped and
            len(stripped) > 1 and
            (stripped[0].isupper() or  # Capitalized
             stripped.isalpha() and not stripped.islower() or  # MixedCase
             any(c.isdigit() for c in stripped) and any(c.isalpha() for c in stripped))  # Alphanumeric
        )

        # Also catch tokens that are continuations (no leading space, follows an entity)
        is_continuation = (
            current_entity and
            stripped and
            not token.startswith(" ") and
            stripped.isalpha()
        )

        if is_entity_token or is_continuation:
            if not current_entity:
                current_start = i
            current_entity.append(i)
        else:
            if len(current_entity) >= 1:
                entity_text = "".join(
                    tokens_data[j]["token"] for j in current_entity
                ).strip()
                # Filter: keep only meaningful entities (not just "The", "A", etc.)
                if len(entity_text) > 2 and not entity_text.lower() in {
                    "the", "this", "that", "these", "those", "here", "there",
                    "when", "where", "which", "what", "who", "how", "its",
                    "has", "had", "have", "was", "were", "are", "been",
                    "also", "can", "may", "will", "would", "could", "should",
                    "not", "but", "and", "for", "with", "from", "into",
                }:
                    entities.append({
                        "text": entity_text,
                        "token_indices": current_entity,
                        "start": current_start,
                    })
            current_entity = []
            current_start = None

    # Flush last entity
    if len(current_entity) >= 1:
        entity_text = "".join(
            tokens_data[j]["token"] for j in current_entity
        ).strip()
        if len(entity_text) > 2:
            entities.append({
                "text": entity_text,
                "token_indices": current_entity,
                "start": current_start,
            })

    return entities


def analyze_output(
    output: dict,
    tokenizer,
    grounding: dict,
    default_grounding: float,
) -> dict:
    """Analyze a single output: compute per-token and per-entity gap metrics."""
    tokens_data = output.get("tokens", [])
    if not tokens_data or not output.get("has_logprobs"):
        return None

    # Compute per-token grounding and gap
    for td in tokens_data:
        tok_ids = tokenize_text(td["token"], tokenizer, "gpt-5.1")
        if tok_ids:
            g = grounding.get(tok_ids[0], default_grounding)
        else:
            g = default_grounding
        td["g_ri"] = g
        td["p_llm"] = np.exp(td["logprob"]) if td.get("logprob") is not None else 0
        td["delta"] = td["p_llm"] - g
        # Log ratio: how much more confident is the model than the baseline?
        td["log_ratio"] = np.log2(max(td["p_llm"], 1e-20) / max(g, 1e-20))
        # Grounding deficit: low G_RI means distributionally unusual
        td["grounding_deficit"] = -np.log2(max(g, 1e-20))

    # Extract entities
    entities = extract_entity_spans(tokens_data)

    entity_results = []
    for ent in entities:
        indices = ent["token_indices"]
        g_values = [tokens_data[i]["g_ri"] for i in indices]
        p_values = [tokens_data[i]["p_llm"] for i in indices]
        delta_values = [tokens_data[i]["delta"] for i in indices]
        log_ratio_values = [tokens_data[i]["log_ratio"] for i in indices]
        deficit_values = [tokens_data[i]["grounding_deficit"] for i in indices]

        entity_results.append({
            "text": ent["text"],
            "n_tokens": len(indices),
            "mean_g_ri": float(np.mean(g_values)),
            "min_g_ri": float(np.min(g_values)),
            "mean_p_llm": float(np.mean(p_values)),
            "mean_delta": float(np.mean(delta_values)),
            "mean_log_ratio": float(np.mean(log_ratio_values)),
            "mean_grounding_deficit": float(np.mean(deficit_values)),
        })

    # Aggregate metrics
    all_g = [td["g_ri"] for td in tokens_data if td.get("logprob") is not None]
    all_deficit = [td["grounding_deficit"] for td in tokens_data if td.get("logprob") is not None]
    all_log_ratio = [td["log_ratio"] for td in tokens_data if td.get("logprob") is not None]

    return {
        "condition": output["condition"],
        "prompt_index": output["prompt_index"],
        "n_tokens": len(tokens_data),
        "n_entities": len(entity_results),
        # Whole-output grounding metrics
        "mean_grounding": float(np.mean(all_g)) if all_g else 0,
        "mean_grounding_deficit": float(np.mean(all_deficit)) if all_deficit else 0,
        "mean_log_ratio": float(np.mean(all_log_ratio)) if all_log_ratio else 0,
        # Entity-level metrics
        "entities": entity_results,
        "entity_mean_grounding": float(np.mean([e["mean_g_ri"] for e in entity_results])) if entity_results else 0,
        "entity_mean_deficit": float(np.mean([e["mean_grounding_deficit"] for e in entity_results])) if entity_results else 0,
        "entity_mean_log_ratio": float(np.mean([e["mean_log_ratio"] for e in entity_results])) if entity_results else 0,
    }


def plot_entity_grounding_comparison(factual_results, halluc_results):
    """Plot: entity-level grounding scores, factual vs hallucination."""
    # Collect all entity grounding values
    factual_entity_g = []
    halluc_entity_g = []
    factual_entity_deficit = []
    halluc_entity_deficit = []

    for r in factual_results:
        for e in r["entities"]:
            factual_entity_g.append(e["mean_g_ri"])
            factual_entity_deficit.append(e["mean_grounding_deficit"])
    for r in halluc_results:
        for e in r["entities"]:
            halluc_entity_g.append(e["mean_g_ri"])
            halluc_entity_deficit.append(e["mean_grounding_deficit"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Entity grounding score distributions
    axes[0].hist(factual_entity_g, bins=40, alpha=0.5, density=True,
                 color="#059669", label=f"Factual (n={len(factual_entity_g)})",
                 edgecolor="black", linewidth=0.3)
    axes[0].hist(halluc_entity_g, bins=40, alpha=0.5, density=True,
                 color="#DC2626", label=f"Hallucination (n={len(halluc_entity_g)})",
                 edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("Entity grounding score G_RI")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Entity Grounding Scores")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # Plot 2: Grounding deficit (higher = less grounded)
    axes[1].hist(factual_entity_deficit, bins=40, alpha=0.5, density=True,
                 color="#059669", label="Factual", edgecolor="black", linewidth=0.3)
    axes[1].hist(halluc_entity_deficit, bins=40, alpha=0.5, density=True,
                 color="#DC2626", label="Hallucination", edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Grounding deficit -log2(G_RI) (bits)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Entity Grounding Deficit\n(higher = less grounded)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    # Plot 3: Output-level entity mean grounding
    f_means = [r["entity_mean_grounding"] for r in factual_results if r["entities"]]
    h_means = [r["entity_mean_grounding"] for r in halluc_results if r["entities"]]

    if f_means and h_means:
        labels = np.concatenate([np.zeros(len(f_means)), np.ones(len(h_means))])
        # Use deficit (inverted grounding) for AUROC — higher deficit = hallucination
        f_deficits = [r["entity_mean_deficit"] for r in factual_results if r["entities"]]
        h_deficits = [r["entity_mean_deficit"] for r in halluc_results if r["entities"]]
        scores = np.concatenate([f_deficits, h_deficits])
        auroc = roc_auc_score(labels, scores)

        fpr, tpr, _ = roc_curve(labels, scores)
        axes[2].plot(fpr, tpr, color="#7C3AED", linewidth=2,
                     label=f"Entity deficit (AUC={auroc:.3f})")

        # Also try whole-output grounding deficit
        f_whole = [r["mean_grounding_deficit"] for r in factual_results]
        h_whole = [r["mean_grounding_deficit"] for r in halluc_results]
        labels_whole = np.concatenate([np.zeros(len(f_whole)), np.ones(len(h_whole))])
        scores_whole = np.concatenate([f_whole, h_whole])
        auroc_whole = roc_auc_score(labels_whole, scores_whole)
        fpr_w, tpr_w, _ = roc_curve(labels_whole, scores_whole)
        axes[2].plot(fpr_w, tpr_w, color="#D97706", linewidth=2,
                     label=f"Whole-output deficit (AUC={auroc_whole:.3f})")

        axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        axes[2].set_xlabel("False Positive Rate")
        axes[2].set_ylabel("True Positive Rate")
        axes[2].set_title("ROC: Grounding Deficit\nas Hallucination Signal")
        axes[2].legend(loc="lower right")
        axes[2].grid(True, alpha=0.2)

    fig.suptitle(
        "Experiment 3b: Entity-Level Grounding Analysis",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "entity_grounding_analysis.png")
    fig.savefig(RESULTS_DIR / "entity_grounding_analysis.pdf")
    print("Saved: entity_grounding_analysis.png/pdf")
    plt.close()

    return auroc if (f_means and h_means) else None


def plot_top_entities(factual_results, halluc_results):
    """Plot: top entities by grounding deficit from each condition."""
    all_entities = []
    for r in factual_results:
        for e in r["entities"]:
            all_entities.append({**e, "condition": "factual"})
    for r in halluc_results:
        for e in r["entities"]:
            all_entities.append({**e, "condition": "hallucination"})

    # Sort by grounding deficit (least grounded first)
    all_entities.sort(key=lambda x: -x["mean_grounding_deficit"])

    # Top 20 from each condition
    top_factual = [e for e in all_entities if e["condition"] == "factual"][:15]
    top_halluc = [e for e in all_entities if e["condition"] == "hallucination"][:15]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, entities, title, color in [
        (axes[0], top_halluc, "Least Grounded (Hallucination)", "#DC2626"),
        (axes[1], top_factual, "Least Grounded (Factual)", "#059669"),
    ]:
        names = [e["text"][:25] for e in entities]
        deficits = [e["mean_grounding_deficit"] for e in entities]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, deficits, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Grounding deficit -log2(G_RI) (bits)")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle("Top Entities by Grounding Deficit", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "top_entities_by_deficit.png")
    fig.savefig(RESULTS_DIR / "top_entities_by_deficit.pdf")
    print("Saved: top_entities_by_deficit.png/pdf")
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading grounding scores...")
    grounding, default_g, rt = load_grounding()
    tokenizer = get_tokenizer("gpt-5.1")

    print("Loading outputs...")
    with open(OUTPUT_DIR / "all_outputs.json") as f:
        all_outputs = json.load(f)

    factual_results = []
    halluc_results = []

    for condition in ["factual", "hallucination"]:
        print(f"\nAnalyzing {condition}...")
        for output in all_outputs.get(condition, []):
            if output.get("error"):
                continue
            result = analyze_output(output, tokenizer, grounding, default_g)
            if result:
                if condition == "factual":
                    factual_results.append(result)
                else:
                    halluc_results.append(result)

    print(f"\nFactual: {len(factual_results)} outputs, "
          f"{sum(r['n_entities'] for r in factual_results)} entities")
    print(f"Hallucination: {len(halluc_results)} outputs, "
          f"{sum(r['n_entities'] for r in halluc_results)} entities")

    # Statistical comparison
    print("\n=== Output-Level Metrics ===")
    for metric in ["mean_grounding", "mean_grounding_deficit", "entity_mean_deficit", "mean_log_ratio"]:
        f_vals = [r[metric] for r in factual_results]
        h_vals = [r[metric] for r in halluc_results]
        if f_vals and h_vals:
            u_stat, u_p = stats.mannwhitneyu(f_vals, h_vals, alternative="two-sided")
            pooled = np.sqrt((np.std(f_vals)**2 + np.std(h_vals)**2) / 2)
            d = (np.mean(h_vals) - np.mean(f_vals)) / pooled if pooled > 0 else 0
            print(f"  {metric}:")
            print(f"    Factual={np.mean(f_vals):.6f}, Halluc={np.mean(h_vals):.6f}")
            print(f"    Cohen's d={d:.4f}, MWU p={u_p:.6f}")

    # Plots
    print("\nGenerating plots...")
    auroc = plot_entity_grounding_comparison(factual_results, halluc_results)
    plot_top_entities(factual_results, halluc_results)

    # Save detailed results
    summary = {
        "factual": [{k: v for k, v in r.items() if k != "entities"} for r in factual_results],
        "hallucination": [{k: v for k, v in r.items() if k != "entities"} for r in halluc_results],
    }
    with open(RESULTS_DIR / "entity_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if auroc:
        print(f"\nEntity-level grounding deficit AUC-ROC: {auroc:.4f}")


if __name__ == "__main__":
    main()
