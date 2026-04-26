"""
Step 1 (Exp03): Generate outputs with token-level logprobs from Ollama.

Uses Ollama's API to get per-token log probabilities for computing
P_LLM(t) in the confidence-grounding gap delta(t) = P_LLM(t) - G_RI(t).
"""

import json
import yaml
import requests
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = Path(__file__).resolve().parent.parent
PROMPTS_PATH = EXP_DIR / "data" / "prompts" / "prompts.yaml"
OUTPUT_DIR = EXP_DIR / "data" / "outputs"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_ID = "llama3.1:8b"


def generate_with_logprobs(prompt: str, temperature: float = 0.7) -> dict:
    """Generate text with per-token logprobs via Ollama raw API.

    Ollama's /api/generate endpoint returns token-level data when streaming.
    We collect all tokens and their logprobs.
    """
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,
        },
    }

    tokens = []
    full_text = ""

    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)

        if chunk.get("response"):
            token_text = chunk["response"]
            full_text += token_text

            token_entry = {
                "token": token_text,
                "logprob": None,  # Ollama doesn't expose logprobs in basic mode
            }
            tokens.append(token_entry)

        if chunk.get("done"):
            break

    return {
        "text": full_text,
        "tokens": tokens,
        "n_tokens": len(tokens),
    }


def generate_with_logprobs_chat(prompt: str, temperature: float = 0.7) -> dict:
    """Generate via Ollama OpenAI-compatible chat endpoint with logprobs."""
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024,
        logprobs=True,
        top_logprobs=10,
    )

    choice = response.choices[0]
    tokens = []

    if choice.logprobs and choice.logprobs.content:
        for lp in choice.logprobs.content:
            tokens.append({
                "token": lp.token,
                "logprob": lp.logprob,
                "prob": 2 ** lp.logprob if lp.logprob else 0,  # convert to probability
                "top_logprobs": [
                    {"token": t.token, "logprob": t.logprob}
                    for t in (lp.top_logprobs or [])
                ],
            })
    else:
        # Fallback: no logprobs available, just tokenize the output
        for ch in (choice.message.content or ""):
            tokens.append({"token": ch, "logprob": None, "prob": None})

    return {
        "text": choice.message.content or "",
        "tokens": tokens,
        "n_tokens": len(tokens),
        "has_logprobs": bool(choice.logprobs and choice.logprobs.content),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROMPTS_PATH) as f:
        prompts = yaml.safe_load(f)

    # Test logprobs availability
    print("Testing logprobs availability...")
    test = generate_with_logprobs_chat("Say hello.", temperature=0.0)
    has_logprobs = test.get("has_logprobs", False)
    print(f"  Logprobs available: {has_logprobs}")
    if has_logprobs and test["tokens"]:
        print(f"  Sample: token='{test['tokens'][0]['token']}', logprob={test['tokens'][0]['logprob']}")

    conditions = {
        "factual": prompts["factual"],
        "hallucination": prompts["hallucination_inducing"],
        "synthetic_base": prompts["synthetic_base"],
    }

    all_outputs = {}

    for condition, condition_prompts in conditions.items():
        print(f"\n=== Condition: {condition} ({len(condition_prompts)} prompts) ===")
        outputs = []

        for i, prompt in enumerate(tqdm(condition_prompts, desc=condition)):
            try:
                result = generate_with_logprobs_chat(prompt, temperature=0.7)
                output = {
                    "condition": condition,
                    "prompt_index": i,
                    "prompt": prompt,
                    "text": result["text"],
                    "tokens": result["tokens"],
                    "n_tokens": result["n_tokens"],
                    "has_logprobs": result.get("has_logprobs", False),
                    "timestamp": datetime.now().isoformat(),
                }
                outputs.append(output)
            except Exception as e:
                print(f"  Error on {condition}[{i}]: {e}")
                outputs.append({
                    "condition": condition,
                    "prompt_index": i,
                    "prompt": prompt,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })

        all_outputs[condition] = outputs

        # Checkpoint after each condition
        checkpoint_path = OUTPUT_DIR / f"{condition}_outputs.json"
        with open(checkpoint_path, "w") as f:
            json.dump(outputs, f, indent=2)
        print(f"  Saved {len(outputs)} outputs to {checkpoint_path}")

    # Save combined
    combined_path = OUTPUT_DIR / "all_outputs.json"
    with open(combined_path, "w") as f:
        json.dump(all_outputs, f, indent=2)

    # Summary
    for cond, outputs in all_outputs.items():
        ok = sum(1 for o in outputs if o.get("text"))
        with_lp = sum(1 for o in outputs if o.get("has_logprobs"))
        print(f"{cond}: {ok} outputs, {with_lp} with logprobs")


if __name__ == "__main__":
    main()
