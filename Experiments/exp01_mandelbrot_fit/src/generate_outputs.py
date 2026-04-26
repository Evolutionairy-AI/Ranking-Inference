"""
Step 1: Generate LLM outputs for rank-frequency analysis.

Calls GPT-5.1, Claude 4.6 Sonnet, and Llama 3.3 with curated prompts
across 5 domains. Saves raw outputs for downstream frequency analysis.
"""

import json
import yaml
import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load API keys from API_KEYS/ directory
from shared.utils.api_keys import load_api_keys
load_api_keys()

EXP_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = EXP_DIR / "config" / "experiment_config.yaml"
PROMPTS_PATH = EXP_DIR / "data" / "prompts" / "prompts.yaml"
OUTPUT_DIR = EXP_DIR / "data" / "outputs"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_prompts():
    with open(PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def generate_openai(model_id: str, prompt: str, api_params: dict) -> dict:
    """Generate output via OpenAI API."""
    from openai import OpenAI
    client = OpenAI()

    # GPT-5.x+ uses max_completion_tokens; older models use max_tokens
    max_tok_param = "max_completion_tokens" if "5." in model_id else "max_tokens"
    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": api_params.get("temperature", 0.7),
        max_tok_param: api_params.get("max_tokens", 1500),
        "logprobs": api_params.get("logprobs", False),
    }
    if api_params.get("logprobs"):
        kwargs["top_logprobs"] = api_params.get("top_logprobs", 5)

    response = client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    result = {
        "text": choice.message.content,
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }

    # Extract logprobs if available
    if choice.logprobs and choice.logprobs.content:
        result["logprobs"] = [
            {
                "token": lp.token,
                "logprob": lp.logprob,
                "top_logprobs": [
                    {"token": t.token, "logprob": t.logprob}
                    for t in (lp.top_logprobs or [])
                ],
            }
            for lp in choice.logprobs.content
        ]

    return result


def generate_anthropic(model_id: str, prompt: str, api_params: dict) -> dict:
    """Generate output via Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model_id,
        max_tokens=api_params.get("max_tokens", 1500),
        messages=[{"role": "user", "content": prompt}],
        temperature=api_params.get("temperature", 0.7),
    )

    return {
        "text": response.content[0].text,
        "finish_reason": response.stop_reason,
        "usage": {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        },
    }


def generate_ollama(model_id: str, prompt: str, api_params: dict) -> dict:
    """Generate output via local Ollama (OpenAI-compatible API).

    Ollama serves at http://localhost:11434/v1 by default. Works for any
    model pulled via `ollama pull <name>` (llama3.1:8b, qwen2.5:7b, ...).
    """
    from openai import OpenAI

    base_url = os.environ.get("LLAMA_API_BASE", "http://localhost:11434/v1")
    client = OpenAI(base_url=base_url, api_key="ollama")

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=api_params.get("temperature", 0.7),
        max_tokens=api_params.get("max_tokens", 1500),
    )

    choice = response.choices[0]
    return {
        "text": choice.message.content,
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
        },
    }


# Backward-compat alias
generate_llama = generate_ollama


def generate_gemini(model_id: str, prompt: str, api_params: dict) -> dict:
    """Generate output via Google Gemini API.

    Gemini 2.5 Pro is a reasoning model: hidden thinking tokens consume the
    output budget, so max_output_tokens must be generous (default 8000).
    Transient 503 UNAVAILABLE is common; retry with exponential backoff.
    """
    import time
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set")
    client = genai.Client(api_key=api_key)

    max_out = api_params.get("max_tokens", 8000)
    temperature = api_params.get("temperature", 0.7)

    last_err = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_out,
                ),
            )
            break
        except genai_errors.APIError as e:
            last_err = e
            if attempt >= 4:
                raise
            wait = 5 * (2 ** attempt)
            print(f"  Gemini error (attempt {attempt+1}): {e}; retry in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Gemini failed after 5 attempts: {last_err}")

    text = getattr(response, "text", "") or ""
    usage = getattr(response, "usage_metadata", None)
    return {
        "text": text,
        "finish_reason": str(getattr(response.candidates[0], "finish_reason", "stop"))
            if response.candidates else "stop",
        "usage": {
            "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "completion_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        },
    }


def generate_mistral(model_id: str, prompt: str, api_params: dict) -> dict:
    """Generate output via Mistral API (mistralai >=1.0).

    SDK quirk: in v2.3.2, Mistral class lives at `mistralai.client.Mistral`,
    not at top-level `mistralai`.
    """
    try:
        from mistralai.client import Mistral
    except ImportError:
        from mistralai import Mistral

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    client = Mistral(api_key=api_key)

    response = client.chat.complete(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=api_params.get("temperature", 0.7),
        max_tokens=api_params.get("max_tokens", 1500),
    )

    choice = response.choices[0]
    return {
        "text": choice.message.content,
        "finish_reason": str(choice.finish_reason),
        "usage": {
            "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
        },
    }


GENERATORS = {
    "openai": generate_openai,
    "anthropic": generate_anthropic,
    "ollama": generate_ollama,
    "google": generate_gemini,
    "gemini": generate_gemini,
    "mistral": generate_mistral,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Mandelbrot fit outputs")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model names to generate (default: all in config)")
    parser.add_argument("--skip", default=None,
                        help="Comma-separated model names to skip")
    args = parser.parse_args()

    config = load_config()
    prompts = load_prompts()

    include = set(m.strip() for m in args.models.split(",")) if args.models else None
    skip = set(m.strip() for m in args.skip.split(",")) if args.skip else set()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_config in config["models"]:
        if include is not None and model_config["name"] not in include:
            print(f"Skipping {model_config['name']} (not in --models filter)")
            continue
        if model_config["name"] in skip:
            print(f"Skipping {model_config['name']} (in --skip list)")
            continue
        model_name = model_config["name"]
        provider = model_config["provider"]
        model_id = model_config["model_id"]
        api_params = model_config.get("api_params", {})

        # Choose generator by provider; ollama handles any locally-served model
        if provider in GENERATORS:
            generator = GENERATORS[provider]
        elif "llama" in model_name.lower() or "qwen" in model_name.lower():
            generator = generate_ollama
        else:
            raise ValueError(f"Unknown provider '{provider}' for model '{model_name}'")

        model_output_dir = OUTPUT_DIR / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing outputs to enable resumption
        existing_file = model_output_dir / "all_outputs.json"
        existing_outputs = {}
        if existing_file.exists():
            with open(existing_file) as f:
                existing_data = json.load(f)
            for item in existing_data.get("outputs", []):
                key = f"{item['domain']}_{item['prompt_index']}"
                existing_outputs[key] = item

        all_outputs = []
        total_tokens = 0

        for domain in config["generation"]["domains"]:
            domain_prompts = prompts.get(domain, [])
            n_prompts = min(
                config["generation"]["prompts_per_domain"],
                len(domain_prompts),
            )

            print(f"\n--- {model_name} / {domain} ({n_prompts} prompts) ---")

            for i, prompt in enumerate(tqdm(domain_prompts[:n_prompts], desc=domain)):
                key = f"{domain}_{i}"

                # Skip if already generated
                if key in existing_outputs:
                    all_outputs.append(existing_outputs[key])
                    total_tokens += existing_outputs[key]["usage"]["completion_tokens"]
                    continue

                try:
                    result = generator(model_id, prompt, api_params)
                    output = {
                        "domain": domain,
                        "prompt_index": i,
                        "prompt": prompt,
                        "model": model_name,
                        "timestamp": datetime.now().isoformat(),
                        **result,
                    }
                    all_outputs.append(output)
                    total_tokens += result["usage"]["completion_tokens"]
                except Exception as e:
                    print(f"  Error on {domain}[{i}]: {e}")
                    all_outputs.append({
                        "domain": domain,
                        "prompt_index": i,
                        "prompt": prompt,
                        "model": model_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    })

            # Save after each domain (checkpoint)
            with open(existing_file, "w") as f:
                json.dump({
                    "model": model_name,
                    "model_id": model_id,
                    "total_outputs": len(all_outputs),
                    "total_completion_tokens": total_tokens,
                    "generated_at": datetime.now().isoformat(),
                    "outputs": all_outputs,
                }, f, indent=2)

        print(f"\n{model_name}: {len(all_outputs)} outputs, ~{total_tokens} tokens")
        print(f"Saved to {existing_file}")


if __name__ == "__main__":
    main()
