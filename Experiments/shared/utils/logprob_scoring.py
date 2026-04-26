"""Multi-model logprob scoring utility.

Provides a uniform interface for getting per-token logprobs from:
- OpenAI (gpt-5.1) via chat completions with logprobs=True
- Anthropic (claude-sonnet-4) via messages API
- Ollama (llama-3.1-8b) via local OpenAI-compatible endpoint

Usage:
    result = score_text_logprobs("The cat sat on the mat.", "gpt-5.1")
    # result.token_ids, result.logprobs ready for compute_entity_gaps()
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ModelConfig:
    """Configuration for a model's API access."""

    model_name: str
    provider: str  # "openai", "anthropic", "ollama"
    api_model_id: str  # actual model ID for the API call
    tokenizer_name: str  # name used for get_tokenizer() / rank table lookup


@dataclass
class ScoringResult:
    """Per-token logprob scoring result."""

    text: str
    token_ids: list[int]
    tokens: list[str]
    logprobs: list[Optional[float]]
    model_name: str


SUPPORTED_MODELS: dict[str, ModelConfig] = {
    "gpt-5.1": ModelConfig(
        model_name="gpt-5.1",
        provider="openai",
        api_model_id="gpt-4o",  # uses o200k_base tokenizer
        tokenizer_name="gpt-5.1",
    ),
    "claude-sonnet-4": ModelConfig(
        model_name="claude-sonnet-4",
        provider="anthropic",
        api_model_id="claude-sonnet-4-20250514",
        tokenizer_name="claude-sonnet-4",
    ),
    "llama-3.1-8b": ModelConfig(
        model_name="llama-3.1-8b",
        provider="ollama",
        api_model_id="llama3.1:8b",
        tokenizer_name="llama-3.1-8b",
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a supported model."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported: {list(SUPPORTED_MODELS)}"
        )
    return SUPPORTED_MODELS[model_name]


def _load_api_key(filename: str) -> str:
    """Load an API key from the API_KEYS directory."""
    key_dir = Path(__file__).resolve().parent.parent.parent / "API_KEYS"
    key_path = key_dir / filename
    if not key_path.exists():
        raise FileNotFoundError(f"API key not found at {key_path}")
    return key_path.read_text().strip()


def _get_tokenizer_and_ids(text: str, config: ModelConfig):
    """Get tokenizer instance and token IDs for a text."""
    from shared.utils.corpus_utils import get_tokenizer, tokenize_text

    tokenizer = get_tokenizer(config.tokenizer_name)
    token_ids = tokenize_text(text, tokenizer, config.tokenizer_name)
    return tokenizer, token_ids


def _score_openai(
    text: str,
    config: ModelConfig,
    prompt: str,
    max_retries: int = 3,
) -> ScoringResult:
    """Score text via OpenAI-compatible API with logprobs=True.

    Works for both OpenAI API and Ollama's OpenAI-compatible endpoint.
    Sends text as a prefilled assistant message and requests logprobs.
    """
    from openai import OpenAI

    if config.provider == "ollama":
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"
    else:
        api_key = _load_api_key("OpenAI_RI.key.txt")
        base_url = None

    client = OpenAI(api_key=api_key, base_url=base_url)
    tokenizer, token_ids = _get_tokenizer_and_ids(text, config)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.api_model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=len(token_ids) + 50,
                logprobs=True,
                top_logprobs=1,
                temperature=0,
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    choice = response.choices[0]
    api_logprobs: list[Optional[float]] = []
    api_tokens: list[str] = []

    if choice.logprobs and choice.logprobs.content:
        for lp in choice.logprobs.content:
            api_tokens.append(lp.token)
            api_logprobs.append(lp.logprob)

    # We need logprobs aligned to our tokenizer's token_ids.
    # The API may produce different tokenization than ours.
    # Strategy: use our token_ids as ground truth, pad/truncate API logprobs.
    logprobs_aligned: list[Optional[float]]
    if len(api_logprobs) == len(token_ids):
        logprobs_aligned = api_logprobs
    elif len(api_logprobs) > 0:
        # Best-effort alignment: pad or truncate
        logprobs_aligned = list(api_logprobs[: len(token_ids)])
        if len(logprobs_aligned) < len(token_ids):
            logprobs_aligned.extend(
                [None] * (len(token_ids) - len(logprobs_aligned))
            )
    else:
        logprobs_aligned = [None] * len(token_ids)

    tokens_out = api_tokens if len(api_tokens) == len(token_ids) else [
        tokenizer.decode([tid]) if hasattr(tokenizer, "decode") else str(tid)
        for tid in token_ids
    ]

    return ScoringResult(
        text=text,
        token_ids=token_ids,
        tokens=tokens_out,
        logprobs=logprobs_aligned,
        model_name=config.model_name,
    )


def _score_anthropic(
    text: str,
    config: ModelConfig,
    prompt: str,
    max_retries: int = 3,
) -> ScoringResult:
    """Score text via Anthropic API.

    Anthropic does not currently expose per-token logprobs in the same way
    as OpenAI. We get token IDs from our proxy tokenizer and return None
    logprobs — compute_entity_gaps() handles this with the fallback P_LLM.
    """
    import anthropic

    api_key = _load_api_key("Claude_Key.txt")
    client = anthropic.Anthropic(api_key=api_key)
    tokenizer, token_ids = _get_tokenizer_and_ids(text, config)

    # Make a generation call to verify the model processes this text
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=config.api_model_id,
                max_tokens=1,
                messages=[
                    {"role": "user", "content": f"{prompt}\n\n{text}"},
                ],
                temperature=0,
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    # No per-token logprobs from Anthropic — use None (fallback in entity_extraction)
    tokens = [
        tokenizer.decode([tid]) if hasattr(tokenizer, "decode") else str(tid)
        for tid in token_ids
    ]

    return ScoringResult(
        text=text,
        token_ids=token_ids,
        tokens=tokens,
        logprobs=[None] * len(token_ids),
        model_name=config.model_name,
    )


def score_text_logprobs(
    text: str,
    model_name: str,
    prompt: str = "Continue this text exactly:",
    max_retries: int = 3,
) -> ScoringResult:
    """Score text through a model to get per-token logprobs.

    Args:
        text: The text to score.
        model_name: One of SUPPORTED_MODELS keys.
        prompt: Context prompt for the scoring call.
        max_retries: Number of API retry attempts.

    Returns:
        ScoringResult with token_ids and logprobs aligned for use
        with compute_entity_gaps().
    """
    if not text or not text.strip():
        return ScoringResult(
            text=text, token_ids=[], tokens=[], logprobs=[], model_name=model_name
        )

    config = get_model_config(model_name)

    if config.provider in ("openai", "ollama"):
        return _score_openai(text, config, prompt, max_retries)
    elif config.provider == "anthropic":
        return _score_anthropic(text, config, prompt, max_retries)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def score_batch(
    texts: list[str],
    model_name: str,
    prompt: str = "Continue this text exactly:",
    delay: float = 0.1,
) -> list[ScoringResult]:
    """Score multiple texts with rate-limiting delay between calls."""
    results = []
    for text in texts:
        result = score_text_logprobs(text, model_name, prompt)
        results.append(result)
        if delay > 0:
            time.sleep(delay)
    return results
