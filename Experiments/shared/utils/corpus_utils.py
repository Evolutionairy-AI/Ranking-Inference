"""
Corpus loading and processing utilities.

Handles downloading/loading reference corpora for the 5 target domains
and tokenizing them with different tokenizers.
"""

import tiktoken
from pathlib import Path
from typing import Generator
from datasets import load_dataset
from tqdm import tqdm


# Domain corpus configurations
DOMAIN_CORPORA = {
    "news": {
        "dataset": "cc_news",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "max_docs": 10000,
        "description": "CC-News: Common Crawl news articles",
    },
    "biomedical": {
        "dataset": "pubmed_qa",
        "subset": "pqa_artificial",
        "split": "train",
        "text_field": "long_answer",
        "max_docs": 10000,
        "description": "PubMed QA: biomedical abstracts and answers",
    },
    "legal": {
        "dataset": "nguha/legalbench",
        "subset": "contract_nli_explicit_identification",
        "split": "test",
        "text_field": "text",
        "max_docs": 10000,
        "description": "LegalBench: legal contract text",
    },
    "code": {
        "dataset": "sahil2801/CodeAlpaca-20k",
        "subset": None,
        "split": "train",
        "text_field": "output",
        "max_docs": 10000,
        "description": "CodeAlpaca: code instruction-response pairs",
    },
    "social_media": {
        "dataset": "mteb/tweet_sentiment_extraction",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "max_docs": 10000,
        "description": "Tweet sentiment: social media text",
    },
}

# Wikipedia as global reference corpus
REFERENCE_CORPUS = {
    "dataset": "wikimedia/wikipedia",
    "subset": "20231101.en",
    "split": "train",
    "text_field": "text",
    "max_docs": 50000,
    "description": "English Wikipedia (Nov 2023 dump)",
}


def get_tokenizer(name: str):
    """Get a tokenizer by name.

    Supported:
        - "gpt-5.1": OpenAI GPT-5.1 tokenizer (o200k_base or latest)
        - "claude-sonnet-4-6": Claude tokenizer (via anthropic)
        - "llama-3.1": Llama 3.1 tokenizer (via transformers)
    """
    if "gpt" in name.lower():
        # GPT-5.1 likely uses o200k_base (GPT-4o family)
        # Fall back to cl100k_base if o200k not available
        try:
            return tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    elif "claude" in name.lower():
        # Claude doesn't expose its tokenizer publicly.
        # Use cl100k_base as a reasonable BPE proxy for frequency analysis.
        # The BPE invariance argument (Section 7.3) predicts the power-law
        # form is preserved across BPE tokenizers, so this is acceptable
        # for rank-frequency analysis.
        return tiktoken.get_encoding("cl100k_base")
    elif "llama" in name.lower() or "gemini" in name.lower() or "mistral" in name.lower() or "qwen" in name.lower():
        # Llama/Gemini/Mistral/Qwen: analyse tokens via Llama 3.1 8B BPE as
        # the apples-to-apples analysis tokenizer (see paper Section 7.3 BPE
        # invariance argument). The HF repo is gated, so fall back to
        # tiktoken's o200k_base if transformers can't load it.
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                trust_remote_code=True,
            )
        except (OSError, ImportError):
            return tiktoken.get_encoding("o200k_base")
    else:
        raise ValueError(f"Unknown tokenizer: {name}")


def tokenize_text(text: str, tokenizer, tokenizer_name: str) -> list[int]:
    """Tokenize text using the given tokenizer, returning token IDs."""
    if hasattr(tokenizer, "encode_ordinary"):
        # tiktoken interface
        return tokenizer.encode(text)
    else:
        # HuggingFace transformers interface
        return tokenizer.encode(text, add_special_tokens=False)


def load_corpus_texts(
    corpus_config: dict,
    max_docs: int | None = None,
) -> Generator[str, None, None]:
    """Stream texts from a HuggingFace dataset.

    Yields individual document texts up to max_docs.
    """
    max_docs = max_docs or corpus_config.get("max_docs", 10000)

    ds = load_dataset(
        corpus_config["dataset"],
        corpus_config.get("subset"),
        split=corpus_config["split"],
        streaming=True,
    )

    count = 0
    for example in ds:
        text = example.get(corpus_config["text_field"], "")
        if text and len(text.strip()) > 50:  # skip very short docs
            yield text.strip()
            count += 1
            if count >= max_docs:
                break


def tokenize_corpus(
    corpus_config: dict,
    tokenizer,
    tokenizer_name: str,
    max_docs: int | None = None,
    progress: bool = True,
) -> list[int]:
    """Load and tokenize an entire corpus, returning flat list of token IDs."""
    all_tokens = []
    max_docs = max_docs or corpus_config.get("max_docs", 10000)

    texts = load_corpus_texts(corpus_config, max_docs)
    if progress:
        texts = tqdm(texts, total=max_docs, desc=f"Tokenizing {corpus_config.get('description', '')}")

    for text in texts:
        tokens = tokenize_text(text, tokenizer, tokenizer_name)
        all_tokens.extend(tokens)

    return all_tokens
