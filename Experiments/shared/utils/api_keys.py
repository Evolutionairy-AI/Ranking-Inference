"""
API key loader. Reads keys from API_KEYS/ directory.
"""

import os
from pathlib import Path

API_KEYS_DIR = Path(__file__).resolve().parent.parent.parent / "API_KEYS"


def load_api_keys():
    """Load all API keys from API_KEYS/*.key.txt files and set as env vars."""
    if not API_KEYS_DIR.exists():
        return

    for key_file in API_KEYS_DIR.glob("*.txt"):
        key_value = key_file.read_text().strip()
        name = key_file.stem  # e.g. "OpenAI_RI"

        if "openai" in name.lower():
            os.environ.setdefault("OPENAI_API_KEY", key_value)
        elif "anthropic" in name.lower() or "claude" in name.lower():
            os.environ.setdefault("ANTHROPIC_API_KEY", key_value)
        elif "together" in name.lower() or "llama" in name.lower():
            os.environ.setdefault("TOGETHER_API_KEY", key_value)
        elif "gemini" in name.lower() or "google" in name.lower():
            os.environ.setdefault("GOOGLE_API_KEY", key_value)
            os.environ.setdefault("GEMINI_API_KEY", key_value)
        elif "mistral" in name.lower():
            os.environ.setdefault("MISTRAL_API_KEY", key_value)
