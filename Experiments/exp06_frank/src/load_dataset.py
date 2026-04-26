"""FRANK dataset loading with error-type to RI taxonomy mapping.

Loads the FRANK benchmark (Pagnoni et al., 2021) from the artidoro/frank
GitHub repository. FRANK provides 2,246 summarization examples with
sentence-level error annotations from 3 annotators across 6 error types
that map directly to the Ranking Inference taxonomy tiers.

Data source: human_annotations_sentence.json from artidoro/frank.
Each summary sentence has per-annotator error type labels.
We use majority vote (2/3 annotators agree) for error assignment.
"""

import json
import requests
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# RI taxonomy mapping
# ---------------------------------------------------------------------------

ERROR_TYPE_TO_TIER = {
    "OutE": "tier1",
    "EntE": "tier1",
    "CircE": "tier1.5",
    "RelE": "tier1.5",   # RelE in FRANK = relational/circumstance
    "PredE": "tier2",
    "LinkE": "tier2",
    "CorefE": "tier2",
}

# Predicted signal strength for gradient ordering (higher = stronger RI signal)
TIER_SIGNAL_ORDER = {
    "OutE": 6,   # strongest — out-of-article entity
    "EntE": 5,   # entity swap
    "CircE": 4,  # circumstance error
    "RelE": 4,   # relational error (same tier as CircE)
    "PredE": 3,  # predicate swap
    "LinkE": 2,  # discourse link error
    "CorefE": 1, # weakest — coreference error
}

ALL_ERROR_TYPES = list(TIER_SIGNAL_ORDER.keys())

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ErrorSpan:
    """A sentence-level error annotation in a summary."""

    text: str
    char_start: int
    char_end: int
    error_type: str
    tier: str


@dataclass
class FRANKExample:
    """One FRANK benchmark example with article, summary, and error annotations."""

    article_id: str
    article_text: str
    summary_text: str
    error_spans: list[ErrorSpan]
    system: str
    has_errors: bool


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

_FRANK_SENTENCE_URL = (
    "https://raw.githubusercontent.com/artidoro/frank/main/data/"
    "human_annotations_sentence.json"
)


def _download_frank(cache_dir: Path) -> Path:
    """Download FRANK sentence annotations from GitHub if not cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "frank_sentence_annotations.json"

    if cache_path.exists():
        return cache_path

    print(f"Downloading FRANK from {_FRANK_SENTENCE_URL} ...")
    resp = requests.get(_FRANK_SENTENCE_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"Saved FRANK annotations to {cache_path} ({len(data)} examples)")
    return cache_path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _majority_vote_errors(annotations: dict) -> list[str]:
    """Get error types with majority vote (>=2 of 3 annotators agree).

    annotations: {"annotator_0": ["EntE", "CircE"], "annotator_1": ["EntE"], ...}
    Returns: list of error types with >=2 annotator agreement.
    """
    all_types = Counter()
    n_annotators = len(annotations)
    threshold = max(2, (n_annotators + 1) // 2)  # majority

    for annotator_key, error_list in annotations.items():
        if isinstance(error_list, list):
            for etype in error_list:
                if etype and etype.strip():
                    all_types[etype.strip()] += 1

    # Return types with majority agreement
    return [etype for etype, count in all_types.items() if count >= threshold]


def _parse_example(raw: dict, idx: int) -> Optional[FRANKExample]:
    """Parse one FRANK sentence-annotated example.

    Fields: hash, model_name, article, summary, reference,
    summary_sentences, summary_sentences_annotations, split
    """
    article_text = raw.get("article", "")
    summary_text = raw.get("summary", "")
    system = raw.get("model_name", "unknown")

    if not article_text or not summary_text:
        return None

    sentences = raw.get("summary_sentences", [])
    sentence_annotations = raw.get("summary_sentences_annotations", [])

    error_spans = []

    for sent_text, sent_ann in zip(sentences, sentence_annotations):
        if not isinstance(sent_ann, dict):
            continue

        # Majority vote across annotators
        majority_errors = _majority_vote_errors(sent_ann)

        # Also include any error type present (union) for more data
        # But use majority for primary analysis
        for etype in majority_errors:
            # Filter to known error types
            if etype not in ERROR_TYPE_TO_TIER:
                # Try normalizing: GramE, Other are not in our taxonomy
                if etype in ("GramE", "Other", "NoE"):
                    continue
                continue

            # Find sentence position in summary
            char_start = summary_text.find(sent_text)
            if char_start < 0:
                char_start = 0
            char_end = char_start + len(sent_text)

            tier = ERROR_TYPE_TO_TIER[etype]
            error_spans.append(ErrorSpan(
                text=sent_text,
                char_start=char_start,
                char_end=char_end,
                error_type=etype,
                tier=tier,
            ))

    return FRANKExample(
        article_id=str(raw.get("hash", idx)),
        article_text=article_text,
        summary_text=summary_text,
        error_spans=error_spans,
        system=system,
        has_errors=len(error_spans) > 0,
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_frank(
    data_dir: Optional[Path] = None,
    max_examples: Optional[int] = None,
) -> list[FRANKExample]:
    """Load FRANK benchmark with sentence-level error annotations.

    Downloads from GitHub if not cached locally. Uses majority vote
    across 3 annotators for error type assignment.

    Args:
        data_dir: directory for cached data files. Defaults to exp06_frank/data/.
        max_examples: maximum number of examples to load (None = all).

    Returns:
        List of FRANKExample with parsed error spans and RI tier mappings.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"

    annotations_path = _download_frank(data_dir)

    with open(annotations_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected list, got {type(raw_data)}")

    examples = []
    for idx, raw in enumerate(raw_data):
        if max_examples is not None and len(examples) >= max_examples:
            break
        example = _parse_example(raw, idx)
        if example is not None:
            examples.append(example)

    n_with_errors = sum(1 for e in examples if e.has_errors)
    print(f"Loaded {len(examples)} FRANK examples ({n_with_errors} with errors)")

    # Error type distribution
    error_type_counts: dict[str, int] = {}
    for ex in examples:
        for span in ex.error_spans:
            error_type_counts[span.error_type] = error_type_counts.get(span.error_type, 0) + 1
    if error_type_counts:
        print("Error type distribution (majority vote):")
        for etype in ALL_ERROR_TYPES:
            count = error_type_counts.get(etype, 0)
            if count > 0:
                print(f"  {etype} ({ERROR_TYPE_TO_TIER[etype]}): {count}")

    return examples
