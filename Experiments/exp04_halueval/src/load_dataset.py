"""HaluEval dataset loading and preprocessing.

Loads the HaluEval benchmark from HuggingFace (pminervini/HaluEval) with
three task configurations: QA, dialogue, and summarization. Each example
contains a single response with a hallucination yes/no label.
"""

import datasets


# Mapping from short task name to HuggingFace config name
_CONFIG_MAP = {
    "qa": "qa_samples",
    "dialogue": "dialogue_samples",
    "summarization": "summarization_samples",
}


def _normalise_example(raw: dict, task: str, idx: int) -> dict:
    """Convert a raw HuggingFace example to our unified format.

    HaluEval (pminervini/HaluEval) fields:
      QA:            knowledge, question, answer, hallucination (yes/no)
      Dialogue:      knowledge, dialogue_history, response, hallucination (yes/no)
      Summarization: document, summary, hallucination (yes/no)
    """
    if task == "qa":
        prompt_parts = []
        if raw.get("knowledge"):
            prompt_parts.append(f"Context: {raw['knowledge']}")
        prompt_parts.append(f"Question: {raw['question']}")
        prompt = "\n".join(prompt_parts)
        response_text = raw["answer"]

    elif task == "dialogue":
        prompt_parts = []
        if raw.get("knowledge"):
            prompt_parts.append(f"Knowledge: {raw['knowledge']}")
        prompt_parts.append(f"Dialogue history: {raw['dialogue_history']}")
        prompt = "\n".join(prompt_parts)
        response_text = raw["response"]

    elif task == "summarization":
        prompt = f"Document: {raw['document']}"
        response_text = raw["summary"]

    else:
        raise ValueError(f"Unknown task: {task}. Expected: qa, dialogue, summarization")

    # Label: 1 = hallucinated, 0 = correct
    hallucination_flag = raw.get("hallucination", "no")
    is_hallucinated = 1 if hallucination_flag.lower().strip() == "yes" else 0

    return {
        "example_id": f"{task}_{idx}",
        "task": task,
        "prompt": prompt,
        "response_text": response_text,
        "label": is_hallucinated,
    }


def load_halueval_split(
    task: str,
    split: str = "data",
    max_examples: int | None = None,
) -> list[dict]:
    """Load one HaluEval task split.

    Args:
        task: "qa", "dialogue", or "summarization"
        split: dataset split name (HaluEval uses "data" as its only split)
        max_examples: limit number of examples (None = load all)

    Returns:
        List of dicts with keys: example_id, task, prompt, response_text, label
        where label is 1 (hallucinated) or 0 (correct).
    """
    if task not in _CONFIG_MAP:
        raise ValueError(
            f"Unknown task: {task}. Expected one of: {list(_CONFIG_MAP.keys())}"
        )

    config_name = _CONFIG_MAP[task]
    ds = datasets.load_dataset("pminervini/HaluEval", config_name, split=split)

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    examples = []
    for idx, raw in enumerate(ds):
        examples.append(_normalise_example(raw, task, idx))

    return examples


def load_all_tasks(
    split: str = "data",
    max_per_task: int | None = None,
) -> list[dict]:
    """Load all three HaluEval task splits combined.

    Args:
        split: dataset split name
        max_per_task: limit examples per task (None = load all)

    Returns:
        Combined list of normalised examples across all tasks.
    """
    all_examples = []
    for task in _CONFIG_MAP:
        task_examples = load_halueval_split(task, split=split, max_examples=max_per_task)
        all_examples.extend(task_examples)
    return all_examples
