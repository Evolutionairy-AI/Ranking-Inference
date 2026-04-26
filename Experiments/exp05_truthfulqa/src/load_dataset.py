"""TruthfulQA dataset loading.

Loads TruthfulQA from HuggingFace (truthfulqa/truthful_qa). The MC targets
come from the 'multiple_choice' config, and category metadata comes from
the 'generation' config. Both are merged by question text.
"""

import datasets


def load_truthfulqa(max_examples: int | None = None) -> list[dict]:
    """Load TruthfulQA validation set.

    Returns list of dicts with keys:
        - question_idx: int
        - question: str
        - category: str (one of 38 categories, or "unknown")
        - mc1_targets: dict with 'choices' (list[str]) and 'labels' (list[int])
        - mc2_targets: dict with 'choices' (list[str]) and 'labels' (list[int])
    """
    # Load MC targets
    ds_mc = datasets.load_dataset(
        "truthfulqa/truthful_qa",
        "multiple_choice",
        split="validation",
    )

    # Load generation config for category metadata
    try:
        ds_gen = datasets.load_dataset(
            "truthfulqa/truthful_qa",
            "generation",
            split="validation",
        )
        # Build question -> category lookup
        q_to_category = {}
        for row in ds_gen:
            q_to_category[row["question"].strip()] = row.get("category", "unknown")
    except Exception:
        q_to_category = {}

    if max_examples is not None:
        ds_mc = ds_mc.select(range(min(max_examples, len(ds_mc))))

    examples: list[dict] = []
    for idx, raw in enumerate(ds_mc):
        question = raw["question"]
        category = q_to_category.get(question.strip(), "unknown")

        examples.append({
            "question_idx": idx,
            "question": question,
            "category": category,
            "mc1_targets": {
                "choices": raw["mc1_targets"]["choices"],
                "labels": raw["mc1_targets"]["labels"],
            },
            "mc2_targets": {
                "choices": raw["mc2_targets"]["choices"],
                "labels": raw["mc2_targets"]["labels"],
            },
        })

    return examples
