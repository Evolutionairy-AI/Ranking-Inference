"""
Rank computation utilities.

Handles tokenization, frequency counting, rank assignment,
and rank deviation computation across different tokenizers.
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class RankTable:
    """Precomputed rank lookup table for a corpus + tokenizer combination."""
    token_to_rank: dict[int, int]       # token_id -> global rank (1-indexed)
    rank_to_token: dict[int, int]       # rank -> token_id
    token_to_freq: dict[int, int]       # token_id -> raw frequency count
    rank_to_freq: np.ndarray            # rank (0-indexed into array) -> frequency
    tokenizer_name: str
    corpus_name: str
    total_tokens: int
    vocab_size: int

    def get_rank(self, token_id: int) -> int:
        """Get global rank for a token. Returns vocab_size+1 for unseen tokens."""
        return self.token_to_rank.get(token_id, self.vocab_size + 1)

    def save(self, path: Path):
        """Save rank table to disk."""
        data = {
            "token_to_rank": {str(k): v for k, v in self.token_to_rank.items()},
            "token_to_freq": {str(k): v for k, v in self.token_to_freq.items()},
            "tokenizer_name": self.tokenizer_name,
            "corpus_name": self.corpus_name,
            "total_tokens": self.total_tokens,
            "vocab_size": self.vocab_size,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "RankTable":
        """Load rank table from disk."""
        with open(path) as f:
            data = json.load(f)

        token_to_rank = {int(k): v for k, v in data["token_to_rank"].items()}
        token_to_freq = {int(k): v for k, v in data["token_to_freq"].items()}

        # Reconstruct rank_to_token and rank_to_freq
        rank_to_token = {v: k for k, v in token_to_rank.items()}
        max_rank = max(token_to_rank.values()) if token_to_rank else 0
        rank_to_freq = np.zeros(max_rank + 1, dtype=np.int64)
        for token_id, rank in token_to_rank.items():
            rank_to_freq[rank] = token_to_freq.get(token_id, 0)

        return cls(
            token_to_rank=token_to_rank,
            rank_to_token=rank_to_token,
            token_to_freq=token_to_freq,
            rank_to_freq=rank_to_freq,
            tokenizer_name=data["tokenizer_name"],
            corpus_name=data["corpus_name"],
            total_tokens=data["total_tokens"],
            vocab_size=data["vocab_size"],
        )


def build_rank_table(
    token_ids: list[int],
    tokenizer_name: str,
    corpus_name: str,
) -> RankTable:
    """Build a rank table from a list of token IDs.

    Args:
        token_ids: flat list of all token IDs in the corpus
        tokenizer_name: name of the tokenizer used
        corpus_name: name of the source corpus

    Returns:
        RankTable with frequency-ordered ranks (rank 1 = most frequent)
    """
    freq_counter = Counter(token_ids)
    # Sort by frequency descending, then by token_id for stability
    sorted_tokens = sorted(freq_counter.items(), key=lambda x: (-x[1], x[0]))

    token_to_rank = {}
    rank_to_token = {}
    token_to_freq = dict(freq_counter)

    for rank_0indexed, (token_id, freq) in enumerate(sorted_tokens):
        rank = rank_0indexed + 1  # 1-indexed
        token_to_rank[token_id] = rank
        rank_to_token[rank] = token_id

    max_rank = len(sorted_tokens)
    rank_to_freq = np.zeros(max_rank + 1, dtype=np.int64)  # index 0 unused
    for rank, token_id in rank_to_token.items():
        rank_to_freq[rank] = freq_counter[token_id]

    return RankTable(
        token_to_rank=token_to_rank,
        rank_to_token=rank_to_token,
        token_to_freq=token_to_freq,
        rank_to_freq=rank_to_freq,
        tokenizer_name=tokenizer_name,
        corpus_name=corpus_name,
        total_tokens=len(token_ids),
        vocab_size=max_rank,
    )


def compute_rank_deviations(
    local_token_ids: list[int],
    global_rank_table: RankTable,
) -> np.ndarray:
    """Compute rank deviation Delta_r = log2(r_global / r_local) for each token.

    Args:
        local_token_ids: token IDs from a local context (e.g., one domain corpus)
        global_rank_table: precomputed global rank table (e.g., from Wikipedia)

    Returns:
        Array of Delta_r values, one per unique token in local context
    """
    local_freq = Counter(local_token_ids)
    # Build local ranks
    sorted_local = sorted(local_freq.items(), key=lambda x: (-x[1], x[0]))
    local_ranks = {token_id: i + 1 for i, (token_id, _) in enumerate(sorted_local)}

    deviations = []
    for token_id, local_rank in local_ranks.items():
        global_rank = global_rank_table.get_rank(token_id)
        # Delta_r = log2(r_global / r_local), measured in bits
        if local_rank > 0 and global_rank > 0:
            delta_r = np.log2(global_rank / local_rank)
            deviations.append(delta_r)

    return np.array(deviations)


def compute_token_level_deviations(
    token_ids: list[int],
    global_rank_table: RankTable,
    local_rank_table: RankTable,
) -> list[dict]:
    """Compute per-token rank deviations with full detail.

    Returns a list of dicts with token_id, global_rank, local_rank, delta_r
    for every token occurrence.
    """
    results = []
    for tid in token_ids:
        g_rank = global_rank_table.get_rank(tid)
        l_rank = local_rank_table.get_rank(tid)
        if g_rank > 0 and l_rank > 0:
            delta_r = np.log2(g_rank / l_rank)
        else:
            delta_r = float("nan")
        results.append({
            "token_id": tid,
            "global_rank": g_rank,
            "local_rank": l_rank,
            "delta_r": delta_r,
        })
    return results
