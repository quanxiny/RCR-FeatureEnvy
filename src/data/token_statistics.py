from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch


def document_frequency(graphs: dict, vocab: dict[str, int]) -> dict:
    counts = Counter()
    for graph_pair in graphs.values():
        counts.update(set(graph_pair["h1_index"]) | set(graph_pair["h2_index"]))
    size = max(vocab.values(), default=-1) + 1
    frequencies = [int(counts[index]) for index in range(size)]
    inverse_vocab = {index: token for token, index in vocab.items()}
    return {
        "documents": len(graphs),
        "vocab_size": size,
        "document_frequency": frequencies,
        "tokens": [inverse_vocab.get(index, "") for index in range(size)],
    }


def save_token_statistics(path: Path, statistics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(statistics, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    temporary.replace(path)


def load_token_statistics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def informative_token_mask(statistics: dict, max_document_ratio: float,
                           device=None) -> torch.Tensor:
    if not 0 < max_document_ratio <= 1:
        raise ValueError("max_document_ratio must be in (0, 1]")
    documents = max(int(statistics["documents"]), 1)
    mask = [0 < int(frequency) / documents <= max_document_ratio
            for frequency in statistics["document_frequency"]]
    return torch.tensor(mask, dtype=torch.bool, device=device)
