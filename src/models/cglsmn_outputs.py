from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class LocalAttention(TypedDict):
    similarity: Tensor
    graph1_weights: Tensor
    graph2_weights: Tensor


class LocalFeatures(TypedDict):
    graph1: Tensor
    graph2: Tensor


class CGLSMNForwardOutput(TypedDict, total=False):
    logits: Tensor
    probabilities: Tensor
    pair_embedding: Tensor
    graph1_embedding: Tensor
    graph2_embedding: Tensor
    attention: dict[str, LocalAttention]
    local_features: dict[str, LocalFeatures]


def local_attention(similarity: Tensor, graph1_weights: Tensor,
                    graph2_weights: Tensor) -> LocalAttention:
    return {
        "similarity": similarity,
        "graph1_weights": graph1_weights,
        "graph2_weights": graph2_weights,
    }


def local_features(graph1: Tensor, graph2: Tensor) -> LocalFeatures:
    return {"graph1": graph1, "graph2": graph2}
