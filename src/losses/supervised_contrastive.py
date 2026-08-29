from __future__ import annotations

import torch
import torch.nn.functional as F


def _canonical_mask(canonical_ids: list[str] | None, size: int, device):
    if canonical_ids is None:
        return torch.ones((size, size), dtype=torch.bool, device=device)
    return torch.tensor(
        [[left != right for right in canonical_ids] for left in canonical_ids],
        dtype=torch.bool,
        device=device,
    )


def supervised_contrastive_loss(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.10,
        canonical_ids: list[str] | None = None) -> tuple[torch.Tensor, dict[str, int]]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    size = embeddings.shape[0]
    if size < 2:
        zero = embeddings.sum() * 0.0
        return zero, {"label_positive_pairs": 0, "ordinary_negative_pairs": 0,
                      "valid_label_anchors": 0}

    z = F.normalize(embeddings, dim=-1)
    logits = torch.mm(z, z.t()) / temperature
    diagonal = torch.eye(size, dtype=torch.bool, device=embeddings.device)
    distinct_graph = _canonical_mask(canonical_ids, size, embeddings.device)
    eligible = ~diagonal & distinct_graph
    positive = labels.reshape(-1, 1).eq(labels.reshape(1, -1)) & eligible
    negative = ~labels.reshape(-1, 1).eq(labels.reshape(1, -1)) & eligible

    masked_logits = logits.masked_fill(~eligible, float("-inf"))
    log_denominator = torch.logsumexp(masked_logits, dim=1, keepdim=True)
    log_probability = logits - log_denominator
    positive_count = positive.sum(dim=1)
    positive_log_probability = torch.where(
        positive, log_probability, torch.zeros_like(log_probability)).sum(dim=1)
    valid = positive_count > 0
    if valid.any():
        loss = -(positive_log_probability[valid] / positive_count[valid]).mean()
    else:
        loss = embeddings.sum() * 0.0
    upper = torch.triu(torch.ones_like(eligible), diagonal=1).bool()
    stats = {
        "label_positive_pairs": int((positive & upper).sum().item()),
        "ordinary_negative_pairs": int((negative & upper).sum().item()),
        "valid_label_anchors": int(valid.sum().item()),
    }
    return loss, stats
