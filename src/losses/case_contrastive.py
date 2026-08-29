from __future__ import annotations

import torch
import torch.nn.functional as F


def _pair_masks(case_ids: list[str], states: list[str], labels: torch.Tensor,
                canonical_ids: list[str], device):
    size = len(case_ids)
    diagonal = torch.eye(size, dtype=torch.bool, device=device)
    valid_case = torch.tensor(
        [[bool(left) and bool(right) for right in case_ids] for left in case_ids],
        dtype=torch.bool, device=device)
    same_case = torch.tensor(
        [[left == right for right in case_ids] for left in case_ids],
        dtype=torch.bool, device=device) & valid_case
    same_state = torch.tensor(
        [[left == right for right in states] for left in states],
        dtype=torch.bool, device=device)
    distinct_graph = torch.tensor(
        [[left != right for right in canonical_ids] for left in canonical_ids],
        dtype=torch.bool, device=device)
    same_label = labels.reshape(-1, 1).eq(labels.reshape(1, -1))
    eligible = ~diagonal & distinct_graph
    strong_positive = same_case & same_state & same_label & eligible
    weak_positive = ~same_case & valid_case & same_label & eligible
    hard_negative = same_case & ~same_state & ~same_label & eligible
    ordinary_negative = ~same_case & valid_case & ~same_label & eligible
    return strong_positive, weak_positive, hard_negative, ordinary_negative


def pre_post_hard_contrastive_loss(
        embeddings: torch.Tensor, case_ids: list[str], states: list[str],
        labels: torch.Tensor, canonical_ids: list[str], margin: float = 0.20,
        ) -> tuple[torch.Tensor, dict[str, int]]:
    z = F.normalize(embeddings, dim=-1)
    _, _, hard_negative, _ = _pair_masks(
        case_ids, states, labels, canonical_ids, embeddings.device)
    upper = torch.triu(torch.ones_like(hard_negative), diagonal=1).bool()
    pairs = hard_negative & upper
    similarities = torch.mm(z, z.t())[pairs]
    if similarities.numel():
        loss = F.relu(similarities - margin).mean()
    else:
        loss = embeddings.sum() * 0.0
    return loss, {"pre_post_hard_negative_pairs": int(pairs.sum().item())}


def case_aware_contrastive_loss(
        embeddings: torch.Tensor, case_ids: list[str], states: list[str],
        labels: torch.Tensor, canonical_ids: list[str], temperature: float = 0.10,
        strong_positive_weight: float = 1.0,
        weak_positive_weight: float = 0.5,
        hard_negative_weight: float = 2.0,
        ordinary_negative_weight: float = 1.0,
        ) -> tuple[torch.Tensor, dict[str, int]]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = F.normalize(embeddings, dim=-1)
    strong, weak, hard, ordinary = _pair_masks(
        case_ids, states, labels, canonical_ids, embeddings.device)
    positive = strong | weak
    denominator_mask = positive | hard | ordinary
    logits = torch.mm(z, z.t()) / temperature
    denominator_weight = torch.ones_like(logits)
    denominator_weight = torch.where(
        hard, torch.full_like(logits, hard_negative_weight), denominator_weight)
    denominator_weight = torch.where(
        ordinary, torch.full_like(logits, ordinary_negative_weight), denominator_weight)
    weighted_logits = logits + denominator_weight.clamp_min(1e-12).log()
    weighted_logits = weighted_logits.masked_fill(~denominator_mask, float("-inf"))
    log_denominator = torch.logsumexp(weighted_logits, dim=1, keepdim=True)
    log_probability = logits - log_denominator
    positive_weight = (
        strong.to(logits.dtype) * strong_positive_weight
        + weak.to(logits.dtype) * weak_positive_weight)
    anchor_weight = positive_weight.sum(dim=1)
    weighted_positive_log = torch.where(
        positive, log_probability * positive_weight,
        torch.zeros_like(log_probability)).sum(dim=1)
    valid = anchor_weight > 0
    finite = torch.isfinite(weighted_positive_log) & torch.isfinite(anchor_weight)
    valid = valid & finite
    if valid.any():
        loss = -(weighted_positive_log[valid] / anchor_weight[valid]).mean()
    else:
        loss = embeddings.sum() * 0.0
    upper = torch.triu(torch.ones_like(positive), diagonal=1).bool()
    return loss, {
        "same_case_same_state_positive_pairs": int((strong & upper).sum().item()),
        "different_case_same_label_positive_pairs": int((weak & upper).sum().item()),
        "pre_post_hard_negative_pairs": int((hard & upper).sum().item()),
        "ordinary_negative_pairs": int((ordinary & upper).sum().item()),
        "valid_case_anchors": int(valid.sum().item()),
    }
