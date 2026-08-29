from __future__ import annotations

import torch
import torch.nn.functional as F


STATISTIC_KEYS = (
    "local_positive_pairs",
    "local_token_positive_pairs",
    "local_mutual_positive_pairs",
    "local_infonce_negative_pairs",
    "local_hard_negative_pairs",
)


def _empty_stats() -> dict[str, int]:
    return {key: 0 for key in STATISTIC_KEYS}


def _positive_direction_loss(
        anchor_features: torch.Tensor,
        other_features: torch.Tensor,
        anchor_tokens: torch.Tensor,
        other_tokens: torch.Tensor,
        informative_tokens: torch.Tensor,
        temperature: float,
        positive_threshold: float,
        negative_threshold: float,
        negative_margin: float,
        negatives_per_anchor: int,
        max_positive_anchors: int):
    """Pull reliable local matches together inside one positive graph pair."""
    zero = anchor_features.sum() * 0.0
    empty_stats = _empty_stats()
    if anchor_features.shape[0] == 0 or other_features.shape[0] < 2:
        return zero, empty_stats

    anchor = F.normalize(anchor_features, dim=-1)
    other = F.normalize(other_features, dim=-1)
    similarity = torch.mm(anchor, other.t())
    detached = similarity.detach()
    informative_anchor = informative_tokens[anchor_tokens]
    exact_token = (
        anchor_tokens.reshape(-1, 1).eq(other_tokens.reshape(1, -1))
        & informative_anchor.reshape(-1, 1))

    # An informative exact-token match is the strongest available anchor.
    # Detached mutual nearest neighbours provide a conservative fallback when
    # the two graphs express the same local relation with different tokens.
    token_scores = detached.masked_fill(~exact_token, float("-inf"))
    token_score, token_index = token_scores.max(dim=1)
    has_token = torch.isfinite(token_score)

    row_score, row_index = detached.max(dim=1)
    column_index = detached.max(dim=0).indices
    anchor_index = torch.arange(anchor.shape[0], device=anchor.device)
    mutual = (
        column_index[row_index].eq(anchor_index)
        & row_score.ge(positive_threshold))

    positive_index = torch.where(has_token, token_index, row_index)
    positive_score = detached[anchor_index, positive_index]
    valid_positive = has_token | mutual
    valid_indices = valid_positive.nonzero(as_tuple=False).reshape(-1)
    if valid_indices.numel() == 0:
        return zero, empty_stats
    if valid_indices.numel() > max_positive_anchors:
        confidence = positive_score[valid_indices]
        keep = confidence.topk(max_positive_anchors, largest=True).indices
        valid_indices = valid_indices[keep]

    selected_positive = positive_index[valid_indices]
    candidates = ~exact_token[valid_indices]
    candidates.scatter_(1, selected_positive.reshape(-1, 1), False)
    maximum_negatives = min(negatives_per_anchor, other.shape[0] - 1)
    if maximum_negatives < 1:
        return zero, empty_stats
    candidate_scores = detached[valid_indices].masked_fill(~candidates, float("inf"))
    negative_detached, negative_index = candidate_scores.topk(
        maximum_negatives, dim=1, largest=False)
    reliable_negative = (
        torch.isfinite(negative_detached)
        & negative_detached.le(negative_threshold))
    valid_anchor = reliable_negative.any(dim=1)
    if not valid_anchor.any():
        return zero, empty_stats

    valid_indices = valid_indices[valid_anchor]
    selected_positive = selected_positive[valid_anchor]
    negative_index = negative_index[valid_anchor]
    reliable_negative = reliable_negative[valid_anchor]
    positive_logits = similarity[valid_indices, selected_positive].reshape(-1, 1)
    negative_logits = similarity[valid_indices.reshape(-1, 1), negative_index]
    logits = torch.cat([positive_logits, negative_logits], dim=1) / temperature
    logit_mask = torch.cat([
        torch.ones((logits.shape[0], 1), dtype=torch.bool, device=logits.device),
        reliable_negative,
    ], dim=1)
    logits = logits.masked_fill(~logit_mask, float("-inf"))
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    info_nce = F.cross_entropy(logits, targets)
    reliable_values = negative_logits[reliable_negative]
    margin_loss = (
        F.relu(reliable_values - negative_margin).mean()
        if reliable_values.numel() else zero)
    loss = info_nce + 0.5 * margin_loss

    token_count = int(has_token[valid_indices].sum().item())
    mutual_count = int((~has_token[valid_indices] & mutual[valid_indices]).sum().item())
    stats = _empty_stats()
    stats.update({
        "local_positive_pairs": int(valid_indices.numel()),
        "local_token_positive_pairs": token_count,
        "local_mutual_positive_pairs": mutual_count,
        "local_infonce_negative_pairs": int(reliable_negative.sum().item()),
    })
    return loss, stats


def _negative_pair_direction_loss(
        anchor_features: torch.Tensor,
        other_features: torch.Tensor,
        anchor_tokens: torch.Tensor,
        other_tokens: torch.Tensor,
        informative_tokens: torch.Tensor,
        negative_pair_margin: float,
        max_negative_anchors: int):
    """Repel the strongest contextual pseudo-match inside one negative pair."""
    zero = anchor_features.sum() * 0.0
    empty_stats = _empty_stats()
    if anchor_features.shape[0] == 0 or other_features.shape[0] == 0:
        return zero, empty_stats

    similarity = torch.mm(
        F.normalize(anchor_features, dim=-1),
        F.normalize(other_features, dim=-1).t())
    informative_anchor = informative_tokens[anchor_tokens]
    informative_other = informative_tokens[other_tokens]
    eligible = informative_anchor.reshape(-1, 1) & informative_other.reshape(1, -1)
    detached = similarity.detach().masked_fill(~eligible, float("-inf"))
    row_score, row_index = detached.max(dim=1)
    column_index = detached.max(dim=0).indices
    anchor_index = torch.arange(anchor_features.shape[0], device=anchor_features.device)
    mutual = (
        torch.isfinite(row_score)
        & column_index[row_index].eq(anchor_index))
    selected = mutual.nonzero(as_tuple=False).reshape(-1)
    if selected.numel() == 0:
        return zero, empty_stats
    if selected.numel() > max_negative_anchors:
        keep = row_score[selected].topk(max_negative_anchors, largest=True).indices
        selected = selected[keep]
    hard_similarity = similarity[selected, row_index[selected]]
    loss = F.relu(hard_similarity - negative_pair_margin).square().mean()
    stats = _empty_stats()
    stats["local_hard_negative_pairs"] = int(selected.numel())
    return loss, stats


def local_graph_pair_contrastive_loss(
        features_by_level: dict[str, dict[str, torch.Tensor]],
        graph1_tokens: torch.Tensor,
        graph2_tokens: torch.Tensor,
        informative_tokens: torch.Tensor,
        label: int,
        temperature: float = 0.10,
        positive_threshold: float = 0.70,
        negative_threshold: float = 0.25,
        negative_margin: float = 0.10,
        negative_pair_margin: float = 0.30,
        negatives_per_anchor: int = 16,
        max_positive_anchors: int = 64,
        max_negative_anchors: int = 64,
        ) -> tuple[torch.Tensor, dict[str, int]]:
    """Label-conditioned local contrast for one graph pair.

    A positive feature-envy pair pulls rare-token or confident mutual local
    matches together with InfoNCE. A negative pair repels its strongest
    contextual mutual matches. The text level is omitted for negative pairs:
    identical tokens have identical context-free embeddings there, so asking
    the shared embedding table to separate them conditionally is contradictory.
    """
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    losses = []
    totals = _empty_stats()
    for level in ("text", "type", "call"):
        features = features_by_level[level]
        if label == 1:
            loss_function = _positive_direction_loss
            arguments = (
                informative_tokens, temperature, positive_threshold,
                negative_threshold, negative_margin, negatives_per_anchor,
                max_positive_anchors)
        elif level == "text":
            for key in STATISTIC_KEYS:
                totals[f"{level}_{key}"] = 0
            continue
        else:
            loss_function = _negative_pair_direction_loss
            arguments = (
                informative_tokens, negative_pair_margin,
                max_negative_anchors)

        forward_loss, forward_stats = loss_function(
            features["graph1"], features["graph2"], graph1_tokens,
            graph2_tokens, *arguments)
        backward_loss, backward_stats = loss_function(
            features["graph2"], features["graph1"], graph2_tokens,
            graph1_tokens, *arguments)
        relation_key = (
            "local_positive_pairs" if label == 1
            else "local_hard_negative_pairs")
        directional_count = (
            int(forward_stats[relation_key] > 0)
            + int(backward_stats[relation_key] > 0))
        if directional_count:
            losses.append((forward_loss + backward_loss) / directional_count)
        for key in STATISTIC_KEYS:
            value = forward_stats[key] + backward_stats[key]
            totals[key] += value
            totals[f"{level}_{key}"] = value
    if losses:
        loss = torch.stack(losses).mean()
    else:
        first = features_by_level["text"]["graph1"]
        loss = first.sum() * 0.0
    return loss, totals
