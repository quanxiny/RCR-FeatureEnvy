from __future__ import annotations

import torch
import torch.nn.functional as F


STATISTIC_KEYS = (
    "directed_evidence_pairs",
    "directed_mutual_pairs",
    "directed_infonce_negatives",
    "directed_negative_pair_matches",
)
LEVELS = ("type", "call")


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _pair_infonce(
    similarity: torch.Tensor,
    positive_rows: torch.Tensor,
    positive_columns: torch.Tensor,
    exact_mask: torch.Tensor,
    *,
    temperature: float,
    negatives_per_anchor: int,
) -> tuple[torch.Tensor, int]:
    zero = _zero(similarity)
    if not positive_rows.numel() or similarity.shape[1] < 2:
        return zero, 0
    candidates = ~exact_mask[positive_rows]
    candidates.scatter_(1, positive_columns.reshape(-1, 1), False)
    count = min(negatives_per_anchor, similarity.shape[1] - 1)
    if count < 1:
        return zero, 0
    detached = similarity.detach()[positive_rows].masked_fill(
        ~candidates, float("-inf")
    )
    negative_scores, negative_columns = detached.topk(
        count, dim=1, largest=True
    )
    valid_negatives = torch.isfinite(negative_scores)
    valid_rows = valid_negatives.any(dim=1)
    if not valid_rows.any():
        return zero, 0
    positive_rows = positive_rows[valid_rows]
    positive_columns = positive_columns[valid_rows]
    negative_columns = negative_columns[valid_rows]
    valid_negatives = valid_negatives[valid_rows]
    positive_logits = similarity[
        positive_rows, positive_columns
    ].reshape(-1, 1)
    negative_logits = similarity[
        positive_rows.reshape(-1, 1), negative_columns
    ]
    logits = torch.cat([positive_logits, negative_logits], dim=1) / temperature
    mask = torch.cat(
        [
            torch.ones(
                (logits.shape[0], 1), dtype=torch.bool, device=logits.device
            ),
            valid_negatives,
        ],
        dim=1,
    )
    logits = logits.masked_fill(~mask, float("-inf"))
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets), int(valid_negatives.sum().item())


def _directed_positive_level(
    method_features: torch.Tensor,
    class_features: torch.Tensor,
    method_tokens: torch.Tensor,
    class_tokens: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    informative_tokens: torch.Tensor,
    *,
    temperature: float,
    mutual_threshold: float,
    negatives_per_anchor: int,
    max_anchors: int,
    enable_mutual: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    method = F.normalize(method_features, dim=-1)
    target = F.normalize(class_features, dim=-1)
    similarity = torch.mm(method, target.t())
    zero = _zero(similarity)
    if not method.shape[0] or class_features.shape[0] < 2:
        return zero, zero, {key: 0 for key in STATISTIC_KEYS}

    if anchor_token_ids.numel():
        anchor_method = torch.isin(method_tokens, anchor_token_ids)
    else:
        anchor_method = torch.zeros_like(method_tokens, dtype=torch.bool)
    exact = (
        method_tokens.reshape(-1, 1).eq(class_tokens.reshape(1, -1))
        & anchor_method.reshape(-1, 1)
    )
    exact_scores = similarity.detach().masked_fill(~exact, float("-inf"))
    best_score, best_column = exact_scores.max(dim=1)
    evidence_rows = torch.isfinite(best_score).nonzero(as_tuple=False).reshape(-1)
    if evidence_rows.numel() > max_anchors:
        keep = best_score[evidence_rows].topk(max_anchors).indices
        evidence_rows = evidence_rows[keep]
    evidence_columns = best_column[evidence_rows]
    evidence_loss, evidence_negatives = _pair_infonce(
        similarity,
        evidence_rows,
        evidence_columns,
        exact,
        temperature=temperature,
        negatives_per_anchor=negatives_per_anchor,
    )

    if not enable_mutual:
        return evidence_loss, zero, {
            "directed_evidence_pairs": int(evidence_rows.numel()),
            "directed_mutual_pairs": 0,
            "directed_infonce_negatives": evidence_negatives,
            "directed_negative_pair_matches": 0,
        }

    detached = similarity.detach()
    row_score, row_column = detached.max(dim=1)
    column_row = detached.max(dim=0).indices
    method_indices = torch.arange(method.shape[0], device=method.device)
    eligible = (
        informative_tokens[method_tokens]
        & ~anchor_method
        & column_row[row_column].eq(method_indices)
        & row_score.ge(mutual_threshold)
    )
    mutual_rows = eligible.nonzero(as_tuple=False).reshape(-1)
    if mutual_rows.numel() > max_anchors:
        keep = row_score[mutual_rows].topk(max_anchors).indices
        mutual_rows = mutual_rows[keep]
    mutual_columns = row_column[mutual_rows]
    mutual_exact = torch.zeros_like(exact)
    if mutual_rows.numel():
        mutual_exact[mutual_rows, mutual_columns] = True
    mutual_loss, mutual_negatives = _pair_infonce(
        similarity,
        mutual_rows,
        mutual_columns,
        mutual_exact,
        temperature=temperature,
        negatives_per_anchor=negatives_per_anchor,
    )
    return evidence_loss, mutual_loss, {
        "directed_evidence_pairs": int(evidence_rows.numel()),
        "directed_mutual_pairs": int(mutual_rows.numel()),
        "directed_infonce_negatives": evidence_negatives + mutual_negatives,
        "directed_negative_pair_matches": 0,
    }


def _directed_negative_level(
    method_features: torch.Tensor,
    class_features: torch.Tensor,
    method_tokens: torch.Tensor,
    class_tokens: torch.Tensor,
    informative_tokens: torch.Tensor,
    *,
    margin: float,
    topk: int,
) -> tuple[torch.Tensor, dict]:
    method = F.normalize(method_features, dim=-1)
    target = F.normalize(class_features, dim=-1)
    similarity = torch.mm(method, target.t())
    zero = _zero(similarity)
    eligible = (
        informative_tokens[method_tokens].reshape(-1, 1)
        & informative_tokens[class_tokens].reshape(1, -1)
    )
    detached = similarity.detach().masked_fill(~eligible, float("-inf"))
    row_score, row_column = detached.max(dim=1)
    column_row = detached.max(dim=0).indices
    method_indices = torch.arange(method.shape[0], device=method.device)
    mutual = (
        torch.isfinite(row_score)
        & column_row[row_column].eq(method_indices)
    )
    rows = mutual.nonzero(as_tuple=False).reshape(-1)
    if not rows.numel():
        return zero, {key: 0 for key in STATISTIC_KEYS}
    count = min(topk, rows.numel())
    keep = row_score[rows].topk(count).indices
    rows = rows[keep]
    scores = similarity[rows, row_column[rows]]
    pair_score = scores.mean()
    loss = F.relu(pair_score - margin).square()
    stats = {key: 0 for key in STATISTIC_KEYS}
    stats["directed_negative_pair_matches"] = int(rows.numel())
    return loss, stats


def directed_method_class_contrastive_loss(
    features_by_level: dict[str, dict[str, torch.Tensor]],
    method_tokens: torch.Tensor,
    class_tokens: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    informative_tokens: torch.Tensor,
    label: int,
    *,
    temperature: float = 0.10,
    mutual_threshold: float = 0.80,
    negative_margin: float = 0.65,
    negatives_per_anchor: int = 16,
    max_anchors: int = 64,
    negative_topk: int = 8,
    enable_mutual: bool = True,
    enable_negative: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Directed method→class relation losses for one graph-pair sample."""
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    reference = features_by_level["type"]["graph1"]
    evidence_losses = []
    mutual_losses = []
    negative_losses = []
    totals = {key: 0 for key in STATISTIC_KEYS}
    for level in LEVELS:
        features = features_by_level[level]
        if label == 1:
            evidence, mutual, stats = _directed_positive_level(
                features["graph1"],
                features["graph2"],
                method_tokens,
                class_tokens,
                anchor_token_ids,
                informative_tokens,
                temperature=temperature,
                mutual_threshold=mutual_threshold,
                negatives_per_anchor=negatives_per_anchor,
                max_anchors=max_anchors,
                enable_mutual=enable_mutual,
            )
            if stats["directed_evidence_pairs"]:
                evidence_losses.append(evidence)
            if stats["directed_mutual_pairs"]:
                mutual_losses.append(mutual)
        elif enable_negative:
            negative, stats = _directed_negative_level(
                features["graph1"],
                features["graph2"],
                method_tokens,
                class_tokens,
                informative_tokens,
                margin=negative_margin,
                topk=negative_topk,
            )
            if stats["directed_negative_pair_matches"]:
                negative_losses.append(negative)
        else:
            stats = {key: 0 for key in STATISTIC_KEYS}
        for key, value in stats.items():
            totals[key] += value
            totals[f"{level}_{key}"] = value

    def mean_or_zero(losses):
        return torch.stack(losses).mean() if losses else _zero(reference)

    return {
        "evidence": mean_or_zero(evidence_losses),
        "mutual": mean_or_zero(mutual_losses),
        "negative": mean_or_zero(negative_losses),
    }, totals
