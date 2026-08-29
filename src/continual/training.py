from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn.functional as F

from src.losses.directed_relation_contrastive import (
    directed_method_class_contrastive_loss,
)
from src.losses.supervised_contrastive import supervised_contrastive_loss
from src.utils.metrics import classification_metrics


def balanced_epoch(lines: Iterable[str], rng: random.Random) -> list[str]:
    """Reproduce the original label-balanced epoch sampler."""
    shuffled = list(lines)
    rng.shuffle(shuffled)
    positives = [line for line in shuffled if int(line.split()[3]) == 1]
    negatives = [line for line in shuffled if int(line.split()[3]) == 0]
    count = min(len(positives), len(negatives))
    selected = positives[:count] + negatives[:count]
    rng.shuffle(selected)
    return selected


def balanced_epoch_size(lines: Iterable[str], batch_size: int) -> int:
    labels = [int(line.split()[3]) for line in lines]
    count = min(labels.count(0), labels.count(1)) * 2
    return (count // batch_size) * batch_size


def one_hot_target(label: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([[0.0, 1.0] if label else [1.0, 0.0]], device=device)


def _unit_distribution(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1).abs().clamp_min(1e-8)
    return values / values.sum().clamp_min(1e-8)


def relation_distillation_loss(
    student: dict,
    teacher: dict,
    *,
    logit_coefficient: float,
    embedding_coefficient: float,
    attention_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve a replay sample's decision and cross-graph relation state."""
    reference = student["logits"]
    zero = reference.new_zeros(())
    logit_loss = zero
    embedding_loss = zero
    attention_loss = zero
    if logit_coefficient > 0.0:
        logit_loss = F.mse_loss(student["logits"], teacher["logits"])
    if embedding_coefficient > 0.0:
        embedding_loss = (
            1.0
            - F.cosine_similarity(
                student["pair_embedding"],
                teacher["pair_embedding"],
                dim=-1,
            ).mean()
        )
    if attention_coefficient > 0.0:
        terms = []
        for level in ("text", "type", "call"):
            for graph in ("graph1_weights", "graph2_weights"):
                current = _unit_distribution(student["attention"][level][graph])
                target = _unit_distribution(
                    teacher["attention"][level][graph]
                ).detach()
                terms.append(
                    F.kl_div(current.log(), target, reduction="sum")
                )
        attention_loss = torch.stack(terms).mean() if terms else zero
    effective = (
        logit_coefficient * logit_loss
        + embedding_coefficient * embedding_loss
        + attention_coefficient * attention_loss
    )
    return effective, {
        "logit_distillation_loss": logit_loss,
        "embedding_distillation_loss": embedding_loss,
        "attention_distillation_loss": attention_loss,
        "effective_distillation_loss": effective,
    }


def directed_auxiliary_loss(
    output: dict,
    graph1_tokens: torch.Tensor,
    graph2_tokens: torch.Tensor,
    label: int,
    label_line: str,
    classification_loss: torch.Tensor,
    epoch: int,
    options: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the proven CL6 method→class auxiliary objective."""
    component_losses, statistics = directed_method_class_contrastive_loss(
        output["local_features"],
        graph1_tokens,
        graph2_tokens,
        torch.as_tensor(
            options["anchors"].get(label_line, ()),
            dtype=torch.long,
            device=graph1_tokens.device,
        ),
        options["informative_tokens"],
        label,
        temperature=options["temperature"],
        mutual_threshold=options["mutual_threshold"],
        negative_margin=options["negative_margin"],
        negatives_per_anchor=options["negatives_per_anchor"],
        max_anchors=options["max_anchors"],
        negative_topk=options["negative_topk"],
        enable_mutual=options["lambda_mutual"] > 0.0,
        enable_negative=options["lambda_negative"] > 0.0,
    )
    weighted = (
        options["lambda_positive"] * component_losses["evidence"]
        + options["lambda_mutual"] * component_losses["mutual"]
        + options["lambda_negative"] * component_losses["negative"]
    )
    ramp = min(
        1.0,
        (epoch - options["warmup_epochs"] + 1)
        / max(options["ramp_epochs"], 1),
    )
    weighted_value = float(weighted.detach())
    scale = 0.0
    if weighted_value > 0.0:
        maximum = (
            options["max_aux_ratio"]
            * float(classification_loss.detach())
        )
        scale = min(1.0, maximum / max(weighted_value, 1e-12))
    effective = weighted * (ramp * scale)
    contributing = int(
        (
            options["lambda_positive"] > 0.0
            and statistics["directed_evidence_pairs"] > 0
        )
        or (
            options["lambda_mutual"] > 0.0
            and statistics["directed_mutual_pairs"] > 0
        )
        or (
            options["lambda_negative"] > 0.0
            and statistics["directed_negative_pair_matches"] > 0
        )
    )
    return effective, {
        "directed_evidence_loss": float(
            component_losses["evidence"].detach()
        ),
        "directed_mutual_loss": float(
            component_losses["mutual"].detach()
        ),
        "directed_negative_loss": float(
            component_losses["negative"].detach()
        ),
        "directed_effective_aux_loss": float(effective.detach()),
        "directed_aux_scale": scale,
        "directed_ramp": ramp,
        "directed_contributing_samples": contributing,
        "directed_evidence_pairs": statistics["directed_evidence_pairs"],
        "directed_mutual_pairs": statistics["directed_mutual_pairs"],
        "directed_infonce_negatives": statistics[
            "directed_infonce_negatives"
        ],
        "directed_negative_pair_matches": statistics[
            "directed_negative_pair_matches"
        ],
    }


def train_epoch(
    model,
    optimizer,
    criterion,
    adapter,
    train_lines: list[str],
    batch_size: int,
    backward_chunk: int,
    device: torch.device,
    rng: random.Random,
    regularizer=None,
    replay_lines: set[str] | None = None,
    teacher_model=None,
    logit_distillation: float = 0.0,
    embedding_distillation: float = 0.0,
    attention_distillation: float = 0.0,
    epoch: int = 0,
    directed_options: dict | None = None,
    projection=None,
    supcon_options: dict | None = None,
) -> dict:
    model.train()
    if projection is not None:
        projection.train()
    selected = balanced_epoch(train_lines, rng)
    usable = (len(selected) // batch_size) * batch_size
    selected = selected[:usable]
    started = time.perf_counter()
    total_loss = 0.0
    correct = 0
    found = 0
    total_regularization = 0.0
    total_distillation = {
        "logit_distillation_loss": 0.0,
        "embedding_distillation_loss": 0.0,
        "attention_distillation_loss": 0.0,
        "effective_distillation_loss": 0.0,
    }
    distilled_samples = 0
    total_directed = {
        "directed_evidence_loss": 0.0,
        "directed_mutual_loss": 0.0,
        "directed_negative_loss": 0.0,
        "directed_effective_aux_loss": 0.0,
        "directed_aux_scale": 0.0,
        "directed_ramp": 0.0,
        "directed_contributing_samples": 0,
        "directed_evidence_pairs": 0,
        "directed_mutual_pairs": 0,
        "directed_infonce_negatives": 0,
        "directed_negative_pair_matches": 0,
    }
    total_supcon_loss = 0.0
    total_supcon_effective_loss = 0.0
    total_supcon_positive_pairs = 0
    total_supcon_negative_pairs = 0
    total_supcon_valid_anchors = 0
    supcon_batches = 0
    replay_lines = replay_lines or set()
    distillation_enabled = (
        teacher_model is not None
        and any(
            value > 0.0
            for value in (
                logit_distillation,
                embedding_distillation,
                attention_distillation,
            )
        )
    )
    directed_active = (
        directed_options is not None
        and epoch >= directed_options["warmup_epochs"]
    )
    supcon_active = projection is not None and supcon_options is not None
    if teacher_model is not None:
        teacher_model.eval()
    for batch_start in range(0, usable, batch_size):
        optimizer.zero_grad(set_to_none=True)
        pending_loss = None
        pending_count = 0
        batch_found = 0
        pair_embeddings = []
        pair_labels = []
        pair_ids = []
        batch = selected[batch_start:batch_start + batch_size]
        for line in batch:
            tensors = adapter.graph_tensors(line, device)
            if tensors is None or tensors[1].numel() == 0 or tensors[3].numel() == 0:
                continue
            h1, e1, h2, e2, label = tensors
            should_distill = distillation_enabled and line in replay_lines
            rich_forward = should_distill or directed_active or supcon_active
            if rich_forward:
                student = model(
                    h1,
                    e1,
                    h2,
                    e2,
                    return_features=should_distill or supcon_active,
                    return_attention=(
                        should_distill and attention_distillation > 0.0
                    ),
                    return_local_features=directed_active,
                )
                output = student["probabilities"]
                if supcon_active:
                    pair_embeddings.append(student["pair_embedding"])
                    pair_labels.append(label)
                    pair_ids.append(line)
            else:
                output = model(h1, e1, h2, e2)
            classification_loss = criterion(
                output, one_hot_target(label, device)
            )
            directed_loss = classification_loss.new_zeros(())
            if directed_active:
                directed_loss, directed_metrics = directed_auxiliary_loss(
                    student,
                    h1,
                    h2,
                    label,
                    line,
                    classification_loss,
                    epoch,
                    directed_options,
                )
                for name, value in directed_metrics.items():
                    total_directed[name] += value
            distillation_loss = classification_loss.new_zeros(())
            if should_distill:
                with torch.no_grad():
                    teacher = teacher_model(
                        h1,
                        e1,
                        h2,
                        e2,
                        return_features=True,
                        return_attention=attention_distillation > 0.0,
                    )
                distillation_loss, components = relation_distillation_loss(
                    student,
                    teacher,
                    logit_coefficient=logit_distillation,
                    embedding_coefficient=embedding_distillation,
                    attention_coefficient=attention_distillation,
                )
                for name, value in components.items():
                    total_distillation[name] += float(value.detach())
                distilled_samples += 1
            loss = classification_loss + directed_loss + distillation_loss
            if directed_active or should_distill:
                # CL6 local features and relation attention retain NxM
                # cross-graph autograd matrices. Backpropagate immediately so
                # they are released before the next graph pair.
                if pending_loss is not None:
                    pending_loss.backward()
                    pending_loss = None
                    pending_count = 0
                loss.backward()
            else:
                pending_loss = (
                    loss if pending_loss is None else pending_loss + loss
                )
                pending_count += 1
            total_loss += float(classification_loss.detach())
            correct += int(output.argmax(dim=1).item() == label)
            found += 1
            batch_found += 1
            if pending_count >= backward_chunk and not supcon_active:
                pending_loss.backward()
                pending_loss = None
                pending_count = 0
        supcon_loss = None
        supcon_effective = None
        if supcon_active and pair_embeddings:
            embeddings = torch.cat(pair_embeddings, dim=0)
            projected = F.normalize(projection(embeddings), dim=-1)
            labels = torch.as_tensor(pair_labels, device=device)
            supcon_loss, supcon_stats = supervised_contrastive_loss(
                projected,
                labels,
                temperature=supcon_options["temperature"],
                canonical_ids=pair_ids,
            )
            supcon_effective = (
                len(pair_embeddings)
                * supcon_options["coefficient"]
                * supcon_loss
            )
            total_supcon_loss += float(supcon_loss.detach())
            total_supcon_effective_loss += float(supcon_effective.detach())
            total_supcon_positive_pairs += supcon_stats["label_positive_pairs"]
            total_supcon_negative_pairs += supcon_stats["ordinary_negative_pairs"]
            total_supcon_valid_anchors += supcon_stats["valid_label_anchors"]
            supcon_batches += 1
        if pending_loss is not None:
            if supcon_effective is not None:
                pending_loss = pending_loss + supcon_effective
            pending_loss.backward()
        if regularizer is not None and batch_found:
            # The published optimizer uses a sum of per-sample losses. Scale
            # the EWC term by the number of usable samples so lambda retains
            # its usual mean-objective interpretation.
            regularization = regularizer(model)
            (regularization * batch_found).backward()
            total_regularization += float(regularization.detach()) * batch_found
        optimizer.step()
        completed = batch_start + batch_size
        if completed % (batch_size * 25) == 0 or completed == usable:
            print(
                f"train {completed}/{usable} loss={total_loss / max(found, 1):.6f} "
                f"acc={correct / max(found, 1):.4f}",
                flush=True,
            )
    return {
        "train_seconds": time.perf_counter() - started,
        "train_loss": total_loss / max(found, 1),
        "train_accuracy": correct / max(found, 1),
        "regularization_loss": total_regularization / max(found, 1),
        **{
            name: value / max(distilled_samples, 1)
            for name, value in total_distillation.items()
        },
        **{
            name: (
                value
                if name.endswith("_samples")
                or name.endswith("_pairs")
                or name.endswith("_negatives")
                or name.endswith("_matches")
                else value / max(found, 1)
            )
            for name, value in total_directed.items()
        },
        "distilled_samples": distilled_samples,
        "supervised_contrastive_loss": (
            total_supcon_loss / max(supcon_batches, 1)
        ),
        "supervised_contrastive_effective_loss": (
            total_supcon_effective_loss / max(found, 1)
        ),
        "supervised_contrastive_positive_pairs": total_supcon_positive_pairs,
        "supervised_contrastive_negative_pairs": total_supcon_negative_pairs,
        "supervised_contrastive_valid_anchors": total_supcon_valid_anchors,
        "train_samples": found,
        "balanced_samples": usable,
        "not_found": usable - found,
    }


def ewc_penalty(model, consolidations: list[dict], coefficient: float) -> torch.Tensor:
    parameters = dict(model.named_parameters())
    first = next(iter(parameters.values()))
    penalty = first.new_zeros(())
    for consolidation in consolidations:
        for name, parameter in parameters.items():
            fisher = consolidation["fisher"][name]
            anchor = consolidation["anchor"][name]
            penalty = penalty + (fisher * (parameter - anchor).pow(2)).sum()
    return coefficient * penalty


def move_consolidations(consolidations: list[dict], device: torch.device) -> list[dict]:
    return [
        {
            "stage": item["stage"],
            "samples": item["samples"],
            "anchor": {name: value.to(device) for name, value in item["anchor"].items()},
            "fisher": {name: value.to(device) for name, value in item["fisher"].items()},
        }
        for item in consolidations
    ]


def estimate_diagonal_fisher(
    model,
    adapter,
    lines: list[str],
    device: torch.device,
    max_samples: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int, float]:
    """Estimate empirical diagonal Fisher from a fixed balanced subset."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    rng = random.Random(seed)
    positives = [line for line in lines if int(line.split()[3]) == 1]
    negatives = [line for line in lines if int(line.split()[3]) == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    per_class = min(len(positives), len(negatives), max_samples // 2)
    selected = positives[:per_class] + negatives[:per_class]
    rng.shuffle(selected)
    parameters = dict(model.named_parameters())
    fisher = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
    was_training = model.training
    # cuDNN RNNs only retain the backward reserve space in training mode.
    # Stage evaluation leaves the CG-LSMN model in eval mode, so empirical
    # Fisher estimation must temporarily switch it back before backpropagating.
    model.train()
    started = time.perf_counter()
    found = 0
    try:
        for index, line in enumerate(selected, 1):
            tensors = adapter.graph_tensors(line, device)
            if tensors is None or tensors[1].numel() == 0 or tensors[3].numel() == 0:
                continue
            h1, e1, h2, e2, label = tensors
            model.zero_grad(set_to_none=True)
            probabilities = model(h1, e1, h2, e2)
            negative_log_likelihood = -torch.log(
                probabilities[0, label].clamp_min(1e-8)
            )
            negative_log_likelihood.backward()
            for name, parameter in parameters.items():
                if parameter.grad is not None:
                    fisher[name].add_(parameter.grad.detach().pow(2))
            found += 1
            if index % 128 == 0 or index == len(selected):
                print(f"estimate Fisher {index}/{len(selected)}", flush=True)
        if not found:
            raise RuntimeError("no usable samples for Fisher estimation")
        for value in fisher.values():
            value.div_(found)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    return fisher, found, time.perf_counter() - started


@torch.no_grad()
def evaluate(model, adapter, lines: list[str], device: torch.device, name: str) -> dict:
    model.eval()
    started = time.perf_counter()
    y_true: list[int] = []
    y_pred: list[int] = []
    positive_scores: list[float] = []
    all_scores: list[list[float]] = []
    not_found = 0
    for index, line in enumerate(lines, 1):
        tensors = adapter.graph_tensors(line, device)
        if tensors is None:
            not_found += 1
            continue
        h1, e1, h2, e2, label = tensors
        output = model(h1, e1, h2, e2)
        scores = output[0].detach().cpu().tolist()
        y_true.append(label)
        y_pred.append(int(output.argmax(dim=1).item()))
        positive_scores.append(scores[1])
        all_scores.append(scores)
        if index % 1000 == 0 or index == len(lines):
            print(f"evaluate {name} {index}/{len(lines)}", flush=True)
    metrics = classification_metrics(y_true, y_pred, positive_scores, all_scores)
    metrics.update({
        "inference_seconds": time.perf_counter() - started,
        "samples": len(y_true),
        "not_found": not_found,
    })
    return metrics


def select_replay_case_ids(rows: list[dict], ratio: float, seed: int) -> list[str]:
    """Select a deterministic, project-diverse buffer of complete cases."""
    if not 0.0 < ratio < 1.0:
        raise ValueError("replay ratio must be between zero and one")
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    target = max(1, round(len(by_case) * ratio))
    by_project: dict[str, list[str]] = defaultdict(list)
    for case_id, case_rows in by_case.items():
        projects = {row["split_project"] for row in case_rows}
        if len(projects) != 1:
            raise ValueError(f"replay case spans projects: {case_id}")
        by_project[next(iter(projects))].append(case_id)

    rng = random.Random(seed)
    projects = sorted(by_project)
    rng.shuffle(projects)
    for project in projects:
        cases = sorted(by_project[project])
        rng.shuffle(cases)
        # Prefer complete pre/post and binary-label cases, preserving random
        # ordering among equally complete candidates.
        cases.sort(
            key=lambda case_id: (
                {row["state"] for row in by_case[case_id]} >= {"pre", "post"}
                and {row["label"] for row in by_case[case_id]} >= {"0", "1"}
            ),
            # ``pop()`` below consumes from the end, so complete cases sort
            # after incomplete ones and are selected first.
            reverse=False,
        )
        by_project[project] = cases

    selected: list[str] = []
    while len(selected) < target:
        progressed = False
        for project in projects:
            if by_project[project] and len(selected) < target:
                selected.append(by_project[project].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != target:
        raise AssertionError(f"selected {len(selected)} replay cases, expected {target}")
    return sorted(selected)


def rows_for_cases(rows: Iterable[dict], case_ids: Iterable[str]) -> list[dict]:
    selected = set(case_ids)
    return [row for row in rows if row["case_id"] in selected]
