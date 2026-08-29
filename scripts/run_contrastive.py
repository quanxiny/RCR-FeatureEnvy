#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.contrastive_sampler import (
    ContrastiveBatchSampler,
    as_contrastive_samples,
    count_contrastive_relations,
    load_metadata_index,
)
from src.data.directed_anchors import build_directed_anchor_index
from src.data.fixed_splits import inner_project_validation
from src.data.original_adapter import OriginalDatasetAdapter
from src.data.token_statistics import (
    informative_token_mask,
    load_token_statistics,
)
from src.data.task1_directed_anchors import (
    build_task1_directed_anchor_index,
    orient_source_target,
)
from src.losses.case_contrastive import (
    case_aware_contrastive_loss,
    pre_post_hard_contrastive_loss,
)
from src.losses.original_loss import FocalLoss
from src.losses.local_matching_contrastive import (
    STATISTIC_KEYS as LOCAL_STATISTIC_KEYS,
    local_graph_pair_contrastive_loss,
)
from src.losses.directed_relation_contrastive import (
    STATISTIC_KEYS as DIRECTED_STATISTIC_KEYS,
    directed_method_class_contrastive_loss,
)
from src.losses.supervised_contrastive import supervised_contrastive_loss
from src.models.cglsmn_baseline import CGLSMNBaseline
from src.models.contrastive_head import PairProjectionHead
from src.utils.checkpoint import atomic_torch_save
from src.utils.metrics import classification_metrics
from src.utils.seed import capture_rng_state, restore_rng_state, seed_everything


METHODS = ("CL0", "CL1", "CL2", "CL3", "CL4", "CL5", "CL6", "CL7")
DIRECTED_METHODS = {"CL6", "CL7"}
LOCAL_METHODS = {"CL5", *DIRECTED_METHODS}


def one_hot_target(label: int, device):
    return torch.tensor([[0.0, 1.0] if label else [1.0, 0.0]], device=device)


def balanced_epoch(lines: list[str], rng: random.Random) -> list[str]:
    shuffled = list(lines)
    rng.shuffle(shuffled)
    positives = [line for line in shuffled if int(line.split()[3]) == 1]
    negatives = [line for line in shuffled if int(line.split()[3]) == 0]
    count = min(len(positives), len(negatives))
    selected = positives[:count] + negatives[:count]
    rng.shuffle(selected)
    return selected


def train_epoch(model, projection, optimizer, criterion, adapter, train_lines,
                metadata, method, batch_size, device, rng, epoch,
                temperature, lambda_label, margin, lambda_hard, lambda_case,
                max_batches=None, local_options=None):
    model.train()
    if projection is not None:
        projection.train()
    selected = balanced_epoch(train_lines, rng)
    usable = (len(selected) // batch_size) * batch_size
    selected = selected[:usable]
    contexts = as_contrastive_samples(selected, metadata)
    if method == "CL0" or method in LOCAL_METHODS:
        batches = [contexts[start:start + batch_size]
                   for start in range(0, usable, batch_size)]
    else:
        batches = ContrastiveBatchSampler(contexts, batch_size, rng)

    started = time.perf_counter()
    total_classification = 0.0
    total_label = 0.0
    total_hard = 0.0
    total_case = 0.0
    total_local = 0.0
    total_local_statistics = {key: 0 for key in LOCAL_STATISTIC_KEYS}
    total_directed = {
        "directed_evidence_loss": 0.0,
        "directed_mutual_loss": 0.0,
        "directed_negative_loss": 0.0,
        "directed_effective_aux_loss": 0.0,
        "directed_aux_scale": 0.0,
        "directed_ramp": 0.0,
        "directed_contributing_samples": 0,
    }
    total_directed_statistics = {
        key: 0 for key in DIRECTED_STATISTIC_KEYS
    }
    correct = 0
    found = 0
    batch_records = []
    auxiliary_enabled = (
        method == "CL5"
        and local_options["lambda"] > 0.0
    ) or (
        method in DIRECTED_METHODS
        and any(
            local_options[name] > 0.0
            for name in (
                "lambda_positive",
                "lambda_mutual",
                "lambda_negative",
            )
        )
    )
    auxiliary_active = (
        auxiliary_enabled
        and epoch >= local_options["warmup_epochs"]
    )
    per_sample_backward = method in LOCAL_METHODS and auxiliary_active
    chunked_classification_backward = (
        method in LOCAL_METHODS and not auxiliary_active
    )
    classification_backward_chunk = 16
    for batch_index, batch in enumerate(batches):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        classification_loss = None
        classification_pending = 0
        classification_value = 0.0
        local_value = 0.0
        local_statistics = {}
        local_contributing_samples = 0
        directed_values = {
            "directed_evidence_loss": 0.0,
            "directed_mutual_loss": 0.0,
            "directed_negative_loss": 0.0,
            "directed_effective_aux_loss": 0.0,
            "directed_aux_scale": 0.0,
            "directed_ramp": 0.0,
            "directed_contributing_samples": 0,
        }
        directed_statistics = {}
        pair_embeddings = []
        used = []
        batch_correct = 0
        for sample in batch:
            tensors = adapter.graph_tensors(sample.line, device)
            if tensors is None or tensors[1].numel() == 0 or tensors[3].numel() == 0:
                continue
            h1, e1, h2, e2, label = tensors
            if method == "CL0" or (
                    method in LOCAL_METHODS and not auxiliary_active):
                probabilities = model(h1, e1, h2, e2)
            elif method in LOCAL_METHODS:
                output = model(
                    h1, e1, h2, e2, return_local_features=True)
                probabilities = output["probabilities"]
            else:
                output = model(h1, e1, h2, e2, return_features=True)
                probabilities = output["probabilities"]
                pair_embeddings.append(output["pair_embedding"])
            loss = criterion(probabilities, one_hot_target(label, device))
            if method == "CL5":
                local_loss = loss.new_zeros(())
                sample_statistics = {}
                if auxiliary_active:
                    local_loss, sample_statistics = local_graph_pair_contrastive_loss(
                        output["local_features"], h1, h2,
                        local_options["informative_tokens"], label,
                        temperature=local_options["temperature"],
                        positive_threshold=local_options["positive_threshold"],
                        negative_threshold=local_options["negative_threshold"],
                        negative_margin=local_options["negative_margin"],
                        negative_pair_margin=local_options["negative_pair_margin"],
                        negatives_per_anchor=local_options["negatives_per_anchor"],
                        max_positive_anchors=local_options["max_positive_anchors"],
                        max_negative_anchors=local_options["max_negative_anchors"])
                    local_contributing_samples += int(
                        sample_statistics["local_positive_pairs"] > 0
                        or sample_statistics["local_hard_negative_pairs"] > 0)
                    for key, value in sample_statistics.items():
                        local_statistics[key] = local_statistics.get(key, 0) + value
                sample_joint = loss + local_options["lambda"] * local_loss
                if not torch.isfinite(sample_joint):
                    raise FloatingPointError(
                        f"non-finite CL5 loss at epoch {epoch} batch {batch_index}")
                if per_sample_backward:
                    # Active CL5 retains a large local NxM similarity graph.
                    # Release it before the next graph pair while retaining one
                    # optimizer step per batch.
                    sample_joint.backward()
                else:
                    classification_loss = (
                        loss
                        if classification_loss is None
                        else classification_loss + loss
                    )
                    classification_pending += 1
                    if classification_pending >= classification_backward_chunk:
                        classification_loss.backward()
                        classification_loss = None
                        classification_pending = 0
                classification_value += float(loss.detach())
                local_value += float(local_loss.detach())
            elif method in DIRECTED_METHODS:
                component_losses = {
                    name: loss.new_zeros(())
                    for name in ("evidence", "mutual", "negative")
                }
                sample_statistics = {
                    key: 0 for key in DIRECTED_STATISTIC_KEYS
                }
                effective_aux = loss.new_zeros(())
                scale = 0.0
                ramp = 0.0
                if auxiliary_active:
                    anchor_entry = local_options["anchors"].get(sample.line)
                    if method == "CL7":
                        anchor_entry = anchor_entry or {
                            "source_graph": 0,
                            "anchor_token_ids": (),
                        }
                        source_graph = int(anchor_entry["source_graph"])
                        anchor_values = anchor_entry["anchor_token_ids"]
                    else:
                        source_graph = 1
                        anchor_values = anchor_entry or ()
                    anchor_ids = torch.as_tensor(
                        anchor_values,
                        dtype=torch.long,
                        device=device,
                    )
                    if source_graph:
                        oriented_features, source_tokens, target_tokens = (
                            orient_source_target(
                                output["local_features"], h1, h2, source_graph
                            )
                        )
                        component_losses, sample_statistics = (
                            directed_method_class_contrastive_loss(
                                oriented_features,
                                source_tokens,
                                target_tokens,
                                anchor_ids,
                                local_options["informative_tokens"],
                                label,
                                temperature=local_options["temperature"],
                                mutual_threshold=local_options["mutual_threshold"],
                                negative_margin=local_options["negative_margin"],
                                negatives_per_anchor=local_options["negatives_per_anchor"],
                                max_anchors=local_options["max_anchors"],
                                negative_topk=local_options["negative_topk"],
                                enable_mutual=(
                                    local_options["lambda_mutual"] > 0.0
                                ),
                                enable_negative=(
                                    local_options["lambda_negative"] > 0.0
                                ),
                            )
                        )
                    weighted_aux = (
                        local_options["lambda_positive"]
                        * component_losses["evidence"]
                        + local_options["lambda_mutual"]
                        * component_losses["mutual"]
                        + local_options["lambda_negative"]
                        * component_losses["negative"]
                    )
                    ramp = min(
                        1.0,
                        (
                            epoch
                            - local_options["warmup_epochs"]
                            + 1
                        )
                        / max(local_options["ramp_epochs"], 1),
                    )
                    weighted_value = float(weighted_aux.detach())
                    if weighted_value > 0.0:
                        maximum_aux = (
                            local_options["max_aux_ratio"]
                            * float(loss.detach())
                        )
                        scale = min(
                            1.0, maximum_aux / max(weighted_value, 1e-12)
                        )
                    effective_aux = weighted_aux * (ramp * scale)
                    contributing = int(
                        (
                            local_options["lambda_positive"] > 0.0
                            and sample_statistics["directed_evidence_pairs"] > 0
                        )
                        or (
                            local_options["lambda_mutual"] > 0.0
                            and sample_statistics["directed_mutual_pairs"] > 0
                        )
                        or (
                            local_options["lambda_negative"] > 0.0
                            and sample_statistics[
                                "directed_negative_pair_matches"
                            ]
                            > 0
                        )
                    )
                    directed_values["directed_contributing_samples"] += contributing
                    for key, value in sample_statistics.items():
                        directed_statistics[key] = (
                            directed_statistics.get(key, 0) + value
                        )
                sample_joint = loss + effective_aux
                if not torch.isfinite(sample_joint):
                    raise FloatingPointError(
                        f"non-finite {method} loss at epoch {epoch} "
                        f"batch {batch_index}"
                    )
                if per_sample_backward:
                    sample_joint.backward()
                else:
                    classification_loss = (
                        loss
                        if classification_loss is None
                        else classification_loss + loss
                    )
                    classification_pending += 1
                    if classification_pending >= classification_backward_chunk:
                        classification_loss.backward()
                        classification_loss = None
                        classification_pending = 0
                classification_value += float(loss.detach())
                directed_values["directed_evidence_loss"] += float(
                    component_losses["evidence"].detach()
                )
                directed_values["directed_mutual_loss"] += float(
                    component_losses["mutual"].detach()
                )
                directed_values["directed_negative_loss"] += float(
                    component_losses["negative"].detach()
                )
                directed_values["directed_effective_aux_loss"] += float(
                    effective_aux.detach()
                )
                directed_values["directed_aux_scale"] += scale
                directed_values["directed_ramp"] += ramp
            else:
                classification_loss = (
                    loss if classification_loss is None
                    else classification_loss + loss)
            batch_correct += int(probabilities.argmax(dim=1).item() == label)
            used.append(sample)
        if not used:
            continue

        reference_loss = loss
        label_loss = reference_loss.new_zeros(())
        hard_loss = reference_loss.new_zeros(())
        case_loss = reference_loss.new_zeros(())
        if method != "CL0" and method not in LOCAL_METHODS:
            embeddings = torch.cat(pair_embeddings, dim=0)
            projected = F.normalize(projection(embeddings), dim=-1)
            labels = torch.tensor([sample.label for sample in used], device=device)
            canonical_ids = [sample.canonical_sample_id for sample in used]
            case_ids = [sample.case_id for sample in used]
            states = [sample.state for sample in used]
            if method in {"CL1", "CL4"}:
                label_loss, _ = supervised_contrastive_loss(
                    projected, labels, temperature, canonical_ids)
            if method in {"CL2", "CL4"}:
                hard_loss, _ = pre_post_hard_contrastive_loss(
                    projected, case_ids, states, labels, canonical_ids, margin)
            if method == "CL3":
                case_loss, _ = case_aware_contrastive_loss(
                    projected, case_ids, states, labels, canonical_ids, temperature)

        sample_count = len(used)
        if chunked_classification_backward:
            if classification_loss is not None:
                classification_loss.backward()
        elif not per_sample_backward:
            joint_loss = classification_loss
            if method in {"CL1", "CL4"}:
                joint_loss = joint_loss + sample_count * lambda_label * label_loss
            if method in {"CL2", "CL4"}:
                joint_loss = joint_loss + sample_count * lambda_hard * hard_loss
            if method == "CL3":
                joint_loss = joint_loss + sample_count * lambda_case * case_loss
            if not torch.isfinite(joint_loss):
                raise FloatingPointError(
                    f"non-finite {method} loss at epoch {epoch} batch {batch_index}")
            joint_loss.backward()
            classification_value = float(classification_loss.detach())
        optimizer.step()

        relations = count_contrastive_relations(used)
        batch_records.append({
            "epoch": epoch,
            "batch": batch_index,
            "samples": sample_count,
            "classification_loss": classification_value / sample_count,
            "label_contrastive_loss": float(label_loss.detach()),
            "hard_contrastive_loss": float(hard_loss.detach()),
            "case_contrastive_loss": float(case_loss.detach()),
            **relations,
        })
        if method == "CL5":
            batch_records[-1].update({
                "local_contrastive_loss": local_value / sample_count,
                "local_contributing_samples": local_contributing_samples,
                **local_statistics,
            })
            total_local += local_value
            for key in LOCAL_STATISTIC_KEYS:
                total_local_statistics[key] += local_statistics.get(key, 0)
        elif method in DIRECTED_METHODS:
            batch_records[-1].update(
                {
                    key: value / sample_count
                    if key != "directed_contributing_samples"
                    else value
                    for key, value in directed_values.items()
                }
            )
            batch_records[-1].update(directed_statistics)
            for key, value in directed_values.items():
                total_directed[key] += value
            for key in DIRECTED_STATISTIC_KEYS:
                total_directed_statistics[key] += directed_statistics.get(key, 0)
        total_classification += classification_value
        total_label += float(label_loss.detach())
        total_hard += float(hard_loss.detach())
        total_case += float(case_loss.detach())
        correct += batch_correct
        found += sample_count
        completed = (batch_index + 1) * batch_size
        if completed % (batch_size * 25) == 0:
            print(
                f"train {completed}/{usable} class={total_classification/max(found,1):.6f} "
                f"acc={correct/max(found,1):.4f}", flush=True)
    elapsed = time.perf_counter() - started
    batch_count = max(len(batch_records), 1)
    relation_totals = {
        key: sum(row[key] for row in batch_records)
        for key in (
            "label_positive_pairs", "ordinary_negative_pairs",
            "same_case_same_state_positive_pairs", "pre_post_hard_negative_pairs")
    }
    metrics = {
        "train_seconds": elapsed,
        "train_loss": total_classification / max(found, 1),
        "train_accuracy": correct / max(found, 1),
        "train_samples": found,
        "label_contrastive_loss": total_label / batch_count,
        "hard_contrastive_loss": total_hard / batch_count,
        "case_contrastive_loss": total_case / batch_count,
        **relation_totals,
    }
    if method == "CL5":
        metrics.update({
            "local_contrastive_loss": total_local / max(found, 1),
            **total_local_statistics,
        })
    elif method in DIRECTED_METHODS:
        metrics.update(
            {
                key: value / max(found, 1)
                if key != "directed_contributing_samples"
                else value
                for key, value in total_directed.items()
            }
        )
        metrics.update(total_directed_statistics)
    return metrics, batch_records


@torch.no_grad()
def evaluate(model, adapter, lines, device, max_samples=None):
    model.eval()
    started = time.perf_counter()
    y_true, y_pred, positive_scores, all_scores = [], [], [], []
    not_found = 0
    limit = min(len(lines), max_samples) if max_samples else len(lines)
    for index, line in enumerate(lines[:limit], 1):
        tensors = adapter.graph_tensors(line, device)
        if tensors is None:
            not_found += 1
            continue
        h1, e1, h2, e2, label = tensors
        probabilities = model(h1, e1, h2, e2)
        scores = probabilities[0].detach().cpu().tolist()
        y_true.append(label)
        y_pred.append(int(probabilities.argmax(dim=1).item()))
        positive_scores.append(scores[1])
        all_scores.append(scores)
        if index % 1000 == 0 or index == limit:
            print(f"evaluate {index}/{limit}", flush=True)
    metrics = classification_metrics(y_true, y_pred, positive_scores, all_scores)
    metrics.update({
        "inference_seconds": time.perf_counter() - started,
        "test_samples": len(y_true),
        "not_found": not_found,
    })
    return metrics


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(path)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_completed_epochs(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["epoch"]) for row in csv.DictReader(handle)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Model/sampler seed; the project-disjoint folds remain fixed.",
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum tuning epochs; validation early stopping may finish sooner.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-label", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--lambda-hard", type=float, default=0.10)
    parser.add_argument("--lambda-case", type=float, default=0.10)
    parser.add_argument("--lambda-local", type=float, default=0.10)
    parser.add_argument("--local-temperature", type=float, default=0.10)
    parser.add_argument("--local-positive-threshold", type=float, default=0.70)
    parser.add_argument("--local-negative-threshold", type=float, default=0.25)
    parser.add_argument("--local-negative-margin", type=float, default=0.10)
    parser.add_argument("--local-negative-pair-margin", type=float, default=0.30)
    parser.add_argument("--local-negatives", type=int, default=16)
    parser.add_argument("--local-max-positive-anchors", type=int, default=64)
    parser.add_argument("--local-max-negative-anchors", type=int, default=64)
    parser.add_argument("--local-max-document-ratio", type=float, default=0.05)
    parser.add_argument("--local-warmup-epochs", type=int, default=1)
    parser.add_argument("--directed-lambda-positive", type=float, default=0.01)
    parser.add_argument("--directed-lambda-mutual", type=float, default=0.0)
    parser.add_argument("--directed-lambda-negative", type=float, default=0.0)
    parser.add_argument("--directed-temperature", type=float, default=0.10)
    parser.add_argument("--directed-mutual-threshold", type=float, default=0.80)
    parser.add_argument("--directed-negative-margin", type=float, default=0.65)
    parser.add_argument("--directed-negatives", type=int, default=16)
    parser.add_argument("--directed-max-anchors", type=int, default=64)
    parser.add_argument("--directed-negative-topk", type=int, default=8)
    parser.add_argument("--directed-warmup-epochs", type=int, default=3)
    parser.add_argument("--directed-ramp-epochs", type=int, default=2)
    parser.add_argument("--directed-max-aux-ratio", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--validation-seed", type=int, default=101)
    parser.add_argument("--selection-metric", choices=("mcc", "pr_auc", "f1"),
                        default="mcc")
    parser.add_argument("--patience", type=int, default=5,
                        help="Validation epochs without improvement before stopping; 0 disables.")
    parser.add_argument("--minimum-epochs", type=int, default=1)
    parser.add_argument("--full-train-epochs", type=int,
                        help="Disable validation and train on the full outer train fold for this many epochs.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Tuning stage only: stop after selecting the validation checkpoint.")
    parser.add_argument(
        "--legacy-test-selection", action="store_true",
        help="Paper-compatible secondary protocol: train on the full outer fold "
             "and select epochs directly by outer-test metrics.")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("CGLSMN_DATA_ROOT", PROJECT_ROOT / "data")),
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=PROJECT_ROOT / "artifacts" / "metadata" / "all_samples.csv")
    parser.add_argument(
        "--token-statistics", type=Path,
        help="Token document-frequency JSON; defaults to the selected task artifact.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gate:
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_samples = args.max_eval_samples or 256
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    phase = "gates" if args.gate else "runs"
    run_name = (
        f"{args.method.lower()}_task{args.task}_fold{args.fold}_seed{args.seed}_{args.tag}")
    result_dir = PROJECT_ROOT / "results" / "contrastive" / phase / run_name
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "contrastive" / phase / run_name
    curve_path = result_dir / "training_curves.csv"
    relation_path = result_dir / "batch_relations.jsonl"
    test_path = result_dir / "test_metrics.csv"
    tuning_path = result_dir / "tuning_complete.json"
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    existing = [curve_path, relation_path, test_path, tuning_path, last_path, best_path]
    if args.fresh and any(path.exists() for path in existing):
        raise RuntimeError("--fresh refuses to overwrite an existing run")
    completion_path = tuning_path if args.skip_test else test_path
    if completion_path.exists():
        print(f"{run_name} already complete; skipping")
        return

    print(f"loading immutable task {args.task} graph cache", flush=True)
    adapter = OriginalDatasetAdapter(args.data_root, args.task, split_seed=100)
    outer_train, outer_test = adapter.fold(args.fold)
    metadata = load_metadata_index(args.metadata, args.task, args.fold, "train")
    if args.full_train_epochs is not None:
        if args.full_train_epochs < 1:
            raise ValueError("--full-train-epochs must be positive")
        if args.skip_test:
            raise ValueError("--full-train-epochs and --skip-test are mutually exclusive")
        train_lines = outer_train
        selection_lines = []
        selection_split = "fixed_epoch_full_train"
        validation_projects = set()
        target_epochs = args.full_train_epochs
    elif args.legacy_test_selection:
        if args.skip_test:
            raise ValueError(
                "--legacy-test-selection and --skip-test are mutually exclusive")
        train_lines = outer_train
        selection_lines = outer_test
        selection_split = "test_legacy"
        validation_projects = set()
        target_epochs = args.epochs
    elif args.method == "CL0":
        train_lines = outer_train
        selection_lines = outer_test
        selection_split = "test_legacy"
        validation_projects = set()
        target_epochs = args.epochs
    else:
        train_lines, selection_lines, validation_projects = inner_project_validation(
            outer_train, args.task, args.fold,
            args.validation_fraction, args.validation_seed)
        selection_split = "validation"
        target_epochs = args.epochs
    print(
        f"{run_name}: train={len(train_lines)} selection={len(selection_lines)} "
        f"test={len(outer_test)} selection_split={selection_split} device={device}",
        flush=True)

    model = CGLSMNBaseline(adapter.vocab_size).to(device)
    projection = (
        None if args.method == "CL0" or args.method in LOCAL_METHODS
        else PairProjectionHead().to(device))
    local_options = None
    if args.method == "CL5":
        if args.local_warmup_epochs < 0:
            raise ValueError("--local-warmup-epochs cannot be negative")
        token_statistics_path = (
            args.token_statistics
            or PROJECT_ROOT / "artifacts" / "token_statistics"
            / f"task{args.task}.json")
        if not token_statistics_path.exists():
            raise FileNotFoundError(
                f"missing {token_statistics_path}; run build_token_statistics.py "
                f"--task {args.task}")
        token_statistics = load_token_statistics(token_statistics_path)
        if int(token_statistics["vocab_size"]) != adapter.vocab_size:
            raise ValueError(
                "token statistics vocabulary does not match the selected task")
        informative_tokens = informative_token_mask(
            token_statistics, args.local_max_document_ratio, device)
        local_options = {
            "informative_tokens": informative_tokens,
            "lambda": args.lambda_local,
            "temperature": args.local_temperature,
            "positive_threshold": args.local_positive_threshold,
            "negative_threshold": args.local_negative_threshold,
            "negative_margin": args.local_negative_margin,
            "negative_pair_margin": args.local_negative_pair_margin,
            "negatives_per_anchor": args.local_negatives,
            "max_positive_anchors": args.local_max_positive_anchors,
            "max_negative_anchors": args.local_max_negative_anchors,
            "warmup_epochs": args.local_warmup_epochs,
        }
        print(
            f"CL5 informative tokens={int(informative_tokens.sum())}/"
            f"{informative_tokens.numel()} max_document_ratio="
            f"{args.local_max_document_ratio}", flush=True)
    elif args.method in DIRECTED_METHODS:
        expected_task = 2 if args.method == "CL6" else 1
        if args.task != expected_task:
            description = (
                "method→class" if args.method == "CL6"
                else "source-class→target-class"
            )
            raise ValueError(
                f"{args.method} directed {description} loss supports "
                f"Task {expected_task} only"
            )
        if args.directed_warmup_epochs < 0:
            raise ValueError("--directed-warmup-epochs cannot be negative")
        if args.directed_ramp_epochs < 1:
            raise ValueError("--directed-ramp-epochs must be positive")
        if not 0.0 < args.directed_max_aux_ratio <= 1.0:
            raise ValueError("--directed-max-aux-ratio must be in (0, 1]")
        for name, value in (
            ("positive", args.directed_lambda_positive),
            ("mutual", args.directed_lambda_mutual),
            ("negative", args.directed_lambda_negative),
        ):
            if value < 0.0:
                raise ValueError(f"directed lambda {name} cannot be negative")
        token_statistics_path = (
            args.token_statistics
            or PROJECT_ROOT / "artifacts" / "token_statistics"
            / f"task{args.task}.json")
        if not token_statistics_path.exists():
            raise FileNotFoundError(
                f"missing {token_statistics_path}; run build_token_statistics.py "
                f"--task {args.task}")
        token_statistics = load_token_statistics(token_statistics_path)
        if int(token_statistics["vocab_size"]) != adapter.vocab_size:
            raise ValueError(
                "token statistics vocabulary does not match the selected task")
        informative_tokens = informative_token_mask(
            token_statistics, args.local_max_document_ratio, device)
        if args.method == "CL6":
            directed_anchors, directed_anchor_audit = (
                build_directed_anchor_index(metadata, adapter.vocab)
            )
        else:
            directed_anchors, directed_anchor_audit = (
                build_task1_directed_anchor_index(metadata, adapter.vocab)
            )
        local_options = {
            "anchors": directed_anchors,
            "informative_tokens": informative_tokens,
            "lambda_positive": args.directed_lambda_positive,
            "lambda_mutual": args.directed_lambda_mutual,
            "lambda_negative": args.directed_lambda_negative,
            "temperature": args.directed_temperature,
            "mutual_threshold": args.directed_mutual_threshold,
            "negative_margin": args.directed_negative_margin,
            "negatives_per_anchor": args.directed_negatives,
            "max_anchors": args.directed_max_anchors,
            "negative_topk": args.directed_negative_topk,
            "warmup_epochs": args.directed_warmup_epochs,
            "ramp_epochs": args.directed_ramp_epochs,
            "max_aux_ratio": args.directed_max_aux_ratio,
        }
        print(
            f"{args.method} directed anchor audit="
            f"{json.dumps(directed_anchor_audit, sort_keys=True)} "
            f"informative_tokens={int(informative_tokens.sum())}/"
            f"{informative_tokens.numel()}",
            flush=True,
        )
    parameters = list(model.parameters())
    if projection is not None:
        parameters += list(projection.parameters())
    optimizer = optim.Adam(
        parameters, lr=0.001, weight_decay=args.weight_decay)
    criterion = FocalLoss().to(device)
    sampling_rng = random.Random(args.seed + args.task * 100 + args.fold)
    start_epoch = 0
    best_score = -math.inf
    epochs_without_improvement = 0
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        if projection is not None:
            projection.load_state_dict(state["projection"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["next_epoch"])
        best_score = float(state.get("best_score", state.get("best_f1", -math.inf)))
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        restore_rng_state(state["rng"], sampling_rng)
        print(f"resuming at epoch {start_epoch}", flush=True)
    completed_epochs = read_completed_epochs(curve_path)

    for epoch in range(start_epoch, target_epochs):
        print(f"epoch {epoch}/{target_epochs - 1}", flush=True)
        train_metrics, batch_records = train_epoch(
            model, projection, optimizer, criterion, adapter, train_lines,
            metadata, args.method, args.batch_size, device, sampling_rng, epoch,
            args.temperature, args.lambda_label, args.margin, args.lambda_hard,
            args.lambda_case, args.max_train_batches, local_options)
        if selection_split == "fixed_epoch_full_train":
            selection_metrics = {}
            selection_score = float(epoch)
            selection_eligible = True
            is_best = True
        else:
            selection_metrics = evaluate(
                model, adapter, selection_lines, device, args.max_eval_samples)
            selection_score = float(selection_metrics[args.selection_metric])
            selection_eligible = not (
                args.method in DIRECTED_METHODS
                and epoch < local_options["warmup_epochs"]
            )
            is_best = selection_eligible and selection_score > best_score
        if is_best:
            best_score = selection_score
            epochs_without_improvement = 0
            atomic_torch_save({
                "model": model.state_dict(),
                "projection": projection.state_dict() if projection is not None else None,
                "method": args.method,
                "task": args.task,
                "fold": args.fold,
                "seed": args.seed,
                "epoch": epoch,
                "selection_split": selection_split,
                "selection_metric": args.selection_metric,
                "selection_score": selection_score,
                "selection_metrics": selection_metrics,
                "parameters": vars(args),
            }, best_path)
        elif selection_eligible:
            epochs_without_improvement += 1
        curve = {
            "method": args.method,
            "task": args.task,
            "fold": args.fold,
            "seed": args.seed,
            "epoch": epoch,
            "selection_split": selection_split,
            "selection_metric": args.selection_metric,
            "selection_score": selection_score,
            "selection_eligible": int(selection_eligible),
            **train_metrics,
            **{f"selection_{key}": value for key, value in selection_metrics.items()},
            "is_best": int(is_best),
        }
        if epoch not in completed_epochs:
            append_csv(curve_path, curve)
            append_jsonl(relation_path, batch_records)
        atomic_torch_save({
            "model": model.state_dict(),
            "projection": projection.state_dict() if projection is not None else None,
            "optimizer": optimizer.state_dict(),
            "next_epoch": epoch + 1,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "rng": capture_rng_state(sampling_rng),
        }, last_path)
        print(json.dumps(curve, sort_keys=True), flush=True)

        if (selection_split in {"validation", "test_legacy"}
                and args.patience > 0
                and epoch + 1 >= args.minimum_epochs
                and epochs_without_improvement >= args.patience):
            print(
                f"early stopping after epoch {epoch}; best {args.selection_metric}="
                f"{best_score:.6f}", flush=True)
            break

    if args.skip_test:
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        write_json(tuning_path, {
            "method": args.method,
            "task": args.task,
            "fold": args.fold,
            "seed": args.seed,
            "selection_metric": args.selection_metric,
            "selection_score": best["selection_score"],
            "selected_epoch": best["epoch"],
            "selected_train_epochs": int(best["epoch"]) + 1,
            "selection_metrics": best["selection_metrics"],
            "validation_projects": sorted(validation_projects),
        })
        # The immutable best checkpoint is the tuning deliverable.  The
        # optimizer-heavy recovery checkpoint is useful only before completion.
        last_path.unlink(missing_ok=True)
        print(tuning_path.read_text(encoding="utf-8"), flush=True)
        return

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    if projection is not None:
        projection.load_state_dict(best["projection"], strict=True)
    test_metrics = evaluate(model, adapter, outer_test, device, args.max_eval_samples)
    result = {
        "method": args.method,
        "task": args.task,
        "fold": args.fold,
        "seed": args.seed,
        "gate": int(args.gate),
        "selected_epoch": best["epoch"],
        "selection_split": selection_split,
        "selection_metric": args.selection_metric,
        "selection_score": best.get("selection_score", ""),
        "training_protocol": (
            "full_train_fixed_epoch" if args.full_train_epochs is not None
            else "legacy_test_selected" if args.legacy_test_selection
            else "validation_selected_development"),
        "validation_projects": ";".join(sorted(validation_projects)),
        "temperature": args.temperature,
        "lambda_label": args.lambda_label,
        "margin": args.margin,
        "lambda_hard": args.lambda_hard,
        "lambda_case": args.lambda_case,
        "weight_decay": args.weight_decay,
        "lambda_local": args.lambda_local if args.method == "CL5" else "",
        "local_temperature": (
            args.local_temperature if args.method == "CL5" else ""),
        "local_max_document_ratio": (
            args.local_max_document_ratio if args.method == "CL5" else ""),
        "local_positive_threshold": (
            args.local_positive_threshold if args.method == "CL5" else ""),
        "local_negative_threshold": (
            args.local_negative_threshold if args.method == "CL5" else ""),
        "local_negative_pair_margin": (
            args.local_negative_pair_margin if args.method == "CL5" else ""),
        "directed_lambda_positive": (
            args.directed_lambda_positive
            if args.method in DIRECTED_METHODS else ""),
        "directed_lambda_mutual": (
            args.directed_lambda_mutual
            if args.method in DIRECTED_METHODS else ""),
        "directed_lambda_negative": (
            args.directed_lambda_negative
            if args.method in DIRECTED_METHODS else ""),
        "directed_mutual_threshold": (
            args.directed_mutual_threshold
            if args.method in DIRECTED_METHODS else ""),
        "directed_negative_margin": (
            args.directed_negative_margin
            if args.method in DIRECTED_METHODS else ""),
        "directed_warmup_epochs": (
            args.directed_warmup_epochs
            if args.method in DIRECTED_METHODS else ""),
        "directed_ramp_epochs": (
            args.directed_ramp_epochs
            if args.method in DIRECTED_METHODS else ""),
        "directed_max_aux_ratio": (
            args.directed_max_aux_ratio
            if args.method in DIRECTED_METHODS else ""),
        **test_metrics,
    }
    write_csv(test_path, result)
    # Preserve best.pt and every metric artifact while reclaiming the redundant
    # optimizer/RNG recovery state after the atomic result has been committed.
    last_path.unlink(missing_ok=True)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
