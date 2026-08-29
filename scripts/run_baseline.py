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
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.original_adapter import OriginalDatasetAdapter
from src.losses.original_loss import FocalLoss
from src.models.cglsmn_baseline import CGLSMNBaseline
from src.utils.checkpoint import atomic_torch_save
from src.utils.metrics import classification_metrics
from src.utils.seed import capture_rng_state, restore_rng_state, seed_everything


METRIC_FIELDS = [
    "task", "fold", "seed", "epoch", "precision", "recall", "f1", "accuracy",
    "roc_auc", "roc_auc_legacy", "pr_auc", "mcc", "tp", "fp", "tn", "fn",
    "train_seconds", "inference_seconds", "train_loss", "train_accuracy",
    "train_samples", "test_samples", "not_found", "is_best",
]


def balanced_epoch(lines, rng: random.Random):
    shuffled = list(lines)
    rng.shuffle(shuffled)
    positives = [line for line in shuffled if int(line.split()[3]) == 1]
    negatives = [line for line in shuffled if int(line.split()[3]) == 0]
    count = min(len(positives), len(negatives))
    selected = positives[:count] + negatives[:count]
    rng.shuffle(selected)
    return selected


def one_hot_target(label: int, device):
    return torch.tensor([[0.0, 1.0] if label else [1.0, 0.0]], device=device)


def train_epoch(model, optimizer, criterion, adapter, train_lines, batch_size,
                backward_chunk, device, rng):
    model.train()
    selected = balanced_epoch(train_lines, rng)
    usable = (len(selected) // batch_size) * batch_size
    selected = selected[:usable]
    started = time.perf_counter()
    total_loss = 0.0
    correct = 0
    found = 0
    for batch_start in range(0, usable, batch_size):
        optimizer.zero_grad(set_to_none=True)
        batch = selected[batch_start:batch_start + batch_size]
        pending_loss = None
        pending_count = 0
        for line in batch:
            tensors = adapter.graph_tensors(line, device)
            if tensors is None or tensors[1].numel() == 0 or tensors[3].numel() == 0:
                continue
            h1, e1, h2, e2, label = tensors
            output = model(h1, e1, h2, e2)
            loss = criterion(output, one_hot_target(label, device))
            pending_loss = loss if pending_loss is None else pending_loss + loss
            pending_count += 1
            total_loss += float(loss.detach())
            correct += int(output.argmax(dim=1).item() == label)
            found += 1
            if pending_count == backward_chunk:
                pending_loss.backward()
                pending_loss = None
                pending_count = 0
        if pending_loss is not None:
            pending_loss.backward()
        optimizer.step()
        completed = batch_start + batch_size
        if completed % (batch_size * 25) == 0 or completed == usable:
            print(
                f"train {completed}/{usable} loss={total_loss/max(found,1):.6f} "
                f"acc={correct/max(found,1):.4f}", flush=True)
    return {
        "train_seconds": time.perf_counter() - started,
        "train_loss": total_loss / max(found, 1),
        "train_accuracy": correct / max(found, 1),
        "train_samples": found,
    }


@torch.no_grad()
def evaluate(model, adapter, test_lines, device):
    model.eval()
    started = time.perf_counter()
    y_true, y_pred, positive_scores, all_scores = [], [], [], []
    not_found = 0
    for index, line in enumerate(test_lines, 1):
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
        if index % 1000 == 0 or index == len(test_lines):
            print(f"test {index}/{len(test_lines)}", flush=True)
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(y_true, y_pred, positive_scores, all_scores)
    metrics.update({
        "inference_seconds": elapsed,
        "test_samples": len(y_true),
        "not_found": not_found,
    })
    return metrics


def read_existing_epochs(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["epoch"]) for row in csv.DictReader(handle)}


def append_metric(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row[field] for field in METRIC_FIELDS})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backward-chunk", type=int, default=64,
                        help="Graph-pair losses combined per backward call; optimizer step remains per batch.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("CGLSMN_DATA_ROOT", PROJECT_ROOT / "data")),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size % args.backward_chunk != 0:
        raise ValueError("--backward-chunk must divide --batch-size")
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_name = f"task{args.task}_fold{args.fold}_seed{args.seed}"
    result_dir = PROJECT_ROOT / "results" / "baseline" / run_name
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "baseline" / run_name
    metric_path = result_dir / "metrics.csv"
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    if args.fresh and (metric_path.exists() or last_path.exists() or best_path.exists()):
        raise RuntimeError("--fresh refuses to overwrite an existing run")
    completed_epochs = read_existing_epochs(metric_path)
    if len(completed_epochs) >= args.epochs:
        print(f"{run_name} already complete; skipping")
        return

    print(f"loading immutable task {args.task} graph cache", flush=True)
    adapter = OriginalDatasetAdapter(args.data_root, args.task, split_seed=100)
    train_lines, test_lines = adapter.fold(args.fold)
    print(
        f"{run_name}: vocab={adapter.vocab_size} train={len(train_lines)} "
        f"test={len(test_lines)} device={device}", flush=True)
    model = CGLSMNBaseline(adapter.vocab_size).to(device)
    criterion = FocalLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    sampling_rng = random.Random(args.seed + args.task * 100 + args.fold)
    start_epoch = 0
    best_f1 = -math.inf
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["next_epoch"])
        best_f1 = float(state["best_f1"])
        restore_rng_state(state["rng"], sampling_rng)
        print(f"resuming at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"epoch {epoch}/{args.epochs - 1}", flush=True)
        train_metrics = train_epoch(
            model, optimizer, criterion, adapter, train_lines, args.batch_size,
            args.backward_chunk, device, sampling_rng)
        test_metrics = evaluate(model, adapter, test_lines, device)
        is_best = test_metrics["f1"] > best_f1
        if is_best:
            best_f1 = test_metrics["f1"]
            atomic_torch_save({
                "model": model.state_dict(),
                "task": args.task,
                "fold": args.fold,
                "seed": args.seed,
                "epoch": epoch,
                "metrics": test_metrics,
            }, best_path)
        row = {
            "task": args.task,
            "fold": args.fold,
            "seed": args.seed,
            "epoch": epoch,
            **test_metrics,
            **train_metrics,
            "is_best": int(is_best),
        }
        if epoch not in completed_epochs:
            append_metric(metric_path, row)
        atomic_torch_save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "next_epoch": epoch + 1,
            "best_f1": best_f1,
            "rng": capture_rng_state(sampling_rng),
        }, last_path)
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
