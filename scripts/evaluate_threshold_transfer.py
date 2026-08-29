#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, matthews_corrcoef

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fixed_splits import inner_project_validation
from src.data.original_adapter import OriginalDatasetAdapter
from src.models.cglsmn_baseline import CGLSMNBaseline
from src.utils.metrics import classification_metrics


@torch.no_grad()
def predict(model, adapter, lines, device):
    model.eval()
    labels, positive_scores, all_scores = [], [], []
    for index, line in enumerate(lines, 1):
        tensors = adapter.graph_tensors(line, device)
        if tensors is None:
            continue
        h1, e1, h2, e2, label = tensors
        probabilities = model(h1, e1, h2, e2)[0]
        scores = probabilities.detach().cpu().tolist()
        labels.append(label)
        positive_scores.append(scores[1])
        all_scores.append(scores)
        if index % 1000 == 0 or index == len(lines):
            print(f"predict {index}/{len(lines)}", flush=True)
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(positive_scores, dtype=np.float64),
        all_scores,
    )


def select_threshold(labels, scores, objective):
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    best = None
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        value = (
            matthews_corrcoef(labels, predictions)
            if objective == "mcc"
            else f1_score(labels, predictions, zero_division=0))
        candidate = (float(value), -abs(float(threshold) - 0.5), float(threshold))
        if best is None or candidate > best:
            best = candidate
    return best[2], best[0]


def metrics_at_threshold(labels, scores, all_scores, threshold):
    predictions = (scores >= threshold).astype(np.int64)
    return classification_metrics(labels, predictions, scores, all_scores)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--tuning-run", required=True)
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--validation-seed", type=int, default=101)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("CGLSMN_DATA_ROOT", PROJECT_ROOT / "data")),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(adapter, run_name, device):
    checkpoint_path = (
        PROJECT_ROOT / "checkpoints" / "contrastive" / "runs"
        / run_name / "best.pt")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CGLSMNBaseline(adapter.vocab_size).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model, checkpoint_path


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"loading immutable task {args.task} graph cache", flush=True)
    adapter = OriginalDatasetAdapter(args.data_root, args.task, split_seed=100)
    outer_train, _outer_test = adapter.fold(args.fold)
    _, validation, validation_projects = inner_project_validation(
        outer_train, args.task, args.fold,
        args.validation_fraction, args.validation_seed)

    tuning_model, tuning_checkpoint = load_model(
        adapter, args.tuning_run, device)
    validation_labels, validation_scores, validation_all_scores = predict(
        tuning_model, adapter, validation, device)
    thresholds = {}
    validation_metrics = {}
    for objective in ("mcc", "f1"):
        threshold, score = select_threshold(
            validation_labels, validation_scores, objective)
        thresholds[objective] = threshold
        validation_metrics[objective] = {
            "selection_score": score,
            **metrics_at_threshold(
                validation_labels, validation_scores,
                validation_all_scores, threshold),
        }
    # The primary runner has already evaluated the fixed 0.5 decision rule on
    # the outer test exactly once.  Replaying the outer test here, even with a
    # validation-only threshold, would violate that frozen one-evaluation
    # policy.  Keep calibration strictly on the inner validation split.
    primary_metrics = (
        PROJECT_ROOT / "results" / "contrastive" / "runs"
        / args.full_run / "test_metrics.csv"
    )
    if not primary_metrics.exists():
        raise FileNotFoundError(primary_metrics)
    with primary_metrics.open(newline="", encoding="utf-8") as handle:
        primary_rows = list(csv.DictReader(handle))
    if len(primary_rows) != 1:
        raise RuntimeError(
            f"expected one locked primary outer-test row, found {len(primary_rows)}"
        )
    full_checkpoint = (
        PROJECT_ROOT / "checkpoints" / "contrastive" / "runs"
        / args.full_run / "best.pt"
    )
    if not full_checkpoint.exists():
        raise FileNotFoundError(full_checkpoint)

    payload = {
        "task": args.task,
        "fold": args.fold,
        "tuning_run": args.tuning_run,
        "full_run": args.full_run,
        "tuning_checkpoint": str(tuning_checkpoint),
        "full_checkpoint": str(full_checkpoint),
        "validation_projects": sorted(validation_projects),
        "thresholds": thresholds,
        "validation_metrics": validation_metrics,
        "outer_test_evaluated": False,
        "locked_primary_outer_test_metrics": str(primary_metrics),
        "policy": (
            "thresholds are validation-only deployment diagnostics; the outer "
            "test is not replayed and the fixed-0.5 primary result is unchanged"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
