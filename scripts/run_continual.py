#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.optim as optim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.continual.stream import load_fold_records, sha256_file
from src.continual.training import (
    balanced_epoch_size,
    estimate_diagonal_fisher,
    ewc_penalty,
    evaluate,
    move_consolidations,
    rows_for_cases,
    select_replay_case_ids,
    train_epoch,
)
from src.data.directed_anchors import build_directed_anchor_index
from src.data.original_adapter import OriginalDatasetAdapter
from src.data.token_statistics import (
    informative_token_mask,
    load_token_statistics,
)
from src.losses.original_loss import FocalLoss
from src.models.cglsmn_baseline import CGLSMNBaseline
from src.models.contrastive_head import PairProjectionHead
from src.utils.checkpoint import atomic_torch_save
from src.utils.seed import capture_rng_state, restore_rng_state, seed_everything


TRAIN_FIELDS = [
    "run_name", "method", "task", "fold", "seed", "train_stage", "epoch",
    "stage_epochs", "input_samples", "current_samples", "replay_samples",
    "replay_cases", "balanced_samples", "train_samples", "not_found",
    "train_loss", "regularization_loss", "logit_distillation_loss",
    "embedding_distillation_loss", "attention_distillation_loss",
    "effective_distillation_loss", "distilled_samples",
    "directed_evidence_loss", "directed_mutual_loss",
    "directed_negative_loss", "directed_effective_aux_loss",
    "directed_aux_scale", "directed_ramp",
    "directed_contributing_samples", "directed_evidence_pairs",
    "directed_mutual_pairs", "directed_infonce_negatives",
    "directed_negative_pair_matches",
    "supervised_contrastive_loss",
    "supervised_contrastive_effective_loss",
    "supervised_contrastive_positive_pairs",
    "supervised_contrastive_negative_pairs",
    "supervised_contrastive_valid_anchors",
    "train_accuracy", "train_seconds",
]
STAGE_FIELDS = [
    "run_name", "method", "task", "fold", "seed", "train_stage",
    "eval_split", "eval_increment", "precision", "recall", "f1", "accuracy",
    "roc_auc", "roc_auc_legacy", "pr_auc", "mcc", "tp", "fp", "tn", "fn",
    "samples", "not_found", "inference_seconds",
]
REPLAY_METHODS = {
    "replay",
    "replay_kd",
    "replay_relation",
    "replay_relation_ewc",
}
DISTILLATION_METHODS = {
    "replay_kd",
    "replay_relation",
    "replay_relation_ewc",
}
EWC_METHODS = {"ewc", "replay_relation_ewc"}


def uses_replay(method: str) -> bool:
    return method in REPLAY_METHODS


def uses_distillation(method: str) -> bool:
    return method in DISTILLATION_METHODS


def uses_ewc(method: str) -> bool:
    return method in EWC_METHODS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "joint",
            "naive",
            "replay",
            "ewc",
            "replay_kd",
            "replay_relation",
            "replay_relation_ewc",
        ),
        required=True,
    )
    parser.add_argument("--task", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--base-objective",
        choices=("focal", "supcon", "cl6"),
        default="focal",
        help="Detection objective used inside every continual stage.",
    )
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backward-chunk", type=int, default=64)
    parser.add_argument("--buffer-ratio", type=float, default=0.10)
    parser.add_argument("--ewc-lambda", type=float, default=10.0)
    parser.add_argument("--logit-distillation", type=float, default=0.50)
    parser.add_argument("--embedding-distillation", type=float, default=0.10)
    parser.add_argument("--attention-distillation", type=float, default=0.10)
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
    parser.add_argument("--local-max-document-ratio", type=float, default=0.05)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--supcon-lambda", type=float, default=0.20)
    parser.add_argument("--fisher-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("CGLSMN_DATA_ROOT", PROJECT_ROOT / "data")),
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=PROJECT_ROOT / "artifacts" / "metadata" / "all_samples.csv",
    )
    parser.add_argument("--stream", type=Path)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _upsert_csv(path: Path, row: dict, fields: list[str], keys: list[str]):
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    key = tuple(str(row[field]) for field in keys)
    rows = [
        existing for existing in rows
        if tuple(str(existing[field]) for field in keys) != key
    ]
    rows.append({field: row[field] for field in fields})
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _row_statistics(rows: list[dict]) -> dict:
    labels = Counter(row["label"] for row in rows)
    states = Counter(row["state"] for row in rows)
    return {
        "samples": len(rows),
        "cases": len({row["case_id"] for row in rows}),
        "projects": len({row["split_project"] for row in rows}),
        "labels": dict(sorted(labels.items())),
        "states": dict(sorted(states.items())),
    }


def load_stream_rows(args):
    stream_path = args.stream or (
        PROJECT_ROOT / "artifacts" / "continual"
        / f"task{args.task}_fold{args.fold}_seed{args.seed}.json"
    )
    artifact = json.loads(stream_path.read_text(encoding="utf-8"))
    if (artifact["task"], artifact["fold"], artifact["seed"]) != (
        args.task, args.fold, args.seed
    ):
        raise ValueError("stream task/fold/seed does not match run")
    if artifact["audit"]["status"] != "passed":
        raise ValueError("continual stream did not pass its audit")
    if sha256_file(args.metadata) != artifact["source"]["metadata_sha256"]:
        raise ValueError("canonical metadata hash differs from the audited stream source")
    train_rows, test_rows = load_fold_records(args.metadata, args.task, args.fold)
    lookup = {row["sample_id"]: row for row in train_rows + test_rows}
    if len(lookup) != len(train_rows) + len(test_rows):
        raise ValueError("sample IDs are not unique in selected fold")

    increments = []
    for item in artifact["increments"]:
        increments.append({
            "id": item["id"],
            "train": [lookup[sample_id] for sample_id in item["train_sample_ids"]],
            "probe": [lookup[sample_id] for sample_id in item["probe_sample_ids"]],
        })
    fixed_test = [
        lookup[sample_id] for sample_id in artifact["fixed_outer_test"]["sample_ids"]
    ]
    return stream_path, artifact, increments, fixed_test


def build_plan(args, increments: list[dict]) -> list[dict]:
    if args.stage_epochs <= 0:
        raise ValueError("stage epochs must be positive")
    if args.method == "joint":
        current = [row for increment in increments for row in increment["train"]]
        return [{
            "train_stage": 3,
            "name": "joint",
            "current": current,
            "replay": [],
            "evaluate_through": len(increments),
            "replay_case_ids": [],
        }]

    plan = []
    prior: list[dict] = []
    for index, increment in enumerate(increments, 1):
        replay_case_ids: list[str] = []
        replay_rows: list[dict] = []
        if uses_replay(args.method) and prior:
            replay_case_ids = select_replay_case_ids(
                prior,
                args.buffer_ratio,
                args.seed + args.task * 10000 + args.fold * 100 + index,
            )
            replay_rows = rows_for_cases(prior, replay_case_ids)
        plan.append({
            "train_stage": index,
            "name": increment["id"],
            "current": increment["train"],
            "replay": replay_rows,
            "evaluate_through": index,
            "replay_case_ids": replay_case_ids,
        })
        prior.extend(increment["train"])
    return plan


def plan_summary(args, plan: list[dict]) -> dict:
    stages = []
    total_balanced = 0
    for stage in plan:
        inputs = stage["current"] + stage["replay"]
        balanced = balanced_epoch_size(
            [row["label_line"] for row in inputs], args.batch_size
        )
        total_balanced += balanced * args.stage_epochs
        stages.append({
            "train_stage": stage["train_stage"],
            "name": stage["name"],
            "current": _row_statistics(stage["current"]),
            "replay": _row_statistics(stage["replay"]),
            "input": _row_statistics(inputs),
            "balanced_samples_per_epoch": balanced,
            "epochs": args.stage_epochs,
            "balanced_samples_total": balanced * args.stage_epochs,
            "evaluate_through": stage["evaluate_through"],
            "replay_case_ids": stage["replay_case_ids"],
        })
    return {
        "method": args.method,
        "stage_epochs": args.stage_epochs,
        "batch_size": args.batch_size,
        "total_balanced_training_samples": total_balanced,
        "stages": stages,
    }


def run_name(args) -> str:
    if args.base_objective == "cl6":
        objective = "_cl6"
    elif args.base_objective == "supcon":
        objective = (
            f"_supcon_t{args.supcon_temperature:g}"
            f"_l{args.supcon_lambda:g}"
        )
    else:
        objective = ""
    suffix = (
        f"_r{int(round(args.buffer_ratio * 100)):02d}"
        if uses_replay(args.method)
        else ""
    )
    if args.method == "ewc":
        suffix = f"_l{args.ewc_lambda:g}"
    elif args.method == "replay_kd":
        suffix += f"_kd{args.logit_distillation:g}"
    elif args.method == "replay_relation":
        suffix += (
            f"_kd{args.logit_distillation:g}"
            f"_z{args.embedding_distillation:g}"
            f"_a{args.attention_distillation:g}"
        )
    elif args.method == "replay_relation_ewc":
        suffix += (
            f"_kd{args.logit_distillation:g}"
            f"_z{args.embedding_distillation:g}"
            f"_a{args.attention_distillation:g}"
            f"_l{args.ewc_lambda:g}"
        )
    return (
        f"task{args.task}_fold{args.fold}_seed{args.seed}_"
        f"{args.method}{objective}{suffix}_e{args.stage_epochs}"
    )


def main():
    args = parse_args()
    if args.batch_size % args.backward_chunk != 0:
        raise ValueError("--backward-chunk must divide --batch-size")
    for name, value in (
        ("logit distillation", args.logit_distillation),
        ("embedding distillation", args.embedding_distillation),
        ("attention distillation", args.attention_distillation),
    ):
        if value < 0.0:
            raise ValueError(f"{name} coefficient cannot be negative")
    if args.base_objective == "cl6":
        if args.task != 2:
            raise ValueError("CL6 continual objective supports Task 2 only")
        if args.directed_warmup_epochs < 0:
            raise ValueError("directed warmup epochs cannot be negative")
        if args.directed_ramp_epochs < 1:
            raise ValueError("directed ramp epochs must be positive")
        if not 0.0 < args.directed_max_aux_ratio <= 1.0:
            raise ValueError("directed max auxiliary ratio must be in (0, 1]")
        for value in (
            args.directed_lambda_positive,
            args.directed_lambda_mutual,
            args.directed_lambda_negative,
        ):
            if value < 0.0:
                raise ValueError("directed coefficients cannot be negative")
    if args.base_objective == "supcon":
        if args.supcon_temperature <= 0.0:
            raise ValueError("SupCon temperature must be positive")
        if args.supcon_lambda < 0.0:
            raise ValueError("SupCon coefficient cannot be negative")
    stream_path, artifact, increments, fixed_test = load_stream_rows(args)
    plan = build_plan(args, increments)
    summary = plan_summary(args, plan)
    if args.plan_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    name = run_name(args)
    result_dir = PROJECT_ROOT / "results" / "continual" / "runs" / name
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "continual" / name
    complete_path = result_dir / "complete.json"
    train_path = result_dir / "train_metrics.csv"
    stage_path = result_dir / "stage_metrics.csv"
    last_path = checkpoint_dir / "last.pt"
    if complete_path.exists():
        print(f"{name} already complete; skipping", flush=True)
        return
    if args.fresh and (result_dir.exists() or checkpoint_dir.exists()):
        raise RuntimeError("--fresh refuses to overwrite an existing run")
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": name,
        "method": args.method,
        "base_objective": args.base_objective,
        "task": args.task,
        "fold": args.fold,
        "seed": args.seed,
        "stage_epochs": args.stage_epochs,
        "batch_size": args.batch_size,
        "backward_chunk": args.backward_chunk,
        "buffer_ratio": args.buffer_ratio if uses_replay(args.method) else None,
        "ewc_lambda": args.ewc_lambda if uses_ewc(args.method) else None,
        "fisher_samples": args.fisher_samples if uses_ewc(args.method) else None,
        "logit_distillation": (
            args.logit_distillation if uses_distillation(args.method) else 0.0
        ),
        "embedding_distillation": (
            args.embedding_distillation
            if args.method in {"replay_relation", "replay_relation_ewc"}
            else 0.0
        ),
        "attention_distillation": (
            args.attention_distillation
            if args.method in {"replay_relation", "replay_relation_ewc"}
            else 0.0
        ),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "supcon_temperature": (
            args.supcon_temperature if args.base_objective == "supcon" else None
        ),
        "supcon_lambda": (
            args.supcon_lambda if args.base_objective == "supcon" else None
        ),
        "model": "original CG-LSMN classifier",
        "loss": (
            (
                "Focal + CL6 directed method-to-class"
                if args.base_objective == "cl6"
                else (
                    "Focal + supervised contrastive pair objective"
                    if args.base_objective == "supcon"
                    else "Focal"
                )
            )
            + "(new+replay) + replay logit/pair/attention distillation"
            + (" + EWC" if uses_ewc(args.method) else "")
            if uses_distillation(args.method)
            else (
                "Focal + CL6 directed method-to-class"
                if args.base_objective == "cl6"
                else (
                    "Focal + supervised contrastive pair objective"
                    if args.base_objective == "supcon"
                    else "original FocalLoss"
                )
            )
            + (" + EWC" if uses_ewc(args.method) else "")
        ),
        "directed": (
            {
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
                "local_max_document_ratio": (
                    args.local_max_document_ratio
                ),
            }
            if args.base_objective == "cl6"
            else None
        ),
        "model_selection": "none; fixed budget, probes are evaluation-only",
        "optimizer_state_across_stages": "retained",
        "stream_path": str(stream_path.resolve()),
        "stream_sha256": sha256_file(stream_path),
        "metadata_sha256": artifact["source"]["metadata_sha256"],
        "plan": summary,
    }
    _atomic_json(result_dir / "run_config.json", config)
    _atomic_json(
        result_dir / "replay_buffers.json",
        {
            f"stage_{stage['train_stage']}": stage["replay_case_ids"]
            for stage in plan
        },
    )

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"loading immutable task {args.task} graph cache", flush=True)
    adapter = OriginalDatasetAdapter(args.data_root, args.task, split_seed=100)
    directed_options = None
    if args.base_objective == "cl6":
        token_statistics = load_token_statistics(
            PROJECT_ROOT / "artifacts" / "token_statistics"
            / f"task{args.task}.json"
        )
        informative_tokens = informative_token_mask(
            token_statistics,
            args.local_max_document_ratio,
            device,
        )
        outer_train_rows = [
            row
            for increment in increments
            for row in increment["train"] + increment["probe"]
        ]
        directed_anchors, directed_audit = build_directed_anchor_index(
            {row["label_line"]: row for row in outer_train_rows},
            adapter.vocab,
        )
        directed_options = {
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
        config["directed"]["anchor_audit"] = directed_audit
        config["directed"]["informative_tokens"] = int(
            informative_tokens.sum()
        )
        _atomic_json(result_dir / "run_config.json", config)
        print(
            "CL6 continual anchor audit="
            f"{json.dumps(directed_audit, sort_keys=True)}",
            flush=True,
        )
    model = CGLSMNBaseline(adapter.vocab_size).to(device)
    projection = (
        PairProjectionHead().to(device)
        if args.base_objective == "supcon"
        else None
    )
    criterion = FocalLoss().to(device)
    parameters = list(model.parameters())
    if projection is not None:
        parameters.extend(projection.parameters())
    optimizer = optim.Adam(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    sampling_rng = random.Random(args.seed + args.task * 100 + args.fold)
    consolidations: list[dict] = []
    teacher_state = None
    stage_cursor = 0
    epoch_cursor = 0
    total_started = time.perf_counter()
    prior_elapsed = 0.0
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["run_name"] != name:
            raise ValueError("checkpoint belongs to a different run")
        model.load_state_dict(state["model"])
        if projection is not None:
            projection.load_state_dict(state["projection"])
        optimizer.load_state_dict(state["optimizer"])
        stage_cursor = int(state["stage_cursor"])
        epoch_cursor = int(state["epoch_cursor"])
        prior_elapsed = float(state.get("elapsed_seconds", 0.0))
        consolidations = move_consolidations(
            state.get("ewc_consolidations", []), device
        )
        teacher_state = state.get("teacher_model")
        restore_rng_state(state["rng"], sampling_rng)
        print(
            f"resuming {name} at plan stage {stage_cursor}, epoch {epoch_cursor}",
            flush=True,
        )

    for plan_index in range(stage_cursor, len(plan)):
        stage = plan[plan_index]
        start_epoch = epoch_cursor if plan_index == stage_cursor else 0
        input_rows = stage["current"] + stage["replay"]
        input_lines = [row["label_line"] for row in input_rows]
        print(
            f"{name} stage={stage['name']} current={len(stage['current'])} "
            f"replay={len(stage['replay'])} cases={len(stage['replay_case_ids'])}",
            flush=True,
        )
        teacher_model = None
        if uses_distillation(args.method) and stage["replay"]:
            teacher_model = CGLSMNBaseline(adapter.vocab_size).to(device)
            source = teacher_state or {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            teacher_model.load_state_dict(source, strict=True)
            teacher_model.eval()
            for parameter in teacher_model.parameters():
                parameter.requires_grad_(False)
        replay_lines = {
            row["label_line"] for row in stage["replay"]
        }
        relation_distillation = args.method in {
            "replay_relation",
            "replay_relation_ewc",
        }
        for epoch in range(start_epoch, args.stage_epochs):
            print(
                f"{name} stage={stage['name']} epoch={epoch}/{args.stage_epochs - 1}",
                flush=True,
            )
            metrics = train_epoch(
                model, optimizer, criterion, adapter, input_lines,
                args.batch_size, args.backward_chunk, device, sampling_rng,
                regularizer=(
                    (lambda current_model: ewc_penalty(
                        current_model, consolidations, args.ewc_lambda
                    ))
                    if uses_ewc(args.method) and consolidations else None
                ),
                replay_lines=replay_lines,
                teacher_model=teacher_model,
                logit_distillation=(
                    args.logit_distillation
                    if uses_distillation(args.method)
                    else 0.0
                ),
                embedding_distillation=(
                    args.embedding_distillation
                    if relation_distillation
                    else 0.0
                ),
                attention_distillation=(
                    args.attention_distillation
                    if relation_distillation
                    else 0.0
                ),
                epoch=plan_index * args.stage_epochs + epoch,
                directed_options=directed_options,
                projection=projection,
                supcon_options=(
                    {
                        "temperature": args.supcon_temperature,
                        "coefficient": args.supcon_lambda,
                    }
                    if projection is not None
                    else None
                ),
            )
            row = {
                "run_name": name,
                "method": args.method,
                "task": args.task,
                "fold": args.fold,
                "seed": args.seed,
                "train_stage": stage["train_stage"],
                "epoch": epoch,
                "stage_epochs": args.stage_epochs,
                "input_samples": len(input_rows),
                "current_samples": len(stage["current"]),
                "replay_samples": len(stage["replay"]),
                "replay_cases": len(stage["replay_case_ids"]),
                **metrics,
            }
            _upsert_csv(
                train_path, row, TRAIN_FIELDS,
                ["run_name", "train_stage", "epoch"],
            )
            elapsed = prior_elapsed + time.perf_counter() - total_started
            atomic_torch_save({
                "run_name": name,
                "model": model.state_dict(),
                "projection": (
                    projection.state_dict() if projection is not None else None
                ),
                "optimizer": optimizer.state_dict(),
                "stage_cursor": plan_index,
                "epoch_cursor": epoch + 1,
                "elapsed_seconds": elapsed,
                "rng": capture_rng_state(sampling_rng),
                "ewc_consolidations": consolidations,
                "teacher_model": (
                    teacher_model.state_dict()
                    if teacher_model is not None
                    else None
                ),
            }, last_path)
            print(json.dumps(row, sort_keys=True), flush=True)

        evaluation_sets = [
            (
                "probe",
                index,
                [row["label_line"] for row in increments[index - 1]["probe"]],
            )
            for index in range(1, stage["evaluate_through"] + 1)
        ]
        evaluation_sets.append(
            ("outer_test", 0, [row["label_line"] for row in fixed_test])
        )
        stage_metrics = []
        for eval_split, eval_increment, lines in evaluation_sets:
            eval_name = f"{eval_split}_I{eval_increment}" if eval_increment else eval_split
            metrics = evaluate(model, adapter, lines, device, eval_name)
            row = {
                "run_name": name,
                "method": args.method,
                "task": args.task,
                "fold": args.fold,
                "seed": args.seed,
                "train_stage": stage["train_stage"],
                "eval_split": eval_split,
                "eval_increment": eval_increment,
                **metrics,
            }
            _upsert_csv(
                stage_path, row, STAGE_FIELDS,
                ["run_name", "train_stage", "eval_split", "eval_increment"],
            )
            stage_metrics.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

        if uses_ewc(args.method):
            fisher, fisher_count, fisher_seconds = estimate_diagonal_fisher(
                model,
                adapter,
                [row["label_line"] for row in stage["current"]],
                device,
                args.fisher_samples,
                args.seed + args.task * 10000 + args.fold * 100 + stage["train_stage"],
            )
            consolidations.append({
                "stage": stage["train_stage"],
                "samples": fisher_count,
                "anchor": {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                },
                "fisher": fisher,
            })
            fisher_stats_path = result_dir / "fisher_statistics.json"
            if fisher_stats_path.exists():
                fisher_stats = json.loads(fisher_stats_path.read_text(encoding="utf-8"))
            else:
                fisher_stats = {}
            fisher_stats[f"stage_{stage['train_stage']}"] = {
                "samples": fisher_count,
                "seconds": fisher_seconds,
                "parameter_tensors": len(fisher),
                "fisher_sum": sum(float(value.sum()) for value in fisher.values()),
                "fisher_max": max(float(value.max()) for value in fisher.values()),
            }
            _atomic_json(fisher_stats_path, fisher_stats)

        atomic_torch_save({
            "run_name": name,
            "model": model.state_dict(),
            "projection": (
                projection.state_dict() if projection is not None else None
            ),
            "train_stage": stage["train_stage"],
            "stage_name": stage["name"],
            "metrics": stage_metrics,
        }, checkpoint_dir / f"stage_{stage['train_stage']}.pt")
        elapsed = prior_elapsed + time.perf_counter() - total_started
        epoch_cursor = 0
        teacher_state = None
        atomic_torch_save({
            "run_name": name,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stage_cursor": plan_index + 1,
            "epoch_cursor": 0,
            "elapsed_seconds": elapsed,
            "rng": capture_rng_state(sampling_rng),
            "ewc_consolidations": consolidations,
            "teacher_model": None,
        }, last_path)

    elapsed = prior_elapsed + time.perf_counter() - total_started
    atomic_torch_save({
        "run_name": name,
        "model": model.state_dict(),
        "projection": (
            projection.state_dict() if projection is not None else None
        ),
        "config": config,
        "elapsed_seconds": elapsed,
        # Fisher/anchor tensors are training-only recovery state.  Their scalar
        # audit is retained in fisher_statistics.json; duplicating all tensors
        # in a completed final model makes EWC checkpoints several times larger.
        "ewc_consolidation_count": len(consolidations),
    }, checkpoint_dir / "final.pt")
    _atomic_json(complete_path, {
        "run_name": name,
        "status": "complete",
        "elapsed_seconds": elapsed,
        "stage_metrics": str(stage_path),
        "train_metrics": str(train_path),
    })
    # Completion is atomic and final.pt plus stage checkpoints remain.  last.pt
    # only duplicates the model, optimizer and recovery tensors after success.
    last_path.unlink(missing_ok=True)
    print(f"completed {name} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
