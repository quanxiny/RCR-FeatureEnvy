#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "results" / "continual"
RUNS = ROOT / "runs"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    os.replace(temporary, path)


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def run_summary(config: dict, rows: list[dict]) -> tuple[dict, list[dict]]:
    method = config["method"]
    final_stage = max(int(row["train_stage"]) for row in rows)
    probes = [row for row in rows if row["eval_split"] == "probe"]
    matrix = []
    by_cell = {}
    for row in probes:
        train_stage = int(row["train_stage"])
        eval_increment = int(row["eval_increment"])
        by_cell[(train_stage, eval_increment)] = row
        matrix.append({
            "run_name": config["run_name"],
            "method": method,
            "task": config["task"],
            "fold": config["fold"],
            "seed": config["seed"],
            "train_stage": train_stage,
            "eval_increment": eval_increment,
            "accuracy": row["accuracy"],
            "f1": row["f1"],
            "pr_auc": row["pr_auc"],
            "mcc": row["mcc"],
        })

    final_probe_rows = [
        row for row in probes if int(row["train_stage"]) == final_stage
    ]
    final_average_accuracy = mean([as_float(row, "accuracy") for row in final_probe_rows])
    forgetting_values: list[float] = []
    bwt_values: list[float] = []
    per_increment_forgetting = {}
    if method != "joint":
        for increment in range(1, final_stage):
            prior = [
                as_float(row, "accuracy") for row in probes
                if int(row["eval_increment"]) == increment
                and int(row["train_stage"]) < final_stage
            ]
            final = by_cell[(final_stage, increment)]
            diagonal = by_cell[(increment, increment)]
            forgetting = max(prior) - as_float(final, "accuracy")
            bwt = as_float(final, "accuracy") - as_float(diagonal, "accuracy")
            forgetting_values.append(forgetting)
            bwt_values.append(bwt)
            per_increment_forgetting[f"I{increment}"] = forgetting

    outer_candidates = [
        row for row in rows
        if row["eval_split"] == "outer_test"
        and int(row["train_stage"]) == final_stage
    ]
    if len(outer_candidates) != 1:
        raise ValueError(f"expected one final outer test row for {config['run_name']}")
    outer = outer_candidates[0]
    average_forgetting = mean(forgetting_values)
    maximum_forgetting = max(forgetting_values) if forgetting_values else math.nan
    material = bool(forgetting_values) and (
        average_forgetting >= 0.02 or maximum_forgetting >= 0.05
    )
    summary = {
        "run_name": config["run_name"],
        "method": method,
        "task": config["task"],
        "fold": config["fold"],
        "seed": config["seed"],
        "stage_epochs": config["stage_epochs"],
        "buffer_ratio": config["buffer_ratio"] if config["buffer_ratio"] is not None else "",
        "final_average_probe_accuracy": final_average_accuracy,
        "average_forgetting": average_forgetting if forgetting_values else "",
        "maximum_forgetting": maximum_forgetting if forgetting_values else "",
        "backward_transfer": mean(bwt_values) if bwt_values else "",
        "material_forgetting": int(material),
        "per_increment_forgetting": json.dumps(per_increment_forgetting, sort_keys=True),
        "outer_precision": outer["precision"],
        "outer_recall": outer["recall"],
        "outer_f1": outer["f1"],
        "outer_accuracy": outer["accuracy"],
        "outer_roc_auc": outer["roc_auc"],
        "outer_pr_auc": outer["pr_auc"],
        "outer_mcc": outer["mcc"],
        "outer_roc_auc_legacy": outer["roc_auc_legacy"],
        "outer_tp": outer["tp"],
        "outer_fp": outer["fp"],
        "outer_tn": outer["tn"],
        "outer_fn": outer["fn"],
        "outer_inference_seconds": outer["inference_seconds"],
        "outer_samples": outer["samples"],
    }
    return summary, matrix


def main():
    stage_rows = []
    summaries = []
    matrix_rows = []
    if RUNS.exists():
        for run_dir in sorted(path for path in RUNS.iterdir() if path.is_dir()):
            if not (run_dir / "complete.json").exists():
                continue
            config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            rows = read_csv(run_dir / "stage_metrics.csv")
            summary, matrix = run_summary(config, rows)
            train_rows = read_csv(run_dir / "train_metrics.csv")
            complete = json.loads(
                (run_dir / "complete.json").read_text(encoding="utf-8")
            )
            summary["training_seconds_total"] = sum(
                as_float(row, "train_seconds") for row in train_rows
            )
            summary["wall_seconds_total"] = float(complete["elapsed_seconds"])
            stage_rows.extend(rows)
            summaries.append(summary)
            matrix_rows.extend(matrix)

    stage_fields = [
        "run_name", "method", "task", "fold", "seed", "train_stage",
        "eval_split", "eval_increment", "precision", "recall", "f1", "accuracy",
        "roc_auc", "roc_auc_legacy", "pr_auc", "mcc", "tp", "fp", "tn", "fn",
        "samples", "not_found", "inference_seconds",
    ]
    matrix_fields = [
        "run_name", "method", "task", "fold", "seed", "train_stage",
        "eval_increment", "accuracy", "f1", "pr_auc", "mcc",
    ]
    summary_fields = [
        "run_name", "method", "task", "fold", "seed", "stage_epochs",
        "buffer_ratio", "final_average_probe_accuracy", "average_forgetting",
        "maximum_forgetting", "backward_transfer", "material_forgetting",
        "per_increment_forgetting", "outer_precision", "outer_recall", "outer_f1",
        "outer_accuracy", "outer_roc_auc", "outer_pr_auc", "outer_mcc",
        "outer_roc_auc_legacy", "outer_tp", "outer_fp", "outer_tn", "outer_fn",
        "outer_inference_seconds", "outer_samples", "training_seconds_total",
        "wall_seconds_total",
    ]
    write_csv(ROOT / "stage_metrics.csv", stage_rows, stage_fields)
    write_csv(ROOT / "forgetting_matrix.csv", matrix_rows, matrix_fields)
    write_csv(ROOT / "summary.csv", summaries, summary_fields)
    print(json.dumps({
        "complete_runs": len(summaries),
        "stage_metric_rows": len(stage_rows),
        "forgetting_matrix_rows": len(matrix_rows),
        "outputs": [
            str(ROOT / "stage_metrics.csv"),
            str(ROOT / "forgetting_matrix.csv"),
            str(ROOT / "summary.csv"),
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
