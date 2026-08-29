#!/usr/bin/env python3
"""Summarize the EI-paper contrastive and continual-learning extension."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_task2_continual_sci import (  # noqa: E402
    METRICS,
    REPORTING_METRICS,
    method_aggregate,
    paired_comparison,
)


WORKFLOW_ROOT = PROJECT_ROOT / "results" / "workflow"
AGGREGATE = PROJECT_ROOT / "results" / "continual" / "summary.csv"
DETECTION = WORKFLOW_ROOT / "task2_comparison_extension_summary.json"
SUMMARY = WORKFLOW_ROOT / "task2_ei_extension_summary.json"
RAW = WORKFLOW_ROOT / "task2_ei_extension_raw.csv"
REPORT = PROJECT_ROOT / "reports" / "26_task2_ei_extension.md"
SEEDS = (42, 123, 2024)
FOLDS = (1, 2, 3, 4, 5)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def run_name(label: str, seed: int, fold: int, ratio: int = 10) -> str:
    names = {
        "finetune_cl6": f"task2_fold{fold}_seed{seed}_naive_cl6_e8",
        "ewc_cl6": f"task2_fold{fold}_seed{seed}_ewc_cl6_l10_e8",
        "focal_replay": f"task2_fold{fold}_seed{seed}_replay_r{ratio:02d}_e8",
        "supcon_replay": (
            f"task2_fold{fold}_seed{seed}_"
            f"replay_supcon_t0.07_l0.2_r{ratio:02d}_e8"
        ),
        "cl6_replay": f"task2_fold{fold}_seed{seed}_replay_cl6_r{ratio:02d}_e8",
        "relation_replay": (
            f"task2_fold{fold}_seed{seed}_replay_relation_cl6_"
            f"r{ratio:02d}_kd0.5_z0.1_a0.1_e8"
        ),
    }
    return names[label]


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def read_rows() -> dict[str, dict]:
    with AGGREGATE.open(newline="", encoding="utf-8") as handle:
        return {row["run_name"]: row for row in csv.DictReader(handle)}


def selected_row(source: dict, label: str) -> dict:
    return {
        "label": label,
        "run_name": source["run_name"],
        "method": source["method"],
        "seed": int(source["seed"]),
        "fold": int(source["fold"]),
        "buffer_ratio": (
            float(source["buffer_ratio"])
            if source.get("buffer_ratio", "") != ""
            else None
        ),
        **{
            metric: (
                float(source[metric]) if source.get(metric, "") != "" else None
            )
            for metric in REPORTING_METRICS
        },
    }


def rows_for(
    aggregate: dict[str, dict], label: str, seeds=SEEDS, ratio: int = 10
) -> list[dict]:
    rows = []
    for seed in seeds:
        for fold in FOLDS:
            name = run_name(label, seed, fold, ratio)
            if name not in aggregate:
                raise RuntimeError(f"missing completed aggregate row: {name}")
            rows.append(selected_row(aggregate[name], label))
    return rows


def comparison(candidate: list[dict], reference: list[dict]) -> dict:
    candidate_confirmatory = [
        row for row in candidate
        if (int(row["seed"]), int(row["fold"])) != (42, 1)
    ]
    reference_confirmatory = [
        row for row in reference
        if (int(row["seed"]), int(row["fold"])) != (42, 1)
    ]
    return paired_comparison(candidate_confirmatory, reference_confirmatory)


def buffer_summary(rows: list[dict]) -> dict:
    return {
        "units": len(rows),
        **{
            metric: {
                "mean": mean([float(row[metric]) for row in rows]),
                "std": statistics.stdev(
                    [float(row[metric]) for row in rows]
                ) if len(rows) > 1 else 0.0,
            }
            for metric in (
                "final_average_probe_accuracy",
                "average_forgetting",
                "backward_transfer",
                "outer_f1",
                "outer_mcc",
                "outer_pr_auc",
                "training_seconds_total",
                "wall_seconds_total",
            )
        },
    }


def main() -> None:
    aggregate = read_rows()
    groups = {
        label: rows_for(aggregate, label)
        for label in (
            "finetune_cl6",
            "ewc_cl6",
            "focal_replay",
            "supcon_replay",
            "cl6_replay",
            "relation_replay",
        )
    }
    budget = {
        "5": rows_for(aggregate, "relation_replay", seeds=(42,), ratio=5),
        "10": [
            row for row in groups["relation_replay"] if int(row["seed"]) == 42
        ],
        "20": rows_for(aggregate, "relation_replay", seeds=(42,), ratio=20),
    }
    raw_rows = [row for rows in groups.values() for row in rows]
    raw_rows.extend(
        {**row, "label": f"relation_replay_r{ratio}"}
        for ratio, rows in budget.items() if ratio != "10"
        for row in rows
    )
    fields = list(raw_rows[0])
    temporary = RAW.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)
    os.replace(temporary, RAW)

    static = json.loads(DETECTION.read_text(encoding="utf-8"))
    payload = {
        "protocol": {
            "task": "Method-Class feature-envy detection",
            "outer_split": "five project-disjoint folds",
            "stream_orders_and_training_seeds": [42, 123, 2024],
            "increments": 3,
            "epochs_per_increment": 8,
            "primary_replay_budget": 0.10,
            "development_unit": "seed 42/fold 1",
            "confirmatory_units": 14,
        },
        "static_detection_confirmatory_10_units": static["confirmatory_10_units"],
        "continual_aggregates_15_units": {
            label: method_aggregate(rows) for label, rows in groups.items()
        },
        "confirmatory_14_unit_comparisons": {
            "supcon_replay_vs_focal_replay": comparison(
                groups["supcon_replay"], groups["focal_replay"]
            ),
            "cl6_replay_vs_focal_replay": comparison(
                groups["cl6_replay"], groups["focal_replay"]
            ),
            "cl6_replay_vs_supcon_replay": comparison(
                groups["cl6_replay"], groups["supcon_replay"]
            ),
            "relation_replay_vs_cl6_replay": comparison(
                groups["relation_replay"], groups["cl6_replay"]
            ),
            "relation_replay_vs_focal_replay": comparison(
                groups["relation_replay"], groups["focal_replay"]
            ),
            "relation_replay_vs_finetune_cl6": comparison(
                groups["relation_replay"], groups["finetune_cl6"]
            ),
            "relation_replay_vs_ewc_cl6": comparison(
                groups["relation_replay"], groups["ewc_cl6"]
            ),
        },
        "buffer_sensitivity_seed42_fivefold": {
            ratio: buffer_summary(rows) for ratio, rows in budget.items()
        },
        "statistical_policy": {
            "test": "Nadeau-Bengio corrected repeated five-fold t-test",
            "multiplicity": "Holm correction over the four primary metrics within each contrast",
            "interval": "crossed seed-fold cluster bootstrap",
            "interpretation": "positive oriented benefit favors the named candidate",
        },
    }
    atomic_text(
        SUMMARY,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    labels = {
        "finetune_cl6": "Fine-tune (LRC; internal run objective CL6)",
        "ewc_cl6": "EWC (LRC; internal run objective CL6)",
        "focal_replay": "Focal + Replay",
        "supcon_replay": "SupCon + Replay",
        "cl6_replay": "LRC + Replay",
        "relation_replay": "LRC + Replay + relation distillation (RCR)",
    }
    lines = [
        "# Task 2：EI会议论文补充实验",
        "",
        "协议：项目互斥五折、3个项目顺序/训练种子、3阶段增量流、"
        "每阶段固定8轮。seed42/fold1作为开发单元，从确认性检验中排除。",
        "",
        "## 持续学习主结果（15个单元描述均值）",
        "",
        "| 方法 | Probe Acc | Forgetting↓ | BWT | Outer F1 | Outer MCC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        row = payload["continual_aggregates_15_units"][label]
        lines.append(
            f"| {labels[label]} | "
            f"{row['final_average_probe_accuracy']['mean']:.4f} | "
            f"{row['average_forgetting']['mean']:.4f} | "
            f"{row['backward_transfer']['mean']:.4f} | "
            f"{row['outer_f1']['mean']:.4f} | "
            f"{row['outer_mcc']['mean']:.4f} | "
            f"{row['outer_pr_auc']['mean']:.4f} |"
        )
    lines.extend([
        "",
        "## 缓存比例敏感性（seed42五折）",
        "",
        "| 缓存比例 | Probe Acc | Forgetting↓ | BWT | Outer F1 | Outer MCC | 训练小时 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for ratio in ("5", "10", "20"):
        row = payload["buffer_sensitivity_seed42_fivefold"][ratio]
        lines.append(
            f"| {ratio}% | "
            f"{row['final_average_probe_accuracy']['mean']:.4f} | "
            f"{row['average_forgetting']['mean']:.4f} | "
            f"{row['backward_transfer']['mean']:.4f} | "
            f"{row['outer_f1']['mean']:.4f} | "
            f"{row['outer_mcc']['mean']:.4f} | "
            f"{row['training_seconds_total']['mean'] / 3600:.2f} |"
        )
    lines.extend([
        "",
        "机器可读结果：",
        "",
        "- `results/workflow/task2_ei_extension_summary.json`",
        "- `results/workflow/task2_ei_extension_raw.csv`",
    ])
    atomic_text(REPORT, "\n".join(lines) + "\n")
    print(json.dumps({
        "summary": str(SUMMARY),
        "raw": str(RAW),
        "report": str(REPORT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
