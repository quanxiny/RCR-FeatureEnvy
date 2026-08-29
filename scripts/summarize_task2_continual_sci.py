#!/usr/bin/env python3
"""Summarize the multi-fold, multi-order Task-2 continual study."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_task2_detection_sci import (
    corrected_repeated_kfold,
    flat_bootstrap_ci,
    crossed_cluster_ci,
    holm_adjust,
    wilcoxon_p,
)


WORKFLOW_ROOT = PROJECT_ROOT / "results" / "workflow"
MANIFEST = WORKFLOW_ROOT / "task2_continual_sci_manifest.json"
AGGREGATE = PROJECT_ROOT / "results" / "continual" / "summary.csv"
RAW = WORKFLOW_ROOT / "task2_continual_sci_raw.csv"
SUMMARY = WORKFLOW_ROOT / "task2_continual_sci_summary.json"
REPORT = PROJECT_ROOT / "reports" / "16_task2_continual_sci.md"
METRICS = (
    "final_average_probe_accuracy",
    "average_forgetting",
    "maximum_forgetting",
    "backward_transfer",
    "outer_f1",
    "outer_mcc",
    "outer_pr_auc",
)
REPORTING_METRICS = (
    *METRICS,
    "outer_precision",
    "outer_recall",
    "outer_accuracy",
    "outer_roc_auc",
    "outer_roc_auc_legacy",
    "outer_tp",
    "outer_fp",
    "outer_tn",
    "outer_fn",
    "outer_inference_seconds",
    "outer_samples",
    "training_seconds_total",
    "wall_seconds_total",
)
PRIMARY = (
    "final_average_probe_accuracy", "average_forgetting", "outer_f1", "outer_mcc"
)
METHOD_LABELS = {
    "naive": "Fine-tune",
    "ewc": "EWC",
    "replay": "Replay",
    "replay_relation": "CL6 + Replay + relation distillation",
    "joint": "Joint upper bound",
}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def mean(values) -> float:
    return statistics.fmean(float(value) for value in values)


def std(values) -> float:
    values = [float(value) for value in values]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def orient(metric: str, proposed: float, baseline: float) -> float:
    if metric in {"average_forgetting", "maximum_forgetting"}:
        return baseline - proposed
    return proposed - baseline


def paired_comparison(
    proposed_rows: list[dict], baseline_rows: list[dict]
) -> dict:
    proposed = {
        (int(row["seed"]), int(row["fold"])): row for row in proposed_rows
    }
    baseline = {
        (int(row["seed"]), int(row["fold"])): row for row in baseline_rows
    }
    units = sorted(set(proposed) & set(baseline))
    output = {}
    primary_p = {}
    for metric in METRICS:
        proposed_values = [float(proposed[unit][metric]) for unit in units]
        baseline_values = [float(baseline[unit][metric]) for unit in units]
        benefits = [
            orient(metric, new, old)
            for new, old in zip(proposed_values, baseline_values)
        ]
        keyed = {unit: value for unit, value in zip(units, benefits)}
        corrected = corrected_repeated_kfold(benefits)
        if metric in PRIMARY:
            primary_p[metric] = corrected["p"]
        benefit_std = std(benefits)
        output[metric] = {
            "units": len(units),
            "proposed_mean": mean(proposed_values),
            "baseline_mean": mean(baseline_values),
            "oriented_benefit": mean(benefits),
            "benefit_definition": (
                "baseline - proposed (lower is better)"
                if metric in {"average_forgetting", "maximum_forgetting"}
                else "proposed - baseline (higher is better)"
            ),
            "crossed_seed_fold_bootstrap_95_ci": (
                crossed_cluster_ci(keyed)
                if len({unit[0] for unit in units}) > 1
                and len({unit[1] for unit in units}) > 1
                else flat_bootstrap_ci(benefits)
            ),
            "wins": sum(value > 0.0 for value in benefits),
            "ties": sum(value == 0.0 for value in benefits),
            "losses": sum(value < 0.0 for value in benefits),
            "wilcoxon_p": wilcoxon_p(benefits),
            "corrected_repeated_kfold": corrected,
            "paired_cohen_dz": (
                mean(benefits) / benefit_std if benefit_std > 0.0 else math.inf
            ),
        }
    adjusted = holm_adjust(primary_p)
    for metric, value in adjusted.items():
        output[metric]["holm_corrected_p_primary_metrics"] = value
    return output


def method_aggregate(rows: list[dict]) -> dict:
    output = {"units": len(rows)}
    for metric in REPORTING_METRICS:
        values = [
            row[metric] for row in rows
            if row.get(metric) is not None and row.get(metric) != ""
        ]
        output[metric] = {
            "mean": mean(values) if values else None,
            "std": std(values) if values else None,
            "defined_units": len(values),
        }
    return output


def fivefold_ablation(
    candidate: list[dict], reference: list[dict]
) -> dict:
    candidate_by_fold = {int(row["fold"]): row for row in candidate}
    reference_by_fold = {int(row["fold"]): row for row in reference}
    if set(candidate_by_fold) != set(range(1, 6)):
        raise RuntimeError("continual ablation is not complete on five folds")
    output = {}
    for metric in METRICS:
        candidate_values = [
            float(candidate_by_fold[fold][metric]) for fold in range(1, 6)
        ]
        reference_values = [
            float(reference_by_fold[fold][metric]) for fold in range(1, 6)
        ]
        benefits = [
            orient(metric, reference_value, candidate_value)
            for candidate_value, reference_value in zip(
                candidate_values, reference_values
            )
        ]
        # Here benefit is full-reference over the ablated candidate.
        output[metric] = {
            "candidate_mean": mean(candidate_values),
            "full_reference_mean": mean(reference_values),
            "full_model_oriented_benefit": mean(benefits),
            "bootstrap_95_ci": flat_bootstrap_ci(benefits),
            "full_model_wins": sum(value > 0.0 for value in benefits),
            "wilcoxon_p": wilcoxon_p(benefits),
        }
    return output


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with AGGREGATE.open(newline="", encoding="utf-8") as handle:
        aggregate_by_run = {
            row["run_name"]: row for row in csv.DictReader(handle)
        }
    selected = []
    for record in manifest["runs"]:
        if record["run_name"] not in aggregate_by_run:
            raise RuntimeError(f"missing aggregate row for {record['run_name']}")
        source = aggregate_by_run[record["run_name"]]
        selected.append({
            "kind": record["kind"],
            "variant": record.get("variant", ""),
            "run_name": record["run_name"],
            "method": record["method"],
            "base_objective": record["base_objective"],
            "seed": int(record["seed"]),
            "fold": int(record["fold"]),
            **{
                metric: (
                    float(source[metric]) if source.get(metric, "") != ""
                    else None
                )
                for metric in REPORTING_METRICS
            },
        })
    selected.sort(key=lambda row: (
        row["kind"], row["seed"], row["fold"], row["method"], row["variant"]
    ))
    selected_keys = [
        (
            row["kind"], int(row["seed"]), int(row["fold"]), row["method"],
            row["base_objective"], row["variant"],
        )
        for row in selected
    ]
    expected_keys = {
        ("primary", seed, fold, method, "cl6", "")
        for seed in (42, 123, 2024)
        for fold in range(1, 6)
        for method in ("naive", "ewc", "replay", "replay_relation")
    }
    expected_keys.update({
        ("upper_bound", 42, fold, "joint", "cl6", "")
        for fold in range(1, 6)
    })
    expected_keys.update({
        ("ablation", 42, fold, method, objective, variant)
        for fold in range(1, 6)
        for variant, method, objective in (
            ("focal_replay", "replay", "focal"),
            ("logit_only", "replay_kd", "cl6"),
            ("no_attention", "replay_relation", "cl6"),
        )
    })
    if (
        len(selected_keys) != len(set(selected_keys))
        or set(selected_keys) != expected_keys
    ):
        raise RuntimeError(
            "continual matrix contains duplicate, missing, or unexpected units"
        )
    write_csv(RAW, selected)

    primary = [row for row in selected if row["kind"] == "primary"]
    by_method = {
        method: [row for row in primary if row["method"] == method]
        for method in ("naive", "ewc", "replay", "replay_relation")
    }
    for method, rows in by_method.items():
        if len(rows) != 15:
            raise RuntimeError(f"{method} has {len(rows)} primary units, expected 15")
    proposed = by_method["replay_relation"]
    confirmatory_by_method = {
        method: [
            row for row in rows
            if (int(row["seed"]), int(row["fold"])) != (42, 1)
        ]
        for method, rows in by_method.items()
    }
    for method, rows in confirmatory_by_method.items():
        if len(rows) != 14:
            raise RuntimeError(
                f"{method} has {len(rows)} confirmatory units, expected 14"
            )
    confirmatory_proposed = confirmatory_by_method["replay_relation"]
    comparisons = {
        method: paired_comparison(
            confirmatory_proposed, confirmatory_by_method[method]
        )
        for method in ("naive", "ewc", "replay")
    }
    descriptive_comparisons = {
        method: paired_comparison(proposed, rows)
        for method, rows in by_method.items()
        if method != "replay_relation"
    }
    upper = [row for row in selected if row["kind"] == "upper_bound"]
    if len(upper) != 5:
        raise RuntimeError("joint upper bound is not complete on five folds")
    full_seed42 = [row for row in proposed if int(row["seed"]) == 42]
    ablation_rows = [row for row in selected if row["kind"] == "ablation"]
    ablations = {}
    for variant in ("logit_only", "no_attention"):
        ablations[variant] = fivefold_ablation(
            [row for row in ablation_rows if row["variant"] == variant],
            full_seed42,
        )
    # Focal replay isolates the CL6 detector objective and is compared with
    # CL6 replay, not with the relation-distillation method.
    ablations["focal_replay"] = fivefold_ablation(
        [row for row in ablation_rows if row["variant"] == "focal_replay"],
        [row for row in by_method["replay"] if int(row["seed"]) == 42],
    )
    payload = {
        "protocol": manifest["protocol"],
        "method_aggregates_15_units": {
            method: method_aggregate(rows) for method, rows in by_method.items()
        },
        "proposed_vs_baselines": comparisons,
        "proposed_vs_baselines_confirmatory_14_units": comparisons,
        "proposed_vs_baselines_all_15_units_descriptive": descriptive_comparisons,
        "joint_upper_bound_seed42_fivefold": method_aggregate(upper),
        "continual_ablation_seed42_fivefold": ablations,
        "statistical_policy": {
            "oriented_benefit": "positive always favors the proposed/full method",
            "primary_metrics": list(PRIMARY),
            "primary_test": "Nadeau-Bengio corrected repeated five-fold t-test",
            "multiplicity": "Holm within each proposed-vs-baseline comparison",
            "interval": "crossed-cluster bootstrap over stream seed and outer fold",
            "development_unit": "seed 42 fold 1, excluded from confirmatory tests",
            "confirmatory_units": 14,
            "alpha": 0.05,
        },
    }
    atomic_text(
        SUMMARY,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    lines = [
        "# Task 2 CL6：持续学习完整实验",
        "",
        "采用 5 个项目互斥外层折和 3 个项目顺序/训练种子。所有方法使用相同的"
        "三阶段项目流、固定 8 epoch/stage 和相同探针；探针与外层测试只用于评估。",
        "",
        "| 方法 | Final probe acc | Avg forgetting↓ | BWT | Outer F1 | Outer MCC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("naive", "ewc", "replay", "replay_relation"):
        row = payload["method_aggregates_15_units"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{row['final_average_probe_accuracy']['mean']:.4f} | "
            f"{row['average_forgetting']['mean']:.4f} | "
            f"{row['backward_transfer']['mean']:.4f} | "
            f"{row['outer_f1']['mean']:.4f} | "
            f"{row['outer_mcc']['mean']:.4f} |"
        )
    lines.extend([
        "",
        "完整方法相对各基线的确认性定向收益（排除开发单元 seed42/fold1，n=14；"
        "遗忘指标已反向，因此正数始终表示更好）：",
        "",
        "| 基线 | Probe acc 收益 | 遗忘减少 | Outer F1 收益 | Outer MCC 收益 |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in ("naive", "ewc", "replay"):
        row = comparisons[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{row['final_average_probe_accuracy']['oriented_benefit']:+.4f} | "
            f"{row['average_forgetting']['oriented_benefit']:+.4f} | "
            f"{row['outer_f1']['oriented_benefit']:+.4f} | "
            f"{row['outer_mcc']['oriented_benefit']:+.4f} |"
        )
    lines.extend([
        "",
        "消融中，Focal Replay 对比 CL6 Replay；logit-only/no-attention 对比完整关系蒸馏。",
        "下表为完整方案的定向收益。",
        "",
        "| 消融 | Probe acc | 遗忘减少 | Outer F1 | Outer MCC |",
        "|---|---:|---:|---:|---:|",
    ])
    labels = {
        "focal_replay": "Replay 去掉 CL6",
        "logit_only": "仅 logit 蒸馏",
        "no_attention": "去掉注意力蒸馏",
    }
    for variant, label in labels.items():
        row = ablations[variant]
        lines.append(
            f"| {label} | "
            f"{row['final_average_probe_accuracy']['full_model_oriented_benefit']:+.4f} | "
            f"{row['average_forgetting']['full_model_oriented_benefit']:+.4f} | "
            f"{row['outer_f1']['full_model_oriented_benefit']:+.4f} | "
            f"{row['outer_mcc']['full_model_oriented_benefit']:+.4f} |"
        )
    lines.extend([
        "",
        "机器可读结果：",
        "",
        "- `results/workflow/task2_continual_sci_summary.json`",
        "- `results/workflow/task2_continual_sci_raw.csv`",
        "- `results/continual/forgetting_matrix.csv`",
        "",
    ])
    atomic_text(REPORT, "\n".join(lines))
    print(json.dumps({
        "summary": str(SUMMARY), "report": str(REPORT),
        "primary_runs": len(primary), "all_selected_runs": len(selected),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
