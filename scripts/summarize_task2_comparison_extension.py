#!/usr/bin/env python3
"""Aggregate the standard-contrastive and seed-100 publication controls."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
WORKFLOW_ROOT = PROJECT_ROOT / "results" / "workflow"
MANIFEST = WORKFLOW_ROOT / "task2_comparison_extension_manifest.json"
DETECTION_RAW = WORKFLOW_ROOT / "task2_detection_sci_raw.csv"
RAW = WORKFLOW_ROOT / "task2_comparison_extension_raw.csv"
SUMMARY = WORKFLOW_ROOT / "task2_comparison_extension_summary.json"
REPORT = PROJECT_ROOT / "reports" / "19_task2_comparison_extension.md"
METRICS = ("f1", "mcc", "pr_auc", "accuracy", "precision", "recall", "roc_auc")
PRIMARY = ("f1", "mcc", "pr_auc")
RAW_METRICS = (*METRICS, "roc_auc_legacy", "tp", "fp", "tn", "fn")


from scripts.summarize_task2_detection_sci import (  # noqa: E402
    corrected_repeated_kfold,
    crossed_cluster_ci,
    flat_bootstrap_ci,
    holm_adjust,
    run_efficiency,
    wilcoxon_p,
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def single_csv(path: Path) -> dict:
    rows = read_csv(path)
    if len(rows) != 1:
        raise RuntimeError(f"expected one test row in {path}, got {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty comparison matrix")
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def mean(values) -> float:
    return statistics.fmean(float(value) for value in values)


def std(values) -> float:
    values = [float(value) for value in values]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def materialize_run(item: dict, group: str, objective: str) -> dict:
    metrics_path = PROJECT_ROOT / item["test_metrics"]
    metrics = single_csv(metrics_path)
    efficiency = run_efficiency(metrics_path, metrics)
    selected_epochs = int(item["selected_train_epochs"])
    if int(efficiency["training_epochs"]) != selected_epochs:
        raise RuntimeError(
            f"{metrics_path}: observed {efficiency['training_epochs']} epochs, "
            f"expected {selected_epochs}"
        )
    if int(item.get("outer_test_evaluations", 0)) != 1:
        raise RuntimeError(f"{metrics_path}: outer-test policy is not single-pass")
    return {
        "comparison_group": group,
        "task": 2,
        "seed": int(item["seed"]),
        "fold": int(item["fold"]),
        "objective": objective,
        "run_name": item["full_run"],
        "selected_train_epochs": selected_epochs,
        "outer_test_evaluations": 1,
        **{
            metric: (
                int(metrics[metric])
                if metric in {"tp", "fp", "tn", "fn"}
                else float(metrics[metric])
            )
            for metric in RAW_METRICS
        },
        **efficiency,
    }


def comparison_statistics(
    reference_rows: list[dict],
    candidate_rows: list[dict],
    *,
    clustered_seeds: bool,
) -> dict:
    reference = {
        (int(row["seed"]), int(row["fold"])): row for row in reference_rows
    }
    candidate = {
        (int(row["seed"]), int(row["fold"])): row for row in candidate_rows
    }
    units = sorted(set(reference) & set(candidate))
    if len(units) != len(reference) or len(units) != len(candidate):
        raise RuntimeError("paired comparison contains unmatched units")
    output = {}
    corrected = {}
    for metric in METRICS:
        old = [float(reference[unit][metric]) for unit in units]
        new = [float(candidate[unit][metric]) for unit in units]
        differences = [right - left for left, right in zip(old, new)]
        keyed = {unit: value for unit, value in zip(units, differences)}
        correction = corrected_repeated_kfold(differences)
        corrected[metric] = correction["p"]
        delta_std = std(differences)
        output[metric] = {
            "units": len(units),
            "reference_mean": mean(old),
            "reference_std": std(old),
            "candidate_mean": mean(new),
            "candidate_std": std(new),
            "mean_delta": mean(differences),
            "delta_std": delta_std,
            "bootstrap_95_ci": (
                crossed_cluster_ci(keyed)
                if clustered_seeds else flat_bootstrap_ci(differences)
            ),
            "wins": sum(value > 0.0 for value in differences),
            "ties": sum(value == 0.0 for value in differences),
            "losses": sum(value < 0.0 for value in differences),
            "wilcoxon_p": wilcoxon_p(differences),
            "corrected_repeated_kfold": correction,
            "paired_cohen_dz": (
                mean(differences) / delta_std if delta_std > 0.0 else math.inf
            ),
        }
    adjusted = holm_adjust({metric: corrected[metric] for metric in PRIMARY})
    for metric, value in adjusted.items():
        output[metric]["holm_corrected_p_primary_metrics"] = value
    return output


def detection_rows_by_objective() -> dict[str, list[dict]]:
    rows = [
        row for row in read_csv(DETECTION_RAW)
        if int(row["seed"]) in {123, 2024}
    ]
    output = {
        "focal": [row for row in rows if row["objective"] == "control"],
        "cl6": [row for row in rows if row["objective"] == "improved"],
    }
    if any(len(values) != 10 for values in output.values()):
        raise RuntimeError("existing confirmatory detector matrix is incomplete")
    return output


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    development = manifest["supcon_development_grid"]
    if len(development) != 9 or any(
        row.get("outer_test_evaluated") is not False for row in development
    ):
        raise RuntimeError("SupCon development grid is incomplete or test-tainted")
    selected = manifest["selected_supcon"]

    extension_rows = [
        materialize_run(item, "standard_contrastive", "supcon")
        for item in manifest["supcon_confirmatory_runs"]
    ]
    extension_rows.extend(
        materialize_run(item, "seed100_alignment", item["objective"])
        for item in manifest["seed100_alignment_runs"]
    )
    expected_keys = {
        ("standard_contrastive", seed, fold, "supcon")
        for seed in (123, 2024) for fold in range(1, 6)
    }
    expected_keys.update({
        ("seed100_alignment", 100, fold, objective)
        for fold in range(1, 6) for objective in ("focal", "cl6")
    })
    keys = [
        (
            row["comparison_group"], int(row["seed"]), int(row["fold"]),
            row["objective"],
        )
        for row in extension_rows
    ]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise RuntimeError("comparison extension has duplicate or missing units")
    extension_rows.sort(
        key=lambda row: (
            row["comparison_group"], row["seed"], row["fold"], row["objective"]
        )
    )
    write_csv(RAW, extension_rows)

    prior = detection_rows_by_objective()
    supcon = [
        row for row in extension_rows
        if row["comparison_group"] == "standard_contrastive"
    ]
    seed100_focal = [
        row for row in extension_rows
        if row["comparison_group"] == "seed100_alignment"
        and row["objective"] == "focal"
    ]
    seed100_cl6 = [
        row for row in extension_rows
        if row["comparison_group"] == "seed100_alignment"
        and row["objective"] == "cl6"
    ]
    payload = {
        "protocol": manifest["protocol"],
        "supcon_definition": {
            "sample_unit": "one Method-Class graph pair embedding",
            "positive_rule": "different graph-pair samples with the same binary label",
            "negative_rule": "different graph-pair samples with different binary labels",
            "total_loss": "original Focal + lambda_label * supervised contrastive loss",
            "projection_head": "128-ReLU-64, L2 normalized, training only",
            "selected_temperature": float(selected["temperature"]),
            "selected_lambda_label": float(selected["lambda_label"]),
            "selection_source": selected["selection_source"],
        },
        "supcon_development_grid_inner_validation_only": development,
        "confirmatory_10_units": {
            "supcon_vs_focal": comparison_statistics(
                prior["focal"], supcon, clustered_seeds=True
            ),
            "cl6_vs_supcon": comparison_statistics(
                supcon, prior["cl6"], clustered_seeds=True
            ),
            "method_means": {
                objective: {
                    metric: mean(row[metric] for row in rows)
                    for metric in METRICS
                }
                for objective, rows in (
                    ("focal", prior["focal"]),
                    ("supcon", supcon),
                    ("cl6", prior["cl6"]),
                )
            },
        },
        "seed100_strict_fivefold": {
            "cl6_vs_focal": comparison_statistics(
                seed100_focal, seed100_cl6, clustered_seeds=False
            ),
            "published_original_external_reference": {
                "seed_in_released_training_code": 100,
                "f1": 0.7421,
                "accuracy": 0.8592,
                "legacy_auc": 0.9303,
                "comparison_role": (
                    "descriptive external reference only because the released "
                    "Java augmentation was unseeded and the original protocol "
                    "selected epochs on the outer test fold"
                ),
            },
        },
        "statistical_policy": {
            "primary_metrics": list(PRIMARY),
            "primary_standard_baseline_contrast": "CL6 minus SupCon",
            "primary_seed100_contrast": "CL6 minus Focal",
            "interval": (
                "crossed seed-fold bootstrap for confirmatory runs; paired fold "
                "bootstrap for seed100"
            ),
            "test": "Nadeau-Bengio corrected repeated five-fold t-test",
            "multiplicity": "Holm over F1, MCC and PR-AUC within each contrast",
            "secondary_test": "paired Wilcoxon signed-rank",
        },
    }
    atomic_json(SUMMARY, payload)

    means = payload["confirmatory_10_units"]["method_means"]
    cl6_supcon = payload["confirmatory_10_units"]["cl6_vs_supcon"]
    seed100 = payload["seed100_strict_fivefold"]["cl6_vs_focal"]
    lines = [
        "# Task 2投稿补充：标准对比学习与seed100对齐",
        "",
        "## 标准图对SupCon对照",
        "",
        "一个Method–Class代码图对作为一个样本；图对嵌入经过训练期投影头后，"
        "同标签图对互为正样本、异标签图对互为负样本。总损失为原Focal损失加"
        "标准监督对比损失。9组温度/权重组合只在seed42、fold1内层验证集选择，"
        "没有读取其外层测试集。",
        "",
        f"选定温度为{selected['temperature']}，对比损失权重为"
        f"{selected['lambda_label']}。正式比较使用seed123/2024的10个确认性单元。",
        "",
        "| 指标 | Focal | SupCon | CL6 | CL6−SupCon | 95% CI | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in PRIMARY:
        row = cl6_supcon[metric]
        interval = row["bootstrap_95_ci"]
        lines.append(
            f"| {metric} | {means['focal'][metric]:.4f} | "
            f"{means['supcon'][metric]:.4f} | {means['cl6'][metric]:.4f} | "
            f"{row['mean_delta']:+.4f} | "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] | "
            f"{row['holm_corrected_p_primary_metrics']:.4g} |"
        )
    lines.extend([
        "",
        "## seed100严格五折对齐",
        "",
        "以下Focal和CL6使用完全相同的数据、五折、seed100、内层验证选轮次和"
        "单次外层测试。原论文数值仍只作外部参照，因为原Java增强没有固定种子，"
        "且原协议按测试集挑选轮次。",
        "",
        "| 指标 | Focal | CL6 | 提升 | 95% CI | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for metric in PRIMARY:
        row = seed100[metric]
        interval = row["bootstrap_95_ci"]
        lines.append(
            f"| {metric} | {row['reference_mean']:.4f} | "
            f"{row['candidate_mean']:.4f} | {row['mean_delta']:+.4f} | "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] | "
            f"{row['holm_corrected_p_primary_metrics']:.4g} |"
        )
    lines.extend([
        "",
        "原论文外部参考：F1=0.7421、ACC=0.8592、legacy AUC=0.9303。",
        "",
        "机器可读结果：",
        "",
        "- `results/workflow/task2_comparison_extension_manifest.json`",
        "- `results/workflow/task2_comparison_extension_raw.csv`",
        "- `results/workflow/task2_comparison_extension_summary.json`",
        "",
    ])
    atomic_text(REPORT, "\n".join(lines))
    print(json.dumps({
        "raw_rows": len(extension_rows),
        "selected_supcon": payload["supcon_definition"],
        "summary": str(SUMMARY),
        "report": str(REPORT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
