#!/usr/bin/env python3
"""Aggregate repeated five-fold Task-2 detection and fixed-budget ablations."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = PROJECT_ROOT / "results" / "workflow"
MANIFEST = WORKFLOW_ROOT / "task2_detection_sci_manifest.json"
RAW = WORKFLOW_ROOT / "task2_detection_sci_raw.csv"
ABLATION_RAW = WORKFLOW_ROOT / "task2_detection_sci_ablation_raw.csv"
MECHANISM_RAW = WORKFLOW_ROOT / "task2_detection_sci_mechanism_raw.csv"
SUMMARY = WORKFLOW_ROOT / "task2_detection_sci_summary.json"
REPORT = PROJECT_ROOT / "reports" / "15_task2_detection_sci.md"
METRICS = (
    "f1", "mcc", "pr_auc", "accuracy", "precision", "recall", "roc_auc",
)
RAW_METRICS = (*METRICS, "roc_auc_legacy", "tp", "fp", "tn", "fn")
PRIMARY = ("f1", "mcc", "pr_auc")
EFFICIENCY_METRICS = ("training_ms_per_sample", "inference_ms_per_sample")
RAW_EFFICIENCY_METRICS = (
    "training_seconds_total", "training_epochs", "training_samples_total",
    "training_ms_per_sample", "inference_seconds", "test_samples",
    "inference_ms_per_sample",
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


def single_csv(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one outer-test row in {path}, found {len(rows)}"
        )
    return rows[0]


def run_efficiency(test_metrics_path: Path, test_metrics: dict) -> dict[str, float]:
    """Normalize measured full-train and inference wall time by sample count."""
    curve_path = test_metrics_path.parent / "training_curves.csv"
    with curve_path.open(newline="", encoding="utf-8") as handle:
        curves = list(csv.DictReader(handle))
    if not curves:
        raise RuntimeError(f"training curve is empty: {curve_path}")
    train_seconds = sum(float(row["train_seconds"]) for row in curves)
    train_samples = sum(int(row["train_samples"]) for row in curves)
    inference_seconds = float(test_metrics["inference_seconds"])
    test_samples = int(test_metrics["test_samples"])
    if (
        train_seconds <= 0.0 or train_samples <= 0
        or inference_seconds <= 0.0 or test_samples <= 0
    ):
        raise RuntimeError(f"invalid efficiency evidence for {test_metrics_path}")
    return {
        "training_seconds_total": train_seconds,
        "training_epochs": len(curves),
        "training_samples_total": train_samples,
        "training_ms_per_sample": 1000.0 * train_seconds / train_samples,
        "inference_seconds": inference_seconds,
        "test_samples": test_samples,
        "inference_ms_per_sample": 1000.0 * inference_seconds / test_samples,
    }


def mechanism_row(
    *, seed: int, fold: int, run_name: str, curves: list[dict]
) -> dict:
    """Summarize observed CL6 activation without using any test outcome."""
    active = [row for row in curves if float(row["directed_ramp"]) > 0.0]
    if not active:
        raise RuntimeError(f"{run_name}: CL6 never became active")
    train_exposures = sum(int(row["train_samples"]) for row in active)
    contributing = sum(
        int(row["directed_contributing_samples"]) for row in active
    )
    evidence_pairs = sum(int(row["directed_evidence_pairs"]) for row in active)
    infonce_negatives = sum(
        int(row["directed_infonce_negatives"]) for row in active
    )
    classification_total = sum(
        float(row["train_loss"]) * int(row["train_samples"])
        for row in active
    )
    auxiliary_total = sum(
        float(row["directed_effective_aux_loss"]) * int(row["train_samples"])
        for row in active
    )
    if train_exposures <= 0 or contributing <= 0 or classification_total <= 0.0:
        raise RuntimeError(f"{run_name}: invalid CL6 mechanism evidence")
    epoch_ratios = [
        float(row["directed_effective_aux_loss"]) / float(row["train_loss"])
        for row in active if float(row["train_loss"]) > 0.0
    ]
    return {
        "seed": seed,
        "fold": fold,
        "run_name": run_name,
        "active_epochs": len(active),
        "active_sample_exposures": train_exposures,
        "contributing_sample_exposures": contributing,
        "contributing_exposure_rate": contributing / train_exposures,
        "directed_evidence_pairs": evidence_pairs,
        "evidence_pairs_per_contributing_exposure": evidence_pairs / contributing,
        "directed_infonce_negatives": infonce_negatives,
        "effective_aux_to_classification_ratio": (
            auxiliary_total / classification_total
        ),
        "max_epoch_aux_to_classification_ratio": max(epoch_ratios),
    }


def mechanism_statistics(rows: list[dict]) -> dict:
    fields = (
        "contributing_exposure_rate",
        "evidence_pairs_per_contributing_exposure",
        "effective_aux_to_classification_ratio",
        "max_epoch_aux_to_classification_ratio",
    )
    return {
        "runs": len(rows),
        "definitions": {
            "exposure": "one graph-pair presentation in an active CL6 epoch",
            "contributing": "a positive exposure with at least one directed evidence pair",
            "aux_ratio": "sample-weighted effective auxiliary loss / classification loss",
        },
        "aggregates": {
            field: {
                "mean": mean(row[field] for row in rows),
                "std": std(row[field] for row in rows),
                "min": min(float(row[field]) for row in rows),
                "max": max(float(row[field]) for row in rows),
            }
            for field in fields
        },
    }


def threshold_policy(seed: int, payload: dict) -> str:
    """Audit that confirmatory threshold diagnostics never replay outer test."""
    if seed == 42 and payload.get("outer_test_evaluated") is not False:
        return "development_legacy_secondary_outer_analysis"
    if payload.get("outer_test_evaluated") is not False:
        raise RuntimeError(
            f"seed {seed}: confirmatory threshold diagnostics replayed outer test"
        )
    if not payload.get("locked_primary_outer_test_metrics"):
        raise RuntimeError(
            f"seed {seed}: threshold diagnostics lack locked primary result reference"
        )
    return "validation_only_no_outer_replay"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def crossed_cluster_ci(
    differences: dict[tuple[int, int], float], repetitions: int = 20_000
) -> list[float]:
    """Pigeonhole bootstrap for crossed seed and outer-fold clusters.

    Seed and fold are crossed rather than nested: all seeds reuse the same
    outer folds. Independently resampling both cluster dimensions preserves
    both sources of dependence and supports an incomplete crossed matrix when
    explicitly identified development cells must be removed.
    """
    if not differences:
        raise ValueError("crossed cluster bootstrap requires observations")
    seeds = np.asarray(sorted({seed for seed, _fold in differences}))
    folds = np.asarray(sorted({fold for _seed, fold in differences}))
    rng = np.random.default_rng(42)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        while True:
            selected_seeds = rng.choice(seeds, len(seeds), replace=True)
            selected_folds = rng.choice(folds, len(folds), replace=True)
            seed_weights = {
                int(seed): int(np.count_nonzero(selected_seeds == seed))
                for seed in seeds
            }
            fold_weights = {
                int(fold): int(np.count_nonzero(selected_folds == fold))
                for fold in folds
            }
            numerator = 0.0
            denominator = 0
            for (seed, fold), value in differences.items():
                weight = seed_weights[seed] * fold_weights[fold]
                numerator += weight * float(value)
                denominator += weight
            if denominator:
                draws[index] = numerator / denominator
                break
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


# Import-compatible alias for older local analysis notebooks. New reports and
# machine-readable output use the scientifically precise name.
hierarchical_ci = crossed_cluster_ci


def flat_bootstrap_ci(values: list[float], repetitions: int = 20_000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    draws = array[indices].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def wilcoxon_p(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if np.allclose(array, 0.0):
        return 1.0
    return float(
        stats.wilcoxon(
            array, zero_method="pratt", alternative="two-sided", method="auto"
        ).pvalue
    )


def corrected_repeated_kfold(values: list[float], folds: int = 5) -> dict:
    """Nadeau-Bengio corrected repeated-k-fold t test.

    The correction uses n_test/n_train = 1/(k-1), appropriate for the fixed
    project-disjoint five-fold protocol. It is deliberately more conservative
    than treating all 15 fold/seed observations as independent.
    """
    values = [float(value) for value in values]
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    correction = 1.0 / len(values) + 1.0 / (folds - 1)
    standard_error = math.sqrt(correction * variance)
    if standard_error == 0.0:
        statistic = math.inf if mean(values) != 0.0 else 0.0
        p_value = 0.0 if statistic else 1.0
    else:
        statistic = mean(values) / standard_error
        p_value = float(2.0 * stats.t.sf(abs(statistic), df=len(values) - 1))
    return {
        "t": statistic,
        "df": len(values) - 1,
        "p": p_value,
        "standard_error": standard_error,
        "correction": correction,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def paired_statistics(
    rows: list[dict], *, include: set[tuple[int, int]] | None = None
) -> dict:
    by_key = {
        (int(row["seed"]), int(row["fold"]), row["objective"]): row
        for row in rows
    }
    units = sorted({
        (seed, fold) for seed, fold, objective in by_key
        if objective == "control"
        and (seed, fold, "improved") in by_key
        and (include is None or (seed, fold) in include)
    })
    output = {}
    corrected_p = {}
    for metric in METRICS:
        control = [float(by_key[(*unit, "control")][metric]) for unit in units]
        improved = [float(by_key[(*unit, "improved")][metric]) for unit in units]
        differences = [new - old for old, new in zip(control, improved)]
        keyed = {unit: value for unit, value in zip(units, differences)}
        correction = corrected_repeated_kfold(differences)
        corrected_p[metric] = correction["p"]
        effect_dz = (
            mean(differences) / std(differences)
            if std(differences) > 0.0 else math.inf
        )
        output[metric] = {
            "units": len(units),
            "control_mean": mean(control),
            "control_std": std(control),
            "improved_mean": mean(improved),
            "improved_std": std(improved),
            "mean_delta": mean(differences),
            "delta_std": std(differences),
            "crossed_seed_fold_bootstrap_95_ci": (
                crossed_cluster_ci(keyed)
                if len({unit[0] for unit in units}) > 1
                and len({unit[1] for unit in units}) > 1
                else flat_bootstrap_ci(differences)
            ),
            "wins": sum(value > 0.0 for value in differences),
            "ties": sum(value == 0.0 for value in differences),
            "losses": sum(value < 0.0 for value in differences),
            "wilcoxon_p": wilcoxon_p(differences),
            "corrected_repeated_kfold": correction,
            "paired_cohen_dz": effect_dz,
        }
    adjusted = holm_adjust({metric: corrected_p[metric] for metric in PRIMARY})
    for metric, value in adjusted.items():
        output[metric]["holm_corrected_p_primary_metrics"] = value
    return output


def efficiency_statistics(rows: list[dict]) -> dict:
    by_key = {
        (int(row["seed"]), int(row["fold"]), row["objective"]): row
        for row in rows
    }
    units = sorted({
        (seed, fold) for seed, fold, objective in by_key
        if objective == "control" and (seed, fold, "improved") in by_key
    })
    output = {}
    for metric in EFFICIENCY_METRICS:
        control = [float(by_key[(*unit, "control")][metric]) for unit in units]
        improved = [float(by_key[(*unit, "improved")][metric]) for unit in units]
        differences = [new - old for old, new in zip(control, improved)]
        percent_changes = [
            100.0 * (new / old - 1.0) for old, new in zip(control, improved)
        ]
        output[metric] = {
            "units": len(units),
            "control_mean": mean(control),
            "improved_mean": mean(improved),
            "mean_absolute_change": mean(differences),
            "mean_paired_percent_change": mean(percent_changes),
            "crossed_seed_fold_bootstrap_95_ci_absolute_change": crossed_cluster_ci({
                unit: value for unit, value in zip(units, differences)
            }),
            "crossed_seed_fold_bootstrap_95_ci_percent_change": crossed_cluster_ci({
                unit: value for unit, value in zip(units, percent_changes)
            }),
        }
    output["parameter_policy"] = {
        "classifier_parameters_added_at_inference": 0,
        "training_parameters_added": 0,
        "architecture": "the original GCN classifier is unchanged",
    }
    return output


def ablation_statistics(rows: list[dict]) -> dict:
    by_key = {
        (int(row["fold"]), row["variant"]): row for row in rows
    }
    variants = sorted({row["variant"] for row in rows if row["variant"] != "cl6_full"})
    output = {}
    for variant in variants:
        metrics = {}
        for metric in METRICS:
            reference = [
                float(by_key[(fold, "cl6_full")][metric]) for fold in range(1, 6)
            ]
            candidate = [
                float(by_key[(fold, variant)][metric]) for fold in range(1, 6)
            ]
            differences = [new - full for full, new in zip(reference, candidate)]
            metrics[metric] = {
                "variant_mean": mean(candidate),
                "cl6_full_mean": mean(reference),
                "mean_delta_vs_cl6_full": mean(differences),
                "bootstrap_95_ci": flat_bootstrap_ci(differences),
                "wins": sum(value > 0.0 for value in differences),
                "wilcoxon_p": wilcoxon_p(differences),
            }
        output[variant] = metrics
    return output


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # Every seed-42 fold was inspected before the multi-seed protocol freeze.
    # Normalize the persisted label here because a detector worker launched
    # under the earlier label may still be alive. Numerical settings do not
    # change; only the two wholly new seeds are confirmatory.
    manifest["protocol"].update({
        "development_units": (
            "all five seed-42 folds observed before protocol freeze"
        ),
        "confirmatory_units": (
            "all five folds for new seeds 123 and 2024 (10 paired units)"
        ),
    })
    manifest["protocol"].pop("development_unit", None)
    atomic_json(MANIFEST, manifest)
    detector_rows = []
    for item in manifest["detector_runs"]:
        for objective in ("control", "improved"):
            test_metrics_path = PROJECT_ROOT / item[objective]["test_metrics"]
            values = single_csv(test_metrics_path)
            efficiency = run_efficiency(test_metrics_path, values)
            selected_train_epochs = int(
                item[objective]["selected_train_epochs"]
            )
            if int(efficiency["training_epochs"]) != selected_train_epochs:
                raise RuntimeError(
                    f"{test_metrics_path}: trained {efficiency['training_epochs']} "
                    f"epochs, selected protocol requires {selected_train_epochs}"
                )
            threshold_path = PROJECT_ROOT / item[objective]["threshold_transfer"]
            threshold_payload = json.loads(
                threshold_path.read_text(encoding="utf-8")
            )
            test_policy = threshold_policy(int(item["seed"]), threshold_payload)
            detector_rows.append({
                "task": 2,
                "seed": int(item["seed"]),
                "fold": int(item["fold"]),
                "objective": objective,
                "run_name": item[objective]["full_run"],
                "selected_train_epochs": selected_train_epochs,
                "outer_test_evaluations": 1,
                "threshold_diagnostic_policy": test_policy,
                **{
                    metric: (
                        int(values[metric])
                        if metric in {"tp", "fp", "tn", "fn"}
                        else float(values[metric])
                    )
                    for metric in RAW_METRICS
                },
                **efficiency,
            })
    expected = 3 * 5 * 2
    if len(detector_rows) != expected:
        raise RuntimeError(
            f"expected {expected} detector result rows, got {len(detector_rows)}"
        )
    detector_keys = [
        (int(row["seed"]), int(row["fold"]), row["objective"])
        for row in detector_rows
    ]
    expected_detector_keys = {
        (seed, fold, objective)
        for seed in (42, 123, 2024)
        for fold in range(1, 6)
        for objective in ("control", "improved")
    }
    if (
        len(detector_keys) != len(set(detector_keys))
        or set(detector_keys) != expected_detector_keys
    ):
        raise RuntimeError(
            "detector matrix contains duplicate or missing fold/seed/objective units"
        )
    detector_rows.sort(
        key=lambda row: (row["seed"], row["fold"], row["objective"])
    )
    write_csv(RAW, detector_rows)

    mechanism_rows = []
    for row in detector_rows:
        if row["objective"] != "improved":
            continue
        curve_path = (
            PROJECT_ROOT / "results" / "contrastive" / "runs"
            / row["run_name"] / "training_curves.csv"
        )
        with curve_path.open(newline="", encoding="utf-8") as handle:
            curves = list(csv.DictReader(handle))
        mechanism_rows.append(mechanism_row(
            seed=int(row["seed"]), fold=int(row["fold"]),
            run_name=row["run_name"], curves=curves,
        ))
    if len(mechanism_rows) != 15:
        raise RuntimeError(
            f"expected 15 CL6 mechanism rows, got {len(mechanism_rows)}"
        )
    write_csv(MECHANISM_RAW, mechanism_rows)

    detector_lookup = {
        (int(row["seed"]), int(row["fold"]), row["objective"]): row
        for row in detector_rows
    }
    ablation_rows = []
    for fold in range(1, 6):
        source = detector_lookup[(42, fold, "improved")]
        ablation_rows.append({
            "seed": 42, "fold": fold, "variant": "cl6_full",
            "run_name": source["run_name"],
            "fixed_train_epochs": int(source["selected_train_epochs"]),
            "epoch_budget_source": "full_cl6_validation_selected_epoch",
            **{metric: source[metric] for metric in RAW_METRICS},
            **{metric: source[metric] for metric in RAW_EFFICIENCY_METRICS},
        })
    for item in manifest["ablation_runs"]:
        test_metrics_path = PROJECT_ROOT / item["test_metrics"]
        values = single_csv(test_metrics_path)
        efficiency = run_efficiency(test_metrics_path, values)
        selected_train_epochs = int(
            item.get("selected_train_epochs", item.get("fixed_train_epochs", 0))
        )
        if int(efficiency["training_epochs"]) != selected_train_epochs:
            raise RuntimeError(
                f"{test_metrics_path}: trained {efficiency['training_epochs']} "
                f"epochs, fixed ablation budget requires {selected_train_epochs}"
            )
        ablation_rows.append({
            "seed": int(item["seed"]),
            "fold": int(item["fold"]),
            "variant": item["variant"],
            "run_name": item["run_name"],
            "fixed_train_epochs": selected_train_epochs,
            "epoch_budget_source": item.get(
                "reference_epoch_source", "full_cl6_validation_selected_epoch"
            ),
            **{
                metric: (
                    int(values[metric])
                    if metric in {"tp", "fp", "tn", "fn"}
                    else float(values[metric])
                )
                for metric in RAW_METRICS
            },
            **efficiency,
        })
    expected_ablation = 5 * (1 + 5)
    if len(ablation_rows) != expected_ablation:
        raise RuntimeError(
            f"expected {expected_ablation} ablation rows, got {len(ablation_rows)}"
        )
    ablation_keys = [
        (int(row["seed"]), int(row["fold"]), row["variant"])
        for row in ablation_rows
    ]
    expected_ablation_keys = {
        (42, fold, variant)
        for fold in range(1, 6)
        for variant in (
            "remove_directed_loss", "cl6_full", "no_curriculum",
            "no_aux_cap", "add_mutual", "add_negative",
        )
    }
    if (
        len(ablation_keys) != len(set(ablation_keys))
        or set(ablation_keys) != expected_ablation_keys
    ):
        raise RuntimeError(
            "detector ablation matrix contains duplicate or missing fold/variant units"
        )
    for fold in range(1, 6):
        fold_rows = [row for row in ablation_rows if int(row["fold"]) == fold]
        budgets = {int(row["fixed_train_epochs"]) for row in fold_rows}
        reference_budget = int(
            detector_lookup[(42, fold, "improved")]["selected_train_epochs"]
        )
        if len(fold_rows) != 6 or budgets != {reference_budget}:
            raise RuntimeError(
                f"fold {fold}: ablation epochs are not matched to CL6 "
                f"budget {reference_budget}: {sorted(budgets)}"
            )
    ablation_rows.sort(key=lambda row: (row["fold"], row["variant"]))
    write_csv(ABLATION_RAW, ablation_rows)

    all_units = {(seed, fold) for seed in (42, 123, 2024) for fold in range(1, 6)}
    confirmatory = {
        (seed, fold) for seed in (123, 2024) for fold in range(1, 6)
    }
    folds_two_to_five = {
        (seed, fold)
        for seed in (42, 123, 2024)
        for fold in range(2, 6)
    }
    payload = {
        "protocol": manifest["protocol"],
        "published_reference_external_only": {
            "source": "CG-LSMN-main/result.xlsx",
            "precision": 0.7917,
            "recall": 0.7006,
            "f1": 0.7421,
            "accuracy": 0.8592,
            "legacy_auc": 0.9303,
            "reason_not_paired": (
                "the original Java data augmentation was unseeded; causal "
                "comparison uses protocol-matched Focal and CL6 runs"
            ),
        },
        "all_15_units": paired_statistics(detector_rows),
        "confirmatory_10_units_new_seeds_only": paired_statistics(
            detector_rows, include=confirmatory
        ),
        "descriptive_folds_2_to_5_all_seeds": paired_statistics(
            detector_rows, include=folds_two_to_five
        ),
        "efficiency_15_units": efficiency_statistics(detector_rows),
        "mechanism_15_cl6_runs": mechanism_statistics(mechanism_rows),
        "ablation_seed42_fivefold": ablation_statistics(ablation_rows),
        "statistical_policy": {
            "primary_metrics": list(PRIMARY),
            "primary_test": "Nadeau-Bengio corrected repeated five-fold t-test",
            "multiplicity": "Holm correction over F1, MCC and PR-AUC",
            "interval": "crossed-cluster bootstrap over seed and outer fold",
            "secondary_test": "paired Wilcoxon signed-rank",
            "alpha": 0.05,
        },
        "outer_test_policy_audit": {
            "primary_evaluations_per_run": 1,
            "confirmatory_threshold_diagnostics": "validation only; no outer replay",
            "seed42_development_note": (
                "pre-protocol threshold files included a secondary deterministic "
                "outer replay; these diagnostics are excluded from every primary "
                "and confirmatory statistic"
            ),
        },
    }
    atomic_json(SUMMARY, payload)

    all_stats = payload["all_15_units"]
    confirm_stats = payload["confirmatory_10_units_new_seeds_only"]
    lines = [
        "# Task 2 CL6：五折三种子检测与消融",
        "",
        "协议：固定的项目互斥五折；每个训练种子都只用内层项目互斥验证集选择轮数，"
        "随后在完整外层训练集按固定轮数重训，并且只评估一次外层测试集。seed 42 "
        "五折均在协议冻结前观察过，只作开发结果；seed 123/2024 的 10 个新单元"
        "单独报告为确认性结果。",
        "",
        "原论文 `result.xlsx` 的 Task 2 五折 F1 为 0.7421。由于原 Java 数据增强"
        "没有固定随机数，该值只作外部量级参照；下表的提升均来自本次同数据、同折、"
        "同种子的严格配对。",
        "seed 123/2024 的确认性运行每个模型只执行一次固定 0.5 外层测试；"
        "验证阈值诊断不再次读取外层测试。seed 42 的历史开发阈值文件包含一次"
        "辅助重放，但不进入任何主指标或确认性统计。",
        "",
        "全部 15 个单元的描述统计（含开发 seed42，不用于确认性检验）：",
        "",
        "| 指标 | Focal | CL6 | 平均提升 | 胜/平/负 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in PRIMARY:
        row = all_stats[metric]
        lines.append(
            f"| {metric} | {row['control_mean']:.4f} | "
            f"{row['improved_mean']:.4f} | {row['mean_delta']:+.4f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend([
        "",
        "新种子确认性结果（seed 123/2024，共 10 个单元）：",
        "",
        "| 指标 | Focal | CL6 | 平均提升 | 95% CI | 胜/平/负 | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for metric in PRIMARY:
        row = confirm_stats[metric]
        interval = row["crossed_seed_fold_bootstrap_95_ci"]
        lines.append(
            f"| {metric} | {row['control_mean']:.4f} | "
            f"{row['improved_mean']:.4f} | {row['mean_delta']:+.4f} | "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['holm_corrected_p_primary_metrics']:.4g} |"
        )
    lines.extend([
        "",
        "效率为固定轮次完整外层训练的实测墙钟时间，并按实际处理样本数归一化；"
        "它是次要描述性指标，不参与模型选择。",
        "",
        "| 效率指标 | Focal | CL6 | 配对变化 |",
        "|---|---:|---:|---:|",
    ])
    efficiency = payload["efficiency_15_units"]
    for metric, label in (
        ("training_ms_per_sample", "训练 ms/样本"),
        ("inference_ms_per_sample", "推理 ms/样本"),
    ):
        row = efficiency[metric]
        lines.append(
            f"| {label} | {row['control_mean']:.3f} | "
            f"{row['improved_mean']:.3f} | "
            f"{row['mean_paired_percent_change']:+.1f}% |"
        )
    lines.extend([
        "",
        "CL6 直接约束原模型已有的局部表征，训练期和推理期均不增加模型参数；"
        "原 GCN 分类器结构不变。",
        "",
        "训练曲线机制审计（15 个 CL6 运行均值）：",
        "",
        "| 有效曝光率 | 每个有效曝光的证据对 | 辅助/分类损失比 | 单轮最大比例 |",
        "|---:|---:|---:|---:|",
    ])
    mechanism = payload["mechanism_15_cl6_runs"]["aggregates"]
    lines.append(
        f"| {mechanism['contributing_exposure_rate']['mean']:.2%} | "
        f"{mechanism['evidence_pairs_per_contributing_exposure']['mean']:.2f} | "
        f"{mechanism['effective_aux_to_classification_ratio']['mean']:.2%} | "
        f"{mechanism['max_epoch_aux_to_classification_ratio']['max']:.2%} |"
    )
    lines.extend([
        "",
        "消融采用 seed 42 五折，并固定为完整 CL6 在内层验证集选出的轮数。"
        "下表差值为变体减去完整 CL6；负值表示移除/修改该机制后退化。",
        "",
        "| 变体 | ΔF1 | ΔMCC | ΔPR-AUC |",
        "|---|---:|---:|---:|",
    ])
    labels = {
        "remove_directed_loss": "去掉方向正证据（仅 Focal）",
        "no_curriculum": "去掉 warm-up/ramp",
        "no_aux_cap": "去掉 25% 辅助损失上限",
        "add_mutual": "加入 mutual 项",
        "add_negative": "加入 negative 项",
    }
    for variant, label in labels.items():
        row = payload["ablation_seed42_fivefold"][variant]
        lines.append(
            f"| {label} | {row['f1']['mean_delta_vs_cl6_full']:+.4f} | "
            f"{row['mcc']['mean_delta_vs_cl6_full']:+.4f} | "
            f"{row['pr_auc']['mean_delta_vs_cl6_full']:+.4f} |"
        )
    lines.extend([
        "",
        "机器可读结果：",
        "",
        "- `results/workflow/task2_detection_sci_summary.json`",
        "- `results/workflow/task2_detection_sci_raw.csv`",
        "- `results/workflow/task2_detection_sci_ablation_raw.csv`",
        "- `results/workflow/task2_detection_sci_mechanism_raw.csv`",
        "",
    ])
    atomic_text(REPORT, "\n".join(lines))
    print(json.dumps({
        "summary": str(SUMMARY), "report": str(REPORT),
        "units": len(all_units), "ablation_rows": len(ablation_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
