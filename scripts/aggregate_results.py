#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc", "mcc",
    "roc_auc_legacy", "inference_seconds", "training_seconds_total")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_union_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if keys:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def aggregate_baseline():
    rows = []
    pattern = "task*_fold*_seed*/metrics.csv"
    for path in sorted((ROOT / "results" / "baseline").glob(pattern)):
        run_rows = read_rows(path)
        if run_rows:
            best = max(run_rows, key=lambda row: float(row["f1"]))
            rows.append({"run_name": path.parent.name, **best})
    raw_path = ROOT / "results" / "baseline" / "raw.csv"
    write_union_csv(raw_path, rows)
    print(f"aggregated {len(rows)} completed baseline runs into {raw_path}")


def objective_name(row: dict[str, str]) -> str:
    method = row.get("method", "")
    try:
        local_weight = float(row.get("lambda_local", "") or "nan")
    except ValueError:
        local_weight = math.nan
    if method == "CL5" and local_weight == 0:
        return "Focal-only-control"
    if method == "CL5":
        return "CL5-graph-pair-local"
    if method == "CL6":
        return "CL6-directed-method-class"
    return method


def aggregate_contrastive():
    result_root = ROOT / "results" / "contrastive"
    raw_rows = []
    curve_rows = []
    for run_dir in sorted((result_root / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        curves = read_rows(run_dir / "training_curves.csv")
        for curve in curves:
            curve_rows.append({"run_name": run_dir.name, **curve})
        test_rows = read_rows(run_dir / "test_metrics.csv")
        if not test_rows:
            continue
        training_seconds = sum(
            float(row.get("train_seconds", 0) or 0) for row in curves)
        for row in test_rows:
            protocol = row.get("training_protocol", "")
            raw_rows.append({
                "run_name": run_dir.name,
                "objective": objective_name(row),
                "reportable": int(protocol in {
                    "full_train_fixed_epoch", "legacy_test_selected"}),
                "training_seconds_total": training_seconds,
                **row,
            })

    raw_path = result_root / "raw.csv"
    curves_path = result_root / "training_curves.csv"
    write_union_csv(raw_path, raw_rows)
    write_union_csv(curves_path, curve_rows)

    groups = defaultdict(list)
    for row in raw_rows:
        if not int(row["reportable"]):
            continue
        key = (
            row.get("task", ""), row.get("objective", ""),
            row.get("training_protocol", ""), row.get("lambda_local", ""),
            row.get("directed_lambda_positive", ""),
            row.get("directed_lambda_mutual", ""),
            row.get("directed_lambda_negative", ""),
        )
        groups[key].append(row)
    summary_rows = []
    for key, rows in sorted(groups.items()):
        summary = {
            "task": key[0],
            "objective": key[1],
            "training_protocol": key[2],
            "lambda_local": key[3],
            "directed_lambda_positive": key[4],
            "directed_lambda_mutual": key[5],
            "directed_lambda_negative": key[6],
            "runs": len(rows),
        }
        for metric in METRICS:
            values = [
                float(row[metric]) for row in rows
                if row.get(metric, "") not in ("", None)]
            if values:
                summary[f"{metric}_mean"] = statistics.fmean(values)
                summary[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0)
        summary_rows.append(summary)
    summary_path = result_root / "summary.csv"
    write_union_csv(summary_path, summary_rows)
    print(
        f"aggregated {len(raw_rows)} contrastive tests and {len(curve_rows)} "
        f"epoch rows into {result_root}")


def main():
    aggregate_baseline()
    aggregate_contrastive()


if __name__ == "__main__":
    main()
