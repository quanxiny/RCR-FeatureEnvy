#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.continual.stream import build_continual_stream, load_fold_records
from src.data.fixed_splits import split_labels
from src.data.original_adapter import dataset_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=(1, 2), default=1)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=100)
    parser.add_argument("--increments", type=int, default=3)
    parser.add_argument("--probe-fraction", type=float, default=0.10)
    parser.add_argument(
        "--metadata", type=Path,
        default=PROJECT_ROOT / "artifacts" / "metadata" / "all_samples.csv",
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("CGLSMN_DATA_ROOT", PROJECT_ROOT / "data")),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or (
        PROJECT_ROOT / "artifacts" / "continual"
        / f"task{args.task}_fold{args.fold}_seed{args.seed}.json"
    )
    train_rows, test_rows = load_fold_records(args.metadata, args.task, args.fold)

    # An independent adapter comparison proves that the stored outer test list
    # is exactly the published fold rather than merely a disjoint substitute.
    files = dataset_files(args.data_root, args.task)
    label_lines = files.labels.read_text(encoding="utf-8").splitlines()
    adapter_train, adapter_test = split_labels(
        label_lines, args.fold, split_seed=args.split_seed
    )
    if [row["label_line"] for row in train_rows] != adapter_train:
        raise AssertionError("metadata outer-training order/content differs from adapter")
    if [row["label_line"] for row in test_rows] != adapter_test:
        raise AssertionError("metadata fixed outer test differs from published adapter fold")

    artifact = build_continual_stream(
        train_rows,
        test_rows,
        task=args.task,
        fold=args.fold,
        seed=args.seed,
        split_seed=args.split_seed,
        increment_count=args.increments,
        probe_fraction=args.probe_fraction,
        metadata_path=args.metadata,
    )
    artifact["audit"]["checks"]["fixed_outer_test_matches_published_adapter"] = True
    artifact["audit"]["checks"]["outer_train_matches_published_adapter"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    summary = {
        "output": str(output),
        "audit": artifact["audit"],
        "increments": {
            item["id"]: item["statistics"] for item in artifact["increments"]
        },
        "fixed_outer_test": artifact["fixed_outer_test"]["statistics"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
