#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.original_adapter import OriginalDatasetAdapter
from src.data.token_statistics import document_frequency, save_token_statistics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("CGLSMN_DATA_ROOT", PROJECT_ROOT / "data")),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "artifacts" / "token_statistics")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"loading immutable task {args.task} cache", flush=True)
    adapter = OriginalDatasetAdapter(args.data_root, args.task, split_seed=100)
    statistics = document_frequency(adapter.graphs, adapter.vocab)
    output = args.output_dir / f"task{args.task}.json"
    save_token_statistics(output, statistics)
    summary = {
        "task": args.task,
        "documents": statistics["documents"],
        "vocab_size": statistics["vocab_size"],
        "nonzero_tokens": sum(value > 0 for value in statistics["document_frequency"]),
        "output": str(output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
