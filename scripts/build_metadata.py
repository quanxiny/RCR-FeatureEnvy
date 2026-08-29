#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.metadata_builder import build_metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("CGLSMN_DATA_ROOT", PROJECT_ROOT / "data")),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "artifacts" / "metadata",
    )
    parser.add_argument(
        "--report", type=Path,
        default=PROJECT_ROOT / "reports" / "00_sample_metadata.md",
    )
    parser.add_argument("--split-seed", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_metadata(args.data_root, args.output_dir, args.report, args.split_seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
