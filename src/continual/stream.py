from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fold_records(metadata_path: Path, task: int, fold: int) -> tuple[list[dict], list[dict]]:
    """Load one task/fold without modifying the canonical metadata artifact."""
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["task"]) != task or int(row["fold"]) != fold:
                continue
            if row["metadata_status"] != "resolved" or not row["case_id"]:
                raise ValueError(
                    f"unresolved task {task} fold {fold} row: {row['record_id']}"
                )
            if row["split"] == "train":
                train_rows.append(row)
            elif row["split"] == "test":
                test_rows.append(row)
            else:
                raise ValueError(f"unknown split {row['split']!r}")
    if not train_rows or not test_rows:
        raise ValueError(f"no complete metadata for task {task} fold {fold}")
    return train_rows, test_rows


def _balanced_chunks(values: list[str], count: int) -> list[list[str]]:
    if count < 2:
        raise ValueError("increment_count must be at least two")
    if len(values) < count:
        raise ValueError("fewer projects than requested increments")
    quotient, remainder = divmod(len(values), count)
    chunks = []
    start = 0
    for index in range(count):
        size = quotient + int(index < remainder)
        chunks.append(values[start:start + size])
        start += size
    return chunks


def _probe_cases(rows: list[dict], fraction: float, seed: int) -> set[str]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("probe_fraction must be between zero and one")
    project_cases: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        project_cases[row["split_project"]].add(row["case_id"])
    case_count = len({row["case_id"] for row in rows})
    target = max(1, min(case_count - 1, round(case_count * fraction)))

    # Select round-robin by project so a small probe is not dominated by a
    # project with many cases. All ordering is local and deterministic.
    rng = random.Random(seed)
    projects = sorted(project_cases)
    rng.shuffle(projects)
    queues: dict[str, list[str]] = {}
    for project in projects:
        cases = sorted(project_cases[project])
        rng.shuffle(cases)
        queues[project] = cases
    chosen: list[str] = []
    while len(chosen) < target:
        progressed = False
        for project in projects:
            if queues[project] and len(chosen) < target:
                chosen.append(queues[project].pop())
                progressed = True
        if not progressed:
            break
    if len(chosen) != target:
        raise AssertionError(f"selected {len(chosen)} probe cases, expected {target}")
    return set(chosen)


def _statistics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    labels = Counter(row["label"] for row in rows)
    states = Counter(row["state"] for row in rows)
    return {
        "samples": len(rows),
        "cases": len({row["case_id"] for row in rows}),
        "projects": len({row["split_project"] for row in rows}),
        "labels": {key: labels[key] for key in sorted(labels)},
        "states": {key: states[key] for key in sorted(states)},
        "positive_rate": labels.get("1", 0) / len(rows) if rows else 0.0,
    }


def _sample_ids(rows: Iterable[dict]) -> list[str]:
    return [row["sample_id"] for row in rows]


def _audit_stream(
    train_rows: list[dict],
    test_rows: list[dict],
    increments: list[dict],
    probe_fraction: float,
) -> dict:
    train_ids = {row["sample_id"] for row in train_rows}
    test_ids = {row["sample_id"] for row in test_rows}
    train_cases = {row["case_id"] for row in train_rows}
    assigned_ids: list[str] = []
    assigned_cases: list[str] = []
    project_sets: list[set[str]] = []
    probe_ratios: dict[str, float] = {}

    for increment in increments:
        assigned_ids.extend(increment["train_sample_ids"])
        assigned_ids.extend(increment["probe_sample_ids"])
        assigned_cases.extend(increment["train_case_ids"])
        assigned_cases.extend(increment["probe_case_ids"])
        project_sets.append(set(increment["projects"]))
        total_cases = increment["statistics"]["all"]["cases"]
        probe_ratios[increment["id"]] = (
            increment["statistics"]["probe"]["cases"] / total_cases
        )

    checks = {
        "outer_train_and_test_samples_disjoint": train_ids.isdisjoint(test_ids),
        "all_outer_train_samples_assigned_once": (
            set(assigned_ids) == train_ids and len(assigned_ids) == len(set(assigned_ids))
        ),
        "all_outer_train_cases_assigned_once": (
            set(assigned_cases) == train_cases and len(assigned_cases) == len(set(assigned_cases))
        ),
        "projects_disjoint_across_increments": all(
            project_sets[left].isdisjoint(project_sets[right])
            for left in range(len(project_sets))
            for right in range(left + 1, len(project_sets))
        ),
        "train_probe_cases_disjoint": all(
            set(increment["train_case_ids"]).isdisjoint(increment["probe_case_ids"])
            for increment in increments
        ),
        "probe_fraction_within_one_case": all(
            abs(increment["statistics"]["probe"]["cases"]
                - increment["statistics"]["all"]["cases"] * probe_fraction) <= 1.0
            for increment in increments
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"continual stream audit failed: {', '.join(failed)}")
    return {
        "status": "passed",
        "checks": checks,
        "probe_case_ratios": probe_ratios,
        "outer_train_samples": len(train_ids),
        "outer_test_samples": len(test_ids),
    }


def build_continual_stream(
    train_rows: list[dict],
    test_rows: list[dict],
    *,
    task: int,
    fold: int,
    seed: int,
    split_seed: int = 100,
    increment_count: int = 3,
    probe_fraction: float = 0.10,
    metadata_path: Path | None = None,
) -> dict:
    """Build project-incremental partitions while preserving complete cases."""
    if any(int(row["task"]) != task or int(row["fold"]) != fold for row in train_rows + test_rows):
        raise ValueError("rows do not match requested task/fold")
    if any(row["split"] != "train" for row in train_rows):
        raise ValueError("train_rows contains a non-training row")
    if any(row["split"] != "test" for row in test_rows):
        raise ValueError("test_rows contains a non-test row")

    case_projects: dict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        case_projects[row["case_id"]].add(row["split_project"])
    ambiguous = [case_id for case_id, projects in case_projects.items() if len(projects) != 1]
    if ambiguous:
        raise ValueError(f"cases mapped to multiple projects: {ambiguous[:3]}")

    project_order_seed = seed + task * 100 + fold
    projects = sorted({row["split_project"] for row in train_rows})
    random.Random(project_order_seed).shuffle(projects)
    project_chunks = _balanced_chunks(projects, increment_count)
    increments = []
    for index, project_chunk in enumerate(project_chunks, 1):
        project_set = set(project_chunk)
        rows = [row for row in train_rows if row["split_project"] in project_set]
        probes = _probe_cases(
            rows, probe_fraction, seed + task * 1000 + fold * 100 + index
        )
        training = [row for row in rows if row["case_id"] not in probes]
        probe = [row for row in rows if row["case_id"] in probes]
        increments.append({
            "id": f"I{index}",
            "projects": sorted(project_chunk),
            "train_case_ids": sorted({row["case_id"] for row in training}),
            "probe_case_ids": sorted(probes),
            "train_sample_ids": _sample_ids(training),
            "probe_sample_ids": _sample_ids(probe),
            "statistics": {
                "all": _statistics(rows),
                "train": _statistics(training),
                "probe": _statistics(probe),
            },
        })

    artifact = {
        "schema_version": 1,
        "task": task,
        "fold": fold,
        "seed": seed,
        "outer_split_seed": split_seed,
        "strategy": {
            "kind": "project_incremental_single_task",
            "increment_count": increment_count,
            "project_order_seed": project_order_seed,
            "project_order_method": "sorted project names shuffled by local Python Random",
            "probe_fraction": probe_fraction,
            "probe_unit": "complete case_id",
            "time_increment_attempted": True,
            "time_increment_used": False,
            "time_increment_rejection_reason": (
                "Only commit SHA values are available; the dataset contains no reliable "
                "commit timestamps or repository history."
            ),
        },
        "source": {
            "metadata_path": str(metadata_path.resolve()) if metadata_path else None,
            "metadata_sha256": sha256_file(metadata_path) if metadata_path else None,
        },
        "fixed_outer_test": {
            "projects": sorted({row["split_project"] for row in test_rows}),
            "case_ids": sorted({row["case_id"] for row in test_rows}),
            "sample_ids": _sample_ids(test_rows),
            "statistics": _statistics(test_rows),
        },
        "increments": increments,
    }
    artifact["audit"] = _audit_stream(
        train_rows, test_rows, increments, probe_fraction
    )
    return artifact
