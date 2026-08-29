from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .fixed_splits import project_name, published_project_folds


ITEM_STATE = re.compile(r"_item_(pos|neg)_(\d+)$")


@dataclass(frozen=True)
class GroundTruthCase:
    commit_sha: str
    project: str
    moved_method: str
    source_class: str
    target_class: str

    @property
    def source_name(self) -> str:
        return Path(self.source_class).stem

    @property
    def target_name(self) -> str:
        return Path(self.target_class).stem

    @property
    def lookup_key(self) -> tuple[str, str, str, str]:
        return self.project, self.moved_method, self.source_name, self.target_name

    @property
    def case_id(self) -> str:
        fields = (
            self.project,
            self.commit_sha,
            self.source_class,
            self.target_class,
            self.moved_method,
        )
        digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()[:24]
        return f"case_{digest}"


def state_and_augmentation(item_path: str) -> tuple[str, int]:
    match = ITEM_STATE.search(item_path)
    if match is None:
        return "unknown", -1
    # The published Java generators read sourceCode/<project>/a for positive
    # examples and sourceCode/<project>/b for negative examples. The ground
    # truth defines a as pre-refactoring and b as post-refactoring.
    return ("pre" if match.group(1) == "pos" else "post"), int(match.group(2))


def load_ground_truth(path: Path) -> dict[tuple[str, str, str, str], GroundTruthCase]:
    cases: dict[tuple[str, str, str, str], GroundTruthCase] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            case = GroundTruthCase(
                commit_sha=row["Commit_SHA"],
                project=row["Project_name"],
                moved_method=row["Method_name"],
                source_class=row["Src_Class"],
                target_class=row["Tag_Class"],
            )
            if case.lookup_key in cases:
                raise ValueError(f"ambiguous ground-truth key: {case.lookup_key}")
            cases[case.lookup_key] = case
    return cases


def _canonical_sample_id(task: int, label_line: str) -> str:
    digest = hashlib.sha256(f"{task}\x1f{label_line}".encode("utf-8")).hexdigest()[:24]
    return f"task{task}_{digest}"


def _dataset_name(task: int) -> str:
    if task == 1:
        return "Dataset_Class_Class"
    if task == 2:
        return "Dataset_Method_Class"
    raise ValueError(f"unsupported task: {task}")


def build_canonical_records(data_root: Path, task: int,
                            ground_truth: dict) -> list[dict]:
    dataset_dir = data_root / _dataset_name(task)
    label_lines = (dataset_dir / "new_labels.txt").read_text(encoding="utf-8").splitlines()
    occurrences: Counter[str] = Counter()
    first_sample_id: dict[str, str] = {}
    info_cache: dict[str, dict] = {}
    records = []

    for label_line in label_lines:
        fields = label_line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid task {task} label line: {label_line!r}")
        item_path, graph1_name, graph2_name, label_text = fields
        label = int(label_text)
        canonical_id = _canonical_sample_id(task, label_line)
        occurrences[canonical_id] += 1
        occurrence = occurrences[canonical_id]
        sample_id = f"{canonical_id}_{occurrence:02d}"
        duplicate_of = first_sample_id.setdefault(canonical_id, sample_id)

        item_dir = dataset_dir / item_path
        info_path = item_dir / "dataItemInfo.json"
        reasons = []
        if item_path not in info_cache:
            if info_path.exists():
                info_cache[item_path] = json.loads(info_path.read_text(encoding="utf-8"))
            else:
                info_cache[item_path] = {}
        info = info_cache[item_path]
        if not info:
            reasons.append("missing_data_item_info")

        method = info.get("methodName") or info.get("needRefactmethodName") or ""
        lookup_key = (
            info.get("projectName", ""),
            method,
            info.get("srcClassName", ""),
            info.get("tagClassName", ""),
        )
        case = ground_truth.get(lookup_key)
        if case is None:
            reasons.append("ground_truth_case_not_found")

        state, augmentation_index = state_and_augmentation(item_path)
        if state == "unknown":
            reasons.append("state_not_recoverable")
        expected_label = 1 if state == "pre" else 0 if state == "post" else None
        if expected_label is not None and label != expected_label:
            reasons.append("label_state_mismatch")
        if "label" in info and int(info["label"]) != label:
            reasons.append("label_info_mismatch")

        graph1_path = item_dir / f"{graph1_name}.java"
        graph2_path = item_dir / f"{graph2_name}.java"
        if not graph1_path.exists():
            reasons.append("graph1_source_not_found")
        if not graph2_path.exists():
            reasons.append("graph2_source_not_found")

        records.append({
            "sample_id": sample_id,
            "canonical_sample_id": canonical_id,
            "case_id": case.case_id if case else "",
            "task": task,
            "project": case.project if case else info.get("projectName", ""),
            "split_project": project_name(label_line),
            "commit_sha": case.commit_sha if case else "",
            "source_class": case.source_class if case else info.get("srcClassName", ""),
            "target_class": case.target_class if case else info.get("tagClassName", ""),
            "moved_method": case.moved_method if case else method,
            "state": state,
            "label": label,
            "graph1_path": str(graph1_path.resolve()),
            "graph2_path": str(graph2_path.resolve()),
            "item_path": item_path,
            "label_line": label_line,
            "augmentation_index": augmentation_index,
            "is_duplicate": int(occurrence > 1),
            "duplicate_of": duplicate_of if occurrence > 1 else "",
            "metadata_status": "resolved" if not reasons else "unresolved",
            "unresolved_reason": ";".join(reasons),
        })
    return records


def expand_fixed_splits(records: Iterable[dict], label_lines: list[str],
                        split_seed: int = 100) -> list[dict]:
    folds = published_project_folds(label_lines, split_seed=split_seed)
    expanded = []
    for record in records:
        for fold, projects in folds.items():
            split = "test" if record["split_project"] in projects["test"] else "train"
            expanded.append({
                "record_id": f'{record["sample_id"]}_fold{fold}',
                **record,
                "fold": fold,
                "split": split,
            })
    return expanded


def case_statistics(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["case_id"]:
            grouped[record["case_id"]].append(record)
    rows = []
    for case_id, samples in sorted(grouped.items()):
        first = samples[0]
        tasks = {sample["task"] for sample in samples}
        task1 = [sample for sample in samples if sample["task"] == 1]
        task2 = [sample for sample in samples if sample["task"] == 2]
        states = {sample["state"] for sample in samples}
        rows.append({
            "case_id": case_id,
            "project": first["project"],
            "commit_sha": first["commit_sha"],
            "source_class": first["source_class"],
            "target_class": first["target_class"],
            "moved_method": first["moved_method"],
            "task1_samples": len(task1),
            "task2_samples": len(task2),
            "total_samples": len(samples),
            "unique_samples": len({sample["canonical_sample_id"] for sample in samples}),
            "task1_pre": sum(sample["state"] == "pre" for sample in task1),
            "task1_post": sum(sample["state"] == "post" for sample in task1),
            "task2_pre": sum(sample["state"] == "pre" for sample in task2),
            "task2_post": sum(sample["state"] == "post" for sample in task2),
            "has_pre_post": int({"pre", "post"}.issubset(states)),
            "aligned_tasks": int(tasks == {1, 2}),
            "duplicate_samples": sum(sample["is_duplicate"] for sample in samples),
        })
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def build_metadata(data_root: Path, output_dir: Path, report_path: Path,
                   split_seed: int = 100) -> dict:
    ground_truth = load_ground_truth(data_root / "groundTruth.csv")
    canonical = []
    expanded = []
    task_counts = {}
    for task in (1, 2):
        records = build_canonical_records(data_root, task, ground_truth)
        label_path = data_root / _dataset_name(task) / "new_labels.txt"
        label_lines = label_path.read_text(encoding="utf-8").splitlines()
        canonical.extend(records)
        expanded.extend(expand_fixed_splits(records, label_lines, split_seed))
        task_counts[task] = len(records)

    stats = case_statistics(canonical)
    unresolved = [record for record in canonical if record["metadata_status"] != "resolved"]
    all_fields = [
        "record_id", "sample_id", "canonical_sample_id", "case_id", "task",
        "project", "split_project", "commit_sha", "source_class", "target_class",
        "moved_method", "state", "label", "fold", "split", "graph1_path",
        "graph2_path", "item_path", "label_line", "augmentation_index",
        "is_duplicate", "duplicate_of", "metadata_status", "unresolved_reason",
    ]
    unresolved_fields = [field for field in all_fields if field not in {"record_id", "fold", "split"}]
    _write_csv(output_dir / "all_samples.csv", expanded, all_fields)
    _write_jsonl(output_dir / "all_samples.jsonl", expanded)
    _write_csv(output_dir / "case_statistics.csv", stats)
    _write_csv(output_dir / "unresolved_samples.csv", unresolved, unresolved_fields)

    by_case_task_fold: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    by_case_fold_task: dict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    for row in expanded:
        if row["case_id"]:
            by_case_task_fold[(row["case_id"], row["task"], row["fold"])].add(row["split"])
            by_case_fold_task[(row["case_id"], row["fold"])][row["task"]] = row["split"]
    within_task_leakage = sum(
        len(splits) > 1 for splits in by_case_task_fold.values())
    cross_task_disagreement = sum(
        len(task_splits) == 2 and len(set(task_splits.values())) > 1
        for task_splits in by_case_fold_task.values())
    summary = {
        "ground_truth_cases": len(ground_truth),
        "task1_samples": task_counts[1],
        "task2_samples": task_counts[2],
        "expanded_fold_records": len(expanded),
        "resolved_samples": len(canonical) - len(unresolved),
        "unresolved_samples": len(unresolved),
        "exact_duplicate_samples": sum(row["is_duplicate"] for row in canonical),
        "cases_in_metadata": len(stats),
        "cases_with_pre_and_post": sum(row["has_pre_post"] for row in stats),
        "cases_aligned_across_tasks": sum(row["aligned_tasks"] for row in stats),
        "within_task_case_fold_split_leakage": within_task_leakage,
        "cross_task_fold_split_disagreement": cross_task_disagreement,
        "split_seed": split_seed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Unified sample metadata\n\n"
        "Metadata is derived from immutable `groundTruth.csv`, each sample's "
        "`dataItemInfo.json`, and the exact published project-fold algorithm. "
        "The Java generators map `item_pos` to pre-refactoring source tree `a` "
        "and `item_neg` to post-refactoring source tree `b`.\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in summary.items())
        + "\n\nThe published protocol has no validation split. These files preserve its "
        "outer train/test folds; improved-method tuning will derive a deterministic "
        "inner validation subset from the outer training projects only.\n",
        encoding="utf-8",
    )
    return summary
