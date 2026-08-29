#!/usr/bin/env python3
"""Run publication-scale Task-2 continual-learning experiments."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
WORKFLOW_ROOT = PROJECT_ROOT / "results" / "workflow"
STATUS = WORKFLOW_ROOT / "task2_continual_sci_status.json"
MANIFEST = WORKFLOW_ROOT / "task2_continual_sci_manifest.json"
LOCK = WORKFLOW_ROOT / "task2_continual_sci.lock"
LOG_ROOT = PROJECT_ROOT / "logs" / "workflow"
SEEDS = (42, 123, 2024)
FOLDS = (1, 2, 3, 4, 5)
# Alternation also balances the two fixed GPU workers: each receives one
# cheaper and one more expensive method per stream unit.
PRIMARY_METHODS = ("naive", "ewc", "replay_relation", "replay")
RELATION_ARGS = (
    "--logit-distillation", "0.5",
    "--embedding-distillation", "0.1",
    "--attention-distillation", "0.1",
)
ABLATIONS = {
    "focal_replay": {
        "method": "replay", "base_objective": "focal", "extra": (),
    },
    "logit_only": {
        "method": "replay_kd", "base_objective": "cl6",
        "extra": ("--logit-distillation", "0.5"),
    },
    "no_attention": {
        "method": "replay_relation", "base_objective": "cl6",
        "extra": (
            "--logit-distillation", "0.5",
            "--embedding-distillation", "0.1",
            "--attention-distillation", "0.0",
        ),
    },
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def acquire_lock() -> None:
    if LOCK.exists():
        try:
            pid = int(json.loads(LOCK.read_text(encoding="utf-8"))["pid"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pid = -1
        if pid_alive(pid):
            raise RuntimeError(f"Task-2 continual workflow already runs as PID {pid}")
    atomic_json(LOCK, {"pid": os.getpid()})


def execute(argv: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + " ".join(argv) + "\n")
        handle.flush()
        completed = subprocess.run(
            argv, cwd=PROJECT_ROOT, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}); see {log_path}"
        )


def stream_path(seed: int, fold: int) -> Path:
    return (
        PROJECT_ROOT / "artifacts" / "continual"
        / f"task2_fold{fold}_seed{seed}.json"
    )


def validate_stream_artifact(
    path: Path, artifact: dict, seed: int, fold: int
) -> None:
    """Reject incomplete or internally inconsistent continual streams early."""
    if (
        int(artifact.get("task", -1)),
        int(artifact.get("fold", -1)),
        int(artifact.get("seed", -1)),
    ) != (2, fold, seed):
        raise RuntimeError(f"stream identity does not match filename: {path}")
    audit = artifact.get("audit", {})
    checks = audit.get("checks", {})
    if audit.get("status") != "passed" or not checks or not all(checks.values()):
        raise RuntimeError(f"stream audit is not passed: {path}")
    increments = artifact.get("increments", [])
    if [item.get("id") for item in increments] != ["I1", "I2", "I3"]:
        raise RuntimeError(f"stream must contain ordered I1/I2/I3: {path}")
    train_ids = [
        sample_id
        for item in increments
        for sample_id in item.get("train_sample_ids", [])
    ]
    probe_ids = [
        sample_id
        for item in increments
        for sample_id in item.get("probe_sample_ids", [])
    ]
    outer_ids = artifact.get("fixed_outer_test", {}).get("sample_ids", [])
    if not train_ids or not probe_ids or not outer_ids:
        raise RuntimeError(f"stream contains an empty required split: {path}")
    if len(train_ids) != len(set(train_ids)):
        raise RuntimeError(f"duplicate training samples in stream: {path}")
    if len(probe_ids) != len(set(probe_ids)):
        raise RuntimeError(f"duplicate probe samples in stream: {path}")
    if len(outer_ids) != len(set(outer_ids)):
        raise RuntimeError(f"duplicate fixed outer-test samples in stream: {path}")
    if set(train_ids) & set(probe_ids):
        raise RuntimeError(f"training/probe sample leakage in stream: {path}")
    if (set(train_ids) | set(probe_ids)) & set(outer_ids):
        raise RuntimeError(f"outer-test sample leakage in stream: {path}")
    expected_outer = int(audit.get("outer_test_samples", -1))
    if expected_outer != len(outer_ids):
        raise RuntimeError(f"outer-test audit count mismatch: {path}")


def ensure_streams(status: dict) -> None:
    log_path = LOG_ROOT / "task2_continual_sci_streams.log"
    for seed in SEEDS:
        for fold in FOLDS:
            path = stream_path(seed, fold)
            if not path.exists():
                status["active"] = {
                    "stage": "stream", "seed": seed, "fold": fold
                }
                atomic_json(STATUS, status)
                execute([
                    str(PYTHON), "-u", "scripts/build_continual_stream.py",
                    "--task", "2", "--fold", str(fold),
                    "--seed", str(seed), "--output", str(path),
                ], log_path)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            validate_stream_artifact(path, artifact, seed, fold)


def run_name(
    seed: int,
    fold: int,
    method: str,
    base_objective: str,
    extra: tuple[str, ...] = (),
) -> str:
    objective = "_cl6" if base_objective == "cl6" else ""
    suffix = ""
    if method in {"replay", "replay_kd", "replay_relation"}:
        suffix = "_r10"
    values = dict(zip(extra[::2], extra[1::2]))
    if method == "ewc":
        suffix = "_l10"
    elif method == "replay_kd":
        suffix += f"_kd{float(values.get('--logit-distillation', 0.5)):g}"
    elif method == "replay_relation":
        suffix += (
            f"_kd{float(values.get('--logit-distillation', 0.5)):g}"
            f"_z{float(values.get('--embedding-distillation', 0.1)):g}"
            f"_a{float(values.get('--attention-distillation', 0.1)):g}"
        )
    return (
        f"task2_fold{fold}_seed{seed}_{method}{objective}{suffix}_e8"
    )


def complete(run: str) -> bool:
    return (
        PROJECT_ROOT / "results" / "continual" / "runs"
        / run / "complete.json"
    ).exists()


def command(
    seed: int,
    fold: int,
    method: str,
    base_objective: str,
    device: str,
    extra: tuple[str, ...] = (),
) -> list[str]:
    return [
        str(PYTHON), "-u", "scripts/run_continual.py",
        "--method", method,
        "--base-objective", base_objective,
        "--task", "2", "--fold", str(fold), "--seed", str(seed),
        "--stream", str(stream_path(seed, fold)),
        "--stage-epochs", "8", "--buffer-ratio", "0.10",
        "--batch-size", "64", "--backward-chunk", "64",
        "--ewc-lambda", "10", "--fisher-samples", "512",
        "--device", device, *extra,
    ]


def primary_jobs() -> list[dict]:
    jobs = []
    # Cycle methods inside each stream so partial results remain paired.
    for seed in SEEDS:
        for fold in FOLDS:
            for method in PRIMARY_METHODS:
                extra = RELATION_ARGS if method == "replay_relation" else ()
                jobs.append({
                    "kind": "primary", "seed": seed, "fold": fold,
                    "method": method, "base_objective": "cl6",
                    "extra": extra,
                })
    return jobs


def secondary_jobs() -> list[dict]:
    jobs = []
    # Joint training is an upper bound, not a continual method, so one full
    # five-fold seed is sufficient and avoids misrepresenting it as a stream.
    for fold in FOLDS:
        jobs.append({
            "kind": "upper_bound", "seed": 42, "fold": fold,
            "method": "joint", "base_objective": "cl6", "extra": (),
        })
    for fold in FOLDS:
        for label, specification in ABLATIONS.items():
            jobs.append({
                "kind": "ablation", "variant": label,
                "seed": 42, "fold": fold, **specification,
            })
    return jobs


def run_worker(
    device: str,
    jobs: list[dict],
    manifest: dict,
    status: dict,
    mutex: threading.Lock,
) -> None:
    log_path = LOG_ROOT / f"task2_continual_sci_{device[-1]}.log"
    for job in jobs:
        name = run_name(
            job["seed"], job["fold"], job["method"],
            job["base_objective"], job["extra"],
        )
        with mutex:
            status["active"][device] = {
                key: value for key, value in job.items() if key != "extra"
            } | {"run_name": name}
            atomic_json(STATUS, status)
        if not complete(name):
            execute(command(
                job["seed"], job["fold"], job["method"],
                job["base_objective"], device, job["extra"],
            ), log_path)
        record = {
            key: value for key, value in job.items() if key != "extra"
        } | {
            "extra": list(job["extra"]),
            "run_name": name,
            "result_dir": str(
                Path("results") / "continual" / "runs" / name
            ),
        }
        with mutex:
            manifest["runs"] = [
                row for row in manifest["runs"] if row["run_name"] != name
            ]
            manifest["runs"].append(record)
            manifest["runs"].sort(key=lambda row: (
                row["kind"], int(row["seed"]), int(row["fold"]),
                row["method"], row.get("variant", ""),
            ))
            atomic_json(MANIFEST, manifest)
    with mutex:
        status["active"].pop(device, None)
        atomic_json(STATUS, status)


def run_phase(
    jobs: list[dict],
    devices: tuple[str, str] | list[str],
    manifest: dict,
    status: dict,
    mutex: threading.Lock,
) -> None:
    pending = []
    recorded = {row["run_name"] for row in manifest["runs"]}
    for job in jobs:
        name = run_name(
            job["seed"], job["fold"], job["method"],
            job["base_objective"], job["extra"],
        )
        if name not in recorded or not complete(name):
            pending.append(job)
    partitions = [pending[::2], pending[1::2]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_worker, device, partition, manifest, status, mutex
            )
            for device, partition in zip(devices, partitions)
            if partition
        ]
        for future in as_completed(futures):
            future.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs=2, default=("cuda:0", "cuda:1"))
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = {
        "task": 2, "folds": list(FOLDS), "seeds": list(SEEDS),
        "primary_methods": list(PRIMARY_METHODS),
        "primary_runs": len(FOLDS) * len(SEEDS) * len(PRIMARY_METHODS),
        "joint_upper_bound_runs": len(FOLDS),
        "ablation_variants": list(ABLATIONS),
        "ablation_runs": len(FOLDS) * len(ABLATIONS),
        "devices": list(args.devices),
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    acquire_lock()
    mutex = threading.Lock()
    status = {
        "state": "running", "pid": os.getpid(),
        "stage": "continual stream construction and audit",
        "plan": plan, "active": {},
    }
    manifest = {
        "protocol": {
            "task": 2,
            "kind": "project-incremental single-task learning",
            "increments": 3,
            "model_objective": "Focal + CL6 unless explicitly ablated",
            "stage_epochs": 8,
            "replay_ratio": 0.10,
            "replay_unit": "complete refactoring case",
            "seeds": list(SEEDS), "folds": list(FOLDS),
            "outer_test_policy": "evaluation only after each fixed-budget stage",
            "development_unit": "fold 1 seed 42",
        },
        "runs": [],
    }
    if MANIFEST.exists():
        manifest["runs"] = json.loads(
            MANIFEST.read_text(encoding="utf-8")
        ).get("runs", [])
    atomic_json(STATUS, status)
    atomic_json(MANIFEST, manifest)
    try:
        ensure_streams(status)
        status.update({"stage": "four-method three-seed five-fold comparison", "active": {}})
        atomic_json(STATUS, status)
        run_phase(primary_jobs(), args.devices, manifest, status, mutex)
        status.update({"stage": "joint upper bound and continual ablations", "active": {}})
        atomic_json(STATUS, status)
        run_phase(secondary_jobs(), args.devices, manifest, status, mutex)
        status.update({"stage": "continual statistics", "active": {}})
        atomic_json(STATUS, status)
        execute(
            [str(PYTHON), "-u", "scripts/aggregate_continual.py"],
            LOG_ROOT / "task2_continual_sci_summary.log",
        )
        execute(
            [str(PYTHON), "-u", "scripts/summarize_task2_continual_sci.py"],
            LOG_ROOT / "task2_continual_sci_summary.log",
        )
        status.update({
            "state": "complete", "stage": "Task 2 continual experiments complete",
            "summary": "results/workflow/task2_continual_sci_summary.json",
            "report": "reports/16_task2_continual_sci.md",
        })
    except Exception as exc:
        status.update({
            "state": "failed", "active": {},
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    finally:
        atomic_json(STATUS, status)
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
