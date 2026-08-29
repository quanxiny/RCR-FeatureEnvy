#!/usr/bin/env python3
"""Complete the Task-2 contrastive/continual matrix for the EI paper.

The workflow is restart-safe: completed run directories are reused and every
training command retains the checkpoint/resume behaviour of run_continual.py.
"""
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
STATUS = WORKFLOW_ROOT / "task2_ei_extension_status.json"
MANIFEST = WORKFLOW_ROOT / "task2_ei_extension_manifest.json"
LOCK = WORKFLOW_ROOT / "task2_ei_extension.lock"
LOG_ROOT = PROJECT_ROOT / "logs" / "workflow"
SEEDS = (42, 123, 2024)
FOLDS = (1, 2, 3, 4, 5)
RELATION_ARGS = (
    "--logit-distillation", "0.5",
    "--embedding-distillation", "0.1",
    "--attention-distillation", "0.1",
)


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
            raise RuntimeError(f"Task-2 EI extension already runs as PID {pid}")
    atomic_json(LOCK, {"pid": os.getpid()})


def run_name(job: dict) -> str:
    objective = ""
    if job["base_objective"] == "cl6":
        objective = "_cl6"
    elif job["base_objective"] == "supcon":
        objective = "_supcon_t0.07_l0.2"
    ratio = int(round(float(job["buffer_ratio"]) * 100))
    suffix = f"_r{ratio:02d}"
    if job["method"] == "replay_relation":
        suffix += "_kd0.5_z0.1_a0.1"
    return (
        f"task2_fold{job['fold']}_seed{job['seed']}_"
        f"{job['method']}{objective}{suffix}_e8"
    )


def complete(name: str) -> bool:
    return (
        PROJECT_ROOT / "results" / "continual" / "runs"
        / name / "complete.json"
    ).exists()


def command(job: dict, device: str) -> list[str]:
    argv = [
        str(PYTHON), "-u", "scripts/run_continual.py",
        "--method", job["method"],
        "--base-objective", job["base_objective"],
        "--task", "2",
        "--fold", str(job["fold"]),
        "--seed", str(job["seed"]),
        "--stream", str(
            PROJECT_ROOT / "artifacts" / "continual"
            / f"task2_fold{job['fold']}_seed{job['seed']}.json"
        ),
        "--stage-epochs", "8",
        "--buffer-ratio", str(job["buffer_ratio"]),
        "--batch-size", "64",
        "--backward-chunk", "64",
        "--device", device,
    ]
    if job["base_objective"] == "supcon":
        argv.extend([
            "--supcon-temperature", "0.07",
            "--supcon-lambda", "0.20",
        ])
    if job["method"] == "replay_relation":
        argv.extend(RELATION_ARGS)
    return argv


def jobs() -> list[dict]:
    output = []
    # Complete the objective comparison at the fixed 10% replay budget.
    for seed in SEEDS:
        for fold in FOLDS:
            output.extend([
                {
                    "kind": "objective_ablation",
                    "variant": "focal_replay",
                    "method": "replay",
                    "base_objective": "focal",
                    "buffer_ratio": 0.10,
                    "seed": seed,
                    "fold": fold,
                },
                {
                    "kind": "objective_ablation",
                    "variant": "supcon_replay",
                    "method": "replay",
                    "base_objective": "supcon",
                    "buffer_ratio": 0.10,
                    "seed": seed,
                    "fold": fold,
                },
            ])
    # The 10% relation-aware runs already exist for all seeds.  Budget
    # sensitivity is a five-fold development analysis, not a new confirmatory
    # family, so seed 42 is sufficient for 5% and 20%.
    for fold in FOLDS:
        for ratio in (0.05, 0.20):
            output.append({
                "kind": "buffer_sensitivity",
                "variant": f"relation_r{int(ratio * 100):02d}",
                "method": "replay_relation",
                "base_objective": "cl6",
                "buffer_ratio": ratio,
                "seed": 42,
                "fold": fold,
            })
    for job in output:
        job["run_name"] = run_name(job)
    return output


def execute(job: dict, device: str, status: dict, mutex: threading.Lock) -> None:
    name = job["run_name"]
    if complete(name):
        with mutex:
            status["runs"][name] = "reused"
            atomic_json(STATUS, status)
        return
    log_path = LOG_ROOT / f"task2_ei_extension_{device[-1]}.log"
    argv = command(job, device)
    with mutex:
        status["active"][device] = name
        status["runs"][name] = "running"
        atomic_json(STATUS, status)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + " ".join(argv) + "\n")
        handle.flush()
        completed = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    with mutex:
        status["active"].pop(device, None)
        status["runs"][name] = (
            "complete" if completed.returncode == 0 and complete(name) else "failed"
        )
        atomic_json(STATUS, status)
    if completed.returncode or not complete(name):
        raise RuntimeError(f"{name} failed; see {log_path}")


def execute_queue(
    device: str,
    assigned: list[dict],
    status: dict,
    mutex: threading.Lock,
) -> None:
    for job in assigned:
        execute(job, device, status, mutex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    args = parser.parse_args()
    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    if not devices:
        raise ValueError("at least one CUDA device is required")
    acquire_lock()
    try:
        all_jobs = jobs()
        atomic_json(MANIFEST, {
            "protocol": {
                "task": 2,
                "folds": list(FOLDS),
                "seeds": list(SEEDS),
                "stage_epochs": 8,
                "primary_buffer_ratio": 0.10,
                "supcon_temperature": 0.07,
                "supcon_lambda": 0.20,
                "budget_sensitivity_seed": 42,
                "devices": list(devices),
            },
            "runs": all_jobs,
        })
        status = {
            "pid": os.getpid(),
            "state": "running",
            "active": {},
            "runs": {},
        }
        mutex = threading.Lock()
        atomic_json(STATUS, status)
        # Round-robin submission preserves a mixture of the slower SupCon and
        # cheaper reused/Focal jobs on both GPUs.
        assignments = {
            device: all_jobs[index::len(devices)]
            for index, device in enumerate(devices)
        }
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = {
                executor.submit(execute_queue, device, assigned, status, mutex): device
                for device, assigned in assignments.items()
            }
            for future in as_completed(futures):
                future.result()
        aggregate_log = LOG_ROOT / "task2_ei_extension_aggregate.log"
        with aggregate_log.open("a", encoding="utf-8") as handle:
            for script in (
                "scripts/aggregate_continual.py",
                "scripts/summarize_task2_ei_extension.py",
            ):
                completed = subprocess.run(
                    [str(PYTHON), "-u", script],
                    cwd=PROJECT_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if completed.returncode:
                    raise RuntimeError(f"summary failed; see {aggregate_log}")
        status["state"] = "complete"
        atomic_json(STATUS, status)
    except Exception as error:
        if "status" in locals():
            status["state"] = "failed"
            status["error"] = repr(error)
            atomic_json(STATUS, status)
        raise
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
