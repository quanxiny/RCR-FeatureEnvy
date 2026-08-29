#!/usr/bin/env python3
"""Run the confirmatory Task-2 detector and fixed-budget CL6 ablations."""
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
STATUS = WORKFLOW_ROOT / "task2_detection_sci_status.json"
MANIFEST = WORKFLOW_ROOT / "task2_detection_sci_manifest.json"
LOCK = WORKFLOW_ROOT / "task2_detection_sci.lock"
LOG_ROOT = PROJECT_ROOT / "logs" / "workflow"
LEGACY_MANIFEST = WORKFLOW_ROOT / "fivefold_detection_manifest.json"
SEEDS = (42, 123, 2024)
FOLDS = (1, 2, 3, 4, 5)

CL6_ARGUMENTS = (
    "--method", "CL6",
    "--directed-lambda-positive", "0.01",
    "--directed-lambda-mutual", "0.0",
    "--directed-lambda-negative", "0.0",
    "--directed-warmup-epochs", "3",
    "--directed-ramp-epochs", "2",
    "--directed-max-aux-ratio", "0.25",
)
FOCAL_ARGUMENTS = (
    "--method", "CL5",
    "--lambda-local", "0.0",
    "--local-warmup-epochs", "1",
)
ABLATIONS = {
    # Train the Focal-only objective for the exact epoch budget selected by
    # the corresponding full CL6 run, so loss removal is not confounded by
    # an independently selected stopping epoch.
    "remove_directed_loss": (),
    "no_curriculum": (
        "--directed-lambda-positive", "0.01",
        "--directed-lambda-mutual", "0.0",
        "--directed-lambda-negative", "0.0",
        "--directed-warmup-epochs", "0",
        "--directed-ramp-epochs", "1",
        "--directed-max-aux-ratio", "0.25",
    ),
    "no_aux_cap": (
        "--directed-lambda-positive", "0.01",
        "--directed-lambda-mutual", "0.0",
        "--directed-lambda-negative", "0.0",
        "--directed-warmup-epochs", "3",
        "--directed-ramp-epochs", "2",
        "--directed-max-aux-ratio", "1.0",
    ),
    "add_mutual": (
        "--directed-lambda-positive", "0.01",
        "--directed-lambda-mutual", "0.005",
        "--directed-lambda-negative", "0.0",
        "--directed-warmup-epochs", "3",
        "--directed-ramp-epochs", "2",
        "--directed-max-aux-ratio", "0.25",
    ),
    "add_negative": (
        "--directed-lambda-positive", "0.01",
        "--directed-lambda-mutual", "0.0",
        "--directed-lambda-negative", "0.005",
        "--directed-warmup-epochs", "3",
        "--directed-ramp-epochs", "2",
        "--directed-max-aux-ratio", "0.25",
    ),
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
            raise RuntimeError(f"Task-2 detector workflow already runs as PID {pid}")
    atomic_json(LOCK, {"pid": os.getpid()})


def result_path(run_name: str, filename: str) -> Path:
    return (
        PROJECT_ROOT / "results" / "contrastive" / "runs"
        / run_name / filename
    )


def checkpoint_path(run_name: str) -> Path:
    return (
        PROJECT_ROOT / "checkpoints" / "contrastive" / "runs"
        / run_name / "best.pt"
    )


def execute(argv: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}); see {log_path}"
        )


def selected_epochs(tuning_run: str) -> int:
    payload = json.loads(
        result_path(tuning_run, "tuning_complete.json").read_text(
            encoding="utf-8"
        )
    )
    epochs = int(payload["selected_train_epochs"])
    if epochs < 1:
        raise ValueError(f"invalid epoch count for {tuning_run}: {epochs}")
    return epochs


def detector_names(seed: int, fold: int, improved: bool) -> tuple[str, str]:
    objective = "cl6" if improved else "focal"
    method = "cl6" if improved else "cl5"
    tune_tag = f"sci_{objective}_tune"
    tuning = f"{method}_task2_fold{fold}_seed{seed}_{tune_tag}"
    if not result_path(tuning, "tuning_complete.json").exists():
        return tuning, ""
    epochs = selected_epochs(tuning)
    full = (
        f"{method}_task2_fold{fold}_seed{seed}_"
        f"sci_{objective}_full_e{epochs}"
    )
    return tuning, full


def tuning_command(
    *, seed: int, fold: int, improved: bool, device: str, tag: str
) -> list[str]:
    objective = CL6_ARGUMENTS if improved else FOCAL_ARGUMENTS
    return [
        str(PYTHON), "-u", "scripts/run_contrastive.py",
        *objective,
        "--task", "2", "--fold", str(fold), "--seed", str(seed),
        "--epochs", "50", "--patience", "5",
        "--selection-metric", "mcc", "--skip-test",
        "--tag", tag, "--device", device,
    ]


def full_command(
    *, seed: int, fold: int, improved: bool, device: str,
    tag: str, epochs: int
) -> list[str]:
    objective = CL6_ARGUMENTS if improved else FOCAL_ARGUMENTS
    return [
        str(PYTHON), "-u", "scripts/run_contrastive.py",
        *objective,
        "--task", "2", "--fold", str(fold), "--seed", str(seed),
        "--selection-metric", "mcc", "--full-train-epochs", str(epochs),
        "--tag", tag, "--device", device,
    ]


def threshold_command(
    *, fold: int, tuning: str, full: str, device: str
) -> list[str]:
    return [
        str(PYTHON), "-u", "scripts/evaluate_threshold_transfer.py",
        "--task", "2", "--fold", str(fold),
        "--tuning-run", tuning, "--full-run", full,
        "--device", device,
        "--output", str(result_path(full, "threshold_transfer.json")),
    ]


def legacy_seed42_entries() -> list[dict]:
    payload = json.loads(LEGACY_MANIFEST.read_text(encoding="utf-8"))
    entries = [
        row for row in payload["runs"]
        if int(row["task"]) == 2 and int(row["seed"]) == 42
    ]
    if {int(row["fold"]) for row in entries} != set(FOLDS):
        raise RuntimeError("legacy seed-42 Task-2 five-fold results are incomplete")
    return sorted(entries, key=lambda row: int(row["fold"]))


def update_entry(manifest: dict, entry: dict, mutex: threading.Lock) -> None:
    with mutex:
        manifest["detector_runs"] = [
            row for row in manifest["detector_runs"]
            if (int(row["seed"]), int(row["fold"]))
            != (int(entry["seed"]), int(entry["fold"]))
        ]
        manifest["detector_runs"].append(entry)
        manifest["detector_runs"].sort(
            key=lambda row: (int(row["seed"]), int(row["fold"]))
        )
        atomic_json(MANIFEST, manifest)


def run_seed(
    seed: int,
    device: str,
    manifest: dict,
    status: dict,
    mutex: threading.Lock,
) -> None:
    log_path = LOG_ROOT / f"task2_detection_sci_seed{seed}.log"
    for fold in FOLDS:
        entry = {"task": 2, "fold": fold, "seed": seed}
        for improved in (False, True):
            label = "improved" if improved else "control"
            with mutex:
                status["active"][device] = {
                    "stage": "detector", "seed": seed, "fold": fold,
                    "objective": label,
                }
                atomic_json(STATUS, status)
            tuning, full = detector_names(seed, fold, improved)
            tune_tag = f"sci_{'cl6' if improved else 'focal'}_tune"
            if not result_path(tuning, "tuning_complete.json").exists():
                execute(
                    tuning_command(
                        seed=seed, fold=fold, improved=improved,
                        device=device, tag=tune_tag,
                    ),
                    log_path,
                )
            epochs = selected_epochs(tuning)
            if not full:
                full = (
                    f"{'cl6' if improved else 'cl5'}_task2_fold{fold}_"
                    f"seed{seed}_sci_{'cl6' if improved else 'focal'}_"
                    f"full_e{epochs}"
                )
            full_tag = full.split(f"_seed{seed}_", 1)[1]
            if not result_path(full, "test_metrics.csv").exists():
                execute(
                    full_command(
                        seed=seed, fold=fold, improved=improved,
                        device=device, tag=full_tag, epochs=epochs,
                    ),
                    log_path,
                )
            threshold = result_path(full, "threshold_transfer.json")
            if not threshold.exists():
                execute(
                    threshold_command(
                        fold=fold, tuning=tuning, full=full, device=device
                    ),
                    log_path,
                )
            entry[label] = {
                "tuning_run": tuning,
                "full_run": full,
                "selected_train_epochs": epochs,
                "test_metrics": str(
                    result_path(full, "test_metrics.csv").relative_to(PROJECT_ROOT)
                ),
                "threshold_transfer": str(threshold.relative_to(PROJECT_ROOT)),
                "checkpoint": str(checkpoint_path(full).relative_to(PROJECT_ROOT)),
            }
        update_entry(manifest, entry, mutex)
    with mutex:
        status["active"].pop(device, None)
        atomic_json(STATUS, status)


def run_ablation_worker(
    device: str,
    jobs: list[tuple[int, str]],
    manifest: dict,
    status: dict,
    mutex: threading.Lock,
) -> None:
    log_path = LOG_ROOT / f"task2_detection_sci_ablation_{device[-1]}.log"
    detector_lookup = {
        (int(row["seed"]), int(row["fold"])): row
        for row in manifest["detector_runs"]
    }
    for fold, variant in jobs:
        reference = detector_lookup[(42, fold)]["improved"]
        epochs = int(reference["selected_train_epochs"])
        tag = f"sci_ablation_{variant}_e{epochs}"
        run_name = ablation_run_name(fold, variant, epochs)
        if not result_path(run_name, "test_metrics.csv").exists():
            argv = [
                str(PYTHON), "-u", "scripts/run_contrastive.py",
                *ablation_objective(variant),
                "--task", "2", "--fold", str(fold), "--seed", "42",
                "--selection-metric", "mcc",
                "--full-train-epochs", str(epochs),
                "--tag", tag, "--device", device,
            ]
            execute(argv, log_path)
        row = {
            "fold": fold,
            "seed": 42,
            "variant": variant,
            "run_name": run_name,
            "reference_epoch_source": reference["tuning_run"],
            "fixed_train_epochs": epochs,
            "selected_train_epochs": epochs,
            "test_metrics": str(
                result_path(run_name, "test_metrics.csv").relative_to(
                    PROJECT_ROOT
                )
            ),
        }
        with mutex:
            manifest["ablation_runs"] = [
                existing for existing in manifest["ablation_runs"]
                if (int(existing["fold"]), existing["variant"])
                != (fold, variant)
            ]
            manifest["ablation_runs"].append(row)
            manifest["ablation_runs"].sort(
                key=lambda item: (int(item["fold"]), item["variant"])
            )
            status["active"][device] = {
                "stage": "ablation", "fold": fold, "variant": variant,
            }
            atomic_json(MANIFEST, manifest)
            atomic_json(STATUS, status)
    with mutex:
        status["active"].pop(device, None)
        atomic_json(STATUS, status)


def ablation_objective(variant: str) -> tuple[str, ...]:
    if variant == "remove_directed_loss":
        return FOCAL_ARGUMENTS
    return ("--method", "CL6", *ABLATIONS[variant])


def ablation_run_name(fold: int, variant: str, epochs: int) -> str:
    method_name = "cl5" if variant == "remove_directed_loss" else "cl6"
    return (
        f"{method_name}_task2_fold{fold}_seed42_"
        f"sci_ablation_{variant}_e{epochs}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs=2, default=("cuda:0", "cuda:1"))
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = {
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "detector_pairs": len(SEEDS) * len(FOLDS),
        "fixed_budget_ablation_runs": len(ABLATIONS) * len(FOLDS),
        "ablation_variants": list(ABLATIONS),
        "devices": list(args.devices),
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    acquire_lock()
    mutex = threading.Lock()
    status = {
        "state": "running", "pid": os.getpid(),
        "stage": "Task 2 three-seed detector",
        "plan": plan, "active": {},
    }
    manifest = {
        "protocol": {
            "task": 2,
            "outer_split": "published project-disjoint five-fold",
            "inner_selection": "project-disjoint validation MCC",
            "outer_test_policy": "one fixed-epoch evaluation per run",
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "development_units": "all five seed-42 folds observed before protocol freeze",
            "confirmatory_units": "all five folds for new seeds 123 and 2024 (10 paired units)",
            "primary_threshold": 0.5,
        },
        "detector_runs": legacy_seed42_entries(),
        "ablation_protocol": {
            "seed": 42,
            "epochs": "reuse the corresponding full CL6 validation-selected epoch",
            "outer_test_policy": "one evaluation; no ablation retuning on test",
            "variants": {name: list(values) for name, values in ABLATIONS.items()},
        },
        "ablation_runs": [],
    }
    if MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["detector_runs"] = previous.get(
            "detector_runs", manifest["detector_runs"]
        )
        manifest["ablation_runs"] = previous.get("ablation_runs", [])
    # Always refresh the already-complete development seed from its audited source.
    for entry in legacy_seed42_entries():
        update_entry(manifest, entry, mutex)
    atomic_json(STATUS, status)
    atomic_json(MANIFEST, manifest)
    try:
        pending_seeds = [
            seed for seed in SEEDS
            if seed != 42
            or any(
                (seed, fold) not in {
                    (int(row["seed"]), int(row["fold"]))
                    for row in manifest["detector_runs"]
                }
                for fold in FOLDS
            )
        ]
        if pending_seeds:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(
                        run_seed, seed, args.devices[index % 2],
                        manifest, status, mutex,
                    ): seed
                    for index, seed in enumerate(pending_seeds)
                }
                for future in as_completed(futures):
                    future.result()
        status["stage"] = "Task 2 fixed-budget CL6 ablations"
        atomic_json(STATUS, status)
        completed = {
            (int(row["fold"]), row["variant"])
            for row in manifest["ablation_runs"]
            if (PROJECT_ROOT / row["test_metrics"]).exists()
        }
        jobs = [
            (fold, variant)
            for fold in FOLDS for variant in ABLATIONS
            if (fold, variant) not in completed
        ]
        partitions = [jobs[::2], jobs[1::2]]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_ablation_worker, device, partition,
                    manifest, status, mutex,
                )
                for device, partition in zip(args.devices, partitions)
                if partition
            ]
            for future in as_completed(futures):
                future.result()
        status["stage"] = "Task 2 detector statistics"
        atomic_json(STATUS, status)
        execute(
            [str(PYTHON), "-u", "scripts/summarize_task2_detection_sci.py"],
            LOG_ROOT / "task2_detection_sci_summary.log",
        )
        status.update({
            "state": "complete",
            "stage": "Task 2 detector and ablations complete",
            "active": {},
            "summary": "results/workflow/task2_detection_sci_summary.json",
            "report": "reports/15_task2_detection_sci.md",
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
