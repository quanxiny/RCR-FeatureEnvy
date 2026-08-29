#!/usr/bin/env python3
"""Run the final Task-2 publication controls.

The extension closes two deliberately narrow comparison gaps:

* a conventional supervised contrastive objective over graph-pair embeddings;
* a strict seed-100 five-fold Focal/CL6 pairing aligned with the released code.

All development choices use inner validation data.  Every full-training run
evaluates the immutable outer test fold once, after its epoch budget is fixed.
The workflow is idempotent and resumes from atomic run artifacts.
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
STATUS = WORKFLOW_ROOT / "task2_comparison_extension_status.json"
MANIFEST = WORKFLOW_ROOT / "task2_comparison_extension_manifest.json"
LOCK = WORKFLOW_ROOT / "task2_comparison_extension.lock"
LOG_ROOT = PROJECT_ROOT / "logs" / "workflow"
FOLDS = (1, 2, 3, 4, 5)
CONFIRMATORY_SEEDS = (123, 2024)
SUPCON_TEMPERATURES = (0.07, 0.10, 0.20)
SUPCON_WEIGHTS = (0.05, 0.10, 0.20)

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
            raise RuntimeError(f"comparison extension already runs as PID {pid}")
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


def float_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "")


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


def tuning_command(
    *, method_arguments: tuple[str, ...], seed: int, fold: int,
    tag: str, device: str,
) -> list[str]:
    return [
        str(PYTHON), "-u", "scripts/run_contrastive.py",
        *method_arguments,
        "--task", "2", "--fold", str(fold), "--seed", str(seed),
        "--epochs", "50", "--patience", "5",
        "--selection-metric", "mcc", "--skip-test",
        "--tag", tag, "--device", device,
    ]


def full_command(
    *, method_arguments: tuple[str, ...], seed: int, fold: int,
    tag: str, epochs: int, device: str,
) -> list[str]:
    return [
        str(PYTHON), "-u", "scripts/run_contrastive.py",
        *method_arguments,
        "--task", "2", "--fold", str(fold), "--seed", str(seed),
        "--selection-metric", "mcc",
        "--full-train-epochs", str(epochs),
        "--tag", tag, "--device", device,
    ]


def selected_epochs(run_name: str) -> int:
    payload = json.loads(
        result_path(run_name, "tuning_complete.json").read_text(
            encoding="utf-8"
        )
    )
    epochs = int(payload["selected_train_epochs"])
    if epochs < 1:
        raise ValueError(f"invalid selected epoch count for {run_name}: {epochs}")
    return epochs


def supcon_arguments(temperature: float, weight: float) -> tuple[str, ...]:
    return (
        "--method", "CL1",
        "--temperature", str(temperature),
        "--lambda-label", str(weight),
    )


def supcon_dev_name(temperature: float, weight: float) -> str:
    tag = (
        f"pub_supcon_dev_t{float_tag(temperature)}_"
        f"l{float_tag(weight)}"
    )
    return f"cl1_task2_fold1_seed42_{tag}"


def run_supcon_development(
    temperature: float,
    weight: float,
    device: str,
    status: dict,
    mutex: threading.Lock,
) -> dict:
    run_name = supcon_dev_name(temperature, weight)
    tag = run_name.split("_seed42_", 1)[1]
    with mutex:
        status["active"][device] = {
            "stage": "supcon_development",
            "temperature": temperature,
            "lambda_label": weight,
        }
        atomic_json(STATUS, status)
    if not result_path(run_name, "tuning_complete.json").exists():
        execute(
            tuning_command(
                method_arguments=supcon_arguments(temperature, weight),
                seed=42,
                fold=1,
                tag=tag,
                device=device,
            ),
            LOG_ROOT / f"task2_supcon_dev_{device[-1]}.log",
        )
    payload = json.loads(
        result_path(run_name, "tuning_complete.json").read_text(
            encoding="utf-8"
        )
    )
    with mutex:
        status["active"].pop(device, None)
        atomic_json(STATUS, status)
    return {
        "temperature": temperature,
        "lambda_label": weight,
        "run_name": run_name,
        "selected_train_epochs": int(payload["selected_train_epochs"]),
        "selection_score": float(payload["selection_score"]),
        "selection_metrics": payload["selection_metrics"],
        "outer_test_evaluated": False,
    }


def development_rank(row: dict) -> tuple[float, float, float, float]:
    metrics = row["selection_metrics"]
    # The final two terms prefer the lower-complexity/default setting on an
    # exact metric tie without consulting any outer-test result.
    return (
        float(row["selection_score"]),
        float(metrics["pr_auc"]),
        -abs(float(row["temperature"]) - 0.10),
        -abs(float(row["lambda_label"]) - 0.10),
    )


def supcon_names(
    seed: int, fold: int, temperature: float, weight: float
) -> tuple[str, str]:
    suffix = f"t{float_tag(temperature)}_l{float_tag(weight)}"
    tuning = f"cl1_task2_fold{fold}_seed{seed}_pub_supcon_tune_{suffix}"
    if not result_path(tuning, "tuning_complete.json").exists():
        return tuning, ""
    epochs = selected_epochs(tuning)
    full = (
        f"cl1_task2_fold{fold}_seed{seed}_"
        f"pub_supcon_full_{suffix}_e{epochs}"
    )
    return tuning, full


def run_supcon_formal(
    seed: int,
    fold: int,
    temperature: float,
    weight: float,
    device: str,
    status: dict,
    mutex: threading.Lock,
) -> dict:
    with mutex:
        status["active"][device] = {
            "stage": "supcon_confirmatory", "seed": seed, "fold": fold,
        }
        atomic_json(STATUS, status)
    arguments = supcon_arguments(temperature, weight)
    tuning, full = supcon_names(seed, fold, temperature, weight)
    tune_tag = tuning.split(f"_seed{seed}_", 1)[1]
    log_path = LOG_ROOT / f"task2_comparison_extension_{device[-1]}.log"
    if not result_path(tuning, "tuning_complete.json").exists():
        execute(
            tuning_command(
                method_arguments=arguments,
                seed=seed,
                fold=fold,
                tag=tune_tag,
                device=device,
            ),
            log_path,
        )
    epochs = selected_epochs(tuning)
    if not full:
        suffix = f"t{float_tag(temperature)}_l{float_tag(weight)}"
        full = (
            f"cl1_task2_fold{fold}_seed{seed}_"
            f"pub_supcon_full_{suffix}_e{epochs}"
        )
    full_tag = full.split(f"_seed{seed}_", 1)[1]
    if not result_path(full, "test_metrics.csv").exists():
        execute(
            full_command(
                method_arguments=arguments,
                seed=seed,
                fold=fold,
                tag=full_tag,
                epochs=epochs,
                device=device,
            ),
            log_path,
        )
    with mutex:
        status["active"].pop(device, None)
        atomic_json(STATUS, status)
    return {
        "seed": seed,
        "fold": fold,
        "tuning_run": tuning,
        "full_run": full,
        "selected_train_epochs": epochs,
        "test_metrics": str(
            result_path(full, "test_metrics.csv").relative_to(PROJECT_ROOT)
        ),
        "checkpoint": str(checkpoint_path(full).relative_to(PROJECT_ROOT)),
        "outer_test_evaluations": 1,
    }


def seed100_names(fold: int, objective: str) -> tuple[str, str]:
    method = "cl6" if objective == "cl6" else "cl5"
    tuning = (
        f"{method}_task2_fold{fold}_seed100_"
        f"pub_seed100_{objective}_tune"
    )
    if not result_path(tuning, "tuning_complete.json").exists():
        return tuning, ""
    epochs = selected_epochs(tuning)
    full = (
        f"{method}_task2_fold{fold}_seed100_"
        f"pub_seed100_{objective}_full_e{epochs}"
    )
    return tuning, full


def run_seed100(
    fold: int,
    objective: str,
    device: str,
    status: dict,
    mutex: threading.Lock,
) -> dict:
    if objective not in {"focal", "cl6"}:
        raise ValueError(objective)
    with mutex:
        status["active"][device] = {
            "stage": "seed100_alignment", "fold": fold,
            "objective": objective,
        }
        atomic_json(STATUS, status)
    arguments = CL6_ARGUMENTS if objective == "cl6" else FOCAL_ARGUMENTS
    tuning, full = seed100_names(fold, objective)
    tune_tag = tuning.split("_seed100_", 1)[1]
    log_path = LOG_ROOT / f"task2_comparison_extension_{device[-1]}.log"
    if not result_path(tuning, "tuning_complete.json").exists():
        execute(
            tuning_command(
                method_arguments=arguments,
                seed=100,
                fold=fold,
                tag=tune_tag,
                device=device,
            ),
            log_path,
        )
    epochs = selected_epochs(tuning)
    if not full:
        method = "cl6" if objective == "cl6" else "cl5"
        full = (
            f"{method}_task2_fold{fold}_seed100_"
            f"pub_seed100_{objective}_full_e{epochs}"
        )
    full_tag = full.split("_seed100_", 1)[1]
    if not result_path(full, "test_metrics.csv").exists():
        execute(
            full_command(
                method_arguments=arguments,
                seed=100,
                fold=fold,
                tag=full_tag,
                epochs=epochs,
                device=device,
            ),
            log_path,
        )
    with mutex:
        status["active"].pop(device, None)
        atomic_json(STATUS, status)
    return {
        "seed": 100,
        "fold": fold,
        "objective": objective,
        "tuning_run": tuning,
        "full_run": full,
        "selected_train_epochs": epochs,
        "test_metrics": str(
            result_path(full, "test_metrics.csv").relative_to(PROJECT_ROOT)
        ),
        "checkpoint": str(checkpoint_path(full).relative_to(PROJECT_ROOT)),
        "outer_test_evaluations": 1,
    }


def run_jobs(
    jobs: list[tuple],
    worker,
    devices: tuple[str, str],
    status: dict,
    mutex: threading.Lock,
) -> list[dict]:
    pending = list(jobs)
    queue_mutex = threading.Lock()

    def run_device(device: str) -> list[dict]:
        completed = []
        while True:
            with queue_mutex:
                if not pending:
                    break
                job = pending.pop(0)
            completed.append(worker(*job, device, status, mutex))
        return completed

    output: list[dict] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_device, device)
            for device in devices
            if jobs
        ]
        for future in as_completed(futures):
            output.extend(future.result())
    return output


def run_final_matrix_job(
    kind: str,
    parameters: tuple,
    device: str,
    status: dict,
    mutex: threading.Lock,
) -> dict:
    if kind == "supcon":
        return run_supcon_formal(*parameters, device, status, mutex)
    if kind == "seed100":
        return run_seed100(*parameters, device, status, mutex)
    raise ValueError(f"unknown final matrix job kind: {kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs=2, default=("cuda:0", "cuda:1"))
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = {
        "supcon_development_tuning_runs": (
            len(SUPCON_TEMPERATURES) * len(SUPCON_WEIGHTS)
        ),
        "supcon_confirmatory_runs": len(CONFIRMATORY_SEEDS) * len(FOLDS),
        "seed100_alignment_runs": len(FOLDS) * 2,
        "devices": list(args.devices),
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    acquire_lock()
    mutex = threading.Lock()
    status = {
        "state": "running", "pid": os.getpid(), "plan": plan, "active": {},
        "stage": "SupCon development",
    }
    atomic_json(STATUS, status)
    try:
        development_jobs = [
            (temperature, weight)
            for temperature in SUPCON_TEMPERATURES
            for weight in SUPCON_WEIGHTS
        ]
        development = run_jobs(
            development_jobs,
            run_supcon_development,
            tuple(args.devices),
            status,
            mutex,
        )
        development.sort(
            key=lambda row: (row["temperature"], row["lambda_label"])
        )
        selected = max(development, key=development_rank)
        manifest = {
            "protocol": {
                "task": 2,
                "outer_split": "published project-disjoint five-fold",
                "inner_selection": "project-disjoint validation MCC",
                "outer_test_policy": "one fixed-epoch evaluation per full run",
                "supcon_development_unit": "seed42 fold1 inner validation only",
                "supcon_confirmatory_units": (
                    "seeds123/2024 x five folds; no development outer tests"
                ),
                "seed100_role": (
                    "strict same-data alignment with the released training seed"
                ),
            },
            "supcon_development_grid": development,
            "selected_supcon": {
                "temperature": selected["temperature"],
                "lambda_label": selected["lambda_label"],
                "selection_source": selected["run_name"],
                "selection_metric": "mcc",
            },
            "supcon_confirmatory_runs": [],
            "seed100_alignment_runs": [],
        }
        atomic_json(MANIFEST, manifest)

        status["stage"] = "dynamic dual-GPU final comparison matrix"
        atomic_json(STATUS, status)
        final_jobs = [
            (
                "supcon",
                (
                    seed, fold,
                    float(selected["temperature"]),
                    float(selected["lambda_label"]),
                ),
            )
            for seed in CONFIRMATORY_SEEDS for fold in FOLDS
        ]
        final_jobs.extend(
            ("seed100", (fold, objective))
            for fold in FOLDS for objective in ("focal", "cl6")
        )
        completed_final = run_jobs(
            final_jobs, run_final_matrix_job, tuple(args.devices), status, mutex
        )
        formal = [
            row for row in completed_final
            if int(row["seed"]) in CONFIRMATORY_SEEDS
        ]
        formal.sort(key=lambda row: (row["seed"], row["fold"]))
        manifest["supcon_confirmatory_runs"] = formal
        atomic_json(MANIFEST, manifest)

        aligned = [
            row for row in completed_final if int(row["seed"]) == 100
        ]
        aligned.sort(key=lambda row: (row["fold"], row["objective"]))
        manifest["seed100_alignment_runs"] = aligned
        atomic_json(MANIFEST, manifest)

        status["stage"] = "comparison statistics and report"
        atomic_json(STATUS, status)
        execute(
            [
                str(PYTHON), "-u",
                "scripts/summarize_task2_comparison_extension.py",
            ],
            LOG_ROOT / "task2_comparison_extension_summary.log",
        )
        status.update({
            "state": "complete",
            "stage": "publication comparison extension complete",
            "active": {},
            "summary": "results/workflow/task2_comparison_extension_summary.json",
            "report": "reports/19_task2_comparison_extension.md",
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
