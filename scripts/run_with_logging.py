#!/usr/bin/env python3
"""Run a command while saving reproducibility metadata and logs."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.result_schema import RESULT_COLUMNS, normalize_result_row


EXTRA_REGISTRY_COLUMNS = [
    "raw_log_path",
    "stdout_path",
    "stderr_path",
    "returncode",
    "runtime_seconds",
    "dirty_status",
    "exact_command",
]


def run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def solver_version() -> tuple[str, str]:
    try:
        gp = importlib.import_module("gurobipy")
        version = getattr(gp, "gurobi").version()
        return "gurobi", ".".join(str(part) for part in version)
    except Exception:
        pass
    return "", ""


def hardware_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }
    torch_version = package_version("torch")
    if torch_version:
        summary["torch"] = torch_version
        code = (
            "import json, torch; "
            "payload={'cuda_available': bool(torch.cuda.is_available()), "
            "'mps_available': bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())}; "
            "payload['cuda_device_count'] = torch.cuda.device_count() if payload['cuda_available'] else 0; "
            "payload['cuda_device_name'] = torch.cuda.get_device_name(0) if payload['cuda_available'] else ''; "
            "print(json.dumps(payload))"
        )
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
                env=env,
            )
            if result.returncode == 0 and result.stdout.strip():
                summary.update(json.loads(result.stdout))
            else:
                summary["torch_probe_error"] = result.stderr.strip()[:500]
        except Exception as exc:
            summary["torch_probe_error"] = str(exc)
    return summary


def load_metric_files(paths: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            if payload:
                payload = payload[-1]
            else:
                payload = {}
        if not isinstance(payload, dict):
            raise TypeError(f"metrics file must contain an object or non-empty list: {path}")
        merged.update(payload)
    return merged


def append_registry(path: Path, row: dict[str, Any]) -> None:
    columns = RESULT_COLUMNS + EXTRA_REGISTRY_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--generator", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--alpha", default="")
    parser.add_argument("--beta", default="")
    parser.add_argument("--num-generated-samples", default="")
    parser.add_argument("--mixture-components", default="")
    parser.add_argument("--covariance-type", default="")
    parser.add_argument("--learning-rate", default="")
    parser.add_argument("--batch-size", default="")
    parser.add_argument("--epochs", default="")
    parser.add_argument("--q-architecture", default="")
    parser.add_argument("--p-theta-architecture", default="")
    parser.add_argument("--q-pretrain-epochs", default="")
    parser.add_argument("--p-theta-pretrain-epochs", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--metrics-json", action="append", default=[])
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--logs-dir", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("run_with_logging.py requires a command after --")

    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    results_dir = Path(args.results_dir)
    task_log_dir = args.logs_dir or os.path.join("logs", args.task or "runs")
    logs_dir = Path(task_log_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = logs_dir / f"{run_id}.stdout.log"
    stderr_path = logs_dir / f"{run_id}.stderr.log"
    raw_log_path = results_dir / f"{run_id}.json"
    exact_command = shlex.join(command)

    solver_name, solver_ver = solver_version()
    dirty_status = run_git(["status", "--short"])
    metric_data = load_metric_files(args.metrics_json)

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "task": args.task,
        "model": args.model,
        "generator": args.generator,
        "seed": args.seed,
        "alpha": args.alpha,
        "beta": args.beta,
        "num_generated_samples": args.num_generated_samples,
        "mixture_components": args.mixture_components,
        "covariance_type": args.covariance_type,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "solver": solver_name,
        "solver_version": solver_ver,
        "hardware": hardware_summary(),
        "q_architecture": args.q_architecture,
        "p_theta_architecture": args.p_theta_architecture or args.generator,
        "q_pretrain_epochs": args.q_pretrain_epochs,
        "p_theta_pretrain_epochs": args.p_theta_pretrain_epochs,
        "notes": args.notes,
        "dirty_status": dirty_status,
        "exact_command": exact_command,
        "versions": {
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "pyepo": package_version("pyepo"),
            "gurobipy": package_version("gurobipy"),
        },
    }

    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.run(command, stdout=stdout_handle, stderr=stderr_handle, text=True)
    runtime_seconds = time.perf_counter() - start

    status = "success" if process.returncode == 0 else "failed"
    record: dict[str, Any] = {
        **metadata,
        **metric_data,
        "status": status,
        "returncode": process.returncode,
        "runtime_seconds": runtime_seconds,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "raw_log_path": str(raw_log_path),
    }
    normalized = normalize_result_row(record)
    registry_row = {
        **normalized,
        "raw_log_path": str(raw_log_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "returncode": process.returncode,
        "runtime_seconds": runtime_seconds,
        "dirty_status": dirty_status,
        "exact_command": exact_command,
    }

    with raw_log_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    append_registry(results_dir / "run_registry.csv", registry_row)

    print(json.dumps(registry_row, indent=2, sort_keys=True))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
