"""Canonical result schema and normalization helpers.

The experiment scripts in this repository predate the reproducibility harness
and use task-specific metric names. This module keeps the required reporting
schema in one place and maps legacy output names into the normalized columns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


RESULT_COLUMNS = [
    "run_id",
    "timestamp",
    "git_commit",
    "task",
    "model",
    "generator",
    "seed",
    "alpha",
    "beta",
    "num_generated_samples",
    "mixture_components",
    "covariance_type",
    "learning_rate",
    "batch_size",
    "epochs",
    "solver",
    "solver_version",
    "hardware",
    "train_time_seconds",
    "eval_time_seconds",
    "inference_time_seconds",
    "metric_regret",
    "metric_cvar_regret",
    "metric_cvar_01_regret",
    "metric_objective",
    "metric_proxy_regret",
    "metric_true_regret",
    "metric_nll",
    "metric_true_nll",
    "metric_q_nll",
    "metric_mse",
    "metric_feasibility_violation",
    "w2_proxy_true",
    "sliced_wasserstein_true_model",
    "proxy_shift_norm",
    "q_architecture",
    "p_theta_architecture",
    "q_pretrain_epochs",
    "p_theta_pretrain_epochs",
    "status",
    "notes",
]


METRIC_ALIASES = {
    "average_regret": "metric_regret",
    "avg_regret": "metric_regret",
    "regret": "metric_regret",
    "true_regret": "metric_true_regret",
    "proxy_regret": "metric_proxy_regret",
    "average_objective": "metric_objective",
    "avg_objective": "metric_objective",
    "objective": "metric_objective",
    "cvar_regret": "metric_cvar_regret",
    "cvar_01_regret": "metric_cvar_01_regret",
    "average_mse": "metric_mse",
    "mse": "metric_mse",
    "final_nll_loss": "metric_nll",
    "nll": "metric_nll",
    "true_nll": "metric_true_nll",
    "q_nll": "metric_q_nll",
    "num_epochs": "epochs",
    "pretrain_epochs": "p_theta_pretrain_epochs",
    "pretrain_time_seconds": "train_time_seconds",
    "runtime_seconds": "train_time_seconds",
}


def stringify_value(value: Any) -> Any:
    """Return CSV-friendly scalar values while preserving missing values."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        # NumPy scalar support without importing NumPy in every script.
        if hasattr(value, "item"):
            return value.item()
    except ValueError:
        pass
    return json.dumps(value, sort_keys=True)


def flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten common wrapper structures into one shallow record."""
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if key in {"metadata", "metrics"} and isinstance(value, Mapping):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def normalize_result_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map a raw experiment record into the canonical result schema."""
    flat = flatten_record(record)
    row = {column: "" for column in RESULT_COLUMNS}

    for key, value in flat.items():
        target = METRIC_ALIASES.get(key, key)
        if target in row:
            row[target] = stringify_value(value)

    if not row["metric_true_regret"] and row["metric_regret"]:
        row["metric_true_regret"] = row["metric_regret"]
    if not row["metric_proxy_regret"] and row["metric_regret"]:
        row["metric_proxy_regret"] = row["metric_regret"]
    if not row["p_theta_architecture"] and row["generator"]:
        row["p_theta_architecture"] = row["generator"]
    if not row["q_architecture"] and "proxy_architecture" in flat:
        row["q_architecture"] = stringify_value(flat["proxy_architecture"])

    return row

