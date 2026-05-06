#!/usr/bin/env python3
"""Run a Portfolio experiment matrix from a simple YAML config.

The project environment does not currently include PyYAML, so this runner
supports the small config subset used under ``configs/portfolio``:

- ``key: value`` mappings
- inline lists such as ``seeds: [42, 43, 44]``
- one-level ``extends: path/to/base.yaml``
"""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SWEEP_KEYS = [
    "generator",
    "mixture_components",
    "num_generated_samples",
    "beta",
    "alpha",
    "n",
    "m",
    "deg",
    "noise_width",
    "num_epochs",
    "batch_size",
    "learning_rate",
    "contextual_pretrain_epochs",
    "p_theta_pretrain_epochs",
]
SEQUENCE_ONLY_KEYS = {"seeds", *SWEEP_KEYS}


class ConfigError(ValueError):
    """Raised for user-facing config problems."""


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line in {path}: {raw_line}")
        key, value = line.split(":", 1)
        data[key.strip()] = parse_scalar(value)
    return data


def load_config(path: Path) -> dict[str, Any]:
    data = load_simple_yaml(path)
    parent = data.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = ROOT / parent_path
        base = load_config(parent_path)
        base.update(data)
        return base
    return data


def validate_config_shape(config: dict[str, Any], config_path: Path) -> None:
    for key, value in config.items():
        if isinstance(value, list) and key not in SEQUENCE_ONLY_KEYS:
            allowed = ", ".join(["seeds", *SWEEP_KEYS])
            raise ConfigError(
                f"{config_path}: key '{key}' received a list, but this field is scalar-only. "
                f"List-valued sweeps are currently supported only for: {allowed}."
            )
    if config.get("generator") != "gmm" and isinstance(config.get("mixture_components"), list):
        raise ConfigError(
            f"{config_path}: 'mixture_components' is a GMM-only sweep. "
            "Set generator: gmm or remove mixture_components."
        )


def csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def seed_list(config: dict[str, Any], override: str | None, config_path: Path) -> list[int]:
    raw = csv_ints(override) if override else config.get("seeds", [config.get("seed", 42)])
    if isinstance(raw, int):
        raw = [raw]
    if not isinstance(raw, list):
        raise ConfigError(f"{config_path}: seeds must be an integer or list of integers.")
    try:
        seeds = [int(seed) for seed in raw]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{config_path}: seeds must contain only integers, got {raw!r}.") from exc
    if not seeds:
        raise ConfigError("Config must provide at least one seed.")
    return seeds


def add_option(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def build_gendfl_command(config: dict[str, Any], seed: int, python_bin: str) -> list[str]:
    generator = str(config.get("generator", "cnf"))
    command = [
        python_bin,
        "scripts/run_with_logging.py",
        "--task",
        "portfolio",
        "--model",
        str(config.get("model", "gen-dfl")),
        "--generator",
        generator,
        "--seed",
        str(seed),
        "--alpha",
        str(config.get("alpha", 1.0)),
        "--beta",
        str(config.get("beta", 100)),
        "--num-generated-samples",
        str(config.get("num_generated_samples", 200)),
        "--batch-size",
        str(config.get("batch_size", 32)),
        "--epochs",
        str(config.get("num_epochs", config.get("epochs", 1))),
        "--learning-rate",
        str(config.get("learning_rate", 0.001)),
        "--q-pretrain-epochs",
        str(config.get("contextual_pretrain_epochs", 30)),
        "--p-theta-pretrain-epochs",
        str(config.get("p_theta_pretrain_epochs", 50)),
    ]
    if generator == "gmm":
        command.extend(
            [
                "--mixture-components",
                str(config.get("mixture_components", 1)),
                "--covariance-type",
                str(config.get("covariance_type", "diagonal")),
            ]
        )

    command.extend(
        [
            "--",
            python_bin,
            "end2end_cflowdfl_portfolio_alpha.py",
            "--generator",
            generator,
            "--seed",
            str(seed),
            "--betas",
            str(config.get("beta", 100)),
            "--n",
            str(config.get("n", 200)),
            "--m",
            str(config.get("m", 50)),
            "--num_epochs",
            str(config.get("num_epochs", config.get("epochs", 1))),
            "--batch_size",
            str(config.get("batch_size", 32)),
            "--num_generated_samples",
            str(config.get("num_generated_samples", 200)),
            "--contextual_pretrain_epochs",
            str(config.get("contextual_pretrain_epochs", 30)),
            "--p_theta_pretrain_epochs",
            str(config.get("p_theta_pretrain_epochs", 50)),
            "--learning_rate",
            str(config.get("learning_rate", 0.001)),
            "--noise_width",
            str(config.get("noise_width", 20)),
            "--deg",
            str(config.get("deg", 5)),
            "--num_experiments",
            "1",
            "--alpha",
            str(config.get("alpha", 1.0)),
        ]
    )
    add_option(command, "--mixture_components", config.get("mixture_components") if generator == "gmm" else None)
    add_option(command, "--rank", config.get("rank"))
    add_option(command, "--raw_results_path", config.get("raw_results_path"))
    return command


def build_baseline_command(config: dict[str, Any], seed: int, python_bin: str) -> list[str]:
    loss_func = str(config.get("loss_func", config.get("model", "spo+")))
    return [
        python_bin,
        "scripts/run_with_logging.py",
        "--task",
        "portfolio",
        "--model",
        loss_func,
        "--generator",
        "predictive",
        "--seed",
        str(seed),
        "--batch-size",
        str(config.get("batch_size", 32)),
        "--epochs",
        str(config.get("num_epochs", config.get("epochs", 1))),
        "--",
        python_bin,
        "pred_dfl_portfolio.py",
        "--loss_func",
        loss_func,
        "--seed",
        str(seed),
        "--deg",
        str(config.get("deg", 5)),
        "--m",
        str(config.get("m", 50)),
        "--n",
        str(config.get("n", 200)),
        "--noise_widths",
        str(config.get("noise_width", 20)),
        "--num_experiments",
        "1",
        "--batch_size",
        str(config.get("batch_size", 32)),
        "--num_epochs",
        str(config.get("num_epochs", config.get("epochs", 1))),
    ]


def build_command(config: dict[str, Any], seed: int, python_bin: str) -> list[str]:
    model = str(config.get("model", "gen-dfl"))
    if model == "gen-dfl":
        return build_gendfl_command(config, seed, python_bin)
    return build_baseline_command(config, seed, python_bin)


def expand_sweep_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand list-valued sweep keys into one concrete config per setting."""
    configs = [dict(config)]
    for key in SWEEP_KEYS:
        value = config.get(key)
        if not isinstance(value, list):
            continue
        expanded = []
        for current in configs:
            for item in value:
                next_config = dict(current)
                next_config[key] = item
                expanded.append(next_config)
        configs = expanded
    return configs


def run_label(config: dict[str, Any], seed: int) -> str:
    parts = [f"seed={seed}", f"generator={config.get('generator', 'cnf')}"]
    if config.get("generator") == "gmm":
        parts.append(f"K={config.get('mixture_components', 1)}")
    return " ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a configs/portfolio YAML file.")
    parser.add_argument("--python-bin", default="/Users/zilikons/conda/envs/gendfl/bin/python")
    parser.add_argument("--seeds", help="Comma-separated seed override, for example 42,43,44,45,46.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        config = load_config(config_path)
        validate_config_shape(config, config_path)
        seeds = seed_list(config, args.seeds, config_path)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    concrete_configs = expand_sweep_configs(config)
    failures = []
    for current_config in concrete_configs:
        for seed in seeds:
            command = build_command(current_config, int(seed), args.python_bin)
            print(f"# {run_label(current_config, int(seed))}", flush=True)
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
            if result.returncode != 0:
                failures.append((seed, result.returncode, current_config))
                if not args.keep_going:
                    return result.returncode

    if failures:
        print(f"Failed seeds: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
