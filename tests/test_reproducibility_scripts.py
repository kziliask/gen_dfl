import csv
import json
import subprocess
import sys
from pathlib import Path

from src.evaluation.result_schema import RESULT_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


def test_run_with_logging_records_success(tmp_path):
    raw_dir = tmp_path / "raw"
    logs_dir = tmp_path / "logs"
    command = [
        sys.executable,
        "scripts/run_with_logging.py",
        "--task",
        "smoke",
        "--model",
        "metadata",
        "--generator",
        "none",
        "--seed",
        "7",
        "--results-dir",
        str(raw_dir),
        "--logs-dir",
        str(logs_dir),
        "--",
        sys.executable,
        "--version",
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    records = [path for path in raw_dir.glob("*.json") if path.name != "run_registry.csv"]
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "success"
    assert Path(record["stdout_path"]).exists()
    assert (raw_dir / "run_registry.csv").exists()


def test_aggregate_results_normalizes_legacy_metrics(tmp_path):
    raw_dir = tmp_path / "raw"
    processed = tmp_path / "processed.csv"
    raw_dir.mkdir()
    (raw_dir / "fixture.json").write_text(
        json.dumps(
            {
                "task": "portfolio",
                "generator": "cnf",
                "seed": 42,
                "average_regret": 0.12,
                "cvar_regret": 0.34,
                "cvar_01_regret": 0.56,
                "average_objective": 1.23,
                "final_nll_loss": 4.5,
                "status": "success",
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_results.py",
            "--input",
            str(raw_dir),
            "--output",
            str(processed),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with processed.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert list(rows[0].keys()) == RESULT_COLUMNS
    assert rows[0]["metric_regret"] == "0.12"
    assert rows[0]["metric_cvar_regret"] == "0.34"
    assert rows[0]["metric_nll"] == "4.5"


def test_portfolio_config_runner_dry_run_uses_extended_seed_matrix():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_portfolio_config.py",
            "configs/portfolio/cnf_reproduction.yaml",
            "--seeds",
            "42,43,44,45,46",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    commands = [line for line in result.stdout.splitlines() if "end2end_cflowdfl_portfolio_alpha.py" in line]
    assert len(commands) == 5
    assert "--seed 42" in commands[0]
    assert "--seed 46" in commands[-1]


def test_portfolio_config_runner_expands_mixture_component_sweep():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_portfolio_config.py",
            "configs/portfolio/gmm_k_sweep.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    commands = [line for line in result.stdout.splitlines() if "end2end_cflowdfl_portfolio_alpha.py" in line]
    assert len(commands) == 25
    assert "--mixture_components 1" in commands[0]
    assert "--mixture_components 10" in commands[-1]


def test_portfolio_config_runner_rejects_unsupported_list_fields(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        "\n".join(
            [
                "extends: configs/portfolio/cnf_reproduction.yaml",
                "rank: [1, 2]",
            ]
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_portfolio_config.py",
            str(config),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Config error:" in result.stderr
    assert "rank" in result.stderr
    assert "scalar-only" in result.stderr
