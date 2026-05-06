#!/usr/bin/env python3
"""Create draft figures from processed CSVs when required columns exist."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/processed/aggregated_results.csv")
    parser.add_argument("--output-dir", default="figures/draft")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import pandas as pd
    import matplotlib.pyplot as plt

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        print(f"No processed CSV found at {input_path}")
        return 0

    data = pd.read_csv(input_path)
    needed = {"generator", "metric_regret", "train_time_seconds"}
    if not needed.issubset(data.columns) or data.empty:
        print("Processed CSV does not yet contain regret/runtime columns with data")
        return 0

    plot_data = data.copy()
    plot_data["metric_regret"] = pd.to_numeric(plot_data["metric_regret"], errors="coerce")
    plot_data["train_time_seconds"] = pd.to_numeric(plot_data["train_time_seconds"], errors="coerce")
    plot_data = plot_data.dropna(subset=["metric_regret", "train_time_seconds"])
    if plot_data.empty:
        print("No numeric regret/runtime data available")
        return 0

    fig, ax = plt.subplots(figsize=(6, 4))
    for generator, group in plot_data.groupby("generator"):
        ax.scatter(group["train_time_seconds"], group["metric_regret"], label=generator or "unknown")
    ax.set_xlabel("Train time (seconds)")
    ax.set_ylabel("Regret")
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "regret_vs_runtime.pdf"
    fig.savefig(output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

