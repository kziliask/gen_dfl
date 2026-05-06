#!/usr/bin/env python3
"""Normalize raw JSON/JSONL experiment records into a schema-stable CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.evaluation.result_schema import RESULT_COLUMNS, normalize_result_row


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield payload
        return

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            for item in payload["results"]:
                if isinstance(item, dict):
                    yield item
        else:
            yield payload


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = list(input_path.rglob("*.json")) + list(input_path.rglob("*.jsonl"))
    return sorted(path for path in files if not path.name.startswith("."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw")
    parser.add_argument("--output", default="results/processed/aggregated_results.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = []
    for path in collect_input_files(input_path):
        for record in iter_records(path):
            rows.append(normalize_result_row(record))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} normalized rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

