#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/zilikons/conda/envs/gendfl/bin/python}"

"${PYTHON_BIN}" scripts/run_with_logging.py \
  --task smoke_tests \
  --model metadata \
  --generator none \
  --seed 0 \
  --notes "wrapper metadata smoke" \
  -- "${PYTHON_BIN}" --version

"${PYTHON_BIN}" scripts/aggregate_results.py \
  --input results/raw \
  --output results/processed/schema_check.csv

