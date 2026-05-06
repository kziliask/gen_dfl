#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/zilikons/conda/envs/gendfl/bin/python}"

"${PYTHON_BIN}" scripts/run_with_logging.py \
  --task secondary \
  --model gen-dfl \
  --generator cnf \
  --seed "${SEED:-42}" \
  --alpha "${ALPHA:-1.0}" \
  --beta "${BETA:-100}" \
  --epochs "${EPOCHS:-1}" \
  -- "${PYTHON_BIN}" end2end_cflowdfl_shortp.py \
    --betas "${BETA:-100}" \
    --n "${N:-20}" \
    --deg "${DEG:-5}" \
    --num_epochs "${EPOCHS:-1}" \
    --num_experiments 1 \
    --alpha "${ALPHA:-1.0}"

