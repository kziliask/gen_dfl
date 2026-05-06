#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/zilikons/conda/envs/gendfl/bin/python}"

"${PYTHON_BIN}" scripts/run_with_logging.py \
  --task portfolio \
  --model gen-dfl \
  --generator gmm \
  --seed "${SEED:-42}" \
  --alpha "${ALPHA:-1.0}" \
  --beta "${BETA:-100}" \
  --num-generated-samples "${NUM_GENERATED_SAMPLES:-200}" \
  --mixture-components "${MIXTURE_COMPONENTS:-1}" \
  --covariance-type diagonal \
  --batch-size "${BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-1}" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --q-pretrain-epochs "${CONTEXTUAL_PRETRAIN_EPOCHS:-30}" \
  --p-theta-pretrain-epochs "${P_THETA_PRETRAIN_EPOCHS:-50}" \
  -- "${PYTHON_BIN}" end2end_cflowdfl_portfolio_alpha.py \
    --generator gmm \
    --mixture_components "${MIXTURE_COMPONENTS:-1}" \
    --seed "${SEED:-42}" \
    --betas "${BETA:-100}" \
    --n "${N:-20}" \
    --m "${M:-5}" \
    --num_epochs "${EPOCHS:-1}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --num_generated_samples "${NUM_GENERATED_SAMPLES:-200}" \
    --contextual_pretrain_epochs "${CONTEXTUAL_PRETRAIN_EPOCHS:-30}" \
    --p_theta_pretrain_epochs "${P_THETA_PRETRAIN_EPOCHS:-50}" \
    --learning_rate "${LEARNING_RATE:-0.001}" \
    --noise_width "${NOISE_WIDTH:-20}" \
    --deg "${DEG:-5}" \
    --num_experiments 1 \
    --alpha "${ALPHA:-1.0}"
