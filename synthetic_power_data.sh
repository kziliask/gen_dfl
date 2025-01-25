#!/usr/bin/env bash

#
# run_synthdata.sh
#
# Example usage:
#   bash run_synthdata.sh
#
# This script runs synthetic_data.py for multiple seeds or other parameters,
# saving data in the synthetic_power_data/ folder. 
#

# Make sure the script is executable: chmod +x run_synthdata.sh

# You can define parameter arrays and loop over them:
# SEEDS=(42 123 999)
# CITIES=(3 5)
SEEDS=(42)
CITIES=(3)
TIME_START=0
TIME_END=50
TIME_STEPS=51
OUTFOLDER="synthetic_power_data"

for seed in "${SEEDS[@]}"; do
  for nc in "${CITIES[@]}"; do
    echo "Generating data for seed=$seed, num_cities=$nc"
    python synthetic_power_data.py \
      --seed "$seed" \
      --num_cities "$nc" \
      --t_start "$TIME_START" \
      --t_end "$TIME_END" \
      --t_steps "$TIME_STEPS" \
      --outfolder "$OUTFOLDER"
  done
done

echo "Done generating synthetic data."
