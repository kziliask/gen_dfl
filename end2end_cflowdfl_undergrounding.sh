python end2end_cflowdfl_undergrounding.py \
  --train_file synthetic_power_data/synthetic_data_seed42_train.pt \
  --test_file synthetic_power_data/synthetic_data_seed42_test.pt \
  --capacity 2 \
  --batch_size 2 \
  --num_epochs 30 \
  --pretrain_epochs 10 \
  --alpha 0.5 \
  --beta 5 \
  --outdir results_undergrounding