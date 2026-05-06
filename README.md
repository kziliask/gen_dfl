# gen_dfl

This repository contains a clone of the official implementation of `{wang2026gendfl}`, which is the codebase for the paper:

**Gen-DFL: Decision-Focused Generative Learning for Robust Decision Making** (ICLR 2026)

## Reproducibility Workflow

Activate the project environment before running experiments:

```bash
mamba activate gendfl
```

For non-interactive runs, use the resolved interpreter directly:

```bash
/Users/zilikons/conda/envs/gendfl/bin/python scripts/run_with_logging.py \
  --task smoke_tests --model metadata --generator none --seed 0 \
  -- /Users/zilikons/conda/envs/gendfl/bin/python --version
```

Normalize raw JSON/JSONL outputs into the canonical CSV schema:

```bash
/Users/zilikons/conda/envs/gendfl/bin/python scripts/aggregate_results.py \
  --input results/raw \
  --output results/processed/aggregated_results.csv
```

Portfolio CNF and GMM smoke wrappers are available at:

```bash
scripts/run_portfolio_cnf.sh
scripts/run_portfolio_gmm.sh
```

For multi-seed runs, prefer the config runner:

```bash
/Users/zilikons/conda/envs/gendfl/bin/python scripts/run_portfolio_config.py \
  configs/portfolio/cnf_main.yaml
```

Preview the exact commands without running them:

```bash
/Users/zilikons/conda/envs/gendfl/bin/python scripts/run_portfolio_config.py \
  configs/portfolio/cnf_main.yaml --dry-run
```

Implementation details, deviations, environment notes, and claim tracking live under `docs/`.

## Citation

~~~bibtex
@inproceedings{wang2026gendfl,
  title     = {Gen-DFL: Decision-Focused Generative Learning for Robust Decision Making},
  author    = {Wang, Prince Zizhuang and Chen, Shuyi and Liang, Jinhao and Fioretto, Ferdinando and Zhu, Shixiang},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
~~~
