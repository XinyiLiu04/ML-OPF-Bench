# ML-OPF-Bench

A shared training and evaluation pipeline for machine-learning methods for AC and DC optimal power flow.

[Dataset](https://huggingface.co/datasets/xinyi-liu/ML-OPF-Bench) · [Project page](https://xinyiliu04.github.io/ml-opf-bench-webpage/)

## Installation

```bash
git clone https://github.com/XinyiLiu04/ML-OPF-Bench.git
cd ML-OPF-Bench
python -m pip install -e ".[test,plots]"
```

## Data

```bash
hf download xinyi-liu/ML-OPF-Bench --repo-type dataset --local-dir ./ML-OPF-Bench-data
export ML_OPF_BENCH_DATA="$PWD/ML-OPF-Bench-data"
```

The data root contains `ac_dataset/` and `dc_dataset/`. Existing OPF labels are provided; regeneration is unnecessary.

## Reproducing the benchmark

```bash
# Run one method (118-bus runs also evaluate distribution shifts).
ml-opf-bench run --formulation ac --method DNN --case case118 --seed 42

# Inspect and run the standard experiment suite.
ml-opf-bench plan --seed 42
ml-opf-bench suite --seed 42 --output-root runs/benchmark
```

Use `--help` for options and `--device cpu` without a GPU. Each run saves its configuration, data/source hashes, split, checkpoint, logs, and metrics.

## Repository layout

- `ac_methods/`, `dc_methods/`: models and physical-system utilities.
- `src/ml_opf_bench/`: experiment runner and evaluation pipeline.
- `data_generalization/`: optional Julia dataset-generation notebooks.
- `reports/`, `tests/`: result processing, audits, and checks.
