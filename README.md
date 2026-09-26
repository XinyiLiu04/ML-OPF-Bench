# ML-OPF-Bench
ML-OPF-Bench: Benchmarking Machine Learning for Optimal Power Flow

A benchmark of machine-learning methods for the AC and DC optimal power flow problems,
with a shared data pipeline and a shared evaluation per problem so that published numbers
are comparable across methods.

Within each problem, all method families are implemented against one interface. Each one
predicts generator setpoints and is scored with the same metrics: prediction error against
the reference OPF solution, constraint violation by category, and cost optimality gap.

Dataset: [`xinyi-liu/ML-OPF-Bench`](https://huggingface.co/datasets/xinyi-liu/ML-OPF-Bench)

## Installation

```bash
git clone https://github.com/XinyiLiu04/ML-OPF-Bench.git
cd ML-OPF-Bench
python -m pip install -e ".[test,plots]"
```

## Getting the data

Download the dataset repository, then point the code at it:

```bash
hf download xinyi-liu/ML-OPF-Bench --repo-type dataset \
  --local-dir ./ML-OPF-Bench-data

export ML_OPF_BENCH_DATA=$PWD/ML-OPF-Bench-data
```

`ML_OPF_BENCH_DATA` must point at the folder containing both `ac_dataset/` and
`dc_dataset/`; the AC and DC code read from the same root.

## Reproducing the benchmark

The package provides one entry point for cross-system, distribution-shift, and
training-size experiments. Dataset generation is separate and is not required.

```bash
ml-opf-bench plan --seed 42
ml-opf-bench run --formulation ac --method DNN --case case118 --seed 42
ml-opf-bench run --formulation dc --method KKT --mode scaling --train-size 1000
ml-opf-bench suite --seed 42 --data-root "$ML_OPF_BENCH_DATA" --output-root runs/paper-seed42
```

Cross-system experiments draw 12,000 samples and use a 10,000/1,000/1,000
train/validation/test split. The same split and seed are shared by all methods.
Scaling uses nested training subsets of 1K, 5K, 12K, 20K, 30K and 35K, with
validation and test each fixed at 10% of the full 118-bus dataset.

A cross-system 118-bus run also evaluates its frozen checkpoint under larger
load variance and heavier API loads. Larger variance uses the typical network
constraints. API evaluation reads its own constraints, while retaining the
training scalers and learned output parametrization. Set `--no-shifts` to run
only the in-distribution evaluation.

All scalers are fitted on training rows. AC GNN inputs are recomputed from the
current constraints and cached under `runs/derived_features/`, or the directory
specified by `ML_OPF_FEATURE_CACHE`. Existing sample and feature CSVs are retained.
Input construction (DCOPF followed by AC power flow) is included in GNN latency.
AC metrics evaluate the power-flow-verified dispatch and report nonconvergence
separately. Accuracy and violation means condition on successful power flows.

The manuscript widths and budgets are defined in `src/ml_opf_bench/config.py`.
NGT normalizes losses using first-epoch training losses; no OPF labels set its
training loss scale. RL uses physical action bounds and validation-reward early
stopping. Active-set methods recover candidates from their top three predictions. The AC
implementation ranks candidates within active-power bounds; full AC feasibility
is measured separately after power flow. The CP run records both raw and repaired outputs, so FR
and QP do not require a duplicate training run. Scaling includes KKT for both
formulations and excludes PR2 variants.

Every attempt gets a new directory containing the full configuration, source and
data hashes, software versions, splits, checkpoint, logs, aggregate metrics and
sample-level predictions. Failed attempts are retained. A suite resumes only
completed experiments with matching settings, source hashes, and base/OOD data
signatures. Checkpoint metadata records executed epochs or RL environment steps.

For a quick execution check, use `--epochs 1 --pool-size 120 --eval-limit 4
--device cpu`. These settings are smoke tests and must not populate paper tables.
The `--epochs` override also shortens the RL rollout and labeled-data budgets.

Legacy entry points remain available after installing the package:

```bash
python ac_methods/acopf_dnn.py
python dc_methods/dcopf_dnn.py
```

Their standalone settings remain in `ac_methods/ac_configuration/acopf_config.py`
and `dc_methods/dc_configuration/dcopf_config.py`.

## Adding a case

Add one line to `CASES` in the config of the problem. The folder names follow from the
short name:

```python
CASES = {
    'case57': {'full_name': 'pglib_opf_case57_ieee', 'short_name': 'case57'},
}
```
## License and citation