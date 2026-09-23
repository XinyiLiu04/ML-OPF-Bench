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
git clone https://github.com/<user>/ML-OPF-Bench.git
cd ML-OPF-Bench
pip install -r requirements.txt
```

## Getting the data

Download the dataset repository, then point the code at it:

```bash
huggingface-cli download xinyi-liu/ML-OPF-Bench --repo-type dataset \
  --local-dir ./ML-OPF-Bench-data

export ML_OPF_BENCH_DATA=$PWD/ML-OPF-Bench-data
```

`ML_OPF_BENCH_DATA` must point at the folder containing both `ac_dataset/` and
`dc_dataset/`; the AC and DC code read from the same root.

## Running a method

Every method reads the configuration of its problem and is launched the same way:

```bash
python ac_methods/acopf_dnn.py
python dc_methods/dcopf_dnn.py
```

AC and DC each have one configuration file, `ac_methods/ac_configuration/acopf_config.py`
and `dc_methods/dc_configuration/dcopf_config.py`. The two are separate, since the
problems have different constraint schemas and metrics, but expose the same settings:

| Setting | Meaning |
| --- | --- |
| `CASE` | a key of `CASES`, e.g. `'case30'`, `'case118'` or `'case300'` |
| `VARIANCE` | load sampling spread, e.g. `'v=0.12'` |
| `N_TRAIN_USE` | how many samples to draw from the case before splitting |
| `SEED` | controls the split |
| `N_EPOCHS_MAX`, `EARLY_STOP_PATIENCE`, `LEARNING_RATE`, `BATCH_SIZE`, `HIDDEN_SIZES` | training budget |
| `DEVICE` | `'auto'`, `'cuda'`, `'mps'` or `'cpu'` |

## Adding a case

Add one line to `CASES` in the config of the problem. The folder names follow from the
short name:

```python
CASES = {
    'case57': {'full_name': 'pglib_opf_case57_ieee', 'short_name': 'case57'},
}
```
## License and citation