# ML-OPF-Bench

A Python package for AC/DC OPF datasets, ML baselines, and shared evaluation.

[Dataset](https://huggingface.co/datasets/xinyi-liu/ML-OPF-Bench) · [Project page](https://xinyiliu04.github.io/ml-opf-bench-webpage/)

## Install

Requires Python 3.11 or later.

```bash
python -m pip install git+https://github.com/XinyiLiu04/ML-OPF-Bench.git
```

## Get the data

```bash
hf download xinyi-liu/ML-OPF-Bench --repo-type dataset --local-dir ./ML-OPF-Bench-data
export ML_OPF_BENCH_DATA="$PWD/ML-OPF-Bench-data"
```

## Run

```bash
ml-opf-bench methods --formulation dc
ml-opf-bench run --formulation ac --method DNN --case case30 --device cpu
```

```python
from ml_opf_bench import Experiment, load_dataset, run_experiment

data = load_dataset("./ML-OPF-Bench-data", "dc", "case30")
split = data.split(seed=42)

spec = Experiment("dc", "LR", case="case30", device="cpu")
run_experiment(spec, "./ML-OPF-Bench-data", "./runs")
```

Each run saves its configuration, checkpoint, split indices, and evaluation results
in a new directory under `runs/`.

## Add a method

Implement `fit(train, validation, *, seed)` and `predict(inputs)`, returning a
`Prediction`. Register the class with `register_method("dc", "MY-METHOD", MyMethod)`
and pass its name to `Experiment`.

See [the minimal example](examples/custom_method.py) and [the interface](src/ml_opf_bench/methods.py).
Importable methods also work with `--plugin my_method:MyMethod --method MY-METHOD`.
