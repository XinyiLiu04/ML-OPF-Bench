# ML-OPF-Bench

ML-OPF-Bench is a comprehensive benchmark for machine learning methods for AC and DC optimal power flow (OPF), bringing together OPF datasets and a broad collection of method implementations. Through systematic comparisons across power systems, training data sizes, and load distributions, it examines the trade-offs among prediction accuracy, constraint satisfaction, generation cost, and computational efficiency. These comparisons reveal insights into the strengths and limitations of existing approaches, offering guidance for the development and practical application of future learning-based OPF methods. The accompanying Python package provides an extensible framework for reproducible evaluation, making it easy to integrate new methods and compare them with existing baselines.

[![ML-OPF-Bench workflow](docs/images/workflow.png?v=20260928)](docs/images/workflow.pdf)

## Install

Requires Python 3.11 or later.

```bash
python -m pip install git+https://github.com/XinyiLiu04/ML-OPF-Bench.git
```

## Get the data 

([Dataset](https://huggingface.co/datasets/xinyi-liu/ML-OPF-Bench))

```bash
hf download xinyi-liu/ML-OPF-Bench --repo-type dataset --local-dir ./ML-OPF-Bench-data
export ML_OPF_BENCH_DATA="$PWD/ML-OPF-Bench-data"
```

## Run

**Command line**

```bash
# List available AC methods
ml-opf-bench methods --formulation ac
# Run the AC DNN baseline on case30
ml-opf-bench run --formulation ac --method DNN --case case30 --device cpu
```

**Python API**

```python
from ml_opf_bench import Experiment, run_experiment

# Run the same AC DNN baseline on case30
spec = Experiment("ac", "DNN", case="case30", device="cpu")
run_experiment(spec, "./ML-OPF-Bench-data", "./runs")
```

Each run saves its configuration, checkpoint, split indices, and evaluation results
in a new directory under `runs/`.

## Add a method

ML-OPF-Bench is designed to support new methods through a shared interface. We welcome researchers and developers to implement their own approaches and evaluate them within the benchmark.

Implement `fit(train, validation, *, seed)` and `predict(inputs)`, returning a
`Prediction`. Register the class with `register_method("dc", "MY-METHOD", MyMethod)`
and pass its name to `Experiment`.

See [the minimal example](examples/custom_method.py) and [the interface](src/ml_opf_bench/methods.py).
Importable methods also work with `--plugin my_method:MyMethod --method MY-METHOD`.
