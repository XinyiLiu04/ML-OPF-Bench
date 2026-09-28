"""Public dataset and method-extension interfaces for ML-OPF-Bench."""

from .config import DatasetPaths, Experiment, dataset_paths
from .datasets import Dataset, DatasetSplit, Inputs, Partition, load_dataset
from .methods import Method, Postprocessor, Prediction, predict_method
from .evaluation import EvaluationResult, evaluate_predictions, evaluate_baseline
from .registry import create_method, list_methods, register_method

__version__ = "0.2.0"

__all__ = [
    "DatasetPaths", "Experiment", "dataset_paths", "Dataset", "DatasetSplit",
    "Inputs", "Partition", "load_dataset", "Method", "Postprocessor", "Prediction",
    "predict_method", "create_method", "list_methods", "register_method",
    "EvaluationResult", "evaluate_predictions", "evaluate_baseline", "run_experiment",
]


def run_experiment(spec, data_root, output_root):
    from .runner import run_experiment as run
    return run(spec, data_root, output_root)
