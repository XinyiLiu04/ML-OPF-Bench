# -*- coding: utf-8 -*-
"""DCOPF configuration. Set ROOT_DIR, CASE and VARIANCE."""

import os

# =====================================================================
# Dataset location
# =====================================================================
# Dataset: https://huggingface.co/datasets/xinyi-liu/ML-OPF-Bench
# Point this at the local copy of the dataset repository, i.e. the folder that
# contains both ac_dataset/ and dc_dataset/. The environment variable wins when set:
#   export ML_OPF_BENCH_DATA=/path/to/ML-OPF-Bench
# `or` rather than a get default, so an environment variable set to an empty
# string does not silently turn every path into a relative one
ROOT_DIR = (os.environ.get("ML_OPF_BENCH_DATA")
            or "/Users/xinyiliu/Projects/ml-opf-bench/ML-OPF-Bench")

DATA_SUBDIR = os.path.join("dc_dataset", "dcopf_datasets")
CONSTRAINTS_SUBDIR = os.path.join("dc_dataset", "dcopf_constraints")

# Expected directory layout:
#   ROOT_DIR/DATA_SUBDIR/<short_name>(<variance>)/<full_name>_dataset_with_duals.csv
#   ROOT_DIR/CONSTRAINTS_SUBDIR/<short_name>/<full_name>_{gen_limits,gen_costs,branch_limits,
#                                             ptdf_matrix,bus_gen_map,bus_ids,base_mva}.csv

# =====================================================================
# Case registry - add an entry here to support a new case
# =====================================================================
CASES = {
    'case14': {'full_name': 'pglib_opf_case14_ieee', 'short_name': 'case14'},
    'case30': {'full_name': 'pglib_opf_case30_ieee', 'short_name': 'case30'},
    'case118': {'full_name': 'pglib_opf_case118_ieee', 'short_name': 'case118'},
    'case300': {'full_name': 'pglib_opf_case300_ieee', 'short_name': 'case300'},
}

# =====================================================================
# Experiment configuration
# =====================================================================
CASE = 'case30'
VARIANCE = 'v=0.12'

# Changing SEED or N_TRAIN_USE changes the test set and invalidates every collected number
N_TRAIN_USE = 35000  # size of the shuffled pool that is split 10:1:1 into train/val/test
N_EPOCHS_MAX = 1000
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 1e-6
LEARNING_RATE = 1e-3
HIDDEN_SIZES = [128, 64]
BATCH_SIZE = 64
SEED = 42
DEVICE = 'auto'  # 'auto', 'cuda', 'mps' or 'cpu'


# =====================================================================
# Path generation
# =====================================================================
def get_case_info(case_key):
    """Look up a case entry in CASES."""
    if case_key not in CASES:
        raise ValueError(f"Unknown case: {case_key}, options: {list(CASES.keys())}")
    return CASES[case_key]


def get_data_path(case_key, variance):
    """Build the path to the single sample CSV, which also carries the duals."""
    case_info = get_case_info(case_key)
    return os.path.join(ROOT_DIR, DATA_SUBDIR, f"{case_info['short_name']}({variance})",
                        f"{case_info['full_name']}_dataset_with_duals.csv")


def get_params_path(case_key):
    """Build the path to the folder holding the case constraint CSVs."""
    return os.path.join(ROOT_DIR, CONSTRAINTS_SUBDIR, get_case_info(case_key)['short_name'])


def get_all_paths():
    """Return the paths consumed by the experiment entry point."""
    return {
        'case_name': get_case_info(CASE)['full_name'],
        'params_path': get_params_path(CASE),
        'data_path': get_data_path(CASE, VARIANCE),
    }


def resolve_device(requested=None):
    """Map 'auto' to cuda -> mps -> cpu; an unavailable explicit choice falls back to cpu with a warning."""
    import torch

    requested = requested or DEVICE
    available = {
        'cuda': torch.cuda.is_available(),
        'mps': getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available(),
        'cpu': True,
    }
    if requested == 'auto':
        return next(d for d in ('cuda', 'mps', 'cpu') if available[d])
    if requested not in available:
        raise ValueError(f"Unknown device '{requested}', expected one of auto/cuda/mps/cpu")
    if not available[requested]:
        print(f"Warning: {requested.upper()} requested but unavailable, falling back to CPU")
        return 'cpu'
    return requested


def synchronize(device):
    """Block until queued kernels finish, so wall-clock timings cover the actual computation."""
    import torch

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def get_all_params():
    """Return the training hyperparameters consumed by the experiment entry point."""
    return {
        'n_train_use': N_TRAIN_USE,
        'seed': SEED,
        'n_epochs': N_EPOCHS_MAX,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'early_stop_min_delta': EARLY_STOP_MIN_DELTA,
        'learning_rate': LEARNING_RATE,
        'hidden_sizes': HIDDEN_SIZES,
        'batch_size': BATCH_SIZE,
        'device': resolve_device(),
    }


def print_config():
    """Print the resolved configuration and paths."""
    print("\n" + "=" * 70)
    print("Experiment Configuration")
    print("=" * 70)
    print(f"Case: {get_case_info(CASE)['full_name']}")
    print(f"Variance: {VARIANCE}")
    print(f"Training Samples: {N_TRAIN_USE}")
    print(f"Max Epochs: {N_EPOCHS_MAX}")
    print(f"Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"Early Stop Min Delta: {EARLY_STOP_MIN_DELTA}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Hidden Layers: {HIDDEN_SIZES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Random Seed: {SEED}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    print("\nGenerated Paths:")
    for key, value in get_all_paths().items():
        print(f"  {key}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()

    print("\nPath Verification:")
    paths = get_all_paths()
    for key in ('data_path', 'params_path'):
        status = "OK" if os.path.exists(paths[key]) else "MISSING"
        print(f"[{status}] {key}: {paths[key]}")