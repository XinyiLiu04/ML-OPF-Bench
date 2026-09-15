# -*- coding: utf-8 -*-
"""ACOPF configuration. Set ROOT_DIR, CASE and VARIANCE."""

import os

# =====================================================================
# Dataset location
# =====================================================================
# Dataset: https://huggingface.co/datasets/xinyi-liu/ML-OPF-Bench
# Point ROOT_DIR at the local copy of the dataset repository, i.e. the folder
# that contains ac_dataset/.
ROOT_DIR = "/lambda/nfs/lxy/acopf_project/ML-OPF-Bench"

DATA_SUBDIR = os.path.join("ac_dataset", "acopf_datasets")
CONSTRAINTS_SUBDIR = os.path.join("ac_dataset", "acopf_constraints")
DC_CONSTRAINTS_SUBDIR = os.path.join("dc_dataset", "dcopf_constraints")

# Expected directory layout:
#   ROOT_DIR/DATA_SUBDIR/<short_name>(<variance>)/<full_name>_{pd,qd,pg,qg,vm,va}.csv
#   ROOT_DIR/DATA_SUBDIR/<short_name>(<variance>)_with_duals/<full_name>_mu_*.csv
#   ROOT_DIR/CONSTRAINTS_SUBDIR/<short_name>/<full_name>_{bus_data,gen_data,branch_data,bus_gen_map,base_mva}.csv
#   ROOT_DIR/DC_CONSTRAINTS_SUBDIR/<short_name>/<full_name>_{ptdf_matrix,gen_limits,gen_costs,branch_limits,bus_gen_map,base_mva}.csv
# Only methods that learn from dual variables read the _with_duals folder, and only
# the sub-optimal state generator reads the DCOPF constraints.

# =====================================================================
# Case registry - add an entry here to support a new case
# =====================================================================
CASES = {
    'case30': {'full_name': 'pglib_opf_case30_ieee', 'short_name': 'case30'},
    'case118': {'full_name': 'pglib_opf_case118_ieee', 'short_name': 'case118'},
    'case300': {'full_name': 'pglib_opf_case300_ieee', 'short_name': 'case300'},
}

# =====================================================================
# Experiment configuration
# =====================================================================
CASE = 'case30'
VARIANCE = 'v=0.12'

N_TRAIN_USE = 12000
N_EPOCHS_MAX = 1000
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 1e-6
LEARNING_RATE = 1e-3
HIDDEN_SIZES = [64, 32]
BATCH_SIZE = 32  # None means full batch
SEED = 42
DEVICE = 'cuda'  # 'cuda' or 'cpu'


# =====================================================================
# Path generation
# =====================================================================
def get_case_info(case_key):
    """Look up a case entry in CASES."""
    if case_key not in CASES:
        raise ValueError(f"Unknown case: {case_key}, options: {list(CASES.keys())}")
    return CASES[case_key]


def get_data_path(case_key, variance):
    """Build the path to <case>_pd.csv; sibling files live in the same folder."""
    case_info = get_case_info(case_key)
    return os.path.join(
        ROOT_DIR,
        DATA_SUBDIR,
        f"{case_info['short_name']}({variance})",
        f"{case_info['full_name']}_pd.csv"
    )


def get_params_path(case_key):
    """Build the path to the folder holding the case constraint CSVs."""
    case_info = get_case_info(case_key)
    return os.path.join(ROOT_DIR, CONSTRAINTS_SUBDIR, case_info['short_name'])


def get_dc_params_path(case_key=None):
    """Build the path to the folder holding the DCOPF constraint CSVs."""
    case_info = get_case_info(case_key or CASE)
    return os.path.join(ROOT_DIR, DC_CONSTRAINTS_SUBDIR, case_info['short_name'])


def get_duals_path(case_key=None, variance=None):
    """Build the path to the folder holding the dual-variable CSVs."""
    case_info = get_case_info(case_key or CASE)
    return os.path.join(
        ROOT_DIR,
        DATA_SUBDIR,
        f"{case_info['short_name']}({variance or VARIANCE})_with_duals"
    )


def get_all_paths():
    """Return the paths consumed by the experiment entry point."""
    return {
        'case_name': get_case_info(CASE)['full_name'],
        'params_path': get_params_path(CASE),
        'data_path': get_data_path(CASE, VARIANCE),
    }


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
        'device': DEVICE,
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

    if os.path.exists(paths['data_path']):
        print(f"[OK] Data found: {paths['data_path']}")
    else:
        print(f"[MISSING] Data not found: {paths['data_path']}")

    if os.path.exists(paths['params_path']):
        print(f"[OK] Params found: {paths['params_path']}")
    else:
        print(f"[MISSING] Params not found: {paths['params_path']}")