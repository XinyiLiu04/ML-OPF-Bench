# -*- coding: utf-8 -*-
"""
ACOPF DNN Unified Configuration File (V2 — M_C^D Replication)

Changes from V1:
- Removed HIDDEN_SIZES (architecture now parameterized by l, n, g per paper)
- Added LAGRANGIAN_LR (ρ step size)
- Added EVAL_GENERALIZATION, EVAL_API flags for multi-test evaluation
- Paper defaults: epochs=80, batch=64, lr=0.001, ρ=0.01

Usage:
1. Set ROOT_DIR to your dataset root
2. Set TRAIN_CASE and TRAIN_VARIANCE
3. Set DATA_MODE (random_split, fixed_valtest, generalization, api_test)
4. Set EVAL_GENERALIZATION / EVAL_API to True for multi-test evaluation
5. Run lagrangian_acopf_main_full.py
"""

import os

# =====================================================================
# Global Configuration
# =====================================================================
ROOT_DIR = "/lambda/nfs/lxy/acopf_project/data"

# =====================================================================
# Case Information Library
# =====================================================================
CASES = {
    'case118': {
        'full_name': 'pglib_opf_case118_ieee',
        'short_name': 'case118',
        'has_api_suffix': False,
    },
    'case118_api': {
        'full_name': 'pglib_opf_case118_ieee__api',
        'short_name': 'case118(api)',
        'has_api_suffix': True,
    },
    # Add more cases as needed
}

# =====================================================================
# Experiment Configuration
# =====================================================================

# ---- Data Mode ----
# Options: 'random_split', 'fixed_valtest', 'generalization', 'api_test'
DATA_MODE = 'fixed_valtest'

# ---- Training Data ----
TRAIN_CASE = 'case118'
TRAIN_VARIANCE = 'v=0.12'

# ---- Test Data (for generalization / api_test modes) ----
TEST_VARIANCE = 'v=0.25'      # GENERALIZATION mode
TEST_CASE = 'case118_api'     # API_TEST mode

# ---- Multi-Test Evaluation ----
# When True, after training, also evaluate on these additional test sets.
# Works with any DATA_MODE (the primary test is always from DATA_MODE).
EVAL_GENERALIZATION = False   # Also evaluate on TEST_VARIANCE data?
EVAL_API = False               # Also evaluate on TEST_CASE (API) data?

# ---- Training Parameters ----
N_TRAIN_USE = 35000            # random_split: total samples; fixed_valtest: train samples
N_TEST_SAMPLES = 1000          # Test samples for generalization / api_test
N_EPOCHS_MAX = 80              # Paper default: 80
EARLY_STOP_PATIENCE = None     # None = disabled (paper default: fixed epochs)
EARLY_STOP_MIN_DELTA = 1e-6
LEARNING_RATE = 1e-3           # Paper: α = 0.001
LAGRANGIAN_LR = 0.01           # Paper: ρ = 0.01
LAMBDA_MAX = 100.0             # Upper bound for λ clipping (prevents runaway)
BATCH_SIZE = 64                # Paper: b = 64
SEED = 42
DEVICE = 'cuda'


# =====================================================================
# Path Auto-generation (no need to modify)
# =====================================================================

def get_case_info(case_key):
    if case_key not in CASES:
        raise ValueError(f"Unknown case: {case_key}, options: {list(CASES.keys())}")
    return CASES[case_key]


def get_data_path(case_key, variance):
    case_info = get_case_info(case_key)
    if variance:
        folder_name = f"{case_info['short_name']}({variance})"
    else:
        folder_name = case_info['short_name']
    return os.path.join(ROOT_DIR, "ACOPF dataset", folder_name,
                        f"{case_info['full_name']}_pd.csv")


def get_params_path(case_key):
    case_info = get_case_info(case_key)
    return os.path.join(ROOT_DIR, "ACOPF Constraints", case_info['short_name'])


def get_result_folder():
    train_info = get_case_info(TRAIN_CASE)
    if DATA_MODE in ('random_split', 'fixed_valtest'):
        return f"{train_info['short_name']}({TRAIN_VARIANCE})"
    elif DATA_MODE == 'generalization':
        return f"{train_info['short_name']}_{TRAIN_VARIANCE}_to_{TEST_VARIANCE}"
    elif DATA_MODE == 'api_test':
        return f"{train_info['short_name']}_{TRAIN_VARIANCE}_to_api"
    else:
        raise ValueError(f"Unknown data mode: {DATA_MODE}")


def get_all_paths():
    train_info = get_case_info(TRAIN_CASE)
    paths = {
        'case_name': train_info['full_name'],
        'params_path': get_params_path(TRAIN_CASE),
        'data_path': get_data_path(TRAIN_CASE, TRAIN_VARIANCE),
    }

    if DATA_MODE == 'generalization':
        paths['test_data_path'] = get_data_path(TRAIN_CASE, TEST_VARIANCE)
        paths['test_params_path'] = None
    elif DATA_MODE == 'api_test':
        paths['test_data_path'] = get_data_path(TEST_CASE, None)
        paths['test_params_path'] = get_params_path(TEST_CASE)
    else:
        paths['test_data_path'] = None
        paths['test_params_path'] = None

    result_folder = get_result_folder()
    paths['log_path'] = os.path.join(ROOT_DIR, "Results", "ACOPF_DNN",
                                     result_folder, "training_log.csv")
    paths['results_path'] = os.path.join(ROOT_DIR, "Results", "ACOPF_DNN",
                                         result_folder, "results.json")
    return paths


def get_all_params():
    return {
        'data_mode': DATA_MODE,
        'n_train_use': N_TRAIN_USE,
        'n_test_samples': N_TEST_SAMPLES,
        'seed': SEED,
        'n_epochs': N_EPOCHS_MAX,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'early_stop_min_delta': EARLY_STOP_MIN_DELTA,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'device': DEVICE,
    }


def print_config():
    train_info = get_case_info(TRAIN_CASE)
    print("\n" + "=" * 70)
    print("Experiment Configuration (M_C^D)")
    print("=" * 70)
    print(f"Case: {train_info['full_name']}")
    print(f"Data Mode: {DATA_MODE}")
    print(f"Training Data: {TRAIN_VARIANCE}")
    if DATA_MODE == 'generalization':
        print(f"Test Data: {TEST_VARIANCE}")
    elif DATA_MODE == 'api_test':
        test_info = get_case_info(TEST_CASE)
        print(f"Test Data: API ({test_info['short_name']})")
    print(f"Eval Generalization: {EVAL_GENERALIZATION}")
    print(f"Eval API: {EVAL_API}")
    print(f"Samples: {N_TRAIN_USE} | Epochs: {N_EPOCHS_MAX}")
    print(f"LR: {LEARNING_RATE} | Lagrangian ρ: {LAGRANGIAN_LR}")
    print(f"Batch: {BATCH_SIZE} | Seed: {SEED} | Device: {DEVICE}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
    paths = get_all_paths()
    print("\nGenerated Paths:")
    for key, value in paths.items():
        print(f"  {key}: {value}")