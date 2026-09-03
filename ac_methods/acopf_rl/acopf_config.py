# -*- coding: utf-8 -*-
"""
ACOPF DNN Unified Configuration File

Usage:
1. Modify ROOT_DIR to your dataset root directory
2. Select experiment mode (RANDOM_SPLIT, FIXED_VALTEST, GENERALIZATION, API_TEST)
3. Modify corresponding configuration parameters based on mode
4. Run acopf_dnn_main.py, which will automatically read this configuration

Open Source Usage:
Other users only need to modify ROOT_DIR, other paths will be auto-generated
"""

import os

# =====================================================================
# Global Configuration - Users only need to modify here
# =====================================================================
ROOT_DIR = "/lambda/nfs/lxy/acopf_project/data"   # ← Modify to your dataset root directory

# =====================================================================
# Case Information Library
# =====================================================================
CASES = {
    # Standard cases
    'case300': {
        'full_name': 'pglib_opf_case300_ieee',  # Full case name
        'short_name': 'case300',  # Short name (for folders)
        'has_api_suffix': False  # Standard case without __api
    },

    # API cases (note double underscore __api)
    'case118_api': {
        'full_name': 'pglib_opf_case118_ieee__api',  # Full case name (with __api)
        'short_name': 'case118(api)',  # Short name (for folders)
        'has_api_suffix': True  # API case has __api
    },

    # You can add more cases
    # 'case300': {
    #     'full_name': 'pglib_opf_case300_ieee',
    #     'short_name': 'case300',
    #     'has_api_suffix': False
    # },
}

# =====================================================================
# Experiment Configuration - Modify these parameters to run different experiments
# =====================================================================

# -------------------- Data Mode --------------------
# Options: 'random_split', 'fixed_valtest', 'generalization', 'api_test', 'combined'
# 'combined': Train once, test on random_split + generalization + api_test (PINN only)
DATA_MODE = 'random_split'

# -------------------- Training Data Configuration --------------------
TRAIN_CASE = 'case300'  # Training case (select from CASES)
TRAIN_VARIANCE = 'v=0.12'  # Training data variance

# -------------------- Test Data Configuration (depends on mode) --------------------
# RANDOM_SPLIT / FIXED_VALTEST: Not needed, uses training data
# GENERALIZATION: Set TEST_VARIANCE
# API_TEST: Set TEST_CASE
# COMBINED: Set both TEST_VARIANCE and TEST_CASE

TEST_VARIANCE = 'v=0.25'  # Used in GENERALIZATION / COMBINED mode
TEST_CASE = 'case118_api'  # Used in API_TEST / COMBINED mode (select from CASES)

# -------------------- Common Training Parameters --------------------
N_TRAIN_USE = 12000  # random_split: total samples; others: training samples
N_TEST_SAMPLES = 1000 # Test samples for GENERALIZATION/API_TEST
N_EPOCHS_MAX = 100  # Maximum number of training epochs
EARLY_STOP_PATIENCE = 20  # Early stopping patience (epochs without improvement)
EARLY_STOP_MIN_DELTA = 1e-6  # Minimum improvement to reset patience counter
LEARNING_RATE = 3e-4  # Learning rate
HIDDEN_SIZES = [512, 256]      # DNN hidden layer structure (used by acopf_dnn_main.py)
BATCH_SIZE = 128  # Batch size, None means full batch
SEED = 42  # Random seed
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# -------------------- PINN-Specific Parameters (used by acopf_PINN_main.py) --------------------
# Three independent branches, each with its own hidden layer structure
# Paper Table II example: Case 39/118/162 uses G=[20,30,50], V=[20,30,50], Lm=[20,30,50]
# Adjust neuron counts per branch as needed
HIDDEN_SIZES_G = [512, 256]     # G branch (generation: pg_non_slack + qg)
HIDDEN_SIZES_V = [512, 256]    # V branch (voltage: Vr + Vi)
HIDDEN_SIZES_LG =  [512, 256]  # Lm branch (dual variables)

# PINN loss weights (Paper Eq. 24: Λ_P, Λ_V, Λ_L, Λ_ε)
LAMBDA_P = 10.0       # Weight for generation prediction loss (MAE_g)
LAMBDA_V = 10.0       # Weight for voltage prediction loss (MAE_v)
LAMBDA_L = 1e-3      # Weight for dual variable prediction loss (MAE_l)
LAMBDA_EPS = 1e-4    # Weight for KKT physics loss (MAE_ε)

# Collocation: fraction of training data used as collocation points (paper: 0.5)
COLLOCATION_RATIO = 0.5


# =====================================================================
# Path Auto-generation Functions (no need to modify)
# =====================================================================

def get_case_info(case_key):
    """Get case information"""
    if case_key not in CASES:
        raise ValueError(f"Unknown case: {case_key}, options: {list(CASES.keys())}")
    return CASES[case_key]


def get_data_path(case_key, variance):
    """
    Auto-generate data path

    Args:
        case_key: Case key (e.g. 'case118', 'case118_api')
        variance: Data variance (e.g. 'v=0.12'), can be None for API case

    Returns:
        Data file path (points to _pd.csv)
    """
    case_info = get_case_info(case_key)

    if variance:
        # Standard data: case118(v=0.12)
        folder_name = f"{case_info['short_name']}({variance})"
    else:
        # API data: case118(api)
        folder_name = case_info['short_name']

    data_path = os.path.join(
        ROOT_DIR,
        "ACOPF dataset",
        folder_name,
        f"{case_info['full_name']}_pd.csv"
    )

    return data_path


def get_params_path(case_key):
    """
    Auto-generate parameters path

    Args:
        case_key: Case key (e.g. 'case118', 'case118_api')

    Returns:
        Parameters folder path
    """
    case_info = get_case_info(case_key)

    params_path = os.path.join(
        ROOT_DIR,
        "ACOPF Constraints",
        case_info['short_name']
    )

    return params_path


def get_result_folder():
    """
    Auto-generate result folder name based on data mode

    Returns:
        Result folder name
    """
    train_info = get_case_info(TRAIN_CASE)

    if DATA_MODE == 'random_split' or DATA_MODE == 'fixed_valtest':
        # case118(v=0.12)
        return f"{train_info['short_name']}({TRAIN_VARIANCE})"

    elif DATA_MODE == 'generalization':
        # case118_v=0.12_to_v=0.25
        return f"{train_info['short_name']}_{TRAIN_VARIANCE}_to_{TEST_VARIANCE}"

    elif DATA_MODE == 'api_test':
        # case118_v=0.12_to_api
        return f"{train_info['short_name']}_{TRAIN_VARIANCE}_to_api"

    elif DATA_MODE == 'combined':
        # case118(v=0.12)_combined
        return f"{train_info['short_name']}({TRAIN_VARIANCE})_combined"

    else:
        raise ValueError(f"Unknown data mode: {DATA_MODE}")


def get_all_paths():
    """
    Get all path configurations

    Returns:
        Dictionary containing all paths
    """
    train_info = get_case_info(TRAIN_CASE)

    # Base paths
    paths = {
        'case_name': train_info['full_name'],
        'params_path': get_params_path(TRAIN_CASE),
        'data_path': get_data_path(TRAIN_CASE, TRAIN_VARIANCE),
    }

    # Test data paths (depends on mode)
    if DATA_MODE == 'generalization':
        paths['test_data_path'] = get_data_path(TRAIN_CASE, TEST_VARIANCE)
        paths['test_params_path'] = None  # Use training params

    elif DATA_MODE == 'api_test':
        paths['test_data_path'] = get_data_path(TEST_CASE, None)
        paths['test_params_path'] = get_params_path(TEST_CASE)

    elif DATA_MODE == 'combined':
        # Combined mode needs both generalization and api_test paths
        paths['test_data_path'] = get_data_path(TRAIN_CASE, TEST_VARIANCE)
        paths['test_params_path'] = get_params_path(TEST_CASE)

    else:  # random_split, fixed_valtest
        paths['test_data_path'] = None
        paths['test_params_path'] = None

    # Result save paths (not used, kept for compatibility)
    result_folder = get_result_folder()
    paths['log_path'] = os.path.join(ROOT_DIR, "Results", "ACOPF_DNN", result_folder, "training_log.csv")
    paths['results_path'] = os.path.join(ROOT_DIR, "Results", "ACOPF_DNN", result_folder, "results.json")

    return paths


def get_all_params():
    """
    Get all training parameters

    Returns:
        Dictionary containing all training parameters
    """
    return {
        'data_mode': DATA_MODE,
        'n_train_use': N_TRAIN_USE,
        'n_test_samples': N_TEST_SAMPLES,
        'seed': SEED,
        'n_epochs': N_EPOCHS_MAX,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'early_stop_min_delta': EARLY_STOP_MIN_DELTA,
        'learning_rate': LEARNING_RATE,
        'hidden_sizes': HIDDEN_SIZES,
        'batch_size': BATCH_SIZE,
        'device': DEVICE,
    }


def get_pinn_params():
    """
    Get PINN-specific parameters (used by acopf_PINN_main.py)

    Returns:
        Dictionary containing PINN-specific parameters
    """
    return {
        'hidden_sizes_V': HIDDEN_SIZES_V,
        'hidden_sizes_G': HIDDEN_SIZES_G,
        'hidden_sizes_Lg': HIDDEN_SIZES_LG,
        'lambda_P': LAMBDA_P,
        'lambda_V': LAMBDA_V,
        'lambda_L': LAMBDA_L,
        'lambda_eps': LAMBDA_EPS,
        'collocation_ratio': COLLOCATION_RATIO,
    }


def print_config():
    """
    Print current configuration (for debugging and confirmation)
    """
    train_info = get_case_info(TRAIN_CASE)

    print("\n" + "=" * 70)
    print("Experiment Configuration")
    print("=" * 70)
    print(f"Case: {train_info['full_name']}")
    print(f"Data Mode: {DATA_MODE}")
    print(f"Training Data: {TRAIN_VARIANCE}")

    if DATA_MODE == 'generalization':
        print(f"Test Data: {TEST_VARIANCE}")
    elif DATA_MODE == 'api_test':
        test_info = get_case_info(TEST_CASE)
        print(f"Test Data: API ({test_info['short_name']})")
        print(f"Test Samples: {N_TEST_SAMPLES}")
    elif DATA_MODE == 'combined':
        test_info = get_case_info(TEST_CASE)
        print(f"Generalization Test: {TEST_VARIANCE}")
        print(f"API Test: {test_info['short_name']}")
        print(f"Test Samples: {N_TEST_SAMPLES}")

    print(f"Training Samples: {N_TRAIN_USE}")
    print(f"Max Epochs: {N_EPOCHS_MAX}")
    print(f"Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"Early Stop Min Delta: {EARLY_STOP_MIN_DELTA}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"DNN Hidden Layers: {HIDDEN_SIZES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Random Seed: {SEED}")
    print(f"Device: {DEVICE}")

    print(f"\nPINN Parameters:")
    print(f"  G Network:  {HIDDEN_SIZES_G}")
    print(f"  V Network:  {HIDDEN_SIZES_V}")
    print(f"  Lg Network: {HIDDEN_SIZES_LG}")
    print(f"  Λ_P={LAMBDA_P}, Λ_V={LAMBDA_V}, Λ_L={LAMBDA_L}, Λ_ε={LAMBDA_EPS}")
    print(f"  Collocation Ratio: {COLLOCATION_RATIO}")
    print("=" * 70)

    print("\nGenerated Paths:")
    paths = get_all_paths()
    for key, value in paths.items():
        print(f"  {key}: {value}")
    print("=" * 70)


# =====================================================================
# Main Function Test (optional)
# =====================================================================
if __name__ == "__main__":
    # Test configuration
    print_config()

    # Verify paths exist
    print("\nPath Verification:")
    paths = get_all_paths()

    # Check training data
    if os.path.exists(paths['data_path']):
        print(f"✓ Training data exists: {paths['data_path']}")
    else:
        print(f"✗ Training data not found: {paths['data_path']}")

    # Check training parameters
    if os.path.exists(paths['params_path']):
        print(f"✓ Training params exist: {paths['params_path']}")
    else:
        print(f"✗ Training params not found: {paths['params_path']}")

    # Check test data
    if paths['test_data_path']:
        if os.path.exists(paths['test_data_path']):
            print(f"✓ Test data exists: {paths['test_data_path']}")
        else:
            print(f"✗ Test data not found: {paths['test_data_path']}")

    # Check test parameters
    if paths['test_params_path']:
        if os.path.exists(paths['test_params_path']):
            print(f"✓ Test params exist: {paths['test_params_path']}")
        else:
            print(f"✗ Test params not found: {paths['test_params_path']}")