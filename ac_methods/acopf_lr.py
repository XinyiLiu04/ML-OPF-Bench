# -*- coding: utf-8 -*-
"""
Linear Regression for AC-OPF (V8 - Simplified Output)

Predicts Pg + Vm, aligned with DNN/PINN methods

Modifications (V8):
- DNN only predicts non-slack pg and generator vm
- Simplified output: only test set metrics
- All comments and outputs in English
- Removed all file saving (JSON/CSV/model)
- All violations in p.u. units
"""

import numpy as np
import pandas as pd
import time
import os
import sys
from sklearn.linear_model import LinearRegression
from pathlib import Path

# Import configuration
try:
    import acopf_config
except ImportError:
    print("Error: Unable to import acopf_config.py")
    print("Please ensure acopf_config.py is in the same directory")
    sys.exit(1)

# Import data modules
try:
    from acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        DataMode,
        prepare_data_splits,
        load_generalization_test_data,
        load_api_test_data,
        reconstruct_full_pg  # Import reconstruction function
    )
except ImportError:
    print("Error: Unable to import 'acopf_data_setup' module.")
    sys.exit(1)

# Import evaluation modules
try:
    from acopf_violation_metrics import evaluate_acopf_predictions
except ImportError:
    print("Error: Unable to import 'acopf_violation_metrics' module.")
    sys.exit(1)

# PyPower
from pypower.runpf import runpf
from pypower.ppoption import ppoption

# =====================================================================
# Global Variables
# =====================================================================
GLOBAL_CASE_DATA = None
PPOPT = None


# =====================================================================
# PyPower Interface
# =====================================================================
def init_pypower_options():
    """Initialize PyPower options"""
    global PPOPT
    ppopt = ppoption()
    PPOPT = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)


def load_case_from_csv(case_name, constraints_path):
    """
    Load PyPower case data from CSV files

    Note: case_name should be the full case name (including __api suffix)
    """
    base_path = Path(constraints_path)

    base_mva_df = pd.read_csv(base_path / f"{case_name}_base_mva.csv")
    bus_df = pd.read_csv(base_path / f"{case_name}_bus_data.csv")
    gen_df = pd.read_csv(base_path / f"{case_name}_gen_data.csv")
    branch_df = pd.read_csv(base_path / f"{case_name}_branch_data.csv")

    baseMVA = base_mva_df['value'].iloc[0]

    # BUS Matrix
    bus = np.zeros((len(bus_df), 13))
    bus[:, 0] = bus_df['bus_id'].values
    bus[:, 1] = bus_df['type'].values
    bus[:, 2] = bus_df['pd_pu'].values * baseMVA
    bus[:, 3] = bus_df['qd_pu'].values * baseMVA
    bus[:, 6] = 1
    bus[:, 7] = bus_df['vm_pu'].values
    bus[:, 8] = bus_df['va_deg'].values
    bus[:, 9] = bus_df['base_kv'].values if 'base_kv' in bus_df.columns else 1.0
    bus[:, 10] = 1
    bus[:, 11] = bus_df['vmax_pu'].values
    bus[:, 12] = bus_df['vmin_pu'].values

    # GEN Matrix
    gen = np.zeros((len(gen_df), 21))
    gen[:, 0] = gen_df['bus_id'].values
    gen[:, 3] = gen_df['qg_max_pu'].values * baseMVA
    gen[:, 4] = gen_df['qg_min_pu'].values * baseMVA
    gen[:, 5] = gen_df['vg_pu'].values
    gen[:, 6] = baseMVA
    gen[:, 7] = 1
    gen[:, 8] = gen_df['pg_max_pu'].values * baseMVA
    gen[:, 9] = gen_df['pg_min_pu'].values * baseMVA

    # BRANCH Matrix
    branch = np.zeros((len(branch_df), 13))
    branch[:, 0] = branch_df['f_bus'].values
    branch[:, 1] = branch_df['t_bus'].values
    branch[:, 2] = branch_df['r_pu'].values
    branch[:, 3] = branch_df['x_pu'].values
    branch[:, 4] = branch_df['b_pu'].values
    branch[:, 5] = branch_df['rate_a_pu'].values * baseMVA
    branch[:, 6] = branch[:, 5]
    branch[:, 7] = branch[:, 5]
    branch[:, 8] = branch_df['tap_ratio'].values
    branch[:, 9] = branch_df['shift_deg'].values
    branch[:, 10] = 1
    branch[:, 11] = -360
    branch[:, 12] = 360

    rate_a_values = branch_df['rate_a_pu'].values
    branch[:, 5:8][np.isnan(rate_a_values) | np.isinf(rate_a_values), :] = 9900.0

    # GENCOST Matrix
    gencost = np.zeros((len(gen_df), 7))
    gencost[:, 0] = 2
    gencost[:, 3] = 3
    gencost[:, 4] = gen_df['cost_c2'].values
    gencost[:, 5] = gen_df['cost_c1'].values
    gencost[:, 6] = gen_df['cost_c0'].values

    ppc = {
        'version': '2',
        'baseMVA': baseMVA,
        'bus': bus,
        'gen': gen,
        'branch': branch,
        'gencost': gencost
    }

    return ppc


def solve_pf_custom_optimized(pd, qd, pg_non_slack, vm_gen, params):
    """
    Run power flow using predicted Pg + Vm

    Args:
        pd: (n_loads,) - Load active power (p.u.)
        qd: (n_loads,) - Load reactive power (p.u.)
        pg_non_slack: (n_gen_non_slack,) - Predicted non-slack generator active power (p.u.)
        vm_gen: (n_gen,) - Predicted generator bus voltage (p.u.)
    """
    global GLOBAL_CASE_DATA, PPOPT

    BASE_MVA = params['general']['BASE_MVA']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    n_gen = params['general']['n_gen']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    load_bus_ids = params['general']['load_bus_ids']

    mpc_pf = {
        'version': GLOBAL_CASE_DATA['version'],
        'baseMVA': GLOBAL_CASE_DATA['baseMVA'],
        'bus': GLOBAL_CASE_DATA['bus'].copy(),
        'gen': GLOBAL_CASE_DATA['gen'].copy(),
        'branch': GLOBAL_CASE_DATA['branch'],
        'gencost': GLOBAL_CASE_DATA['gencost']
    }

    # Set loads (using load_bus_ids mapping)
    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc_pf["bus"][bus_idx, 2] = pd[i] * BASE_MVA
            mpc_pf["bus"][bus_idx, 3] = qd[i] * BASE_MVA

    # Set generator active power (only non-slack generators)
    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc_pf["gen"][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    # Set generator voltage
    for i in range(n_gen):
        mpc_pf["gen"][i, 5] = vm_gen[i]

    return runpf(mpc_pf, PPOPT)


# =====================================================================
# Linear Regression Model Class
# =====================================================================
class LinearRegressionACOPF:
    """
    Linear Regression predicts Pg + Vm

    Architecture:
    - n_gen_non_slack models predict Pg_non_slack
    - n_gen models predict Vm_gen
    """

    def __init__(self, n_gen_non_slack, n_gen):
        self.n_gen_non_slack = n_gen_non_slack
        self.n_gen = n_gen

        self.pg_models = [LinearRegression() for _ in range(n_gen_non_slack)]
        self.vm_models = [LinearRegression() for _ in range(n_gen)]

        self.is_fitted = False

    def fit(self, X_train, y_pg_non_slack_train, y_vm_gen_train):
        """Train models"""
        print(f"\nTraining Linear Regression models...")
        print(f"  - {self.n_gen_non_slack} Pg models (non-slack)")
        print(f"  - {self.n_gen} Vm models (generator buses)")

        t_start = time.time()

        for gen_idx in range(self.n_gen_non_slack):
            self.pg_models[gen_idx].fit(X_train, y_pg_non_slack_train[:, gen_idx])
            if (gen_idx + 1) % 20 == 0 or gen_idx == self.n_gen_non_slack - 1:
                print(f"    Trained {gen_idx + 1}/{self.n_gen_non_slack} Pg models")

        for gen_idx in range(self.n_gen):
            self.vm_models[gen_idx].fit(X_train, y_vm_gen_train[:, gen_idx])
            if (gen_idx + 1) % 20 == 0 or gen_idx == self.n_gen - 1:
                print(f"    Trained {gen_idx + 1}/{self.n_gen} Vm models")

        t_train = time.time() - t_start
        self.is_fitted = True

        print(f"✓ Training completed in {t_train:.2f} seconds")
        return t_train

    def predict(self, X):
        """Predict Pg_non_slack and Vm_gen"""
        if not self.is_fitted:
            raise ValueError("Model not trained!")

        n_samples = X.shape[0]
        y_pg_non_slack_pred = np.zeros((n_samples, self.n_gen_non_slack))
        y_vm_gen_pred = np.zeros((n_samples, self.n_gen))

        for gen_idx in range(self.n_gen_non_slack):
            y_pg_non_slack_pred[:, gen_idx] = self.pg_models[gen_idx].predict(X)

        for gen_idx in range(self.n_gen):
            y_vm_gen_pred[:, gen_idx] = self.vm_models[gen_idx].predict(X)

        return y_pg_non_slack_pred, y_vm_gen_pred


# =====================================================================
# Evaluation Function
# =====================================================================
def evaluate_model(model, X, indices, raw_data, params, scalers, split_name, verbose=True):
    """
    Evaluate model - calls acopf_violation_metrics.evaluate_acopf_predictions
    """
    if verbose:
        print(f"\n{split_name} Evaluation:")

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Predict
    y_pred_pg_non_slack, y_pred_vm_gen = model.predict(X)

    # Reconstruct full arrays
    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)

    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_all = np.zeros((len(X), n_buses), dtype=y_pred_vm_gen.dtype)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen
    non_gen_mask = np.ones(n_buses, dtype=bool)
    non_gen_mask[gen_bus_indices] = False
    y_pred_vm_all[:, non_gen_mask] = 1.0

    # True values
    y_true_pg = raw_data['pg'][indices]
    y_true_vm = raw_data['vm'][indices]
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

    # Loads
    x_raw_data = scalers['x'].inverse_transform(X)
    pd_pu = x_raw_data[:, :n_loads]
    qd_pu = x_raw_data[:, n_loads:]

    # Run power flow calculation
    n_samples = len(X)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_custom_optimized(pd_pu[i], qd_pu[i], y_pred_pg_non_slack[i], y_pred_vm_gen[i], params)
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except:
            pf_results_list.append(({'success': False, 'gen': np.zeros((n_gen, 21)),
                                     'bus': np.zeros((n_buses, 13)), 'branch': np.zeros((1, 17))},))
            converge_flags.append(False)

    if verbose:
        print(f"    ✓ Converged: {sum(converge_flags)}/{n_samples}")

    # Call unified evaluation function
    metrics = evaluate_acopf_predictions(
        y_pred_pg=y_pred_pg_full,
        y_pred_vm=y_pred_vm_all,
        y_true_pg=y_true_pg,
        y_true_vm=y_true_vm,
        y_true_qg=y_true_qg,
        y_true_va_rad=y_true_va_rad,
        pf_results_list=pf_results_list,
        converge_flags=converge_flags,
        params=params,
        verbose=verbose
    )

    return metrics


# =====================================================================
# Main Experiment Function
# =====================================================================
def linear_regression_experiment(
        case_name,
        params_path,
        data_path,
        log_path,
        results_path,
        # Data mode parameters
        data_mode='random_split',
        n_train_use=None,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=None,
        # Other
        seed=42,
        device='cuda',  # For interface consistency, but Linear Regression doesn't need it
        **kwargs  # Accept other unnecessary parameters (like n_epochs, learning_rate, early_stop_*, etc.)
):
    """
    Linear Regression AC-OPF experiment

    Supports four data modes:
    - RANDOM_SPLIT: Random split
    - FIXED_VALTEST: Fixed validation/test sets
    - GENERALIZATION: Cross-distribution generalization test
    - API_TEST: API data test (different topology)
    """
    global GLOBAL_CASE_DATA, PPOPT

    np.random.seed(seed)

    print(f"\n{'=' * 70}")
    print(f"Linear Regression for AC-OPF")
    print(f"{'=' * 70}")
    print(f"Case: {case_name}")
    print(f"Data Mode: {data_mode}")
    print(f"{'=' * 70}")

    # ========================================================================
    # 1. Load training parameters and PyPower case data
    # ========================================================================
    print(f"\n[Step 1] Loading training parameters and case data...")
    init_pypower_options()
    params = load_parameters_from_csv(case_name, params_path)
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)
    print(f"  ✓ Training params and PyPower case data loaded")

    # ========================================================================
    # 2. Load training data and fit scalers
    # ========================================================================
    print(f"\n[Step 2] Loading training data...")
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_loads = params['general']['n_loads']
    baseMVA = params['general']['BASE_MVA']

    print(f"\n[Training Data Info]")
    print(
        f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_non_slack}), Loads: {n_loads}, Base MVA: {baseMVA}")
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # ========================================================================
    # 3. Load and split data based on data mode
    # ========================================================================
    if data_mode == DataMode.API_TEST:
        print(f"\n{'=' * 60}")
        print(f"Data Mode: API_TEST")
        print(f"{'=' * 60}")

        if test_data_path is None or test_params_path is None:
            raise ValueError("API_TEST mode requires test_data_path and test_params_path")

        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=DataMode.API_TEST,
            n_train_use=n_train_use,
            seed=seed
        )

        test_params, test_x_scaled, test_y_scaled, test_raw_data, _ = \
            load_api_test_data(
                test_data_path,
                test_params_path,
                scalers,
                n_test_samples=n_test_samples or 1000,
                seed=seed
            )

        test_idx = np.arange(len(test_x_scaled))

        test_case_name = os.path.basename(test_data_path)
        if test_case_name.endswith('_pd.csv'):
            test_case_name = test_case_name[:-7]

        GLOBAL_CASE_DATA_TEST = load_case_from_csv(test_case_name, test_params_path)
        print(f"  ✓ API test PyPower case data loaded")

        print(f"\n[API Test Data Info]")
        print(f"  Buses: {test_params['general']['n_buses']}")
        print(
            f"  Generators: {test_params['general']['n_gen']} (Non-Slack: {test_params['general']['n_gen_non_slack']})")
        print(f"  Loads: {test_params['general']['n_loads']}")
        print(f"  Base MVA: {test_params['general']['BASE_MVA']}")

    elif data_mode == DataMode.GENERALIZATION:
        print(f"\n{'=' * 60}")
        print(f"Data Mode: GENERALIZATION")
        print(f"{'=' * 60}")

        if test_data_path is None:
            raise ValueError("GENERALIZATION mode requires test_data_path")

        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=DataMode.GENERALIZATION,
            n_train_use=n_train_use,
            seed=seed
        )

        test_x_scaled, test_y_scaled, test_raw_data, _ = \
            load_generalization_test_data(
                test_data_path,
                params,
                scalers,
                n_test_samples=n_test_samples or 1000,
                seed=seed
            )

        test_idx = np.arange(len(test_x_scaled))
        test_params = params
        GLOBAL_CASE_DATA_TEST = GLOBAL_CASE_DATA

    else:
        print(f"\n{'=' * 60}")
        print(f"Data Mode: {data_mode}")
        print(f"{'=' * 60}")

        train_idx, val_idx, test_idx = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=data_mode,
            n_train_use=n_train_use,
            seed=seed
        )

        test_x_scaled, test_y_scaled, test_raw_data = x_data_scaled, y_data_scaled, raw_data
        test_params = params
        GLOBAL_CASE_DATA_TEST = GLOBAL_CASE_DATA

    # ========================================================================
    # 4. Prepare training data
    # ========================================================================
    print(f"\n[Dataset Sizes]")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    X_train = x_data_scaled[train_idx]
    X_test = test_x_scaled[test_idx]

    y_pg_non_slack_train = raw_data['pg_non_slack'][train_idx]
    y_vm_gen_train = raw_data['vm_gen'][train_idx]

    # ========================================================================
    # 5. Create and train model
    # ========================================================================
    print(f"\n[Step 3] Creating and training Linear Regression models...")
    print(f"  Total: {n_gen_non_slack + n_gen} linear models")

    model = LinearRegressionACOPF(n_gen_non_slack, n_gen)
    train_time = model.fit(X_train, y_pg_non_slack_train, y_vm_gen_train)

    # ========================================================================
    # 6. Model evaluation (using test params)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    GLOBAL_CASE_DATA_BACKUP = GLOBAL_CASE_DATA
    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_TEST

    if data_mode == DataMode.API_TEST:
        test_split_name = "API Test"
    elif data_mode == DataMode.GENERALIZATION:
        test_split_name = "Generalization Test"
    else:
        test_split_name = "Test"

    test_metrics = evaluate_model(
        model, X_test, test_idx, test_raw_data,
        test_params,
        scalers, test_split_name, verbose=True
    )

    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_BACKUP

    # ========================================================================
    # 7. Inference speed test
    # ========================================================================
    times = []
    for _ in range(100):
        t_start = time.perf_counter()
        _ = model.predict(X_test[:1])
        times.append(time.perf_counter() - t_start)

    inference_time_ms = np.mean(times) * 1000

    # ========================================================================
    # 8. Print final results (simplified)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Final Results Summary")
    print(f"{'=' * 70}")

    print(f"\nData Mode: {data_mode}")
    print(f"Test Case: {case_name}")

    print(f"\n--- Accuracy Metrics ---")
    print(f"MAE_Pg (Non-Slack): {test_metrics['mae_pg_non_slack_percent']:.4f}%")
    print(f"MAE_Vm (Generator): {test_metrics['mae_vm_percent']:.4f}%")
    print(f"MAE_Qg (All Gens):  {test_metrics['mae_qg_percent']:.4f}%")
    print(f"MAE_Va (All Buses): {test_metrics['mae_va_deg']:.4f} degrees")

    print(f"\n--- Violations (p.u.) ---")
    print(f"Pg_viol (Non-Slack): {test_metrics['mean_pg_viol_non_slack_pu']:.6f} p.u.")
    print(f"Pg_viol (Slack):     {test_metrics['mean_pg_viol_slack_pu']:.6f} p.u.")
    print(f"Qg_viol (All Gens):  {test_metrics['mean_max_qg_viol_pu']:.6f} p.u.")
    print(f"Vm_viol (All Buses): {test_metrics['mean_max_vm_viol_pu']:.6f} p.u.")
    print(f"Branch_viol:         {test_metrics['mean_max_branch_viol_pu']:.6f} p.u. (1.0 = 100% overload)")

    print(f"\n--- Cost Metrics ---")
    print(f"Cost Gap: {test_metrics['cost_optimality_gap_percent']:.4f}%")

    print(f"\n--- Performance ---")
    print(f"Inference Time: {inference_time_ms:.4f} ms/sample")
    print(f"Training Time:  {train_time:.2f} s")
    print(f"Convergence Rate: {test_metrics['convergence_rate_percent']:.2f}%")

    print(f"{'=' * 70}")

    # No file saving - removed all JSON saving

    return test_metrics


# =====================================================================
# Main Function
# =====================================================================
if __name__ == "__main__":
    # =========================================================================
    # Read configuration from acopf_config.py and run experiment
    # =========================================================================

    # Print configuration
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    # Get all paths
    paths = acopf_config.get_all_paths()

    # Get all training parameters
    params = acopf_config.get_all_params()

    print(f"\n[Linear Regression Note]")
    print(f"  Linear Regression doesn't need iterative training, fit() is done once")
    print(f"  Therefore, n_epochs, early_stop_patience, learning_rate, etc. are not used for this method")
    print("=" * 70)

    # Execute experiment
    results = linear_regression_experiment(**paths, **params)

    print("\n✓ Experiment completed successfully!")