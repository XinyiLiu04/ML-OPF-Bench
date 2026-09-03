# -*- coding: utf-8 -*-
"""
Linear Regression for DCOPF - Main Experiment Script
Simple Baseline Method

Version: v5.1 - Fixed virtual bus handling for case300

Fixes in v5.1:
- BUG FIX 1: load_and_prepare_lr_data now uses bus_id_to_idx mapping
  instead of assuming bus_id == array_index (bus_id - 1)
- BUG FIX 2: pg columns now explicitly follow g_bus order from params,
  instead of unsorted column scanning

Features:
- N-1 independent linear models (one per non-Slack generator)
- Slack generators determined by power balance
- Pure data-driven, no physics constraints
- Baseline for comparison with DNN/PINN methods
- Auto-identify and handle Slack Bus
- Support API_TEST mode with dual constraint parameters

Evaluation Metrics (6 metrics aligned with DNN):
- MAE Pg (%) - Non-Slack, Slack
- Pg Violation (p.u., Mean of Max) - Non-Slack, Slack
- Branch Violation (p.u., Mean of Max)
- Cost Gap (%)
- Training Time (s)
- Inference Time (ms)
"""

import numpy as np
import pandas as pd
import time
import os
import sys
from sklearn.linear_model import LinearRegression

# Import custom modules
sys.path.append(os.path.dirname(__file__))

from dcopf_data_setup import (
    load_parameters_from_csv,
    DataSplitMode,
    split_data_by_mode
)
from dcopf_violation_metrics import (
    feasibility,
    compute_branch_violation_pu,
    compute_cost,
    compute_cost_gap_percentage
)
from dcopf_slack_utils import (
    identify_slack_bus_and_gens,
    update_params_with_slack_info,
    reconstruct_full_pg,
    compute_detailed_mae,
    compute_detailed_pg_violations_pu
)


# =====================================================================
# Data Loading Function (v5.1 FIXED)
# =====================================================================

def load_and_prepare_lr_data(file_path, params, column_names):
    """
    Load and prepare Linear Regression training data

    v5.1 FIXES:
    -----------
    1. Load demand mapping: uses bus_id_to_idx dict instead of bus_id-1 indexing.
       This is critical for case300 where bus IDs like 7001, 9533 exist.
    2. Generator columns: explicitly built from g_bus order in params,
       instead of unsorted column scanning which may produce wrong column order.
    """
    full_df = pd.read_csv(file_path)
    n_samples = len(full_df)
    n_buses = params['general']['n_buses']

    # ------------------------------------------------------------------ #
    # FIX 1: Load demand data — use bus_id_to_idx mapping                #
    # ------------------------------------------------------------------ #
    load_prefix = column_names['load_prefix']
    load_cols = [col for col in full_df.columns if col.startswith(load_prefix)]

    bus_id_to_idx = params['general']['bus_id_to_idx']
    x_data_raw = np.zeros((n_samples, n_buses), dtype='float32')

    for col_name in load_cols:
        bus_id = int(col_name[len(load_prefix):])
        if bus_id in bus_id_to_idx:
            x_data_raw[:, bus_id_to_idx[bus_id]] = full_df[col_name].values.astype('float32')
        else:
            print(f"[WARNING] load bus {bus_id} not in bus_id_to_idx, skipping")

    # ------------------------------------------------------------------ #
    # FIX 2: Generator columns — MUST match g_bus order from params      #
    # ------------------------------------------------------------------ #
    g_bus = params['general']['g_bus']
    gen_prefix = column_names['gen_prefix']
    pg_cols = [f"{gen_prefix}{int(gen_id)}" for gen_id in g_bus]

    missing = [c for c in pg_cols if c not in full_df.columns]
    if missing:
        raise ValueError(
            f"[load_and_prepare_lr_data] Missing pg columns in CSV: {missing}\n"
            f"Available pg columns (first 10): "
            f"{sorted([c for c in full_df.columns if c.startswith(gen_prefix)])[:10]}"
        )

    y_pg_raw_all = full_df[pg_cols].values.astype('float32')

    expected_n_gen = params['general']['n_g']
    if len(pg_cols) != expected_n_gen:
        print(f"[WARNING] Number of generators in CSV ({len(pg_cols)}) "
              f"does not match parameters file ({expected_n_gen})!")

    # Extract non-Slack generator data
    non_slack_indices = params['general']['non_slack_gen_indices']
    y_pg_raw_non_slack = y_pg_raw_all[:, non_slack_indices]

    return x_data_raw, y_pg_raw_non_slack, y_pg_raw_all


# =====================================================================
# Linear Regression Model
# =====================================================================

class LinearRegressionDCOPF:
    """DCOPF Linear Regression model (supports Slack Bus)"""

    def __init__(self, n_gen_non_slack):
        """
        Initialize model

        Args:
            n_gen_non_slack: Number of non-Slack generators
        """
        self.n_gen = n_gen_non_slack
        self.pg_models = [LinearRegression() for _ in range(n_gen_non_slack)]
        self.is_fitted = False

    def fit(self, X_train, y_pg_train_non_slack):
        """
        Train model

        Args:
            X_train: (n_samples, n_buses) - Load data [Pd]
            y_pg_train_non_slack: (n_samples, n_g_non_slack) - Non-Slack generator outputs (p.u.)

        Returns:
            train_time: Training time in seconds
        """
        t_start = time.time()

        for gen_idx in range(self.n_gen):
            self.pg_models[gen_idx].fit(X_train, y_pg_train_non_slack[:, gen_idx])

        t_train = time.time() - t_start
        self.is_fitted = True

        return t_train

    def predict(self, X):
        """
        Predict non-Slack generator outputs

        Args:
            X: (n_samples, n_buses) - Load data

        Returns:
            y_pg_pred_non_slack: (n_samples, n_g_non_slack) - Predicted non-Slack generator outputs
        """
        if not self.is_fitted:
            raise ValueError("Model not trained!")

        n_samples = X.shape[0]
        y_pg_pred_non_slack = np.zeros((n_samples, self.n_gen))

        for gen_idx in range(self.n_gen):
            y_pg_pred_non_slack[:, gen_idx] = self.pg_models[gen_idx].predict(X)

        return y_pg_pred_non_slack


# =====================================================================
# Evaluation Function
# =====================================================================

def evaluate_model(
        model,
        X,
        indices,
        raw_data,
        params,
        test_data_external=None,
        test_params=None
):
    """Evaluate Linear Regression model (aligned with DNN)"""

    eval_params = test_params if test_params is not None else params

    if test_data_external is not None:
        X_eval = test_data_external['x']
        y_true_pg_all = test_data_external['y_pg_all']
        x_pd_raw = X_eval
    else:
        X_eval = X
        y_true_pg_all = raw_data['y_pg_all'][indices]
        x_pd_raw = raw_data['x'][indices]

    # Predict non-Slack
    y_pred_non_slack = model.predict(X_eval)

    # Reconstruct full Pg
    pd_total = x_pd_raw.sum(axis=1)
    y_pred_all = reconstruct_full_pg(
        pg_non_slack=y_pred_non_slack,
        pd_total=pd_total,
        params=eval_params
    )

    # Calculate MAE
    mae_dict = compute_detailed_mae(
        y_true_all=y_true_pg_all,
        y_pred_non_slack=y_pred_non_slack,
        y_pred_all=y_pred_all,
        params=eval_params
    )

    # Calculate violations (p.u.)
    gen_up_viol, gen_lo_viol, line_viol, balance_err = feasibility(
        y_pred_pg=y_pred_all,
        x_pd=x_pd_raw,
        params=eval_params
    )

    viol_dict = compute_detailed_pg_violations_pu(
        gen_up_viol=gen_up_viol,
        gen_lo_viol=gen_lo_viol,
        params=eval_params
    )

    viol_branch_pu = compute_branch_violation_pu(
        line_viol=line_viol,
        Pl_max=eval_params['constraints']['Pl_max']
    )

    # Calculate Cost
    cost_coeffs = {
        'C2': eval_params['constraints'].get('C_Pg_c2', np.zeros(y_true_pg_all.shape[1])),
        'C1': eval_params['constraints']['C_Pg'],
        'C0': eval_params['constraints'].get('C_Pg_c0', np.zeros(y_true_pg_all.shape[1]))
    }

    cost_true = compute_cost(y_true_pg_all, cost_coeffs)
    cost_pred = compute_cost(y_pred_all, cost_coeffs)
    cost_gap_pct = compute_cost_gap_percentage(cost_true, cost_pred)

    return {
        'mae_pg_non_slack': mae_dict['mae_non_slack'],
        'mae_pg_slack': mae_dict['mae_slack'],
        'viol_pg_non_slack': viol_dict['viol_non_slack'],
        'viol_pg_slack': viol_dict['viol_slack'],
        'viol_branch': viol_branch_pu,
        'cost_gap_percent': cost_gap_pct,
    }


# =====================================================================
# Main Training Function
# =====================================================================

def linear_regression_dcopf_experiment(
        case_name,
        params_path,
        dataset_path,
        column_names,
        n_train_use=10000,
        seed=42,
        split_mode=DataSplitMode.RANDOM_SPLIT,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=1000
):
    """Linear Regression DCOPF experiment (aligned with DNN version)"""

    np.random.seed(seed)

    # Load parameters
    params = load_parameters_from_csv(case_name, params_path, is_api=False)
    slack_info = identify_slack_bus_and_gens(params)
    params = update_params_with_slack_info(params, slack_info)

    test_params = None
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError('API_TEST mode requires test_params_path')
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_slack_info = identify_slack_bus_and_gens(test_params)
        test_params = update_params_with_slack_info(test_params, test_slack_info)
    else:
        test_params = params

    n_g_non_slack = params['general']['n_g_non_slack']

    # Load data
    x_data_raw, y_pg_raw_non_slack, y_pg_raw_all = load_and_prepare_lr_data(
        dataset_path, params, column_names
    )

    raw_data = {
        'x': x_data_raw,
        'y_pg_non_slack': y_pg_raw_non_slack,
        'y_pg_all': y_pg_raw_all
    }

    # Data split
    train_idx, val_idx, test_idx, x_test_external, y_test_external = split_data_by_mode(
        x_data_raw=x_data_raw,
        y_pg_raw=y_pg_raw_all,
        mode=split_mode,
        n_train_use=n_train_use,
        seed=seed,
        test_data_path=test_data_path,
        params=params,
        column_names=column_names,
        n_test_samples=n_test_samples
    )

    X_train = x_data_raw[train_idx]

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        X_test = None
    else:
        X_test = x_data_raw[test_idx]

    y_pg_train_non_slack = y_pg_raw_non_slack[train_idx]

    # Train model
    model = LinearRegressionDCOPF(n_g_non_slack)
    train_time = model.fit(X_train, y_pg_train_non_slack)

    # Evaluation (test set only)
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }
        test_metrics = evaluate_model(
            model=model,
            X=None,
            indices=None,
            raw_data=raw_data,
            params=params,
            test_data_external=test_data_external_dict,
            test_params=test_params
        )
    else:
        test_metrics = evaluate_model(
            model=model,
            X=X_test,
            indices=test_idx,
            raw_data=raw_data,
            params=params,
            test_params=test_params
        )

    # Speed test
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_sample = x_test_external[:1]
    else:
        test_sample = X_test[:1]

    times = []
    for _ in range(100):
        t_start = time.perf_counter()
        _ = model.predict(test_sample)
        times.append(time.perf_counter() - t_start)

    inference_time_ms = np.mean(times) * 1000

    # Print results (aligned with DNN format)
    print("\n" + "=" * 70)
    print("Test Set Results")
    print("=" * 70)
    print(f"\nNon-Slack Generators:")
    print(f"  MAE:        {test_metrics['mae_pg_non_slack']:.4f}%")
    print(f"  Violation:  {test_metrics['viol_pg_non_slack']:.4f} p.u.")
    print(f"\nSlack-Only Generators:")
    print(f"  MAE:        {test_metrics['mae_pg_slack']:.4f}%")
    print(f"  Violation:  {test_metrics['viol_pg_slack']:.4f} p.u.")
    print(f"\nBranch:")
    print(f"  Violation:  {test_metrics['viol_branch']:.4f} p.u.")
    print(f"\nCost Gap:     {test_metrics['cost_gap_percent']:.4f}%")
    print(f"\nTraining Time:   {train_time:.2f} s")
    print(f"Inference Time:  {inference_time_ms:.4f} ms")
    print("\n" + "=" * 70 + "\n")


# =====================================================================
# Main Program
# =====================================================================

if __name__ == "__main__":
    # ===================================================================
    # Experiment Configuration
    # ===================================================================

    # --- 1. Case Configuration ---
    CASE_NAME = 'pglib_opf_case118_ieee'
    CASE_SHORT_NAME = 'case118'

    # --- 2. Data Split Mode ---
    SPLIT_MODE = DataSplitMode.VALID_FIXED

    # --- 3. Training & Test Sample Counts ---
    N_TRAIN_USE = 35000
    N_TEST_SAMPLES = 1000

    # --- 4. Other ---
    SEED = 42

    # --- 5. Path Configuration (Manual) ---
    ROOT_DIR = "/lambda/nfs/lxy/dcopf_project/data"
    TRAIN_VARIANCE = "v=0.12"
    TEST_VARIANCE = "v=0.25"

    COLUMN_NAMES = {
        'load_prefix': 'pd',
        'gen_prefix': 'pg',
        'lambda': 'lambda',
        'mu_g_min_prefix': 'mu_g_min_',
        'mu_g_max_prefix': 'mu_g_max_',
        'mu_line_pos_prefix': 'mu_line_max_',
        'mu_line_neg_prefix': 'mu_line_min_',
    }

    # ===================================================================
    # Path Generation
    # ===================================================================

    params_path = os.path.join(ROOT_DIR, "DCOPF Constraints", CASE_SHORT_NAME)
    train_data_path = os.path.join(
        ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TRAIN_VARIANCE})",
        f"{CASE_NAME}_dataset_with_duals.csv"
    )

    if SPLIT_MODE == DataSplitMode.GENERALIZATION:
        test_data_path = os.path.join(
            ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TEST_VARIANCE})",
            f"{CASE_NAME}_dataset_with_duals.csv"
        )
        test_params_path = None
    elif SPLIT_MODE == DataSplitMode.API_TEST:
        test_data_path = os.path.join(
            ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}(v=api)",
            f"{CASE_NAME}__api_dataset_with_duals.csv"
        )
        test_params_path = os.path.join(
            ROOT_DIR, "DCOPF Constraints", f"{CASE_SHORT_NAME}(api)"
        )
    else:
        test_data_path = None
        test_params_path = None

    # ===================================================================
    # Run Experiment
    # ===================================================================
    linear_regression_dcopf_experiment(
        case_name=CASE_NAME,
        params_path=params_path,
        dataset_path=train_data_path,
        column_names=COLUMN_NAMES,
        n_train_use=N_TRAIN_USE,
        seed=SEED,
        split_mode=SPLIT_MODE,
        test_data_path=test_data_path,
        test_params_path=test_params_path,
        n_test_samples=N_TEST_SAMPLES
    )