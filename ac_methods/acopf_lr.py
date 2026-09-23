# -*- coding: utf-8 -*-
"""Linear regression baseline for ACOPF: one model per non-slack Pg and per generator Vm."""

import numpy as np
import time
import sys
from sklearn.linear_model import LinearRegression

import os

# ac_configuration/ sits in ac_methods/. Appending the parent of this script's own
# directory makes it importable whether this file is directly in ac_methods/ or one
# level down in a grouped method folder, and from any working directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ac_configuration import acopf_config
    from ac_configuration.acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        prepare_data_splits,
        reconstruct_full_pg
    )
    from ac_configuration.acopf_evaluation_metrics import evaluate_acopf_predictions
    from ac_configuration.acopf_pypower import load_case_from_csv, solve_pf_setpoints
except ImportError as e:
    print(f"Error: Unable to import from ac_configuration/ ({e})")
    sys.exit(1)

GLOBAL_CASE_DATA = None


class LinearRegressionACOPF:
    """One independent least-squares model per output: n_gen_non_slack for Pg, n_gen for Vm."""

    def __init__(self, n_gen_non_slack, n_gen):
        self.n_gen_non_slack = n_gen_non_slack
        self.n_gen = n_gen
        self.pg_models = [LinearRegression() for _ in range(n_gen_non_slack)]
        self.vm_models = [LinearRegression() for _ in range(n_gen)]
        self.is_fitted = False

    def fit(self, X_train, y_pg_non_slack_train, y_vm_gen_train):
        """Fit every output model and return the total wall-clock training time."""
        print(f"\nTraining {self.n_gen_non_slack} Pg models and {self.n_gen} Vm models...")
        t_start = time.perf_counter()

        for gen_idx in range(self.n_gen_non_slack):
            self.pg_models[gen_idx].fit(X_train, y_pg_non_slack_train[:, gen_idx])

        for gen_idx in range(self.n_gen):
            self.vm_models[gen_idx].fit(X_train, y_vm_gen_train[:, gen_idx])

        t_train = time.perf_counter() - t_start
        self.is_fitted = True
        print(f"Training completed in {t_train:.2f} seconds")
        return t_train

    def predict(self, X):
        """Return predicted non-slack Pg and generator-bus Vm in physical units."""
        if not self.is_fitted:
            raise ValueError("Model not trained")

        n_samples = X.shape[0]
        y_pg_non_slack_pred = np.zeros((n_samples, self.n_gen_non_slack))
        y_vm_gen_pred = np.zeros((n_samples, self.n_gen))

        for gen_idx in range(self.n_gen_non_slack):
            y_pg_non_slack_pred[:, gen_idx] = self.pg_models[gen_idx].predict(X)

        for gen_idx in range(self.n_gen):
            y_vm_gen_pred[:, gen_idx] = self.vm_models[gen_idx].predict(X)

        return y_pg_non_slack_pred, y_vm_gen_pred


def evaluate_model(model, X, indices, raw_data, params, scalers, split_name, verbose=True):
    """Predict, run power flow per sample, and return the evaluation metrics."""
    if verbose:
        print(f"\n{split_name} Evaluation:")

    n_gen = params['general']['n_gen']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    y_pred_pg_non_slack, y_pred_vm_gen = model.predict(X)

    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)

    # Scatter generator Vm into an all-bus array; load buses are unpredicted, so they
    # are set to nominal voltage and excluded from MAE_Vm downstream
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_all = np.full((len(X), n_buses), 1.0, dtype=y_pred_vm_gen.dtype)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen

    y_true_pg = raw_data['pg'][indices]
    y_true_vm = raw_data['vm'][indices]
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

    x_raw_data = scalers['x'].inverse_transform(X)
    pd_pu = x_raw_data[:, :n_loads]
    qd_pu = x_raw_data[:, n_loads:]

    n_samples = len(X)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_setpoints(
                pd_pu[i], qd_pu[i], y_pred_pg_non_slack[i], y_pred_vm_gen[i],
                params, GLOBAL_CASE_DATA
            )
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except Exception:
            pf_results_list.append((
                {'success': False,
                 'gen': np.zeros((n_gen, 21)),
                 'bus': np.zeros((n_buses, 13)),
                 'branch': np.zeros((1, 17))},
            ))
            converge_flags.append(False)

    if verbose:
        print(f"  Converged: {sum(converge_flags)}/{n_samples}")

    return evaluate_acopf_predictions(
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


def linear_regression_experiment(
        case_name,
        params_path,
        data_path,
        n_train_use=None,
        seed=42,
        **kwargs  # Absorbs iterative-training settings that do not apply to least squares
):
    """Fit least-squares models on a random split and evaluate on the held-out test indices."""
    global GLOBAL_CASE_DATA
    np.random.seed(seed)

    print(f"\n{'=' * 70}")
    print(f"Linear Regression for AC-OPF")
    print(f"{'=' * 70}")
    print(f"Case: {case_name}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # 1. Load network parameters and PyPower case data
    # ------------------------------------------------------------------
    params = load_parameters_from_csv(case_name, params_path)
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)

    # ------------------------------------------------------------------
    # 2. Load dataset and fit scalers
    # ------------------------------------------------------------------
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_loads = params['general']['n_loads']
    baseMVA = params['general']['BASE_MVA']

    print(f"\n[Dataset Info]")
    print(f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_non_slack}), "
          f"Loads: {n_loads}, Base MVA: {baseMVA}")
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # ------------------------------------------------------------------
    # 3. Split. The validation indices go unused here, but the split is kept
    #    identical to the other methods so the test sets are comparable.
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = prepare_data_splits(
        x_data_scaled, y_data_scaled,
        n_train_use=n_train_use,
        seed=seed
    )

    X_train = x_data_scaled[train_idx]
    X_test = x_data_scaled[test_idx]

    # Targets are the unscaled values, so predictions come out in physical units
    y_pg_non_slack_train = raw_data['pg_non_slack'][train_idx]
    y_vm_gen_train = raw_data['vm_gen'][train_idx]

    # ------------------------------------------------------------------
    # 4. Fit
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"Input dim: {x_data_scaled.shape[1]} (pd + qd)")
    print(f"Independent models: {n_gen_non_slack} (pg_non_slack) + {n_gen} (vm_gen)")
    print(f"{'=' * 70}")

    model = LinearRegressionACOPF(n_gen_non_slack, n_gen)
    train_time = model.fit(X_train, y_pg_non_slack_train, y_vm_gen_train)

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    test_metrics = evaluate_model(
        model, X_test, test_idx, raw_data, params, scalers, "Test", verbose=True
    )

    # ------------------------------------------------------------------
    # 6. Inference latency (single sample)
    # ------------------------------------------------------------------
    for _ in range(10):
        model.predict(X_test[:1])

    times = []
    for _ in range(100):
        t_start = time.perf_counter()
        model.predict(X_test[:1])
        times.append(time.perf_counter() - t_start)

    latency_ms = np.mean(times) * 1000

    # ------------------------------------------------------------------
    # 7. Results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Final Results Summary")
    print(f"{'=' * 70}")
    print(f"\nCase: {case_name}")

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
    print(f"Branch_viol:         {test_metrics['mean_max_branch_viol_pu']:.6f} p.u. "
          f"(1.0 = 100% overload)")

    print(f"\n--- Cost Metrics ---")
    print(f"Cost Gap: {test_metrics['cost_optimality_gap_percent']:.4f}%")

    print(f"\n--- Performance ---")
    print(f"Inference Time: {latency_ms:.4f} ms/sample")
    print(f"Training Time:  {train_time:.2f} s")
    print(f"Convergence Rate: {test_metrics['convergence_rate_percent']:.2f}%")
    print(f"{'=' * 70}")

    return test_metrics


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)
    print("Note: least squares has no iterative training, so n_epochs, learning_rate,")
    print("      early stopping, batch size and device are ignored by this method.")

    results = linear_regression_experiment(
        **acopf_config.get_all_paths(),
        **acopf_config.get_all_params()
    )

    print("\nExperiment completed successfully!")