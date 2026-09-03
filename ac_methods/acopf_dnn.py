# -*- coding: utf-8 -*-
"""
Traditional DNN ACOPF Main Experiment File (V10-Simplified-Output)

Modifications:
1. Model output dimension = [pg_non_slack, vm_gen] (only generator bus Vm)
2. solve_pf_custom_optimized: Accepts pg_non_slack, reconstructs full pg
3. evaluate_split: Reconstructs full pg after denormalization
4. Simplified output: only test set metrics
5. Removed all file saving (JSON/CSV/model)
6. All comments in English

Key Logic:
- DNN predicts non-Slack generator Pg and generator bus Vm
- Reconstruct full Pg array before power flow calculation (Slack position filled with 0)
- Evaluation compares full Pg (values after power flow auto-adjustment)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
from pathlib import Path

from pypower.runpf import runpf
from pypower.ppoption import ppoption

# Import configuration
try:
    import acopf_config
except ImportError:
    print("Error: Unable to import acopf_config.py")
    sys.exit(1)

try:
    from acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        DataMode,
        prepare_data_splits,
        load_generalization_test_data,
        load_api_test_data,
        reconstruct_full_pg
    )
except ImportError:
    print("Warning: Unable to import 'acopf_data_setup' module.")
    sys.exit(1)

try:
    from acopf_violation_metrics import evaluate_acopf_predictions
except ImportError:
    print("Warning: Unable to import 'acopf_violation_metrics' module.")
    sys.exit(1)

GLOBAL_CASE_DATA = None
PPOPT = None


class TraditionalNN_ACOPF(nn.Module):
    """DNN Model (structure unchanged, only output dimension changes)"""

    def __init__(self, input_size, output_size, hidden_sizes=[256, 256]):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def init_pypower_options():
    global PPOPT
    ppopt = ppoption()
    PPOPT = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)


def load_case_from_csv(case_name, constraints_path):
    """Load PyPower case data from CSV files (same as original)"""
    base_path = Path(constraints_path)
    base_mva_df = pd.read_csv(base_path / f"{case_name}_base_mva.csv")
    bus_df = pd.read_csv(base_path / f"{case_name}_bus_data.csv")
    gen_df = pd.read_csv(base_path / f"{case_name}_gen_data.csv")
    branch_df = pd.read_csv(base_path / f"{case_name}_branch_data.csv")
    baseMVA = base_mva_df['value'].iloc[0]

    bus = np.zeros((len(bus_df), 13))
    bus[:, 0] = bus_df['bus_id'].values
    bus[:, 1] = bus_df['type'].values
    bus[:, 2] = bus_df['pd_pu'].values
    bus[:, 3] = bus_df['qd_pu'].values
    bus[:, 6] = 1
    bus[:, 7] = bus_df['vm_pu'].values
    bus[:, 8] = bus_df['va_deg'].values
    bus[:, 9] = bus_df['base_kv'].values
    bus[:, 10] = 1
    bus[:, 11] = bus_df['vmax_pu'].values
    bus[:, 12] = bus_df['vmin_pu'].values

    gen = np.zeros((len(gen_df), 21))
    gen[:, 0] = gen_df['bus_id'].values
    gen[:, 3] = gen_df['qg_max_pu'].values
    gen[:, 4] = gen_df['qg_min_pu'].values
    gen[:, 5] = gen_df['vg_pu'].values
    gen[:, 6] = baseMVA
    gen[:, 7] = 1
    gen[:, 8] = gen_df['pg_max_pu'].values
    gen[:, 9] = gen_df['pg_min_pu'].values

    branch = np.zeros((len(branch_df), 13))
    branch[:, 0] = branch_df['f_bus'].values
    branch[:, 1] = branch_df['t_bus'].values
    branch[:, 2] = branch_df['r_pu'].values
    branch[:, 3] = branch_df['x_pu'].values
    branch[:, 4] = branch_df['b_pu'].values
    branch[:, 5] = branch_df['rate_a_pu'].values
    branch[:, 6] = branch[:, 5]
    branch[:, 7] = branch[:, 5]
    branch[:, 8] = branch_df['tap_ratio'].values
    branch[:, 9] = branch_df['shift_deg'].values
    branch[:, 10] = 1
    branch[:, 11] = -360
    branch[:, 12] = 360
    rate_a_values = branch_df['rate_a_pu'].values
    branch[:, 5:8][np.isnan(rate_a_values) | np.isinf(rate_a_values), :] = 9900.0

    gencost = np.zeros((len(gen_df), 7))
    gencost[:, 0] = 2
    gencost[:, 3] = 3
    gencost[:, 4] = gen_df['cost_c2'].values
    gencost[:, 5] = gen_df['cost_c1'].values
    gencost[:, 6] = gen_df['cost_c0'].values

    ppc = {'version': '2', 'baseMVA': baseMVA, 'bus': bus, 'gen': gen, 'branch': branch, 'gencost': gencost}
    ppc['bus'][:, 2] *= baseMVA
    ppc['bus'][:, 3] *= baseMVA
    ppc['gen'][:, 3] *= baseMVA
    ppc['gen'][:, 4] *= baseMVA
    ppc['gen'][:, 8] *= baseMVA
    ppc['gen'][:, 9] *= baseMVA
    mask = (ppc['branch'][:, 5] != 0) & (ppc['branch'][:, 5] < 9000)
    ppc['branch'][mask, 5:8] *= baseMVA
    return ppc


# =====================================================================
# KEY MODIFICATION 1: Accept pg_non_slack, reconstruct full pg array
# =====================================================================
def solve_pf_custom_optimized(pd, qd, pg_non_slack, vm, params):
    """
    Run power flow calculation (modified: accepts non-Slack generator Pg)

    Args:
        pd: Load active power (p.u.), shape (n_loads,)
        qd: Load reactive power (p.u.), shape (n_loads,)
        pg_non_slack: Non-Slack generator active power (p.u.), shape (n_gen_non_slack,)
        vm: Generator bus voltage (p.u.), shape (n_gen,)
        params: Parameters dictionary

    Returns:
        Power flow calculation results
    """
    global GLOBAL_CASE_DATA, PPOPT
    BASE_MVA = params['general']['BASE_MVA']

    mpc_pf = {
        'version': GLOBAL_CASE_DATA['version'],
        'baseMVA': GLOBAL_CASE_DATA['baseMVA'],
        'bus': GLOBAL_CASE_DATA['bus'].copy(),
        'gen': GLOBAL_CASE_DATA['gen'].copy(),
        'branch': GLOBAL_CASE_DATA['branch'],
        'gencost': GLOBAL_CASE_DATA['gencost']
    }

    # Set loads
    load_bus_ids = params['general']['load_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc_pf["bus"][bus_idx, 2] = pd[i] * BASE_MVA
            mpc_pf["bus"][bus_idx, 3] = qd[i] * BASE_MVA

    # 🔥 Set generator active power (only non-Slack generators)
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    n_gen = params['general']['n_gen']

    # For non-Slack generators: set predicted pg
    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc_pf["gen"][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    # For Slack generators: don't set pg (or set to 0), let power flow algorithm auto-balance
    # PyPower will automatically adjust Slack bus pg to satisfy power balance

    # Set generator voltage
    for i in range(n_gen):
        mpc_pf["gen"][i, 5] = vm[i]

    return runpf(mpc_pf, PPOPT)


# =====================================================================
# KEY MODIFICATION 2: Handle model output dimension change
# =====================================================================
def evaluate_split(model, X, indices, raw_data, params, scalers, device, split_name, verbose=True):
    """
    Evaluate model performance on specified dataset (modified: handle non-Slack pg and generator Vm)
    """
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X.to(device))
    y_pred_scaled_np = y_pred_scaled.cpu().numpy()

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # 🔥 Denormalize (key modification)
    # Model output: [pg_non_slack (n_gen_non_slack), vm_gen (n_gen)]
    y_pred_pg_non_slack_scaled = y_pred_scaled_np[:, :n_gen_non_slack]
    y_pred_vm_gen_scaled = y_pred_scaled_np[:, n_gen_non_slack:]

    y_pred_pg_non_slack = scalers['pg'].inverse_transform(y_pred_pg_non_slack_scaled)  # Non-Slack pg
    y_pred_vm_gen = scalers['vm'].inverse_transform(y_pred_vm_gen_scaled)  # Generator bus vm

    # 🔥 Reconstruct full Pg array (for power flow calculation)
    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)

    # 🔥 Reconstruct full Vm array (all buses)
    # We need all bus Vm for evaluation, expand generator Vm to all buses
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_all = np.zeros((len(X), n_buses), dtype=y_pred_vm_gen.dtype)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen

    # For non-generator buses, use nominal voltage (or from power flow calculation)
    # Here we use a simple approach: fill with 1.0 p.u.
    non_gen_mask = np.ones(n_buses, dtype=bool)
    non_gen_mask[gen_bus_indices] = False
    y_pred_vm_all[:, non_gen_mask] = 1.0

    # Extract true values
    y_true_pg = raw_data['pg'][indices]  # All generator true pg
    y_true_vm = raw_data['vm'][indices]  # All bus true vm
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

    # Extract input data
    x_raw_data = scalers['x'].inverse_transform(X.cpu().numpy())
    pd_pu = x_raw_data[:, :n_loads]
    qd_pu = x_raw_data[:, n_loads:]

    # 🔥 Run power flow calculation (use non-Slack pg)
    n_samples = len(X)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            # Pass non-Slack generator pg
            r1_pf = solve_pf_custom_optimized(
                pd_pu[i],
                qd_pu[i],
                y_pred_pg_non_slack[i],  # 🔥 Pass non-Slack pg
                y_pred_vm_gen[i],  # Pass generator bus vm
                params
            )
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except:
            pf_results_list.append((
                {'success': False,
                 'gen': np.zeros((n_gen, 21)),
                 'bus': np.zeros((n_buses, 13)),
                 'branch': np.zeros((1, 17))},
            ))
            converge_flags.append(False)

    if verbose:
        print(f"    ✓ Converged: {sum(converge_flags)}/{n_samples}")

    # Evaluate (using full Pg)
    # Note: evaluate_acopf_predictions will extract full pg from pf_results_list
    # So y_pred_pg_full here is mainly for consistency check
    return evaluate_acopf_predictions(
        y_pred_pg_full,  # 🔥 Pass reconstructed full pg (actual evaluation uses power flow results)
        y_pred_vm_all,  # Pass full bus vm
        y_true_pg,  # True full pg
        y_true_vm,  # True full vm
        y_true_qg,
        y_true_va_rad,
        pf_results_list,
        converge_flags,
        params,
        verbose=verbose
    )


def traditional_nn_acopf_experiment(
        case_name,
        params_path,
        data_path,
        log_path,
        results_path,
        data_mode='random_split',
        n_train_use=None,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=None,
        seed=42,
        n_epochs=1000,
        early_stop_patience=20,
        early_stop_min_delta=1e-6,
        learning_rate=0.001,
        hidden_sizes=[256, 256],
        batch_size=None,
        device='cuda',
        tolerances=None
):
    """
    Traditional DNN ACOPF experiment main function (modified: exclude Slack Pg, only generator Vm)
    """
    global GLOBAL_CASE_DATA, PPOPT
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"ACOPF DNN Experiment")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Case: {case_name}")
    print(f"Data Mode: {data_mode}")
    print(f"{'=' * 70}")

    # ========================================================================
    # 1. Load training data params and PyPower case data
    # ========================================================================
    params = load_parameters_from_csv(case_name, params_path)
    init_pypower_options()
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)
    print(f"✓ Training params and PyPower case data loaded")

    # ========================================================================
    # 2. Load training data and fit scalers
    # ========================================================================
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
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
        print(f"\n{'=' * 70}")
        print(f"Data Mode: API_TEST")
        print(f"{'=' * 70}")

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
        print(f"✓ API test PyPower case data loaded")

        print(f"\n[API Test Data Info]")
        print(f"  Buses: {test_params['general']['n_buses']}")
        print(
            f"  Generators: {test_params['general']['n_gen']} (Non-Slack: {test_params['general']['n_gen_non_slack']})")
        print(f"  Loads: {test_params['general']['n_loads']}")
        print(f"  Base MVA: {test_params['general']['BASE_MVA']}")

    elif data_mode == DataMode.GENERALIZATION:
        print(f"\n{'=' * 70}")
        print(f"Data Mode: GENERALIZATION")
        print(f"{'=' * 70}")

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
        print(f"\n{'=' * 70}")
        print(f"Data Mode: {data_mode}")
        print(f"{'=' * 70}")

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
    X_train = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32, device=device)
    Y_train = torch.tensor(y_data_scaled[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(x_data_scaled[val_idx], dtype=torch.float32, device=device)
    Y_val = torch.tensor(y_data_scaled[val_idx], dtype=torch.float32, device=device)
    X_test = torch.tensor(test_x_scaled[test_idx], dtype=torch.float32, device=device)

    print(f"\n[Dataset Sizes]")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # ========================================================================
    # 5. 🔥 Create model (output dimension changed)
    # ========================================================================
    input_dim = x_data_scaled.shape[1]
    output_dim = y_data_scaled.shape[1]  # n_gen_non_slack + n_gen

    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"Input dim: {input_dim} (Load: pd + qd)")
    print(f"Output dim: {output_dim} (pg_non_slack: {n_gen_non_slack} + vm_gen: {n_gen})")
    print(f"Network: {input_dim} -> {' -> '.join(map(str, hidden_sizes))} -> {output_dim}")
    print(f"Training params: max_epochs={n_epochs}, patience={early_stop_patience}, lr={learning_rate}, batch_size={batch_size or 'full batch'}")
    print(f"{'=' * 70}")

    model = TraditionalNN_ACOPF(input_dim, output_dim, hidden_sizes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ========================================================================
    # 6. Training loop
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Training Progress")
    print(f"{'=' * 70}")

    n_train = len(X_train)
    batch_size = batch_size or n_train
    n_batches = (n_train + batch_size - 1) // batch_size
    train_losses, val_losses = [], []
    t0 = time.perf_counter()

    # Early stopping state
    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        indices = torch.randperm(n_train)
        for i in range(n_batches):
            batch_idx = indices[i * batch_size:min((i + 1) * batch_size, n_train)]
            optimizer.zero_grad()
            loss = criterion(model(X_train[batch_idx]), Y_train[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
        train_losses.append(epoch_loss / n_train)

        model.eval()
        with torch.no_grad():
            val_losses.append(float(criterion(model(X_val), Y_val).item()))

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {train_losses[-1]:.6f} - Val Loss: {val_losses[-1]:.6f}")

        # Early stopping check
        if val_losses[-1] < best_val_loss - early_stop_min_delta:
            best_val_loss = val_losses[-1]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {train_losses[-1]:.6f} - Val Loss: {val_losses[-1]:.6f}")
                print(f"\n⚡ Early stopping triggered at epoch {epoch} (patience={early_stop_patience})")
                break

    # Restore best model weights
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    print(f"✓ Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")

    train_time = time.perf_counter() - t0
    print(f"✓ Training completed in {train_time:.2f} seconds")

    # ========================================================================
    # 7. Model evaluation (use test params)
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

    test_metrics = evaluate_split(
        model, X_test, test_idx, test_raw_data,
        test_params,
        scalers, device, test_split_name,
        verbose=True
    )

    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_BACKUP

    # ========================================================================
    # 8. Inference speed test
    # ========================================================================
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(X_test[:1])

    times = [time.perf_counter() for _ in range(101)]
    with torch.no_grad():
        for i in range(100):
            model(X_test[:1])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times[i + 1] = time.perf_counter()

    latency_ms = np.mean(np.diff(times)) * 1000

    # ========================================================================
    # 9. 🔥 Simplified result summary (only specified metrics)
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
    print(f"Inference Time: {latency_ms:.4f} ms/sample")
    print(f"Training Time:  {train_time:.2f} s")
    print(f"Convergence Rate: {test_metrics['convergence_rate_percent']:.2f}%")

    print(f"{'=' * 70}")

    # ---- Save per-sample arrays for bootstrap over test samples ----
    ps = test_metrics['per_sample']
    save_dir = os.path.join(acopf_config.ROOT_DIR, "bootstrap_per_sample")
    os.makedirs(save_dir, exist_ok=True)
    method_name = "DNN"                      # 每个方法脚本改这里: "LR" / "GNN" / "DC3" / "RL" ...
    system_name = acopf_config.TRAIN_CASE    # 直接读配置, 得到 'case30'/'case118'/'case300'
    np.savez(os.path.join(save_dir, f"{method_name}__{system_name}.npz"),
             infer_ms=latency_ms, formulation="AC", **ps)   # DC 管线写 "DC"
    print(f"✓ Saved per-sample arrays -> {os.path.join(save_dir, f'{method_name}__{system_name}.npz')}")

    return test_metrics


if __name__ == "__main__":
    # =========================================================================
    # Read configuration from acopf_config.py and run experiment
    # =========================================================================

    # Print configuration
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    paths = acopf_config.get_all_paths()
    params = acopf_config.get_all_params()

    # Run experiment
    results = traditional_nn_acopf_experiment(
        **paths,
        **params
    )

    print("\n✓ Experiment completed successfully!")