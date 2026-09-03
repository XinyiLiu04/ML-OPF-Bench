# -*- coding: utf-8 -*-
"""
Q-Correction ACOPF Neural Network (Aligned with DNN Baseline)
VERSION: 2.0 - FIXED QG DIMENSION ISSUE

Modifications (Early Stopping):
- qcorrection_nn_acopf_experiment: added early_stop_patience and early_stop_min_delta parameters
- Saves best model weights during training and restores after training loop ends
- Patience counter and min_delta threshold match acopf_config settings
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
from collections import defaultdict

from pypower.runpf import runpf
from pypower.ppoption import ppoption

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
    print("Error: Unable to import 'acopf_data_setup' module")
    sys.exit(1)

try:
    from acopf_violation_metrics import evaluate_acopf_predictions
except ImportError:
    print("Error: Unable to import 'acopf_violation_metrics' module")
    sys.exit(1)

GLOBAL_CASE_DATA = None
PPOPT = None


# ============================================================================
# 1. Neural Network Architecture
# ============================================================================
class QCorrectionNN_ACOPF(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[256, 256]):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Sigmoid())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 2. Parametrization Functions
# ============================================================================
def parametrize_pg_vm(pg_non_slack, vm_gen, params):
    pg_min = params['generator']['pg_min']
    pg_max = params['generator']['pg_max']
    vm_min = params['bus']['vm_min']
    vm_max = params['bus']['vm_max']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']

    pg_min_non_slack = pg_min[:, non_slack_gen_idx]
    pg_max_non_slack = pg_max[:, non_slack_gen_idx]

    gen_vm_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    vm_min_gen = vm_min[gen_vm_indices]
    vm_max_gen = vm_max[gen_vm_indices]

    alpha = (pg_non_slack - pg_min_non_slack) / (pg_max_non_slack - pg_min_non_slack + 1e-8)
    beta = (vm_gen - vm_min_gen) / (vm_max_gen - vm_min_gen + 1e-8)
    return np.hstack([alpha, beta])


def inverse_parametrize(alpha_beta_pred, params):
    pg_min = params['generator']['pg_min']
    pg_max = params['generator']['pg_max']
    vm_min = params['bus']['vm_min']
    vm_max = params['bus']['vm_max']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    n_gen_non_slack = params['general']['n_gen_non_slack']

    pg_min_non_slack = pg_min[:, non_slack_gen_idx]
    pg_max_non_slack = pg_max[:, non_slack_gen_idx]
    gen_vm_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    vm_min_gen = vm_min[gen_vm_indices]
    vm_max_gen = vm_max[gen_vm_indices]

    alpha = alpha_beta_pred[:, :n_gen_non_slack]
    beta = alpha_beta_pred[:, n_gen_non_slack:]

    pg_non_slack = pg_min_non_slack + alpha * (pg_max_non_slack - pg_min_non_slack)
    vm_gen = vm_min_gen + beta * (vm_max_gen - vm_min_gen)
    return pg_non_slack, vm_gen


# ============================================================================
# 3. Auxiliary Functions
# ============================================================================
def init_pypower_options():
    global PPOPT
    ppopt = ppoption()
    PPOPT = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)


def load_case_from_csv(case_name, params_path):
    base_path = Path(params_path)
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

    ppc = {'version': '2', 'baseMVA': baseMVA, 'bus': bus, 'gen': gen,
           'branch': branch, 'gencost': gencost}
    ppc['bus'][:, 2] *= baseMVA
    ppc['bus'][:, 3] *= baseMVA
    ppc['gen'][:, 3] *= baseMVA
    ppc['gen'][:, 4] *= baseMVA
    ppc['gen'][:, 8] *= baseMVA
    ppc['gen'][:, 9] *= baseMVA
    mask = (ppc['branch'][:, 5] != 0) & (ppc['branch'][:, 5] < 9000)
    ppc['branch'][mask, 5:8] *= baseMVA
    return ppc


# ============================================================================
# 4. Q-Correction Power Flow Solving (Paper Algorithm 1)
# ============================================================================
def solve_pf_with_qg_correction(pd, qd, pg_non_slack, vm_gen, params):
    global GLOBAL_CASE_DATA, PPOPT
    BASE_MVA = params['general']['BASE_MVA']
    qg_min = params['generator']['qg_min'].flatten()
    qg_max = params['generator']['qg_max'].flatten()
    n_gen = params['general']['n_gen']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    load_bus_ids = params['general']['load_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_bus_ids_arr = params['general']['gen_bus_ids']

    # Stage 1: First power flow
    mpc1 = {
        'version': GLOBAL_CASE_DATA['version'],
        'baseMVA': BASE_MVA,
        'bus': GLOBAL_CASE_DATA['bus'].copy(),
        'gen': GLOBAL_CASE_DATA['gen'].copy(),
        'branch': GLOBAL_CASE_DATA['branch'].copy(),
        'gencost': GLOBAL_CASE_DATA['gencost'],
    }

    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc1['bus'][bus_idx, 2] = pd[i] * BASE_MVA
            mpc1['bus'][bus_idx, 3] = qd[i] * BASE_MVA

    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc1['gen'][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    for i in range(min(n_gen, len(mpc1['gen']))):
        mpc1['gen'][i, 5] = vm_gen[i]

    try:
        r1 = runpf(mpc1, PPOPT)
    except Exception:
        fail = {'success': False,
                'gen': np.zeros((n_gen, 21)),
                'bus': np.zeros((params['general']['n_buses'], 13)),
                'branch': np.zeros((1, 17))}
        return (fail,), False

    if not r1[0]['success']:
        return r1, False

    # Stage 2: Check Qg limits
    qg_pf_pu = r1[0]['gen'][:n_gen, 2] / BASE_MVA
    violation_mask = (qg_pf_pu < qg_min) | (qg_pf_pu > qg_max)

    if not np.any(violation_mask):
        return r1, False

    # Stage 3: Second power flow with clipped Qg
    qg_clipped_pu = np.clip(qg_pf_pu, qg_min, qg_max)

    mpc2 = {
        'version': GLOBAL_CASE_DATA['version'],
        'baseMVA': BASE_MVA,
        'bus': GLOBAL_CASE_DATA['bus'].copy(),
        'gen': GLOBAL_CASE_DATA['gen'].copy(),
        'branch': GLOBAL_CASE_DATA['branch'].copy(),
        'gencost': GLOBAL_CASE_DATA['gencost'],
    }

    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc2['bus'][bus_idx, 2] = pd[i] * BASE_MVA
            mpc2['bus'][bus_idx, 3] = qd[i] * BASE_MVA

    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc2['gen'][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    for i in range(min(n_gen, len(mpc2['gen']))):
        if not violation_mask[i]:
            mpc2['gen'][i, 5] = vm_gen[i]

    bus_to_gens = defaultdict(list)
    for gi, bid in enumerate(gen_bus_ids_arr[:n_gen]):
        bus_to_gens[int(bid)].append(gi)

    for bus_id, gens_at in bus_to_gens.items():
        bus_idx = bus_id_to_idx.get(bus_id)
        if bus_idx is None:
            continue
        if int(mpc2['bus'][bus_idx, 1]) == 3:
            continue
        if not all(violation_mask[g] for g in gens_at):
            continue
        for g in gens_at:
            mpc2['gen'][g, 2] = float(qg_clipped_pu[g]) * BASE_MVA
        mpc2['bus'][bus_idx, 1] = 1

    try:
        r2 = runpf(mpc2, PPOPT)
    except Exception:
        return r1, True

    return r2, True


# ============================================================================
# 5. Evaluation Function
# ============================================================================
def evaluate_split_qcorrection(model, X, indices, raw_data, params, scalers, device, split_name, verbose=True):
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()
    with torch.no_grad():
        alpha_beta_pred = model(X.to(device)).cpu().numpy()

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    BASE_MVA = params['general']['BASE_MVA']

    pg_non_slack_pred, vm_gen_pred = inverse_parametrize(alpha_beta_pred, params)

    y_true_pg = raw_data['pg'][indices]
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]
    y_true_vm = raw_data['vm'][indices]

    x_raw_data = scalers['x'].inverse_transform(X.cpu().numpy())
    pd_pu = x_raw_data[:, :n_loads]
    qd_pu = x_raw_data[:, n_loads:]

    n_samples = len(X)
    pf_results_list = []
    converge_flags = []
    qg_correction_count = 0
    non_slack_gen_idx = params['general']['non_slack_gen_idx']

    if verbose:
        print(f"  Computing power flow with Q-correction for {n_samples} samples...")
        print(f"  Debug info:")
        print(f"    - pg_non_slack range: [{pg_non_slack_pred.min():.4f}, {pg_non_slack_pred.max():.4f}] p.u.")
        print(f"    - vm_gen range: [{vm_gen_pred.min():.4f}, {vm_gen_pred.max():.4f}] p.u.")
        print(f"    - n_gen: {n_gen}, n_gen_non_slack: {n_gen_non_slack}")
        print(f"    - non_slack_gen_idx: {non_slack_gen_idx[:5]}... (showing first 5)")

    first_sample_debug = True

    for i in range(n_samples):
        try:
            r_pf, had_correction = solve_pf_with_qg_correction(
                pd_pu[i], qd_pu[i], pg_non_slack_pred[i], vm_gen_pred[i], params
            )
            if first_sample_debug and verbose:
                print(f"\n  First sample debug:")
                print(f"    - Convergence: {r_pf[0]['success']}")
                if not r_pf[0]['success']:
                    print(f"    - Power flow FAILED for first sample")
                    print(f"    - pd range: [{pd_pu[i].min():.4f}, {pd_pu[i].max():.4f}]")
                    print(f"    - pg_non_slack[0]: {pg_non_slack_pred[i][0]:.4f}")
                    print(f"    - vm_gen[0]: {vm_gen_pred[i][0]:.4f}")
                first_sample_debug = False
            pf_results_list.append(r_pf)
            converge_flags.append(r_pf[0]['success'])
            if had_correction:
                qg_correction_count += 1
        except Exception as e:
            if first_sample_debug and verbose:
                print(f"\n  First sample EXCEPTION: {e}")
                import traceback
                traceback.print_exc()
                first_sample_debug = False
            pf_results_list.append(({'success': False, 'gen': np.zeros((n_gen, 21)),
                                     'bus': np.zeros((n_buses, 13)),
                                     'branch': np.zeros((1, 17))},))
            converge_flags.append(False)

    if verbose:
        print(f"    Converged: {sum(converge_flags)}/{n_samples}")
        print(f"    Q-Correction applied: {qg_correction_count}/{n_samples}")

    pg_full_pred = np.zeros((n_samples, n_gen))
    y_pred_vm_all = np.zeros((n_samples, n_buses))
    for i in range(n_samples):
        if converge_flags[i]:
            pg_full_pred[i, :] = pf_results_list[i][0]['gen'][:n_gen, 1] / BASE_MVA
            # Use Vm from PF results (not the NN setpoint), because Q-correction
            # may convert PV buses to PQ buses, changing Vm from the setpoint.
            y_pred_vm_all[i, :] = pf_results_list[i][0]['bus'][:n_buses, 7]
        else:
            pg_full_pred[i, :] = reconstruct_full_pg(pg_non_slack_pred[i], params)
            # Non-converged: fall back to NN setpoint for generator buses
            gen_vm_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
            y_pred_vm_all[i, gen_vm_indices] = vm_gen_pred[i]
            non_gen_mask = np.ones(n_buses, dtype=bool)
            non_gen_mask[gen_vm_indices] = False
            y_pred_vm_all[i, non_gen_mask] = 1.0

    return evaluate_acopf_predictions(
        pg_full_pred, y_pred_vm_all, y_true_pg, y_true_vm, y_true_qg, y_true_va_rad,
        pf_results_list, converge_flags, params, verbose=verbose
    )


# ============================================================================
# 6. Main Experiment Function
# ============================================================================
def qcorrection_nn_acopf_experiment(
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
        early_stop_patience=50,      # [ADDED] from acopf_config.EARLY_STOP_PATIENCE
        early_stop_min_delta=1e-6,   # [ADDED] from acopf_config.EARLY_STOP_MIN_DELTA
        learning_rate=0.001,
        hidden_sizes=None,
        batch_size=None,
        device='cuda',
        tolerances=None
):
    global GLOBAL_CASE_DATA, PPOPT
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"Q-Correction ACOPF Experiment")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Case: {case_name}")
    print(f"Data Mode: {data_mode}")
    print(f"{'=' * 70}")

    # 1. Load training data params and PyPower case data
    params = load_parameters_from_csv(case_name, params_path)
    init_pypower_options()
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)
    print(f"Training params and PyPower case data loaded")

    # 2. Load training data and fit scalers
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    baseMVA = params['general']['BASE_MVA']

    print(f"\n[Training Data Info]")
    print(f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_non_slack}), Loads: {n_loads}, Base MVA: {baseMVA}")
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # 3. Parametrize training data
    print(f"\n[Parametrizing Training Data]")
    print(f"  Original output dimension: {y_data_scaled.shape[1]} (pg_non_slack + vm_gen)")

    y_pg_non_slack_scaled = y_data_scaled[:, :n_gen_non_slack]
    y_vm_gen_scaled = y_data_scaled[:, n_gen_non_slack:]
    y_pg_non_slack = scalers['pg'].inverse_transform(y_pg_non_slack_scaled)
    y_vm_gen = scalers['vm'].inverse_transform(y_vm_gen_scaled)
    y_alpha_beta = parametrize_pg_vm(y_pg_non_slack, y_vm_gen, params)

    print(f"  Parametrized dimension: {y_alpha_beta.shape[1]} (2*{n_gen}-1 = {2 * n_gen - 1})")
    print(f"    - alpha (non-slack): {n_gen_non_slack}")
    print(f"    - beta (generator buses): {n_gen}")

    # 4. Load and split data based on data mode
    if data_mode == DataMode.API_TEST:
        print(f"\n{'=' * 70}\nData Mode: API_TEST\n{'=' * 70}")
        if test_data_path is None or test_params_path is None:
            raise ValueError("API_TEST mode requires test_data_path and test_params_path")

        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, y_alpha_beta, mode=DataMode.API_TEST,
            n_train_use=n_train_use, seed=seed)

        test_params, test_x_scaled, test_y_scaled, test_raw_data, _ = \
            load_api_test_data(test_data_path, test_params_path, scalers,
                               n_test_samples=n_test_samples or 1000, seed=seed)

        test_y_pg_non_slack_scaled = test_y_scaled[:, :test_params['general']['n_gen_non_slack']]
        test_y_vm_gen_scaled = test_y_scaled[:, test_params['general']['n_gen_non_slack']:]
        test_y_pg_non_slack = scalers['pg'].inverse_transform(test_y_pg_non_slack_scaled)
        test_y_vm_gen = scalers['vm'].inverse_transform(test_y_vm_gen_scaled)
        test_y_alpha_beta = parametrize_pg_vm(test_y_pg_non_slack, test_y_vm_gen, test_params)

        test_idx = np.arange(len(test_x_scaled))
        test_case_name = os.path.basename(test_data_path)
        if test_case_name.endswith('_pd.csv'):
            test_case_name = test_case_name[:-7]
        GLOBAL_CASE_DATA_TEST = load_case_from_csv(test_case_name, test_params_path)
        print(f"  API test PyPower case data loaded")
        print(f"\n[API Test Data Info]")
        print(f"  Buses: {test_params['general']['n_buses']}")
        print(f"  Generators: {test_params['general']['n_gen']} (Non-Slack: {test_params['general']['n_gen_non_slack']})")
        print(f"  Loads: {test_params['general']['n_loads']}")
        print(f"  Base MVA: {test_params['general']['BASE_MVA']}")

    elif data_mode == DataMode.GENERALIZATION:
        print(f"\n{'=' * 70}\nData Mode: GENERALIZATION\n{'=' * 70}")
        if test_data_path is None:
            raise ValueError("GENERALIZATION mode requires test_data_path")

        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, y_alpha_beta, mode=DataMode.GENERALIZATION,
            n_train_use=n_train_use, seed=seed)

        test_x_scaled, test_y_scaled, test_raw_data, _ = \
            load_generalization_test_data(test_data_path, params, scalers,
                                          n_test_samples=n_test_samples or 1000, seed=seed)

        test_y_pg_non_slack_scaled = test_y_scaled[:, :n_gen_non_slack]
        test_y_vm_gen_scaled = test_y_scaled[:, n_gen_non_slack:]
        test_y_pg_non_slack = scalers['pg'].inverse_transform(test_y_pg_non_slack_scaled)
        test_y_vm_gen = scalers['vm'].inverse_transform(test_y_vm_gen_scaled)
        test_y_alpha_beta = parametrize_pg_vm(test_y_pg_non_slack, test_y_vm_gen, params)

        test_idx = np.arange(len(test_x_scaled))
        test_params = params
        GLOBAL_CASE_DATA_TEST = GLOBAL_CASE_DATA

    else:
        print(f"\n{'=' * 70}\nData Mode: {data_mode}\n{'=' * 70}")
        train_idx, val_idx, test_idx = prepare_data_splits(
            x_data_scaled, y_alpha_beta, mode=data_mode,
            n_train_use=n_train_use, seed=seed)

        test_x_scaled = x_data_scaled
        test_y_alpha_beta = y_alpha_beta
        test_raw_data = raw_data
        test_params = params
        GLOBAL_CASE_DATA_TEST = GLOBAL_CASE_DATA

    # 5. Prepare training data
    X_train = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32, device=device)
    Y_train = torch.tensor(y_alpha_beta[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(x_data_scaled[val_idx], dtype=torch.float32, device=device)
    Y_val = torch.tensor(y_alpha_beta[val_idx], dtype=torch.float32, device=device)
    X_test = torch.tensor(test_x_scaled[test_idx], dtype=torch.float32, device=device)

    print(f"\n[Dataset Sizes]")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # 6. Create Q-correction neural network
    input_dim = x_data_scaled.shape[1]
    output_dim = y_alpha_beta.shape[1]

    model = QCorrectionNN_ACOPF(input_dim, output_dim, hidden_sizes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"Input dim: {input_dim} (Load: pd + qd)")
    print(f"Output dim: {output_dim} (alpha_non_slack: {n_gen_non_slack} + beta_gen: {n_gen})")
    print(f"Network: {input_dim} -> {' -> '.join(map(str, hidden_sizes))} -> {output_dim}")
    print(f"Activation: All Sigmoid (including output layer)")
    # [ADDED] Print early stopping config BEFORE batch_size is overwritten
    print(f"Training params: max_epochs={n_epochs}, patience={early_stop_patience}, "
          f"lr={learning_rate}, batch_size={batch_size or 'full batch'}")
    print(f"{'=' * 70}")

    # 7. Training loop
    print(f"\n{'=' * 70}")
    print(f"Training Progress")
    print(f"{'=' * 70}")

    n_train = len(X_train)
    batch_size = batch_size or n_train  # NOTE: overwrites batch_size to int here
    n_batches = (n_train + batch_size - 1) // batch_size
    train_losses, val_losses = [], []

    # [ADDED] Initialize early stopping state
    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    t0 = time.perf_counter()

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

        # [ADDED] Early stopping check
        if val_losses[-1] < best_val_loss - early_stop_min_delta:
            best_val_loss = val_losses[-1]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {train_losses[-1]:.6f} - Val Loss: {val_losses[-1]:.6f}")
                print(f"\n Early stopping triggered at epoch {epoch} (patience={early_stop_patience})")
                break

    train_time = time.perf_counter() - t0

    # [ADDED] Restore best model weights after training
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        print(f"Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")

    print(f"\nTraining completed in {train_time:.2f} seconds")

    # 8. Model evaluation
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

    test_metrics = evaluate_split_qcorrection(
        model, X_test, test_idx, test_raw_data,
        test_params, scalers, device, test_split_name, verbose=True
    )

    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_BACKUP

    # 9. Inference speed test (Two-stage timing)
    print(f"\n{'=' * 70}")
    print(f"Inference Speed Test (Two-Stage Breakdown)")
    print(f"{'=' * 70}")

    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(X_test[:1])

    n_timing_samples = min(100, len(X_test))
    stage1_times = []
    stage2_times = []
    total_times = []
    n_stage2_applied = 0

    print(f"  Testing on {n_timing_samples} samples...")

    for i in range(n_timing_samples):
        t_stage1_start = time.perf_counter()

        with torch.no_grad():
            alpha_beta_pred_single = model(X_test[i:i + 1])
        if device.type == 'cuda':
            torch.cuda.synchronize()

        alpha_beta_np = alpha_beta_pred_single.cpu().numpy()
        pg_non_slack_single, vm_gen_single = inverse_parametrize(alpha_beta_np, test_params)

        x_raw_single = scalers['x'].inverse_transform(X_test[i:i + 1].cpu().numpy())
        pd_single = x_raw_single[0, :n_loads]
        qd_single = x_raw_single[0, n_loads:]

        BASE_MVA = test_params['general']['BASE_MVA']
        qg_min_t = test_params['generator']['qg_min'].flatten()
        qg_max_t = test_params['generator']['qg_max'].flatten()
        n_gen_test = test_params['general']['n_gen']
        non_slack_gen_idx = test_params['general']['non_slack_gen_idx']
        load_bus_ids = test_params['general']['load_bus_ids']
        bus_id_to_idx = test_params['general']['bus_id_to_idx']

        mpc_pf = {
            'version': GLOBAL_CASE_DATA_TEST['version'],
            'baseMVA': BASE_MVA,
            'bus': GLOBAL_CASE_DATA_TEST['bus'].copy(),
            'gen': GLOBAL_CASE_DATA_TEST['gen'].copy(),
            'branch': GLOBAL_CASE_DATA_TEST['branch'].copy(),
            'gencost': GLOBAL_CASE_DATA_TEST['gencost']
        }

        for j, bus_id in enumerate(load_bus_ids):
            bus_idx = bus_id_to_idx.get(int(bus_id))
            if bus_idx is not None:
                mpc_pf["bus"][bus_idx, 2] = pd_single[j] * BASE_MVA
                mpc_pf["bus"][bus_idx, 3] = qd_single[j] * BASE_MVA

        for j, gen_idx in enumerate(non_slack_gen_idx):
            mpc_pf["gen"][gen_idx, 1] = pg_non_slack_single[0, j] * BASE_MVA

        for j in range(min(n_gen_test, len(mpc_pf["gen"]))):
            mpc_pf["gen"][j, 5] = vm_gen_single[0, j]

        try:
            r1_pf = runpf(mpc_pf, PPOPT)
            success = r1_pf[0]['success']
        except:
            success = False

        t_stage1_end = time.perf_counter()
        stage1_times.append(t_stage1_end - t_stage1_start)

        stage2_time = 0.0
        if success:
            gen_bus_ids_t = test_params['general']['gen_bus_ids']
            qg_pu_t = r1_pf[0]['gen'][:n_gen_test, 2] / BASE_MVA
            vmask_t = (qg_pu_t < qg_min_t) | (qg_pu_t > qg_max_t)

            if np.any(vmask_t):
                t_s2_start = time.perf_counter()
                n_stage2_applied += 1
                qg_clip_t = np.clip(qg_pu_t, qg_min_t, qg_max_t)

                mpc2_t = {
                    'version': GLOBAL_CASE_DATA_TEST['version'],
                    'baseMVA': BASE_MVA,
                    'bus': GLOBAL_CASE_DATA_TEST['bus'].copy(),
                    'gen': GLOBAL_CASE_DATA_TEST['gen'].copy(),
                    'branch': GLOBAL_CASE_DATA_TEST['branch'].copy(),
                    'gencost': GLOBAL_CASE_DATA_TEST['gencost'],
                }
                for j, bus_id in enumerate(load_bus_ids):
                    bidx = bus_id_to_idx.get(int(bus_id))
                    if bidx is not None:
                        mpc2_t['bus'][bidx, 2] = pd_single[j] * BASE_MVA
                        mpc2_t['bus'][bidx, 3] = qd_single[j] * BASE_MVA
                for j, gi in enumerate(non_slack_gen_idx):
                    mpc2_t['gen'][gi, 1] = pg_non_slack_single[0, j] * BASE_MVA
                for j in range(min(n_gen_test, len(mpc2_t['gen']))):
                    if not vmask_t[j]:
                        mpc2_t['gen'][j, 5] = vm_gen_single[0, j]

                b2g_t = defaultdict(list)
                for gi, bid in enumerate(gen_bus_ids_t[:n_gen_test]):
                    b2g_t[int(bid)].append(gi)
                for bus_id, gens_at in b2g_t.items():
                    bidx = bus_id_to_idx.get(bus_id)
                    if bidx is None or int(mpc2_t['bus'][bidx, 1]) == 3:
                        continue
                    if all(vmask_t[g] for g in gens_at):
                        for g in gens_at:
                            mpc2_t['gen'][g, 2] = float(qg_clip_t[g]) * BASE_MVA
                        mpc2_t['bus'][bidx, 1] = 1
                try:
                    runpf(mpc2_t, PPOPT)
                except Exception:
                    pass
                stage2_time = time.perf_counter() - t_s2_start

        stage2_times.append(stage2_time)
        total_times.append((t_stage1_end - t_stage1_start) + stage2_time)

    stage1_mean = np.mean(stage1_times) * 1000
    stage1_std = np.std(stage1_times) * 1000
    stage2_mean = np.mean(stage2_times) * 1000
    stage2_std = np.std(stage2_times) * 1000
    total_mean = np.mean(total_times) * 1000
    total_std = np.std(total_times) * 1000

    print(f"\n  Two-Stage Timing Breakdown:")
    print(f"    Stage 1 (NN + 1st PF):  {stage1_mean:.4f} +/- {stage1_std:.4f} ms")
    print(f"    Stage 2 (Q-correction): {stage2_mean:.4f} +/- {stage2_std:.4f} ms")
    print(f"    Total:                  {total_mean:.4f} +/- {total_std:.4f} ms")
    print(f"    Stage 1 Percentage:     {(stage1_mean / total_mean * 100):.2f}%")
    print(f"    Stage 2 Percentage:     {(stage2_mean / total_mean * 100):.2f}%")
    print(f"    Stage 2 Applied:        {n_stage2_applied}/{n_timing_samples} samples")

    latency_ms = total_mean

    # 10. Final results summary
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
    print(f"Total Inference: {latency_ms:.4f} ms/sample")
    print(f"  Stage 1 (NN + 1st PF): {stage1_mean:.4f} ms ({(stage1_mean / total_mean * 100):.1f}%)")
    print(f"  Stage 2 (Q-correction): {stage2_mean:.4f} ms ({(stage2_mean / total_mean * 100):.1f}%)")
    print(f"Training Time: {train_time:.2f} s")
    print(f"Convergence Rate: {test_metrics['convergence_rate_percent']:.2f}%")

    print(f"{'=' * 70}")

    return test_metrics


# ============================================================================
# 7. Main Program
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    paths = acopf_config.get_all_paths()

    # get_all_params() returns early_stop_patience and early_stop_min_delta
    # which are now accepted by qcorrection_nn_acopf_experiment()
    params = acopf_config.get_all_params()

    results = qcorrection_nn_acopf_experiment(**paths, **params)

    print("\nExperiment completed successfully!")