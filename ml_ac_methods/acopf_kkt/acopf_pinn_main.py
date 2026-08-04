# -*- coding: utf-8 -*-
"""
ACOPF PINN Main Experiment File (Faithful to Paper, No Worst-Case)
===================================================================
Features:
1. Physics-Informed Neural Network with KKT loss (paper Eq. 24)
2. Three independent branches: G (generation), V (voltage), Lm (duals)
3. Collocation mechanism (split from training data, discard labels)
4. Rectangular voltage [Vr, Vi] for KKT computation
5. Four data modes: random_split, fixed_valtest, generalization, api_test
6. Train once, test on three datasets (random_split + generalization + api_test)
7. Early stopping on validation loss
8. Output format identical to acopf_dnn_main.py
9. No file saving (JSON/CSV/model)

Evaluation pipeline:
  - Extract pg_non_slack from G branch output
  - Convert [Vr, Vi] → Vm at generator buses
  - Reconstruct full Pg, run PyPower power flow
  - Evaluate with acopf_violation_metrics.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

from acopf_pinnmodel import PinnModel

GLOBAL_CASE_DATA = None
PPOPT = None


# =====================================================================
# PyPower Helpers (identical to DNN version)
# =====================================================================
def init_pypower_options():
    global PPOPT
    ppopt = ppoption()
    PPOPT = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)


def load_case_from_csv(case_name, constraints_path):
    """Load PyPower case data from CSV files."""
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


def solve_pf_custom_optimized(pd, qd, pg_non_slack, vm_gen, params):
    """Run power flow: accepts non-slack Pg and generator Vm."""
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

    load_bus_ids = params['general']['load_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc_pf["bus"][bus_idx, 2] = pd[i] * BASE_MVA
            mpc_pf["bus"][bus_idx, 3] = qd[i] * BASE_MVA

    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    n_gen = params['general']['n_gen']
    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc_pf["gen"][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    for i in range(n_gen):
        mpc_pf["gen"][i, 5] = vm_gen[i]

    return runpf(mpc_pf, PPOPT)


# =====================================================================
# Data Loading: Duals + Rectangular Voltage
# =====================================================================
def load_dual_variables(data_dir, case_name, n_samples):
    """
    Load dual variable CSV files from _with_duals directory.

    Returns dict of numpy arrays, or None if files not found.
    """
    duals_dir = data_dir.rstrip('/\\') + '_with_duals'
    if not os.path.isdir(duals_dir):
        print(f"⚠️ Duals directory not found: {duals_dir}")
        return None

    dual_files = {
        'mu_pg_min': 'mu_pg_min', 'mu_pg_max': 'mu_pg_max',
        'mu_qg_min': 'mu_qg_min', 'mu_qg_max': 'mu_qg_max',
        'mu_vm_min': 'mu_vm_min', 'mu_vm_max': 'mu_vm_max',
        'lambda_kcl_r': 'lambda_kcl_r', 'lambda_kcl_i': 'lambda_kcl_i',
        'mu_sm_fr': 'mu_sm_fr', 'mu_sm_to': 'mu_sm_to',
    }

    duals = {}
    for key, fname in dual_files.items():
        fpath = os.path.join(duals_dir, f"{case_name}_{fname}.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            duals[key] = df.values.astype('float32')[:n_samples]
        else:
            print(f"  ⚠️ Missing dual file: {fpath}")
            return None

    print(f"✓ Dual variables loaded from: {duals_dir}")
    for k, v in duals.items():
        nz = np.sum(np.abs(v) > 1e-6)
        print(f"    {k}: shape={v.shape}, non-zero={nz}")

    return duals


def compute_rect_voltage(vm, va_rad):
    """Convert polar (vm, va) → rectangular (Vr, Vi)."""
    Vr = vm * np.cos(va_rad)
    Vi = vm * np.sin(va_rad)
    return np.hstack([Vr, Vi]).astype('float32')


def prepare_pinn_targets(raw_data, indices, params, duals, duals_available=True):
    """
    Prepare PINN training targets from raw data and duals.

    Returns dict of numpy arrays (unscaled, in p.u.).
    """
    non_slack_gen_idx = params['general']['non_slack_gen_idx']

    # G targets: [pg_non_slack, qg]
    pg_ns = raw_data['pg_non_slack'][indices]  # (n, n_gen_non_slack)
    qg = raw_data['qg'][indices]  # (n, n_gen)
    pg_qg = np.hstack([pg_ns, qg]).astype('float32')

    # V targets: [Vr, Vi] from vm (all buses) and va (all buses)
    vm_all = raw_data['vm'][indices]  # (n, n_buses)
    va_rad = raw_data['va'][indices]  # (n, n_buses) in radians
    v_rect = compute_rect_voltage(vm_all, va_rad)

    targets = {'pg_qg': pg_qg, 'v_rect': v_rect}

    # Dual targets
    if duals_available and duals is not None:
        n_buses = params['general']['n_buses']
        n_gen = params['general']['n_gen']
        n_gen_ns = params['general']['n_gen_non_slack']

        # lambda_p: [lambda_kcl_r, lambda_kcl_i]  → (n, 2*n_buses)
        lam_r = duals['lambda_kcl_r'][indices]
        lam_i = duals['lambda_kcl_i'][indices]
        targets['lambda_p'] = np.hstack([lam_r, lam_i]).astype('float32')

        # mu_g_u: [mu_pg_max (non-slack), mu_qg_max (all)]
        mu_pg_max = duals['mu_pg_max'][indices][:, non_slack_gen_idx]
        mu_qg_max = duals['mu_qg_max'][indices]
        targets['mu_g_u'] = np.hstack([mu_pg_max, mu_qg_max]).astype('float32')

        # mu_g_d: [mu_pg_min (non-slack), mu_qg_min (all)]
        mu_pg_min = duals['mu_pg_min'][indices][:, non_slack_gen_idx]
        mu_qg_min = duals['mu_qg_min'][indices]
        targets['mu_g_d'] = np.hstack([mu_pg_min, mu_qg_min]).astype('float32')

        # mu_v_u / mu_v_d
        targets['mu_v_u'] = duals['mu_vm_max'][indices].astype('float32')
        targets['mu_v_d'] = duals['mu_vm_min'][indices].astype('float32')

        # mu_sm_fr / mu_sm_to
        targets['mu_sm_fr'] = duals['mu_sm_fr'][indices].astype('float32')
        targets['mu_sm_to'] = duals['mu_sm_to'][indices].astype('float32')
    else:
        # No duals: fill with zeros (will be masked out for collocation anyway)
        n = len(indices)
        n_buses = params['general']['n_buses']
        n_gen_ns = params['general']['n_gen_non_slack']
        n_gen = params['general']['n_gen']
        n_br = params['general']['n_branches']
        targets['lambda_p'] = np.zeros((n, 2 * n_buses), dtype='float32')
        targets['mu_g_u'] = np.zeros((n, n_gen_ns + n_gen), dtype='float32')
        targets['mu_g_d'] = np.zeros((n, n_gen_ns + n_gen), dtype='float32')
        targets['mu_v_u'] = np.zeros((n, n_buses), dtype='float32')
        targets['mu_v_d'] = np.zeros((n, n_buses), dtype='float32')
        targets['mu_sm_fr'] = np.zeros((n, n_br), dtype='float32')
        targets['mu_sm_to'] = np.zeros((n, n_br), dtype='float32')

    return targets


# =====================================================================
# Evaluation (identical logic to DNN version)
# =====================================================================
def evaluate_split(model, X, indices, raw_data, params, device, split_name, verbose=True):
    """Evaluate model on a dataset split — DNN-direct + After PF metrics."""
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()
    with torch.no_grad():
        pg_non_slack_pred, vm_gen_pred, vm_all_pred, va_all_pred, qg_all_pred = \
            model.predict_for_evaluation(X.to(device))

    pg_non_slack_np = pg_non_slack_pred.cpu().numpy()
    vm_gen_np = vm_gen_pred.cpu().numpy()
    vm_all_np = vm_all_pred.cpu().numpy()
    va_all_np = va_all_pred.cpu().numpy()   # radians
    qg_all_np = qg_all_pred.cpu().numpy()

    if verbose:
        print(f"  [DEBUG] pg_non_slack_pred: mean={pg_non_slack_np.mean():.6f}, "
              f"std={pg_non_slack_np.std():.6f}, "
              f"range=[{pg_non_slack_np.min():.6f}, {pg_non_slack_np.max():.6f}]")
        print(f"  [DEBUG] pg_non_slack TRUE:  mean={raw_data['pg_non_slack'][indices].mean():.6f}, "
              f"range=[{raw_data['pg_non_slack'][indices].min():.6f}, {raw_data['pg_non_slack'][indices].max():.6f}]")
        print(f"  [DEBUG] vm_gen_pred:        mean={vm_gen_np.mean():.6f}, "
              f"range=[{vm_gen_np.min():.6f}, {vm_gen_np.max():.6f}]")
        print(f"  [DEBUG] vm_gen TRUE:        mean={raw_data['vm_gen'][indices].mean():.6f}, "
              f"range=[{raw_data['vm_gen'][indices].min():.6f}, {raw_data['vm_gen'][indices].max():.6f}]")

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Reconstruct full Pg (with slack = 0 placeholder)
    y_pred_pg_full = reconstruct_full_pg(pg_non_slack_np, params)

    # Reconstruct full Vm for PF (only generator buses from prediction)
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_for_pf = np.ones((len(X), n_buses), dtype=np.float32)
    y_pred_vm_for_pf[:, gen_bus_indices] = vm_gen_np

    # True values
    y_true_pg = raw_data['pg'][indices]
    y_true_vm = raw_data['vm'][indices]
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

    # Input data (raw p.u.)
    x_raw = raw_data['x'][indices]
    pd_pu = x_raw[:, :n_loads]
    qd_pu = x_raw[:, n_loads:]

    # ================================================================
    # Run power flow
    # ================================================================
    n_samples = len(X)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_custom_optimized(
                pd_pu[i], qd_pu[i],
                pg_non_slack_np[i], vm_gen_np[i],
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

    # ================================================================
    # After PF metrics (Table 4-5 style)
    # ================================================================
    pf_metrics = evaluate_acopf_predictions(
        y_pred_pg_full, y_pred_vm_for_pf,
        y_true_pg, y_true_vm, y_true_qg, y_true_va_rad,
        pf_results_list, converge_flags, params, verbose=verbose
    )

    # ================================================================
    # DNN-direct metrics (Table 3 style)
    # ================================================================
    from acopf_violation_metrics import compute_mae_percentage, compute_mae_absolute

    # MAE — directly compare DNN outputs vs ground truth
    dnn_mae_pg = compute_mae_percentage(
        y_true_pg[:, ~params['general']['slack_gen_mask']], pg_non_slack_np)
    dnn_mae_vm = compute_mae_percentage(y_true_vm[:, gen_bus_indices],
                                         vm_all_np[:, gen_bus_indices])
    dnn_mae_qg = compute_mae_percentage(y_true_qg, qg_all_np)

    y_true_va_deg = y_true_va_rad * (180.0 / np.pi)
    y_pred_va_deg = va_all_np * (180.0 / np.pi)
    dnn_mae_va = compute_mae_absolute(y_true_va_deg, y_pred_va_deg)

    # Direct violations from DNN predictions (no PF involved)
    pg_min = params['generator']['pg_min'].flatten()
    pg_max = params['generator']['pg_max'].flatten()
    qg_min = params['generator']['qg_min'].flatten()
    qg_max = params['generator']['qg_max'].flatten()
    vm_min = params['bus']['vm_min']
    vm_max = params['bus']['vm_max']

    # Pg violation (non-slack only)
    ns_idx = params['general']['non_slack_gen_idx']
    pg_viol = np.maximum(0, pg_min[ns_idx] - pg_non_slack_np) + \
              np.maximum(0, pg_non_slack_np - pg_max[ns_idx])
    dnn_pg_viol = float(np.mean(np.max(pg_viol, axis=1)))

    # Pg violation (slack) — slack Pg comes from PF
    slack_gen_mask = params['general']['slack_gen_mask']
    slack_gen_idx = np.where(slack_gen_mask)[0]
    if len(slack_gen_idx) > 0 and len(pf_results_list) > 0:
        slack_pg_viols = []
        for i in range(n_samples):
            if converge_flags[i]:
                gen = pf_results_list[i][0]['gen']
                pg_mw = gen[:, 1]
                pg_min_mw = gen[:, 9]
                pg_max_mw = gen[:, 8]
                viols_mw = np.maximum(0, pg_min_mw - pg_mw) + np.maximum(0, pg_mw - pg_max_mw)
                viols_pu = viols_mw / params['general']['BASE_MVA']
                slack_pg_viols.append(viols_pu[slack_gen_idx[0]])
        dnn_pg_slack_viol = float(np.mean(slack_pg_viols)) if slack_pg_viols else 0.0
    else:
        dnn_pg_slack_viol = 0.0

    # DNN-direct cost gap — same as After PF (DNN cannot predict slack Pg)
    dnn_cost_gap = pf_metrics.get('cost_optimality_gap_percent', float('nan'))

    # Qg violation (all generators)
    qg_viol = np.maximum(0, qg_min - qg_all_np) + np.maximum(0, qg_all_np - qg_max)
    dnn_qg_viol = float(np.mean(np.max(qg_viol, axis=1)))

    # Vm violation (all buses)
    vm_viol = np.maximum(0, vm_min - vm_all_np) + np.maximum(0, vm_all_np - vm_max)
    dnn_vm_viol = float(np.mean(np.max(vm_viol, axis=1)))

    # Branch violation from predicted (vm, va) via Ohm's Law (power form)
    # Derive bus indices and branch admittance from raw params
    f_bus = params['branch']['f_bus']
    t_bus = params['branch']['t_bus']
    f_idx = np.array([bus_id_to_idx[int(b)] for b in f_bus])
    t_idx = np.array([bus_id_to_idx[int(b)] for b in t_bus])

    r_pu = params['branch']['r_pu'].astype(np.float64)
    x_pu = params['branch']['x_pu'].astype(np.float64)
    z_sq = r_pu**2 + x_pu**2
    z_sq[z_sq < 1e-20] = 1e-20
    g_br = (r_pu / z_sq).astype(np.float32)
    b_br = (-x_pu / z_sq).astype(np.float32)

    rate_a = params['branch']['rate_a'].copy()
    rate_a[rate_a <= 0] = 1e6

    vi = vm_all_np[:, f_idx]
    vj = vm_all_np[:, t_idx]
    ai = va_all_np[:, f_idx]
    aj = va_all_np[:, t_idx]
    theta_ij = ai - aj
    pf_flow = g_br * vi**2 - vi * vj * (b_br * np.sin(theta_ij) + g_br * np.cos(theta_ij))
    qf_flow = -b_br * vi**2 - vi * vj * (g_br * np.sin(theta_ij) - b_br * np.cos(theta_ij))
    sf_sq = pf_flow**2 + qf_flow**2
    branch_viol = np.maximum(0, sf_sq - rate_a**2)
    dnn_branch_viol = float(np.mean(np.max(branch_viol, axis=1)))

    dnn_metrics = {
        'dnn_mae_pg_percent': dnn_mae_pg,
        'dnn_mae_vm_percent': dnn_mae_vm,
        'dnn_mae_qg_percent': dnn_mae_qg,
        'dnn_mae_va_deg': dnn_mae_va,
        'dnn_pg_viol_pu': dnn_pg_viol,
        'dnn_pg_slack_viol_pu': dnn_pg_slack_viol,
        'dnn_qg_viol_pu': dnn_qg_viol,
        'dnn_vm_viol_pu': dnn_vm_viol,
        'dnn_branch_viol_pu': dnn_branch_viol,
        'dnn_cost_gap_percent': dnn_cost_gap,
    }

    # Merge both metric sets
    metrics = {**pf_metrics, **dnn_metrics}
    return metrics


def print_results(test_metrics, data_mode, case_name, train_time, latency_ms):
    """Print results: DNN-direct (Table 3 style) + PF-based (Table 4-5 style)."""
    print(f"\n  ┌─── {data_mode} | {case_name} ───")
    print(f"  │")
    print(f"  │ [DNN Prediction] (直接从预测值评估, 对标原文Table 3)")
    print(f"  │   MAE: Pg={test_metrics['dnn_mae_pg_percent']:.4f}%  "
          f"Vm={test_metrics['dnn_mae_vm_percent']:.4f}%  "
          f"Qg={test_metrics['dnn_mae_qg_percent']:.4f}%  "
          f"Va={test_metrics['dnn_mae_va_deg']:.4f}°")
    print(f"  │   Viol: Pg(ns)={test_metrics['dnn_pg_viol_pu']:.6f}  "
          f"Pg(slack)={test_metrics['dnn_pg_slack_viol_pu']:.6f}  "
          f"Qg={test_metrics['dnn_qg_viol_pu']:.6f}  "
          f"Vm={test_metrics['dnn_vm_viol_pu']:.6f}  "
          f"Branch={test_metrics['dnn_branch_viol_pu']:.6f}")
    print(f"  │   Cost gap: {test_metrics['dnn_cost_gap_percent']:.4f}%")
    print(f"  │")
    print(f"  │ [After Load Flow] (PF恢复后评估, 对标原文Table 4-5)")
    print(f"  │   Convergence: {test_metrics['convergence_rate_percent']:.1f}% "
          f"({test_metrics['n_converged']}/{test_metrics['n_samples']})")
    print(f"  │   MAE: Pg(ns)={test_metrics['mae_pg_non_slack_percent']:.4f}%  "
          f"Vm={test_metrics['mae_vm_percent']:.4f}%  "
          f"Qg={test_metrics['mae_qg_percent']:.4f}%  "
          f"Va={test_metrics['mae_va_deg']:.4f}°")
    print(f"  │   Viol: Pg(ns)={test_metrics['mean_pg_viol_non_slack_pu']:.6f}  "
          f"Pg(slack)={test_metrics['mean_pg_viol_slack_pu']:.6f}  "
          f"Qg={test_metrics['mean_max_qg_viol_pu']:.6f}  "
          f"Vm={test_metrics['mean_max_vm_viol_pu']:.6f}  "
          f"Branch={test_metrics['mean_max_branch_viol_pu']:.6f}")
    print(f"  │   Cost gap: {test_metrics['cost_optimality_gap_percent']:.4f}%")
    print(f"  │")
    print(f"  │ [Performance]")
    print(f"  │   Inference: {latency_ms:.4f} ms/sample  |  Training: {train_time:.2f} s")
    print(f"  └{'─' * 60}")


# =====================================================================
# Main Experiment
# =====================================================================
def acopf_pinn_experiment(
        case_name, params_path, data_path, log_path, results_path,
        data_mode='random_split',
        n_train_use=None, test_data_path=None, test_params_path=None,
        n_test_samples=None, seed=42, n_epochs=1000,
        early_stop_patience=20, early_stop_min_delta=1e-6,
        learning_rate=0.001,
        hidden_sizes_V=None, hidden_sizes_G=None, hidden_sizes_Lg=None,
        lambda_P=1.0, lambda_V=1.0, lambda_L=1e-3, lambda_eps=1e-2,
        collocation_ratio=0.5,
        batch_size=None, device='cuda', tolerances=None
):
    global GLOBAL_CASE_DATA, PPOPT
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"ACOPF PINN Experiment")
    print(f"{'=' * 70}")
    print(f"Device: {device_obj}")
    print(f"Case: {case_name}")
    print(f"Data Mode: {data_mode}")
    print(f"{'=' * 70}")

    # ========================================================================
    # 1. Load params and PyPower case
    # ========================================================================
    params = load_parameters_from_csv(case_name, params_path)
    init_pypower_options()
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)

    # Add network structure to params
    params['training'] = {
        'neurons_in_hidden_layers_V': hidden_sizes_V,
        'neurons_in_hidden_layers_G': hidden_sizes_G,
        'neurons_in_hidden_layers_Lg': hidden_sizes_Lg,
    }

    # ========================================================================
    # 2. Load training data
    # ========================================================================
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']

    print(f"\n[Training Data Info]")
    print(f"  Buses: {n_buses}, Gens: {n_gen} (Non-Slack: {n_gen_non_slack}), Loads: {n_loads}")

    # ========================================================================
    # 3. Load dual variables
    # ========================================================================
    data_dir = os.path.dirname(data_path)
    base_filename = os.path.basename(data_path)
    if base_filename.endswith('_pd.csv'):
        dual_case_name = base_filename[:-7]
    else:
        dual_case_name = base_filename.rsplit('_', 1)[0]

    duals = load_dual_variables(data_dir, dual_case_name, len(x_data_scaled))
    duals_available = duals is not None

    # ========================================================================
    # 4. Data splitting
    # ========================================================================
    # For combined mode (random_split + generalization + api_test): use random_split
    if data_mode == 'combined':
        split_mode = DataMode.RANDOM_SPLIT
    else:
        split_mode = data_mode

    # Prepare test datasets dict for multi-test
    test_configs = {}

    if data_mode == 'combined':
        # Train once, test on 3 datasets
        train_idx, val_idx, test_idx_rs = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=DataMode.RANDOM_SPLIT, n_train_use=n_train_use, seed=seed
        )
        test_configs['random_split'] = {
            'x': raw_data['x'], 'idx': test_idx_rs,
            'raw': raw_data, 'params': params, 'case_data': GLOBAL_CASE_DATA
        }

        # Generalization test
        if test_data_path:
            gen_x, gen_y, gen_raw, _ = load_generalization_test_data(
                test_data_path, params, scalers, n_test_samples=n_test_samples, seed=seed)
            test_configs['generalization'] = {
                'x': gen_raw['x'], 'idx': np.arange(len(gen_raw['x'])),
                'raw': gen_raw, 'params': params, 'case_data': GLOBAL_CASE_DATA
            }

        # API test
        if test_params_path:
            api_data_path = acopf_config.get_data_path(acopf_config.TEST_CASE, None)
            api_params, api_x, api_y, api_raw, _ = load_api_test_data(
                api_data_path, test_params_path, scalers,
                n_test_samples=n_test_samples, seed=seed)
            api_case_name = os.path.basename(api_data_path)
            if api_case_name.endswith('_pd.csv'):
                api_case_name = api_case_name[:-7]
            api_case_data = load_case_from_csv(api_case_name, test_params_path)
            test_configs['api_test'] = {
                'x': api_raw['x'], 'idx': np.arange(len(api_raw['x'])),
                'raw': api_raw, 'params': api_params, 'case_data': api_case_data
            }

    elif data_mode in [DataMode.RANDOM_SPLIT, 'random_split']:
        train_idx, val_idx, test_idx_rs = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=DataMode.RANDOM_SPLIT, n_train_use=n_train_use, seed=seed)
        test_configs['random_split'] = {
            'x': raw_data['x'], 'idx': test_idx_rs,
            'raw': raw_data, 'params': params, 'case_data': GLOBAL_CASE_DATA
        }

    elif data_mode in [DataMode.FIXED_VALTEST, 'fixed_valtest']:
        train_idx, val_idx, test_idx_rs = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=DataMode.FIXED_VALTEST, n_train_use=n_train_use, seed=seed)
        test_configs['fixed_valtest'] = {
            'x': raw_data['x'], 'idx': test_idx_rs,
            'raw': raw_data, 'params': params, 'case_data': GLOBAL_CASE_DATA
        }

    elif data_mode in [DataMode.GENERALIZATION, 'generalization']:
        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=DataMode.GENERALIZATION, n_train_use=n_train_use, seed=seed)
        gen_x, gen_y, gen_raw, _ = load_generalization_test_data(
            test_data_path, params, scalers, n_test_samples=n_test_samples, seed=seed)
        test_configs['generalization'] = {
            'x': gen_raw['x'], 'idx': np.arange(len(gen_raw['x'])),
            'raw': gen_raw, 'params': params, 'case_data': GLOBAL_CASE_DATA
        }

    elif data_mode in [DataMode.API_TEST, 'api_test']:
        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=DataMode.API_TEST, n_train_use=n_train_use, seed=seed)
        api_params, api_x, api_y, api_raw, _ = load_api_test_data(
            test_data_path, test_params_path, scalers,
            n_test_samples=n_test_samples, seed=seed)
        api_case_name = os.path.basename(test_data_path)
        if api_case_name.endswith('_pd.csv'):
            api_case_name = api_case_name[:-7]
        api_case_data = load_case_from_csv(api_case_name, test_params_path)
        test_configs['api_test'] = {
            'x': api_raw['x'], 'idx': np.arange(len(api_raw['x'])),
            'raw': api_raw, 'params': api_params, 'case_data': api_case_data
        }

    # ========================================================================
    # 5. Split train into supervised + collocation
    # ========================================================================
    rng = np.random.default_rng(seed + 999)
    n_train_total = len(train_idx)
    n_collocation = int(n_train_total * collocation_ratio)
    n_supervised = n_train_total - n_collocation

    perm = rng.permutation(n_train_total)
    supervised_idx = train_idx[perm[:n_supervised]]
    collocation_idx = train_idx[perm[n_supervised:]]

    print(f"\n[Collocation Split]")
    print(f"  Total train: {n_train_total}")
    print(f"  Supervised:  {n_supervised} ({100*(1-collocation_ratio):.0f}%)")
    print(f"  Collocation: {n_collocation} ({100*collocation_ratio:.0f}%)")

    # ========================================================================
    # 6. Prepare PINN targets (p.u., unscaled)
    # ========================================================================
    # Supervised targets
    sup_targets = prepare_pinn_targets(raw_data, supervised_idx, params, duals, duals_available)
    # Collocation targets (zeros — will be masked out)
    col_targets = prepare_pinn_targets(raw_data, collocation_idx, params, None, False)

    # Input data (raw p.u.) for KKT computation
    X_sup_raw = raw_data['x'][supervised_idx]
    X_col_raw = raw_data['x'][collocation_idx]

    # Combine supervised + collocation
    X_train_raw = np.vstack([X_sup_raw, X_col_raw]).astype('float32')
    mask_train = np.vstack([
        np.ones((n_supervised, 1), dtype='float32'),
        np.zeros((n_collocation, 1), dtype='float32')
    ])

    all_targets = {}
    for key in sup_targets:
        all_targets[key] = np.vstack([sup_targets[key], col_targets[key]])

    # Validation targets (all supervised)
    val_targets = prepare_pinn_targets(raw_data, val_idx, params, duals, duals_available)
    X_val_raw = raw_data['x'][val_idx]

    # Convert to tensors
    X_train_t = torch.tensor(X_train_raw, dtype=torch.float32, device=device_obj)
    mask_t = torch.tensor(mask_train, dtype=torch.float32, device=device_obj)
    targets_t = {k: torch.tensor(v, dtype=torch.float32, device=device_obj) for k, v in all_targets.items()}

    X_val_t = torch.tensor(X_val_raw, dtype=torch.float32, device=device_obj)
    val_targets_t = {k: torch.tensor(v, dtype=torch.float32, device=device_obj) for k, v in val_targets.items()}
    val_mask_t = torch.ones(len(X_val_t), 1, dtype=torch.float32, device=device_obj)

    print(f"\n[Dataset Sizes]")
    print(f"  Train (sup+col): {len(X_train_t)} samples")
    print(f"  Val: {len(X_val_t)} samples")
    for name, cfg in test_configs.items():
        print(f"  Test ({name}): {len(cfg['idx'])} samples")

    # ========================================================================
    # 7. Create PINN model
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"V Network: {hidden_sizes_V}")
    print(f"G Network: {hidden_sizes_G}")
    print(f"Lg Network: {hidden_sizes_Lg}")
    print(f"Loss weights: Λ_P={lambda_P}, Λ_V={lambda_V}, Λ_L={lambda_L}, Λ_ε={lambda_eps}")
    print(f"{'=' * 70}")

    model = PinnModel(
        simulation_parameters=params,
        lambda_P=lambda_P, lambda_V=lambda_V,
        lambda_L=lambda_L, lambda_eps=lambda_eps,
        collocation_ratio=collocation_ratio,
        learning_rate=learning_rate, device=device_obj
    ).to(device_obj)

    # ========================================================================
    # 8. Training loop
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Training Progress")
    print(f"{'=' * 70}")

    n_train = len(X_train_t)
    batch_size = batch_size or n_train
    n_batches = (n_train + batch_size - 1) // batch_size
    train_losses, val_losses = [], []
    t0 = time.perf_counter()

    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_kkt = 0.0
        epoch_mae_g = 0.0
        epoch_mae_v = 0.0
        indices = torch.randperm(n_train, device=device_obj)

        for i in range(n_batches):
            batch_idx = indices[i * batch_size:min((i + 1) * batch_size, n_train)]
            X_batch = X_train_t[batch_idx]
            mask_batch = mask_t[batch_idx]
            tgt_batch = {k: v[batch_idx] for k, v in targets_t.items()}

            model.optimizer.zero_grad()
            outputs = model(X_batch)
            total_loss, loss_dict = model.compute_loss(outputs, tgt_batch, mask_batch)
            total_loss.backward()
            model.optimizer.step()

            bs = len(batch_idx)
            epoch_loss += total_loss.item() * bs
            epoch_kkt += loss_dict['mae_eps'] * bs
            epoch_mae_g += loss_dict['mae_g'] * bs
            epoch_mae_v += loss_dict['mae_v'] * bs

        avg_loss = epoch_loss / n_train
        avg_kkt = epoch_kkt / n_train
        avg_mae_g = epoch_mae_g / n_train
        avg_mae_v = epoch_mae_v / n_train
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss, _ = model.compute_loss(val_out, val_targets_t, val_mask_t)
        val_losses.append(val_loss.item())

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Epoch {epoch:4d}/{n_epochs} - Total: {avg_loss:.6f} - Val: {val_losses[-1]:.6f} | "
                  f"mae_g: {avg_mae_g:.6f} mae_v: {avg_mae_v:.6f} KKT: {avg_kkt:.4f}")

        # Early stopping
        if val_losses[-1] < best_val_loss - early_stop_min_delta:
            best_val_loss = val_losses[-1]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Epoch {epoch:4d}/{n_epochs} - Total: {avg_loss:.6f} - Val: {val_losses[-1]:.6f} | "
                      f"mae_g: {avg_mae_g:.6f} mae_v: {avg_mae_v:.6f} KKT: {avg_kkt:.4f}")
                print(f"\n⚡ Early stopping at epoch {epoch} (patience={early_stop_patience})")
                break

    model.load_state_dict({k: v.to(device_obj) for k, v in best_state_dict.items()})
    print(f"✓ Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")

    train_time = time.perf_counter() - t0
    print(f"✓ Training completed in {train_time:.2f} seconds")

    # ========================================================================
    # 9. Evaluate on all test sets
    # ========================================================================
    GLOBAL_CASE_DATA_BACKUP = GLOBAL_CASE_DATA

    # Inference speed test
    model.eval()
    with torch.no_grad():
        # Pick any test set for speed test
        first_cfg = next(iter(test_configs.values()))
        speed_x = torch.tensor(first_cfg['x'][:1], dtype=torch.float32, device=device_obj)
        for _ in range(10):
            model.predict_for_evaluation(speed_x)

    times = [time.perf_counter() for _ in range(101)]
    with torch.no_grad():
        for i in range(100):
            model.predict_for_evaluation(speed_x)
            if device_obj.type == 'cuda':
                torch.cuda.synchronize()
            times[i + 1] = time.perf_counter()
    latency_ms = np.mean(np.diff(times)) * 1000

    all_metrics = {}
    for test_name, cfg in test_configs.items():
        print(f"\n{'=' * 70}")
        print(f"Test Set Evaluation: {test_name}")
        print(f"{'=' * 70}")

        GLOBAL_CASE_DATA = cfg['case_data']
        X_test = torch.tensor(cfg['x'][cfg['idx']], dtype=torch.float32, device=device_obj)

        metrics = evaluate_split(
            model, X_test, cfg['idx'], cfg['raw'],
            cfg['params'], device_obj, test_name, verbose=True
        )
        all_metrics[test_name] = metrics
        print_results(metrics, test_name, case_name, train_time, latency_ms)

    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_BACKUP
    return all_metrics


# =====================================================================
# Entry Point
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    paths = acopf_config.get_all_paths()
    config_params = acopf_config.get_all_params()
    pinn_params = acopf_config.get_pinn_params()

    # Remove DNN-specific params that PINN doesn't use directly
    config_params.pop('hidden_sizes', None)

    # Extract early stopping params (shared between DNN and PINN)
    early_stop_patience = config_params.pop('early_stop_patience', 20)
    early_stop_min_delta = config_params.pop('early_stop_min_delta', 1e-6)

    print(f"\nPINN Network Structures:")
    print(f"  V Network:  {pinn_params['hidden_sizes_V']}")
    print(f"  G Network:  {pinn_params['hidden_sizes_G']}")
    print(f"  Lg Network: {pinn_params['hidden_sizes_Lg']}")
    print(f"\nPINN Loss Weights:")
    print(f"  Λ_P={pinn_params['lambda_P']}, Λ_V={pinn_params['lambda_V']}, "
          f"Λ_L={pinn_params['lambda_L']}, Λ_ε={pinn_params['lambda_eps']}")
    print(f"  Collocation ratio: {pinn_params['collocation_ratio']}")

    results = acopf_pinn_experiment(
        **paths,
        **config_params,
        **pinn_params,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )

    print("\n✓ PINN Experiment completed successfully!")