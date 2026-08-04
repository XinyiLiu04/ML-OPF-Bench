# -*- coding: utf-8 -*-
"""
Spectral GNN Utility Functions for ACOPF
— Paper-faithful sub-optimal state edition —

Key changes:
    1. collate_graph_batch() now builds node features from sub-optimal state
       X = [vm, va, p_inj, q_inj] instead of load-only features.
    2. load_subopt_state() loads pre-computed sub-optimal state CSVs.
    3. evaluate_split() adapted to new input format.
    4. PyPower interface (load_case_from_csv, solve_pf_custom_optimized,
       init_pypower_options, build_adjacency_edge_weight) unchanged.
"""

import numpy as np
import pandas as pd
import re
import torch
from pathlib import Path
from pypower.runpf import runpf
from pypower.ppoption import ppoption

from acopf_violation_metrics import evaluate_acopf_predictions
from acopf_data_setup import reconstruct_full_pg
from gnn_spectral_model_v2 import build_node_features_subopt


# =====================================================================
# Global state
# =====================================================================
GLOBAL_CASE_DATA = None
PPOPT = None


# =====================================================================
# Load sub-optimal state data
# =====================================================================
def load_subopt_state(data_dir, case_name, params):
    """
    Load sub-optimal state CSVs produced by generate_subopt_state.py.

    Returns:
        subopt_x_raw : np.ndarray [n_samples, 4*n_buses]
                       columns: [vm_all | va_all | pinj_all | qinj_all]
        converged    : np.ndarray [n_samples] bool
    """
    bus_ids = params['general']['bus_ids']
    n_buses = params['general']['n_buses']

    vm_df   = pd.read_csv(f"{data_dir}/{case_name}_subopt_vm.csv")
    va_df   = pd.read_csv(f"{data_dir}/{case_name}_subopt_va.csv")
    pinj_df = pd.read_csv(f"{data_dir}/{case_name}_subopt_pinj.csv")
    qinj_df = pd.read_csv(f"{data_dir}/{case_name}_subopt_qinj.csv")
    conv_df = pd.read_csv(f"{data_dir}/{case_name}_subopt_converged.csv")

    # Extract columns in bus_id order
    vm_cols   = [f"vm_{int(bid)}" for bid in bus_ids]
    va_cols   = [f"va_{int(bid)}" for bid in bus_ids]
    pinj_cols = [f"pinj_{int(bid)}" for bid in bus_ids]
    qinj_cols = [f"qinj_{int(bid)}" for bid in bus_ids]

    vm_raw   = vm_df[vm_cols].values.astype('float32')
    va_raw   = va_df[va_cols].values.astype('float32')
    pinj_raw = pinj_df[pinj_cols].values.astype('float32')
    qinj_raw = qinj_df[qinj_cols].values.astype('float32')

    # Stack: [n_samples, 4*n_buses]
    subopt_x_raw = np.hstack([vm_raw, va_raw, pinj_raw, qinj_raw])
    converged = conv_df['converged'].values.astype(bool)

    return subopt_x_raw, converged


# =====================================================================
# Graph structure (unchanged)
# =====================================================================
def build_adjacency_edge_weight(params, kernel='gaussian', scale_k=1.0,
                                threshold=0.0):
    """
    Build bidirectional COO edge list and scalar edge weights for ChebConv.

    Gaussian kernel on branch impedance magnitude:
        w_ij = exp(-k * |z_ij|^2),  z_ij = r_ij + j*x_ij
    """
    bus_id_to_idx = params['general']['bus_id_to_idx']
    f_bus = params['branch']['f_bus']
    t_bus = params['branch']['t_bus']
    r_pu  = params['branch']['r_pu']
    x_pu  = params['branch']['x_pu']

    src_list, dst_list, w_list = [], [], []

    for i in range(len(f_bus)):
        fi = bus_id_to_idx[int(f_bus[i])]
        ti = bus_id_to_idx[int(t_bus[i])]

        if kernel == 'gaussian':
            z_mag_sq = float(r_pu[i]) ** 2 + float(x_pu[i]) ** 2
            w = float(np.exp(-scale_k * z_mag_sq))
        else:
            w = 1.0

        if w <= threshold:
            continue

        src_list += [fi, ti]
        dst_list += [ti, fi]
        w_list   += [w,  w]

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(w_list, dtype=torch.float32)

    n_buses = params['general']['n_buses']
    print(f"\n✓ Graph structure constructed (spectral):")
    print(f"  Nodes       : {n_buses}")
    print(f"  Edges       : {edge_index.shape[1]} (bidirectional)")
    print(f"  Edge kernel : {kernel} (k={scale_k})")
    if len(w_list) > 0:
        print(f"  w range     : [{min(w_list):.4f}, {max(w_list):.4f}]")

    return edge_index, edge_weight


# =====================================================================
# Batched-graph construction (adapted for sub-optimal state)
# =====================================================================
def collate_graph_batch(x_scaled_batch, edge_index, edge_weight,
                        n_buses, device):
    """
    Build disjoint-graph batch from scaled sub-optimal state vectors.

    Args:
        x_scaled_batch : Tensor [B, 4*N]  scaled sub-optimal state
        edge_index     : LongTensor  [2, E]  single-graph edges
        edge_weight    : FloatTensor [E]     single-graph weights
        n_buses        : int
        device         : torch.device

    Returns:
        node_feats  : Tensor [B*N, 4]
        batch_ei    : Tensor [2, B*E]
        batch_ew    : Tensor [B*E]
        B           : int
    """
    B = x_scaled_batch.shape[0]
    N = n_buses
    E = edge_index.shape[1]

    # 1. Node features [B*N, 4]
    node_feats = build_node_features_subopt(x_scaled_batch, N, device)

    # 2. Edge index: replicate + offset
    offsets  = (torch.arange(B, device=device) * N).view(B, 1, 1)
    batch_ei = (edge_index.to(device).unsqueeze(0) + offsets)  # [B, 2, E]
    batch_ei = batch_ei.reshape(2, B * E)

    # 3. Edge weight: simple repeat
    batch_ew = edge_weight.to(device).repeat(B)   # [B*E]

    return node_feats, batch_ei, batch_ew, B


# =====================================================================
# PyPower interface (unchanged)
# =====================================================================
def init_pypower_options():
    global PPOPT
    ppopt = ppoption()
    PPOPT = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)


def load_case_from_csv(case_name, constraints_path):
    """Load PyPower ppc dict from CSV files."""
    base_path = Path(constraints_path)

    base_mva_df = pd.read_csv(base_path / f"{case_name}_base_mva.csv")
    bus_df      = pd.read_csv(base_path / f"{case_name}_bus_data.csv")
    gen_df      = pd.read_csv(base_path / f"{case_name}_gen_data.csv")
    branch_df   = pd.read_csv(base_path / f"{case_name}_branch_data.csv")

    baseMVA = base_mva_df['value'].iloc[0]

    bus = np.zeros((len(bus_df), 13))
    bus[:, 0]  = bus_df['bus_id'].values
    bus[:, 1]  = bus_df['type'].values
    bus[:, 2]  = bus_df['pd_pu'].values
    bus[:, 3]  = bus_df['qd_pu'].values
    bus[:, 6]  = 1
    bus[:, 7]  = bus_df['vm_pu'].values
    bus[:, 8]  = bus_df['va_deg'].values
    bus[:, 9]  = bus_df['base_kv'].values
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
    branch[:, 0]  = branch_df['f_bus'].values
    branch[:, 1]  = branch_df['t_bus'].values
    branch[:, 2]  = branch_df['r_pu'].values
    branch[:, 3]  = branch_df['x_pu'].values
    branch[:, 4]  = branch_df['b_pu'].values
    branch[:, 5]  = branch_df['rate_a_pu'].values
    branch[:, 6]  = branch[:, 5]
    branch[:, 7]  = branch[:, 5]
    branch[:, 8]  = branch_df['tap_ratio'].values
    branch[:, 9]  = branch_df['shift_deg'].values
    branch[:, 10] = 1
    branch[:, 11] = -360
    branch[:, 12] = 360

    rate_a = branch_df['rate_a_pu'].values
    branch[:, 5:8][np.isnan(rate_a) | np.isinf(rate_a), :] = 9900.0

    gencost = np.zeros((len(gen_df), 7))
    gencost[:, 0] = 2
    gencost[:, 3] = 3
    gencost[:, 4] = gen_df['cost_c2'].values
    gencost[:, 5] = gen_df['cost_c1'].values
    gencost[:, 6] = gen_df['cost_c0'].values

    ppc = {
        'version': '2', 'baseMVA': baseMVA,
        'bus': bus, 'gen': gen, 'branch': branch, 'gencost': gencost
    }

    ppc['bus'][:, 2] *= baseMVA
    ppc['bus'][:, 3] *= baseMVA
    ppc['gen'][:, 3] *= baseMVA
    ppc['gen'][:, 4] *= baseMVA
    ppc['gen'][:, 8] *= baseMVA
    ppc['gen'][:, 9] *= baseMVA
    mask = (ppc['branch'][:, 5] != 0) & (ppc['branch'][:, 5] < 9000)
    ppc['branch'][mask, 5:8] *= baseMVA

    return ppc


def solve_pf_custom_optimized(pd_pu, qd_pu, pg_non_slack, vm_gen, params):
    """Run AC power flow for one sample."""
    global GLOBAL_CASE_DATA, PPOPT

    BASE_MVA          = params['general']['BASE_MVA']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    n_gen             = params['general']['n_gen']
    load_bus_ids      = params['general']['load_bus_ids']
    bus_id_to_idx     = params['general']['bus_id_to_idx']

    mpc_pf = {
        'version' : GLOBAL_CASE_DATA['version'],
        'baseMVA' : GLOBAL_CASE_DATA['baseMVA'],
        'bus'     : GLOBAL_CASE_DATA['bus'].copy(),
        'gen'     : GLOBAL_CASE_DATA['gen'].copy(),
        'branch'  : GLOBAL_CASE_DATA['branch'],
        'gencost' : GLOBAL_CASE_DATA['gencost'],
    }

    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc_pf['bus'][bus_idx, 2] = pd_pu[i] * BASE_MVA
            mpc_pf['bus'][bus_idx, 3] = qd_pu[i] * BASE_MVA

    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc_pf['gen'][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    for i in range(n_gen):
        mpc_pf['gen'][i, 5] = vm_gen[i]

    return runpf(mpc_pf, PPOPT)


# =====================================================================
# Evaluation (adapted for sub-optimal state input)
# =====================================================================
_DEFAULT_EVAL_CHUNK = 512


def evaluate_split(model, X_subopt_scaled, indices, raw_data, params, scalers,
                   edge_index, edge_weight, device, split_name,
                   subopt_x_raw=None,
                   verbose=True, eval_chunk_size=_DEFAULT_EVAL_CHUNK):
    """
    Evaluate SpectralGNN_ACOPF on a data split.

    X_subopt_scaled contains the SCALED sub-optimal state [vm|va|pinj|qinj].
    The original (pd, qd) for power flow is recovered from raw_data['x'].

    When predict_vm=False, generator Vm setpoints for power flow are recovered
    from the sub-optimal state (DCOPF+PF vm), NOT from the ground-truth
    ACOPF optimal Vm.  This avoids inflating MAE_Vm to a trivial 0.

    Args:
        model       : SpectralGNN_ACOPF instance
        X_subopt_scaled : FloatTensor [n_samples, 4*n_buses]
        indices     : array of int
        raw_data    : dict with keys 'pg', 'vm', 'qg', 'va', 'x'
        params      : params dict
        scalers     : dict of sklearn scalers (keys: 'subopt_x', 'pg', 'vm')
        edge_index  : LongTensor  [2, E]
        edge_weight : FloatTensor [E]
        device      : torch.device
        split_name  : str
        subopt_x_raw: np.ndarray [n_samples, 4*n_buses] RAW (unscaled)
                      sub-optimal state; needed when predict_vm=False to
                      extract vm setpoints for power flow.
                      If None and predict_vm=False, falls back to 1.0 p.u.
        verbose     : bool
        eval_chunk_size : int

    Returns:
        metrics : dict from evaluate_acopf_predictions()
    """
    if verbose:
        print(f"\n{split_name} Evaluation:")

    n_samples = len(X_subopt_scaled)
    n_buses = params['general']['n_buses']

    # ── Chunked batched-graph inference ───────────────────────────────
    model.eval()
    y_pred_parts = []

    with torch.no_grad():
        for start in range(0, n_samples, eval_chunk_size):
            end   = min(start + eval_chunk_size, n_samples)
            chunk = X_subopt_scaled[start:end].to(device)

            nf, bei, bew, B = collate_graph_batch(
                chunk, edge_index, edge_weight, n_buses, device)

            out = model(nf, bei, bew, batch_size=B, params=params)
            y_pred_parts.append(out.cpu())

    y_pred_scaled = torch.cat(y_pred_parts, dim=0)
    y_pred_np     = y_pred_scaled.numpy()

    # ── Denormalise ───────────────────────────────────────────────────
    n_gen           = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_loads         = params['general']['n_loads']
    gen_bus_ids     = params['general']['gen_bus_ids']
    bus_id_to_idx   = params['general']['bus_id_to_idx']

    predict_vm = hasattr(model, 'predict_vm') and model.predict_vm

    y_pred_pg_non_slack = scalers['pg'].inverse_transform(
        y_pred_np[:, :n_gen_non_slack])

    if predict_vm:
        y_pred_vm_gen = scalers['vm'].inverse_transform(
            y_pred_np[:, n_gen_non_slack:])
    else:
        # predict_vm=False: use sub-optimal state Vm (from DCOPF+PF) as
        # generator Vm setpoints.  This is the Vm the model "sees" as input,
        # so it is the most principled choice for power flow evaluation.
        gen_bus_indices_for_vm = np.array(
            [bus_id_to_idx[int(g)] for g in gen_bus_ids])

        if subopt_x_raw is not None:
            # subopt_x_raw layout: [vm_all | va_all | pinj_all | qinj_all]
            subopt_vm_all = subopt_x_raw[:, :n_buses]  # [n_samples, n_buses]
            y_pred_vm_gen = subopt_vm_all[:, gen_bus_indices_for_vm]
            if verbose:
                print(f"  predict_vm=False → using sub-optimal state Vm for PF"
                      f" (range: [{y_pred_vm_gen.min():.4f}, {y_pred_vm_gen.max():.4f}])")
        else:
            # Fallback: flat voltage profile
            y_pred_vm_gen = np.ones((n_samples, n_gen), dtype=np.float32)
            if verbose:
                print(f"  predict_vm=False → using 1.0 p.u. flat Vm (no subopt_x_raw)")

    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)

    gen_bus_indices = np.array([bus_id_to_idx[int(g)] for g in gen_bus_ids])
    y_pred_vm_all   = np.ones((n_samples, n_buses), dtype=np.float32)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen

    # ── Ground truth ──────────────────────────────────────────────────
    y_true_pg     = raw_data['pg'][indices]
    y_true_vm     = raw_data['vm'][indices]
    y_true_qg     = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

    # ── Recover raw p.u. loads for power flow ─────────────────────────
    x_raw = raw_data['x'][indices]   # [n_samples, 2*n_loads]
    pd_pu = x_raw[:, :n_loads]
    qd_pu = x_raw[:, n_loads:]

    # ── Power flow ────────────────────────────────────────────────────
    pf_results_list = []
    converge_flags  = []
    n_exceptions    = 0

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")
        print(f"  Predicted pg_non_slack range: "
              f"[{y_pred_pg_non_slack.min():.4f}, {y_pred_pg_non_slack.max():.4f}] p.u.")
        print(f"  True pg (all) range:          "
              f"[{y_true_pg.min():.4f}, {y_true_pg.max():.4f}] p.u.")
        print(f"  Vm_gen for PF range:          "
              f"[{y_pred_vm_gen.min():.4f}, {y_pred_vm_gen.max():.4f}] p.u.")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_custom_optimized(
                pd_pu[i], qd_pu[i],
                y_pred_pg_non_slack[i],
                y_pred_vm_gen[i],
                params)
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except Exception as e:
            if n_exceptions < 3 and verbose:
                import traceback
                print(f"  ⚠️ PF exception at sample {i}: {e}")
                traceback.print_exc()
            n_exceptions += 1
            n_gen_pf = params['general']['n_gen']
            pf_results_list.append((
                {'success': False,
                 'gen': np.zeros((n_gen_pf, 21)),
                 'bus': np.zeros((n_buses, 13)),
                 'branch': np.zeros((1, 17))},
            ))
            converge_flags.append(False)

    if verbose:
        n_conv = sum(converge_flags)
        print(f"    ✓ Converged: {n_conv}/{n_samples}"
              + (f" (exceptions: {n_exceptions})" if n_exceptions > 0 else ""))

    return evaluate_acopf_predictions(
        y_pred_pg_full,
        y_pred_vm_all,
        y_true_pg,
        y_true_vm,
        y_true_qg,
        y_true_va_rad,
        pf_results_list,
        converge_flags,
        params,
        verbose=verbose
    )