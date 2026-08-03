# -*- coding: utf-8 -*-
"""
GNN Utility Functions for DCOPF  (Enhanced: susceptance kernel + 6-dim features)

Version: v2.1 — Fixed virtual bus handling for case300

Fixes in v2.1:
- BUG FIX: load_and_prepare_dc now uses bus_id_to_idx mapping
  instead of assuming bus_id == array_index (bus_id - 1).
  The pg column ordering (using g_bus) was already correct.

Changes from v1:
  1. Default graph kernel changed from 'gaussian' to 'susceptance'
  2. collate_graph_batch_dc() updated for 6-dim node features
  3. build_graph_from_branch_info() adds edge weight normalization
  4. evaluate_split_dc() updated for new model signature

Provides:
  - load_branch_info()             : read branch topology
  - build_bus_lookup()             : bus_id → sequential index mapping
  - build_graph_from_branch_info() : COO edge_index + scalar edge_weight
  - load_and_prepare_dc()          : load CSV dataset
  - evaluate_split_dc()            : full DCOPF metric computation
  - collate_graph_batch_dc()       : batched disjoint graph construction
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dcopf_violation_metrics import (
    feasibility as dc_feasibility,
    compute_cost,
    compute_cost_gap_percentage,
    compute_branch_violation_pu,
)
from dcopf_slack_utils import (
    reconstruct_full_pg,
    compute_detailed_mae,
    compute_detailed_pg_violations_pu,
)
from gnn_dcopf_model import build_node_features_dc


# =====================================================================
# Batched-graph construction  (unchanged)
# =====================================================================
def collate_graph_batch_dc(x_pd_scaled_batch, edge_index, edge_weight,
                           static_node_feats, single_edge_index,
                           n_buses, device):
    B = x_pd_scaled_batch.shape[0]
    N = n_buses
    E = edge_index.shape[1]

    node_feats = build_node_features_dc(
        x_pd_scaled_batch, static_node_feats, single_edge_index, device
    )

    offsets  = (torch.arange(B, device=device) * N).view(B, 1, 1)
    batch_ei = (edge_index.to(device).unsqueeze(0) + offsets).reshape(2, B * E)
    batch_ew = edge_weight.to(device).repeat(B)

    return node_feats, batch_ei, batch_ew, B


# =====================================================================
# Branch info loader  (unchanged)
# =====================================================================
def load_branch_info(params_path, case_name, is_api=False):
    suffix   = "__api" if is_api else ""
    csv_path = os.path.join(params_path, f"{case_name}{suffix}_branch_info.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[load_branch_info] not found: {csv_path}")
    df = pd.read_csv(csv_path).sort_values('branch_id').reset_index(drop=True)
    return {
        'branch_id': df['branch_id'].values,
        'f_bus'    : df['f_bus'].values.astype(int),
        't_bus'    : df['t_bus'].values.astype(int),
        'r_pu'     : df['r_pu'].values.astype('float32'),
        'x_pu'     : df['x_pu'].values.astype('float32'),
    }


# =====================================================================
# Bus lookup  (unchanged — already correct)
# =====================================================================
def build_bus_lookup(branch_info, params):
    n_buses = params['general']['n_buses']
    all_bus_ids = set()
    all_bus_ids.update(int(x) for x in branch_info['f_bus'])
    all_bus_ids.update(int(x) for x in branch_info['t_bus'])
    sorted_bus_ids = sorted(all_bus_ids)
    if len(sorted_bus_ids) > n_buses:
        print(f"  [WARNING] Found {len(sorted_bus_ids)} unique bus IDs but n_buses={n_buses}")
    bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(sorted_bus_ids)}
    max_id = sorted_bus_ids[-1] if sorted_bus_ids else 0
    n_unique = len(sorted_bus_ids)
    if max_id > n_unique:
        n_virtual = sum(1 for bid in sorted_bus_ids if bid > n_unique)
        print(f"  [Bus Lookup] Non-contiguous bus IDs detected: {n_unique} buses, max={max_id}")
    else:
        print(f"  [Bus Lookup] Contiguous bus IDs: 1..{n_unique}")
    return bus_id_to_idx


# =====================================================================
# Graph construction  (unchanged — already uses bus_id_to_idx correctly)
# =====================================================================
def build_graph_from_branch_info(branch_info, params, kernel='susceptance',
                                  scale_k=1.0, threshold=0.0):
    n_buses = params['general']['n_buses']
    bus_id_to_idx = build_bus_lookup(branch_info, params)

    f_bus = branch_info['f_bus']
    t_bus = branch_info['t_bus']
    r_pu  = branch_info.get('r_pu', np.zeros_like(branch_info['x_pu']))
    x_pu  = branch_info['x_pu']

    src_list, dst_list, w_list = [], [], []
    n_skipped = 0

    for i in range(len(f_bus)):
        fi = bus_id_to_idx.get(int(f_bus[i]))
        ti = bus_id_to_idx.get(int(t_bus[i]))
        if fi is None or ti is None:
            n_skipped += 1; continue
        if fi >= n_buses or ti >= n_buses:
            n_skipped += 1; continue

        ri = float(r_pu[i]); xi = float(x_pu[i])

        if kernel == 'susceptance':
            if abs(xi) < 1e-12: xi = 1e-12
            w = float(1.0 / abs(xi))
        elif kernel == 'gaussian':
            z_mag_sq = ri ** 2 + xi ** 2
            w = float(np.exp(-scale_k * z_mag_sq))
        else:
            w = 1.0

        if w <= threshold: continue
        src_list += [fi, ti]; dst_list += [ti, fi]; w_list += [w, w]

    if len(w_list) > 0 and kernel != 'uniform':
        w_max = max(w_list)
        if w_max > 0:
            w_list = [w / w_max for w in w_list]

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(w_list, dtype=torch.float32)

    n_edges = edge_index.shape[1]
    print(f"\n✓ Graph structure constructed (DCOPF):")
    print(f"  Nodes: {n_buses}  Edges: {n_edges} (bidirectional)  Kernel: {kernel}")
    if n_skipped > 0:
        print(f"  Skipped: {n_skipped} branches")

    return edge_index, edge_weight


# =====================================================================
# Data loader  (v2.1 FIXED)
# =====================================================================
def load_and_prepare_dc(full_dataset_path, params):
    """
    Load DCOPF dataset CSV and return (x_raw, y_pg_non_slack_raw, y_pg_all_raw).

    v2.1 FIX:
    ---------
    Load demand mapping now uses bus_id_to_idx dict instead of bus_id-1.
    This is critical for case300 where bus IDs like 7001, 9533 exist.
    The pg column ordering (using g_bus) was already correct.
    """
    full_df   = pd.read_csv(full_dataset_path)
    n_samples = len(full_df)
    n_buses   = params['general']['n_buses']

    # ------------------------------------------------------------------ #
    # FIX: Load demand — use bus_id_to_idx mapping                       #
    #                                                                     #
    # OLD (BROKEN for case300):                                           #
    #   if bus_id <= n_buses:                                             #
    #       x_data_raw[:, bus_id - 1] = ...                              #
    # ------------------------------------------------------------------ #
    pd_cols = sorted(
        [c for c in full_df.columns if c.startswith('pd')],
        key=lambda c: int(c[2:])
    )
    bus_id_to_idx = params['general']['bus_id_to_idx']
    x_data_raw = np.zeros((n_samples, n_buses), dtype='float32')
    for col in pd_cols:
        bus_id = int(col[2:])
        if bus_id in bus_id_to_idx:
            x_data_raw[:, bus_id_to_idx[bus_id]] = full_df[col].values.astype('float32')
        else:
            print(f"[WARNING] load bus {bus_id} not in bus_id_to_idx, skipping")

    # Generator columns — already correct (uses g_bus order)
    g_bus   = params['general']['g_bus']
    pg_cols = [f'pg{int(gid)}' for gid in g_bus]
    missing = [c for c in pg_cols if c not in full_df.columns]
    if missing:
        raise ValueError(f"[load_and_prepare_dc] Missing pg columns: {missing}")
    y_pg_raw_all = full_df[pg_cols].values.astype('float32')

    non_slack_idx      = params['general']['non_slack_gen_indices']
    y_pg_raw_non_slack = y_pg_raw_all[:, non_slack_idx]

    print(f"  [load_and_prepare_dc] {n_samples} samples loaded")
    print(f"    x  (pd / bus)   : {x_data_raw.shape}")
    print(f"    y  (all gens)   : {y_pg_raw_all.shape}")
    print(f"    y  (non-slack)  : {y_pg_raw_non_slack.shape}")

    return x_data_raw, y_pg_raw_non_slack, y_pg_raw_all


# =====================================================================
# Evaluation  (unchanged)
# =====================================================================
def evaluate_split_dc(model, edge_index, edge_weight, scalers, params,
                      device, x_raw_eval, y_true_pg_all, batch_size=512):
    model.eval()
    n_samples = len(x_raw_eval)
    n_buses   = params['general']['n_buses']
    x_scaled  = scalers['x'].transform(x_raw_eval)

    all_pred_scaled = []
    ei_dev = edge_index.to(device); ew_dev = edge_weight.to(device)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = torch.tensor(x_scaled[start:end], dtype=torch.float32, device=device)
            nf, bei, bew, B = collate_graph_batch_dc(
                x_batch, ei_dev, ew_dev,
                model.static_node_feats, model.single_edge_index, n_buses, device)
            pred = model(nf, bei, bew, batch_size=B, params=params)
            all_pred_scaled.append(pred.cpu().numpy())

    y_pred_ns_scaled = np.vstack(all_pred_scaled)
    y_pred_non_slack = scalers['y_pg_non_slack'].inverse_transform(y_pred_ns_scaled)
    pd_total      = x_raw_eval.sum(axis=1)
    y_pred_pg_all = reconstruct_full_pg(y_pred_non_slack, pd_total, params)

    mae_dict = compute_detailed_mae(y_true_pg_all, y_pred_non_slack, y_pred_pg_all, params)
    gen_up_viol, gen_lo_viol, line_viol, _ = dc_feasibility(y_pred_pg_all, x_raw_eval, params)
    viol_dict = compute_detailed_pg_violations_pu(gen_up_viol, gen_lo_viol, params)
    branch_viol = compute_branch_violation_pu(line_viol, params['constraints']['Pl_max'])

    cost_coeffs = {
        'C2': params['constraints'].get('C_Pg_c2', np.zeros(y_true_pg_all.shape[1])),
        'C1': params['constraints']['C_Pg'],
        'C0': params['constraints'].get('C_Pg_c0', np.zeros(y_true_pg_all.shape[1])),
    }
    cost_true = compute_cost(y_true_pg_all, cost_coeffs)
    cost_pred = compute_cost(y_pred_pg_all, cost_coeffs)
    cost_gap_pct = compute_cost_gap_percentage(cost_true, cost_pred)

    return {
        'mae_pg_non_slack': float(mae_dict['mae_non_slack']),
        'viol_pg_non_slack': float(viol_dict['viol_non_slack']),
        'viol_pg_slack': float(viol_dict['viol_slack']),
        'viol_branch': float(branch_viol),
        'cost_gap_percent': float(cost_gap_pct),
        'n_samples': n_samples,
    }