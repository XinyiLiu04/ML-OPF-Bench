# -*- coding: utf-8 -*-
"""Graph construction, sub-optimal state loading and evaluation for the spectral GNN."""

import numpy as np
import pandas as pd
import torch
import os
import sys

# ac_configuration/ sits in ac_methods/. Appending the parent of this script's own
# directory makes it importable whether this file is directly in ac_methods/ or one
# level down in a grouped method folder, and from any working directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ac_configuration.acopf_data_setup import reconstruct_full_pg
from ac_configuration.acopf_evaluation_metrics import evaluate_acopf_predictions
from ac_configuration.acopf_pypower import solve_pf_setpoints
from gnn_model import build_node_features_subopt

_DEFAULT_EVAL_CHUNK = 512


def load_subopt_state(data_dir, case_name, params):
    """Load the sub-optimal state CSVs written by generate_subopt_state.py.

    Returns (subopt_x_raw [n_samples, 4*n_buses] laid out as
    [vm_all | va_all | pinj_all | qinj_all], converged [n_samples] bool).
    """
    bus_ids = params['general']['bus_ids']

    vm_df = pd.read_csv(f"{data_dir}/{case_name}_subopt_vm.csv")
    va_df = pd.read_csv(f"{data_dir}/{case_name}_subopt_va.csv")
    pinj_df = pd.read_csv(f"{data_dir}/{case_name}_subopt_pinj.csv")
    qinj_df = pd.read_csv(f"{data_dir}/{case_name}_subopt_qinj.csv")
    conv_df = pd.read_csv(f"{data_dir}/{case_name}_subopt_converged.csv")

    # Columns are selected by bus id so the ordering matches bus_id_to_idx
    subopt_x_raw = np.hstack([
        vm_df[[f"vm_{int(bid)}" for bid in bus_ids]].values.astype('float32'),
        va_df[[f"va_{int(bid)}" for bid in bus_ids]].values.astype('float32'),
        pinj_df[[f"pinj_{int(bid)}" for bid in bus_ids]].values.astype('float32'),
        qinj_df[[f"qinj_{int(bid)}" for bid in bus_ids]].values.astype('float32'),
    ])
    converged = conv_df['converged'].values.astype(bool)

    return subopt_x_raw, converged


def build_adjacency_edge_weight(params, kernel='gaussian', scale_k=1.0, threshold=0.0):
    """Build a bidirectional COO edge list with scalar weights for ChebConv.

    The Gaussian kernel weights each branch by exp(-k * |z|^2) with z = r + jx, so
    electrically close buses are coupled more strongly.
    """
    bus_id_to_idx = params['general']['bus_id_to_idx']
    f_bus = params['branch']['f_bus']
    t_bus = params['branch']['t_bus']
    r_pu = params['branch']['r_pu']
    x_pu = params['branch']['x_pu']

    src_list, dst_list, w_list = [], [], []

    for i in range(len(f_bus)):
        fi = bus_id_to_idx[int(f_bus[i])]
        ti = bus_id_to_idx[int(t_bus[i])]

        if kernel == 'gaussian':
            w = float(np.exp(-scale_k * (float(r_pu[i]) ** 2 + float(x_pu[i]) ** 2)))
        else:
            w = 1.0

        if w <= threshold:
            continue

        src_list += [fi, ti]
        dst_list += [ti, fi]
        w_list += [w, w]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(w_list, dtype=torch.float32)

    print(f"\nGraph structure constructed:")
    print(f"  Nodes: {params['general']['n_buses']}")
    print(f"  Edges: {edge_index.shape[1]} (bidirectional)")
    print(f"  Edge kernel: {kernel} (k={scale_k})")
    if w_list:
        print(f"  Weight range: [{min(w_list):.4f}, {max(w_list):.4f}]")

    return edge_index, edge_weight


def collate_graph_batch(x_scaled_batch, edge_index, edge_weight, n_buses, device):
    """Assemble a batch of identical graphs into one disjoint graph.

    Returns (node_feats [B*N, 4], edge_index [2, B*E], edge_weight [B*E], B).
    """
    B = x_scaled_batch.shape[0]
    N = n_buses
    E = edge_index.shape[1]

    node_feats = build_node_features_subopt(x_scaled_batch, N, device)

    # Graph b occupies rows [b*N, (b+1)*N), so its edges are shifted by b*N
    offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
    batch_ei = (edge_index.to(device).unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, B * E)
    batch_ew = edge_weight.to(device).repeat(B)

    return node_feats, batch_ei, batch_ew, B


def evaluate_split(model, X_subopt_scaled, indices, raw_data, params, scalers,
                   edge_index, edge_weight, device, case_data, split_name,
                   subopt_x_raw=None, verbose=True,
                   eval_chunk_size=_DEFAULT_EVAL_CHUNK):
    """Run the GNN over a split, solve the power flow per sample and return the metrics.

    X_subopt_scaled holds the scaled sub-optimal state; the loads driving the power
    flow come from raw_data['x'], and subopt_x_raw supplies the unscaled Vm used when
    the model does not predict Vm.
    """
    if verbose:
        print(f"\n{split_name} Evaluation:")

    n_samples = len(X_subopt_scaled)
    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_loads = params['general']['n_loads']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_bus_indices = np.array(
        [bus_id_to_idx[int(g)] for g in params['general']['gen_bus_ids']])

    model.eval()
    y_pred_parts = []
    with torch.no_grad():
        for start in range(0, n_samples, eval_chunk_size):
            chunk = X_subopt_scaled[start:min(start + eval_chunk_size, n_samples)].to(device)
            nf, bei, bew, B = collate_graph_batch(
                chunk, edge_index, edge_weight, n_buses, device)
            y_pred_parts.append(model(nf, bei, bew, batch_size=B).cpu())

    y_pred_np = torch.cat(y_pred_parts, dim=0).numpy()

    y_pred_pg_non_slack = scalers['pg'].inverse_transform(
        y_pred_np[:, :n_gen_non_slack])

    if model.predict_vm:
        y_pred_vm_gen = scalers['vm'].inverse_transform(y_pred_np[:, n_gen_non_slack:])
    elif subopt_x_raw is not None:
        # Without a Vm head, the power flow uses the Vm the model was given as input
        # (from DCOPF plus power flow). Taking the ground-truth ACOPF Vm instead would
        # drive MAE_Vm to zero and report an accuracy the model never produced.
        y_pred_vm_gen = subopt_x_raw[:, :n_buses][:, gen_bus_indices]
        if verbose:
            print(f"  predict_vm=False, using sub-optimal state Vm for the power flow "
                  f"(range: [{y_pred_vm_gen.min():.4f}, {y_pred_vm_gen.max():.4f}])")
    else:
        y_pred_vm_gen = np.ones((n_samples, n_gen), dtype=np.float32)
        if verbose:
            print(f"  predict_vm=False and no sub-optimal state given, "
                  f"using a flat 1.0 p.u. profile")

    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)
    y_pred_vm_all = np.full((n_samples, n_buses), 1.0, dtype=np.float32)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen

    y_true_pg = raw_data['pg'][indices]
    y_true_vm = raw_data['vm'][indices]
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

    x_raw = raw_data['x'][indices]
    pd_pu = x_raw[:, :n_loads]
    qd_pu = x_raw[:, n_loads:]

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")
        print(f"  Predicted pg_non_slack range: "
              f"[{y_pred_pg_non_slack.min():.4f}, {y_pred_pg_non_slack.max():.4f}] p.u.")
        print(f"  True pg range: [{y_true_pg.min():.4f}, {y_true_pg.max():.4f}] p.u.")

    pf_results_list = []
    converge_flags = []
    n_exceptions = 0

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_setpoints(
                pd_pu[i], qd_pu[i], y_pred_pg_non_slack[i], y_pred_vm_gen[i],
                params, case_data)
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except Exception:
            n_exceptions += 1
            pf_results_list.append((
                {'success': False,
                 'gen': np.zeros((n_gen, 21)),
                 'bus': np.zeros((n_buses, 13)),
                 'branch': np.zeros((1, 17))},
            ))
            converge_flags.append(False)

    if verbose:
        print(f"  Converged: {sum(converge_flags)}/{n_samples}"
              + (f" (exceptions: {n_exceptions})" if n_exceptions else ""))

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