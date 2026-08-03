"""
DeepOPF-NGT — Semi-Supervised DCOPF  (Paper Algorithm 2, DCOPF adaptation)
===========================================================================
Source: "Unsupervised Learning for Solving AC Optimal Power Flows:
         Design, Analysis, and Experiment"
         Huang, Chen, Low — IEEE Trans. Power Syst., Vol. 39, No. 6, Nov. 2024

Version: v2.1 — Fixed virtual bus handling for case300

Fixes in v2.1:
- BUG FIX 1: _load_raw now uses bus_id_to_idx mapping
  instead of assuming bus_id == array_index (bus_id - 1)
- BUG FIX 2: pg columns now explicitly follow g_bus order from params,
  instead of sorted column scanning which may produce wrong column order

This script adapts Algorithm 2 (Extended DeepOPF-NGT, semi-supervised) from
the ACOPF setting (paper_semi_supervised_acopf.py) to the DCOPF setting,
using the same infrastructure as unsupervised_dcopf.py.

Algorithm 2 — Training of Extended DeepOPF-NGT (DCOPF version):
  Input : D̄ (all data, unlabeled), D (labeled subset with ground-truth Pg)
  For each epoch t = 1, 2, ..., T:
    ── Step 1 (supervised pre-train on labeled data D) ──────────────────────
      Sample mini-batch from D
      Compute: L_sup = k_v·L_v + k_g·L_g + k_slack·L_slack + k_line·L_line
        where L_v = Σ_{i ∈ non-slack} ‖P̂g_i - Pg_i‖²     (DCOPF Eq.14)
      Update: φ ← φ − η·∇_φ L_sup
    ── Step 2 (unsupervised train on all data D̄) ────────────────────────────
      Sample mini-batch from D̄
      Compute: L_uns = k_obj·L_obj + k_g·L_g + k_slack·L_slack + k_line·L_line
      Update: φ ← φ − η·∇_φ L_uns
      Update: k_i^t = min(k_obj·L_obj / L_i, k̄_i)         Eq.(12)

Key design decisions (faithful to paper, adapted for DCOPF):
  * Both steps happen WITHIN EVERY epoch (not two sequential phases).
  * k_v is FIXED throughout; constraint weights k_i updated only in Step 2.
  * L_v is over non-slack generators only (DCOPF has no V/θ, no ZIBs).
  * Weight update follows paper Eq.(12) exactly — no EMA, no decay, no boost.
  * All DCOPF infrastructure (imports, evaluate, data loading) is identical
    to unsupervised_dcopf.py.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dcopf_config import PathConfig
from dcopf_data_setup import (
    load_parameters_from_csv,
    load_and_prepare_data_generalization,
    DataSplitMode,
    split_data_by_mode,
)
from dcopf_slack_utils import (
    identify_slack_bus_and_gens,
    update_params_with_slack_info,
    reconstruct_full_pg,
    compute_detailed_mae,
    compute_detailed_pg_violations_pu,
)
from dcopf_violation_metrics import (
    feasibility as dc_feasibility,
    compute_cost,
    compute_cost_gap_percentage,
    compute_branch_violation_pu,
    compute_mae_percentage,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Neural Network  (identical to unsupervised_dcopf.py)
# ═══════════════════════════════════════════════════════════════════════════════

class PgPredictor(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, output_size), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Physical helper functions (PyTorch) — identical to unsupervised_dcopf.py
# ═══════════════════════════════════════════════════════════════════════════════

def denormalise_pg_non_slack(pg_norm, params, device):
    non_slack_idx = params['general']['non_slack_gen_indices']
    pg_min_all = params['constraints']['Pg_min'].ravel()
    pg_max_all = params['constraints']['Pg_max'].ravel()
    pg_min = torch.tensor(pg_min_all[non_slack_idx], dtype=torch.float32, device=device)
    pg_max = torch.tensor(pg_max_all[non_slack_idx], dtype=torch.float32, device=device)
    return pg_min + pg_norm * (pg_max - pg_min)

def reconstruct_slack_torch(pg_non_slack, Pd_batch, params, device):
    n_slack = params['general']['n_slack_gens']
    pg_slack_total = Pd_batch.sum(dim=1) - pg_non_slack.sum(dim=1)
    pg_slack_per = pg_slack_total / max(n_slack, 1)
    return pg_slack_per.unsqueeze(1).expand(-1, n_slack)

def build_full_pg_torch(pg_non_slack, pg_slack, params, device):
    n_g = params['general']['n_g']
    batch = pg_non_slack.shape[0]
    non_slack_idx = params['general']['non_slack_gen_indices']
    slack_idx = params['general']['slack_gen_indices']
    pg_full = torch.zeros(batch, n_g, dtype=torch.float32, device=device)
    pg_full[:, non_slack_idx] = pg_non_slack
    pg_full[:, slack_idx] = pg_slack
    return pg_full


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Unsupervised Loss  (identical to unsupervised_dcopf.py)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_unsupervised_loss(pg_non_slack, pg_slack, pg_full, Pd_batch, params, device):
    pg_min_all = torch.tensor(params['constraints']['Pg_min'].ravel(), dtype=torch.float32, device=device)
    pg_max_all = torch.tensor(params['constraints']['Pg_max'].ravel(), dtype=torch.float32, device=device)
    c2 = torch.tensor(params['constraints']['C_Pg_c2'], dtype=torch.float32, device=device)
    c1 = torch.tensor(params['constraints']['C_Pg'], dtype=torch.float32, device=device)
    c0 = torch.tensor(params['constraints']['C_Pg_c0'], dtype=torch.float32, device=device)
    Pl_max = torch.tensor(params['constraints']['Pl_max'], dtype=torch.float32, device=device)
    PTDF = torch.tensor(params['constraints']['PTDF'], dtype=torch.float32, device=device)
    Map_g = torch.tensor(params['constraints']['Map_g'], dtype=torch.float32, device=device)
    non_slack_idx = params['general']['non_slack_gen_indices']
    slack_idx = params['general']['slack_gen_indices']

    cost_per_gen = c2 * pg_full ** 2 + c1 * pg_full + c0
    L_obj = torch.mean(cost_per_gen.sum(dim=1))

    pg_min_ns = pg_min_all[non_slack_idx]; pg_max_ns = pg_max_all[non_slack_idx]
    ns_viol = torch.relu(pg_min_ns - pg_non_slack) + torch.relu(pg_non_slack - pg_max_ns)
    L_g = torch.mean((ns_viol ** 2).sum(dim=1))

    pg_min_sl = pg_min_all[slack_idx]; pg_max_sl = pg_max_all[slack_idx]
    sl_viol = torch.relu(pg_min_sl - pg_slack) + torch.relu(pg_slack - pg_max_sl)
    L_slack = torch.mean((sl_viol ** 2).sum(dim=1))

    Pg_bus = torch.matmul(pg_full, Map_g)
    P_inj = Pg_bus - Pd_batch
    line_flows = torch.matmul(P_inj, PTDF)
    valid_mask = (Pl_max > 1e-5) & (Pl_max < 1e10)
    if valid_mask.any():
        lf_v = line_flows[:, valid_mask]; pl_v = Pl_max[valid_mask]
        l_viol = torch.relu(torch.abs(lf_v) - pl_v)
        L_line = torch.mean((l_viol ** 2).sum(dim=1))
    else:
        L_line = torch.tensor(0.0, device=device)

    return {'L_obj': L_obj, 'L_g': L_g, 'L_slack': L_slack, 'L_line': L_line}


def compute_total_loss_unsup(loss_norm, coeffs):
    return (coeffs['k_obj'] * loss_norm['L_obj'] + coeffs['k_g'] * loss_norm['L_g'] +
            coeffs['k_slack'] * loss_norm['L_slack'] + coeffs['k_line'] * loss_norm['L_line'])


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Supervised Loss  (DCOPF version of Eq.13–14)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_supervised_loss(pg_non_slack_pred, pg_non_slack_gt, loss_norm, coeffs, k_v):
    L_v = torch.mean(((pg_non_slack_pred - pg_non_slack_gt) ** 2).sum(dim=1))
    L_cons = (coeffs['k_g'] * loss_norm['L_g'] + coeffs['k_slack'] * loss_norm['L_slack'] +
              coeffs['k_line'] * loss_norm['L_line'])
    return k_v * L_v + L_cons, L_v


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Adaptive weight update — 3-regime
# ═══════════════════════════════════════════════════════════════════════════════

def update_coefficients_3regime(coeffs, loss_ema):
    SATISFIED_THRESH = 0.02; BOOST_THRESH = 0.30; DECAY = 0.95; BOOST_BASE = 50.0
    K_INIT = {'k_g': 100.0, 'k_slack': 100.0, 'k_line': 10.0}
    ema_obj = loss_ema['L_obj']
    for wk, lk in [('k_g', 'L_g'), ('k_slack', 'L_slack'), ('k_line', 'L_line')]:
        ema_i = loss_ema[lk]; k_min = coeffs[f'{wk}_min']; k_max = coeffs[f'{wk}_max']
        k_init = K_INIT[wk]
        if ema_i < SATISFIED_THRESH:
            coeffs[wk] = float(max(coeffs[wk] * DECAY, k_init))
        elif ema_i >= BOOST_THRESH:
            coeffs[wk] = float(np.clip(BOOST_BASE * ema_i, k_min, k_max))
        else:
            if ema_i > 1e-8:
                coeffs[wk] = float(np.clip(coeffs['k_obj'] * ema_obj / ema_i, k_min, k_max))


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Evaluation  (identical to unsupervised_dcopf.py)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, x_raw_eval, y_pg_all_eval, x_scaler, params, device):
    model.eval()
    x_scaled = x_scaler.transform(x_raw_eval)
    X_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        pg_norm = model(X_t)
        pg_ns = denormalise_pg_non_slack(pg_norm, params, device)
    pg_ns_np = pg_ns.cpu().numpy()
    pd_total = x_raw_eval.sum(axis=1)
    pg_all_pred = reconstruct_full_pg(pg_ns_np, pd_total, params)
    mae_dict = compute_detailed_mae(y_pg_all_eval, pg_ns_np, pg_all_pred, params)
    gen_up_viol, gen_lo_viol, line_viol, _ = dc_feasibility(pg_all_pred, x_raw_eval, params)
    viol_dict = compute_detailed_pg_violations_pu(gen_up_viol, gen_lo_viol, params)
    branch_viol = compute_branch_violation_pu(line_viol, params['constraints']['Pl_max'])
    cost_coeffs = {'C2': params['constraints']['C_Pg_c2'], 'C1': params['constraints']['C_Pg'],
                   'C0': params['constraints']['C_Pg_c0']}
    cost_true = compute_cost(y_pg_all_eval, cost_coeffs)
    cost_pred = compute_cost(pg_all_pred, cost_coeffs)
    cost_gap = compute_cost_gap_percentage(cost_true, cost_pred)
    return {'mae_pg_non_slack': mae_dict['mae_non_slack'], 'mae_pg_slack': mae_dict['mae_slack'],
            'viol_pg_non_slack': viol_dict['viol_non_slack'], 'viol_pg_slack': viol_dict['viol_slack'],
            'viol_branch': branch_viol, 'cost_gap': cost_gap}


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Data loading helpers  (v2.1 FIXED — sparse-bus-safe)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_raw(file_path: str, params: dict, column_names: dict) -> tuple:
    """
    Load raw data from CSV.

    v2.1 FIXES:
    -----------
    1. Load demand mapping: uses bus_id_to_idx dict instead of bus_id-1 indexing.
    2. Generator columns: explicitly built from g_bus order in params.
    """
    import pandas as pd
    df      = pd.read_csv(file_path)
    n       = len(df)
    n_buses = params['general']['n_buses']
    n_g     = params['general']['n_g']
    lp      = column_names['load_prefix']
    gp      = column_names['gen_prefix']

    # FIX 1: use bus_id_to_idx mapping
    pd_cols = sorted([c for c in df.columns if c.startswith(lp)],
                     key=lambda c: int(c[len(lp):]))
    bus_id_to_idx = params['general']['bus_id_to_idx']
    x_raw = np.zeros((n, n_buses), dtype='float32')
    for col in pd_cols:
        bus_id = int(col[len(lp):])
        if bus_id in bus_id_to_idx:
            x_raw[:, bus_id_to_idx[bus_id]] = df[col].values.astype('float32')
        else:
            print(f"[WARNING] load bus {bus_id} not in bus_id_to_idx, skipping")

    # FIX 2: pg columns from g_bus order
    g_bus = params['general']['g_bus']
    pg_cols = [f"{gp}{int(gen_id)}" for gen_id in g_bus]
    missing = [c for c in pg_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[_load_raw] Missing pg columns: {missing}")
    y_pg = df[pg_cols].values.astype('float32')
    if y_pg.shape[1] != n_g:
        print(f"  [WARNING] CSV has {y_pg.shape[1]} pg columns, params expect {n_g}.")
    return x_raw, y_pg


def load_train_data(dataset_path, params, column_names):
    x_raw, y_pg = _load_raw(dataset_path, params, column_names)
    print(f"  Loaded {len(x_raw)} samples  x: {x_raw.shape}  y_pg: {y_pg.shape}")
    return x_raw, y_pg

def fit_input_scaler(x_raw, train_idx):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(); scaler.fit(x_raw[train_idx]); return scaler

def load_external_test_data(test_data_path, params, column_names, n_test_samples, seed):
    x_full, y_full = _load_raw(test_data_path, params, column_names)
    n_available = len(x_full); n_actual = min(n_test_samples, n_available)
    if n_actual < n_available:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_available, n_actual, replace=False)
        return x_full[idx], y_full[idx]
    return x_full, y_full


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Main training function  (Algorithm 2 — Semi-Supervised DCOPF)
# ═══════════════════════════════════════════════════════════════════════════════

def train_semi_supervised_dcopf(
        case_name, params_path, dataset_path, column_names,
        split_mode=DataSplitMode.RANDOM_SPLIT, test_data_path=None,
        test_params_path=None, n_test_samples=1000, n_train_use=10000,
        n_labeled=300, seed=42, hidden_sizes=None, n_epochs=100,
        learning_rate=1e-3, batch_size=256, device='cuda',
        k_v=100.0, k_obj=0.1, k_g_0=100.0, k_slack_0=100.0, k_line_0=10.0,
        k_g_max=10000.0, k_slack_max=10000.0, k_line_max=200.0):
    if hidden_sizes is None: hidden_sizes = [256, 256]

    print("\n" + "=" * 70)
    print("Semi-Supervised DCOPF  –  Paper Algorithm 2 (DCOPF adaptation)")
    print("=" * 70)
    print(f"  Case: {case_name}  Mode: {split_mode.value}  Hidden: {hidden_sizes}")
    print(f"  Epochs: {n_epochs}  LR: {learning_rate}  Batch: {batch_size}")
    print(f"  Train N: {n_train_use}  Labeled: {n_labeled}  Seed: {seed}  k_v: {k_v}")
    print("=" * 70)

    torch.manual_seed(seed); np.random.seed(seed)
    _device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {_device}\n")

    # [1] Load params
    train_params = load_parameters_from_csv(case_name, params_path, is_api=False)
    train_params = update_params_with_slack_info(train_params, identify_slack_bus_and_gens(train_params))
    n_g = train_params['general']['n_g']; n_buses = train_params['general']['n_buses']
    n_non_slack = train_params['general']['n_g_non_slack']; n_slack = train_params['general']['n_slack_gens']
    print(f"  Buses: {n_buses}  Gen: {n_g}  Non-slack: {n_non_slack}  Slack: {n_slack}")

    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None: raise ValueError("API_TEST requires test_params_path")
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_params = update_params_with_slack_info(test_params, identify_slack_bus_and_gens(test_params))
    else:
        test_params = train_params

    # [3] Load data
    x_data_raw, y_pg_raw = load_train_data(dataset_path, train_params, column_names)

    # [4] Split
    if split_mode in (DataSplitMode.RANDOM_SPLIT, DataSplitMode.VALID_FIXED):
        train_idx, val_idx, test_idx, _, _ = split_data_by_mode(
            x_data_raw=x_data_raw, y_pg_raw=y_pg_raw, mode=split_mode, n_train_use=n_train_use, seed=seed)
        x_test_raw = x_data_raw[test_idx]; y_pg_test_all = y_pg_raw[test_idx]
    elif split_mode == DataSplitMode.GENERALIZATION:
        train_idx, val_idx, _, _, _ = split_data_by_mode(
            x_data_raw=x_data_raw, y_pg_raw=y_pg_raw, mode=split_mode, n_train_use=n_train_use, seed=seed,
            test_data_path=test_data_path, params=train_params, column_names=column_names, n_test_samples=n_test_samples)
        x_test_raw, y_pg_test_all = load_external_test_data(test_data_path, train_params, column_names, n_test_samples, seed)
    elif split_mode == DataSplitMode.API_TEST:
        train_idx, val_idx, _, _, _ = split_data_by_mode(
            x_data_raw=x_data_raw, y_pg_raw=y_pg_raw, mode=split_mode, n_train_use=n_train_use, seed=seed,
            test_data_path=test_data_path, params=train_params, column_names=column_names, n_test_samples=n_test_samples)
        x_test_raw, y_pg_test_all = load_external_test_data(test_data_path, test_params, column_names, n_test_samples, seed)
    else:
        raise ValueError(f"Unknown: {split_mode}")

    x_train_raw = x_data_raw[train_idx]
    x_scaler = fit_input_scaler(x_data_raw, train_idx)
    x_train_scaled = x_scaler.transform(x_train_raw)
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(x_test_raw)}")

    # [7] Labeled subset
    rng = np.random.default_rng(seed); n_train = len(train_idx)
    lbl_local = rng.choice(n_train, size=min(n_labeled, n_train), replace=False)
    non_slack_idx = train_params['general']['non_slack_gen_indices']
    X_labeled = torch.tensor(x_train_scaled[lbl_local], dtype=torch.float32, device=_device)
    Pg_ns_gt_lbl = torch.tensor(y_pg_raw[train_idx[lbl_local]][:, non_slack_idx], dtype=torch.float32, device=_device)
    Pd_lbl = torch.tensor(x_train_raw[lbl_local], dtype=torch.float32, device=_device)
    n_lbl = len(X_labeled)
    X_train_all = torch.tensor(x_train_scaled, dtype=torch.float32, device=_device)
    Pd_train_all = torch.tensor(x_train_raw, dtype=torch.float32, device=_device)
    n_train_all = len(X_train_all)
    print(f"  Labeled: {n_lbl}  All training: {n_train_all}")

    # [9] Model
    model = PgPredictor(n_buses, n_non_slack, hidden_sizes).to(_device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    coeffs = {'k_obj': k_obj, 'k_g': k_g_0, 'k_slack': k_slack_0, 'k_line': k_line_0,
              'k_g_min': 1.0, 'k_g_max': k_g_max, 'k_slack_min': 1.0, 'k_slack_max': k_slack_max,
              'k_line_min': 1.0, 'k_line_max': k_line_max}
    loss_refs = {'L_obj': 1.0, 'L_g': 1.0, 'L_slack': 1.0, 'L_line': 1.0}
    EMA_ALPHA = 0.3
    loss_ema = {'L_obj': None, 'L_g': None, 'L_slack': None, 'L_line': None}

    # [11] Training
    t_train_start = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_sup = 0.0; ep_Lv = 0.0
        ep_sums = {k: 0.0 for k in ['L_obj', 'L_g', 'L_slack', 'L_line']}

        # Step 1: supervised
        lbl_perm = torch.randperm(n_lbl, device=_device)
        for b_start in range(0, n_lbl, batch_size):
            idx = lbl_perm[b_start: min(b_start + batch_size, n_lbl)]
            X_b = X_labeled[idx]; Pd_b = Pd_lbl[idx]; Pg_gt_b = Pg_ns_gt_lbl[idx]
            opt.zero_grad()
            pg_norm = model(X_b); pg_ns = denormalise_pg_non_slack(pg_norm, train_params, _device)
            pg_sl = reconstruct_slack_torch(pg_ns, Pd_b, train_params, _device)
            pg_full = build_full_pg_torch(pg_ns, pg_sl, train_params, _device)
            loss_dict = compute_unsupervised_loss(pg_ns, pg_sl, pg_full, Pd_b, train_params, _device)
            loss_norm = {k: loss_dict[k] / loss_refs[k] for k in loss_dict}
            L_sup, L_v = compute_supervised_loss(pg_ns, Pg_gt_b, loss_norm, coeffs, k_v)
            L_sup.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); opt.step()
            ep_sup += L_sup.item() * len(idx); ep_Lv += L_v.item() * len(idx)

        # Step 2: unsupervised
        all_perm = torch.randperm(n_train_all, device=_device)
        for b_start in range(0, n_train_all, batch_size):
            idx = all_perm[b_start: min(b_start + batch_size, n_train_all)]
            X_b = X_train_all[idx]; Pd_b = Pd_train_all[idx]
            opt.zero_grad()
            pg_norm = model(X_b); pg_ns = denormalise_pg_non_slack(pg_norm, train_params, _device)
            pg_sl = reconstruct_slack_torch(pg_ns, Pd_b, train_params, _device)
            pg_full = build_full_pg_torch(pg_ns, pg_sl, train_params, _device)
            loss_dict = compute_unsupervised_loss(pg_ns, pg_sl, pg_full, Pd_b, train_params, _device)
            loss_norm = {k: loss_dict[k] / loss_refs[k] for k in loss_dict}
            L_uns = compute_total_loss_unsup(loss_norm, coeffs)
            L_uns.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); opt.step()
            bs = len(idx)
            for k, v in loss_dict.items(): ep_sums[k] += v.item() * bs

        epoch_means_raw = {k: ep_sums[k] / n_train_all for k in ep_sums}
        epoch_means_norm = {k: epoch_means_raw[k] / loss_refs[k] for k in epoch_means_raw}

        if epoch == 1:
            for k in loss_refs:
                raw = epoch_means_raw[k]; loss_refs[k] = float(raw) if raw > 1e-8 else 1.0
            for k in loss_ema: loss_ema[k] = 1.0

        if epoch > 1:
            for k in loss_ema:
                loss_ema[k] = EMA_ALPHA * epoch_means_norm[k] + (1.0 - EMA_ALPHA) * loss_ema[k]
            update_coefficients_3regime(coeffs, loss_ema)

        if epoch % 10 == 0 or epoch == 1:
            r = epoch_means_raw
            print(f"Epoch {epoch:4d}/{n_epochs}  | L_obj: {r['L_obj']:12.2f}  | L_line: {r['L_line']:10.4f}"
                  f"  | L_v:{ep_Lv/n_lbl:.6f}  | k_g:{coeffs['k_g']:.1f}  k_line:{coeffs['k_line']:.1f}")

    train_time = time.perf_counter() - t_train_start
    print(f"\n✅ Training complete in {train_time:.2f} s")

    # Latency
    model.eval()
    single_x = torch.tensor(x_scaler.transform(x_test_raw[:1]), dtype=torch.float32, device=_device)
    with torch.no_grad():
        for _ in range(20): _ = model(single_x)
    if _device.type == 'cuda': torch.cuda.synchronize()
    n_rep, times = 200, []
    with torch.no_grad():
        for _ in range(n_rep):
            t0 = time.perf_counter(); _ = model(single_x)
            if _device.type == 'cuda': torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    latency_ms = float(np.mean(times)) * 1000.0

    # Evaluate
    test_metrics = evaluate(model, x_test_raw, y_pg_test_all, x_scaler, test_params, _device)

    print("\n" + "=" * 70)
    print(f"Final Test Results  [{split_mode.value}]  |  Labeled={n_labeled}")
    print("=" * 70)
    print(f"  MAE Pg Non-Slack: {test_metrics['mae_pg_non_slack']:.4f} %")
    print(f"  MAE Pg Slack:     {test_metrics['mae_pg_slack']:.4f} %")
    print(f"  Pg Non-Slack viol: {test_metrics['viol_pg_non_slack']:.6f} p.u.")
    print(f"  Pg Slack viol:     {test_metrics['viol_pg_slack']:.6f} p.u.")
    print(f"  Branch viol:       {test_metrics['viol_branch']:.6f} p.u.")
    print(f"  Cost Gap:          {test_metrics['cost_gap']:.4f} %")
    print(f"  Training time:     {train_time:.2f} s")
    print(f"  Inference latency: {latency_ms:.4f} ms")
    print("=" * 70)

    return model, train_params, test_params, x_scaler, coeffs, test_metrics


if __name__ == "__main__":
    CASE_NAME = 'pglib_opf_case118_ieee'; CASE_SHORT_NAME = 'case118'
    TRAIN_VARIANCE = 'v=0.12'; TEST_VARIANCE = 'v=0.25'
    SPLIT_MODE = DataSplitMode.API_TEST # Change to VALID_FIXED, GENERALIZATION, or API_TEST as needed
    PARAMS_PATH = PathConfig.get_constraints_path(CASE_SHORT_NAME, is_api=False)
    DATASET_PATH = PathConfig.get_dataset_path(CASE_NAME, CASE_SHORT_NAME, variance=TRAIN_VARIANCE, is_api=False)
    if SPLIT_MODE == DataSplitMode.GENERALIZATION:
        TEST_DATA_PATH = PathConfig.get_dataset_path(CASE_NAME, CASE_SHORT_NAME, variance=TEST_VARIANCE); TEST_PARAMS_PATH = None
    elif SPLIT_MODE == DataSplitMode.API_TEST:
        TEST_DATA_PATH = PathConfig.get_dataset_path(CASE_NAME, CASE_SHORT_NAME, is_api=True)
        TEST_PARAMS_PATH = PathConfig.get_constraints_path(CASE_SHORT_NAME, is_api=True)
    else:
        TEST_DATA_PATH = None; TEST_PARAMS_PATH = None

    COLUMN_NAMES = {'load_prefix': 'pd', 'gen_prefix': 'pg', 'lambda': 'lambda',
                    'mu_g_min_prefix': 'mu_g_min_', 'mu_g_max_prefix': 'mu_g_max_',
                    'mu_line_pos_prefix': 'mu_line_max_', 'mu_line_neg_prefix': 'mu_line_min_'}

    model, train_params, test_params, x_scaler, final_coeffs, test_metrics = \
        train_semi_supervised_dcopf(
            case_name=CASE_NAME, params_path=PARAMS_PATH, dataset_path=DATASET_PATH,
            column_names=COLUMN_NAMES, split_mode=SPLIT_MODE, test_data_path=TEST_DATA_PATH,
            test_params_path=TEST_PARAMS_PATH, n_test_samples=1000, n_train_use=12000,
            n_labeled=5000, n_epochs=3000, learning_rate=1e-3, batch_size=64,
            hidden_sizes=[128, 64], seed=42, device='cuda', k_v=100.0)