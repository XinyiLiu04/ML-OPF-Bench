"""
DeepOPF-NGT — Unsupervised DCOPF  (Paper Algorithm 1, faithful adaptation)
=========================================================================
Source: "Unsupervised Learning for Solving AC Optimal Power Flows:
         Design, Analysis, and Experiment"
         Huang, Chen, Low — IEEE Trans. Power Syst., Vol. 39, No. 6, Nov. 2024

Version: v2.1 — Fixed virtual bus handling for case300

Fixes in v2.1:
- BUG FIX 1: _load_raw now uses bus_id_to_idx mapping
  instead of assuming bus_id == array_index (bus_id - 1)
- BUG FIX 2: pg columns now explicitly follow g_bus order from params,
  instead of sorted column scanning which may produce wrong column order

This script is a direct replication of the reference version
(the version you handed in), differing from project/unsupervised_dcopf_main.py
in ONE place only — the weight update strategy:

  Reference version (this file):
    k_g, k_slack, k_line all use the SAME unified 3-regime loop:
      (A) ema_i < 0.02  → decay toward K_INIT  (×0.95 per epoch)
      (B) 0.02 ≤ ema_i < 0.30  → standard formula  k_i = k_obj·ema_obj/ema_i
      (C) ema_i ≥ 0.30  → proportional boost  k_i = 50·ema_i

  Project version (unsupervised_dcopf_main.py):
    k_g, k_slack use standard formula only (no A/C regimes)
    k_line uses a separate LINE_VIOLATION_THRESHOLD=0.05 boost rule

Everything else is byte-for-byte identical to the project version:
  loss terms, data loading, evaluate(), param keys, architecture,
  auto-norm, EMA(α=0.3), grad clip(1.0), latency measurement, __main__.
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
# 1.  Neural Network
# ═══════════════════════════════════════════════════════════════════════════════

class PgPredictor(nn.Module):
    """
    Fully-connected network with Sigmoid output.
    Output ∈ (0,1) is later mapped to physical [Pg_min, Pg_max].
    """
    def __init__(self, input_size: int, output_size: int,
                 hidden_sizes: list = None):
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
# 2.  Physical helper functions (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

def denormalise_pg_non_slack(pg_norm: torch.Tensor,
                              params: dict,
                              device: torch.device) -> torch.Tensor:
    """Map DNN output (0,1) → Pg_non_slack in [Pg_min, Pg_max] (p.u.)."""
    non_slack_idx = params['general']['non_slack_gen_indices']
    pg_min_all = params['constraints']['Pg_min'].ravel()
    pg_max_all = params['constraints']['Pg_max'].ravel()
    pg_min = torch.tensor(pg_min_all[non_slack_idx], dtype=torch.float32, device=device)
    pg_max = torch.tensor(pg_max_all[non_slack_idx], dtype=torch.float32, device=device)
    return pg_min + pg_norm * (pg_max - pg_min)


def reconstruct_slack_torch(pg_non_slack: torch.Tensor,
                             Pd_batch: torch.Tensor,
                             params: dict,
                             device: torch.device) -> torch.Tensor:
    """
    Reconstruct slack Pg via power balance (PyTorch).
    Pg_slack_total = ΣPd − ΣPg_non_slack  (divided equally among slack gens).
    """
    n_slack = params['general']['n_slack_gens']
    pg_slack_total = Pd_batch.sum(dim=1) - pg_non_slack.sum(dim=1)
    pg_slack_per   = pg_slack_total / max(n_slack, 1)
    return pg_slack_per.unsqueeze(1).expand(-1, n_slack)


def build_full_pg_torch(pg_non_slack: torch.Tensor,
                         pg_slack: torch.Tensor,
                         params: dict,
                         device: torch.device) -> torch.Tensor:
    """Assemble full Pg tensor (batch, n_g)."""
    n_g   = params['general']['n_g']
    batch = pg_non_slack.shape[0]
    non_slack_idx = params['general']['non_slack_gen_indices']
    slack_idx     = params['general']['slack_gen_indices']
    pg_full = torch.zeros(batch, n_g, dtype=torch.float32, device=device)
    pg_full[:, non_slack_idx] = pg_non_slack
    pg_full[:, slack_idx]     = pg_slack
    return pg_full


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Unsupervised Loss
# ═══════════════════════════════════════════════════════════════════════════════

def compute_unsupervised_loss(pg_non_slack: torch.Tensor,
                               pg_slack: torch.Tensor,
                               pg_full: torch.Tensor,
                               Pd_batch: torch.Tensor,
                               params: dict,
                               device: torch.device) -> dict:
    """
    Four physics-based loss terms (all in p.u.).

    L_obj   : mean generation cost          (minimise objective)
    L_g     : squared non-slack Pg bound violations
    L_slack : squared slack Pg bound violations  (replaces L_d)
    L_line  : squared branch flow violations
    """
    pg_min_all = torch.tensor(params['constraints']['Pg_min'].ravel(),
                               dtype=torch.float32, device=device)
    pg_max_all = torch.tensor(params['constraints']['Pg_max'].ravel(),
                               dtype=torch.float32, device=device)
    c2     = torch.tensor(params['constraints']['C_Pg_c2'],
                          dtype=torch.float32, device=device)
    c1     = torch.tensor(params['constraints']['C_Pg'],
                          dtype=torch.float32, device=device)
    c0     = torch.tensor(params['constraints']['C_Pg_c0'],
                          dtype=torch.float32, device=device)
    Pl_max = torch.tensor(params['constraints']['Pl_max'],
                          dtype=torch.float32, device=device)
    PTDF   = torch.tensor(params['constraints']['PTDF'],
                          dtype=torch.float32, device=device)   # (n_buses, n_branches)
    Map_g  = torch.tensor(params['constraints']['Map_g'],
                          dtype=torch.float32, device=device)   # (n_g, n_buses)

    non_slack_idx = params['general']['non_slack_gen_indices']
    slack_idx     = params['general']['slack_gen_indices']

    # ── L_obj ────────────────────────────────────────────────────────────────
    cost_per_gen = c2 * pg_full ** 2 + c1 * pg_full + c0
    L_obj = torch.mean(cost_per_gen.sum(dim=1))

    # ── L_g ──────────────────────────────────────────────────────────────────
    pg_min_ns = pg_min_all[non_slack_idx]
    pg_max_ns = pg_max_all[non_slack_idx]
    ns_viol   = torch.relu(pg_min_ns - pg_non_slack) + torch.relu(pg_non_slack - pg_max_ns)
    L_g = torch.mean((ns_viol ** 2).sum(dim=1))

    # ── L_slack ───────────────────────────────────────────────────────────────
    pg_min_sl = pg_min_all[slack_idx]
    pg_max_sl = pg_max_all[slack_idx]
    sl_viol   = torch.relu(pg_min_sl - pg_slack) + torch.relu(pg_slack - pg_max_sl)
    L_slack = torch.mean((sl_viol ** 2).sum(dim=1))

    # ── L_line ────────────────────────────────────────────────────────────────
    Pg_bus     = torch.matmul(pg_full, Map_g)
    P_inj      = Pg_bus - Pd_batch
    line_flows = torch.matmul(P_inj, PTDF)
    valid_mask = (Pl_max > 1e-5) & (Pl_max < 1e10)
    if valid_mask.any():
        lf_v   = line_flows[:, valid_mask]
        pl_v   = Pl_max[valid_mask]
        l_viol = torch.relu(torch.abs(lf_v) - pl_v)
        L_line = torch.mean((l_viol ** 2).sum(dim=1))
    else:
        L_line = torch.tensor(0.0, device=device)

    return {'L_obj': L_obj, 'L_g': L_g, 'L_slack': L_slack, 'L_line': L_line}


def compute_total_loss(loss_dict: dict, coeffs: dict) -> torch.Tensor:
    return (coeffs['k_obj']   * loss_dict['L_obj']  +
            coeffs['k_g']     * loss_dict['L_g']    +
            coeffs['k_slack'] * loss_dict['L_slack'] +
            coeffs['k_line']  * loss_dict['L_line'])


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Evaluation (NumPy, no PyPower)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model: nn.Module,
             x_raw_eval: np.ndarray,
             y_pg_all_eval: np.ndarray,
             x_scaler,
             params: dict,
             device: torch.device) -> dict:
    model.eval()
    x_scaled = x_scaler.transform(x_raw_eval)
    X_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        pg_norm = model(X_t)
        pg_ns   = denormalise_pg_non_slack(pg_norm, params, device)

    pg_ns_np    = pg_ns.cpu().numpy()
    pd_total    = x_raw_eval.sum(axis=1)
    pg_all_pred = reconstruct_full_pg(pg_ns_np, pd_total, params)

    mae_dict    = compute_detailed_mae(y_pg_all_eval, pg_ns_np, pg_all_pred, params)
    gen_up_viol, gen_lo_viol, line_viol, _ = dc_feasibility(
        pg_all_pred, x_raw_eval, params)
    viol_dict   = compute_detailed_pg_violations_pu(gen_up_viol, gen_lo_viol, params)
    branch_viol = compute_branch_violation_pu(line_viol, params['constraints']['Pl_max'])

    cost_coeffs = {
        'C2': params['constraints']['C_Pg_c2'],
        'C1': params['constraints']['C_Pg'],
        'C0': params['constraints']['C_Pg_c0'],
    }
    cost_true = compute_cost(y_pg_all_eval, cost_coeffs)
    cost_pred = compute_cost(pg_all_pred,   cost_coeffs)
    cost_gap  = compute_cost_gap_percentage(cost_true, cost_pred)

    return {
        'mae_pg_non_slack':  mae_dict['mae_non_slack'],
        'mae_pg_slack':      mae_dict['mae_slack'],
        'viol_pg_non_slack': viol_dict['viol_non_slack'],
        'viol_pg_slack':     viol_dict['viol_slack'],
        'viol_branch':       branch_viol,
        'cost_gap':          cost_gap,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Data loading helpers  (v2.1 FIXED — sparse-bus-safe)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_raw(file_path: str, params: dict, column_names: dict) -> tuple:
    """
    Load raw data from CSV.

    v2.1 FIXES:
    -----------
    1. Load demand mapping: uses bus_id_to_idx dict instead of bus_id-1 indexing.
       This is critical for case300 where bus IDs like 7001, 9533 exist.
    2. Generator columns: explicitly built from g_bus order in params,
       instead of sorted column scanning which may produce wrong column order.
    """
    import pandas as pd
    df      = pd.read_csv(file_path)
    n       = len(df)
    n_buses = params['general']['n_buses']
    n_g     = params['general']['n_g']
    lp      = column_names['load_prefix']
    gp      = column_names['gen_prefix']

    # ------------------------------------------------------------------ #
    # FIX 1: Load demand data — use bus_id_to_idx mapping                #
    #                                                                     #
    # OLD (BROKEN for case300):                                           #
    #   if bus_id <= n_buses:                                             #
    #       x_raw[:, bus_id - 1] = ...                                   #
    #                                                                     #
    # NEW (CORRECT):                                                      #
    #   Uses bus_id_to_idx dict from params (loaded from bus_ids.csv)    #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # FIX 2: Generator columns — MUST match g_bus order from params      #
    #                                                                     #
    # OLD (BROKEN):                                                       #
    #   pg_cols = sorted([c for c in df.columns if c.startswith(gp)],    #
    #                    key=lambda c: int(c[len(gp):]))                  #
    #                                                                     #
    # NEW (CORRECT):                                                      #
    #   Build pg_cols explicitly from g_bus (gen_id order)               #
    # ------------------------------------------------------------------ #
    g_bus = params['general']['g_bus']
    pg_cols = [f"{gp}{int(gen_id)}" for gen_id in g_bus]

    missing = [c for c in pg_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[_load_raw] Missing pg columns in CSV: {missing}\n"
            f"Available pg columns (first 10): "
            f"{sorted([c for c in df.columns if c.startswith(gp)])[:10]}"
        )

    y_pg = df[pg_cols].values.astype('float32')
    if y_pg.shape[1] != n_g:
        print(f"  [WARNING] CSV has {y_pg.shape[1]} pg columns, "
              f"params expect {n_g}.")
    return x_raw, y_pg


def load_train_data(dataset_path: str, params: dict, column_names: dict) -> tuple:
    x_raw, y_pg = _load_raw(dataset_path, params, column_names)
    print(f"  Loaded {len(x_raw)} samples  x: {x_raw.shape}  y_pg: {y_pg.shape}")
    return x_raw, y_pg


def fit_input_scaler(x_raw: np.ndarray, train_idx: np.ndarray):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(x_raw[train_idx])
    return scaler


def load_external_test_data(test_data_path: str,
                             params: dict,
                             column_names: dict,
                             n_test_samples: int,
                             seed: int) -> tuple:
    x_full, y_full = _load_raw(test_data_path, params, column_names)
    n_available = len(x_full)
    n_actual    = min(n_test_samples, n_available)
    if n_actual < n_available:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_available, n_actual, replace=False)
        return x_full[idx], y_full[idx]
    return x_full, y_full


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Main training function
# ═══════════════════════════════════════════════════════════════════════════════

def train_unsupervised_dcopf(
        case_name:        str,
        params_path:      str,
        dataset_path:     str,
        column_names:     dict,
        split_mode:       DataSplitMode = DataSplitMode.RANDOM_SPLIT,
        test_data_path:   str   = None,
        test_params_path: str   = None,
        n_test_samples:   int   = 1000,
        n_train_use:      int   = 10000,
        seed:             int   = 42,
        hidden_sizes:     list  = None,
        n_epochs:         int   = 100,
        learning_rate:    float = 1e-3,
        batch_size:       int   = 256,
        device:           str   = 'cuda',
):
    if hidden_sizes is None:
        hidden_sizes = [256, 256]

    print("\n" + "=" * 70)
    print("Unsupervised DCOPF  –  Dynamic Loss Weight Version")
    print("=" * 70)
    print(f"  Case       : {case_name}")
    print(f"  Mode       : {split_mode.value}")
    print(f"  Hidden     : {hidden_sizes}")
    print(f"  Epochs     : {n_epochs}  |  LR: {learning_rate}  |  Batch: {batch_size}")
    print(f"  Train N    : {n_train_use}  |  Seed: {seed}")
    print("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)
    _device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"  Device     : {_device}\n")

    # ── [1] Load training system parameters ──────────────────────────────────
    print("[1] Loading training system parameters...")
    train_params = load_parameters_from_csv(case_name, params_path, is_api=False)
    slack_info   = identify_slack_bus_and_gens(train_params)
    train_params = update_params_with_slack_info(train_params, slack_info)

    n_g         = train_params['general']['n_g']
    n_buses     = train_params['general']['n_buses']
    n_non_slack = train_params['general']['n_g_non_slack']
    n_slack     = train_params['general']['n_slack_gens']
    print(f"  Buses: {n_buses}  Generators: {n_g}"
          f"  Non-slack: {n_non_slack}  Slack: {n_slack}")

    # ── [2] Load test parameters for API_TEST ────────────────────────────────
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError("API_TEST mode requires test_params_path")
        print("\n[2] Loading API test system parameters...")
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_slack  = identify_slack_bus_and_gens(test_params)
        test_params = update_params_with_slack_info(test_params, test_slack)
        print(f"  API buses: {test_params['general']['n_buses']}"
              f"  API gens: {test_params['general']['n_g']}"
              f"  API non-slack: {test_params['general']['n_g_non_slack']}")
    else:
        test_params = train_params

    # ── [3] Load training data ────────────────────────────────────────────────
    print("\n[3] Loading training dataset...")
    x_data_raw, y_pg_raw = load_train_data(dataset_path, train_params, column_names)

    # ── [4] Data splitting ────────────────────────────────────────────────────
    print(f"\n[4] Splitting data (mode={split_mode.value})...")

    if split_mode in (DataSplitMode.RANDOM_SPLIT, DataSplitMode.VALID_FIXED):
        train_idx, val_idx, test_idx, _, _ = split_data_by_mode(
            x_data_raw=x_data_raw, y_pg_raw=y_pg_raw,
            mode=split_mode, n_train_use=n_train_use, seed=seed)
        x_test_raw    = x_data_raw[test_idx]
        y_pg_test_all = y_pg_raw[test_idx]

    elif split_mode == DataSplitMode.GENERALIZATION:
        if test_data_path is None:
            raise ValueError("GENERALIZATION mode requires test_data_path")
        train_idx, val_idx, _, _, _ = split_data_by_mode(
            x_data_raw=x_data_raw, y_pg_raw=y_pg_raw,
            mode=split_mode, n_train_use=n_train_use, seed=seed,
            test_data_path=test_data_path, params=train_params,
            column_names=column_names, n_test_samples=n_test_samples)
        x_test_raw, y_pg_test_all = load_external_test_data(
            test_data_path, train_params, column_names, n_test_samples, seed)
        test_idx = None
        print(f"  External test samples: {len(x_test_raw)}")

    elif split_mode == DataSplitMode.API_TEST:
        if test_data_path is None:
            raise ValueError("API_TEST mode requires test_data_path")
        train_idx, val_idx, _, _, _ = split_data_by_mode(
            x_data_raw=x_data_raw, y_pg_raw=y_pg_raw,
            mode=split_mode, n_train_use=n_train_use, seed=seed,
            test_data_path=test_data_path, params=train_params,
            column_names=column_names, n_test_samples=n_test_samples)
        x_test_raw, y_pg_test_all = load_external_test_data(
            test_data_path, test_params, column_names, n_test_samples, seed)
        test_idx = None
        print(f"  API test samples: {len(x_test_raw)}")

    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    x_scaler    = fit_input_scaler(x_data_raw, train_idx)
    x_train_raw = x_data_raw[train_idx]
    X_train_t   = torch.tensor(
        x_scaler.transform(x_train_raw), dtype=torch.float32, device=_device)
    Pd_train_t  = torch.tensor(x_train_raw, dtype=torch.float32, device=_device)

    print(f"\n  Train: {len(train_idx)}"
          f"  Val: {len(val_idx)}"
          f"  Test: {len(x_test_raw)}")

    # ── [5] Build model ───────────────────────────────────────────────────────
    print(f"\n[5] Building model  {n_buses} → {hidden_sizes} → {n_non_slack} ...")
    model = PgPredictor(n_buses, n_non_slack, hidden_sizes).to(_device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {n_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ── [6] Loss weight initialisation ────────────────────────────────────────
    coeffs = {
        'k_obj':       0.1,
        'k_g':         100.0,
        'k_slack':     100.0,
        'k_line':      10.0,
        'k_g_min':     1.0,      'k_g_max':     10000.0,
        'k_slack_min': 1.0,      'k_slack_max': 10000.0,
        'k_line_min':  1.0,      'k_line_max':  200.0,
    }
    loss_refs = {'L_obj': 1.0, 'L_g': 1.0, 'L_slack': 1.0, 'L_line': 1.0}
    EMA_ALPHA = 0.3
    loss_ema  = {'L_obj': None, 'L_g': None, 'L_slack': None, 'L_line': None}

    # ── [7] Training loop ─────────────────────────────────────────────────────
    print(f"\n[6] Training ({n_epochs} epochs)...")
    print("-" * 70)

    n_train   = len(X_train_t)
    n_batches = (n_train + batch_size - 1) // batch_size
    t_train_start = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_sums = {k: 0.0 for k in ['L_obj', 'L_g', 'L_slack', 'L_line']}
        perm = torch.randperm(n_train, device=_device)

        for b in range(n_batches):
            s   = b * batch_size
            e   = min(s + batch_size, n_train)
            idx = perm[s:e]

            X_b  = X_train_t[idx]
            Pd_b = Pd_train_t[idx]

            optimizer.zero_grad()
            pg_norm = model(X_b)
            pg_ns   = denormalise_pg_non_slack(pg_norm, train_params, _device)
            pg_sl   = reconstruct_slack_torch(pg_ns, Pd_b, train_params, _device)
            pg_full = build_full_pg_torch(pg_ns, pg_sl, train_params, _device)

            loss_dict = compute_unsupervised_loss(
                pg_ns, pg_sl, pg_full, Pd_b, train_params, _device)

            loss_norm = {k: loss_dict[k] / loss_refs[k] for k in loss_dict}
            loss = compute_total_loss(loss_norm, coeffs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = e - s
            for k in epoch_sums:
                epoch_sums[k] += loss_dict[k].item() * bs

        epoch_means_raw  = {k: epoch_sums[k] / n_train for k in epoch_sums}
        epoch_means_norm = {k: epoch_means_raw[k] / loss_refs[k]
                            for k in epoch_means_raw}

        if epoch == 1:
            for k in loss_refs:
                raw = epoch_means_raw[k]
                loss_refs[k] = float(raw) if raw > 1e-8 else 1.0
            for k in loss_ema:
                loss_ema[k] = 1.0
            print(f"  [Auto-norm] References after epoch 1:")
            print(f"    L_obj_ref  = {loss_refs['L_obj']:.4f}")
            print(f"    L_g_ref    = {loss_refs['L_g']:.6f}")
            print(f"    L_slack_ref= {loss_refs['L_slack']:.6f}")
            print(f"    L_line_ref = {loss_refs['L_line']:.6f}")

        if epoch > 1:
            for k in loss_ema:
                loss_ema[k] = (EMA_ALPHA * epoch_means_norm[k]
                               + (1.0 - EMA_ALPHA) * loss_ema[k])

            ema_obj = loss_ema['L_obj']

            SATISFIED_THRESH = 0.02
            BOOST_THRESH     = 0.30
            DECAY            = 0.95
            BOOST_BASE       = 50.0

            K_INIT = {'k_g': 100.0, 'k_slack': 100.0, 'k_line': 10.0}

            for wk, lk in {'k_g': 'L_g',
                            'k_slack': 'L_slack',
                            'k_line': 'L_line'}.items():
                ema_i  = loss_ema[lk]
                k_min  = coeffs[f'{wk}_min']
                k_max  = coeffs[f'{wk}_max']
                k_init = K_INIT[wk]

                if ema_i < SATISFIED_THRESH:
                    decayed = coeffs[wk] * DECAY
                    coeffs[wk] = float(max(decayed, k_init))

                elif ema_i >= BOOST_THRESH:
                    boosted = BOOST_BASE * ema_i
                    coeffs[wk] = float(np.clip(boosted, k_min, k_max))

                else:
                    if ema_i > 1e-8:
                        new_w = coeffs['k_obj'] * ema_obj / ema_i
                        coeffs[wk] = float(np.clip(new_w, k_min, k_max))

        if epoch % 10 == 0 or epoch == 1:
            r = epoch_means_raw
            n_obj   = r['L_obj']   / loss_refs['L_obj']
            n_line  = r['L_line']  / loss_refs['L_line']
            n_slack = r['L_slack'] / loss_refs['L_slack']
            print(
                f"Epoch {epoch:4d}/{n_epochs}"
                f"  | L_obj: {r['L_obj']:12.2f} (norm {n_obj:.4f})"
                f"  | L_line: {r['L_line']:10.4f} (norm {n_line:.4f})"
                f"  | L_slack: {r['L_slack']:10.6f} (norm {n_slack:.4f})"
            )
            ema_str = (f"  EMA → obj:{loss_ema['L_obj']:.4f}"
                       f"  line:{loss_ema['L_line']:.4f}"
                       f"  slack:{loss_ema['L_slack']:.4f}"
                       if epoch > 1 else "")
            print(
                f"  Weights → k_g:{coeffs['k_g']:.1f}"
                f"  k_slack:{coeffs['k_slack']:.1f}"
                f"  k_line:{coeffs['k_line']:.1f}"
                + ema_str
            )

    train_time = time.perf_counter() - t_train_start
    print(f"\n✅ Training complete in {train_time:.2f} s")

    # ── [8] Inference latency ─────────────────────────────────────────────────
    print("\n[7] Measuring inference latency...")
    model.eval()
    single_x = torch.tensor(
        x_scaler.transform(x_test_raw[:1]),
        dtype=torch.float32, device=_device)
    with torch.no_grad():
        for _ in range(20): _ = model(single_x)
    if _device.type == 'cuda':
        torch.cuda.synchronize()

    n_rep, times = 200, []
    with torch.no_grad():
        for _ in range(n_rep):
            t0 = time.perf_counter()
            _ = model(single_x)
            if _device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    latency_ms = float(np.mean(times)) * 1000.0

    # ── [9] Test set evaluation ───────────────────────────────────────────────
    print("\n[8] Evaluating on test set...")
    eval_params = test_params
    test_metrics = evaluate(
        model, x_test_raw, y_pg_test_all,
        x_scaler, eval_params, _device)

    # ── [10] Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Final Test Results  [{split_mode.value}]")
    print("=" * 70)
    print("\n--- Accuracy (MAE) ---")
    print(f"  MAE Pg  Non-Slack  : {test_metrics['mae_pg_non_slack']:>10.4f} %")
    print("\n--- Constraint Violations (p.u., Mean of Max) ---")
    print(f"  Pg  Non-Slack viol : {test_metrics['viol_pg_non_slack']:>10.6f} p.u.")
    print(f"  Pg  Slack viol     : {test_metrics['viol_pg_slack']:>10.6f} p.u.")
    print(f"  Branch viol        : {test_metrics['viol_branch']:>10.6f} p.u.")
    print("\n--- Cost ---")
    print(f"  Cost Gap           : {test_metrics['cost_gap']:>10.4f} %")
    print("\n--- Speed ---")
    print(f"  Training time      : {train_time:>10.2f} s")
    print(f"  Inference latency  : {latency_ms:>10.4f} ms"
          f"  (single sample, avg {n_rep} runs)")
    print("=" * 70)

    return model, train_params, test_params, x_scaler, coeffs, test_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    CASE_NAME       = 'pglib_opf_case118_ieee'
    CASE_SHORT_NAME = 'case118'
    TRAIN_VARIANCE  = 'v=0.12'
    TEST_VARIANCE   = 'v=0.25'

    SPLIT_MODE = DataSplitMode.API_TEST
    PARAMS_PATH  = PathConfig.get_constraints_path(CASE_SHORT_NAME, is_api=False)
    DATASET_PATH = PathConfig.get_dataset_path(
        CASE_NAME, CASE_SHORT_NAME, variance=TRAIN_VARIANCE, is_api=False)

    if SPLIT_MODE == DataSplitMode.GENERALIZATION:
        TEST_DATA_PATH   = PathConfig.get_dataset_path(
            CASE_NAME, CASE_SHORT_NAME, variance=TEST_VARIANCE, is_api=False)
        TEST_PARAMS_PATH = None
    elif SPLIT_MODE == DataSplitMode.API_TEST:
        TEST_DATA_PATH   = PathConfig.get_dataset_path(
            CASE_NAME, CASE_SHORT_NAME, is_api=True)
        TEST_PARAMS_PATH = PathConfig.get_constraints_path(CASE_SHORT_NAME, is_api=True)
    else:
        TEST_DATA_PATH   = None
        TEST_PARAMS_PATH = None

    COLUMN_NAMES = {
        'load_prefix':        'pd',
        'gen_prefix':         'pg',
        'lambda':             'lambda',
        'mu_g_min_prefix':    'mu_g_min_',
        'mu_g_max_prefix':    'mu_g_max_',
        'mu_line_pos_prefix': 'mu_line_max_',
        'mu_line_neg_prefix': 'mu_line_min_',
    }

    N_TRAIN_USE    = 12000
    N_TEST_SAMPLES = 1000
    N_EPOCHS       = 3000
    LEARNING_RATE  = 1e-3
    BATCH_SIZE     = 64
    HIDDEN_SIZES   = [128, 64]
    SEED           = 42
    DEVICE         = 'cuda'

    model, train_params, test_params, x_scaler, final_coeffs, test_metrics = \
        train_unsupervised_dcopf(
            case_name        = CASE_NAME,
            params_path      = PARAMS_PATH,
            dataset_path     = DATASET_PATH,
            column_names     = COLUMN_NAMES,
            split_mode       = SPLIT_MODE,
            test_data_path   = TEST_DATA_PATH,
            test_params_path = TEST_PARAMS_PATH,
            n_test_samples   = N_TEST_SAMPLES,
            n_train_use      = N_TRAIN_USE,
            n_epochs         = N_EPOCHS,
            learning_rate    = LEARNING_RATE,
            batch_size       = BATCH_SIZE,
            hidden_sizes     = HIDDEN_SIZES,
            seed             = SEED,
            device           = DEVICE,
        )

    save_path = f'model_unsupervised_dcopf_{SPLIT_MODE.value}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_params':     train_params,
        'test_params':      test_params,
        'x_scaler':         x_scaler,
        'final_coeffs':     final_coeffs,
        'test_metrics':     test_metrics,
        'split_mode':       SPLIT_MODE.value,
        'version':          'unsupervised_dcopf_dynamic_loss_v2.1',
    }, save_path)
    print(f"\n✨ Model saved → {save_path}")