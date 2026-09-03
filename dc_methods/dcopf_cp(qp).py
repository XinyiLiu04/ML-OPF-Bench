# -*- coding: utf-8 -*-
"""
DeepOPF (PINN) + Post-processing for DCOPF - Main Experiment Script
Physics-Informed Neural Network with Strict DeepOPF Post-processing

Version: v6.1 - Fixed virtual bus handling for case300

Fixes in v6.1:
- BUG FIX 1: load_and_prepare_deepopf_data now uses bus_id_to_idx mapping
  instead of assuming bus_id == array_index (bus_id - 1)
- BUG FIX 2: pg columns now explicitly follow g_bus order from params,
  instead of unsorted column scanning

Modifications vs v5.0:
- post_process_solution() now strictly follows the original DeepOPF paper (Section 3.5):
    Solve: min ||Pg_all_pred - u||^2
    s.t.   Pg_min_i <= u_i <= Pg_max_i            (generator limits, all gens)
           sum(Map_g @ u) = sum(Pd)                (power balance via bus injection)
           -Pl_max <= PTDF @ (Map_g @ u - Pd) <= Pl_max  (branch flow limits via PTDF)
  Using scipy.optimize.minimize (SLSQP) as the QP solver.
- build_branch_flow_matrix() replaced by direct PTDF usage from params.
- All other logic unchanged.

Post-processing reference:
  Pan et al., "DeepOPF: Deep Neural Network for DC Optimal Power Flow", 2020
  Section 3.5: Project predicted solution into feasible region via convex QP.

Key params fields used (from dcopf_data_setup.py):
  params['constraints']['PTDF']    : shape (n_buses, n_lines)  [stored as ptdf_matrix.T]
                                     -> transposed to (n_lines, n_buses) inside post_process
  params['constraints']['Map_g']   : shape (n_buses, n_gen)   [stored as bus_gen_map.T]
  params['constraints']['Pg_min']  : shape (1, n_gen)
  params['constraints']['Pg_max']  : shape (1, n_gen)
  params['constraints']['Pl_max']  : shape (n_lines,)
"""

import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

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

# Global parameters
GLOBAL_PARAMS = {}
GLOBAL_SCALERS = {}


# ============================================================================
# Post-processing: Strict DeepOPF Convex QP Projection (Section 3.5)
# ============================================================================

def post_process_solution_qp(Pg_pred_all, x_pd, params):
    """
    Strict DeepOPF post-processing (Section 3.5 of Pan et al. 2020).

    For each sample, solves:
        min  0.5 * ||u - Pg_pred_all||^2
        s.t.
            Pg_min_i <= u_i <= Pg_max_i,  for all i       [generator limits]
            sum(Map_g @ u) = sum(Pd)                       [power balance]
            -Pl_max_k <= (PTDF @ (Map_g @ u - Pd))_k
                      <= Pl_max_k, for all k               [branch flow limits]

    In DC-OPF, branch flows are expressed via PTDF:
        f = PTDF @ (P_injection_bus)
          = PTDF @ (Map_g @ Pg_all - Pd)
    where:
        PTDF   : (n_lines, n_buses)  -- power transfer distribution factors
        Map_g  : (n_buses, n_gen)    -- generator-to-bus injection mapping
        Pd     : (n_buses,)          -- bus load demand

    The branch flow constraint becomes:
        -Pl_max <= PTDF @ Map_g @ u - PTDF @ Pd <= Pl_max
    i.e.:
        -Pl_max - PTDF @ Pd <= (PTDF @ Map_g) @ u <= Pl_max - PTDF @ Pd

    This is a convex QP solved via scipy SLSQP (O(1/tau) convergence,
    consistent with the fast dual proximal gradient algorithm in the paper).

    Parameters:
    -----------
    Pg_pred_all : np.ndarray, shape (n_samples, n_gen)
        Predicted full generator outputs (p.u.), including slack.
    x_pd : np.ndarray, shape (n_samples, n_buses)
        Bus load demand (p.u.).
    params : dict
        System parameters from dcopf_data_setup.load_parameters_from_csv().

    Returns:
    --------
    Pg_corrected_all : np.ndarray, shape (n_samples, n_gen)
        Post-processed (feasible) generator outputs (p.u.).
    """
    n_samples, n_gen = Pg_pred_all.shape

    # -------------------------------------------------------------------------
    # Extract constraint data from params
    # -------------------------------------------------------------------------
    Pg_min = params['constraints']['Pg_min'].ravel()        # (n_gen,)
    Pg_max = params['constraints']['Pg_max'].ravel()        # (n_gen,)
    Pl_max = params['constraints']['Pl_max'].ravel()        # (n_lines,)

    # In dcopf_data_setup.py, both matrices are stored transposed:
    #   ptdf_T        = ptdf_matrix.T  -> params['PTDF']  shape (n_buses, n_lines)
    #   bus_gen_map_T = bus_gen_map.T  -> params['Map_g'] shape (n_gen,  n_buses)
    # We need:
    #   PTDF  : (n_lines, n_buses)  for  f = PTDF  @ net_injection_bus
    #   Map_g : (n_buses, n_gen)    for  P_bus_gen = Map_g @ Pg_all
    PTDF  = params['constraints']['PTDF'].T     # (n_buses, n_lines).T -> (n_lines, n_buses)
    Map_g = params['constraints']['Map_g'].T    # (n_gen,  n_buses).T  -> (n_buses, n_gen)

    # Only include branches with finite limits (index along n_lines dimension)
    finite_mask   = Pl_max < 1e10              # (n_lines,)
    PTDF_finite   = PTDF[finite_mask, :]       # (n_finite_lines, n_buses)
    Pl_max_finite = Pl_max[finite_mask]        # (n_finite_lines,)

    # Precompute PTDF_finite @ Map_g  (n_finite_lines, n_gen) -- constant across samples
    # Branch flow: f = PTDF_finite @ (Map_g @ u - Pd)
    PTDF_Mg = PTDF_finite @ Map_g              # (n_finite_lines, n_gen)

    # Generator bounds
    bounds = [(float(Pg_min[g]), float(Pg_max[g])) for g in range(n_gen)]

    Pg_corrected_all = np.zeros_like(Pg_pred_all)

    for i in range(n_samples):
        pg_pred_i = Pg_pred_all[i].astype(np.float64)
        pd_bus_i  = x_pd[i].astype(np.float64)             # (n_buses,)
        pd_total_i = pd_bus_i.sum()

        # PTDF @ Pd  (n_finite_lines,) -- constant offset for this sample
        ptdf_pd_i = PTDF_finite @ pd_bus_i                  # (n_finite_lines,)

        # ---------------------------------------------------------------------
        # Objective: min 0.5 * ||u - pg_pred_i||^2
        # ---------------------------------------------------------------------
        def objective(u):
            diff = u - pg_pred_i
            return 0.5 * np.dot(diff, diff)

        def grad_objective(u):
            return u - pg_pred_i

        # ---------------------------------------------------------------------
        # Equality constraint: sum(Map_g @ u) = sum(Pd)
        # i.e. 1^T Map_g u = pd_total
        # row_sum_Mg: (n_gen,)  = 1^T @ Map_g
        # ---------------------------------------------------------------------
        row_sum_Mg = Map_g.sum(axis=0)                      # (n_gen,) -- sum bus contributions per generator

        eq_constraints = [
            {
                'type': 'eq',
                'fun': lambda u, rs=row_sum_Mg, pd=pd_total_i: rs @ u - pd,
                'jac': lambda u, rs=row_sum_Mg: rs
            }
        ]

        # ---------------------------------------------------------------------
        # Inequality constraints (SLSQP convention: fun(u) >= 0):
        #   Pl_max - (PTDF_Mg @ u - ptdf_pd) >= 0   [upper branch limit]
        #   Pl_max + (PTDF_Mg @ u - ptdf_pd) >= 0   [lower branch limit]
        # ---------------------------------------------------------------------
        ineq_constraints = [
            {
                'type': 'ineq',
                'fun':  lambda u, A=PTDF_Mg, b=ptdf_pd_i, pl=Pl_max_finite:
                            pl - (A @ u - b),
                'jac':  lambda u, A=PTDF_Mg: -A
            },
            {
                'type': 'ineq',
                'fun':  lambda u, A=PTDF_Mg, b=ptdf_pd_i, pl=Pl_max_finite:
                            pl + (A @ u - b),
                'jac':  lambda u, A=PTDF_Mg: A
            }
        ]

        all_constraints = eq_constraints + ineq_constraints

        # ---------------------------------------------------------------------
        # Warm start: clip prediction to bounds then adjust for power balance
        # ---------------------------------------------------------------------
        u0 = np.clip(pg_pred_i, Pg_min, Pg_max)
        residual = pd_total_i - (row_sum_Mg @ u0)
        if abs(residual) > 1e-8:
            if residual > 0:
                cap = Pg_max - u0
                total_cap = (row_sum_Mg * cap).sum()
                if total_cap > 1e-8:
                    ratio = min(1.0, residual / total_cap)
                    u0 += cap * ratio
            else:
                cap = u0 - Pg_min
                total_cap = (row_sum_Mg * cap).sum()
                if total_cap > 1e-8:
                    ratio = min(1.0, abs(residual) / total_cap)
                    u0 -= cap * ratio
        u0 = np.clip(u0, Pg_min, Pg_max)

        # ---------------------------------------------------------------------
        # Solve QP
        # ---------------------------------------------------------------------
        result = minimize(
            fun=objective,
            x0=u0,
            jac=grad_objective,
            method='SLSQP',
            bounds=bounds,
            constraints=all_constraints,
            options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False}
        )

        if result.success:
            Pg_corrected_all[i] = result.x
        else:
            # QP failed: fall back to warm start
            Pg_corrected_all[i] = u0

    return Pg_corrected_all


# ============================================================================
# Physics Constraint Penalty Calculation (Original DeepOPF Paper, Section 3.3-3.4)
# ============================================================================
#
# Follows Pan et al. "DeepOPF: Deep Neural Network for DC Optimal Power Flow"
# exactly:
#
# 1. Line flow constraint is normalised to [-1, 1]:
#        (A * theta_hat)_k = f_k / Pl_max_k,   should lie in [-1, 1]
#
# 2. Penalty function (Eq. 8):
#        p(x) = x^2 - 1
#    which is <= 0 when |x| <= 1 (feasible) and > 0 when |x| > 1 (infeasible).
#
# 3. L_pen (Eq. 12):
#        L_pen = (1/n_a) * sum_{k=1}^{n_a} p( (A * theta_hat)_k )
#
# 4. Because theta = B_inv @ (Pg - Pd) is LINEAR in Pg, and p(x) = x^2 - 1
#    is differentiable, the entire penalty is analytically differentiable
#    w.r.t. the DNN output (scaling factors alpha). Standard PyTorch autograd
#    is used -- no zero-order gradient estimation needed.
#
# Note: compute_dcopf_penalty() is retained (unchanged) for the evaluation
#       pipeline which uses the feasibility() function from dcopf_violation_metrics.
# ============================================================================

def compute_dcopf_penalty(y_pred_pg_non_slack, x_pd, params):
    """Calculate DCOPF constraint violation penalty (for evaluation only)."""
    n_samples = y_pred_pg_non_slack.shape[0]

    pd_total = x_pd.sum(axis=1)
    y_pred_pg_full = reconstruct_full_pg(
        pg_non_slack=y_pred_pg_non_slack,
        pd_total=pd_total,
        params=params
    )

    gen_up_viol, gen_lo_viol, line_viol, balance_err = feasibility(
        y_pred_pg=y_pred_pg_full,
        x_pd=x_pd,
        params=params
    )

    ctol = 1e-4
    penalties = np.zeros(n_samples)

    for i in range(n_samples):
        pg_viol = gen_up_viol[i, :] + gen_lo_viol[i, :]
        pg_viol[pg_viol < ctol] = 0
        pg_penalty = np.sum(np.abs(pg_viol))

        line_v = line_viol[i, :]
        line_v[line_v < ctol] = 0
        line_penalty = np.sum(np.abs(line_v))

        balance_penalty = np.abs(balance_err[i])
        if balance_penalty < ctol:
            balance_penalty = 0

        penalties[i] = pg_penalty + line_penalty + balance_penalty

    return penalties


def build_penalty_tensors(params, device):
    """
    Pre-compute constant torch tensors needed by the differentiable penalty.

    Returns a dict of tensors on `device`.

    Derivation
    ----------
    Line flow for branch k:
        f_k = (PTDF^T @ P_net_injection)_k
            = (PTDF^T @ (Map_g^T @ Pg_all - Pd))_k

    Normalised flow (paper Eq. 10):
        (A theta_hat)_k = f_k / Pl_max_k

    We only penalise branches that have finite limits (Pl_max < 1e10).
    """
    Pg_min = params['constraints']['Pg_min'].ravel().astype('float32')
    Pg_max = params['constraints']['Pg_max'].ravel().astype('float32')
    Pl_max = params['constraints']['Pl_max'].ravel().astype('float32')

    # PTDF stored as (n_buses, n_branches);  Map_g stored as (n_g, n_buses)
    PTDF_np = params['constraints']['PTDF'].astype('float32')   # (n_buses, n_branches)
    Map_g_np = params['constraints']['Map_g'].astype('float32') # (n_g, n_buses)

    # Identify constrained branches
    finite_mask = Pl_max < 1e10
    PTDF_finite = PTDF_np[:, finite_mask]         # (n_buses, n_finite)
    Pl_max_finite = Pl_max[finite_mask]            # (n_finite,)

    # Identify slack and non-slack generators
    slack_gen_indices = params['general']['slack_gen_indices']
    non_slack_gen_indices = params['general']['non_slack_gen_indices']

    tensors = {
        'Pg_min': torch.tensor(Pg_min, device=device),                     # (n_g,)
        'Pg_max': torch.tensor(Pg_max, device=device),                     # (n_g,)
        'Map_g_T': torch.tensor(Map_g_np.T, dtype=torch.float32, device=device),
                                                                           # (n_buses, n_g)
        'PTDF_finite': torch.tensor(PTDF_finite, dtype=torch.float32, device=device),
                                                                           # (n_buses, n_finite)
        'Pl_max_finite': torch.tensor(Pl_max_finite, device=device),       # (n_finite,)
        'n_finite': int(finite_mask.sum()),
        'slack_gen_indices': slack_gen_indices,
        'non_slack_gen_indices': non_slack_gen_indices,
        'n_g': params['general']['n_g'],
    }
    return tensors


def compute_penalty_differentiable(alpha_non_slack, x_pd_scaled, scalers, penalty_tensors):
    """
    Compute the DeepOPF line-flow penalty (Eq. 12) in a fully differentiable way.

    Parameters
    ----------
    alpha_non_slack : torch.Tensor, shape (batch, n_g_non_slack)
        Sigmoid output of the DNN (scaling factors for non-slack generators).
    x_pd_scaled : torch.Tensor, shape (batch, n_buses)
        MinMax-scaled load input (as fed to the DNN).
    scalers : dict
        Contains 'x' and 'y_pg_non_slack' MinMaxScalers.
    penalty_tensors : dict
        Pre-computed constant tensors from build_penalty_tensors().

    Returns
    -------
    L_pen : torch.Tensor, scalar
        Mean penalty over the batch (Eq. 12).
    """
    device = alpha_non_slack.device

    # ------------------------------------------------------------------
    # 1. Inverse-transform scaling factors -> physical Pg (non-slack)
    #    MinMaxScaler: x_raw = x_scaled * (data_max - data_min) + data_min
    # ------------------------------------------------------------------
    scaler_pg = scalers['y_pg_non_slack']
    pg_scale = torch.tensor(scaler_pg.scale_, dtype=torch.float32, device=device)       # 1/(max-min)
    pg_min_val = torch.tensor(scaler_pg.min_, dtype=torch.float32, device=device)       # -min/(max-min)
    # inverse: pg_raw = (alpha - pg_min_val) / pg_scale   ... but MinMaxScaler stores
    #   X_std = (X - X.min) / (X.max - X.min)  =>  X = X_std * (X.max - X.min) + X.min
    #   i.e. data_range_ = X.max - X.min,  data_min_ = X.min
    pg_data_range = torch.tensor(scaler_pg.data_range_, dtype=torch.float32, device=device)
    pg_data_min   = torch.tensor(scaler_pg.data_min_,   dtype=torch.float32, device=device)
    pg_non_slack = alpha_non_slack * pg_data_range + pg_data_min   # (batch, n_g_ns)

    # ------------------------------------------------------------------
    # 2. Inverse-transform scaled load -> physical Pd
    # ------------------------------------------------------------------
    scaler_x = scalers['x']
    x_data_range = torch.tensor(scaler_x.data_range_, dtype=torch.float32, device=device)
    x_data_min   = torch.tensor(scaler_x.data_min_,   dtype=torch.float32, device=device)
    x_pd_raw = x_pd_scaled * x_data_range + x_data_min   # (batch, n_buses)

    # ------------------------------------------------------------------
    # 3. Reconstruct full Pg (insert slack via power balance)
    # ------------------------------------------------------------------
    n_g = penalty_tensors['n_g']
    slack_idx = penalty_tensors['slack_gen_indices']
    non_slack_idx = penalty_tensors['non_slack_gen_indices']
    batch_size = alpha_non_slack.shape[0]

    pg_full = torch.zeros(batch_size, n_g, dtype=torch.float32, device=device)
    pg_full[:, non_slack_idx] = pg_non_slack

    pd_total = x_pd_raw.sum(dim=1)                     # (batch,)
    pg_ns_total = pg_non_slack.sum(dim=1)               # (batch,)
    pg_slack_total = pd_total - pg_ns_total              # (batch,)
    n_slack = len(slack_idx)
    if n_slack > 0:
        pg_slack_each = pg_slack_total / n_slack         # (batch,)
        for s_idx in slack_idx:
            pg_full[:, s_idx] = pg_slack_each

    # ------------------------------------------------------------------
    # 4. Compute line flows:  f = P_net_injection @ PTDF_finite
    #    P_net_injection = Pg_bus - Pd = pg_full @ Map_g - x_pd_raw
    #    Here Map_g is (n_g, n_buses) so pg_full @ Map_g -> (batch, n_buses)
    # ------------------------------------------------------------------
    Map_g_T = penalty_tensors['Map_g_T']                # (n_buses, n_g)
    Pg_bus = torch.matmul(pg_full, Map_g_T.T)           # (batch, n_buses)
    P_net = Pg_bus - x_pd_raw                            # (batch, n_buses)

    PTDF_finite = penalty_tensors['PTDF_finite']        # (n_buses, n_finite)
    line_flows = torch.matmul(P_net, PTDF_finite)       # (batch, n_finite)

    # ------------------------------------------------------------------
    # 5. Normalise flows and compute penalty  p(x) = x^2 - 1  (Eq. 8, 12)
    #    normalised_flow_k = f_k / Pl_max_k  (should be in [-1, 1])
    # ------------------------------------------------------------------
    Pl_max_f = penalty_tensors['Pl_max_finite']         # (n_finite,)
    normalised_flows = line_flows / Pl_max_f.unsqueeze(0)   # (batch, n_finite)

    p_values = normalised_flows ** 2 - 1.0              # (batch, n_finite)

    # L_pen = (1/n_a) * sum_k p((A theta_hat)_k)   (Eq. 12)
    n_a = penalty_tensors['n_finite']
    L_pen_per_sample = p_values.sum(dim=1) / n_a        # (batch,)
    L_pen = L_pen_per_sample.mean()                     # scalar

    return L_pen


# ============================================================================
# PINN Model (Differentiable Penalty — Original DeepOPF Paper)
# ============================================================================

class PINN_DCOPF(nn.Module):
    """
    Physics-Informed Neural Network for DCOPF.

    The penalty is computed as a standard differentiable PyTorch operation,
    following the original DeepOPF paper (Eq. 8, 12). No custom backward
    or zero-order gradient estimation is needed because the mapping from
    DNN output (scaling factors) to line flows is entirely linear and the
    penalty function p(x) = x^2 - 1 is smooth.
    """

    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 256]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x_sol = self.net(x)
        # Penalty is computed externally in the training loop
        return x_sol


# ============================================================================
# Data Loading Function (v6.1 FIXED)
# ============================================================================

def load_and_prepare_deepopf_data(file_path, params, column_names):
    """
    Load and prepare DeepOPF training data

    v6.1 FIXES:
    -----------
    1. Load demand mapping: uses bus_id_to_idx dict instead of bus_id-1 indexing.
       This is critical for case300 where bus IDs like 7001, 9533 exist.
    2. Generator columns: explicitly built from g_bus order in params,
       instead of unsorted column scanning which may produce wrong column order.
    """
    import pandas as pd

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
            f"[load_and_prepare_deepopf_data] Missing pg columns in CSV: {missing}\n"
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


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(
        model,
        X,
        indices,
        raw_data,
        params,
        scalers,
        device,
        test_data_external=None,
        test_params=None
):
    """
    Evaluate model with both NN-only and strict DeepOPF post-processing metrics.

    Returns:
    --------
    result : dict
        Contains both 'nn_only' and 'post_processed' metric dicts.
    """
    eval_params = test_params if test_params is not None else params

    global GLOBAL_PARAMS
    original_global_params = GLOBAL_PARAMS
    GLOBAL_PARAMS = eval_params

    try:
        model.eval()

        if test_data_external is not None:
            x_raw_eval    = test_data_external['x']
            y_true_pg_all = test_data_external['y_pg_all']
            x_scaled  = scalers['x'].transform(x_raw_eval)
            X_eval    = torch.tensor(x_scaled, dtype=torch.float32, device=device)

            with torch.no_grad():
                y_pred_non_slack_scaled = model(X_eval)
        else:
            with torch.no_grad():
                y_pred_non_slack_scaled = model(X.to(device))
            x_raw_eval    = raw_data['x'][indices]
            y_true_pg_all = raw_data['y_pg_all'][indices]

        y_pred_non_slack_scaled_np = y_pred_non_slack_scaled.cpu().numpy()
        y_pred_non_slack = scalers['y_pg_non_slack'].inverse_transform(
            y_pred_non_slack_scaled_np
        )

        non_slack_indices = eval_params['general']['non_slack_gen_indices']

        # ====================================================================
        # Step 1: Reconstruct full Pg (slack filled by power balance)
        # ====================================================================
        pd_total = x_raw_eval.sum(axis=1)
        y_pred_pg_all_raw = reconstruct_full_pg(
            pg_non_slack=y_pred_non_slack,
            pd_total=pd_total,
            params=eval_params
        )

        # ====================================================================
        # Compute NN-only metrics (before post-processing)
        # ====================================================================
        cost_coeffs = {
            'C2': eval_params['constraints'].get('C_Pg_c2', np.zeros(y_true_pg_all.shape[1])),
            'C1': eval_params['constraints']['C_Pg'],
            'C0': eval_params['constraints'].get('C_Pg_c0', np.zeros(y_true_pg_all.shape[1]))
        }
        cost_true = compute_cost(y_true_pg_all, cost_coeffs)

        # NN-only MAE
        nn_mae_dict = compute_detailed_mae(
            y_true_all=y_true_pg_all,
            y_pred_non_slack=y_pred_non_slack,
            y_pred_all=y_pred_pg_all_raw,
            params=eval_params
        )

        # NN-only violations
        nn_gen_up, nn_gen_lo, nn_line_viol, nn_balance = feasibility(
            y_pred_pg=y_pred_pg_all_raw,
            x_pd=x_raw_eval,
            params=eval_params
        )
        nn_viol_dict = compute_detailed_pg_violations_pu(
            gen_up_viol=nn_gen_up, gen_lo_viol=nn_gen_lo, params=eval_params
        )
        nn_viol_branch = compute_branch_violation_pu(
            line_viol=nn_line_viol, Pl_max=eval_params['constraints']['Pl_max']
        )

        # NN-only cost
        nn_cost_pred = compute_cost(y_pred_pg_all_raw, cost_coeffs)
        nn_cost_gap  = compute_cost_gap_percentage(cost_true, nn_cost_pred)

        nn_result = {
            'mae_pg_non_slack':  nn_mae_dict['mae_non_slack'],
            'mae_pg_slack':      nn_mae_dict['mae_slack'],
            'viol_pg_non_slack': nn_viol_dict['viol_non_slack'],
            'viol_pg_slack':     nn_viol_dict['viol_slack'],
            'viol_branch':       nn_viol_branch,
            'cost_gap_percent':  nn_cost_gap,
        }

        # ====================================================================
        # Step 2: Strict DeepOPF QP post-processing (Section 3.5)
        # ====================================================================
        n_samples = len(x_raw_eval)
        print(f"  Running strict QP post-processing for {n_samples} samples...")
        t_pp_start = time.time()

        y_pred_pg_all_pp = post_process_solution_qp(
            Pg_pred_all=y_pred_pg_all_raw,
            x_pd=x_raw_eval,
            params=eval_params
        )

        t_pp_end = time.time()
        pp_total = t_pp_end - t_pp_start
        print(f"  Done: {pp_total:.2f}s total, "
              f"{pp_total / n_samples * 1000:.4f} ms/sample")

        # Post-processed non-slack for MAE
        y_pred_non_slack_pp = y_pred_pg_all_pp[:, non_slack_indices]

        # Post-processed MAE
        pp_mae_dict = compute_detailed_mae(
            y_true_all=y_true_pg_all,
            y_pred_non_slack=y_pred_non_slack_pp,
            y_pred_all=y_pred_pg_all_pp,
            params=eval_params
        )

        # Post-processed violations
        pp_gen_up, pp_gen_lo, pp_line_viol, pp_balance = feasibility(
            y_pred_pg=y_pred_pg_all_pp,
            x_pd=x_raw_eval,
            params=eval_params
        )
        pp_viol_dict = compute_detailed_pg_violations_pu(
            gen_up_viol=pp_gen_up, gen_lo_viol=pp_gen_lo, params=eval_params
        )
        pp_viol_branch = compute_branch_violation_pu(
            line_viol=pp_line_viol, Pl_max=eval_params['constraints']['Pl_max']
        )

        # Post-processed cost
        pp_cost_pred = compute_cost(y_pred_pg_all_pp, cost_coeffs)
        pp_cost_gap  = compute_cost_gap_percentage(cost_true, pp_cost_pred)

        pp_result = {
            'mae_pg_non_slack':  pp_mae_dict['mae_non_slack'],
            'mae_pg_slack':      pp_mae_dict['mae_slack'],
            'viol_pg_non_slack': pp_viol_dict['viol_non_slack'],
            'viol_pg_slack':     pp_viol_dict['viol_slack'],
            'viol_branch':       pp_viol_branch,
            'cost_gap_percent':  pp_cost_gap,
        }

        result = {
            'nn_only': nn_result,
            'post_processed': pp_result,
        }

    finally:
        GLOBAL_PARAMS = original_global_params

    return result


# ============================================================================
# Main Training Function
# ============================================================================

def train_pinn_dcopf(
        case_name,
        params_path,
        dataset_path,
        column_names,
        n_train_use=10000,
        hidden_sizes=[256, 256],
        n_epochs=1000,
        patience=20,
        min_delta=1e-6,
        batch_size=256,
        learning_rate=1e-3,
        penalty_weight=0.1,
        seed=42,
        device='cuda',
        split_mode=DataSplitMode.RANDOM_SPLIT,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=1000
):
    """
    PINN for DCOPF training with strict DeepOPF QP post-processing.

    Early stopping monitors val_loss = MSE + penalty_weight * physics_penalty.
    Best model weights (lowest val_loss) are restored before evaluation.
    """
    global GLOBAL_PARAMS, GLOBAL_SCALERS

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    # Load parameters
    params = load_parameters_from_csv(case_name, params_path, is_api=False)
    slack_info = identify_slack_bus_and_gens(params)
    params = update_params_with_slack_info(params, slack_info)
    GLOBAL_PARAMS = params

    test_params = None
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError('API_TEST mode requires test_params_path')
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_slack_info = identify_slack_bus_and_gens(test_params)
        test_params = update_params_with_slack_info(test_params, test_slack_info)
    else:
        test_params = params

    n_gen_non_slack = params['general']['n_g_non_slack']

    # Load data
    x_data_raw, y_pg_raw_non_slack, y_pg_raw_all = load_and_prepare_deepopf_data(
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

    # Data normalization
    x_scaler = MinMaxScaler().fit(x_data_raw[train_idx])
    y_pg_non_slack_scaler = MinMaxScaler().fit(y_pg_raw_non_slack[train_idx])
    scalers = {'x': x_scaler, 'y_pg_non_slack': y_pg_non_slack_scaler}
    GLOBAL_SCALERS = scalers

    x_train_scaled = x_scaler.transform(x_data_raw[train_idx])
    y_train_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[train_idx])
    x_val_scaled   = x_scaler.transform(x_data_raw[val_idx])
    y_val_scaled   = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[val_idx])

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        X_test = None
    else:
        x_test_scaled = x_scaler.transform(x_data_raw[test_idx])
        X_test = torch.tensor(x_test_scaled, dtype=torch.float32)

    X_train = torch.from_numpy(x_train_scaled).float().to(device)
    Y_train = torch.from_numpy(y_train_scaled).float().to(device)
    X_val   = torch.from_numpy(x_val_scaled).float().to(device)
    Y_val_tensor = torch.from_numpy(y_val_scaled).float().to(device)

    train_dataset = Data.TensorDataset(X_train, Y_train)
    train_loader  = Data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    # Build model
    model = PINN_DCOPF(
        input_dim=x_data_raw.shape[1],
        output_dim=n_gen_non_slack,
        hidden_sizes=hidden_sizes
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    # Build pre-computed penalty tensors (constant matrices on device)
    penalty_tensors = build_penalty_tensors(params, device)

    # --- Early stopping state ---
    best_val_loss    = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"Early stopping — patience: {patience}, min_delta: {min_delta}")
    print(f"Val loss monitored: MSE + {penalty_weight} * physics_penalty\n")

    # ========================================================================
    # Training
    # ========================================================================
    training_start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_total = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            mse_loss = criterion(pred, batch_y)
            penalty = compute_penalty_differentiable(
                pred, batch_x, scalers, penalty_tensors
            )
            total_loss = 1.0 * mse_loss + penalty_weight * penalty
            total_loss.backward()
            optimizer.step()
            epoch_total += total_loss.item() * len(batch_x)

        avg_total = epoch_total / len(X_train)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mse  = criterion(val_pred, Y_val_tensor)
            val_penalty = compute_penalty_differentiable(
                val_pred, X_val, scalers, penalty_tensors
            )
            val_loss = 1.0 * val_mse.item() + penalty_weight * val_penalty.item()

        # --- Early stopping check ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss    = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        print(f"Epoch {epoch}/{n_epochs} - train_loss: {avg_total:.6f} - val_loss: {val_loss:.6f}"
              f" - patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n[Early Stopping] No improvement for {patience} epochs. "
                  f"Best val_loss: {best_val_loss:.6f} (epoch {epoch - patience})")
            break

    # Restore best model weights before evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[Early Stopping] Best model weights restored (val_loss={best_val_loss:.6f})")

    train_time = time.time() - training_start

    # ========================================================================
    # Evaluation
    # ========================================================================
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }
        test_metrics = evaluate_model(
            model=model, X=None, indices=None,
            raw_data=raw_data, params=params, scalers=scalers, device=device,
            test_data_external=test_data_external_dict, test_params=test_params
        )
    else:
        test_metrics = evaluate_model(
            model=model, X=X_test, indices=test_idx,
            raw_data=raw_data, params=params, scalers=scalers, device=device,
            test_params=test_params
        )

    # ========================================================================
    # Speed test - Stage 1: Neural Network
    # ========================================================================
    model.eval()

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_sample     = torch.tensor(
            x_scaler.transform(x_test_external[:1]), dtype=torch.float32, device=device
        )
        test_sample_raw = x_test_external[:1]
    else:
        test_sample     = X_test[:1].to(device)
        test_sample_raw = x_data_raw[test_idx[:1]]

    times_nn = []
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_sample)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        for _ in range(100):
            t_start = time.time()
            pred_scaled = model(test_sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times_nn.append(time.time() - t_start)

    nn_time_ms = np.mean(times_nn) * 1000

    # ========================================================================
    # Speed test - Stage 2: QP Post-processing (single sample)
    # ========================================================================
    with torch.no_grad():
        pred_scaled_single = model(test_sample)
    pred_np    = pred_scaled_single.cpu().numpy()
    pred_raw_ns = scalers['y_pg_non_slack'].inverse_transform(pred_np)

    pd_total_single  = test_sample_raw.sum(axis=1)
    pred_raw_full    = reconstruct_full_pg(
        pg_non_slack=pred_raw_ns,
        pd_total=pd_total_single,
        params=test_params
    )

    times_pp = []
    for _ in range(20):   # Fewer reps: QP is slower than NN
        t_start = time.time()
        _ = post_process_solution_qp(
            Pg_pred_all=pred_raw_full,
            x_pd=test_sample_raw,
            params=test_params
        )
        times_pp.append(time.time() - t_start)

    pp_time_ms = np.mean(times_pp) * 1000

    # ========================================================================
    # Print results
    # ========================================================================
    nn_m = test_metrics['nn_only']
    pp_m = test_metrics['post_processed']

    print("\n" + "=" * 70)
    print("Test Set Results — NN Only (No Post-processing)")
    print("=" * 70)
    print(f"\nNon-Slack Generators:")
    print(f"  MAE:        {nn_m['mae_pg_non_slack']:.4f}%")
    print(f"  Violation:  {nn_m['viol_pg_non_slack']:.4f} p.u.")
    print(f"\nSlack-Only Generators:")
    print(f"  MAE:        {nn_m['mae_pg_slack']:.4f}%")
    print(f"  Violation:  {nn_m['viol_pg_slack']:.4f} p.u.")
    print(f"\nBranch:")
    print(f"  Violation:  {nn_m['viol_branch']:.4f} p.u.")
    print(f"\nCost Gap:     {nn_m['cost_gap_percent']:.4f}%")

    print("\n" + "=" * 70)
    print("Test Set Results — With QP Post-processing")
    print("=" * 70)
    print(f"\nNon-Slack Generators:")
    print(f"  MAE:        {pp_m['mae_pg_non_slack']:.4f}%")
    print(f"  Violation:  {pp_m['viol_pg_non_slack']:.4f} p.u.")
    print(f"\nSlack-Only Generators:")
    print(f"  MAE:        {pp_m['mae_pg_slack']:.4f}%")
    print(f"  Violation:  {pp_m['viol_pg_slack']:.4f} p.u.")
    print(f"\nBranch:")
    print(f"  Violation:  {pp_m['viol_branch']:.4f} p.u.")
    print(f"\nCost Gap:     {pp_m['cost_gap_percent']:.4f}%")

    print("\n" + "=" * 70)
    print("Overall Performance Summary")
    print("=" * 70)
    print(f"\nTraining Time:   {train_time:.2f} s")
    print(f"\nInference Time (per sample):")
    print(f"  Stage 1 (Neural Network):            {nn_time_ms:.4f} ms")
    print(f"  Stage 2 (QP Post-processing, SLSQP): {pp_time_ms:.4f} ms")
    print(f"  Total (NN + QP):                     {nn_time_ms + pp_time_ms:.4f} ms")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# Main Program
# ============================================================================

if __name__ == '__main__':

    CASE_NAME       = 'pglib_opf_case118_ieee'
    CASE_SHORT_NAME = 'case118'
    SPLIT_MODE      = DataSplitMode.VALID_FIXED
    N_TRAIN_USE     = 5000
    N_TEST_SAMPLES  = 1000
    N_EPOCHS        = 1000
    PATIENCE        = 20
    MIN_DELTA       = 1e-6
    LEARNING_RATE   = 1e-3
    BATCH_SIZE      = 64
    HIDDEN_SIZES    = [128, 64]
    PENALTY_WEIGHT  = 0.00001
    SEED            = 42

    ROOT_DIR = "/lambda/nfs/lxy/dcopf_project/data"
    TRAIN_VARIANCE = "v=0.12"
    TEST_VARIANCE  = "v=0.25"

    COLUMN_NAMES = {
        'load_prefix':        'pd',
        'gen_prefix':         'pg',
        'lambda':             'lambda',
        'mu_g_min_prefix':    'mu_g_min_',
        'mu_g_max_prefix':    'mu_g_max_',
        'mu_line_pos_prefix': 'mu_line_max_',
        'mu_line_neg_prefix': 'mu_line_min_',
    }

    params_path     = os.path.join(ROOT_DIR, "DCOPF Constraints", CASE_SHORT_NAME)
    train_data_path = os.path.join(
        ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TRAIN_VARIANCE})",
        f"{CASE_NAME}_dataset_with_duals.csv"
    )

    if SPLIT_MODE == DataSplitMode.GENERALIZATION:
        test_data_path  = os.path.join(
            ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TEST_VARIANCE})",
            f"{CASE_NAME}_dataset_with_duals.csv"
        )
        test_params_path = None
    elif SPLIT_MODE == DataSplitMode.API_TEST:
        test_data_path  = os.path.join(
            ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}(v=api)",
            f"{CASE_NAME}__api_dataset_with_duals.csv"
        )
        test_params_path = os.path.join(
            ROOT_DIR, "DCOPF Constraints", f"{CASE_SHORT_NAME}(api)"
        )
    else:
        test_data_path   = None
        test_params_path = None

    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    train_pinn_dcopf(
        case_name=CASE_NAME,
        params_path=params_path,
        dataset_path=train_data_path,
        column_names=COLUMN_NAMES,
        n_train_use=N_TRAIN_USE,
        hidden_sizes=HIDDEN_SIZES,
        n_epochs=N_EPOCHS,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        penalty_weight=PENALTY_WEIGHT,
        seed=SEED,
        device=device_name,
        split_mode=SPLIT_MODE,
        test_data_path=test_data_path,
        test_params_path=test_params_path,
        n_test_samples=N_TEST_SAMPLES
    )