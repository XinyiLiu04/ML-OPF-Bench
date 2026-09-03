# -*- coding: utf-8 -*-
"""
Lagrangian Dual ACOPF — Strict Replication of Fioretto et al. 2019, Model M_C^D

Key design decisions:
  - Network: 4 sub-networks (Out-v, Out-θ, Out-p^g, Out-q^g), sizes parameterized
    by l (load buses), n (all buses), g (non-slack generators), g_total (all generators).
  - Loss: Lo = MSE(v) + MSE(θ) + MSE(p^g) + MSE(q^g) + Σ λ_c ν_c
  - 9 constraint violation degrees computed analytically on GPU tensors (differentiable).
  - Lagrangian dual update (Algorithm 1): λ updated inside mini-batch loop.
  - Slack bus Pg excluded from DNN prediction; restored by power flow at evaluation.
  - Supports 4 data modes; trains once, evaluates on up to 3 test sets.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import sys
from pathlib import Path

# PyPower
from pypower.api import runpf, ppoption

# Import configuration
try:
    import acopf_config
except ImportError:
    print("Error: Unable to import acopf_config.py")
    sys.exit(1)

# Import data modules
try:
    from acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        DataMode,
        prepare_data_splits,
        load_generalization_test_data,
        load_api_test_data,
        reconstruct_full_pg,
    )
except ImportError:
    print("Error: Unable to import 'acopf_data_setup' module.")
    sys.exit(1)

# Import evaluation
try:
    from acopf_violation_metrics import evaluate_acopf_predictions
except ImportError:
    print("Error: Unable to import 'acopf_violation_metrics' module.")
    sys.exit(1)


# =====================================================================
# PyPower Interface
# =====================================================================
GLOBAL_CASE_DATA = None
PPOPT = None


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
    bus[:, 2] = bus_df['pd_pu'].values * baseMVA
    bus[:, 3] = bus_df['qd_pu'].values * baseMVA
    bus[:, 6] = 1
    bus[:, 7] = bus_df['vm_pu'].values
    bus[:, 8] = bus_df['va_deg'].values
    bus[:, 9] = bus_df['base_kv'].values if 'base_kv' in bus_df.columns else 1.0
    bus[:, 10] = 1
    bus[:, 11] = bus_df['vmax_pu'].values
    bus[:, 12] = bus_df['vmin_pu'].values

    gen = np.zeros((len(gen_df), 21))
    gen[:, 0] = gen_df['bus_id'].values
    gen[:, 3] = gen_df['qg_max_pu'].values * baseMVA
    gen[:, 4] = gen_df['qg_min_pu'].values * baseMVA
    gen[:, 5] = gen_df['vg_pu'].values
    gen[:, 6] = baseMVA
    gen[:, 7] = 1
    gen[:, 8] = gen_df['pg_max_pu'].values * baseMVA
    gen[:, 9] = gen_df['pg_min_pu'].values * baseMVA

    branch = np.zeros((len(branch_df), 13))
    branch[:, 0] = branch_df['f_bus'].values
    branch[:, 1] = branch_df['t_bus'].values
    branch[:, 2] = branch_df['r_pu'].values
    branch[:, 3] = branch_df['x_pu'].values
    branch[:, 4] = branch_df['b_pu'].values
    branch[:, 5] = branch_df['rate_a_pu'].values * baseMVA
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

    return {
        'version': '2', 'baseMVA': baseMVA,
        'bus': bus, 'gen': gen, 'branch': branch, 'gencost': gencost,
    }


def solve_pf_for_evaluation(pd, qd, pg_non_slack, vm_gen, params):
    """Run power flow for a single sample."""
    global GLOBAL_CASE_DATA, PPOPT
    BASE_MVA = params['general']['BASE_MVA']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    n_gen = params['general']['n_gen']
    load_bus_ids = params['general']['load_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    mpc = {
        'version': GLOBAL_CASE_DATA['version'],
        'baseMVA': GLOBAL_CASE_DATA['baseMVA'],
        'bus': GLOBAL_CASE_DATA['bus'].copy(),
        'gen': GLOBAL_CASE_DATA['gen'].copy(),
        'branch': GLOBAL_CASE_DATA['branch'].copy(),
        'gencost': GLOBAL_CASE_DATA['gencost'],
    }

    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc["bus"][bus_idx, 2] = pd[i] * BASE_MVA
            mpc["bus"][bus_idx, 3] = qd[i] * BASE_MVA

    for i, gen_idx in enumerate(non_slack_gen_idx):
        mpc["gen"][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    for g in range(n_gen):
        mpc["gen"][g, 5] = vm_gen[g]

    return runpf(mpc, PPOPT)


# =====================================================================
# Neural Network: M_C architecture (4 sub-networks, paper-parameterized)
# =====================================================================
def _build_subnetwork(dims, use_activation_on_last=False):
    """Build a sub-network from a list of (in, out) dimensions.
    All layers use ReLU except the last (no activation by default)."""
    layers = []
    for i, (d_in, d_out) in enumerate(dims):
        layers.append(nn.Linear(d_in, d_out))
        if i < len(dims) - 1:
            layers.append(nn.ReLU())
        elif use_activation_on_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class OPF_DNN_MC(nn.Module):
    """
    M_C architecture from Fioretto et al. 2019 Appendix (without hot-start).

    Input: (p^d, q^d) of dimension 2*l
    Output: v_hat (n), theta_hat (n), pg_hat (g), qg_hat (g_total)

    Architecture (from Appendix Table for M_C, adapted for no hot-start):
      Input block:  2l -> 4l (ReLU) -> 4l (ReLU)  [shared hidden = 4l]
      Out-v:   4l -> 8l (ReLU) -> 4l (ReLU) -> 2n (ReLU) -> n
      Out-θ:   4l -> 8l (ReLU) -> 4l (ReLU) -> 2n (ReLU) -> n
      Out-p^g: 4l -> 8l (ReLU) -> 4l (ReLU) -> 2g (ReLU) -> g
      Out-q^g: 4l -> 8l (ReLU) -> 4l (ReLU) -> 2g_total (ReLU) -> g_total

    Note: g = n_gen_non_slack for Pg head; g_total = n_gen for Qg head.
    """

    def __init__(self, n_loads, n_buses, n_gen_non_slack, n_gen):
        super().__init__()
        l = n_loads
        n = n_buses
        g = n_gen_non_slack
        g_total = n_gen

        # Shared input block
        self.input_block = _build_subnetwork([
            (2 * l, 4 * l),
            (4 * l, 4 * l),
        ], use_activation_on_last=True)  # output: 4l with ReLU

        # Out-v: predicts voltage magnitudes for all buses
        self.out_v = _build_subnetwork([
            (4 * l, 8 * l),
            (8 * l, 4 * l),
            (4 * l, 2 * n),
            (2 * n, n),
        ])

        # Out-theta: predicts voltage angles for all buses
        self.out_theta = _build_subnetwork([
            (4 * l, 8 * l),
            (8 * l, 4 * l),
            (4 * l, 2 * n),
            (2 * n, n),
        ])

        # Out-pg: predicts active power for non-slack generators
        self.out_pg = _build_subnetwork([
            (4 * l, 8 * l),
            (8 * l, 4 * l),
            (4 * l, 2 * g),
            (2 * g, g),
        ])

        # Out-qg: predicts reactive power for all generators
        self.out_qg = _build_subnetwork([
            (4 * l, 8 * l),
            (8 * l, 4 * l),
            (4 * l, 2 * g_total),
            (2 * g_total, g_total),
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, 2*l) scaled input loads
        Returns:
            v_hat:     (batch, n)        scaled vm predictions
            theta_hat: (batch, n)        scaled va predictions
            pg_hat:    (batch, g)        scaled pg predictions (non-slack)
            qg_hat:    (batch, g_total)  scaled qg predictions (all gen)
        """
        h = self.input_block(x)
        v_hat = self.out_v(h)
        theta_hat = self.out_theta(h)
        pg_hat = self.out_pg(h)
        qg_hat = self.out_qg(h)
        return v_hat, theta_hat, pg_hat, qg_hat


# =====================================================================
# Constraint Violation Computation (Tensor, differentiable)
# =====================================================================
class OPFConstraints:
    """
    Pre-computes and caches all network topology tensors on the target device.
    Provides `compute_violations()` that returns 9 violation degrees as tensors.

    All quantities are in p.u. and on the computation device.
    """

    def __init__(self, params, scalers, device):
        self.device = device
        self.params = params
        self.scalers = scalers

        n_buses = params['general']['n_buses']
        n_gen = params['general']['n_gen']
        n_gen_non_slack = params['general']['n_gen_non_slack']
        n_branches = params['general']['n_branches']
        n_loads = params['general']['n_loads']

        # Bounds (as tensors)
        self.vm_min = torch.tensor(params['bus']['vm_min'], dtype=torch.float32, device=device)
        self.vm_max = torch.tensor(params['bus']['vm_max'], dtype=torch.float32, device=device)
        self.pg_min = torch.tensor(params['generator']['pg_min'].flatten(), dtype=torch.float32, device=device)
        self.pg_max = torch.tensor(params['generator']['pg_max'].flatten(), dtype=torch.float32, device=device)
        self.qg_min = torch.tensor(params['generator']['qg_min'].flatten(), dtype=torch.float32, device=device)
        self.qg_max = torch.tensor(params['generator']['qg_max'].flatten(), dtype=torch.float32, device=device)

        # Non-slack gen indices
        self.non_slack_gen_idx = params['general']['non_slack_gen_idx']

        # Slack bus mask: True for slack buses (excluded from KCL penalty)
        slack_gen_mask = params['general']['slack_gen_mask']
        gen_bus_ids = params['general']['gen_bus_ids']
        bus_id_to_idx_dict = params['general']['bus_id_to_idx']
        slack_bus_indices = set()
        for gi, is_slack in enumerate(slack_gen_mask):
            if is_slack:
                slack_bus_indices.add(bus_id_to_idx_dict[int(gen_bus_ids[gi])])
        # non_slack_bus_mask: True for buses that are NOT slack (KCL is penalized)
        self.non_slack_bus_mask = torch.ones(n_buses, dtype=torch.bool, device=device)
        for idx in slack_bus_indices:
            self.non_slack_bus_mask[idx] = False

        # Branch topology
        self.f_idx = torch.tensor(params['branch']['f_bus_idx'], dtype=torch.long, device=device)
        self.t_idx = torch.tensor(params['branch']['t_bus_idx'], dtype=torch.long, device=device)
        self.g_br = torch.tensor(params['branch']['g_br'], dtype=torch.float32, device=device)
        self.b_br = torch.tensor(params['branch']['b_br'], dtype=torch.float32, device=device)
        self.b_sh = torch.tensor(params['branch']['b_pu'], dtype=torch.float32, device=device)  # shunt susceptance

        # Thermal limits
        rate_a = params['branch']['rate_a'].copy()
        rate_a[rate_a <= 0] = 1e6  # no limit -> very large
        self.s_max_sq = torch.tensor(rate_a ** 2, dtype=torch.float32, device=device)

        # Angle difference limits
        self.theta_diff_max = torch.tensor(
            params['branch']['theta_diff_max'], dtype=torch.float32, device=device)

        # Bus-generator mapping: for KCL (pg/qg at each bus)
        # bus_gen_map: (n_buses, n_gen), 1 if gen g is at bus i
        self.bus_gen_map = torch.tensor(
            params['topology']['bus_gen_map'], dtype=torch.float32, device=device)

        # Load bus mapping: which buses have loads (for KCL)
        bus_id_to_idx = params['general']['bus_id_to_idx']
        load_bus_ids = params['general']['load_bus_ids']
        self.load_bus_indices = torch.tensor(
            [bus_id_to_idx[int(lid)] for lid in load_bus_ids],
            dtype=torch.long, device=device)

        # Pre-compute scaler parameters as tensors for inverse transform
        self._cache_scaler_tensors()

        # Sizes
        self.n_buses = n_buses
        self.n_gen = n_gen
        self.n_gen_non_slack = n_gen_non_slack
        self.n_branches = n_branches
        self.n_loads = n_loads

        # Bus-branch incidence for KCL: sparse-like mapping
        # For each bus i, sum of p_f over branches (ij) where i is the from-bus
        # We use scatter_add for this

    def _cache_scaler_tensors(self):
        """Cache MinMaxScaler parameters as tensors for fast inverse transform."""
        for name in ['vm', 'va', 'pg', 'qg', 'x']:
            scaler = self.scalers[name]
            scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=self.device)
            min_val = torch.tensor(scaler.min_, dtype=torch.float32, device=self.device)
            setattr(self, f'_scaler_{name}_scale', scale)
            setattr(self, f'_scaler_{name}_min', min_val)

    def _inv_transform(self, x_scaled, name):
        """Inverse MinMaxScaler transform on tensor."""
        scale = getattr(self, f'_scaler_{name}_scale')
        min_val = getattr(self, f'_scaler_{name}_min')
        return (x_scaled - min_val) / scale

    def compute_branch_flows(self, vm, va):
        """
        Compute branch power flows from Ohm's Law (Eqs 5a, 5b in paper).

        Args:
            vm: (batch, n_buses) voltage magnitudes in p.u.
            va: (batch, n_buses) voltage angles in radians
        Returns:
            pf: (batch, n_branches) active power flow
            qf: (batch, n_branches) reactive power flow
        """
        vi = vm[:, self.f_idx]    # (batch, n_branches)
        vj = vm[:, self.t_idx]
        ai = va[:, self.f_idx]
        aj = va[:, self.t_idx]

        theta_ij = ai - aj
        cos_t = torch.cos(theta_ij)
        sin_t = torch.sin(theta_ij)

        g = self.g_br.unsqueeze(0)  # (1, n_branches)
        b = self.b_br.unsqueeze(0)

        # Ohm's Law (5a): p_f_ij = g_ij * vi^2 - vi*vj*(b_ij*sin + g_ij*cos)
        pf = g * vi ** 2 - vi * vj * (b * sin_t + g * cos_t)

        # Ohm's Law (5b): q_f_ij = -b_ij * vi^2 - vi*vj*(g_ij*sin - b_ij*cos)
        qf = -b * vi ** 2 - vi * vj * (g * sin_t - b * cos_t)

        return pf, qf

    def compute_violations(self, vm_pred_scaled, va_pred_scaled, pg_pred_scaled,
                           qg_pred_scaled, x_scaled, vm_true_scaled, va_true_scaled):
        """
        Compute all 9 violation degrees (tensor, differentiable).

        All *_scaled inputs are in [0,1] MinMaxScaler range.
        Inverse transforms are applied internally.

        Args:
            vm_pred_scaled: (batch, n_buses)
            va_pred_scaled: (batch, n_buses)
            pg_pred_scaled: (batch, n_gen_non_slack)
            qg_pred_scaled: (batch, n_gen)
            x_scaled:       (batch, 2*n_loads)  input loads
            vm_true_scaled: (batch, n_buses)    ground truth vm (for ν5)
            va_true_scaled: (batch, n_buses)    ground truth va (for ν5)

        Returns:
            dict of {violation_name: scalar tensor (batch-averaged)}
        """
        batch = vm_pred_scaled.shape[0]

        # Inverse transform to physical units (p.u.)
        vm_pred = self._inv_transform(vm_pred_scaled, 'vm')
        va_pred = self._inv_transform(va_pred_scaled, 'va')
        pg_pred_ns = self._inv_transform(pg_pred_scaled, 'pg')  # non-slack only
        qg_pred = self._inv_transform(qg_pred_scaled, 'qg')    # all gen
        x_raw = self._inv_transform(x_scaled, 'x')
        vm_true = self._inv_transform(vm_true_scaled, 'vm')
        va_true = self._inv_transform(va_true_scaled, 'va')

        # Reconstruct full pg (with slack = 0, not penalized)
        pg_pred_full = torch.zeros(batch, self.n_gen, device=self.device)
        pg_pred_full[:, self.non_slack_gen_idx] = pg_pred_ns

        # Loads
        pd_raw = x_raw[:, :self.n_loads]   # (batch, n_loads)
        qd_raw = x_raw[:, self.n_loads:]   # (batch, n_loads)

        # ---- ν_2a: Voltage magnitude bounds ----
        vm_lo = torch.clamp(self.vm_min.unsqueeze(0) - vm_pred, min=0)
        vm_hi = torch.clamp(vm_pred - self.vm_max.unsqueeze(0), min=0)
        nu_2a = (vm_lo + vm_hi).mean(dim=1).mean()  # average over buses then batch

        # ---- ν_2b: Voltage angle difference bounds ----
        ai = va_pred[:, self.f_idx]
        aj = va_pred[:, self.t_idx]
        theta_diff = ai - aj
        theta_max = self.theta_diff_max.unsqueeze(0)
        td_lo = torch.clamp(-theta_diff - theta_max, min=0)
        td_hi = torch.clamp(theta_diff - theta_max, min=0)
        nu_2b = (td_lo + td_hi).mean(dim=1).mean()

        # ---- ν_3a: Pg bounds (only non-slack generators) ----
        pg_min_ns = self.pg_min[self.non_slack_gen_idx].unsqueeze(0)
        pg_max_ns = self.pg_max[self.non_slack_gen_idx].unsqueeze(0)
        pg_lo = torch.clamp(pg_min_ns - pg_pred_ns, min=0)
        pg_hi = torch.clamp(pg_pred_ns - pg_max_ns, min=0)
        nu_3a = (pg_lo + pg_hi).mean(dim=1).mean()

        # ---- ν_3b: Qg bounds (all generators) ----
        qg_lo = torch.clamp(self.qg_min.unsqueeze(0) - qg_pred, min=0)
        qg_hi = torch.clamp(qg_pred - self.qg_max.unsqueeze(0), min=0)
        nu_3b = (qg_lo + qg_hi).mean(dim=1).mean()

        # ---- Branch flows from predictions (for ν4, ν5, ν6) ----
        pf_pred, qf_pred = self.compute_branch_flows(vm_pred, va_pred)

        # ---- ν_4: Thermal limit ----
        sf_sq = pf_pred ** 2 + qf_pred ** 2
        nu_4 = torch.clamp(sf_sq - self.s_max_sq.unsqueeze(0), min=0).mean(dim=1).mean()

        # ---- ν_5a, ν_5b: Ohm's Law (pred flow vs ground truth flow) ----
        pf_true, qf_true = self.compute_branch_flows(vm_true, va_true)
        nu_5a = torch.abs(pf_pred - pf_true).mean(dim=1).mean()
        nu_5b = torch.abs(qf_pred - qf_true).mean(dim=1).mean()

        # ---- ν_6a, ν_6b: Kirchhoff's Current Law ----
        # KCL at bus i: pg_i - pd_i = Σ_{(ij)∈E, from i} pf_ij
        # The paper defines: σ_6a = Σ_{(ij)∈E} p̃f_ij - (p̂g_i - pd_i)
        # This sums flows on branches where bus i is the "from" bus.

        # Also compute reverse flows for to-bus (branch ji perspective)
        # pf_to: flow seen from the to-bus side
        vj_r = vm_pred[:, self.t_idx]
        vi_r = vm_pred[:, self.f_idx]
        aj_r = va_pred[:, self.t_idx]
        ai_r = va_pred[:, self.f_idx]
        theta_ji = aj_r - ai_r
        cos_ji = torch.cos(theta_ji)
        sin_ji = torch.sin(theta_ji)
        g_unsq = self.g_br.unsqueeze(0)
        b_unsq = self.b_br.unsqueeze(0)
        pf_to = g_unsq * vj_r ** 2 - vj_r * vi_r * (b_unsq * sin_ji + g_unsq * cos_ji)
        qf_to = -b_unsq * vj_r ** 2 - vj_r * vi_r * (g_unsq * sin_ji - b_unsq * cos_ji)

        # Assemble bus injection: pg - pd at each bus
        # For KCL, use ground truth pg for slack bus (model doesn't predict it)
        # Only penalize KCL at non-slack buses to avoid coupling with slack recovery
        pg_at_bus = torch.matmul(pg_pred_full, self.bus_gen_map.T)  # (batch, n_buses)
        qg_at_bus = torch.matmul(qg_pred, self.bus_gen_map.T)

        pd_at_bus = torch.zeros(batch, self.n_buses, device=self.device)
        qd_at_bus = torch.zeros(batch, self.n_buses, device=self.device)
        pd_at_bus[:, self.load_bus_indices] = pd_raw
        qd_at_bus[:, self.load_bus_indices] = qd_raw

        # Sum of branch flows: from-bus contributions + to-bus contributions
        pf_sum = torch.zeros(batch, self.n_buses, device=self.device)
        qf_sum = torch.zeros(batch, self.n_buses, device=self.device)
        pf_sum.scatter_add_(1, self.f_idx.unsqueeze(0).expand(batch, -1), pf_pred)
        qf_sum.scatter_add_(1, self.f_idx.unsqueeze(0).expand(batch, -1), qf_pred)
        pf_sum.scatter_add_(1, self.t_idx.unsqueeze(0).expand(batch, -1), pf_to)
        qf_sum.scatter_add_(1, self.t_idx.unsqueeze(0).expand(batch, -1), qf_to)

        # KCL: |Σ pf_ij - (pg_i - pd_i)| averaged over NON-SLACK buses only
        # Slack bus pg is set to 0 (not predicted), so KCL cannot balance there.
        kcl_p = torch.abs(pf_sum - (pg_at_bus - pd_at_bus))  # (batch, n_buses)
        kcl_q = torch.abs(qf_sum - (qg_at_bus - qd_at_bus))

        # Mask out slack buses
        mask = self.non_slack_bus_mask.unsqueeze(0)  # (1, n_buses)
        n_non_slack_buses = self.non_slack_bus_mask.sum().float()
        nu_6a = (kcl_p * mask).sum(dim=1).mean() / n_non_slack_buses
        nu_6b = (kcl_q * mask).sum(dim=1).mean() / n_non_slack_buses

        return {
            'nu_2a': nu_2a, 'nu_2b': nu_2b,
            'nu_3a': nu_3a, 'nu_3b': nu_3b,
            'nu_4': nu_4,
            'nu_5a': nu_5a, 'nu_5b': nu_5b,
            'nu_6a': nu_6a, 'nu_6b': nu_6b,
        }


# =====================================================================
# Y tensor splitting helper
# =====================================================================
def split_y(y_batch, n_buses, n_gen_non_slack, n_gen):
    """Split concatenated Y tensor into (vm, va, pg, qg) components."""
    idx = 0
    vm = y_batch[:, idx:idx + n_buses];        idx += n_buses
    va = y_batch[:, idx:idx + n_buses];         idx += n_buses
    pg = y_batch[:, idx:idx + n_gen_non_slack]; idx += n_gen_non_slack
    qg = y_batch[:, idx:idx + n_gen];           idx += n_gen
    return vm, va, pg, qg


# =====================================================================
# Training Function: Lagrangian Dual (Algorithm 1)
# =====================================================================
def train_lagrangian_dual(model, train_loader, val_loader, scalers, params,
                          n_epochs=80, lr=1e-3, rho=1e-2, device='cpu',
                          lambda_max=100.0):
    """
    Lagrangian Dual training — strict Algorithm 1 replication.

    λ is updated INSIDE the mini-batch loop, immediately after each weight update.
    Uses fixed epochs (paper default: 80) with best-model checkpointing by val MSE.

    lambda_max: upper bound to clip λ values, preventing runaway multipliers.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']

    # Initialize constraint computation module
    opf_constraints = OPFConstraints(params, scalers, device)

    # Algorithm 1, line 1: λ⁰ ← 0  ∀c ∈ C
    violation_names = ['nu_2a', 'nu_2b', 'nu_3a', 'nu_3b', 'nu_4',
                       'nu_5a', 'nu_5b', 'nu_6a', 'nu_6b']
    lambdas = {name: 0.0 for name in violation_names}

    history = {'train_loss': [], 'val_loss': [], 'mse_loss': []}

    print(f"\n{'=' * 80}")
    print(f"Lagrangian Dual Training (M_C^D, Algorithm 1)")
    print(f"{'=' * 80}")
    print(f"  α={lr}, ρ={rho}, epochs={n_epochs}, λ_max={lambda_max}")
    print(f"  9 constraint violations: {violation_names}")
    print(f"\n{'Ep':>4} {'Lo':>9} {'Lc':>9} {'Val':>9} "
          f"{'λ2a':>7} {'λ3a':>7} {'λ4':>7} {'λ5a':>7} {'λ6a':>7} "
          f"{'ν2a':>8} {'ν3a':>8} {'ν6a':>8}")
    print(f"{'-' * 110}")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_lc = 0.0
        n_batches = 0
        epoch_violations = {name: 0.0 for name in violation_names}

        # Algorithm 1, line 3: foreach (x, y) ← minibatch
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()

            # Algorithm 1, line 4: ŷ ← Ô[w](x)
            vm_pred, va_pred, pg_pred, qg_pred = model(X_batch)

            # Split ground truth Y
            vm_true, va_true, pg_true, qg_true = split_y(
                Y_batch, n_buses, n_gen_non_slack, n_gen)

            # Algorithm 1, line 5: Lo(ŷ, y)
            loss_v = criterion(vm_pred, vm_true)
            loss_theta = criterion(va_pred, va_true)
            loss_pg = criterion(pg_pred, pg_true)
            loss_qg = criterion(qg_pred, qg_true)
            lo = loss_v + loss_theta + loss_pg + loss_qg

            # Compute violation degrees (Algorithm 1, line 6)
            violations = opf_constraints.compute_violations(
                vm_pred, va_pred, pg_pred, qg_pred,
                X_batch, vm_true, va_true
            )

            # Algorithm 1, line 6: Lc = Σ λ_c · ν_c
            lc = torch.tensor(0.0, device=device)
            for name in violation_names:
                lc = lc + lambdas[name] * violations[name]

            # Algorithm 1, line 7: w ← w - α∇w(Lo + Lc)
            total_loss = lo + lc
            total_loss.backward()
            optimizer.step()

            # Algorithm 1, lines 8-9: λ^{k+1}_c ← max(0, λ^k_c + ρ·ν_c)
            # With upper bound clipping to prevent runaway multipliers
            with torch.no_grad():
                for name in violation_names:
                    nu_val = violations[name].item()
                    lambdas[name] = min(lambda_max,
                                        max(0.0, lambdas[name] + rho * nu_val))
                    epoch_violations[name] += nu_val

            epoch_loss += total_loss.item()
            epoch_mse += lo.item()
            epoch_lc += lc.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches
        avg_mse = epoch_mse / n_batches
        avg_lc = epoch_lc / n_batches
        avg_viols = {k: v / n_batches for k, v in epoch_violations.items()}
        history['train_loss'].append(avg_train)
        history['mse_loss'].append(avg_mse)

        # Validation loss (pure MSE only — not affected by λ changes)
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                vm_p, va_p, pg_p, qg_p = model(X_val)
                vm_t, va_t, pg_t, qg_t = split_y(Y_val, n_buses, n_gen_non_slack, n_gen)
                loss = criterion(vm_p, vm_t) + criterion(va_p, va_t) + \
                       criterion(pg_p, pg_t) + criterion(qg_p, qg_t)
                val_loss += loss.item() * len(X_val)
                n_val += len(X_val)

        avg_val = val_loss / n_val if n_val > 0 else 0
        history['val_loss'].append(avg_val)

        # Print progress
        if epoch % max(1, n_epochs // 40) == 0 or epoch == n_epochs - 1:
            print(f"{epoch:>4} {avg_mse:>9.6f} {avg_lc:>9.4f} {avg_val:>9.6f} "
                  f"{lambdas['nu_2a']:>7.3f} {lambdas['nu_3a']:>7.3f} "
                  f"{lambdas['nu_4']:>7.3f} {lambdas['nu_5a']:>7.3f} "
                  f"{lambdas['nu_6a']:>7.3f} "
                  f"{avg_viols['nu_2a']:>8.5f} {avg_viols['nu_3a']:>8.5f} "
                  f"{avg_viols['nu_6a']:>8.5f}")

    # Final lambdas
    print(f"\n  Final λ values:")
    for name in violation_names:
        print(f"    {name}: {lambdas[name]:.6f}")

    return history


# =====================================================================
# Evaluation Function
# =====================================================================
def evaluate_on_test_set(model, X_test, test_idx_or_raw, raw_data, scalers, params,
                         device, verbose=True, label="Test"):
    """
    Evaluate model on a test set using power flow.

    Returns metrics dict from acopf_violation_metrics.evaluate_acopf_predictions.
    """
    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_loads = params['general']['n_loads']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']

    model.eval()
    with torch.no_grad():
        vm_pred_s, va_pred_s, pg_pred_s, qg_pred_s = model(X_test.to(device))
        vm_pred_s = vm_pred_s.cpu().numpy()
        va_pred_s = va_pred_s.cpu().numpy()
        pg_pred_s = pg_pred_s.cpu().numpy()
        qg_pred_s = qg_pred_s.cpu().numpy()

    # Inverse transform
    y_pred_vm = scalers['vm'].inverse_transform(vm_pred_s)
    y_pred_va = scalers['va'].inverse_transform(va_pred_s)
    y_pred_pg_ns = scalers['pg'].inverse_transform(pg_pred_s)
    y_pred_qg = scalers['qg'].inverse_transform(qg_pred_s)

    # Reconstruct full Pg
    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_ns, params)

    # Extract Vm at generator buses for power flow
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_gen = y_pred_vm[:, gen_bus_indices]

    # Loads from input
    x_raw = scalers['x'].inverse_transform(X_test.cpu().numpy())
    pd_pu = x_raw[:, :n_loads]
    qd_pu = x_raw[:, n_loads:]

    # True values
    y_true_pg = raw_data['pg']
    y_true_vm = raw_data['vm']
    y_true_qg = raw_data['qg']
    y_true_va_rad = raw_data['va']

    # Run power flow
    n_samples = len(X_test)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"\n  [{label}] Running power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_for_evaluation(
                pd_pu[i], qd_pu[i], y_pred_pg_ns[i], y_pred_vm_gen[i], params)
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except Exception:
            pf_results_list.append(({'success': False, 'gen': np.zeros((n_gen, 21)),
                                     'bus': np.zeros((n_buses, 13)),
                                     'branch': np.zeros((1, 17))},))
            converge_flags.append(False)

    if verbose:
        print(f"    Converged: {sum(converge_flags)}/{n_samples}")

    # ---- PF-based metrics (Table 4-5 style) ----
    pf_metrics = evaluate_acopf_predictions(
        y_pred_pg=y_pred_pg_full,
        y_pred_vm=y_pred_vm,
        y_true_pg=y_true_pg,
        y_true_vm=y_true_vm,
        y_true_qg=y_true_qg,
        y_true_va_rad=y_true_va_rad,
        pf_results_list=pf_results_list,
        converge_flags=converge_flags,
        params=params,
        verbose=False,
    )

    # ---- DNN-direct metrics (Table 3 style) ----
    # Prediction errors: directly compare DNN outputs vs ground truth
    from acopf_violation_metrics import compute_mae_percentage, compute_mae_absolute

    # MAE for each predicted quantity
    dnn_mae_pg = compute_mae_percentage(y_true_pg[:, ~params['general']['slack_gen_mask']],
                                         y_pred_pg_ns)
    dnn_mae_vm = compute_mae_percentage(y_true_vm[:, gen_bus_indices],
                                         y_pred_vm[:, gen_bus_indices])
    dnn_mae_qg = compute_mae_percentage(y_true_qg, y_pred_qg)

    y_true_va_deg = y_true_va_rad * (180.0 / np.pi)
    y_pred_va_deg = y_pred_va * (180.0 / np.pi)
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
    pg_viol = np.maximum(0, pg_min[ns_idx] - y_pred_pg_ns) + \
              np.maximum(0, y_pred_pg_ns - pg_max[ns_idx])
    dnn_pg_viol = float(np.mean(np.max(pg_viol, axis=1)))

    # Pg violation (slack) — slack Pg comes from PF, same as After PF
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

    # DNN-direct cost gap — uses DNN non-slack Pg + PF slack Pg (same as After PF)
    # Since DNN does not predict slack Pg, cost is identical to After PF.
    from acopf_violation_metrics import compute_cost_metrics
    dnn_cost_gap = pf_metrics.get('cost_optimality_gap_percent', float('nan'))

    # Qg violation (all generators)
    qg_viol = np.maximum(0, qg_min - y_pred_qg) + np.maximum(0, y_pred_qg - qg_max)
    dnn_qg_viol = float(np.mean(np.max(qg_viol, axis=1)))

    # Vm violation (all buses)
    vm_viol = np.maximum(0, vm_min - y_pred_vm) + np.maximum(0, y_pred_vm - vm_max)
    dnn_vm_viol = float(np.mean(np.max(vm_viol, axis=1)))

    # Branch violation from predicted (vm, va) via Ohm's Law
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

    vi = y_pred_vm[:, f_idx]
    vj = y_pred_vm[:, t_idx]
    ai = y_pred_va[:, f_idx]
    aj = y_pred_va[:, t_idx]
    theta_ij = ai - aj
    pf = g_br * vi**2 - vi * vj * (b_br * np.sin(theta_ij) + g_br * np.cos(theta_ij))
    qf = -b_br * vi**2 - vi * vj * (g_br * np.sin(theta_ij) - b_br * np.cos(theta_ij))
    sf_sq = pf**2 + qf**2
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


def print_metrics(metrics, label="Test"):
    """Print evaluation metrics: DNN-direct (Table 3 style) + PF-based (Table 4-5 style)."""
    print(f"\n  ┌─── {label} ───")
    print(f"  │")
    print(f"  │ [DNN Prediction] (直接从预测值评估, 对标原文Table 3)")
    print(f"  │   MAE: Pg={metrics['dnn_mae_pg_percent']:.4f}%  "
          f"Vm={metrics['dnn_mae_vm_percent']:.4f}%  "
          f"Qg={metrics['dnn_mae_qg_percent']:.4f}%  "
          f"Va={metrics['dnn_mae_va_deg']:.4f}°")
    print(f"  │   Viol: Pg(ns)={metrics['dnn_pg_viol_pu']:.6f}  "
          f"Pg(slack)={metrics['dnn_pg_slack_viol_pu']:.6f}  "
          f"Qg={metrics['dnn_qg_viol_pu']:.6f}  "
          f"Vm={metrics['dnn_vm_viol_pu']:.6f}  "
          f"Branch={metrics['dnn_branch_viol_pu']:.6f}")
    print(f"  │   Cost gap: {metrics['dnn_cost_gap_percent']:.4f}%")
    print(f"  │")
    print(f"  │ [After Load Flow] (PF恢复后评估, 对标原文Table 4-5)")
    print(f"  │   Convergence: {metrics['convergence_rate_percent']:.1f}% "
          f"({metrics['n_converged']}/{metrics['n_samples']})")
    print(f"  │   MAE: Pg(ns)={metrics['mae_pg_non_slack_percent']:.4f}%  "
          f"Vm={metrics['mae_vm_percent']:.4f}%  "
          f"Qg={metrics['mae_qg_percent']:.4f}%  "
          f"Va={metrics['mae_va_deg']:.4f}°")
    print(f"  │   Viol: Pg(ns)={metrics['mean_pg_viol_non_slack_pu']:.6f}  "
          f"Pg(slack)={metrics['mean_pg_viol_slack_pu']:.6f}  "
          f"Qg={metrics['mean_max_qg_viol_pu']:.6f}  "
          f"Vm={metrics['mean_max_vm_viol_pu']:.6f}  "
          f"Branch={metrics['mean_max_branch_viol_pu']:.6f}")
    print(f"  │   Cost gap: {metrics['cost_optimality_gap_percent']:.4f}%")
    print(f"  └{'─' * 60}")


# =====================================================================
# Main Experiment Function
# =====================================================================
def lagrangian_acopf_experiment(
        case_name, params_path, data_path, log_path, results_path,
        # Data mode
        data_mode='random_split',
        n_train_use=None,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=None,
        # Training parameters
        n_epochs=80,
        learning_rate=0.001,
        lagrangian_lr=0.01,
        lambda_max=100.0,
        batch_size=64,
        seed=42,
        device='cuda',
        # Multi-test evaluation
        eval_generalization=False,
        gen_test_data_path=None,
        gen_n_test_samples=None,
        eval_api=False,
        api_test_data_path=None,
        api_test_params_path=None,
        api_n_test_samples=None,
):
    """
    M_C^D Experiment: Train once, evaluate on up to 3 test sets.
    """
    global GLOBAL_CASE_DATA

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 80}")
    print(f"Lagrangian Dual ACOPF — M_C^D Replication")
    print(f"{'=' * 80}")
    print(f"Case: {case_name}  |  Mode: {data_mode}  |  Device: {device}")

    # ================================================================
    # 1. Load parameters and case data
    # ================================================================
    init_pypower_options()
    params = load_parameters_from_csv(case_name, params_path)
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)

    # ================================================================
    # 2. Load and scale data
    # ================================================================
    x_scaled, y_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params)

    # ================================================================
    # 3. Data splits
    # ================================================================
    train_idx, val_idx, test_idx = prepare_data_splits(
        x_scaled, y_scaled, mode=data_mode, n_train_use=n_train_use, seed=seed)

    X_train = torch.tensor(x_scaled[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(y_scaled[train_idx], dtype=torch.float32)
    X_val = torch.tensor(x_scaled[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(y_scaled[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val),
                            batch_size=batch_size, shuffle=False)

    # ================================================================
    # 4. Build model (paper-parameterized architecture)
    # ================================================================
    n_loads = params['general']['n_loads']
    n_buses = params['general']['n_buses']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_gen = params['general']['n_gen']

    model = OPF_DNN_MC(n_loads, n_buses, n_gen_non_slack, n_gen).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: OPF_DNN_MC  |  Parameters: {total_params:,}")
    print(f"  Architecture: l={n_loads}, n={n_buses}, g={n_gen_non_slack}, g_total={n_gen}")
    print(f"  Input: 2l={2 * n_loads}  →  Output: v({n_buses})+θ({n_buses})"
          f"+pg({n_gen_non_slack})+qg({n_gen})")

    # ================================================================
    # 5. Train
    # ================================================================
    t_start = time.time()
    history = train_lagrangian_dual(
        model, train_loader, val_loader, scalers, params,
        n_epochs=n_epochs, lr=learning_rate, rho=lagrangian_lr,
        device=device,
        lambda_max=lambda_max,
    )
    train_time = time.time() - t_start

    # ================================================================
    # 6. Inference timing (consistent with DNN baseline)
    # ================================================================
    X_test_tensor = torch.tensor(x_scaled[test_idx], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        # Warmup: 10 forward passes to stabilize timing
        for _ in range(10):
            model(X_test_tensor[:1].to(device))

    # Timed: 100 single-sample forward passes
    times = [time.perf_counter() for _ in range(101)]
    with torch.no_grad():
        for i in range(100):
            model(X_test_tensor[:1].to(device))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times[i + 1] = time.perf_counter()

    latency_ms = np.mean(np.diff(times)) * 1000

    # ================================================================
    # 7. Evaluate on primary test set
    # ================================================================
    # Subset raw_data for primary test
    raw_test = {k: v[test_idx] for k, v in raw_data.items()}

    primary_metrics = evaluate_on_test_set(
        model, X_test_tensor, test_idx, raw_test, scalers, params,
        device, verbose=True, label=f"Primary ({data_mode})")

    # ================================================================
    # 8. Evaluate on generalization test set (if requested)
    # ================================================================
    gen_metrics = None
    if eval_generalization and gen_test_data_path:
        print(f"\n--- Generalization Test ---")
        gen_x, gen_y, gen_raw, gen_cost = load_generalization_test_data(
            gen_test_data_path, params, scalers,
            n_test_samples=gen_n_test_samples, seed=seed)

        X_gen = torch.tensor(gen_x, dtype=torch.float32)
        gen_metrics = evaluate_on_test_set(
            model, X_gen, None, gen_raw, scalers, params,
            device, verbose=True, label="Generalization")

    # ================================================================
    # 9. Evaluate on API test set (if requested)
    # ================================================================
    api_metrics = None
    if eval_api and api_test_data_path and api_test_params_path:
        print(f"\n--- API Test ---")
        api_params, api_x, api_y, api_raw, api_cost = load_api_test_data(
            api_test_data_path, api_test_params_path, scalers,
            n_test_samples=api_n_test_samples, seed=seed)

        # Need to reload case data for API topology
        api_case_name = os.path.basename(api_test_data_path)
        if api_case_name.endswith('_pd.csv'):
            api_case_name = api_case_name[:-7]
        GLOBAL_CASE_DATA = load_case_from_csv(api_case_name, api_test_params_path)

        X_api = torch.tensor(api_x, dtype=torch.float32)
        api_metrics = evaluate_on_test_set(
            model, X_api, None, api_raw, scalers, api_params,
            device, verbose=True, label="API")

        # Restore original case data
        GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)

    # ================================================================
    # 10. Print summary
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Training time: {train_time:.1f}s  |  Inference: {latency_ms:.4f} ms/sample")

    print_metrics(primary_metrics, f"Primary ({data_mode})")
    if gen_metrics is not None:
        print_metrics(gen_metrics, "Generalization")
    if api_metrics is not None:
        print_metrics(api_metrics, "API")

    print(f"{'=' * 80}")

    return {
        'primary': primary_metrics,
        'generalization': gen_metrics,
        'api': api_metrics,
        'train_time': train_time,
        'latency_ms': latency_ms,
    }


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    # ── Read config ──
    paths = acopf_config.get_all_paths()
    train_params = acopf_config.get_all_params()

    # Lagrangian-specific (set here or add to config)
    LAGRANGIAN_LR = getattr(acopf_config, 'LAGRANGIAN_LR', 0.01)
    LAMBDA_MAX = getattr(acopf_config, 'LAMBDA_MAX', 100.0)

    # Multi-test configuration
    EVAL_GEN = getattr(acopf_config, 'EVAL_GENERALIZATION', False)
    EVAL_API = getattr(acopf_config, 'EVAL_API', False)

    gen_test_path = None
    api_test_path = None
    api_params_path = None

    if EVAL_GEN:
        gen_test_path = acopf_config.get_data_path(
            acopf_config.TRAIN_CASE, acopf_config.TEST_VARIANCE)

    if EVAL_API:
        api_test_path = acopf_config.get_data_path(acopf_config.TEST_CASE, None)
        api_params_path = acopf_config.get_params_path(acopf_config.TEST_CASE)

    # Print config
    print("\n" + "=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"  Case: {paths['case_name']}")
    print(f"  Mode: {train_params['data_mode']}")
    print(f"  Epochs: {train_params['n_epochs']}")
    print(f"  LR: {train_params['learning_rate']}  |  Lagrangian ρ: {LAGRANGIAN_LR}  |  λ_max: {LAMBDA_MAX}")
    print(f"  Batch: {train_params['batch_size']}  |  Seed: {train_params['seed']}")
    print(f"  Eval generalization: {EVAL_GEN}  |  Eval API: {EVAL_API}")
    print("=" * 80)

    # Run experiment
    results = lagrangian_acopf_experiment(
        case_name=paths['case_name'],
        params_path=paths['params_path'],
        data_path=paths['data_path'],
        log_path=paths.get('log_path', ''),
        results_path=paths.get('results_path', ''),
        data_mode=train_params['data_mode'],
        n_train_use=train_params['n_train_use'],
        test_data_path=paths.get('test_data_path'),
        test_params_path=paths.get('test_params_path'),
        n_test_samples=train_params['n_test_samples'],
        n_epochs=train_params['n_epochs'],
        learning_rate=train_params['learning_rate'],
        lagrangian_lr=LAGRANGIAN_LR,
        lambda_max=LAMBDA_MAX,
        batch_size=train_params['batch_size'],
        seed=train_params['seed'],
        device=train_params['device'],
        # Multi-test
        eval_generalization=EVAL_GEN,
        gen_test_data_path=gen_test_path,
        gen_n_test_samples=train_params['n_test_samples'],
        eval_api=EVAL_API,
        api_test_data_path=api_test_path,
        api_test_params_path=api_params_path,
        api_n_test_samples=train_params['n_test_samples'],
    )

    print("\n✓ Experiment completed successfully!")