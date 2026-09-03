# -*- coding: utf-8 -*-
"""
ACOPF RL Main Experiment File (PPO)

Architecture:
- Environment: Custom Gymnasium env backed by PyPower runpf
- Agent: PPO (stable-baselines3)
- Reward: cost objective + constraint penalty (Summation, with normalization scaling)
- Observation: scaled [pd, qd]
- Action: normalized [0,1] → denormalized pg_non_slack + vm_gen
- Evaluation: identical to acopf_dnn_main.py (converged-only metrics)

Usage:
  Modify acopf_config.py as usual, then run this file.
"""

import os
import sys
import time
import copy

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pypower.runpf import runpf
from pypower.ppoption import ppoption
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import acopf_config
from acopf_data_setup import (
    load_parameters_from_csv,
    load_and_scale_acopf_data,
    DataMode,
    prepare_data_splits,
    load_generalization_test_data,
    load_api_test_data,
    reconstruct_full_pg,
)
from acopf_violation_metrics import evaluate_acopf_predictions

# reward.py lives alongside this script (or on PYTHONPATH)
# We only need the Summation class; estimate_reward_distribution is NOT used
# because it requires OpfEnv-style methods that we don't have here.
try:
    from reward import Summation
except ImportError:
    print("Error: cannot import reward.py. Make sure it is on PYTHONPATH.")
    sys.exit(1)

# =====================================================================
# PyPower helpers  (mirrors load_case_from_csv in acopf_dnn_main.py)
# =====================================================================
from pathlib import Path
import pandas as pd


def load_case_from_csv(case_name: str, constraints_path: str) -> dict:
    """Build a PyPower ppc dict from the CSV constraint files."""
    base_path = Path(constraints_path)

    base_mva_df = pd.read_csv(base_path / f"{case_name}_base_mva.csv")
    bus_df      = pd.read_csv(base_path / f"{case_name}_bus_data.csv")
    gen_df      = pd.read_csv(base_path / f"{case_name}_gen_data.csv")
    branch_df   = pd.read_csv(base_path / f"{case_name}_branch_data.csv")

    baseMVA = base_mva_df['value'].iloc[0]

    # ---- BUS ----
    bus = np.zeros((len(bus_df), 13))
    bus[:, 0]  = bus_df['bus_id'].values
    bus[:, 1]  = bus_df['type'].values
    bus[:, 2]  = bus_df['pd_pu'].values * baseMVA   # MW
    bus[:, 3]  = bus_df['qd_pu'].values * baseMVA   # MVAr
    bus[:, 6]  = 1
    bus[:, 7]  = bus_df['vm_pu'].values
    bus[:, 8]  = bus_df['va_deg'].values
    bus[:, 9]  = bus_df['base_kv'].values
    bus[:, 10] = 1
    bus[:, 11] = bus_df['vmax_pu'].values
    bus[:, 12] = bus_df['vmin_pu'].values

    # ---- GEN ----
    gen = np.zeros((len(gen_df), 21))
    gen[:, 0] = gen_df['bus_id'].values
    gen[:, 3] = gen_df['qg_max_pu'].values * baseMVA
    gen[:, 4] = gen_df['qg_min_pu'].values * baseMVA
    gen[:, 5] = gen_df['vg_pu'].values
    gen[:, 6] = baseMVA
    gen[:, 7] = 1
    gen[:, 8] = gen_df['pg_max_pu'].values * baseMVA
    gen[:, 9] = gen_df['pg_min_pu'].values * baseMVA

    # ---- BRANCH ----
    branch = np.zeros((len(branch_df), 13))
    branch[:, 0]  = branch_df['f_bus'].values
    branch[:, 1]  = branch_df['t_bus'].values
    branch[:, 2]  = branch_df['r_pu'].values
    branch[:, 3]  = branch_df['x_pu'].values
    branch[:, 4]  = branch_df['b_pu'].values
    rate_a        = branch_df['rate_a_pu'].values
    branch[:, 5]  = rate_a * baseMVA
    branch[:, 6]  = branch[:, 5]
    branch[:, 7]  = branch[:, 5]
    branch[:, 8]  = branch_df['tap_ratio'].values
    branch[:, 9]  = branch_df['shift_deg'].values
    branch[:, 10] = 1
    branch[:, 11] = -360
    branch[:, 12] = 360
    # Unlimited / NaN branches → sentinel 9900 MVA
    bad = np.isnan(rate_a) | np.isinf(rate_a) | (rate_a == 0)
    branch[bad, 5:8] = 9900.0

    # ---- GENCOST (cost coefficients stored in p.u.) ----
    gencost = np.zeros((len(gen_df), 7))
    gencost[:, 0] = 2   # polynomial model
    gencost[:, 3] = 3   # 3 coefficients
    # p.u. → MW unit conversion for cost coefficients
    gencost[:, 4] = gen_df['cost_c2'].values / (baseMVA ** 2)
    gencost[:, 5] = gen_df['cost_c1'].values / baseMVA
    gencost[:, 6] = gen_df['cost_c0'].values

    ppc = {
        'version': '2',
        'baseMVA': baseMVA,
        'bus': bus,
        'gen': gen,
        'branch': branch,
        'gencost': gencost,
    }
    return ppc


def make_ppopt():
    ppopt = ppoption()
    return ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)


# =====================================================================
# Reward helpers (objective + penalty computed from pf results)
# =====================================================================

def compute_cost_from_pf(r1_pf, base_mva, cost_c2, cost_c1, cost_c0):
    """
    Compute generation cost ($/h) from a converged PyPower pf result.
    cost coefficients are in p.u. units (as stored in CSV).
    pg from pf is in MW → convert to p.u. first.
    """
    pg_mw  = r1_pf[0]['gen'][:, 1]          # MW
    pg_pu  = pg_mw / base_mva               # p.u.
    cost   = np.sum(cost_c2 * pg_pu**2 + cost_c1 * pg_pu + cost_c0)
    return float(cost)


def compute_penalty_from_pf(r1_pf, base_mva):
    """
    Compute a scalar constraint-violation penalty from a converged pf result.
    Mirrors the logic in acopf_violation_metrics.calculate_single_sample_violations,
    but returns a single summed (negative) penalty value for the reward function.

    All violations are normalised to p.u. scale and summed.
    Returns a value <= 0  (0 when fully feasible).
    """
    gen    = r1_pf[0]['gen']
    bus    = r1_pf[0]['bus']
    branch = r1_pf[0]['branch']

    # --- Pg violation ---
    pg_mw     = gen[:, 1]
    pg_viol   = (np.maximum(0, gen[:, 9] - pg_mw) +
                 np.maximum(0, pg_mw - gen[:, 8]))
    pg_pen    = np.sum(pg_viol) / base_mva

    # --- Qg violation ---
    qg_mvar   = gen[:, 2]
    qg_viol   = (np.maximum(0, gen[:, 4] - qg_mvar) +
                 np.maximum(0, qg_mvar - gen[:, 3]))
    qg_pen    = np.sum(qg_viol) / base_mva

    # --- Vm violation (already p.u.) ---
    vm_pu     = bus[:, 7]
    vm_viol   = (np.maximum(0, bus[:, 12] - vm_pu) +
                 np.maximum(0, vm_pu - bus[:, 11]))
    vm_pen    = np.sum(vm_viol)

    # --- Branch loading violation ---
    rate_a    = branch[:, 5]
    lim_idx   = (rate_a > 0) & (rate_a < 9000)
    br_pen    = 0.0
    if np.any(lim_idx):
        Ff = np.abs(branch[lim_idx, 13] + 1j * branch[lim_idx, 14])
        Ft = np.abs(branch[lim_idx, 15] + 1j * branch[lim_idx, 16])
        ra = rate_a[lim_idx]
        br_pen = float(np.sum(np.maximum(0, Ff / ra - 1) +
                               np.maximum(0, Ft / ra - 1)))

    total_violation = pg_pen + qg_pen + vm_pen + br_pen
    return -total_violation   # negative → penalty


# =====================================================================
# Gymnasium Environment
# =====================================================================

class AcopfEnv(gym.Env):
    """
    Single-step ACOPF environment.

    Observation : scaled [pd, qd]  (MinMaxScaler from training data)
    Action      : [0, 1]^(n_gen_non_slack + n_gen)
                  first  n_gen_non_slack dims → pg_non_slack (p.u.)
                  last   n_gen             dims → vm_gen      (p.u.)
    Reward      : Summation(objective, penalty)  with normalization scaling
    Episode     : 1 step, always terminated=True
    """

    metadata = {}

    def __init__(self,
                 x_scaled: np.ndarray,
                 x_raw: np.ndarray,
                 indices: np.ndarray,
                 params: dict,
                 scalers: dict,
                 ppc_template: dict,
                 ppopt,
                 reward_fn,
                 non_converge_reward: float = -10.0,
                 seed: int = 42):

        super().__init__()

        self.x_scaled   = x_scaled[indices]   # (N, 2*n_loads)
        self.x_raw      = x_raw[indices]       # (N, 2*n_loads)  raw p.u.
        self.n_samples  = len(indices)

        self.params     = params
        self.scalers    = scalers
        self.ppc_tmpl   = ppc_template
        self.ppopt      = ppopt
        self.reward_fn  = reward_fn
        self.non_converge_reward = non_converge_reward

        # Dimensions
        self.n_loads          = params['general']['n_loads']
        self.n_gen            = params['general']['n_gen']
        self.n_gen_non_slack  = params['general']['n_gen_non_slack']
        self.non_slack_idx    = params['general']['non_slack_gen_idx']
        self.gen_bus_ids      = params['general']['gen_bus_ids']
        self.bus_id_to_idx    = params['general']['bus_id_to_idx']
        self.base_mva         = params['general']['BASE_MVA']

        # Cost coefficients (p.u.)
        self.cost_c2 = params['generator']['cost_c2']
        self.cost_c1 = params['generator']['cost_c1']
        self.cost_c0 = params['generator']['cost_c0']

        # Action bounds from scaler (pg_non_slack + vm_gen ranges)
        # pg scaler: shape (n_gen_non_slack,)
        self.pg_min = scalers['pg'].data_min_   # p.u.
        self.pg_max = scalers['pg'].data_max_
        self.vm_min = scalers['vm'].data_min_   # p.u.
        self.vm_max = scalers['vm'].data_max_

        obs_dim = 2 * self.n_loads
        act_dim = self.n_gen_non_slack + self.n_gen

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._current_idx = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_idx = int(self._rng.integers(0, self.n_samples))
        obs = self.x_scaled[self._current_idx].astype(np.float32)
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(action, 0.0, 1.0)

        # --- Denormalise action ---
        pg_ns_norm = action[:self.n_gen_non_slack]
        vm_norm    = action[self.n_gen_non_slack:]

        pg_non_slack = pg_ns_norm * (self.pg_max - self.pg_min) + self.pg_min
        vm_gen       = vm_norm    * (self.vm_max - self.vm_min) + self.vm_min

        # --- Build ppc for this sample ---
        pd_pu = self.x_raw[self._current_idx, :self.n_loads]
        qd_pu = self.x_raw[self._current_idx, self.n_loads:]

        mpc = self._build_mpc(pd_pu, qd_pu, pg_non_slack, vm_gen)

        # --- Run power flow ---
        try:
            r1_pf = runpf(mpc, self.ppopt)
            converged = bool(r1_pf[0]['success'])
        except Exception:
            converged = False

        # --- Compute reward ---
        if not converged:
            reward = float(self.non_converge_reward)
        else:
            objective = -compute_cost_from_pf(
                r1_pf, self.base_mva,
                self.cost_c2, self.cost_c1, self.cost_c0)
            penalty = compute_penalty_from_pf(r1_pf, self.base_mva)
            valid   = (penalty == 0.0)
            reward  = float(self.reward_fn(objective, penalty, valid))

        obs = self.x_scaled[self._current_idx].astype(np.float32)
        return obs, reward, True, False, {}   # terminated every step

    # ------------------------------------------------------------------
    def _build_mpc(self, pd_pu, qd_pu, pg_non_slack, vm_gen):
        """Return a deep-copied ppc with loads, pg, vm set for this sample."""
        mpc = {
            'version':  self.ppc_tmpl['version'],
            'baseMVA':  self.ppc_tmpl['baseMVA'],
            'bus':      self.ppc_tmpl['bus'].copy(),
            'gen':      self.ppc_tmpl['gen'].copy(),
            'branch':   self.ppc_tmpl['branch'],
            'gencost':  self.ppc_tmpl['gencost'],
        }
        base_mva = self.base_mva
        bus_id_to_idx = self.bus_id_to_idx
        load_bus_ids  = self.params['general']['load_bus_ids']

        # Set loads
        for i, bid in enumerate(load_bus_ids):
            bidx = bus_id_to_idx.get(int(bid))
            if bidx is not None:
                mpc['bus'][bidx, 2] = pd_pu[i] * base_mva
                mpc['bus'][bidx, 3] = qd_pu[i] * base_mva

        # Set non-slack generator Pg (MW)
        for i, gen_idx in enumerate(self.non_slack_idx):
            mpc['gen'][gen_idx, 1] = pg_non_slack[i] * base_mva

        # Set generator voltage setpoints
        for i in range(self.n_gen):
            mpc['gen'][i, 5] = vm_gen[i]

        return mpc


# =====================================================================
# Reward-scaling bootstrap
# =====================================================================

def estimate_scaling_params(
        x_raw: np.ndarray,
        train_idx: np.ndarray,
        params: dict,
        scalers: dict,
        ppc_template: dict,
        ppopt,
        num_samples: int = 500,
        seed: int = 42,
) -> dict:
    """
    Estimate the mean and std of objective and penalty under random actions
    so that Summation(reward_scaling='normalization') can scale both terms
    to a similar magnitude.

    Fully self-contained: no dependency on OpfEnv or reward.py helpers.
    Non-converged samples are excluded (NaN-masked) exactly as in evaluation.
    """
    rng          = np.random.default_rng(seed)
    n_loads      = params['general']['n_loads']
    n_gen        = params['general']['n_gen']
    n_gen_ns     = params['general']['n_gen_non_slack']
    non_slack    = params['general']['non_slack_gen_idx']
    bus_id_to_idx= params['general']['bus_id_to_idx']
    load_bus_ids = params['general']['load_bus_ids']
    base_mva     = params['general']['BASE_MVA']
    cost_c2      = params['generator']['cost_c2']
    cost_c1      = params['generator']['cost_c1']
    cost_c0      = params['generator']['cost_c0']

    pg_min = scalers['pg'].data_min_
    pg_max = scalers['pg'].data_max_
    vm_min = scalers['vm'].data_min_
    vm_max = scalers['vm'].data_max_

    sample_pool = rng.choice(train_idx, size=num_samples, replace=True)
    objectives  = []
    penalties   = []

    for sample_idx in sample_pool:
        # Random action in [0, 1]
        action       = rng.random(n_gen_ns + n_gen).astype(np.float32)
        pg_non_slack = action[:n_gen_ns] * (pg_max - pg_min) + pg_min
        vm_gen       = action[n_gen_ns:] * (vm_max - vm_min) + vm_min

        pd_pu = x_raw[sample_idx, :n_loads]
        qd_pu = x_raw[sample_idx, n_loads:]

        mpc = {
            'version': ppc_template['version'],
            'baseMVA': ppc_template['baseMVA'],
            'bus':     ppc_template['bus'].copy(),
            'gen':     ppc_template['gen'].copy(),
            'branch':  ppc_template['branch'],
            'gencost': ppc_template['gencost'],
        }
        for j, bid in enumerate(load_bus_ids):
            bidx = bus_id_to_idx.get(int(bid))
            if bidx is not None:
                mpc['bus'][bidx, 2] = pd_pu[j] * base_mva
                mpc['bus'][bidx, 3] = qd_pu[j] * base_mva
        for j, gen_idx in enumerate(non_slack):
            mpc['gen'][gen_idx, 1] = pg_non_slack[j] * base_mva
        for j in range(n_gen):
            mpc['gen'][j, 5] = vm_gen[j]

        try:
            r1_pf     = runpf(mpc, ppopt)
            converged = bool(r1_pf[0]['success'])
        except Exception:
            converged = False

        if not converged:
            continue

        cost = compute_cost_from_pf(r1_pf, base_mva, cost_c2, cost_c1, cost_c0)
        pen  = compute_penalty_from_pf(r1_pf, base_mva)
        objectives.append(-cost)   # objective is negated cost
        penalties.append(pen)

    objectives = np.array(objectives)
    penalties  = np.array(penalties)

    # Guard against edge cases
    std_obj = float(np.std(objectives)) if len(objectives) > 1 else 1.0
    std_pen = float(np.std(penalties))  if len(penalties)  > 1 else 1.0
    if std_obj == 0:
        std_obj = 1.0
    if std_pen == 0:
        std_pen = 1.0

    return {
        'mean_objective': float(np.mean(objectives)) if len(objectives) else 0.0,
        'std_objective':  std_obj,
        'mean_penalty':   float(np.mean(penalties))  if len(penalties)  else 0.0,
        'std_penalty':    std_pen,
        # extras required by Summation base class (unused for 'normalization')
        'min_objective':  float(np.min(objectives))  if len(objectives) else -1.0,
        'max_objective':  float(np.max(objectives))  if len(objectives) else  0.0,
        'min_penalty':    float(np.min(penalties))   if len(penalties)  else -1.0,
        'max_penalty':    float(np.max(penalties))   if len(penalties)  else  0.0,
    }


# =====================================================================
# Evaluation (identical pipeline to acopf_dnn_main.py)
# =====================================================================

def evaluate_rl_agent(
        model,
        x_scaled: np.ndarray,
        x_raw: np.ndarray,
        indices: np.ndarray,
        raw_data: dict,
        params: dict,
        scalers: dict,
        ppc_template: dict,
        ppopt,
        split_name: str = "Test",
        verbose: bool = True,
):
    n_samples     = len(indices)
    n_gen         = params['general']['n_gen']
    n_buses       = params['general']['n_buses']
    n_loads       = params['general']['n_loads']
    base_mva      = params['general']['BASE_MVA']
    non_slack_idx = params['general']['non_slack_gen_idx']
    gen_bus_ids   = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    pg_min = scalers['pg'].data_min_
    pg_max = scalers['pg'].data_max_
    vm_min = scalers['vm'].data_min_
    vm_max = scalers['vm'].data_max_

    if verbose:
        print(f"\n{split_name} Evaluation:")
        print(f"  Computing power flow for {n_samples} samples...")

    pf_results_list = []
    converge_flags  = []

    # Arrays to collect predictions for MAE calculation
    y_pred_pg_full  = np.zeros((n_samples, n_gen))
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_all   = np.ones((n_samples, n_buses))   # default 1.0 p.u.

    for i, sample_idx in enumerate(indices):
        obs = x_scaled[sample_idx].astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, 0.0, 1.0)

        # Denormalise
        pg_ns_norm   = action[:params['general']['n_gen_non_slack']]
        vm_norm      = action[params['general']['n_gen_non_slack']:]
        pg_non_slack = pg_ns_norm * (pg_max - pg_min) + pg_min
        vm_gen       = vm_norm    * (vm_max - vm_min) + vm_min

        # Store predicted vm for generator buses
        y_pred_vm_all[i, gen_bus_indices] = vm_gen

        # Build mpc
        pd_pu = x_raw[sample_idx, :n_loads]
        qd_pu = x_raw[sample_idx, n_loads:]

        mpc = {
            'version': ppc_template['version'],
            'baseMVA': ppc_template['baseMVA'],
            'bus':     ppc_template['bus'].copy(),
            'gen':     ppc_template['gen'].copy(),
            'branch':  ppc_template['branch'],
            'gencost': ppc_template['gencost'],
        }
        for j, bid in enumerate(params['general']['load_bus_ids']):
            bidx = bus_id_to_idx.get(int(bid))
            if bidx is not None:
                mpc['bus'][bidx, 2] = pd_pu[j] * base_mva
                mpc['bus'][bidx, 3] = qd_pu[j] * base_mva
        for j, gen_idx in enumerate(non_slack_idx):
            mpc['gen'][gen_idx, 1] = pg_non_slack[j] * base_mva
        for j in range(n_gen):
            mpc['gen'][j, 5] = vm_gen[j]

        # Run pf
        try:
            r1_pf     = runpf(mpc, ppopt)
            converged = bool(r1_pf[0]['success'])
        except Exception:
            r1_pf     = ({'success': False,
                          'gen':    np.zeros((n_gen, 21)),
                          'bus':    np.zeros((n_buses, 13)),
                          'branch': np.zeros((1, 17))},)
            converged = False

        pf_results_list.append(r1_pf)
        converge_flags.append(converged)

        if converged:
            pg_mw = r1_pf[0]['gen'][:, 1]
            y_pred_pg_full[i] = pg_mw / base_mva

    if verbose:
        print(f"    ✓ Converged: {sum(converge_flags)}/{n_samples}")

    # True values
    y_true_pg     = raw_data['pg'][indices]
    y_true_vm     = raw_data['vm'][indices]
    y_true_qg     = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

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
        verbose=verbose,
    )


# =====================================================================
# Main experiment
# =====================================================================

def acopf_rl_experiment(
        case_name,
        params_path,
        data_path,
        log_path,           # unused, kept for API compatibility
        results_path,       # unused
        data_mode='random_split',
        n_train_use=None,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=None,
        seed=42,
        n_epochs=100,        # maps to PPO total_timesteps via steps_per_epoch
        learning_rate=3e-4,
        hidden_sizes=None,
        batch_size=256,
        device='cuda',
        **kwargs,
):
    hidden_sizes = hidden_sizes or [256, 256]

    print(f"\n{'=' * 70}")
    print(f"ACOPF RL Experiment (PPO)")
    print(f"{'=' * 70}")
    print(f"Case      : {case_name}")
    print(f"Data Mode : {data_mode}")
    print(f"Device    : {device}")
    print(f"{'=' * 70}")

    ppopt = make_ppopt()

    # ── 1. Load params and case ──────────────────────────────────────
    params = load_parameters_from_csv(case_name, params_path)
    ppc    = load_case_from_csv(case_name, params_path)
    print("✓ Params and PyPower case loaded")

    # ── 2. Load + scale data ─────────────────────────────────────────
    x_scaled, y_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_gen           = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses         = params['general']['n_buses']
    n_loads         = params['general']['n_loads']
    base_mva        = params['general']['BASE_MVA']

    print(f"\n[Data Info]  buses={n_buses}  gen={n_gen}  "
          f"non-slack={n_gen_non_slack}  loads={n_loads}  baseMVA={base_mva}")
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # ── 3. Split data ────────────────────────────────────────────────
    # Build test_configs dict: {name: {x_scaled, x_raw, idx, raw, params, ppc}}
    test_configs = {}

    if data_mode == 'combined':
        # Combined: train with random_split, test on 3 datasets
        train_idx, val_idx, test_idx_rs = prepare_data_splits(
            x_scaled, y_scaled, mode=DataMode.RANDOM_SPLIT,
            n_train_use=n_train_use, seed=seed)

        # 1) random_split test
        test_configs['random_split'] = {
            'x_scaled': x_scaled, 'x_raw': raw_data['x'],
            'idx': test_idx_rs, 'raw': raw_data,
            'params': params, 'ppc': ppc,
        }

        # 2) generalization test
        if test_data_path:
            gen_x_scaled, _, gen_raw, _ = load_generalization_test_data(
                test_data_path, params, scalers,
                n_test_samples=n_test_samples or 1000, seed=seed)
            test_configs['generalization'] = {
                'x_scaled': gen_x_scaled, 'x_raw': gen_raw['x'],
                'idx': np.arange(len(gen_x_scaled)), 'raw': gen_raw,
                'params': params, 'ppc': ppc,
            }

        # 3) api_test
        if test_params_path:
            api_data_path = acopf_config.get_data_path(acopf_config.TEST_CASE, None)
            api_params, api_x_scaled, _, api_raw, _ = load_api_test_data(
                api_data_path, test_params_path, scalers,
                n_test_samples=n_test_samples or 1000, seed=seed)
            api_case_name = os.path.basename(api_data_path)
            if api_case_name.endswith('_pd.csv'):
                api_case_name = api_case_name[:-7]
            api_ppc = load_case_from_csv(api_case_name, test_params_path)
            test_configs['api_test'] = {
                'x_scaled': api_x_scaled, 'x_raw': api_raw['x'],
                'idx': np.arange(len(api_x_scaled)), 'raw': api_raw,
                'params': api_params, 'ppc': api_ppc,
            }

    elif data_mode == DataMode.API_TEST:
        train_idx, val_idx, _ = prepare_data_splits(
            x_scaled, y_scaled, mode=DataMode.API_TEST,
            n_train_use=n_train_use, seed=seed)
        test_params, test_x_scaled, _, test_raw_data, _ = load_api_test_data(
            test_data_path, test_params_path, scalers,
            n_test_samples=n_test_samples or 1000, seed=seed)
        test_idx  = np.arange(len(test_x_scaled))
        test_ppc  = load_case_from_csv(
            test_data_path.split('/')[-1].replace('_pd.csv', ''),
            test_params_path)
        test_configs['api_test'] = {
            'x_scaled': test_x_scaled, 'x_raw': test_raw_data['x'],
            'idx': test_idx, 'raw': test_raw_data,
            'params': test_params, 'ppc': test_ppc,
        }

    elif data_mode == DataMode.GENERALIZATION:
        train_idx, val_idx, _ = prepare_data_splits(
            x_scaled, y_scaled, mode=DataMode.GENERALIZATION,
            n_train_use=n_train_use, seed=seed)
        test_x_scaled, _, test_raw_data, _ = load_generalization_test_data(
            test_data_path, params, scalers,
            n_test_samples=n_test_samples or 1000, seed=seed)
        test_idx    = np.arange(len(test_x_scaled))
        test_configs['generalization'] = {
            'x_scaled': test_x_scaled, 'x_raw': test_raw_data['x'],
            'idx': test_idx, 'raw': test_raw_data,
            'params': params, 'ppc': ppc,
        }

    else:
        # random_split or fixed_valtest
        train_idx, val_idx, test_idx = prepare_data_splits(
            x_scaled, y_scaled, mode=data_mode,
            n_train_use=n_train_use, seed=seed)
        test_configs[data_mode] = {
            'x_scaled': x_scaled, 'x_raw': raw_data['x'],
            'idx': test_idx, 'raw': raw_data,
            'params': params, 'ppc': ppc,
        }

    print(f"\n[Dataset Sizes]  train={len(train_idx)}  val={len(val_idx)}")
    for tname, tcfg in test_configs.items():
        print(f"  Test ({tname}): {len(tcfg['idx'])} samples")

    # ── 4. Build reward function (bootstrap scaling) ─────────────────
    print("\n[Reward] Estimating reward distribution for normalization "
          "(this may take a minute)...")

    norm_params = estimate_scaling_params(
        x_raw=raw_data['x'],
        train_idx=train_idx,
        params=params,
        scalers=scalers,
        ppc_template=ppc,
        ppopt=ppopt,
        num_samples=500,
        seed=seed,
    )
    reward_fn = Summation(
        penalty_weight=0.5,
        reward_scaling='normalization',
        scaling_params=norm_params,
    )
    print("✓ Reward function ready")
    print(f"  objective: mean={norm_params['mean_objective']:.2f}  "
          f"std={norm_params['std_objective']:.2f}")
    print(f"  penalty:   mean={norm_params['mean_penalty']:.4f}  "
          f"std={norm_params['std_penalty']:.4f}")

    # ── 5. Build training env ────────────────────────────────────────
    train_env = AcopfEnv(
        x_scaled=x_scaled,
        x_raw=raw_data['x'],
        indices=train_idx,
        params=params,
        scalers=scalers,
        ppc_template=ppc,
        ppopt=ppopt,
        reward_fn=reward_fn,
        seed=seed,
    )

    # ── 6. PPO model ─────────────────────────────────────────────────
    # Build policy_kwargs from hidden_sizes
    policy_kwargs = dict(net_arch=hidden_sizes)

    steps_per_epoch   = max(len(train_idx), 2048)   # at least one full pass
    total_timesteps   = steps_per_epoch * n_epochs

    print(f"\n{'=' * 70}")
    print(f"PPO Configuration")
    print(f"{'=' * 70}")
    print(f"Input dim  : {2 * n_loads}")
    print(f"Output dim : {n_gen_non_slack + n_gen}  "
          f"(pg_non_slack={n_gen_non_slack}, vm_gen={n_gen})")
    print(f"Network    : {hidden_sizes}")
    print(f"LR         : {learning_rate}")
    print(f"Batch size : {batch_size}")
    print(f"Epochs     : {n_epochs}  (≈ {total_timesteps} timesteps)")
    print(f"{'=' * 70}")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        clip_range=0.1,
        n_steps=min(steps_per_epoch, 4096),
        batch_size=batch_size,
        n_epochs=3,           # PPO inner optimisation epochs per rollout
        policy_kwargs=policy_kwargs,
        device=device,
        seed=seed,
        verbose=1,
    )

    # ── 7. Training ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Training")
    print(f"{'=' * 70}")
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    train_time = time.perf_counter() - t0
    print(f"\n✓ Training completed in {train_time:.2f} s")

    # ── 8. Inference speed ───────────────────────────────────────────
    first_cfg = next(iter(test_configs.values()))
    dummy_obs = first_cfg['x_scaled'][first_cfg['idx'][0]].astype(np.float32)
    for _ in range(10):
        model.predict(dummy_obs, deterministic=True)
    times = []
    for _ in range(100):
        t = time.perf_counter()
        model.predict(dummy_obs, deterministic=True)
        times.append(time.perf_counter() - t)
    latency_ms = np.mean(times) * 1000

    # ── 9. Evaluation on all test sets ───────────────────────────────
    all_metrics = {}
    for test_name, cfg in test_configs.items():
        print(f"\n{'=' * 70}")
        print(f"Test Set Evaluation: {test_name}")
        print(f"{'=' * 70}")

        metrics = evaluate_rl_agent(
            model=model,
            x_scaled=cfg['x_scaled'],
            x_raw=cfg['x_raw'],
            indices=cfg['idx'],
            raw_data=cfg['raw'],
            params=cfg['params'],
            scalers=scalers,
            ppc_template=cfg['ppc'],
            ppopt=ppopt,
            split_name=test_name,
            verbose=True,
        )
        all_metrics[test_name] = metrics

        # ── Print results for this test set ──
        print(f"\n{'=' * 70}")
        print(f"Results Summary: {test_name}")
        print(f"{'=' * 70}")
        print(f"\nData Mode : {data_mode}")
        print(f"Test Set  : {test_name}")

        print(f"\n--- Accuracy Metrics ---")
        print(f"MAE_Pg (Non-Slack): {metrics['mae_pg_non_slack_percent']:.4f}%")
        print(f"MAE_Vm (Generator): {metrics['mae_vm_percent']:.4f}%")
        print(f"MAE_Qg (All Gens):  {metrics['mae_qg_percent']:.4f}%")
        print(f"MAE_Va (All Buses): {metrics['mae_va_deg']:.4f} degrees")

        print(f"\n--- Violations (p.u.) ---")
        print(f"Pg_viol (Non-Slack): {metrics['mean_pg_viol_non_slack_pu']:.6f} p.u.")
        print(f"Pg_viol (Slack):     {metrics['mean_pg_viol_slack_pu']:.6f} p.u.")
        print(f"Qg_viol (All Gens):  {metrics['mean_max_qg_viol_pu']:.6f} p.u.")
        print(f"Vm_viol (All Buses): {metrics['mean_max_vm_viol_pu']:.6f} p.u.")
        print(f"Branch_viol:         {metrics['mean_max_branch_viol_pu']:.6f} p.u. "
              f"(1.0 = 100% overload)")

        print(f"\n--- Cost Metrics ---")
        print(f"Cost Gap: {metrics['cost_optimality_gap_percent']:.4f}%")

        print(f"\n--- Performance ---")
        print(f"Inference Time:   {latency_ms:.4f} ms/sample")
        print(f"Training Time:    {train_time:.2f} s")
        print(f"Convergence Rate: {metrics['convergence_rate_percent']:.2f}%")
        print(f"{'=' * 70}")

    return all_metrics


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    paths  = acopf_config.get_all_paths()
    params = acopf_config.get_all_params()

    # PPO uses learning_rate and hidden_sizes directly from config
    results = acopf_rl_experiment(**paths, **params)

    print("\n✓ RL Experiment completed successfully!")