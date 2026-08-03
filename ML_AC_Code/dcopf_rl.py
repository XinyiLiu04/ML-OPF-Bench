# -*- coding: utf-8 -*-
"""
DCOPF RL Main Experiment (PPO)

Version: v3.1 — Corrected hyperparameters after v3.0 divergence

Root cause of v3.0 failure:
- penalty_weight=10.0 + normalization scaling created a huge constant
  reward bias (~41), drowning out the gradient signal.
- n_epochs=10 (PPO internal) caused catastrophic policy updates,
  evidenced by std rising from 2.7→4.3 and approx_kl > 0.2 constantly.
- ent_coef=0.01 * penalty_weight=10 effectively amplified entropy,
  encouraging randomness.

v3.1 strategy — keep it simple, fix one thing at a time:
- reward_scaling='minmax11' maps both objective and penalty to [-1,1],
  eliminating the constant bias that killed gradients.
- penalty_weight=0.5 — with both terms in [-1,1], equal weighting works.
- target_kl=0.02 — early stops PPO inner loop when KL divergence gets
  too large, preventing the catastrophic updates seen in v3.0.
- gamma=0.0 — correct for single-step episodes.
- n_epochs=5 (PPO internal) — moderate increase from original 3.
- clip_range=0.15 — slightly relaxed from 0.1.
- log_std_init=-0.5 — moderate initial exploration.
- ent_coef=0.0 — no entropy bonus.

Usage:
  Edit the __main__ block at the bottom, then run this file.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ── project imports ───────────────────────────────────────────────────
from dcopf_data_setup import (
    load_parameters_from_csv,
    split_data_by_mode,
    DataSplitMode,
    load_and_prepare_data_generalization,
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
)
from reward import Summation


# =====================================================================
# Data Loading Function (v2.1 FIXED)
# =====================================================================

def load_and_scale_data_safe(file_path, params, column_names):
    """
    Safe data loading function with virtual bus handling.

    v2.1 FIX: Load demand mapping uses bus_id_to_idx dict instead of bus_id-1.
    Critical for case300 where bus IDs like 7001, 9533 exist.

    Returns: (x_data_scaled, y_training, Lg_Max, scalers, y_pg_raw, x_data_raw)
    """
    full_df = pd.read_csv(file_path)
    n_samples = len(full_df)
    n_buses = params['general']['n_buses']

    load_prefix = column_names['load_prefix']
    load_cols = sorted(
        [col for col in full_df.columns if col.startswith(load_prefix)],
        key=lambda c: int(c[len(load_prefix):])
    )

    bus_id_to_idx = params['general']['bus_id_to_idx']
    x_data_raw = np.zeros((n_samples, n_buses), dtype='float32')
    for col_name in load_cols:
        bus_id = int(col_name[len(load_prefix):])
        if bus_id in bus_id_to_idx:
            x_data_raw[:, bus_id_to_idx[bus_id]] = full_df[col_name].values.astype('float32')
        else:
            print(f"[WARNING] load bus {bus_id} not in bus_id_to_idx, skipping")

    g_bus = params['general']['g_bus']
    pg_cols = [f"{column_names['gen_prefix']}{int(gen_id)}" for gen_id in g_bus]
    missing = [c for c in pg_cols if c not in full_df.columns]
    if missing:
        raise ValueError(f"[load_and_scale_data_safe] Missing pg columns: {missing}")
    y_pg_raw = full_df[pg_cols].values.astype('float32')

    y_lambda_raw = full_df[column_names['lambda']].values.reshape(-1, 1)
    mu_g_min_cols = [f"{column_names['mu_g_min_prefix']}{int(i)}" for i in g_bus]
    y_mu_g_min_raw = full_df[mu_g_min_cols].values
    mu_g_max_cols = [f"{column_names['mu_g_max_prefix']}{int(i)}" for i in g_bus]
    y_mu_g_max_raw = full_df[mu_g_max_cols].values

    valid_branch_indices = np.where(params['constraints']['Pl_max'] < 1e10)[0]
    valid_branch_ids = params['general']['branch_ids'][valid_branch_indices]
    mu_line_pos_cols = [f"{column_names['mu_line_pos_prefix']}{i}" for i in valid_branch_ids]
    y_mu_line_pos_raw = full_df[mu_line_pos_cols].values
    mu_line_neg_cols = [f"{column_names['mu_line_neg_prefix']}{i}" for i in valid_branch_ids]
    y_mu_line_neg_raw = full_df[mu_line_neg_cols].values

    x_scaler = MinMaxScaler()
    scalers = {
        "pg": MinMaxScaler(), "lambda": MinMaxScaler(),
        "mu_g_min": MinMaxScaler(), "mu_g_max": MinMaxScaler(),
        "mu_line_pos": MinMaxScaler(), "mu_line_neg": MinMaxScaler(),
        "x": x_scaler,
    }

    x_data_scaled = x_scaler.fit_transform(x_data_raw)
    y_pg_scaled = scalers['pg'].fit_transform(y_pg_raw)
    y_lambda_scaled = scalers['lambda'].fit_transform(y_lambda_raw)
    y_mu_g_min_scaled = scalers['mu_g_min'].fit_transform(y_mu_g_min_raw)
    y_mu_g_max_scaled = scalers['mu_g_max'].fit_transform(y_mu_g_max_raw)
    y_mu_line_pos_scaled = scalers['mu_line_pos'].fit_transform(y_mu_line_pos_raw)
    y_mu_line_neg_scaled = scalers['mu_line_neg'].fit_transform(y_mu_line_neg_raw)

    y_data_scaled = (
        y_pg_scaled, y_lambda_scaled, y_mu_g_min_scaled,
        y_mu_g_max_scaled, y_mu_line_pos_scaled, y_mu_line_neg_scaled
    )
    y_physics = np.zeros((len(x_data_scaled), 1), dtype='float32')
    y_training = y_data_scaled + (y_physics,)

    Lg_Max = [np.max(np.abs(y)) for y in (
        y_lambda_raw, y_mu_g_max_raw, y_mu_g_min_raw,
        y_mu_line_pos_raw, y_mu_line_neg_raw
    )]

    return x_data_scaled, y_training, Lg_Max, scalers, y_pg_raw, x_data_raw


# =====================================================================
# DC reward helpers
# =====================================================================

def compute_dc_cost(pg_non_slack, pg_slack, params):
    n_g = params['general']['n_g']
    slack_idx = params['general']['slack_gen_indices']
    non_slack_idx = params['general']['non_slack_gen_indices']
    pg_full = np.zeros(n_g, dtype=np.float32)
    pg_full[non_slack_idx] = pg_non_slack
    pg_full[slack_idx] = pg_slack
    c2 = params['constraints']['C_Pg_c2']
    c1 = params['constraints']['C_Pg']
    c0 = params['constraints']['C_Pg_c0']
    return float(np.sum(c2 * pg_full ** 2 + c1 * pg_full + c0))


def compute_dc_penalty(pg_non_slack, pg_slack, pd_bus, params):
    n_g = params['general']['n_g']
    slack_idx = params['general']['slack_gen_indices']
    non_slack_idx = params['general']['non_slack_gen_indices']
    pg_full = np.zeros(n_g, dtype=np.float32)
    pg_full[non_slack_idx] = pg_non_slack
    pg_full[slack_idx] = pg_slack
    Pg_min = params['constraints']['Pg_min'].ravel()
    Pg_max = params['constraints']['Pg_max'].ravel()
    Pl_max = params['constraints']['Pl_max']
    PTDF = params['constraints']['PTDF']
    Map_g = params['constraints']['Map_g']
    pg_pen = float(np.sum(np.maximum(0, Pg_min - pg_full) + np.maximum(0, pg_full - Pg_max)))
    pg_bus = Map_g.T @ pg_full
    pl = PTDF.T @ (pg_bus - pd_bus)
    limited = Pl_max < 1e10
    br_pen = 0.0
    if np.any(limited):
        br_pen = float(np.sum(np.maximum(0, np.abs(pl[limited]) - Pl_max[limited])))
    return -(pg_pen + br_pen)


# =====================================================================
# Bootstrap (v3.1: mixed sampling)
# =====================================================================

def estimate_dc_scaling_params(x_raw, train_idx, params, pg_scaler,
                               num_samples=1000, seed=42):
    rng = np.random.default_rng(seed)
    non_slack_idx = params['general']['non_slack_gen_indices']
    slack_idx = params['general']['slack_gen_indices']
    n_non_slack = params['general']['n_g_non_slack']
    pg_min_ns = params['constraints']['Pg_min'].ravel()[non_slack_idx]
    pg_max_ns = params['constraints']['Pg_max'].ravel()[non_slack_idx]
    n_slack = len(slack_idx)

    sample_pool = rng.choice(train_idx, size=num_samples, replace=True)
    objectives, penalties = [], []

    for i, idx in enumerate(sample_pool):
        pd_bus = x_raw[idx]
        pd_total = float(pd_bus.sum())

        if i < num_samples // 2:
            action = rng.random(n_non_slack).astype(np.float32)
        else:
            capacity_range = pg_max_ns - pg_min_ns
            total_capacity = capacity_range.sum()
            if total_capacity > 0:
                slack_share = 1.0 / (n_non_slack + n_slack)
                non_slack_target = pd_total * (1 - slack_share * n_slack)
                target_above_min = non_slack_target - pg_min_ns.sum()
                proportional = target_above_min * capacity_range / total_capacity
                action = np.clip(proportional / (capacity_range + 1e-10), 0, 1)
                action = np.clip(action + rng.normal(0, 0.15, n_non_slack), 0, 1)
            else:
                action = rng.random(n_non_slack).astype(np.float32)

        action = action.astype(np.float32)
        pg_ns = action * (pg_max_ns - pg_min_ns) + pg_min_ns
        pg_slack_total = pd_total - pg_ns.sum()
        pg_slack = np.full(n_slack, pg_slack_total / max(n_slack, 1), dtype=np.float32)
        cost = compute_dc_cost(pg_ns, pg_slack, params)
        penalty = compute_dc_penalty(pg_ns, pg_slack, pd_bus, params)
        objectives.append(-cost)
        penalties.append(penalty)

    objectives = np.array(objectives)
    penalties = np.array(penalties)

    std_obj = float(np.std(objectives)) or 1.0
    std_pen = float(np.std(penalties)) or 1.0

    print(f"\n[Bootstrap Scaling Stats] (n={num_samples})")
    print(f"  Objective: mean={np.mean(objectives):.2f}, std={std_obj:.2f}, "
          f"range=[{np.min(objectives):.2f}, {np.max(objectives):.2f}]")
    print(f"  Penalty:   mean={np.mean(penalties):.2f}, std={std_pen:.2f}, "
          f"range=[{np.min(penalties):.2f}, {np.max(penalties):.2f}]")
    print(f"  Feasible fraction: {np.mean(penalties == 0.0):.1%}")

    return {
        'mean_objective': float(np.mean(objectives)),
        'std_objective': std_obj,
        'mean_penalty': float(np.mean(penalties)),
        'std_penalty': std_pen,
        'min_objective': float(np.min(objectives)),
        'max_objective': float(np.max(objectives)),
        'min_penalty': float(np.min(penalties)),
        'max_penalty': float(np.max(penalties)),
    }


# =====================================================================
# Gymnasium Environment (sequential cycling)
# =====================================================================

class DcopfEnv(gym.Env):
    metadata = {}

    def __init__(self, x_raw, x_scaled, indices, params, pg_scaler,
                 reward_fn, non_converge_reward=-10.0, seed=42):
        super().__init__()
        self.x_raw = x_raw[indices]
        self.x_scaled = x_scaled[indices]
        self.n_samples = len(indices)
        self.params = params
        self.pg_scaler = pg_scaler
        self.reward_fn = reward_fn

        non_slack_idx = params['general']['non_slack_gen_indices']
        slack_idx = params['general']['slack_gen_indices']
        self.non_slack_idx = non_slack_idx
        self.slack_idx = slack_idx
        self.n_non_slack = params['general']['n_g_non_slack']
        self.n_slack = params['general']['n_slack_gens']

        Pg_min = params['constraints']['Pg_min'].ravel()
        Pg_max = params['constraints']['Pg_max'].ravel()
        self.pg_min_ns = Pg_min[non_slack_idx]
        self.pg_max_ns = Pg_max[non_slack_idx]

        obs_dim = x_raw.shape[1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_non_slack,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._current_idx = 0
        self._order = self._rng.permutation(self.n_samples)
        self._position = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_idx = int(self._order[self._position])
        self._position += 1
        if self._position >= self.n_samples:
            self._position = 0
            self._order = self._rng.permutation(self.n_samples)
        return self.x_scaled[self._current_idx].astype(np.float32), {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        pg_ns = action * (self.pg_max_ns - self.pg_min_ns) + self.pg_min_ns
        pd_bus = self.x_raw[self._current_idx]
        pd_total = float(pd_bus.sum())
        pg_slack_total = pd_total - pg_ns.sum()
        pg_slack = np.full(self.n_slack,
                           pg_slack_total / max(self.n_slack, 1),
                           dtype=np.float32)
        objective = -compute_dc_cost(pg_ns, pg_slack, self.params)
        penalty = compute_dc_penalty(pg_ns, pg_slack, pd_bus, self.params)
        valid = (penalty == 0.0)
        reward = float(self.reward_fn(objective, penalty, valid))
        obs = self.x_scaled[self._current_idx].astype(np.float32)
        return obs, reward, True, False, {}


# =====================================================================
# Validation Callback
# =====================================================================

class ValidationCallback(BaseCallback):
    def __init__(self, eval_fn, eval_interval_steps=10000, verbose=1):
        super().__init__(verbose)
        self.eval_fn = eval_fn
        self.eval_interval = eval_interval_steps
        self.best_score = -np.inf
        self.best_params = None
        self.eval_history = []

    def _on_step(self):
        if self.num_timesteps % self.eval_interval == 0 and self.num_timesteps > 0:
            metrics = self.eval_fn(self.model)
            score = -abs(metrics['cost_gap_percent']) - 100 * (
                metrics['viol_pg_non_slack'] + metrics['viol_pg_slack'] +
                metrics['viol_branch'])
            self.eval_history.append({
                'timesteps': self.num_timesteps, 'score': score, **metrics
            })
            if score > self.best_score:
                self.best_score = score
                self.best_params = {
                    k: v.clone() for k, v in self.model.policy.state_dict().items()
                }
                if self.verbose:
                    print(f"  [Val @{self.num_timesteps}] New best! "
                          f"CostGap={metrics['cost_gap_percent']:.2f}% "
                          f"BrViol={metrics['viol_branch']:.4f} "
                          f"Score={score:.4f}")
            elif self.verbose:
                print(f"  [Val @{self.num_timesteps}] "
                      f"CostGap={metrics['cost_gap_percent']:.2f}% "
                      f"BrViol={metrics['viol_branch']:.4f} "
                      f"Score={score:.4f}")
        return True

    def restore_best(self):
        if self.best_params is not None:
            self.model.policy.load_state_dict(self.best_params)
            if self.verbose:
                print(f"  [Val] Restored best model (score={self.best_score:.4f})")


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_rl_agent(model, x_raw, x_scaled, indices, y_pg_all, params,
                      pg_scaler, split_name="Test", verbose=True):
    non_slack_idx = params['general']['non_slack_gen_indices']
    n_g_non_slack = params['general']['n_g_non_slack']
    pg_min_ns = params['constraints']['Pg_min'].ravel()[non_slack_idx]
    pg_max_ns = params['constraints']['Pg_max'].ravel()[non_slack_idx]

    if indices is not None:
        x_raw_eval = x_raw[indices]
        x_scaled_eval = x_scaled[indices]
        y_true_all = y_pg_all[indices]
    else:
        x_raw_eval = x_raw
        x_scaled_eval = x_scaled
        y_true_all = y_pg_all

    n_samples = len(x_raw_eval)
    y_pred_ns = np.zeros((n_samples, n_g_non_slack), dtype=np.float32)
    pd_totals = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        obs = x_scaled_eval[i].astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, 0.0, 1.0)
        pg_ns = action * (pg_max_ns - pg_min_ns) + pg_min_ns
        y_pred_ns[i] = pg_ns
        pd_totals[i] = float(x_raw_eval[i].sum())

    y_pred_all = reconstruct_full_pg(y_pred_ns, pd_totals, params)
    mae_dict = compute_detailed_mae(y_true_all, y_pred_ns, y_pred_all, params)
    gen_up_viol, gen_lo_viol, line_viol, _ = dc_feasibility(y_pred_all, x_raw_eval, params)
    viol_dict = compute_detailed_pg_violations_pu(gen_up_viol, gen_lo_viol, params)
    branch_violation_pu = compute_branch_violation_pu(line_viol, params['constraints']['Pl_max'])
    cost_coeffs = {
        'C2': params['constraints'].get('C_Pg_c2', np.zeros(y_true_all.shape[1])),
        'C1': params['constraints']['C_Pg'],
        'C0': params['constraints'].get('C_Pg_c0', np.zeros(y_true_all.shape[1])),
    }
    cost_true = compute_cost(y_true_all, cost_coeffs)
    cost_pred = compute_cost(y_pred_all, cost_coeffs)
    cost_gap_pct = compute_cost_gap_percentage(cost_true, cost_pred)

    return {
        'mae_pg_non_slack': mae_dict['mae_non_slack'],
        'mae_pg_slack': mae_dict['mae_slack'],
        'viol_pg_non_slack': viol_dict['viol_non_slack'],
        'viol_pg_slack': viol_dict['viol_slack'],
        'viol_branch': branch_violation_pu,
        'cost_gap_percent': cost_gap_pct,
    }


# =====================================================================
# Main experiment (v3.1)
# =====================================================================

def dcopf_rl_experiment(
        case_name, params_path, dataset_path, n_train_use=10000, seed=42,
        n_epochs=100, learning_rate=3e-4, batch_size=256, hidden_layers=None,
        split_mode=DataSplitMode.RANDOM_SPLIT, test_data_path=None,
        test_params_path=None, column_names=None, n_test_samples=1000,
        penalty_weight=0.5, reward_scaling='minmax11',
        bootstrap_samples=1000, eval_interval_epochs=10):

    hidden_layers = hidden_layers or [256, 256]
    np.random.seed(seed)
    if column_names is None:
        column_names = {
            'load_prefix': 'pd', 'gen_prefix': 'pg', 'lambda': 'lambda',
            'mu_g_min_prefix': 'mu_g_min_', 'mu_g_max_prefix': 'mu_g_max_',
            'mu_line_pos_prefix': 'mu_line_max_', 'mu_line_neg_prefix': 'mu_line_min_',
        }

    print(f"\n{'='*70}\nDCOPF RL Experiment (PPO) v3.1\n{'='*70}")
    print(f"Case: {case_name}")
    print(f"Mode: {split_mode.value}")
    print(f"Train samples: {n_train_use}, Epochs: {n_epochs}")
    print(f"Network: {hidden_layers}, LR: {learning_rate}, Batch: {batch_size}")
    print(f"Penalty weight: {penalty_weight}, Reward scaling: {reward_scaling}")
    print(f"{'='*70}")

    # ── Load parameters ──────────────────────────────────────────────
    params = load_parameters_from_csv(case_name, params_path, is_api=False)
    params = update_params_with_slack_info(params, identify_slack_bus_and_gens(params))

    print(f"\nSystem: {params['general']['n_buses']} buses, "
          f"{params['general']['n_g']} generators "
          f"({params['general']['n_g_non_slack']} non-slack, "
          f"{params['general']['n_slack_gens']} slack), "
          f"{params['general']['n_line']} branches")

    # ── Load and scale data ──────────────────────────────────────────
    x_scaled_full, _, _, scalers, y_pg_raw_full, x_raw_full = \
        load_and_scale_data_safe(
            file_path=dataset_path, params=params, column_names=column_names)
    y_pg_all_full = y_pg_raw_full
    pg_scaler = scalers['pg']
    x_scaler = scalers['x']

    # ── Split data ───────────────────────────────────────────────────
    train_idx, val_idx, test_idx, x_test_ext, y_test_ext = split_data_by_mode(
        x_data_raw=x_raw_full, y_pg_raw=y_pg_all_full, mode=split_mode,
        n_train_use=n_train_use, seed=seed, test_data_path=test_data_path,
        params=params, column_names=column_names, n_test_samples=n_test_samples)

    if split_mode == DataSplitMode.API_TEST and test_params_path:
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_params = update_params_with_slack_info(
            test_params, identify_slack_bus_and_gens(test_params))
    else:
        test_params = params

    # ── Reward function setup ────────────────────────────────────────
    norm_params = estimate_dc_scaling_params(
        x_raw_full, train_idx, params, pg_scaler, bootstrap_samples, seed)

    reward_fn = Summation(
        penalty_weight=penalty_weight,
        reward_scaling=reward_scaling,
        scaling_params=norm_params,
    )

    sp = reward_fn.scaling_params
    print(f"\n[Reward Scaling Applied] ({reward_scaling})")
    print(f"  obj_factor={sp['objective_factor']:.6f}, obj_bias={sp['objective_bias']:.6f}")
    print(f"  pen_factor={sp['penalty_factor']:.6f}, pen_bias={sp['penalty_bias']:.6f}")

    # Reward sanity check
    rng_check = np.random.default_rng(seed + 1)
    test_rewards = []
    pg_min_ns = params['constraints']['Pg_min'].ravel()[params['general']['non_slack_gen_indices']]
    pg_max_ns = params['constraints']['Pg_max'].ravel()[params['general']['non_slack_gen_indices']]
    for _ in range(200):
        idx = rng_check.choice(train_idx)
        pd_bus = x_raw_full[idx]
        pd_total = float(pd_bus.sum())
        act = rng_check.random(params['general']['n_g_non_slack']).astype(np.float32)
        pg_ns = act * (pg_max_ns - pg_min_ns) + pg_min_ns
        n_sl = params['general']['n_slack_gens']
        pg_sl = np.full(n_sl, (pd_total - pg_ns.sum()) / max(n_sl, 1), dtype=np.float32)
        obj = -compute_dc_cost(pg_ns, pg_sl, params)
        pen = compute_dc_penalty(pg_ns, pg_sl, pd_bus, params)
        r = float(reward_fn(obj, pen, pen == 0.0))
        test_rewards.append(r)
    test_rewards = np.array(test_rewards)
    print(f"  Reward check (200 random): mean={test_rewards.mean():.3f}, "
          f"std={test_rewards.std():.3f}, range=[{test_rewards.min():.3f}, {test_rewards.max():.3f}]")

    # ── Build environment ────────────────────────────────────────────
    train_env = DcopfEnv(
        x_raw=x_raw_full, x_scaled=x_scaled_full, indices=train_idx,
        params=params, pg_scaler=pg_scaler, reward_fn=reward_fn, seed=seed)

    # ── PPO hyperparameters (v3.1) ───────────────────────────────────
    steps_per_epoch = max(len(train_idx), 2048)
    n_steps_ppo = min(steps_per_epoch, 4096)
    total_timesteps = steps_per_epoch * n_epochs

    print(f"\n[PPO Config]")
    print(f"  n_steps={n_steps_ppo}, total_timesteps={total_timesteps}")
    print(f"  batch_size={batch_size}, ppo_epochs=5, clip_range=0.15")
    print(f"  gamma=0.0, target_kl=0.02, ent_coef=0.0")

    model = PPO(
        "MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps_ppo,
        batch_size=batch_size,
        n_epochs=3,               # moderate (original=3, v3.0=10→diverged)
        policy_kwargs=dict(
            net_arch=hidden_layers,
            log_std_init=-0.5,    # moderate exploration
        ),
        clip_range=0.2,          # slightly relaxed from original 0.1
        gamma=0.0,                # single-step, no discounting
        gae_lambda=1.0,
        ent_coef=0.0,             # no entropy bonus
        vf_coef=1.0,
        max_grad_norm=0.5,
        target_kl=None,           # KEY: early stop PPO when KL too large
        seed=seed,
        verbose=1,
    )

    # ── Validation callback ──────────────────────────────────────────
    eval_interval_steps = steps_per_epoch * eval_interval_epochs

    def val_eval_fn(model_):
        if val_idx is not None and len(val_idx) > 0:
            return evaluate_rl_agent(
                model_, x_raw_full, x_scaled_full, val_idx,
                y_pg_all_full, params, pg_scaler, verbose=False)
        else:
            small_idx = train_idx[:min(500, len(train_idx))]
            return evaluate_rl_agent(
                model_, x_raw_full, x_scaled_full, small_idx,
                y_pg_all_full, params, pg_scaler, verbose=False)

    val_callback = ValidationCallback(
        eval_fn=val_eval_fn,
        eval_interval_steps=eval_interval_steps,
        verbose=1,
    )

    # ── Training ─────────────────────────────────────────────────────
    print(f"\nTraining started...")
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=val_callback,
                progress_bar=False)
    train_time = time.perf_counter() - t0

    val_callback.restore_best()

    # ── Inference latency ────────────────────────────────────────────
    dummy_obs = x_scaled_full[train_idx[0]].astype(np.float32)
    for _ in range(10):
        model.predict(dummy_obs, deterministic=True)
    times = []
    for _ in range(100):
        t = time.perf_counter()
        model.predict(dummy_obs, deterministic=True)
        times.append(time.perf_counter() - t)
    latency_ms = float(np.mean(times)) * 1000

    # ── Test evaluation ──────────────────────────────────────────────
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        x_test_scaled = x_scaler.transform(x_test_ext)
        test_metrics = evaluate_rl_agent(
            model, x_test_ext, x_test_scaled, None,
            y_test_ext, test_params, pg_scaler)
    else:
        test_metrics = evaluate_rl_agent(
            model, x_raw_full, x_scaled_full, test_idx,
            y_pg_all_full, params, pg_scaler)

    # ── Print results ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Test Set Results")
    print("=" * 70)
    print(f"Case: {case_name}  |  Mode: {split_mode.value}")
    print(f"\nNon-Slack: MAE={test_metrics['mae_pg_non_slack']:.4f}%  "
          f"Viol={test_metrics['viol_pg_non_slack']:.4f} p.u.")
    print(f"Slack:     MAE={test_metrics['mae_pg_slack']:.4f}%  "
          f"Viol={test_metrics['viol_pg_slack']:.4f} p.u.")
    print(f"Branch:    Viol={test_metrics['viol_branch']:.4f} p.u.")
    print(f"Cost Gap:  {test_metrics['cost_gap_percent']:.4f}%")
    print(f"Training:  {train_time:.2f} s   Inference: {latency_ms:.4f} ms")
    print("=" * 70 + "\n")

    return test_metrics


if __name__ == "__main__":
    CASE_NAME = 'pglib_opf_case300_ieee'
    CASE_SHORT_NAME = 'case300'
    SPLIT_MODE = DataSplitMode.RANDOM_SPLIT

    ROOT_DIR = "/lambda/nfs/lxy/dcopf_project/data"
    TRAIN_VARIANCE = "v=0.12"
    TEST_VARIANCE = "v=0.25"

    PARAMS_PATH = os.path.join(ROOT_DIR, "DCOPF Constraints", CASE_SHORT_NAME)
    DATASET_PATH = os.path.join(
        ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TRAIN_VARIANCE})",
        f"{CASE_NAME}_dataset_with_duals.csv")

    if SPLIT_MODE == DataSplitMode.GENERALIZATION:
        TEST_DATA_PATH = os.path.join(
            ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TEST_VARIANCE})",
            f"{CASE_NAME}_dataset_with_duals.csv")
        TEST_PARAMS_PATH = None
    elif SPLIT_MODE == DataSplitMode.API_TEST:
        TEST_DATA_PATH = os.path.join(
            ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}(v=api)",
            f"{CASE_NAME}__api_dataset_with_duals.csv")
        TEST_PARAMS_PATH = os.path.join(
            ROOT_DIR, "DCOPF Constraints", f"{CASE_SHORT_NAME}(api)")
    else:
        TEST_DATA_PATH = None
        TEST_PARAMS_PATH = None

    COLUMN_NAMES = {
        'load_prefix': 'pd', 'gen_prefix': 'pg', 'lambda': 'lambda',
        'mu_g_min_prefix': 'mu_g_min_', 'mu_g_max_prefix': 'mu_g_max_',
        'mu_line_pos_prefix': 'mu_line_max_', 'mu_line_neg_prefix': 'mu_line_min_',
    }

    dcopf_rl_experiment(
        case_name=CASE_NAME,
        params_path=PARAMS_PATH,
        dataset_path=DATASET_PATH,
        n_train_use=12000,
        seed=42,
        n_epochs=200,
        learning_rate=1e-4,
        batch_size=128,
        hidden_layers=[256, 128],
        split_mode=SPLIT_MODE,
        test_data_path=TEST_DATA_PATH,
        test_params_path=TEST_PARAMS_PATH,
        column_names=COLUMN_NAMES,
        n_test_samples=1000,
        penalty_weight=0.5,           # restored from original
        reward_scaling='minmax11',    # KEY change: [-1,1] range, no bias
        bootstrap_samples=1000,
        eval_interval_epochs=20,
    )