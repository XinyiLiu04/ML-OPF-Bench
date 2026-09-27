# -*- coding: utf-8 -*-
"""DDPG with non-slack Pg actions and CSV-fixed generator voltage.

Single-step physical-bound variant. A bounded Summation reward uses physical scales and separate constraint diagnostics.
"""

from ml_opf_bench.runtime import TrainingState, is_managed
from pgonly_reward import BoundedSummation, constraint_components, feasible, TOLERANCES

import numpy as np
import time
import sys

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env

import os

# ac_configuration/ sits in ac_methods/. Appending the parent of this script's own
# directory makes it importable whether this file is directly in ac_methods/ or one
# level down in a grouped method folder, and from any working directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ac_configuration import acopf_config
    from ac_configuration.acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        prepare_data_splits,
    )
    from ac_configuration.acopf_evaluation_metrics import evaluate_acopf_predictions
    from ac_configuration.acopf_pypower import load_case_from_csv, solve_pf_setpoints
except ImportError as e:
    print(f"Error: Unable to import from ac_configuration/ ({e})")
    sys.exit(1)

try:
    from reward import Summation
except ImportError:
    print("Error: cannot import reward.py; it must sit next to this script")
    sys.exit(1)

# A sample whose power flow diverges yields no cost and no violation to score, so it
# gets a fixed reward well below the scaled range of a converged sample
NON_CONVERGE_REWARD = -3.0

# Feasibility is decided with a tolerance rather than an exact comparison to zero,
# since the violations are sums of clipped floating point differences
FEASIBILITY_TOL = 1e-9


def compute_cost_from_pf(r1_pf, base_mva, cost_c2, cost_c1, cost_c0):
    """Generation cost in $/h from a converged power flow; Pg arrives in MW."""
    pg_pu = r1_pf[0]['gen'][:, 1] / base_mva
    return float(np.sum(cost_c2 * pg_pu ** 2 + cost_c1 * pg_pu + cost_c0))


def compute_penalty_from_pf(r1_pf, base_mva):
    """Total constraint violation of a converged power flow, returned as a value <= 0.

    The four categories are summed even though their units differ: Pg and Qg are in
    p.u. power, Vm in p.u. voltage, and branch loading is a relative overload. This is
    a deliberate simplification of the reward signal, not a physical quantity, and it
    is not the same thing as the per-category violations the evaluation module reports.
    """
    gen = r1_pf[0]['gen']
    bus = r1_pf[0]['bus']
    branch = r1_pf[0]['branch']

    pg_mw = gen[:, 1]
    pg_pen = np.sum(np.maximum(0, gen[:, 9] - pg_mw)
                    + np.maximum(0, pg_mw - gen[:, 8])) / base_mva

    qg_mvar = gen[:, 2]
    qg_pen = np.sum(np.maximum(0, gen[:, 4] - qg_mvar)
                    + np.maximum(0, qg_mvar - gen[:, 3])) / base_mva

    vm_pu = bus[:, 7]
    vm_pen = np.sum(np.maximum(0, bus[:, 12] - vm_pu)
                    + np.maximum(0, vm_pu - bus[:, 11]))

    rate_a = branch[:, 5]
    limited = np.isfinite(rate_a) & (rate_a > 0)
    br_pen = 0.0
    if np.any(limited):
        Ff = np.abs(branch[limited, 13] + 1j * branch[limited, 14])
        Ft = np.abs(branch[limited, 15] + 1j * branch[limited, 16])
        ra = rate_a[limited]
        br_pen = float(np.sum(np.maximum(0, Ff / ra - 1)
                              + np.maximum(0, Ft / ra - 1)))

    return -(pg_pen + qg_pen + vm_pen + br_pen)


def action_to_setpoints(action, bounds):
    """Only non-slack Pg is controlled; CSV generator voltages stay fixed."""
    action = np.asarray(action)
    if action.shape != bounds['pg_min'].shape or not np.isfinite(action).all():
        raise ValueError("Expected finite non-slack Pg action vector")
    pg = bounds['pg_min'] + np.clip(action, 0, 1) * (bounds['pg_max'] - bounds['pg_min'])
    return pg, bounds['vm_fixed'].copy()


def make_action_bounds(params, case_data):
    ns = params['general']['non_slack_gen_idx']
    return dict(pg_min=params['generator']['pg_min'].flatten()[ns].astype(float),
                pg_max=params['generator']['pg_max'].flatten()[ns].astype(float),
                vm_fixed=case_data['gen'][:, 5].copy())


class AcopfEnv(gym.Env):
    """One sample per episode: observe the load, set the generators, get scored."""

    metadata = {}

    def __init__(self, x_scaled, x_raw, indices, params, case_data, bounds,
                 reward_fn, non_converge_reward=NON_CONVERGE_REWARD, seed=42):
        super().__init__()

        self.x_scaled = x_scaled[indices]
        if not np.isfinite(self.x_scaled).all():
            raise ValueError("RL observations must be finite")
        self.x_raw = x_raw[indices]
        self.n_samples = len(indices)

        self.params = params
        self.case_data = case_data
        self.bounds = bounds
        self.reward_fn = reward_fn
        self.non_converge_reward = non_converge_reward

        self.n_loads = params['general']['n_loads']
        self.n_gen = params['general']['n_gen']
        self.n_gen_non_slack = params['general']['n_gen_non_slack']
        self.base_mva = params['general']['BASE_MVA']

        self.cost_c2 = params['generator']['cost_c2']
        self.cost_c1 = params['generator']['cost_c1']
        self.cost_c0 = params['generator']['cost_c0']

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * self.n_loads,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_gen_non_slack,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._current_idx = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_idx = int(self._rng.integers(0, self.n_samples))
        return self.x_scaled[self._current_idx].astype(np.float32), {}

    def step(self, action):
        pg_non_slack, vm_gen = action_to_setpoints(action, self.bounds)

        pd_pu = self.x_raw[self._current_idx, :self.n_loads]
        qd_pu = self.x_raw[self._current_idx, self.n_loads:]

        r1_pf = solve_pf_setpoints(pd_pu, qd_pu, pg_non_slack, vm_gen,
                                  self.params, self.case_data)
        converged = bool(r1_pf[0]['success'])
        info = dict(pf_converged=converged, feasible=False)
        if converged:
            components = constraint_components(r1_pf[0], self.base_mva)
            cost = compute_cost_from_pf(r1_pf, self.base_mva, self.cost_c2,
                                        self.cost_c1, self.cost_c0)
            reward = self.reward_fn(cost, components)
            info.update(feasible=feasible(components), violations=components, cost=cost)
        else:
            reward = self.non_converge_reward
        obs = self.x_scaled[self._current_idx].astype(np.float32)
        return obs, float(reward), True, False, info


def evaluate_rl_agent(model, x_scaled, x_raw, indices, raw_data, params,
                      case_data, bounds, split_name="Test", verbose=True):
    """Act greedily on every test sample, solve the power flow, and score the result."""
    n_samples = len(indices)
    n_gen = params['general']['n_gen']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    base_mva = params['general']['BASE_MVA']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_bus_indices = np.array(
        [bus_id_to_idx[int(gid)] for gid in params['general']['gen_bus_ids']])

    if verbose:
        print(f"\n{split_name} Evaluation:")
        print(f"  Computing power flow for {n_samples} samples...")

    # One batched prediction rather than one call per sample
    obs_batch = x_scaled[indices].astype(np.float32)
    actions, _ = model.predict(obs_batch, deterministic=True)

    pf_results_list = []
    converge_flags = []
    y_pred_pg_full = np.zeros((n_samples, n_gen))
    y_pred_vm_all = np.full((n_samples, n_buses), 1.0)

    for i, sample_idx in enumerate(indices):
        pg_non_slack, vm_gen = action_to_setpoints(actions[i], bounds)
        y_pred_vm_all[i, gen_bus_indices] = vm_gen

        pd_pu = x_raw[sample_idx, :n_loads]
        qd_pu = x_raw[sample_idx, n_loads:]

        try:
            r1_pf = solve_pf_setpoints(pd_pu, qd_pu, pg_non_slack, vm_gen,
                                       params, case_data)
            converged = bool(r1_pf[0]['success'])
        except Exception:
            r1_pf = ({'success': False,
                      'gen': np.zeros((n_gen, 21)),
                      'bus': np.zeros((n_buses, 13)),
                      'branch': np.zeros((1, 17))},)
            converged = False

        pf_results_list.append(r1_pf)
        converge_flags.append(converged)

        if converged:
            y_pred_pg_full[i] = r1_pf[0]['gen'][:, 1] / base_mva

    if verbose:
        print(f"  Converged: {sum(converge_flags)}/{n_samples}")

    return evaluate_acopf_predictions(
        y_pred_pg_full,
        y_pred_vm_all,
        raw_data['pg'][indices],
        raw_data['vm'][indices],
        raw_data['qg'][indices],
        raw_data['va'][indices],
        pf_results_list,
        converge_flags,
        params,
        verbose=verbose,
    )


def acopf_ddpg_pgonly_experiment(
        case_name,
        params_path,
        data_path,
        n_train_use=None,
        seed=42,
        n_epochs=100,
        learning_rate=3e-4,
        hidden_sizes=None,
        batch_size=256,
        device='cuda',
        total_timesteps=2_000_000,
        penalty_weight=0.5,
        action_bounds='physical',
        n_scaling_probes=500,
        rollout_steps=None,
        learning_starts=1024,
        early_stop_patience=20,
        early_stop_min_delta=1e-6,
        **kwargs  # Absorbs settings that do not apply, such as early stopping
):
    """Train DDPG on a random split and evaluate on the test indices."""
    hidden_sizes = hidden_sizes or [256, 256]

    print(f"\n{'=' * 70}")
    print(f"ACOPF RL Experiment (DDPG Pg-only)")
    print(f"{'=' * 70}")
    print(f"Case: {case_name}  |  Device: {device}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # 1. Network parameters and PyPower case data
    # ------------------------------------------------------------------
    params = load_parameters_from_csv(case_name, params_path)
    case_data = load_case_from_csv(case_name, params_path)

    # ------------------------------------------------------------------
    # 2. Dataset. The labels are used only by the evaluation, never by the agent.
    # ------------------------------------------------------------------
    x_scaled, y_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True,
                                  n_train_use=n_train_use, seed=seed)

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    base_mva = params['general']['BASE_MVA']

    print(f"\n[Dataset Info]")
    print(f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_non_slack}), "
          f"Loads: {n_loads}, Base MVA: {base_mva}")
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = prepare_data_splits(
        x_scaled, y_scaled, n_train_use=n_train_use, seed=seed)

    if action_bounds != "physical":
        raise ValueError("DDPG Pg-only requires physical bounds")
    bounds = make_action_bounds(params, case_data)
    print(f"\n[Action Space] bounds from '{action_bounds}'")
    print(f"  Pg range: [{bounds['pg_min'].min():.4f}, {bounds['pg_max'].max():.4f}] p.u.")
    print(f"  Vm range: [{bounds['vm_fixed'].min():.4f}, {bounds['vm_fixed'].max():.4f}] p.u.")

    # ------------------------------------------------------------------
    # 4. Reward scaling, estimated from random actions
    # ------------------------------------------------------------------
    reward_fn = BoundedSummation(params)
    norm_params = dict(cost_scale=reward_fn.cost_scale, tolerances=TOLERANCES,
                       kind="bounded_summation_v1", failure_reward=-3.0)
    # ------------------------------------------------------------------
    # 5. Environment
    # ------------------------------------------------------------------
    train_env = AcopfEnv(
        x_scaled=x_scaled,
        x_raw=raw_data['x'],
        indices=train_idx,
        params=params,
        case_data=case_data,
        bounds=bounds,
        reward_fn=reward_fn,
        seed=seed,
    )
    check_env(train_env, warn=True)

    # ------------------------------------------------------------------
    # 6. DDPG. Every environment step costs one power flow solve, so the
    #    timestep budget, not the epoch count, is what sets the runtime.
    # ------------------------------------------------------------------
    if total_timesteps is None or total_timesteps <= 0:
        raise ValueError("An explicit positive step budget is required")
    print(f"DDPG Pg-only: {n_gen_non_slack} actions, budget={total_timesteps}")

    model = DDPG(
        "MlpPolicy", train_env, learning_rate=learning_rate,
        buffer_size=1_000_000, learning_starts=learning_starts, batch_size=batch_size,
        train_freq=(1, "step"), gradient_steps=1, gamma=0.0,
        action_noise=NormalActionNoise(np.zeros(n_gen_non_slack),
                                      0.1 * np.ones(n_gen_non_slack)),
        policy_kwargs=dict(net_arch=hidden_sizes), device=device, seed=seed, verbose=1,
    )

    # ------------------------------------------------------------------
    # 7. Training
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Training")
    print(f"{'=' * 70}")
    t0 = time.perf_counter()
    from pgonly_validation import ValidationRewardStopping
    val_env = AcopfEnv(x_scaled, raw_data['x'], val_idx, params, case_data,
                       bounds, reward_fn, seed=seed)
    callback = ValidationRewardStopping(val_env, min(len(train_idx), total_timesteps),
                                         early_stop_patience, early_stop_min_delta)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    train_time = time.perf_counter() - t0
    if is_managed():
        return TrainingState(model.policy, params, train_time, dict(scalers=scalers, bounds=bounds, total_timesteps=model.num_timesteps, algorithm="DDPG", action_mode="pg_only", voltage_source="gen_data.csv:vg_pu", reward_scaling=norm_params, validation_history=callback.history))
    print(f"\nTraining completed in {train_time:.2f} seconds")

    # ------------------------------------------------------------------
    # 8. Inference latency (policy forward pass only)
    # ------------------------------------------------------------------
    dummy_obs = x_scaled[test_idx[0]].astype(np.float32)
    for _ in range(10):
        model.predict(dummy_obs, deterministic=True)

    times = []
    for _ in range(100):
        t = time.perf_counter()
        model.predict(dummy_obs, deterministic=True)
        times.append(time.perf_counter() - t)
    latency_ms = float(np.mean(times)) * 1000

    # ------------------------------------------------------------------
    # 9. Evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    metrics = evaluate_rl_agent(
        model=model,
        x_scaled=x_scaled,
        x_raw=raw_data['x'],
        indices=test_idx,
        raw_data=raw_data,
        params=params,
        case_data=case_data,
        bounds=bounds,
        split_name="Test",
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 10. Results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Final Results Summary")
    print(f"{'=' * 70}")
    print(f"\nCase: {case_name}")

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
    print(f"Inference Time: {latency_ms:.4f} ms/sample (policy forward only)")
    print(f"Training Time:  {train_time:.2f} s")
    print(f"Convergence Rate: {metrics['convergence_rate_percent']:.2f}%")
    print(f"{'=' * 70}")

    return metrics

