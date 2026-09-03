# -*- coding: utf-8 -*-
"""
THIS IS THE NAN VERSION!!!
ACOPF Evaluation Metrics Module (V11-Converged-Only)

Modifications (V11):
  All violation metrics, MAE_Pg, MAE_Qg, MAE_Va and cost metrics are computed
  ONLY over converged samples.  Non-converged samples are excluded entirely
  from these averages.  The convergence rate is still reported over all samples
  so readers can interpret the coverage of the other metrics.

  Rationale: filling non-converged samples with sentinel values (1000 p.u.) or
  zeros produces physically meaningless averages that mask true solution quality.

Previous (V10):
1. evaluate_acopf_predictions: Separate statistics for non-slack, all, and slack MAE_PG
2. All violations in p.u. units
3. Branch violation in p.u. (1.0 = 100% overload)
4. Print separately for three metrics

Includes:
1. Violation calculation: PG/QG/VM/Branch violations
2. MAE calculation: PG/VM/QG/VA accuracy metrics (separate non-slack and all)
3. Cost calculation: Generation cost evaluation
"""
import numpy as np


# =====================================================================
# Violation Calculation Functions - ALL IN P.U. UNITS
# =====================================================================
def calculate_single_sample_violations(r1_pf, is_converged, base_mva):
    """
    Calculate 4 types of maximum violations for single sample (PG/QG/VM/Branch)

    Returns: All violations in p.u. units
    """
    if not is_converged:
        # Return NaN so non-converged samples can be masked out in aggregation.
        # Using a sentinel (e.g. 1000) would pollute the mean when convergence
        # rate is low (e.g. case300 at ~7%).
        return np.nan, np.nan, np.nan, np.nan

    gen = r1_pf[0]['gen']
    bus = r1_pf[0]['bus']
    branch = r1_pf[0]['branch']

    # PG violation (convert to p.u.)
    pg_runpf_mw = gen[:, 1]
    pg_min_mw = gen[:, 9]
    pg_max_mw = gen[:, 8]
    pg_viol_mw = np.maximum(0, pg_min_mw - pg_runpf_mw) + np.maximum(0, pg_runpf_mw - pg_max_mw)
    pg_viol_pu = pg_viol_mw / base_mva
    max_pg_viol_pu = np.max(pg_viol_pu) if pg_viol_pu.size > 0 else 0.0

    # QG violation (convert to p.u.)
    qg_runpf_mvar = gen[:, 2]
    qg_min_mvar = gen[:, 4]
    qg_max_mvar = gen[:, 3]
    qg_viol_mvar = np.maximum(0, qg_min_mvar - qg_runpf_mvar) + np.maximum(0, qg_runpf_mvar - qg_max_mvar)
    qg_viol_pu = qg_viol_mvar / base_mva
    max_qg_viol_pu = np.max(qg_viol_pu) if qg_viol_pu.size > 0 else 0.0

    # VM violation (already in p.u.)
    vm_runpf_pu = bus[:, 7]
    vm_min_pu = bus[:, 12]
    vm_max_pu = bus[:, 11]
    vm_viol_pu = np.maximum(0, vm_min_pu - vm_runpf_pu) + np.maximum(0, vm_runpf_pu - vm_max_pu)
    max_vm_viol_pu = np.max(vm_viol_pu) if vm_viol_pu.size > 0 else 0.0

    # Branch violation (in p.u., 1.0 = 100% overload)
    rate_a_mva = branch[:, 5]
    limit_idx = (rate_a_mva > 0) & (rate_a_mva < 9000)
    max_branch_viol_pu = 0.0

    if np.any(limit_idx):
        Ff_MVA = np.abs(branch[limit_idx, 13] + 1j * branch[limit_idx, 14])
        Ft_MVA = np.abs(branch[limit_idx, 15] + 1j * branch[limit_idx, 16])
        rate_a_MVA = rate_a_mva[limit_idx]

        # Violation in p.u. (1.0 means 100% overload)
        ff_viol_pu = np.maximum(0, (Ff_MVA / rate_a_MVA) - 1)
        ft_viol_pu = np.maximum(0, (Ft_MVA / rate_a_MVA) - 1)

        all_branch_viols = np.concatenate([ff_viol_pu, ft_viol_pu])
        max_branch_viol_pu = np.max(all_branch_viols) if all_branch_viols.size > 0 else 0.0

    return max_pg_viol_pu, max_qg_viol_pu, max_vm_viol_pu, max_branch_viol_pu


def extract_pf_results(r1_pf, is_converged, base_mva, n_gen, n_buses):
    """Extract QG and VA values from power flow results"""
    if not is_converged:
        return np.zeros(n_gen), np.zeros(n_buses)

    gen = r1_pf[0]['gen']
    bus = r1_pf[0]['bus']

    qg_pu = gen[:, 2] / base_mva
    va_deg = bus[:, 8]

    return qg_pu, va_deg


# =====================================================================
# MAE Calculation Functions
# =====================================================================
def compute_mae_percentage(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE)

    Normalized by mean of absolute true values
    """
    epsilon = 1e-8
    mae = np.mean(np.abs(y_true - y_pred))
    mean_true = np.mean(np.abs(y_true)) + epsilon
    return 100.0 * mae / mean_true


def compute_mae_absolute(y_true, y_pred):
    """Calculate Mean Absolute Error (MAE)"""
    return np.mean(np.abs(y_true - y_pred))


# =====================================================================
# Cost Calculation Functions
# =====================================================================
def compute_cost_from_pg(pg_pu, cost_coeffs):
    """Calculate generation cost from pg (p.u.)"""
    cost_c2 = cost_coeffs['cost_c2']
    cost_c1 = cost_coeffs['cost_c1']
    cost_c0 = cost_coeffs['cost_c0']

    if pg_pu.ndim == 1:
        cost = np.sum(cost_c2 * pg_pu ** 2 + cost_c1 * pg_pu + cost_c0)
    else:
        cost_per_gen = (cost_c2.reshape(1, -1) * pg_pu ** 2 +
                        cost_c1.reshape(1, -1) * pg_pu +
                        cost_c0.reshape(1, -1))
        cost = np.sum(cost_per_gen, axis=1)

    return cost


def compute_cost_metrics(pg_true, pg_pred, cost_coeffs):
    """Calculate cost-related metrics"""
    cost_true = compute_cost_from_pg(pg_true, cost_coeffs)
    cost_pred = compute_cost_from_pg(pg_pred, cost_coeffs)

    cost_true_mean = np.mean(cost_true)
    cost_pred_mean = np.mean(cost_pred)
    cost_optimality_gap = np.mean((cost_pred - cost_true) / (cost_true + 1e-8)) * 100

    return {
        'cost_true_mean': cost_true_mean,
        'cost_pred_mean': cost_pred_mean,
        'cost_optimality_gap_percent': cost_optimality_gap
    }


# =====================================================================
# KEY MODIFICATION: Comprehensive evaluation with slack classification
# =====================================================================
def evaluate_acopf_predictions(
        y_pred_pg,  # Predicted pg (p.u.), may be full or only non-slack
        y_pred_vm,  # Predicted vm (p.u.)
        y_true_pg,  # True pg (p.u.), must be full (including slack)
        y_true_vm,  # True vm (p.u.)
        y_true_qg,  # True qg (p.u.)
        y_true_va_rad,  # True va (radians)
        pf_results_list,  # Power flow calculation results list
        converge_flags,  # Convergence flags list
        params,  # Parameters dictionary
        verbose=True
):
    """
    Comprehensive evaluation of ACOPF predictions (V11: Converged-only metrics)

    All violation metrics, MAE_Pg, MAE_Qg, MAE_Va and cost are computed ONLY
    over converged samples.  Non-converged samples are excluded from every
    average so that a low convergence rate does not corrupt quality metrics.
    The convergence rate itself is still computed over all samples.

    If no samples converged, all quality metrics are returned as NaN.
    """
    n_samples = len(y_pred_pg)
    n_gen     = params['general']['n_gen']
    n_buses   = params['general']['n_buses']
    base_mva  = params['general']['BASE_MVA']

    # Check if slack_gen_mask exists (backward compatibility)
    has_slack_info = 'slack_gen_mask' in params['general']

    conv_flags = np.array(converge_flags, dtype=bool)   # (n_samples,)
    n_converged = int(conv_flags.sum())
    convergence_rate = (n_converged / n_samples) * 100

    # ── converged-sample index array ────────────────────────────────────
    conv_idx = np.where(conv_flags)[0]   # indices of converged samples

    # ==================== 1. Per-sample violations (NaN for non-converged) ====================
    # calculate_single_sample_violations now returns NaN for non-converged samples.
    max_pg_viol_per_sample     = np.full(n_samples, np.nan)
    max_qg_viol_per_sample     = np.full(n_samples, np.nan)
    max_vm_viol_per_sample     = np.full(n_samples, np.nan)
    max_branch_viol_per_sample = np.full(n_samples, np.nan)

    # QG / VA arrays: only populated for converged samples; others stay NaN
    # (NaN rows are excluded from MAE via nanmean logic below)
    y_pred_qg_pf = np.full((n_samples, n_gen),   np.nan)
    y_pred_va_pf = np.full((n_samples, n_buses),  np.nan)

    for i in conv_idx:
        qg_pu, va_deg = extract_pf_results(
            pf_results_list[i], True, base_mva, n_gen, n_buses
        )
        y_pred_qg_pf[i, :] = qg_pu
        y_pred_va_pf[i, :] = va_deg

        pg_vio, qg_vio, vm_vio, branch_vio = calculate_single_sample_violations(
            pf_results_list[i], True, base_mva
        )
        max_pg_viol_per_sample[i]     = pg_vio
        max_qg_viol_per_sample[i]     = qg_vio
        max_vm_viol_per_sample[i]     = vm_vio
        max_branch_viol_per_sample[i] = branch_vio

    # ==================== 2. Full Pg from power flow (converged only) ====================
    # Non-converged rows stay NaN; MAE computed via nanmean helpers below.
    y_pred_pg_pf_full = np.full((n_samples, n_gen), np.nan)
    for i in conv_idx:
        pg_mw = pf_results_list[i][0]['gen'][:, 1]
        y_pred_pg_pf_full[i, :] = pg_mw / base_mva

    # ==================== 3. MAE helpers (NaN-aware) ====================
    def _mae_pct_conv(true, pred):
        """MAPE over converged rows only (NaN rows skipped)."""
        mask = ~np.isnan(pred[:, 0])   # row is converged iff first col is not NaN
        if mask.sum() == 0:
            return np.nan
        epsilon = 1e-8
        mae = np.mean(np.abs(true[mask] - pred[mask]))
        mean_true = np.mean(np.abs(true[mask])) + epsilon
        return 100.0 * mae / mean_true

    def _mae_abs_conv(true_deg, pred_deg):
        """MAE (degrees) over converged rows only."""
        mask = ~np.isnan(pred_deg[:, 0])
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs(true_deg[mask] - pred_deg[mask]))

    # ==================== 4. MAE_Pg (converged samples, slack-separated) ====================
    if has_slack_info:
        slack_gen_mask = params['general']['slack_gen_mask']
        mae_pg_non_slack = _mae_pct_conv(
            y_true_pg[:, ~slack_gen_mask],
            y_pred_pg_pf_full[:, ~slack_gen_mask]
        )
        mae_pg_slack = _mae_pct_conv(
            y_true_pg[:, slack_gen_mask],
            y_pred_pg_pf_full[:, slack_gen_mask]
        )
        mae_pg_all = _mae_pct_conv(y_true_pg, y_pred_pg_pf_full)
    else:
        mae_pg_all       = _mae_pct_conv(y_true_pg, y_pred_pg_pf_full)
        mae_pg_non_slack = mae_pg_all
        mae_pg_slack     = np.nan

    # ==================== 5. Other MAE metrics (converged samples) ====================
    # MAE_Vm: generator buses only (OPF decision variable = generator Vm setpoints).
    # This ensures consistent comparison across methods regardless of whether
    # the DNN predicts all-bus Vm or generator-bus Vm only.
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])

    if n_converged == 0:
        mae_vm = mae_qg = mae_va_deg = np.nan
    else:
        mae_vm  = compute_mae_percentage(
            y_true_vm[conv_idx][:, gen_bus_indices],
            y_pred_vm[conv_idx][:, gen_bus_indices]
        )
        mae_qg  = _mae_pct_conv(y_true_qg, y_pred_qg_pf)
        y_true_va_deg = y_true_va_rad * (180.0 / np.pi)
        mae_va_deg    = _mae_abs_conv(y_true_va_deg, y_pred_va_pf)

    # ==================== 6. Cost (converged samples) ====================
    if n_converged == 0:
        cost_metrics = {
            'cost_true_mean': np.nan,
            'cost_pred_mean': np.nan,
            'cost_optimality_gap_percent': np.nan,
        }
    else:
        # For cost we use the full-sample y_true_pg (ground truth doesn't
        # depend on convergence) but restrict predicted pg to converged rows.
        # We fill non-converged rows with the ground truth so they contribute
        # zero gap; then we restrict to converged rows.
        cost_metrics = compute_cost_metrics(
            y_true_pg[conv_idx],
            y_pred_pg_pf_full[conv_idx],
            params['generator']
        )

    # ==================== 7. Violation aggregates (converged samples only) ====================
    # np.nanmean ignores NaN entries → automatically excludes non-converged rows.
    mean_max_pg_viol_pu     = np.nanmean(max_pg_viol_per_sample)     if n_converged > 0 else np.nan
    mean_max_qg_viol_pu     = np.nanmean(max_qg_viol_per_sample)     if n_converged > 0 else np.nan
    mean_max_vm_viol_pu     = np.nanmean(max_vm_viol_per_sample)     if n_converged > 0 else np.nan
    mean_max_branch_viol_pu = np.nanmean(max_branch_viol_per_sample) if n_converged > 0 else np.nan

    # Slack / Non-Slack Pg violations (converged samples only)
    if has_slack_info and n_converged > 0:
        slack_gen_idx = np.where(slack_gen_mask)[0]
        slack_pg_viols     = []
        non_slack_pg_viols = []

        for i in conv_idx:
            gen    = pf_results_list[i][0]['gen']
            pg_mw  = gen[:, 1]
            pg_min = gen[:, 9]
            pg_max = gen[:, 8]
            viols_mw  = np.maximum(0, pg_min - pg_mw) + np.maximum(0, pg_mw - pg_max)
            viols_pu  = viols_mw / base_mva

            slack_pg_viols.append(
                viols_pu[slack_gen_idx[0]] if len(slack_gen_idx) > 0 else 0.0
            )
            non_slack_pg_viols.append(
                np.max(viols_pu[~slack_gen_mask]) if np.any(~slack_gen_mask) else 0.0
            )

        mean_slack_pg_viol     = float(np.mean(slack_pg_viols))
        mean_non_slack_pg_viol = float(np.mean(non_slack_pg_viols))
    elif has_slack_info:
        mean_slack_pg_viol     = np.nan
        mean_non_slack_pg_viol = np.nan
    else:
        mean_slack_pg_viol     = np.nan
        mean_non_slack_pg_viol = mean_max_pg_viol_pu

    # ==================== 8. Return all metrics ====================
    metrics = {
        # MAE metrics — converged samples only
        'mae_pg_non_slack_percent': mae_pg_non_slack,
        'mae_pg_all_percent':       mae_pg_all,
        'mae_pg_slack_percent':     mae_pg_slack,
        'mae_vm_percent':           mae_vm,
        'mae_qg_percent':           mae_qg,
        'mae_va_deg':               mae_va_deg,

        # Cost metrics — converged samples only
        'cost_true_mean':               cost_metrics['cost_true_mean'],
        'cost_pred_mean':               cost_metrics['cost_pred_mean'],
        'cost_optimality_gap_percent':  cost_metrics['cost_optimality_gap_percent'],

        # Violation metrics — converged samples only, all in p.u.
        'mean_max_pg_viol_pu':        mean_max_pg_viol_pu,
        'mean_pg_viol_non_slack_pu':  mean_non_slack_pg_viol,
        'mean_pg_viol_slack_pu':      mean_slack_pg_viol,
        'mean_max_qg_viol_pu':        mean_max_qg_viol_pu,
        'mean_max_vm_viol_pu':        mean_max_vm_viol_pu,
        'mean_max_branch_viol_pu':    mean_max_branch_viol_pu,

        # Convergence — always over all samples
        'convergence_rate_percent':   convergence_rate,
        'n_converged':                n_converged,
        'n_samples':                  n_samples,
    }

    return metrics