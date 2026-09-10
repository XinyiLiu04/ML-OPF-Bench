# -*- coding: utf-8 -*-
"""ACOPF evaluation metrics. Quality metrics are averaged over converged samples only."""

import numpy as np


def calculate_single_sample_violations(r1_pf, is_converged, base_mva):
    """Max Pg/Qg/Vm/branch violation of one sample in p.u.; NaN if the power flow diverged."""
    if not is_converged:
        # NaN rather than a sentinel value: a sentinel would dominate the mean
        # whenever the convergence rate is low.
        return np.nan, np.nan, np.nan, np.nan

    gen = r1_pf[0]['gen']
    bus = r1_pf[0]['bus']
    branch = r1_pf[0]['branch']

    pg_viol_mw = (np.maximum(0, gen[:, 9] - gen[:, 1]) +
                  np.maximum(0, gen[:, 1] - gen[:, 8]))
    pg_viol_pu = pg_viol_mw / base_mva
    max_pg_viol_pu = np.max(pg_viol_pu) if pg_viol_pu.size > 0 else 0.0

    qg_viol_mvar = (np.maximum(0, gen[:, 4] - gen[:, 2]) +
                    np.maximum(0, gen[:, 2] - gen[:, 3]))
    qg_viol_pu = qg_viol_mvar / base_mva
    max_qg_viol_pu = np.max(qg_viol_pu) if qg_viol_pu.size > 0 else 0.0

    vm_viol_pu = (np.maximum(0, bus[:, 12] - bus[:, 7]) +
                  np.maximum(0, bus[:, 7] - bus[:, 11]))
    max_vm_viol_pu = np.max(vm_viol_pu) if vm_viol_pu.size > 0 else 0.0

    # Branch violation is relative: 1.0 means 100% over the rating
    rate_a_mva = branch[:, 5]
    limit_idx = (rate_a_mva > 0) & (rate_a_mva < 9000)
    max_branch_viol_pu = 0.0

    if np.any(limit_idx):
        Ff_MVA = np.abs(branch[limit_idx, 13] + 1j * branch[limit_idx, 14])
        Ft_MVA = np.abs(branch[limit_idx, 15] + 1j * branch[limit_idx, 16])
        rate_a_MVA = rate_a_mva[limit_idx]

        all_branch_viols = np.concatenate([
            np.maximum(0, (Ff_MVA / rate_a_MVA) - 1),
            np.maximum(0, (Ft_MVA / rate_a_MVA) - 1)
        ])
        max_branch_viol_pu = np.max(all_branch_viols) if all_branch_viols.size > 0 else 0.0

    return max_pg_viol_pu, max_qg_viol_pu, max_vm_viol_pu, max_branch_viol_pu


def extract_pf_results(r1_pf, is_converged, base_mva, n_gen, n_buses):
    """Pull Qg (p.u.) and Va (degrees) out of a power flow result."""
    if not is_converged:
        return np.zeros(n_gen), np.zeros(n_buses)

    return r1_pf[0]['gen'][:, 2] / base_mva, r1_pf[0]['bus'][:, 8]


def compute_mae_percentage(y_true, y_pred):
    """MAE normalized by the mean absolute true value, in percent."""
    epsilon = 1e-8
    mae = np.mean(np.abs(y_true - y_pred))
    mean_true = np.mean(np.abs(y_true)) + epsilon
    return 100.0 * mae / mean_true


def compute_cost_from_pg(pg_pu, cost_coeffs):
    """Quadratic generation cost from Pg (p.u.)."""
    cost_c2 = cost_coeffs['cost_c2']
    cost_c1 = cost_coeffs['cost_c1']
    cost_c0 = cost_coeffs['cost_c0']

    if pg_pu.ndim == 1:
        return np.sum(cost_c2 * pg_pu ** 2 + cost_c1 * pg_pu + cost_c0)

    cost_per_gen = (cost_c2.reshape(1, -1) * pg_pu ** 2 +
                    cost_c1.reshape(1, -1) * pg_pu +
                    cost_c0.reshape(1, -1))
    return np.sum(cost_per_gen, axis=1)


def compute_cost_metrics(pg_true, pg_pred, cost_coeffs):
    """True cost, predicted cost and mean relative optimality gap."""
    cost_true = compute_cost_from_pg(pg_true, cost_coeffs)
    cost_pred = compute_cost_from_pg(pg_pred, cost_coeffs)

    return {
        'cost_true_mean': np.mean(cost_true),
        'cost_pred_mean': np.mean(cost_pred),
        'cost_optimality_gap_percent': np.mean((cost_pred - cost_true) / (cost_true + 1e-8)) * 100
    }


def evaluate_acopf_predictions(
        y_pred_pg,
        y_pred_vm,
        y_true_pg,
        y_true_vm,
        y_true_qg,
        y_true_va_rad,
        pf_results_list,
        converge_flags,
        params,
        verbose=True
):
    """Accuracy, violation and cost metrics over converged samples, plus per-sample arrays."""
    n_samples = len(y_pred_pg)
    n_gen = params['general']['n_gen']
    n_buses = params['general']['n_buses']
    base_mva = params['general']['BASE_MVA']
    slack_gen_mask = params['general']['slack_gen_mask']

    conv_flags = np.array(converge_flags, dtype=bool)
    conv_idx = np.where(conv_flags)[0]
    n_converged = int(conv_flags.sum())
    convergence_rate = (n_converged / n_samples) * 100

    # Non-converged rows stay NaN throughout and are dropped from every average
    max_pg_viol_per_sample = np.full(n_samples, np.nan)
    max_qg_viol_per_sample = np.full(n_samples, np.nan)
    max_vm_viol_per_sample = np.full(n_samples, np.nan)
    max_branch_viol_per_sample = np.full(n_samples, np.nan)

    y_pred_qg_pf = np.full((n_samples, n_gen), np.nan)
    y_pred_va_pf = np.full((n_samples, n_buses), np.nan)
    y_pred_pg_pf_full = np.full((n_samples, n_gen), np.nan)

    for i in conv_idx:
        qg_pu, va_deg = extract_pf_results(pf_results_list[i], True, base_mva, n_gen, n_buses)
        y_pred_qg_pf[i, :] = qg_pu
        y_pred_va_pf[i, :] = va_deg

        # Pg is taken from the power flow solution, so the slack generator carries
        # whatever imbalance the predicted setpoints left behind
        y_pred_pg_pf_full[i, :] = pf_results_list[i][0]['gen'][:, 1] / base_mva

        (max_pg_viol_per_sample[i],
         max_qg_viol_per_sample[i],
         max_vm_viol_per_sample[i],
         max_branch_viol_per_sample[i]) = calculate_single_sample_violations(
            pf_results_list[i], True, base_mva)

    def _mae_pct_conv(true, pred):
        """MAE percentage over converged rows, identified by a non-NaN first column."""
        mask = ~np.isnan(pred[:, 0])
        if mask.sum() == 0:
            return np.nan
        mae = np.mean(np.abs(true[mask] - pred[mask]))
        mean_true = np.mean(np.abs(true[mask])) + 1e-8
        return 100.0 * mae / mean_true

    def _mae_abs_conv(true_deg, pred_deg):
        """MAE in degrees over converged rows."""
        mask = ~np.isnan(pred_deg[:, 0])
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs(true_deg[mask] - pred_deg[mask]))

    mae_pg_non_slack = _mae_pct_conv(y_true_pg[:, ~slack_gen_mask],
                                     y_pred_pg_pf_full[:, ~slack_gen_mask])
    mae_pg_slack = _mae_pct_conv(y_true_pg[:, slack_gen_mask],
                                 y_pred_pg_pf_full[:, slack_gen_mask])
    mae_pg_all = _mae_pct_conv(y_true_pg, y_pred_pg_pf_full)

    # MAE_Vm is restricted to generator buses so that methods predicting all-bus Vm
    # and methods predicting only generator Vm remain comparable
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])

    if n_converged == 0:
        mae_vm = mae_qg = mae_va_deg = np.nan
        cost_metrics = {
            'cost_true_mean': np.nan,
            'cost_pred_mean': np.nan,
            'cost_optimality_gap_percent': np.nan,
        }
    else:
        mae_vm = compute_mae_percentage(
            y_true_vm[conv_idx][:, gen_bus_indices],
            y_pred_vm[conv_idx][:, gen_bus_indices]
        )
        mae_qg = _mae_pct_conv(y_true_qg, y_pred_qg_pf)
        mae_va_deg = _mae_abs_conv(y_true_va_rad * (180.0 / np.pi), y_pred_va_pf)
        cost_metrics = compute_cost_metrics(
            y_true_pg[conv_idx],
            y_pred_pg_pf_full[conv_idx],
            params['generator']
        )

    mean_max_pg_viol_pu = np.nanmean(max_pg_viol_per_sample) if n_converged > 0 else np.nan
    mean_max_qg_viol_pu = np.nanmean(max_qg_viol_per_sample) if n_converged > 0 else np.nan
    mean_max_vm_viol_pu = np.nanmean(max_vm_viol_per_sample) if n_converged > 0 else np.nan
    mean_max_branch_viol_pu = np.nanmean(max_branch_viol_per_sample) if n_converged > 0 else np.nan

    # Slack and non-slack Pg violations are reported separately: the slack generator
    # absorbs the power balance error, so its violation reflects a different failure mode
    slack_gen_idx = np.where(slack_gen_mask)[0]
    viol_pg_slack_per_sample = np.full(n_samples, np.nan)
    viol_pg_nonslack_per_sample = np.full(n_samples, np.nan)

    for i in conv_idx:
        gen = pf_results_list[i][0]['gen']
        v = (np.maximum(0, gen[:, 9] - gen[:, 1]) +
             np.maximum(0, gen[:, 1] - gen[:, 8])) / base_mva
        viol_pg_slack_per_sample[i] = v[slack_gen_idx[0]] if len(slack_gen_idx) > 0 else 0.0
        viol_pg_nonslack_per_sample[i] = np.max(v[~slack_gen_mask]) if np.any(~slack_gen_mask) else 0.0

    mean_slack_pg_viol = np.nanmean(viol_pg_slack_per_sample) if n_converged > 0 else np.nan
    mean_non_slack_pg_viol = np.nanmean(viol_pg_nonslack_per_sample) if n_converged > 0 else np.nan

    metrics = {
        'mae_pg_non_slack_percent': mae_pg_non_slack,
        'mae_pg_all_percent': mae_pg_all,
        'mae_pg_slack_percent': mae_pg_slack,
        'mae_vm_percent': mae_vm,
        'mae_qg_percent': mae_qg,
        'mae_va_deg': mae_va_deg,

        'cost_true_mean': cost_metrics['cost_true_mean'],
        'cost_pred_mean': cost_metrics['cost_pred_mean'],
        'cost_optimality_gap_percent': cost_metrics['cost_optimality_gap_percent'],

        'mean_max_pg_viol_pu': mean_max_pg_viol_pu,
        'mean_pg_viol_non_slack_pu': mean_non_slack_pg_viol,
        'mean_pg_viol_slack_pu': mean_slack_pg_viol,
        'mean_max_qg_viol_pu': mean_max_qg_viol_pu,
        'mean_max_vm_viol_pu': mean_max_vm_viol_pu,
        'mean_max_branch_viol_pu': mean_max_branch_viol_pu,

        # Convergence rate is always over all samples, so readers can judge the
        # coverage of the metrics above
        'convergence_rate_percent': convergence_rate,
        'n_converged': n_converged,
        'n_samples': n_samples,
    }

    return metrics