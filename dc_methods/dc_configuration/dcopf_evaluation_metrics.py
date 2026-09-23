"""Constraint violations, cost and accuracy metrics shared by every DCOPF method."""

import numpy as np


def line_flows(pg, pd_bus, params):
    """Branch flows (n_samples, n_branches) from the exported PTDF and nodal net injections."""
    c = params['constraints']
    return (pg @ c['gen_bus_map'].T - pd_bus) @ c['ptdf'].T


def violations(pg, pd_bus, params):
    """Per-sample violations in p.u.; branch violations cover constrained branches only."""
    c = params['constraints']
    mask = c['constrained_branches']
    flows = line_flows(pg, pd_bus, params)[:, mask]
    return {
        'gen_up': np.maximum(0.0, pg - c['pg_max']),
        'gen_lo': np.maximum(0.0, c['pg_min'] - pg),
        'branch': np.maximum(0.0, np.abs(flows) - c['rate_a'][mask]),
        'balance': np.abs(pg.sum(axis=1) - pd_bus.sum(axis=1)),
    }


def compute_cost(pg, params):
    """Total generation cost per sample; coefficients are in the per-unit Pg convention of the export."""
    c = params['constraints']
    return (c['cost_c2'] * pg ** 2 + c['cost_c1'] * pg + c['cost_c0']).sum(axis=1)


def mae_percent(y_true, y_pred):
    """MAE normalized by the mean absolute true value, pooled over every element."""
    return 100.0 * np.mean(np.abs(y_true - y_pred)) / (np.mean(np.abs(y_true)) + 1e-8)


def _mean_of_max(values):
    """Mean over samples of the worst element; 0 when there are no elements to violate."""
    return float(np.mean(values.max(axis=1))) if values.shape[1] else 0.0


def evaluate_dispatch(pg_pred, pg_true, pd_bus, params):
    """Standard metrics dict for a full predicted dispatch (n_samples, n_gens)."""
    general, c = params['general'], params['constraints']
    slack, non_slack = general['slack_gen_idx'], general['non_slack_gen_idx']
    viol = violations(pg_pred, pd_bus, params)
    gen_viol = np.maximum(viol['gen_up'], viol['gen_lo'])
    # Branch violation relative to capacity, so 0.1 means a 10% overload on the worst branch
    branch_ratio = viol['branch'] / c['rate_a'][c['constrained_branches']]

    cost_true = compute_cost(pg_true, params)
    cost_pred = compute_cost(pg_pred, params)
    return {
        'mae_pg_non_slack': float(mae_percent(pg_true[:, non_slack], pg_pred[:, non_slack])),
        'mae_pg_slack': float(mae_percent(pg_true[:, slack], pg_pred[:, slack])),
        'viol_pg_non_slack': _mean_of_max(gen_viol[:, non_slack]),
        'viol_pg_slack': _mean_of_max(gen_viol[:, slack]),
        'viol_branch': _mean_of_max(branch_ratio),
        'viol_balance': float(np.mean(viol['balance'])),
        'cost_gap_percent': float(100.0 * np.mean((cost_pred - cost_true) / (np.abs(cost_true) + 1e-8))),
    }


def print_metrics(metrics):
    """Uniform result printout; timing entries are printed only when present."""
    print("\n" + "=" * 70)
    print("Test Set Results")
    print("=" * 70)
    print(f"Non-slack generators  MAE {metrics['mae_pg_non_slack']:.4f}%   "
          f"violation {metrics['viol_pg_non_slack']:.6f} p.u. (mean of max)")
    print(f"Slack generators      MAE {metrics['mae_pg_slack']:.4f}%   "
          f"violation {metrics['viol_pg_slack']:.6f} p.u. (mean of max)")
    print(f"Branch violation      {metrics['viol_branch']:.6f} x capacity (mean of max)")
    print(f"Power balance error   {metrics['viol_balance']:.3e} p.u. (mean)")
    print(f"Cost gap              {metrics['cost_gap_percent']:.4f}%")
    if 'train_time_s' in metrics:
        print(f"Training time         {metrics['train_time_s']:.2f} s")
    if 'inference_ms' in metrics:
        print(f"Inference time        {metrics['inference_ms']:.4f} ms ({metrics['inference_scope']})")
    print("=" * 70 + "\n")