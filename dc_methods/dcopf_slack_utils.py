"""
dcopf_slack_utils.py
================================

Automatically identify slack bus and slack generators
Provide helper functions for non-slack prediction and full Pg reconstruction

"""

import numpy as np
from typing import Dict, Tuple


def identify_slack_bus_and_gens(params: Dict) -> Dict:
    """
    Automatically identify slack bus and slack generators from PTDF matrix and Map_g
    No need to modify Julia scripts, all processing done in Python

    Parameters:
    -----------
    params : Dict
        System parameters dictionary, must contain:
        - params['constraints']['PTDF']: PTDF matrix (n_buses, n_branches)
        - params['constraints']['Map_g']: Generator-bus mapping (n_g, n_buses)
        - params['general']['n_g']: Number of generators
        - params['general']['n_buses']: Number of buses
        - params['general']['g_bus']: Generator connected bus IDs
        - params['general']['bus_ids']: Real bus IDs array (supports non-contiguous numbering)

    Returns:
    --------
    slack_info : Dict
        Dictionary containing slack information:
        - 'slack_bus_idx': Slack bus index (0-based)
        - 'slack_bus_id': Slack bus ID (inferred from parameters)
        - 'slack_gen_indices': Slack generator indices array (0-based)
        - 'slack_gen_ids': Slack generator IDs array
        - 'non_slack_gen_indices': Non-slack generator indices array (0-based)
        - 'n_slack_gens': Number of slack generators
        - 'n_non_slack_gens': Number of non-slack generators
    """
    PTDF = params['constraints']['PTDF']  # (n_buses, n_branches)
    Map_g = params['constraints']['Map_g']  # (n_g, n_buses)
    n_g = params['general']['n_g']
    n_buses = params['general']['n_buses']
    g_bus = params['general']['g_bus']  # Generator connected bus IDs
    bus_ids = params['general']['bus_ids']  # Real bus IDs (may be non-contiguous, e.g. case300)

    # ============ Method 1: Identify slack bus through PTDF matrix ============
    # The PTDF column corresponding to slack bus should be all zeros (reference angle)
    ptdf_col_norms = np.linalg.norm(PTDF, axis=1)  # L2 norm of each column
    slack_bus_idx = np.argmin(ptdf_col_norms)  # Column with minimum norm

    # Validation: Check if this column is indeed close to 0
    if ptdf_col_norms[slack_bus_idx] > 1e-6:
        print(f"[WARNING] PTDF column norm is large: {ptdf_col_norms[slack_bus_idx]:.2e}")
        print(f"[WARNING] Identified slack bus may not be accurate, please check")

    # Look up the real bus ID from bus_ids array (handles sparse numbering like case300)
    slack_bus_id = int(bus_ids[slack_bus_idx])

    print(f"\nIdentified Slack Bus:")
    print(f"  Slack bus index: {slack_bus_idx} (0-based)")
    print(f"  Slack bus ID: {slack_bus_id}")
    print(f"  PTDF column norm: {ptdf_col_norms[slack_bus_idx]:.2e}")

    # ============ Method 2: Find generators connected to slack bus ============
    # Map_g[i, j] > 0 means generator i is connected to bus j
    slack_gen_mask = Map_g[:, slack_bus_idx] > 0
    slack_gen_indices = np.where(slack_gen_mask)[0]
    non_slack_gen_indices = np.where(~slack_gen_mask)[0]

    # Get slack generator IDs
    slack_gen_ids = g_bus[slack_gen_indices]

    print(f"\nSlack Generators:")
    print(f"  Count: {len(slack_gen_indices)}")
    print(f"  Indices (0-based): {slack_gen_indices}")
    print(f"  IDs: {slack_gen_ids}")
    print(f"  Non-Slack generator count: {len(non_slack_gen_indices)}")

    # ============ Assemble return results ============
    slack_info = {
        'slack_bus_idx': int(slack_bus_idx),
        'slack_bus_id': int(slack_bus_id),
        'slack_gen_indices': slack_gen_indices,
        'slack_gen_ids': slack_gen_ids,
        'non_slack_gen_indices': non_slack_gen_indices,
        'n_slack_gens': len(slack_gen_indices),
        'n_non_slack_gens': len(non_slack_gen_indices)
    }

    return slack_info


def update_params_with_slack_info(params: Dict, slack_info: Dict) -> Dict:
    """
    Add identified slack information to params dictionary

    Parameters:
    -----------
    params : Dict
        Original parameters dictionary
    slack_info : Dict
        Slack information obtained from identify_slack_bus_and_gens

    Returns:
    --------
    params : Dict
        Updated parameters dictionary
    """
    params['general']['slack_bus_idx'] = slack_info['slack_bus_idx']
    params['general']['slack_bus_id'] = slack_info['slack_bus_id']
    params['general']['slack_gen_indices'] = slack_info['slack_gen_indices']
    params['general']['slack_gen_ids'] = slack_info['slack_gen_ids']
    params['general']['non_slack_gen_indices'] = slack_info['non_slack_gen_indices']
    params['general']['n_slack_gens'] = slack_info['n_slack_gens']
    params['general']['n_g_non_slack'] = slack_info['n_non_slack_gens']

    return params


def reconstruct_full_pg(
        pg_non_slack: np.ndarray,
        pd_total: np.ndarray,
        params: Dict
) -> np.ndarray:
    """
    Reconstruct full Pg vector from non-slack generator outputs

    Core Principle:
    ---------------
    Slack generator output is automatically determined by power balance constraint:
        Pg_slack = Σ Pd - Σ Pg_non_slack

    If there are multiple slack generators, divide slack power evenly

    Parameters:
    -----------
    pg_non_slack : np.ndarray
        Non-slack generator outputs, shape (n_samples, n_g_non_slack), p.u.
    pd_total : np.ndarray
        Total load, shape (n_samples,), p.u.
    params : Dict
        System parameters dictionary, must contain:
        - params['general']['n_g']: Total number of generators
        - params['general']['slack_gen_indices']: Slack generator indices (0-based)
        - params['general']['non_slack_gen_indices']: Non-slack generator indices (0-based)

    Returns:
    --------
    pg_full : np.ndarray
        Full generator output vector, shape (n_samples, n_g), p.u.

    Example:
    --------
    >>> pg_non_slack = np.array([[0.5, 0.6, 0.7]])  # 3 non-slack generators
    >>> pd_total = np.array([2.0])  # Total load
    >>> params = {
    ...     'general': {
    ...         'n_g': 4,
    ...         'slack_gen_indices': np.array([0]),
    ...         'non_slack_gen_indices': np.array([1, 2, 3])
    ...     }
    ... }
    >>> pg_full = reconstruct_full_pg(pg_non_slack, pd_total, params)
    >>> print(pg_full)  # [[0.2, 0.5, 0.6, 0.7]]
    >>> # Verification: 0.2 + 0.5 + 0.6 + 0.7 = 2.0 ✓
    """
    n_samples = pg_non_slack.shape[0]
    n_g = params['general']['n_g']
    slack_indices = params['general']['slack_gen_indices']
    non_slack_indices = params['general']['non_slack_gen_indices']

    # Initialize full Pg vector
    pg_full = np.zeros((n_samples, n_g), dtype=pg_non_slack.dtype)

    # Fill non-slack generators
    pg_full[:, non_slack_indices] = pg_non_slack

    # Calculate total slack generator output (power balance)
    pg_non_slack_total = pg_non_slack.sum(axis=1)
    pg_slack_total = pd_total - pg_non_slack_total

    # If there are multiple slack generators, divide evenly
    if len(slack_indices) > 0:
        pg_slack_per_gen = pg_slack_total / len(slack_indices)
        for idx in slack_indices:
            pg_full[:, idx] = pg_slack_per_gen

    return pg_full


def compute_detailed_mae(
        y_true_all: np.ndarray,
        y_pred_non_slack: np.ndarray,
        y_pred_all: np.ndarray,
        params: Dict
) -> Dict[str, float]:
    """
    Calculate detailed MAE metrics (non-slack, slack, all)

    Parameters:
    -----------
    y_true_all : np.ndarray
        True values (all generators), shape (n_samples, n_g), p.u.
    y_pred_non_slack : np.ndarray
        Predicted values (non-slack generators), shape (n_samples, n_g_non_slack), p.u.
    y_pred_all : np.ndarray
        Predicted values (all generators, including reconstructed slack), shape (n_samples, n_g), p.u.
    params : Dict
        System parameters

    Returns:
    --------
    mae_dict : Dict[str, float]
        Contains three MAE values (percentage):
        - 'mae_non_slack': MAE for non-slack generators
        - 'mae_slack': MAE for slack generators only
        - 'mae_all': MAE for all generators
    """
    slack_indices = params['general']['slack_gen_indices']
    non_slack_indices = params['general']['non_slack_gen_indices']

    # Extract true values
    y_true_non_slack = y_true_all[:, non_slack_indices]
    y_true_slack = y_true_all[:, slack_indices]

    # Extract slack portion from predictions
    y_pred_slack = y_pred_all[:, slack_indices]

    # Calculate MAE (%)
    def mae_percent(y_true, y_pred):
        epsilon = 1e-8
        mae = np.mean(np.abs(y_true - y_pred))
        mean_true = np.mean(np.abs(y_true)) + epsilon
        return 100.0 * mae / mean_true

    mae_dict = {
        'mae_non_slack': mae_percent(y_true_non_slack, y_pred_non_slack),
        'mae_slack': mae_percent(y_true_slack, y_pred_slack) if len(slack_indices) > 0 else 0.0,
        'mae_all': mae_percent(y_true_all, y_pred_all)
    }

    return mae_dict


def compute_pg_violation_pu(
        gen_up_viol: np.ndarray,
        gen_lo_viol: np.ndarray,
        indices: np.ndarray
) -> float:
    """
    Calculate generator violation in p.u. (Mean of Max)

    Physical meaning:
    -----------------
    - 0.01 p.u. = 1% of base power
    - 0.10 p.u. = 10% of base power (serious!)
    - Values are directly comparable across different cases

    Formula:
    --------
    mean_max_viol = mean over samples [max over generators (viol_pu)]

    Parameters:
    -----------
    gen_up_viol : np.ndarray
        Generator upper limit violations, shape (n_samples, n_g), p.u.
    gen_lo_viol : np.ndarray
        Generator lower limit violations, shape (n_samples, n_g), p.u.
    indices : np.ndarray
        Generator indices to consider (0-based)

    Returns:
    --------
    viol_pu : float
        Generator violation in p.u. (Mean of Max)
    """
    if len(indices) == 0:
        return 0.0

    # Extract violations for specified generators
    up_viol = gen_up_viol[:, indices]  # (n_samples, n_indices)
    lo_viol = gen_lo_viol[:, indices]

    # For each sample, take max across generators (already in p.u.)
    max_up_per_sample = np.max(up_viol, axis=1)  # (n_samples,)
    max_lo_per_sample = np.max(lo_viol, axis=1)
    max_viol_per_sample = np.maximum(max_up_per_sample, max_lo_per_sample)

    # Calculate mean (Mean of Max) - no conversion needed!
    return float(np.mean(max_viol_per_sample))


def compute_detailed_pg_violations_pu(
        gen_up_viol: np.ndarray,
        gen_lo_viol: np.ndarray,
        params: Dict
) -> Dict[str, float]:
    """
    Calculate detailed generator violation metrics in p.u. (non-slack, slack)

    Parameters:
    -----------
    gen_up_viol : np.ndarray
        Generator upper limit violations, shape (n_samples, n_g), p.u.
    gen_lo_viol : np.ndarray
        Generator lower limit violations, shape (n_samples, n_g), p.u.
    params : Dict
        System parameters

    Returns:
    --------
    viol_dict : Dict[str, float]
        Contains violation values in p.u. (Mean of Max):
        - 'viol_non_slack': Non-slack generator violations
        - 'viol_slack': Slack generator violations only

    Note:
    -----
    No baseMVA needed! Values are directly comparable across cases.
    """
    slack_indices = params['general']['slack_gen_indices']
    non_slack_indices = params['general']['non_slack_gen_indices']

    # Calculate violations for each category
    viol_dict = {
        'viol_non_slack': compute_pg_violation_pu(
            gen_up_viol, gen_lo_viol, non_slack_indices
        ),
        'viol_slack': compute_pg_violation_pu(
            gen_up_viol, gen_lo_viol, slack_indices
        )
    }

    return viol_dict