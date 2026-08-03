"""
DCOPF Violation Metrics - Streamlined Version
===============================================

Core functionality for ML-DCOPF benchmark:
1. Calculate DCOPF constraint violations accurately
2. Support PyTorch and NumPy inputs
3. Unified metrics framework for all ML-DCOPF methods

Constraint Types:
- Generator limits: Pg_min ≤ Pg ≤ Pg_max
- Line flow limits: -Pl_max ≤ Pl ≤ Pl_max
- Power balance: ∑Pg = ∑Pd

Metrics:
- MAE (%): Prediction accuracy
- Viol_pg (%): Generator violations relative to capacity (Mean of Max)
- Viol_branch (%): Line violations relative to capacity (Mean of Max)
- Cost Gap (%): Economic optimality

Author: Auto-generated
Date: 2025-01-05
"""

import numpy as np
from typing import Tuple, Dict


# =====================================================================
# Core DCOPF Violation Calculation
# =====================================================================

def feasibility(
        y_pred_pg: np.ndarray,
        x_pd: np.ndarray,
        params: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate DCOPF constraint violations (strict engineering version)

    Physical Model:
    ---------------
    1. Generator constraints: Pg_min ≤ Pg ≤ Pg_max
    2. Line flow: Pl = PTDF @ (Pg_bus - Pd_bus)
    3. Line constraints: |Pl| ≤ Pl_max
    4. Power balance: ∑Pg = ∑Pd

    Parameters:
    -----------
    y_pred_pg : np.ndarray
        Predicted generator outputs, shape (n_samples, n_g), unit: p.u.
        Must be unscaled true physical values
    x_pd : np.ndarray
        Load demand, shape (n_samples, n_buses), unit: p.u.
        Must be unscaled true physical values
    params : Dict
        System parameters dictionary, contains:
        - params['constraints']['Pg_min']: shape (1, n_g) or (n_g,), p.u.
        - params['constraints']['Pg_max']: shape (1, n_g) or (n_g,), p.u.
        - params['constraints']['Pl_max']: shape (n_branches,), p.u.
        - params['constraints']['PTDF']: shape (n_buses, n_branches)
        - params['constraints']['Map_g']: shape (n_g, n_buses)

    Returns:
    --------
    gen_up_viol : np.ndarray
        Generator upper limit violations, shape (n_samples, n_g), unit: p.u.
        gen_up_viol[i,j] = max(0, Pg[i,j] - Pg_max[j])
    gen_lo_viol : np.ndarray
        Generator lower limit violations, shape (n_samples, n_g), unit: p.u.
        gen_lo_viol[i,j] = max(0, Pg_min[j] - Pg[i,j])
    line_viol : np.ndarray
        Line flow violations, shape (n_samples, n_branches), unit: p.u.
        line_viol[i,k] = max(0, |Pl[i,k]| - Pl_max[k])
    balance_err : np.ndarray
        Power balance error, shape (n_samples,), unit: p.u.
        balance_err[i] = |∑Pg[i] - ∑Pd[i]|

    Notes:
    ------
    1. All inputs must be in p.u. unit (per-unit values)
    2. No additional scaling or normalization
    3. PTDF matrix definition: Pl = PTDF^T @ P_inj, where P_inj = Pg_bus - Pd_bus
    4. Map_g is the generator-to-bus mapping matrix
    """

    # ============ Input validation ============
    assert y_pred_pg.ndim == 2, f"y_pred_pg must be 2D, got shape {y_pred_pg.shape}"
    assert x_pd.ndim == 2, f"x_pd must be 2D, got shape {x_pd.shape}"
    assert y_pred_pg.shape[0] == x_pd.shape[0], "Batch size mismatch"

    # ============ Extract parameters ============
    Pg_min = params['constraints']['Pg_min']  # (1, n_g) or (n_g,)
    Pg_max = params['constraints']['Pg_max']  # (1, n_g) or (n_g,)
    Pl_max = params['constraints']['Pl_max']  # (n_branches,)
    PTDF = params['constraints']['PTDF']  # (n_buses, n_branches)
    Map_g = params['constraints']['Map_g']  # (n_g, n_buses)

    # Ensure Pg_min and Pg_max are 1D arrays
    if Pg_min.ndim == 2:
        Pg_min = Pg_min.ravel()
    if Pg_max.ndim == 2:
        Pg_max = Pg_max.ravel()

    n_samples = y_pred_pg.shape[0]
    n_g = y_pred_pg.shape[1]
    n_buses = x_pd.shape[1]
    n_branches = PTDF.shape[1]

    # ============ 1. Generator constraint violations ============
    # Upper limit violation: max(0, Pg - Pg_max)
    gen_up_viol = np.maximum(0.0, y_pred_pg - Pg_max[np.newaxis, :])

    # Lower limit violation: max(0, Pg_min - Pg)
    gen_lo_viol = np.maximum(0.0, Pg_min[np.newaxis, :] - y_pred_pg)

    # ============ 2. Line flow calculation ============
    # Map generator output to buses: Pg_bus = Pg @ Map_g
    # Map_g: (n_g, n_buses), y_pred_pg: (n_samples, n_g)
    # Pg_bus: (n_samples, n_buses)
    Pg_bus = np.dot(y_pred_pg, Map_g)

    # Net injection at nodes: P_inj = Pg_bus - Pd_bus
    # P_inj: (n_samples, n_buses)
    P_net_injection = Pg_bus - x_pd

    # Line flows: Pl = PTDF^T @ P_inj
    # PTDF: (n_buses, n_branches), P_net_injection: (n_samples, n_buses)
    # line_flows: (n_samples, n_branches)
    line_flows = np.dot(P_net_injection, PTDF)

    # ============ 3. Line constraint violations ============
    # Line violation: max(0, |Pl| - Pl_max)
    # line_viol: (n_samples, n_branches)
    line_viol = np.maximum(0.0, np.abs(line_flows) - Pl_max[np.newaxis, :])

    # ============ 4. Power balance error ============
    # Balance error: |∑Pg - ∑Pd|
    # balance_err: (n_samples,)
    total_gen = np.sum(y_pred_pg, axis=1)  # (n_samples,)
    total_load = np.sum(x_pd, axis=1)  # (n_samples,)
    balance_err = np.abs(total_gen - total_load)

    return gen_up_viol, gen_lo_viol, line_viol, balance_err


# =====================================================================
# Cost Calculation Functions
# =====================================================================

def compute_cost(
        pg: np.ndarray,
        cost_coeffs: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Calculate generation cost

    Formula: Cost = Σ(c2*pg² + c1*pg + c0) for all generators

    Parameters:
    -----------
    pg : np.ndarray
        Generator outputs, shape (n_samples, n_g), unit: p.u.
    cost_coeffs : Dict[str, np.ndarray]
        Cost coefficient dictionary, contains:
        - 'C2': shape (n_g,), quadratic term coefficient
        - 'C1': shape (n_g,), linear term coefficient
        - 'C0': shape (n_g,), constant term

    Returns:
    --------
    cost : np.ndarray
        Total cost, shape (n_samples,), unit: $
    """
    c2 = cost_coeffs.get('C2', np.zeros(pg.shape[1]))
    c1 = cost_coeffs.get('C1', np.zeros(pg.shape[1]))
    c0 = cost_coeffs.get('C0', np.zeros(pg.shape[1]))

    # Ensure coefficients are 1D arrays
    if c2.ndim == 2:
        c2 = c2.ravel()
    if c1.ndim == 2:
        c1 = c1.ravel()
    if c0.ndim == 2:
        c0 = c0.ravel()

    # Calculate cost for each generator: c2*pg² + c1*pg + c0
    cost_per_gen = c2[np.newaxis, :] * pg ** 2 + c1[np.newaxis, :] * pg + c0[np.newaxis, :]

    # Sum across all generators to get total cost
    total_cost = np.sum(cost_per_gen, axis=1)  # (n_samples,)

    return total_cost


def compute_cost_gap_percentage(
        cost_true: np.ndarray,
        cost_pred: np.ndarray
) -> float:
    """
    Calculate cost optimality gap (Cost Gap %)

    Formula: Cost Gap% = mean((cost_pred - cost_true) / cost_true) × 100%

    Parameters:
    -----------
    cost_true : np.ndarray
        True cost, shape (n_samples,), unit: $
    cost_pred : np.ndarray
        Predicted cost, shape (n_samples,), unit: $

    Returns:
    --------
    cost_gap_pct : float
        Cost gap percentage
    """
    epsilon = 1e-8
    gap = (cost_pred - cost_true) / (np.abs(cost_true) + epsilon)
    return 100.0 * np.mean(gap)


# =====================================================================
# Branch Violation Percentage (Mean of Max)
# =====================================================================

def compute_branch_violation_pu(
        line_viol: np.ndarray,
        Pl_max: np.ndarray
) -> float:
    """
    Calculate line violation (p.u.) - Mean of Max

    Physical meaning: violation as a multiple of line capacity
    - 0.01 p.u. = 1% overload
    - 0.10 p.u. = 10% overload
    - 1.00 p.u. = 100% overload (doubled capacity)
    - 6.39 p.u. = 639% overload

    Calculation Method:
    -------------------
    1. Filter constrained lines (Pl_max < 1e10)
    2. Calculate violation ratio: viol_ratio[i,k] = line_viol[i,k] / Pl_max[k]
    3. For each sample, take maximum: max_viol_ratio[i] = max(viol_ratio[i])
    4. Calculate mean: vio_branch = mean(max_viol_ratio)

    Parameters:
    -----------
    line_viol : np.ndarray
        Line flow violations, shape (n_samples, n_branches), unit: p.u.
    Pl_max : np.ndarray
        Line capacity limits, shape (n_branches,), unit: p.u.

    Returns:
    --------
    branch_viol_pu : float
        Line violation (p.u.), Mean of Max
        Represents violation as a multiple of line capacity

    Note:
    -----
    Only calculates for constrained lines (Pl_max < 1e10)
    """
    # Filter constrained lines
    valid_indices = np.where(Pl_max < 1e10)[0]

    if len(valid_indices) == 0:
        # No line constraints
        return 0.0

    # Extract constrained line data
    line_viol_valid = line_viol[:, valid_indices]  # (n_samples, n_valid)
    Pl_max_valid = Pl_max[valid_indices]  # (n_valid,)

    # Calculate violation as ratio (not percentage!)
    viol_ratio = line_viol_valid / Pl_max_valid[np.newaxis, :]  # (n_samples, n_valid)

    # Maximum violation ratio for each sample
    max_viol_per_sample = np.max(viol_ratio, axis=1)  # (n_samples,)

    # Calculate mean
    branch_viol_pu = np.mean(max_viol_per_sample)

    return float(branch_viol_pu)


# =====================================================================
# MAE Percentage Calculation
# =====================================================================

def compute_mae_percentage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean absolute percentage error (MAE %)

    Formula: MAE% = (MAE / mean(|y_true|)) × 100%

    Parameters:
    -----------
    y_true : np.ndarray
        True values, any shape
    y_pred : np.ndarray
        Predicted values, same shape as y_true

    Returns:
    --------
    mae_pct : float
        MAE percentage
    """
    epsilon = 1e-8
    mae = np.mean(np.abs(y_true - y_pred))
    mean_true = np.mean(np.abs(y_true)) + epsilon
    return 100.0 * mae / mean_true
