"""
Algebraic Power Flow Computation with Kron Reduction (FIXED VERSION)

Key fixes:
1. Correct power injection formula (P_i = Σ V_i*V_j*(G_ij*cos + B_ij*sin))
2. Proper vectorization with broadcasting
3. Robust error handling for Kron Reduction
4. Sign conventions consistent with MATPOWER/PyPower

Date: 2025-01-31 (Fixed)
"""

import torch
import numpy as np


def build_admittance_matrix(params, device='cpu'):
    """
    Build admittance matrix Y = G + jB from branch parameters

    Returns:
        G: Conductance matrix (n_buses × n_buses)
        B: Susceptance matrix (n_buses × n_buses)
    """
    n_buses = params['general']['n_buses']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Initialize complex admittance matrix
    Y = np.zeros((n_buses, n_buses), dtype=complex)

    # Extract branch parameters
    f_bus = params['branch']['f_bus']
    t_bus = params['branch']['t_bus']
    r_pu = params['branch']['r_pu']
    x_pu = params['branch']['x_pu']
    b_pu = params['branch']['b_pu']
    tap_ratio = params['branch']['tap_ratio']
    shift_deg = params['branch']['shift_deg']

    # Build Y matrix
    for k in range(len(f_bus)):
        i = bus_id_to_idx[int(f_bus[k])]
        j = bus_id_to_idx[int(t_bus[k])]

        # Branch impedance and admittance
        z = r_pu[k] + 1j * x_pu[k]
        y = 1.0 / z if abs(z) > 1e-10 else 0.0

        # Shunt admittance
        y_shunt = 1j * b_pu[k] / 2.0

        # Transformer tap ratio and phase shift
        tap = tap_ratio[k]
        shift = shift_deg[k] * np.pi / 180.0
        tap_complex = tap * np.exp(1j * shift)

        # Add to Y matrix (standard pi-model)
        if abs(tap - 1.0) < 1e-6 and abs(shift) < 1e-6:
            # No transformer
            Y[i, i] += y + y_shunt
            Y[j, j] += y + y_shunt
            Y[i, j] -= y
            Y[j, i] -= y
        else:
            # With transformer
            Y[i, i] += (y + y_shunt) / (tap ** 2)
            Y[j, j] += y + y_shunt
            Y[i, j] -= y / tap_complex
            Y[j, i] -= y / np.conj(tap_complex)

    # Convert to PyTorch tensors
    G = torch.tensor(Y.real, dtype=torch.float32, device=device)
    B = torch.tensor(Y.imag, dtype=torch.float32, device=device)

    return G, B


def compute_zib_voltages_kron(v_alpha, theta_alpha, params, G, B, device='cpu'):
    """
    Compute ZIB voltages using Kron Reduction (Equation 2)

    Solves: A * [e_β, f_β]^T = b
    where e = V*cos(θ), f = V*sin(θ)

    Returns:
        v_beta: ZIB voltage magnitudes (batch × n_zib)
        theta_beta: ZIB voltage angles (batch × n_zib)
        zib_idx: Indices of ZIB buses
        nonzib_idx: Indices of non-ZIB buses
    """
    batch_size = v_alpha.shape[0]
    n_buses = G.shape[0]

    # Get non-ZIB indices from params (computed in main script)
    if 'nonzib_indices' in params['general']:
        nonzib_list = params['general']['nonzib_indices']
    else:
        # Fallback: identify from load/gen buses
        bus_ids = [int(bid) for bid in params['general']['bus_ids']]
        load_bus_ids = set([int(bid) for bid in params['general']['load_bus_ids']])
        gen_bus_ids = set([int(bid) for bid in params['general']['gen_bus_ids']])
        nonzib_list = [idx for idx, bid in enumerate(bus_ids)
                       if bid in load_bus_ids or bid in gen_bus_ids]

    # Compute ZIB indices
    nonzib_set = set(nonzib_list)
    zib_list = [idx for idx in range(n_buses) if idx not in nonzib_set]

    nonzib_idx = torch.tensor(nonzib_list, device=device, dtype=torch.long)
    zib_idx = torch.tensor(zib_list, device=device, dtype=torch.long)

    if len(zib_list) == 0:
        # No ZIBs
        return (torch.empty(batch_size, 0, device=device),
                torch.empty(batch_size, 0, device=device),
                zib_idx, nonzib_idx)

    # Convert to Cartesian coordinates
    e_alpha = v_alpha * torch.cos(theta_alpha)
    f_alpha = v_alpha * torch.sin(theta_alpha)

    # Partition admittance matrix
    G_ba = G[zib_idx][:, nonzib_idx]
    B_ba = B[zib_idx][:, nonzib_idx]
    G_bb = G[zib_idx][:, zib_idx]
    B_bb = B[zib_idx][:, zib_idx]

    # Build linear system
    # [G_ββ  -B_ββ] [e_β]   [-(G_βα*e_α - B_βα*f_α)]
    # [B_ββ   G_ββ] [f_β] = [-(B_βα*e_α + G_βα*f_α)]
    A_top = torch.cat([G_bb, -B_bb], dim=1)
    A_bot = torch.cat([B_bb, G_bb], dim=1)
    A = torch.cat([A_top, A_bot], dim=0)

    # RHS for each sample in batch
    # rhs_e = -(e_α @ G_βα^T - f_α @ B_βα^T)
    rhs_e = -(e_alpha @ G_ba.t() - f_alpha @ B_ba.t())
    rhs_f = -(e_alpha @ B_ba.t() + f_alpha @ G_ba.t())
    rhs = torch.cat([rhs_e, rhs_f], dim=1)  # (batch, 2*n_zib)

    # Solve linear system
    try:
        # A: (2*n_zib, 2*n_zib), rhs: (batch, 2*n_zib)
        # Need to transpose for solve
        ef_beta = torch.linalg.solve(A, rhs.t()).t()  # (batch, 2*n_zib)
    except RuntimeError as e:
        print(f"⚠️ Kron Reduction solve failed: {e}")
        print(f"   A condition number: {torch.linalg.cond(A).item():.2e}")
        ef_beta = torch.zeros_like(rhs)

    # Extract e and f
    e_beta = ef_beta[:, :len(zib_list)]
    f_beta = ef_beta[:, len(zib_list):]

    # Convert back to polar
    v_beta = torch.sqrt(e_beta ** 2 + f_beta ** 2)
    theta_beta = torch.atan2(f_beta, e_beta)

    return v_beta, theta_beta, zib_idx, nonzib_idx


def compute_power_injection(v_all, theta_all, G, B):
    """
    Vectorized computation of net power injections (CORRECT FORMULA)

    Standard AC power flow equations:
    P_i = Σ_j V_i*V_j*(G_ij*cos(θ_ij) + B_ij*sin(θ_ij))
    Q_i = Σ_j V_i*V_j*(G_ij*sin(θ_ij) - B_ij*cos(θ_ij))

    Args:
        v_all: (batch × n_buses) voltage magnitudes
        theta_all: (batch × n_buses) voltage angles
        G, B: (n_buses × n_buses) admittance matrices

    Returns:
        P_inject: (batch × n_buses) active power injections
        Q_inject: (batch × n_buses) reactive power injections
    """
    batch_size, n_buses = v_all.shape

    # Expand dimensions for broadcasting
    # Shape: (batch, n_buses, 1) and (batch, 1, n_buses)
    v_i = v_all.unsqueeze(2)  # (batch, n_buses, 1)
    v_j = v_all.unsqueeze(1)  # (batch, 1, n_buses)
    theta_i = theta_all.unsqueeze(2)
    theta_j = theta_all.unsqueeze(1)

    # Angle differences θ_ij = θ_i - θ_j
    theta_ij = theta_i - theta_j  # (batch, n_buses, n_buses)

    # Voltage product V_i * V_j
    v_prod = v_i * v_j  # (batch, n_buses, n_buses)

    # Expand G and B for broadcasting
    G_exp = G.unsqueeze(0)  # (1, n_buses, n_buses)
    B_exp = B.unsqueeze(0)

    # Trigonometric terms
    cos_theta = torch.cos(theta_ij)
    sin_theta = torch.sin(theta_ij)

    # Power flow equations
    # P_ij = V_i * V_j * (G_ij*cos(θ_ij) + B_ij*sin(θ_ij))
    P_ij = v_prod * (G_exp * cos_theta + B_exp * sin_theta)

    # Q_ij = V_i * V_j * (G_ij*sin(θ_ij) - B_ij*cos(θ_ij))
    Q_ij = v_prod * (G_exp * sin_theta - B_exp * cos_theta)

    # Sum over j: P_i = Σ_j P_ij
    P_inject = torch.sum(P_ij, dim=2)  # (batch, n_buses)
    Q_inject = torch.sum(Q_ij, dim=2)  # (batch, n_buses)

    return P_inject, Q_inject


def compute_branch_power(v_all, theta_all, params, G, B, device='cpu'):
    """
    Compute branch power flows P_ij, Q_ij

    Uses vectorized computation with proper indexing
    """
    batch_size = v_all.shape[0]

    f_bus = [int(fb) for fb in params['branch']['f_bus']]
    t_bus = [int(tb) for tb in params['branch']['t_bus']]
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Get bus indices for all branches
    f_idx = [bus_id_to_idx[fb] for fb in f_bus]
    t_idx = [bus_id_to_idx[tb] for tb in t_bus]

    f_idx_t = torch.tensor(f_idx, device=device, dtype=torch.long)
    t_idx_t = torch.tensor(t_idx, device=device, dtype=torch.long)

    # Extract voltages for from/to buses
    v_f = v_all[:, f_idx_t]  # (batch, n_branches)
    v_t = v_all[:, t_idx_t]
    theta_f = theta_all[:, f_idx_t]
    theta_t = theta_all[:, t_idx_t]

    # Angle differences
    theta_ft = theta_f - theta_t  # (batch, n_branches)

    # Get admittances for branches
    g_ij = G[f_idx_t, t_idx_t]  # (n_branches,)
    b_ij = B[f_idx_t, t_idx_t]

    # Trigonometric terms
    cos_t = torch.cos(theta_ft)
    sin_t = torch.sin(theta_ft)

    # Branch power flows (from bus i perspective)
    # P_ij = -V_i*V_j*(G_ij*cos(θ_ij) + B_ij*sin(θ_ij)) + V_i^2*G_ij
    P_branch = -v_f * v_t * (g_ij * cos_t + b_ij * sin_t) + v_f ** 2 * g_ij

    # Q_ij = -V_i*V_j*(G_ij*sin(θ_ij) - B_ij*cos(θ_ij)) - V_i^2*B_ij
    Q_branch = -v_f * v_t * (g_ij * sin_t - b_ij * cos_t) - v_f ** 2 * b_ij

    return P_branch, Q_branch


def compute_algebraic_acopf(v_alpha, theta_alpha, Pd, Qd, params, G, B, device='cpu'):
    """
    Complete algebraic AC-OPF pipeline (FULLY DIFFERENTIABLE)

    Steps:
    1. Predict V_α, θ_α (non-ZIB buses)
    2. Kron Reduction → V_β, θ_β (ZIB buses)
    3. Merge → V_all, θ_all
    4. Compute injections → P_i, Q_i
    5. Compute generation → P_gi, Q_gi
    6. Compute branch flows → P_ij, Q_ij

    Returns:
        results: Dictionary with all computed variables
    """
    batch_size = v_alpha.shape[0]
    n_buses = G.shape[0]

    # Step 1: Kron Reduction for ZIB voltages
    v_beta, theta_beta, zib_idx, nonzib_idx = compute_zib_voltages_kron(
        v_alpha, theta_alpha, params, G, B, device
    )

    # Step 2: Reconstruct full voltage vectors
    v_all = torch.zeros(batch_size, n_buses, device=device)
    theta_all = torch.zeros(batch_size, n_buses, device=device)

    # Use advanced indexing to assign values
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    v_all[batch_idx, nonzib_idx] = v_alpha
    theta_all[batch_idx, nonzib_idx] = theta_alpha

    if len(zib_idx) > 0:
        v_all[batch_idx, zib_idx] = v_beta
        theta_all[batch_idx, zib_idx] = theta_beta

    # Step 3: Compute net power injections (CORRECTED FORMULA)
    P_inject, Q_inject = compute_power_injection(v_all, theta_all, G, B)

    # Step 4: Compute branch power flows
    P_branch, Q_branch = compute_branch_power(v_all, theta_all, params, G, B, device)

    # Step 5: Compute generator outputs
    gen_bus_ids = [int(gid) for gid in params['general']['gen_bus_ids']]
    load_bus_ids = [int(lid) for lid in params['general']['load_bus_ids']]
    bus_id_to_idx = params['general']['bus_id_to_idx']

    gen_indices = [bus_id_to_idx[gid] for gid in gen_bus_ids]
    load_indices = [bus_id_to_idx[lid] for lid in load_bus_ids]

    # Create full load vectors
    Pd_full = torch.zeros(batch_size, n_buses, device=device)
    Qd_full = torch.zeros(batch_size, n_buses, device=device)
    Pd_full[:, load_indices] = Pd
    Qd_full[:, load_indices] = Qd

    # Generation = Injection + Load
    # P_gi = P_i + P_di (positive convention for generation)
    Pg_all = P_inject + Pd_full
    Qg_all = Q_inject + Qd_full

    # Extract only generator buses
    Pg = Pg_all[:, gen_indices]
    Qg = Qg_all[:, gen_indices]

    # Return complete results
    return {
        'Pg': Pg,
        'Qg': Qg,
        'v_all': v_all,
        'theta_all': theta_all,
        'P_branch': P_branch,
        'Q_branch': Q_branch,
        'v_beta': v_beta,
        'theta_beta': theta_beta,
        'Pd_satisfied': Pd,
        'Qd_satisfied': Qd,
        'P_inject': P_inject,
        'Q_inject': Q_inject
    }


if __name__ == "__main__":
    print("✅ Algebraic Power Flow Module (Fixed Version)")
    print("=" * 70)
    print("Key fixes:")
    print("  1. Corrected power injection formula")
    print("  2. Proper vectorization with broadcasting")
    print("  3. Robust error handling")
    print("=" * 70)