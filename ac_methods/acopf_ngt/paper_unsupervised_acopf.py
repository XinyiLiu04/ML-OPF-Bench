import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

try:
    import acopf_config
    from acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        prepare_data_splits,
        load_generalization_test_data,
        load_api_test_data,
        DataMode,
    )
    from algebraic_power_flow import (
        build_admittance_matrix,
        compute_algebraic_acopf,
    )
    from acopf_violation_metrics import evaluate_acopf_predictions
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Neural Network  (Sec. IV-A)
#     Fully-connected, ReLU hidden, Sigmoid output.
#     Output ∈ (0,1)^{2·|N_α|} → denormalised to (V̂_α, θ̂_α).
# ─────────────────────────────────────────────────────────────────────────────

class DeepOPF_NGT(nn.Module):
    """DNN: (P_d, Q_d) → (V̂_α, θ̂_α) for non-ZIB buses."""

    def __init__(self, input_size: int, output_size: int,
                 hidden_sizes: list = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, output_size), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Denormalisation  (Sec. IV-A)
#     DNN output ∈ (0,1) → physical (V̂_α, θ̂_α).
#     Voltage magnitudes: linear map to [V_i_min, V_i_max].
#     Voltage angles    : linear map to [θ_ij_min, θ_ij_max] per bus.
#     Paper uses branch angle limits (1h); we apply them per-bus using
#     the system-wide maximum |θ_ij| as a proxy, consistent with the
#     paper's implementation detail (Sec. IV-A, ±30° commonly used).
# ─────────────────────────────────────────────────────────────────────────────

def denormalise_output(y_norm: torch.Tensor,
                       params: dict,
                       theta_max_rad: float = np.pi / 6.0):
    """
    Map DNN output (0,1) to physical (V̂_α, θ̂_α).

    Parameters
    ----------
    y_norm       : (batch, 2·|N_α|)  DNN sigmoid output
    params       : system parameter dict
    theta_max_rad: maximum absolute angle (default π/6 = 30°)

    Returns
    -------
    v_alpha   : (batch, |N_α|)  voltage magnitudes [p.u.]
    theta_alpha: (batch, |N_α|) voltage angles [rad]
    """
    nonzib_idx = params['general']['nonzib_indices']
    n_nz = len(nonzib_idx)

    vm_min = torch.as_tensor(params['bus']['vm_min'],
                              dtype=torch.float32, device=y_norm.device)
    vm_max = torch.as_tensor(params['bus']['vm_max'],
                              dtype=torch.float32, device=y_norm.device)
    v_min = vm_min[nonzib_idx]
    v_max = vm_max[nonzib_idx]

    v_norm     = y_norm[:, :n_nz]
    theta_norm = y_norm[:, n_nz:]

    # V̂ ∈ [V_min, V_max]
    v_alpha = v_min + v_norm * (v_max - v_min)
    # θ̂ ∈ [-theta_max, +theta_max]
    theta_alpha = (theta_norm - 0.5) * 2.0 * theta_max_rad

    return v_alpha, theta_alpha


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Loss Function  (Sec. IV-A, Eqs. 3–9)
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss_terms(results: dict,
                       Pd: torch.Tensor,
                       Qd: torch.Tensor,
                       params: dict,
                       device: torch.device):
    """
    Compute each loss term from Eqs. (3)–(9).

    results : output of compute_algebraic_acopf
              keys: Pg, Qg, v_all, theta_all, P_branch, Q_branch,
                    Pd_pred, Qd_pred   (predicted loads at load buses)
    Pd, Qd  : demanded active/reactive loads  (batch, n_loads)  [p.u.]

    Returns dict with keys: L_obj, L_g, L_Sl, L_theta, L_z, L_d
    """
    Pg        = results['Pg']          # (batch, n_gen)
    Qg        = results['Qg']          # (batch, n_gen)
    v_all     = results['v_all']       # (batch, n_buses)
    theta_all = results['theta_all']   # (batch, n_buses)
    P_branch  = results['P_branch']    # (batch, n_branches)
    Q_branch  = results['Q_branch']    # (batch, n_branches)

    gen_p  = params['generator']
    bus_p  = params['bus']
    br_p   = params['branch']
    gen_p_all = params['general']

    pg_min = torch.as_tensor(gen_p['pg_min'], dtype=torch.float32, device=device).squeeze()
    pg_max = torch.as_tensor(gen_p['pg_max'], dtype=torch.float32, device=device).squeeze()
    qg_min = torch.as_tensor(gen_p['qg_min'], dtype=torch.float32, device=device).squeeze()
    qg_max = torch.as_tensor(gen_p['qg_max'], dtype=torch.float32, device=device).squeeze()
    c2     = torch.as_tensor(gen_p['cost_c2'], dtype=torch.float32, device=device)
    c1     = torch.as_tensor(gen_p['cost_c1'], dtype=torch.float32, device=device)
    c0     = torch.as_tensor(gen_p['cost_c0'], dtype=torch.float32, device=device)

    vm_min = torch.as_tensor(bus_p['vm_min'], dtype=torch.float32, device=device)
    vm_max = torch.as_tensor(bus_p['vm_max'], dtype=torch.float32, device=device)
    rate_a = torch.as_tensor(br_p['rate_a'],  dtype=torch.float32, device=device)

    # ── L_obj  Eq. (1a) / (3) ────────────────────────────────────────────────
    # Σ_i C_i(P̂_gi) using the Pg reconstructed from power balance (raw, not clamped)
    L_obj = torch.mean(torch.sum(c2 * Pg ** 2 + c1 * Pg + c0, dim=1))

    # ── L_g  Eq. (5)  generator capacity violations ───────────────────────────
    pg_viol = (torch.relu(Pg - pg_max) ** 2 +
               torch.relu(pg_min - Pg) ** 2)
    qg_viol = (torch.relu(Qg - qg_max) ** 2 +
               torch.relu(qg_min - Qg) ** 2)
    L_g = torch.mean(torch.sum(pg_viol + qg_viol, dim=1))

    # ── L_Sl  Eq. (6)  branch apparent power violations ──────────────────────
    S_branch  = torch.sqrt(P_branch ** 2 + Q_branch ** 2 + 1e-12)
    valid_br  = (rate_a > 1e-5) & (rate_a < 9000.0)
    if valid_br.any():
        s_viol = torch.relu(S_branch[:, valid_br] - rate_a[valid_br]) ** 2
        L_Sl   = torch.mean(torch.sum(s_viol, dim=1))
    else:
        L_Sl = torch.tensor(0.0, device=device)

    # ── L_theta  Eq. (7)  branch angle difference violations ─────────────────
    f_bus        = br_p['f_bus']
    t_bus        = br_p['t_bus']
    bus_id_to_idx = gen_p_all['bus_id_to_idx']

    # Retrieve per-branch angle limits if available; else use ±30°
    theta_lb = br_p.get('theta_min', None)
    theta_ub = br_p.get('theta_max', None)
    default_max = 30.0 * np.pi / 180.0

    theta_diffs = []
    for k in range(len(f_bus)):
        i_idx = bus_id_to_idx[int(f_bus[k])]
        j_idx = bus_id_to_idx[int(t_bus[k])]
        diff  = theta_all[:, i_idx] - theta_all[:, j_idx]
        theta_diffs.append(diff)

    if theta_diffs:
        theta_diff = torch.stack(theta_diffs, dim=1)   # (batch, n_branches)
        if theta_lb is not None and theta_ub is not None:
            lb = torch.as_tensor(theta_lb, dtype=torch.float32, device=device)
            ub = torch.as_tensor(theta_ub, dtype=torch.float32, device=device)
            viol = (torch.relu(theta_diff - ub) ** 2 +
                    torch.relu(lb - theta_diff) ** 2)
        else:
            viol = torch.relu(torch.abs(theta_diff) - default_max) ** 2
        L_theta = torch.mean(torch.sum(viol, dim=1))
    else:
        L_theta = torch.tensor(0.0, device=device)

    # ── L_z  Eq. (8)  ZIB voltage magnitude violations ───────────────────────
    zib_mask = gen_p_all['zib_mask']          # (n_buses,) bool
    zib_idx  = np.where(zib_mask)[0]
    if len(zib_idx) > 0:
        v_zib   = v_all[:, zib_idx]           # (batch, n_ZIB)
        vmin_z  = vm_min[zib_idx]
        vmax_z  = vm_max[zib_idx]
        z_viol  = (torch.relu(v_zib - vmax_z) ** 2 +
                   torch.relu(vmin_z - v_zib) ** 2)
        L_z = torch.mean(torch.sum(z_viol, dim=1))
    else:
        L_z = torch.tensor(0.0, device=device)

    # ── L_d  Eq. (9)  load satisfaction  ─────────────────────────────────────
    # L_d = Σ_{i∈N_L\N_Z} [(P̂_di - P_di)² + (Q̂_di - Q_di)²]
    # P̂_di, Q̂_di are the predicted loads derived from bus voltage via eq.(1e).
    # The algebraic_power_flow module returns Pd_pred / Qd_pred at load buses.
    if 'Pd_pred' in results and 'Qd_pred' in results:
        Pd_pred = results['Pd_pred']   # (batch, n_loads)
        Qd_pred = results['Qd_pred']   # (batch, n_loads)
        load_err = (Pd_pred - Pd) ** 2 + (Qd_pred - Qd) ** 2
        L_d = torch.mean(torch.sum(load_err, dim=1))
    else:
        # Fallback: scalar active power balance  ΣP_g ≈ ΣP_d
        L_d = torch.mean((torch.sum(Pg, dim=1) - torch.sum(Pd, dim=1)) ** 2)

    return {
        'L_obj':   L_obj,
        'L_g':     L_g,
        'L_Sl':    L_Sl,
        'L_theta': L_theta,
        'L_z':     L_z,
        'L_d':     L_d,
    }


def compute_total_loss(loss_dict: dict, coeffs: dict) -> torch.Tensor:
    """L = k_obj·L_obj + k_g·L_g + k_Sl·L_Sl + k_θ·L_θ + k_z·L_z + k_d·L_d  Eq.(3)/(10)"""
    return (coeffs['k_obj']   * loss_dict['L_obj']
          + coeffs['k_g']     * loss_dict['L_g']
          + coeffs['k_Sl']    * loss_dict['L_Sl']
          + coeffs['k_theta'] * loss_dict['L_theta']
          + coeffs['k_z']     * loss_dict['L_z']
          + coeffs['k_d']     * loss_dict['L_d'])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Adaptive weight update  (Algorithm 1, Eq. 12)
#
#     k_i^t = min( k_obj · L_obj / L_i,  k̄_i )    for t > 1
#
#     Computed per mini-batch (as described in paper).
#     First epoch uses fixed initial values.
# ─────────────────────────────────────────────────────────────────────────────

def update_coefficients(coeffs: dict,
                        loss_dict: dict,
                        k_upper: dict):
    """
    Paper Algorithm 1 / Eq. (12):
        k_i^t = min( k_obj · L_obj / L_i,  k̄_i )

    Parameters
    ----------
    coeffs   : current coefficient dict (modified in-place)
    loss_dict: {L_obj, L_g, L_Sl, L_theta, L_z, L_d}  — scalar Python floats
    k_upper  : {k_g, k_Sl, k_theta, k_z, k_d}  — upper bounds k̄_i
    """
    # Use abs(L_obj) so that cost functions with large negative c0 constants
    # (common in some MATPOWER cases) do not produce negative weights.
    L_obj_abs = abs(loss_dict['L_obj'])
    k_obj = coeffs['k_obj']

    for name, loss_key in [('k_g',     'L_g'),
                            ('k_Sl',    'L_Sl'),
                            ('k_theta', 'L_theta'),
                            ('k_z',     'L_z'),
                            ('k_d',     'L_d')]:
        Li = loss_dict[loss_key]
        if Li > 1e-12:
            new_k = k_obj * L_obj_abs / Li
        else:
            new_k = k_upper[name]     # Li ≈ 0 → constraint satisfied; cap at upper bound
        coeffs[name] = float(min(new_k, k_upper[name]))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training  (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

def train_deepopf_ngt(
        case_name:      str,
        params_path:    str,
        data_path:      str,
        data_mode:      str  = 'random_split',
        n_train_use:    int  = 10000,
        n_test_samples: int  = None,
        test_data_path: str  = None,
        test_params_path: str = None,
        seed:           int  = 42,
        n_epochs:       int  = 100,
        learning_rate:  float = 1e-3,
        hidden_sizes:   list = None,
        batch_size:     int  = 256,
        # --- initial / upper-bound coefficients (paper: fine-tuned manually) ---
        k_obj:   float = 1.0,
        k_g_0:   float = 1.0,   k_g_max:   float = 1000.0,
        k_Sl_0:  float = 1.0,   k_Sl_max:  float = 1000.0,
        k_th_0:  float = 1.0,   k_th_max:  float = 1000.0,
        k_z_0:   float = 1.0,   k_z_max:   float = 1000.0,
        k_d_0:   float = 1.0,   k_d_max:   float = 1000.0,
        theta_max_deg: float = 30.0,
        device:  str   = 'cpu',
):
    """
    Paper-faithful training of DeepOPF-NGT (Algorithm 1).

    Key design decisions matching the paper:
      • Weights updated per mini-batch via Eq.(12), not per epoch.
      • L_obj uses raw P̂_g (not clamped) as reconstructed from power balance.
      • L_d = predicted-vs-demanded load deviation (Eq. 9).
      • No gradient clipping, no EMA, no lower-bound on k_i.
    """
    if hidden_sizes is None:
        hidden_sizes = [256, 256]

    print(f"\n{'='*70}")
    print(f"DeepOPF-NGT  —  Unsupervised ACOPF  (Paper Algorithm 1)")
    print(f"{'='*70}")
    print(f"  k_obj={k_obj} | upper bounds: k_g={k_g_max}, k_Sl={k_Sl_max}, "
          f"k_θ={k_th_max}, k_z={k_z_max}, k_d={k_d_max}")
    print(f"  Epochs={n_epochs}, lr={learning_rate}, batch={batch_size}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ── [1] Load system parameters ────────────────────────────────────────────
    params = load_parameters_from_csv(case_name, params_path)
    G, B   = build_admittance_matrix(params, device_obj)

    # Identify ZIBs / non-ZIBs (Kron reduction, Sec. III-A)
    zib_mask      = params['general']['zib_mask']
    nonzib_indices = np.where(~zib_mask)[0].tolist()
    params['general']['nonzib_indices'] = nonzib_indices

    n_buses  = params['general']['n_buses']
    n_nonzib = len(nonzib_indices)
    print(f"  Buses: {n_buses}  |  Non-ZIB (predicted): {n_nonzib}")

    # ── [2] Load and split data ───────────────────────────────────────────────
    x_data_scaled, _, scalers, raw_data, cost_baseline = load_and_scale_acopf_data(
        data_path, params, fit_scalers=True
    )

    if data_mode == DataMode.API_TEST:
        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, x_data_scaled,
            mode=DataMode.API_TEST, n_train_use=n_train_use, seed=seed)
        test_params, test_x_scaled, _, test_raw_data, _ = load_api_test_data(
            test_data_path, test_params_path, scalers,
            n_test_samples=n_test_samples, seed=seed)
        test_idx = np.arange(len(test_x_scaled))
    elif data_mode == DataMode.GENERALIZATION:
        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, x_data_scaled,
            mode=DataMode.GENERALIZATION, n_train_use=n_train_use, seed=seed)
        test_x_scaled, _, test_raw_data, _ = load_generalization_test_data(
            test_data_path, params, scalers,
            n_test_samples=n_test_samples, seed=seed)
        test_idx = np.arange(len(test_x_scaled))
        test_params = params
    else:
        train_idx, val_idx, test_idx = prepare_data_splits(
            x_data_scaled, x_data_scaled,
            mode=data_mode, n_train_use=n_train_use, seed=seed)
        test_x_scaled = x_data_scaled
        test_raw_data = raw_data
        test_params   = params

    n_loads  = params['general']['n_loads']
    X_train  = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32, device=device_obj)
    X_test   = torch.tensor(test_x_scaled[test_idx],  dtype=torch.float32, device=device_obj)
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
    if cost_baseline:
        print(f"  Reference cost: {cost_baseline:.2f} $/h")

    # ── [3] Build model ───────────────────────────────────────────────────────
    input_dim  = n_loads * 2          # (P_d, Q_d)
    output_dim = n_nonzib * 2         # (V̂_α, θ̂_α)
    model = DeepOPF_NGT(input_dim, output_dim, hidden_sizes).to(device_obj)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    theta_max_rad = theta_max_deg * np.pi / 180.0

    # ── [4] Coefficient initialisation ───────────────────────────────────────
    coeffs = {
        'k_obj':   k_obj,
        'k_g':     k_g_0,
        'k_Sl':    k_Sl_0,
        'k_theta': k_th_0,
        'k_z':     k_z_0,
        'k_d':     k_d_0,
    }
    k_upper = {
        'k_g':     k_g_max,
        'k_Sl':    k_Sl_max,
        'k_theta': k_th_max,
        'k_z':     k_z_max,
        'k_d':     k_d_max,
    }

    # ── [5] Training loop  (Algorithm 1) ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Training  (Algorithm 1)")
    print(f"{'─'*70}")
    n_train   = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss        = 0.0
        epoch_loss_sums   = {k: 0.0 for k in
                             ['L_obj','L_g','L_Sl','L_theta','L_z','L_d']}

        indices = torch.randperm(n_train, device=device_obj)

        for b in range(n_batches):
            start = b * batch_size
            end   = min(start + batch_size, n_train)
            X_b   = X_train[indices[start:end]]

            # De-normalise inputs to get physical (P_d, Q_d)
            x_np   = X_b.cpu().numpy()
            x_phys = scalers['x'].inverse_transform(x_np)
            Pd_b = torch.tensor(x_phys[:, :n_loads], dtype=torch.float32, device=device_obj)
            Qd_b = torch.tensor(x_phys[:, n_loads:], dtype=torch.float32, device=device_obj)

            optimiser.zero_grad()

            # Forward pass
            y_pred = model(X_b)                                      # (batch, 2·n_nonzib)
            v_alpha, theta_alpha = denormalise_output(y_pred, params, theta_max_rad)

            # Reconstruct all variables via algebraic power flow
            results = compute_algebraic_acopf(
                v_alpha, theta_alpha, Pd_b, Qd_b, params, G, B, device_obj)

            # Compute loss terms (Eqs. 3–9)
            loss_dict = compute_loss_terms(results, Pd_b, Qd_b, params, device_obj)

            # Total loss  Eq. (3)
            loss = compute_total_loss(loss_dict, coeffs)

            loss.backward()
            optimiser.step()

            # ── Adaptive weight update per mini-batch  Eq. (12) ──────────────
            # Paper: "k_i^t calculated per constraint by summing L_obj and L_i
            #         over all training samples in each mini-batch"
            # We use detached scalar values (already summed over batch via mean)
            if epoch > 1:
                batch_loss_vals = {k: v.item() for k, v in loss_dict.items()}
                update_coefficients(coeffs, batch_loss_vals, k_upper)

            bs = end - start
            epoch_loss += loss.item() * bs
            for k, v in loss_dict.items():
                epoch_loss_sums[k] += v.item() * bs

        # ── Per-epoch logging ─────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            emean = {k: epoch_loss_sums[k] / n_train
                     for k in epoch_loss_sums}
            cost_gap = ((emean['L_obj'] - cost_baseline) / cost_baseline * 100
                        if cost_baseline else 0.0)
            print(f"Epoch {epoch:4d}/{n_epochs} | "
                  f"L={epoch_loss/n_train:.4f} | "
                  f"L_obj={emean['L_obj']:.1f} ({cost_gap:+.2f}%) | "
                  f"L_g={emean['L_g']:.4f} | L_Sl={emean['L_Sl']:.4f} | "
                  f"L_θ={emean['L_theta']:.4f} | L_z={emean['L_z']:.4f} | "
                  f"L_d={emean['L_d']:.4f}")
            print(f"  Weights: k_g={coeffs['k_g']:.2f}, k_Sl={coeffs['k_Sl']:.2f}, "
                  f"k_θ={coeffs['k_theta']:.2f}, k_z={coeffs['k_z']:.2f}, "
                  f"k_d={coeffs['k_d']:.2f}")

    print(f"\n  Training complete in {time.time()-t0:.2f} s")

    # ── [6] Test evaluation ───────────────────────────────────────────────────
    model.eval()
    print(f"\n{'='*70}")
    print(f"  Test Evaluation")
    print(f"{'='*70}")

    n_test        = len(X_test)
    test_params_e = test_params
    test_raw_e    = test_raw_data

    # raw_data['x'] = [Pd | Qd] concatenated; split by n_loads
    n_loads_e = test_params_e['general']['n_loads']
    _x        = test_raw_e['x'][test_idx]          # select test rows
    Pd_test   = _x[:, :n_loads_e]
    Qd_test   = _x[:, n_loads_e:]
    y_true_pg = test_raw_e.get('pg',  None)        # all generators (including slack)
    y_true_vm = test_raw_e.get('vm',  None)        # all buses
    y_true_qg = test_raw_e.get('qg',  None)
    y_true_va = test_raw_e.get('va',  None)
    # Index into test rows for ground-truth arrays
    if y_true_pg is not None: y_true_pg = y_true_pg[test_idx]
    if y_true_vm is not None: y_true_vm = y_true_vm[test_idx]
    if y_true_qg is not None: y_true_qg = y_true_qg[test_idx]
    if y_true_va is not None: y_true_va = y_true_va[test_idx]

    n_gen    = test_params_e['general']['n_gen']
    n_buses_t = test_params_e['general']['n_buses']
    base_mva = test_params_e['general'].get('base_mva', 100.0)

    y_pred_pg_list = []
    y_pred_vm_list = []
    pf_results_list = []
    converge_flags  = []

    t_pred_start = time.time()
    with torch.no_grad():
        for i in range(n_test):
            Pd_i = torch.tensor(Pd_test[i:i+1], dtype=torch.float32, device=device_obj)
            Qd_i = torch.tensor(Qd_test[i:i+1], dtype=torch.float32, device=device_obj)

            # Scale input
            x_i_np = np.concatenate([Pd_test[i:i+1], Qd_test[i:i+1]], axis=1)
            x_i_sc = scalers['x'].transform(x_i_np)
            X_i    = torch.tensor(x_i_sc, dtype=torch.float32, device=device_obj)

            y_pred = model(X_i)
            v_alpha, theta_alpha = denormalise_output(y_pred, params, theta_max_rad)

            results = compute_algebraic_acopf(
                v_alpha, theta_alpha, Pd_i, Qd_i, test_params_e, G, B, device_obj)

            Pg       = results['Pg'].cpu().numpy()[0]
            Qg       = results['Qg'].cpu().numpy()[0]
            v_all    = results['v_all'].cpu().numpy()[0]
            th_all   = results['theta_all'].cpu().numpy()[0]
            P_branch = results['P_branch'].cpu().numpy()[0]
            Q_branch = results['Q_branch'].cpu().numpy()[0]

            y_pred_pg_list.append(Pg)
            y_pred_vm_list.append(v_all)

            n_branches = len(test_params_e['branch']['f_bus'])
            pf = {
                'success': True,
                'gen':    np.zeros((n_gen, 21)),
                'bus':    np.zeros((n_buses_t, 13)),
                'branch': np.zeros((n_branches, 17)),
            }
            pf['gen'][:, 1] = Pg     * base_mva
            pf['gen'][:, 2] = Qg     * base_mva
            pf['gen'][:, 8] = test_params_e['generator']['pg_max'].flatten() * base_mva
            pf['gen'][:, 9] = test_params_e['generator']['pg_min'].flatten() * base_mva
            pf['gen'][:, 3] = test_params_e['generator']['qg_max'].flatten() * base_mva
            pf['gen'][:, 4] = test_params_e['generator']['qg_min'].flatten() * base_mva
            pf['bus'][:,  7] = v_all
            pf['bus'][:,  8] = th_all * 180.0 / np.pi
            pf['bus'][:, 11] = test_params_e['bus']['vm_max']
            pf['bus'][:, 12] = test_params_e['bus']['vm_min']
            pf['branch'][:, 13] = P_branch * base_mva
            pf['branch'][:, 14] = Q_branch * base_mva
            pf['branch'][:, 15] = P_branch * base_mva
            pf['branch'][:, 16] = Q_branch * base_mva
            pf['branch'][:,  5] = test_params_e['branch']['rate_a'] * base_mva
            pf_results_list.append([pf])
            converge_flags.append(True)

    t_pred_end = time.time()
    avg_inf_ms = (t_pred_end - t_pred_start) / n_test * 1000.0

    y_pred_pg = np.array(y_pred_pg_list)
    y_pred_vm = np.array(y_pred_vm_list)

    metrics = evaluate_acopf_predictions(
        y_pred_pg, y_pred_vm,
        y_true_pg, y_true_vm, y_true_qg, y_true_va,
        pf_results_list, converge_flags,
        test_params_e, verbose=False)

    print(f"\n--- Accuracy ---")
    print(f"  MAE_Pg (non-slack): {metrics['mae_pg_non_slack_percent']:.4f}%")
    print(f"  MAE_Vm:             {metrics['mae_vm_percent']:.4f}%")
    print(f"--- Violations ---")
    print(f"  Pg_viol:   {metrics['mean_max_pg_viol_pu']:.6f} p.u.")
    print(f"  Qg_viol:   {metrics['mean_max_qg_viol_pu']:.6f} p.u.")
    print(f"  Vm_viol:   {metrics['mean_max_vm_viol_pu']:.6f} p.u.")
    print(f"  Branch_viol: {metrics['mean_max_branch_viol_pu']:.6f} p.u.")
    print(f"--- Optimality ---")
    print(f"  Cost Gap:  {metrics['cost_optimality_gap_percent']:.4f}%")
    print(f"--- Speed ---")
    print(f"  Avg inference: {avg_inf_ms:.4f} ms/sample")
    print(f"  Training time: {time.time()-t0:.2f} s")
    print(f"{'='*70}")

    return model, params, G, B, scalers, coeffs, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    paths        = acopf_config.get_all_paths()
    params_cfg   = acopf_config.get_all_params()

    model, params, G, B, scalers, coeffs, metrics = train_deepopf_ngt(
        case_name       = paths['case_name'],
        params_path     = paths['params_path'],
        data_path       = paths['data_path'],
        test_data_path  = paths.get('test_data_path'),
        test_params_path= paths.get('test_params_path'),
        data_mode       = params_cfg['data_mode'],
        n_train_use     = params_cfg['n_train_use'],
        n_test_samples  = params_cfg.get('n_test_samples'),
        seed            = params_cfg['seed'],
        n_epochs        = params_cfg['n_epochs'],
        learning_rate   = params_cfg['learning_rate'],
        hidden_sizes    = params_cfg['hidden_sizes'],
        batch_size      = params_cfg['batch_size'],
        device          = params_cfg['device'],
    )

    torch.save({
        'model_state_dict': model.state_dict(),
        'params':           params,
        'scalers':          scalers,
        'coeffs':           coeffs,
        'test_metrics':     metrics,
        'version':          'paper_algorithm1_faithful',
    }, 'deepopf_ngt_paper.pth')
    print('\nModel saved to deepopf_ngt_paper.pth')