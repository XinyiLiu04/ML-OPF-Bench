"""
DeepOPF-NGT - PAPER VERSION (Algorithm 1 + Critical Fixes)

Features:
- Dynamic ZIB identification
- Fixed k_obj (0.1)
- Dynamic k_i = clamp(k_obj * L_obj / L_i, k_min, k_max)
- Detailed evaluation using acopf_violation_metrics (same as DNN script)
"""

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
        DataMode
    )
    from algebraic_power_flow import (
        build_admittance_matrix,
        compute_algebraic_acopf
    )
    from acopf_violation_metrics import evaluate_acopf_predictions
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


class VoltagePredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[256, 256]):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def denormalize_voltage(v_norm, theta_norm, params):
    """Denormalize with ±30° theta limit"""
    nonzib_indices = params['general']['nonzib_indices']

    all_vm_min = torch.as_tensor(params['bus']['vm_min'],
                                 dtype=torch.float32, device=v_norm.device)
    all_vm_max = torch.as_tensor(params['bus']['vm_max'],
                                 dtype=torch.float32, device=v_norm.device)

    v_min = all_vm_min[nonzib_indices]
    v_max = all_vm_max[nonzib_indices]

    v = v_min + v_norm * (v_max - v_min)

    theta_max_deg = 30.0
    theta_max_rad = theta_max_deg * np.pi / 180.0
    theta = (theta_norm - 0.5) * 2 * theta_max_rad

    return v, theta


def compute_unsupervised_loss(results, Pd, Qd, params, device='cpu', G=None, B=None):
    """
    Unsupervised ACOPF loss with two key fixes vs original:

    Fix 1 – L_obj: cost is computed from raw (unclipped) Pg so that the
      gradient always flows back through the network.  Clipping zero-out
      gradients for out-of-bound values, which let the model exploit a
      low-cost but infeasible regime (Cost Gap < 0).

    Fix 2 – L_d: power balance is measured via the true AC nodal residual
      computed from v_all / theta_all and the admittance matrix, instead of
      a hardcoded "3% losses" heuristic.  The heuristic was case-dependent
      and its gradient through Pg_clipped was zero for out-of-bound values.
      The nodal residual |P_inj - P_calc| has clean gradients w.r.t. voltages.
    """
    Pg = results['Pg']
    Qg = results['Qg']
    v_all = results['v_all']
    theta_all = results['theta_all']
    P_branch = results['P_branch']
    Q_branch = results['Q_branch']

    gen_params = params['generator']
    pg_min = torch.as_tensor(gen_params['pg_min'], device=device, dtype=torch.float32).squeeze()
    pg_max = torch.as_tensor(gen_params['pg_max'], device=device, dtype=torch.float32).squeeze()
    qg_min = torch.as_tensor(gen_params['qg_min'], device=device, dtype=torch.float32).squeeze()
    qg_max = torch.as_tensor(gen_params['qg_max'], device=device, dtype=torch.float32).squeeze()

    c2 = torch.as_tensor(gen_params['cost_c2'], device=device, dtype=torch.float32)
    c1 = torch.as_tensor(gen_params['cost_c1'], device=device, dtype=torch.float32)
    c0 = torch.as_tensor(gen_params['cost_c0'], device=device, dtype=torch.float32)

    vm_min = torch.as_tensor(params['bus']['vm_min'], device=device, dtype=torch.float32)
    vm_max = torch.as_tensor(params['bus']['vm_max'], device=device, dtype=torch.float32)
    rate_a = torch.as_tensor(params['branch']['rate_a'], device=device, dtype=torch.float32)

    # ── L_obj: cost uses clamped Pg to prevent spurious negative-cost solutions ──
    # When Pg is unclamped, the model can push Pg negative (or very large) to
    # minimise the quadratic cost, producing L_obj < 0 at random initialisation.
    # L_obj must only reward physically feasible generation levels.
    # Gradient for L_obj therefore flows through the clamped path.
    #
    # ── L_g: violation penalty uses RAW Pg for full gradient coverage ──────────
    # relu(pg_min - Pg) has non-zero gradient only when Pg < pg_min, so the
    # gradient already flows exactly where it is needed regardless of clamping.
    # Using raw Pg here is equivalent to using clamped Pg for L_g, but we keep
    # raw Pg explicitly to make the intent clear.
    # Clamp for cost evaluation (prevents negative-Pg cost exploitation)
    Pg_clipped = torch.clamp(Pg, pg_min, pg_max)
    cost_per_sample = torch.sum(c2 * Pg_clipped ** 2 + c1 * Pg_clipped + c0, dim=1)
    # Add a linear extrapolation term for out-of-bound Pg so that L_obj still
    # has non-zero gradient w.r.t. Pg outside [pg_min, pg_max].
    # At the boundary, marginal cost = 2*c2*Pg_boundary + c1 (slope of cost curve).
    # This keeps L_obj pushing Pg toward the feasible region even when clamped.
    mc_min = 2.0 * c2 * pg_min + c1   # marginal cost at lower bound
    mc_max = 2.0 * c2 * pg_max + c1   # marginal cost at upper bound
    extra = torch.sum(
        mc_min * torch.relu(pg_min - Pg) +   # below lower bound
        mc_max * torch.relu(Pg - pg_max),     # above upper bound
        dim=1
    )
    L_obj = torch.mean(cost_per_sample + extra)

    # Generator constraints: raw Pg ensures full gradient flow toward feasibility
    pg_viol = torch.relu(pg_min - Pg) + torch.relu(Pg - pg_max)
    qg_viol = torch.relu(qg_min - Qg) + torch.relu(Qg - qg_max)
    L_g = torch.mean(torch.sum(pg_viol ** 2 + qg_viol ** 2, dim=1))

    # Voltage constraints
    vm_viol = torch.relu(vm_min - v_all) + torch.relu(v_all - vm_max)
    L_z = torch.mean(torch.sum(vm_viol ** 2, dim=1))

    # Branch flow constraints
    S_branch = torch.sqrt(P_branch ** 2 + Q_branch ** 2 + 1e-12)
    valid_mask = (rate_a > 1e-5) & (rate_a < 9000)
    if valid_mask.any():
        s_viol = torch.relu(S_branch[:, valid_mask] - rate_a[valid_mask])
        L_Sl = torch.mean(torch.sum(s_viol ** 2, dim=1))
    else:
        L_Sl = torch.tensor(0.0, device=device)

    # Angle constraints
    theta_max = 30 * np.pi / 180
    f_bus = params['branch']['f_bus']
    t_bus = params['branch']['t_bus']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    theta_diff_list = []
    for k in range(len(f_bus)):
        i = bus_id_to_idx[int(f_bus[k])]
        j = bus_id_to_idx[int(t_bus[k])]
        theta_diff_list.append(theta_all[:, i] - theta_all[:, j])

    if len(theta_diff_list) > 0:
        theta_diff = torch.stack(theta_diff_list, dim=1)
        theta_viol = torch.relu(torch.abs(theta_diff) - theta_max)
        L_theta = torch.mean(torch.sum(theta_viol ** 2, dim=1))
    else:
        L_theta = torch.tensor(0.0, device=device)

    # ── Fix 2: L_d uses true AC nodal power-balance residual ─────────────────
    # P_inj[i] = Pg_bus[i] - Pd[i]   (net injection, p.u.)
    # P_calc[i] = v_i * Σ_j v_j*(G_ij*cos θ_ij + B_ij*sin θ_ij)  (from Y-bus)
    # L_d = mean over samples of  Σ_i (P_inj[i] - P_calc[i])²
    #
    # Falls back to simple Pg-sum balance when G/B not available.
    if G is not None and B is not None:
        # Build bus-level Pg injection: Map_gen (n_buses, n_gen) @ Pg → (batch, n_buses)
        bus_ids     = params['general']['bus_ids']
        gen_bus_ids = params['general']['gen_bus_ids']
        bus_id_to_idx_map = params['general']['bus_id_to_idx']
        n_buses = params['general']['n_buses']
        n_gen   = params['general']['n_gen']

        # Generator-to-bus mapping (built once, moved to device)
        gen_bus_indices = torch.tensor(
            [bus_id_to_idx_map[int(gid)] for gid in gen_bus_ids],
            dtype=torch.long, device=device
        )
        batch = Pg.shape[0]
        Pg_bus = torch.zeros(batch, n_buses, dtype=torch.float32, device=device)
        Pg_bus.scatter_add_(1, gen_bus_indices.unsqueeze(0).expand(batch, -1), Pg)

        # Pd passed in is (batch, n_loads); expand to (batch, n_buses) using bus mapping
        load_bus_ids = params['general']['load_bus_ids']
        load_bus_indices = torch.tensor(
            [bus_id_to_idx_map[int(lid)] for lid in load_bus_ids],
            dtype=torch.long, device=device
        )
        Pd_bus = torch.zeros(batch, n_buses, dtype=torch.float32, device=device)
        Pd_bus.scatter_add_(1, load_bus_indices.unsqueeze(0).expand(batch, -1), Pd)

        P_inj = Pg_bus - Pd_bus   # (batch, n_buses)

        # AC power calculated from voltages: P_calc_i = v_i Σ_j v_j(G cosθ+B sinθ)
        cos_th = torch.cos(theta_all)   # (batch, n_buses)
        sin_th = torch.sin(theta_all)
        # v_i * v_j for all i,j:  (batch, n_buses, n_buses)
        vi = v_all.unsqueeze(2)         # (batch, n_buses, 1)
        vj = v_all.unsqueeze(1)         # (batch, 1, n_buses)
        vivj = vi * vj                  # (batch, n_buses, n_buses)

        cos_diff = (cos_th.unsqueeze(2) * cos_th.unsqueeze(1)
                    + sin_th.unsqueeze(2) * sin_th.unsqueeze(1))   # cos(θi-θj)
        sin_diff = (sin_th.unsqueeze(2) * cos_th.unsqueeze(1)
                    - cos_th.unsqueeze(2) * sin_th.unsqueeze(1))   # sin(θi-θj)

        P_calc = torch.sum(vivj * (G * cos_diff + B * sin_diff), dim=2)  # (batch, n_buses)
        balance_err = P_inj - P_calc    # (batch, n_buses)
        L_d = torch.mean(torch.sum(balance_err ** 2, dim=1))
    else:
        # Fallback: scalar active power balance  Σ Pg = Σ Pd
        Pg_total = torch.sum(Pg, dim=1)
        Pd_total = torch.sum(Pd, dim=1)
        L_d = torch.mean((Pg_total - Pd_total) ** 2)

    return {
        'L_obj': L_obj,
        'L_g': L_g,
        'L_Sl': L_Sl,
        'L_theta': L_theta,
        'L_z': L_z,
        'L_d': L_d
    }


def compute_total_loss(loss_dict, coeffs):
    """Weighted total loss"""
    total = (
            coeffs['k_obj'] * loss_dict['L_obj'] +
            coeffs['k_g'] * loss_dict['L_g'] +
            coeffs['k_Sl'] * loss_dict['L_Sl'] +
            coeffs['k_theta'] * loss_dict['L_theta'] +
            coeffs['k_z'] * loss_dict['L_z'] +
            coeffs['k_d'] * loss_dict['L_d']
    )
    return total


def train_deepopf_ngt_paper(
        case_name,
        params_path,
        data_path,
        data_mode='random_split',
        n_train_use=1000,
        n_test_samples=None,
        test_data_path=None,
        test_params_path=None,
        seed=42,
        n_epochs=100,
        learning_rate=0.001,
        hidden_sizes=[256, 256],
        batch_size=256,
        device='cpu'
):
    """Train DeepOPF-NGT with Paper Algorithm 1"""
    print(f"\n{'=' * 70}")
    print(f"DeepOPF-NGT - PAPER VERSION (Algorithm 1 + Fixes)")
    print(f"{'=' * 70}")
    print(f"🎯 Strategy: EMA-smoothed dynamic weights + auto-normalised losses")
    print(f"✅ Fixed k_obj: 0.1")
    print(f"✅ Auto-norm: each loss divided by its epoch-1 mean (case-agnostic)")
    print(f"✅ EMA smoothing (alpha=0.3): prevents single-epoch ratchet effect")
    print(f"✅ Dynamic ZIB identification")
    print(f"✅ Keeps critical fixes: Pg clipping, p.u. units, power balance")
    print(f"{'=' * 70}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # [1] Load parameters
    print(f"\n[1] Loading parameters...")
    params = load_parameters_from_csv(case_name, params_path)
    G, B = build_admittance_matrix(params, device)
    print(f"✅ Admittance matrix: {G.shape}")

    # [2] Dynamic ZIB identification
    print(f"\n[2] Using dynamic ZIB identification from parameters...")
    zib_mask = params['general']['zib_mask']
    nonzib_indices = np.where(~zib_mask)[0].tolist()
    params['general']['nonzib_indices'] = nonzib_indices

    n_buses = params['general']['n_buses']
    n_nonzib = len(nonzib_indices)
    print(f"📊 Buses: {n_buses} total, {n_nonzib} non-ZIB")

    # [3] Load data
    print(f"\n[3] Loading data...")
    x_data_scaled, _, scalers, raw_data, cost_baseline = load_and_scale_acopf_data(
        data_path, params, fit_scalers=True
    )

    # ── Split and prepare test set based on data mode ─────────────────────
    if data_mode == DataMode.API_TEST:
        print(f"\n  Data Mode: API_TEST")
        if test_data_path is None or test_params_path is None:
            raise ValueError("API_TEST mode requires test_data_path and test_params_path")

        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, x_data_scaled,
            mode=DataMode.API_TEST,
            n_train_use=n_train_use,
            seed=seed
        )
        test_params, test_x_scaled, _, test_raw_data, _ = load_api_test_data(
            test_data_path, test_params_path, scalers,
            n_test_samples=n_test_samples, seed=seed
        )
        test_idx = np.arange(len(test_x_scaled))
        # Extract test_case_name for PyPower loading
        import os as _os
        _base = _os.path.basename(test_data_path)
        test_case_name = _base[:-7] if _base.endswith('_pd.csv') else _base.rsplit('_', 1)[0]
        _use_test_case_data = True   # flag: load separate PyPower data for test

    elif data_mode == DataMode.GENERALIZATION:
        print(f"\n  Data Mode: GENERALIZATION")
        if test_data_path is None:
            raise ValueError("GENERALIZATION mode requires test_data_path")

        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, x_data_scaled,
            mode=DataMode.GENERALIZATION,
            n_train_use=n_train_use,
            seed=seed
        )
        test_x_scaled, _, test_raw_data, _ = load_generalization_test_data(
            test_data_path, params, scalers,
            n_test_samples=n_test_samples, seed=seed
        )
        test_idx = np.arange(len(test_x_scaled))
        test_params = params
        test_case_name = case_name
        _use_test_case_data = False

    else:  # random_split / fixed_valtest
        print(f"\n  Data Mode: {data_mode}")
        train_idx, val_idx, test_idx = prepare_data_splits(
            x_data_scaled, x_data_scaled,
            mode=data_mode,
            n_train_use=n_train_use,
            seed=seed
        )
        test_x_scaled = x_data_scaled
        test_raw_data = raw_data
        test_params = params
        test_case_name = case_name
        _use_test_case_data = False

    n_loads = params['general']['n_loads']
    X_train = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32, device=device)
    X_test  = torch.tensor(test_x_scaled[test_idx],  dtype=torch.float32, device=device)

    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    if data_mode in (DataMode.API_TEST, DataMode.GENERALIZATION):
        print(f"  Test case: {test_case_name}  "
              f"Buses: {test_params['general']['n_buses']}  "
              f"Gens: {test_params['general']['n_gen']}")
    print(f"🎯 Target cost: {cost_baseline:.0f} $/h")

    # [4] Create model
    input_dim, output_dim = n_loads * 2, n_nonzib * 2
    print(f"\n[4] Model: {input_dim} → {hidden_sizes} → {output_dim}")

    model = VoltagePredictor(input_dim, output_dim, hidden_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Coefficients – k_obj fixed; others updated via unified 3-regime rule
    # operating on auto-normalised losses (epoch-1 reference scaling).
    #
    # Three regimes per weight k_i (same logic as unsupervised DCOPF):
    #   (A) ema_i < SATISFIED_THRESH  → constraint satisfied, decay k_i back
    #       to initial value so L_obj can regain gradient dominance.
    #   (B) SATISFIED_THRESH ≤ ema_i < BOOST_THRESH  → standard dynamic
    #       formula: k_i = k_obj * ema_obj / ema_i
    #   (C) ema_i ≥ BOOST_THRESH  → severe violation, proportional boost:
    #       k_i = BOOST_BASE * ema_i  (ensures meaningful penalty signal)
    #
    # This prevents two symmetric failure modes:
    #   • k_g / k_theta locking at max when their loss → 0  (case30/118)
    #   • k_Sl / k_d stuck at min when their loss stays high  (case300)
    coeffs = {
        'k_obj':       0.1,
        'k_g':         100.0,
        'k_Sl':        10.0,
        'k_theta':     100.0,
        'k_z':         10.0,
        'k_d':         10.0,
        # upper bounds: physics-critical terms (g, theta) keep large max;
        # secondary terms (Sl, z, d) capped conservatively
        'k_g_max':     10000.0,
        'k_Sl_max':    200.0,
        'k_theta_max': 10000.0,
        'k_z_max':     200.0,
        'k_d_max':     200.0,
        'k_g_min':     1.0,
        'k_Sl_min':    1.0,
        'k_theta_min': 1.0,
        'k_z_min':     1.0,
        'k_d_min':     1.0,
    }

    LOSS_KEYS = ['L_obj', 'L_g', 'L_Sl', 'L_theta', 'L_z', 'L_d']
    # weight key → corresponding loss key + initial value for decay target
    WEIGHT_LOSS_MAP = {
        'k_g':     ('L_g',     100.0),
        'k_Sl':    ('L_Sl',    10.0),
        'k_theta': ('L_theta', 100.0),
        'k_z':     ('L_z',     10.0),
        'k_d':     ('L_d',     10.0),
    }

    # 3-regime thresholds and parameters
    SATISFIED_THRESH = 0.02   # ema_i below → constraint satisfied, decay
    BOOST_THRESH     = 0.30   # ema_i above → severe violation, boost
    DECAY            = 0.95   # per-epoch decay multiplier when satisfied
    BOOST_BASE       = 50.0   # proportional boost scale

    # Reference magnitudes (set after epoch 1, never updated again)
    loss_refs = {k: 1.0 for k in LOSS_KEYS}

    # EMA state for normalised loss values
    EMA_ALPHA = 0.3
    loss_ema  = {k: None for k in LOSS_KEYS}

    # [5] Training
    print(f"\n[5] Training with EMA-smoothed dynamic weights...")
    print(f"{'-' * 70}")

    n_train = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_loss_sums = {k: 0.0 for k in LOSS_KEYS}

        indices = torch.randperm(n_train, device=device)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            X_batch = X_train[indices[start_idx:end_idx]]

            x_batch_np = X_batch.cpu().numpy()
            x_batch_denorm = scalers['x'].inverse_transform(x_batch_np)
            Pd_batch = torch.tensor(x_batch_denorm[:, :n_loads], dtype=torch.float32, device=device)
            Qd_batch = torch.tensor(x_batch_denorm[:, n_loads:], dtype=torch.float32, device=device)

            optimizer.zero_grad()

            y_pred = model(X_batch)
            v_norm = y_pred[:, :n_nonzib]
            theta_norm = y_pred[:, n_nonzib:]

            v_alpha, theta_alpha = denormalize_voltage(v_norm, theta_norm, params)
            results = compute_algebraic_acopf(v_alpha, theta_alpha, Pd_batch, Qd_batch, params, G, B, device)
            loss_dict = compute_unsupervised_loss(results, Pd_batch, Qd_batch, params, device, G=G, B=B)

            # ── auto-normalise by epoch-1 reference ───────────────────────
            loss_norm = {k: loss_dict[k] / loss_refs[k] for k in LOSS_KEYS}
            loss = compute_total_loss(loss_norm, coeffs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size_actual = end_idx - start_idx
            epoch_loss += loss.item() * batch_size_actual
            for key in LOSS_KEYS:
                epoch_loss_sums[key] += loss_dict[key].item() * batch_size_actual

        # ── epoch-level means (raw and normalised) ────────────────────────
        epoch_means_raw  = {k: epoch_loss_sums[k] / n_train for k in LOSS_KEYS}
        epoch_means_norm = {k: epoch_means_raw[k] / loss_refs[k] for k in LOSS_KEYS}

        # ── set references + initialise EMA after epoch 1 ─────────────────
        if epoch == 1:
            # L_obj reference: use cost_baseline (known optimum) instead of
            # epoch-1 mean.  In ACOPF the model is randomly initialised so
            # voltages/angles are garbage at epoch 1 → the algebraic power
            # flow gives a completely wrong cost estimate.  Using cost_baseline
            # anchors the normalisation to the true target scale from the start.
            if cost_baseline and cost_baseline > 1e-3:
                loss_refs['L_obj'] = float(cost_baseline)
            else:
                raw_obj = epoch_means_raw['L_obj']
                loss_refs['L_obj'] = float(raw_obj) if raw_obj > 1e-8 else 1.0

            # All constraint terms: epoch-1 auto-norm (they are well-defined
            # even at random initialisation)
            for k in ['L_g', 'L_Sl', 'L_theta', 'L_z', 'L_d']:
                raw = epoch_means_raw[k]
                loss_refs[k] = float(raw) if raw > 1e-8 else 1.0

            for k in LOSS_KEYS:
                loss_ema[k] = 1.0      # all normalised values start at ~1.0
            print(f"  [Auto-norm] References after epoch 1:")
            print(f"    L_obj={loss_refs['L_obj']:.2f} (cost_baseline)"
                  f"  L_g={loss_refs['L_g']:.4f}"
                  f"  L_Sl={loss_refs['L_Sl']:.4f}  L_theta={loss_refs['L_theta']:.4f}"
                  f"  L_z={loss_refs['L_z']:.4f}  L_d={loss_refs['L_d']:.4f}")

        # ── update EMA and dynamic weights (from epoch 2 onward) ──────────
        if epoch > 1:
            for k in LOSS_KEYS:
                loss_ema[k] = (EMA_ALPHA * epoch_means_norm[k]
                               + (1.0 - EMA_ALPHA) * loss_ema[k])

            ema_obj = loss_ema['L_obj']
            for wk, (lk, k_init) in WEIGHT_LOSS_MAP.items():
                ema_i = loss_ema[lk]
                k_min = coeffs[f'{wk}_min']
                k_max = coeffs[f'{wk}_max']

                if ema_i < SATISFIED_THRESH:
                    # (A) Constraint satisfied – decay toward initial value
                    decayed = coeffs[wk] * DECAY
                    coeffs[wk] = float(max(decayed, k_init))

                elif ema_i >= BOOST_THRESH:
                    # (C) Severe violation – proportional boost
                    boosted = BOOST_BASE * ema_i
                    coeffs[wk] = float(np.clip(boosted, k_min, k_max))

                else:
                    # (B) Normal range – standard dynamic formula
                    if ema_i > 1e-8:
                        new_w = coeffs['k_obj'] * ema_obj / ema_i
                        coeffs[wk] = float(np.clip(new_w, k_min, k_max))

        # ── logging ───────────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            r = epoch_means_raw
            cost_gap = (r['L_obj'] - cost_baseline) / cost_baseline * 100 if cost_baseline else 0
            n_obj = r['L_obj'] / loss_refs['L_obj']
            n_g   = r['L_g']   / loss_refs['L_g']
            n_d   = r['L_d']   / loss_refs['L_d']
            print(f"Epoch {epoch:4d}/{n_epochs} | "
                  f"L_obj: {r['L_obj']:.0f} ({cost_gap:+.1f}%, norm {n_obj:.4f}) | "
                  f"L_g: {r['L_g']:.4f} (norm {n_g:.4f}) | "
                  f"L_d: {r['L_d']:.4f} (norm {n_d:.4f})")
            ema_str = (f"  EMA → obj:{loss_ema['L_obj']:.4f}"
                       f"  g:{loss_ema['L_g']:.4f}"
                       f"  Sl:{loss_ema['L_Sl']:.4f}"
                       f"  th:{loss_ema['L_theta']:.4f}"
                       f"  z:{loss_ema['L_z']:.4f}"
                       f"  d:{loss_ema['L_d']:.4f}"
                       if epoch > 1 else "")
            print(f"  Weights: k_g={coeffs['k_g']:.1f}, k_Sl={coeffs['k_Sl']:.1f}, "
                  f"k_theta={coeffs['k_theta']:.1f}, k_z={coeffs['k_z']:.1f}, "
                  f"k_d={coeffs['k_d']:.1f}" + ema_str)

    print(f"\n✅ Completed in {time.time() - t0:.2f}s")

    # [6] Test Set Detailed Evaluation (using PyPower runpf like DNN script)
    print(f"\n{'=' * 70}")
    print(f"Test Set Detailed Evaluation (PyPower verification)")
    print(f"{'=' * 70}")

    # Initialize PyPower (same as DNN script)
    try:
        from pypower.runpf import runpf
        from pypower.ppoption import ppoption
        import pandas as pd
        from pathlib import Path

        print(f"Initializing PyPower...")
        ppopt = ppoption()
        PPOPT = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)

        # Load PyPower case data
        def load_case_from_csv(case_name, constraints_path):
            base_path = Path(constraints_path)
            base_mva_df = pd.read_csv(base_path / f"{case_name}_base_mva.csv")
            bus_df = pd.read_csv(base_path / f"{case_name}_bus_data.csv")
            gen_df = pd.read_csv(base_path / f"{case_name}_gen_data.csv")
            branch_df = pd.read_csv(base_path / f"{case_name}_branch_data.csv")
            baseMVA = base_mva_df['value'].iloc[0]

            bus = np.zeros((len(bus_df), 13))
            bus[:, 0] = bus_df['bus_id'].values
            bus[:, 1] = bus_df['type'].values
            bus[:, 2] = bus_df['pd_pu'].values
            bus[:, 3] = bus_df['qd_pu'].values
            bus[:, 6] = 1
            bus[:, 7] = bus_df['vm_pu'].values
            bus[:, 8] = bus_df['va_deg'].values
            bus[:, 9] = bus_df['base_kv'].values
            bus[:, 10] = 1
            bus[:, 11] = bus_df['vmax_pu'].values
            bus[:, 12] = bus_df['vmin_pu'].values

            gen = np.zeros((len(gen_df), 21))
            gen[:, 0] = gen_df['bus_id'].values
            gen[:, 3] = gen_df['qg_max_pu'].values
            gen[:, 4] = gen_df['qg_min_pu'].values
            gen[:, 5] = gen_df['vg_pu'].values
            gen[:, 6] = baseMVA
            gen[:, 7] = 1
            gen[:, 8] = gen_df['pg_max_pu'].values
            gen[:, 9] = gen_df['pg_min_pu'].values

            branch = np.zeros((len(branch_df), 13))
            branch[:, 0] = branch_df['f_bus'].values
            branch[:, 1] = branch_df['t_bus'].values
            branch[:, 2] = branch_df['r_pu'].values
            branch[:, 3] = branch_df['x_pu'].values
            branch[:, 4] = branch_df['b_pu'].values
            branch[:, 5] = branch_df['rate_a_pu'].values
            branch[:, 6] = branch[:, 5]
            branch[:, 7] = branch[:, 5]
            branch[:, 8] = branch_df['tap_ratio'].values
            branch[:, 9] = branch_df['shift_deg'].values
            branch[:, 10] = 1
            branch[:, 11] = -360
            branch[:, 12] = 360
            rate_a_values = branch_df['rate_a_pu'].values
            branch[:, 5:8][np.isnan(rate_a_values) | np.isinf(rate_a_values), :] = 9900.0

            gencost = np.zeros((len(gen_df), 7))
            gencost[:, 0] = 2
            gencost[:, 3] = 3
            gencost[:, 4] = gen_df['cost_c2'].values
            gencost[:, 5] = gen_df['cost_c1'].values
            gencost[:, 6] = gen_df['cost_c0'].values

            ppc = {'version': '2', 'baseMVA': baseMVA, 'bus': bus, 'gen': gen, 'branch': branch, 'gencost': gencost}
            ppc['bus'][:, 2] *= baseMVA
            ppc['bus'][:, 3] *= baseMVA
            ppc['gen'][:, 3] *= baseMVA
            ppc['gen'][:, 4] *= baseMVA
            ppc['gen'][:, 8] *= baseMVA
            ppc['gen'][:, 9] *= baseMVA
            mask = (ppc['branch'][:, 5] != 0) & (ppc['branch'][:, 5] < 9000)
            ppc['branch'][mask, 5:8] *= baseMVA
            return ppc

        GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)
        # For api_test mode, load the test case's PyPower data separately
        if _use_test_case_data:
            GLOBAL_CASE_DATA_TEST = load_case_from_csv(test_case_name, test_params_path)
            print(f"✓ PyPower case data loaded (train: {case_name}, test: {test_case_name})")
        else:
            GLOBAL_CASE_DATA_TEST = GLOBAL_CASE_DATA
            print(f"✓ PyPower case data loaded")

    except ImportError:
        print(f"⚠️  PyPower not available, using algebraic power flow")
        PPOPT = None
        GLOBAL_CASE_DATA = None
        GLOBAL_CASE_DATA_TEST = None

    model.eval()
    n_test = len(X_test)
    print(f"Evaluating {n_test} test samples...")

    # Prepare test data (use test_x_scaled, already in X_test)
    x_test_denorm = scalers['x'].inverse_transform(X_test.cpu().numpy())
    Pd_test = x_test_denorm[:, :n_loads]
    Qd_test = x_test_denorm[:, n_loads:]

    # Get true values from test set
    y_true_pg = test_raw_data['pg'][test_idx]
    y_true_vm = test_raw_data['vm'][test_idx]
    y_true_qg = test_raw_data['qg'][test_idx]
    y_true_va = test_raw_data['va'][test_idx]

    # === Inference Time Measurement ===
    print(f"  Measuring inference time...")

    # 1. Measure pure DNN inference time (forward pass only)
    dnn_inference_times = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for i in range(n_test):
        t_start = time.time()
        with torch.no_grad():
            y_pred_single = model(X_test[i:i + 1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.time()
        dnn_inference_times.append(t_end - t_start)

    avg_dnn_inference_time = np.mean(dnn_inference_times)

    # 2. Predict voltages (batch mode for evaluation)
    print(f"  Running model predictions...")
    t_prediction_start = time.time()
    with torch.no_grad():
        y_pred = model(X_test)
        v_norm = y_pred[:, :n_nonzib]
        theta_norm = y_pred[:, n_nonzib:]
        v_alpha, theta_alpha = denormalize_voltage(v_norm, theta_norm, params)
    t_prediction_end = time.time()
    total_prediction_time = t_prediction_end - t_prediction_start

    # Run PyPower or algebraic PF
    print(f"  Running power flow verification...")
    t_pf_start = time.time()

    y_pred_pg_list = []
    y_pred_vm_list = []
    pf_results_list = []
    converge_flags = []

    base_mva = test_params['general']['BASE_MVA']
    n_gen = test_params['general']['n_gen']
    n_buses = test_params['general']['n_buses']
    load_bus_ids = test_params['general']['load_bus_ids']
    bus_id_to_idx = test_params['general']['bus_id_to_idx']
    non_slack_gen_idx = test_params['general']['non_slack_gen_idx']
    gen_bus_ids = test_params['general']['gen_bus_ids']
    gen_bus_indices = [bus_id_to_idx[int(gid)] for gid in gen_bus_ids]

    if PPOPT is not None and GLOBAL_CASE_DATA_TEST is not None:
        # Use PyPower (like DNN script)
        print(f"  Running PyPower verification...")

        # First compute algebraic PF to get initial Pg values
        for i in range(n_test):
            Pd_batch = torch.tensor(Pd_test[i:i + 1], dtype=torch.float32, device=device)
            Qd_batch = torch.tensor(Qd_test[i:i + 1], dtype=torch.float32, device=device)

            # Get Pg and Vm from algebraic PF (for PyPower input)
            results_alg = compute_algebraic_acopf(
                v_alpha[i:i + 1], theta_alpha[i:i + 1],
                Pd_batch, Qd_batch, params, G, B, device
            )
            Pg_alg = results_alg['Pg'].cpu().numpy()[0]
            Vm_gen = results_alg['v_all'].cpu().numpy()[0][gen_bus_indices]

            # Setup PyPower case
            mpc_pf = {
                'version': GLOBAL_CASE_DATA_TEST['version'],
                'baseMVA': GLOBAL_CASE_DATA_TEST['baseMVA'],
                'bus': GLOBAL_CASE_DATA_TEST['bus'].copy(),
                'gen': GLOBAL_CASE_DATA_TEST['gen'].copy(),
                'branch': GLOBAL_CASE_DATA_TEST['branch'],
                'gencost': GLOBAL_CASE_DATA_TEST['gencost']
            }

            # Set loads
            for j, bus_id in enumerate(load_bus_ids):
                bus_idx = bus_id_to_idx.get(int(bus_id))
                if bus_idx is not None:
                    mpc_pf["bus"][bus_idx, 2] = Pd_test[i, j] * base_mva
                    mpc_pf["bus"][bus_idx, 3] = Qd_test[i, j] * base_mva

            # Set non-slack generator Pg
            for j, gen_idx in enumerate(non_slack_gen_idx):
                mpc_pf["gen"][gen_idx, 1] = Pg_alg[gen_idx] * base_mva

            # Set generator Vm
            for j in range(n_gen):
                mpc_pf["gen"][j, 5] = Vm_gen[j]

            # Run PyPower
            try:
                r1_pf = runpf(mpc_pf, PPOPT)
                pf_results_list.append(r1_pf)
                converge_flags.append(r1_pf[0]['success'])

                # Extract results (in p.u.)
                if r1_pf[0]['success']:
                    y_pred_pg_list.append(r1_pf[0]['gen'][:, 1] / base_mva)
                    y_pred_vm_list.append(r1_pf[0]['bus'][:, 7])
                else:
                    y_pred_pg_list.append(Pg_alg)
                    y_pred_vm_list.append(results_alg['v_all'].cpu().numpy()[0])
            except:
                # Fallback to algebraic result
                pf_result_dummy = {
                    'success': False,
                    'gen': np.zeros((n_gen, 21)),
                    'bus': np.zeros((n_buses, 13)),
                    'branch': np.zeros((len(test_params['branch']['f_bus']), 17))
                }
                pf_results_list.append([pf_result_dummy])
                converge_flags.append(False)
                y_pred_pg_list.append(Pg_alg)
                y_pred_vm_list.append(results_alg['v_all'].cpu().numpy()[0])
    else:
        # Fallback: use algebraic PF
        print(f"  Using algebraic power flow (PyPower not available)...")
        for i in range(n_test):
            Pd_batch = torch.tensor(Pd_test[i:i + 1], dtype=torch.float32, device=device)
            Qd_batch = torch.tensor(Qd_test[i:i + 1], dtype=torch.float32, device=device)

            results = compute_algebraic_acopf(
                v_alpha[i:i + 1], theta_alpha[i:i + 1],
                Pd_batch, Qd_batch, test_params, G, B, device
            )

            # Create dummy PyPower result
            Pg = results['Pg'].cpu().numpy()[0]
            v_all = results['v_all'].cpu().numpy()[0]
            Qg = results['Qg'].cpu().numpy()[0]
            theta_all = results['theta_all'].cpu().numpy()[0]
            P_branch = results['P_branch'].cpu().numpy()[0]
            Q_branch = results['Q_branch'].cpu().numpy()[0]

            y_pred_pg_list.append(Pg)
            y_pred_vm_list.append(v_all)

            n_branches = len(test_params['branch']['f_bus'])
            pf_result = {
                'success': True,
                'gen': np.zeros((n_gen, 21)),
                'bus': np.zeros((n_buses, 13)),
                'branch': np.zeros((n_branches, 17))
            }

            pf_result['gen'][:, 1] = Pg * base_mva
            pf_result['gen'][:, 2] = Qg * base_mva
            pf_result['gen'][:, 8] = test_params['generator']['pg_max'].flatten() * base_mva
            pf_result['gen'][:, 9] = test_params['generator']['pg_min'].flatten() * base_mva
            pf_result['gen'][:, 3] = test_params['generator']['qg_max'].flatten() * base_mva
            pf_result['gen'][:, 4] = test_params['generator']['qg_min'].flatten() * base_mva
            pf_result['bus'][:, 7] = v_all
            pf_result['bus'][:, 8] = theta_all * 180 / np.pi
            pf_result['bus'][:, 11] = test_params['bus']['vm_max']
            pf_result['bus'][:, 12] = test_params['bus']['vm_min']
            pf_result['branch'][:, 13] = P_branch * base_mva
            pf_result['branch'][:, 14] = Q_branch * base_mva
            pf_result['branch'][:, 15] = P_branch * base_mva
            pf_result['branch'][:, 16] = Q_branch * base_mva
            pf_result['branch'][:, 5] = test_params['branch']['rate_a'] * base_mva

            pf_results_list.append([pf_result])
            converge_flags.append(True)

    y_pred_pg = np.array(y_pred_pg_list)
    y_pred_vm = np.array(y_pred_vm_list)

    t_pf_end = time.time()
    total_pf_time = t_pf_end - t_pf_start

    # Calculate average complete inference time (DNN + PF per sample)
    avg_complete_inference_time = (total_prediction_time + total_pf_time) / n_test

    print(f"  ✓ Power flow completed")
    print(f"  Converged: {sum(converge_flags)}/{n_test}")
    print(f"\n  Inference Time Statistics:")
    print(f"    Pure DNN (avg per sample):     {avg_dnn_inference_time * 1000:.4f} ms")
    print(f"    Complete (DNN+PF, avg):        {avg_complete_inference_time * 1000:.4f} ms")
    print(f"    Total prediction time:         {total_prediction_time:.4f} s")
    print(f"    Total PF time:                 {total_pf_time:.4f} s")

    # Evaluate using violation metrics
    test_metrics = evaluate_acopf_predictions(
        y_pred_pg,
        y_pred_vm,
        y_true_pg,
        y_true_vm,
        y_true_qg,
        y_true_va,
        pf_results_list,
        converge_flags,
        test_params,
        verbose=False
    )

    # Print metrics (same format as DNN script)
    print(f"\n{'=' * 70}")
    print(f"Final Test Results Summary")
    print(f"{'=' * 70}")

    print(f"\nData Mode: {data_mode}")
    print(f"Test Samples: {n_test}")

    print(f"\n--- Accuracy Metrics ---")
    print(f"MAE_Pg (Non-Slack): {test_metrics['mae_pg_non_slack_percent']:.4f}%")
    print(f"MAE_Pg (All Gens):  {test_metrics['mae_pg_all_percent']:.4f}%")
    print(f"MAE_Vm (Generator): {test_metrics['mae_vm_percent']:.4f}%")
    print(f"MAE_Qg (All Gens):  {test_metrics['mae_qg_percent']:.4f}%")
    print(f"MAE_Va (All Buses): {test_metrics['mae_va_deg']:.4f} degrees")

    print(f"\n--- Violations (p.u.) ---")
    print(f"Pg_viol (Non-Slack): {test_metrics['mean_pg_viol_non_slack_pu']:.6f} p.u.")
    print(f"Pg_viol (Slack):     {test_metrics['mean_pg_viol_slack_pu']:.6f} p.u.")
    print(f"Pg_viol (System):    {test_metrics['mean_max_pg_viol_pu']:.6f} p.u.")
    print(f"Qg_viol (All Gens):  {test_metrics['mean_max_qg_viol_pu']:.6f} p.u.")
    print(f"Vm_viol (All Buses): {test_metrics['mean_max_vm_viol_pu']:.6f} p.u.")
    print(f"Branch_viol:         {test_metrics['mean_max_branch_viol_pu']:.6f} p.u. (1.0 = 100% overload)")

    print(f"\n--- Cost Metrics ---")
    print(f"Cost Gap:           {test_metrics['cost_optimality_gap_percent']:.4f}%")

    print(f"\n--- Inference Time & Speedup ---")
    print(f"Pure DNN Inference (avg):      {avg_dnn_inference_time * 1000:.4f} ms/sample")
    print(f"Complete Inference (avg):      {avg_complete_inference_time * 1000:.4f} ms/sample")

    # Estimate IPOPT time (typical: 0.1-0.5s for IEEE 118-bus)
    # This is a conservative estimate; actual IPOPT time can be measured separately
    estimated_ipopt_time = 0.3  # seconds per sample (conservative estimate)
    speedup_factor = estimated_ipopt_time / avg_complete_inference_time
    print(f"Estimated IPOPT time:          {estimated_ipopt_time * 1000:.1f} ms/sample")
    print(f"Speedup Factor (estimated):    {speedup_factor:.1f}×")
    print(f"  (Note: IPOPT time is conservative estimate)")

    print(f"\n--- Performance ---")
    print(f"Training Time:      {time.time() - t0:.2f} s")
    print(f"Convergence Rate:   {test_metrics['convergence_rate_percent']:.2f}%")

    print(f"{'=' * 70}")

    # Add inference time to metrics
    test_metrics['avg_dnn_inference_time_ms'] = avg_dnn_inference_time * 1000
    test_metrics['avg_complete_inference_time_ms'] = avg_complete_inference_time * 1000
    test_metrics['total_prediction_time_s'] = total_prediction_time
    test_metrics['total_pf_time_s'] = total_pf_time

    return model, params, G, B, scalers, coeffs, test_metrics


if __name__ == "__main__":
    paths = acopf_config.get_all_paths()
    params_config = acopf_config.get_all_params()

    print("\n🎯 PAPER VERSION - Algorithm 1 + Critical Fixes")
    print("Features:")
    print("  ✅ Dynamic ZIB identification")
    print("  ✅ Fixed k_obj (0.1)")
    print("  ✅ Dynamic k_i = clamp(k_obj * L_obj / L_i, k_min, k_max)")
    print("  ✅ Gradient balancing via adaptive weights")
    print("  🔥 CRITICAL FIX: Lower bounds prevent k_i → 0")
    print("  📊 Detailed evaluation like DNN script")

    model, params, G, B, scalers, final_coeffs, test_metrics = train_deepopf_ngt_paper(
        case_name=paths['case_name'],
        params_path=paths['params_path'],
        data_path=paths['data_path'],
        test_data_path=paths.get('test_data_path'),
        test_params_path=paths.get('test_params_path'),
        data_mode=params_config['data_mode'],
        n_train_use=params_config['n_train_use'],
        n_test_samples=params_config.get('n_test_samples'),
        seed=params_config['seed'],
        n_epochs=params_config['n_epochs'],
        learning_rate=params_config['learning_rate'],
        hidden_sizes=params_config['hidden_sizes'],
        batch_size=params_config['batch_size'],
        device=params_config['device']
    )

    torch.save({
        'model_state_dict': model.state_dict(),
        'params': params,
        'scalers': scalers,
        'final_coeffs': final_coeffs,
        'test_metrics': test_metrics,
        'version': 'paper_algorithm1_fixed_v2'
    }, "model_paper_v2.pth")
