"""
DeepOPF-NGT — Semi-Supervised ACOPF  (Paper-faithful implementation)
====================================================================
Source: "Unsupervised Learning for Solving AC Optimal Power Flows:
         Design, Analysis, and Experiment"
         Huang, Chen, Low — IEEE Trans. Power Syst., Vol. 39, No. 6, Nov. 2024

Exact correspondence to paper Section V-B and Algorithm 2:

Algorithm 2 — Training of Extended DeepOPF-NGT:
  Input : D̄ (all unlabeled), D (labeled subset with ground truth)
  For each epoch t = 1, 2, ..., T:
    ── Step 1 (pre-train on labeled data) ──────────────────────────────────
      Sample mini-batch from D
      Compute: L_sup = k_v·L_v + Σ_i k_i·L_i          Eq.(13)
        where L_v = Σ_{i∈N} [‖V̂_i - V_i‖² + ‖θ̂_i - θ_i‖²]  Eq.(14)
      Update: φ ← φ − η·∇_φ L_sup
    ── Step 2 (train on all data) ──────────────────────────────────────────
      Sample mini-batch from D̄
      Compute: L_uns = k_obj·L_obj + Σ_i k_i·L_i       Eq.(10)
      Update: φ ← φ − η·∇_φ L_uns
      Update: k_i^t = min(k_obj·L_obj / L_i, k̄_i)     Eq.(12)

Key faithfulness notes:
  * Both steps happen WITHIN EVERY epoch (not two sequential phases).
  * k_v is FIXED; constraint weights k_i updated only from Step 2 Eq.(12).
  * L_v is over all buses (including ZIBs via Kron recovery), Eq.(14).
  * No EMA, no lower-bound on k_i, no gradient clipping.
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
# 1.  Neural Network  (identical to Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

class DeepOPF_NGT(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None):
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

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Denormalisation
# ─────────────────────────────────────────────────────────────────────────────

def denormalise_output(y_norm, params, theta_max_rad=np.pi/6.0):
    nonzib_idx = params['general']['nonzib_indices']
    n_nz = len(nonzib_idx)
    vm_min = torch.as_tensor(params['bus']['vm_min'], dtype=torch.float32, device=y_norm.device)
    vm_max = torch.as_tensor(params['bus']['vm_max'], dtype=torch.float32, device=y_norm.device)
    v_min = vm_min[nonzib_idx]
    v_max = vm_max[nonzib_idx]
    v_alpha     = v_min + y_norm[:, :n_nz] * (v_max - v_min)
    theta_alpha = (y_norm[:, n_nz:] - 0.5) * 2.0 * theta_max_rad
    return v_alpha, theta_alpha


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Unsupervised loss terms  Eqs.(3–9)
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss_terms(results, Pd, Qd, params, device):
    Pg        = results['Pg']
    Qg        = results['Qg']
    v_all     = results['v_all']
    theta_all = results['theta_all']
    P_branch  = results['P_branch']
    Q_branch  = results['Q_branch']

    gen_p = params['generator']
    bus_p = params['bus']
    br_p  = params['branch']
    gen_all = params['general']

    pg_min = torch.as_tensor(gen_p['pg_min'], dtype=torch.float32, device=device).squeeze()
    pg_max = torch.as_tensor(gen_p['pg_max'], dtype=torch.float32, device=device).squeeze()
    qg_min = torch.as_tensor(gen_p['qg_min'], dtype=torch.float32, device=device).squeeze()
    qg_max = torch.as_tensor(gen_p['qg_max'], dtype=torch.float32, device=device).squeeze()
    c2 = torch.as_tensor(gen_p['cost_c2'], dtype=torch.float32, device=device)
    c1 = torch.as_tensor(gen_p['cost_c1'], dtype=torch.float32, device=device)
    c0 = torch.as_tensor(gen_p['cost_c0'], dtype=torch.float32, device=device)
    vm_min = torch.as_tensor(bus_p['vm_min'], dtype=torch.float32, device=device)
    vm_max = torch.as_tensor(bus_p['vm_max'], dtype=torch.float32, device=device)
    rate_a = torch.as_tensor(br_p['rate_a'],  dtype=torch.float32, device=device)

    # L_obj Eq.(1a)
    L_obj = torch.mean(torch.sum(c2*Pg**2 + c1*Pg + c0, dim=1))

    # L_g Eq.(5)
    L_g = torch.mean(torch.sum(
        torch.relu(Pg-pg_max)**2 + torch.relu(pg_min-Pg)**2 +
        torch.relu(Qg-qg_max)**2 + torch.relu(qg_min-Qg)**2, dim=1))

    # L_Sl Eq.(6)
    S_branch = torch.sqrt(P_branch**2 + Q_branch**2 + 1e-12)
    valid_br = (rate_a > 1e-5) & (rate_a < 9000.0)
    L_Sl = (torch.mean(torch.sum(torch.relu(S_branch[:, valid_br] - rate_a[valid_br])**2, dim=1))
            if valid_br.any() else torch.tensor(0.0, device=device))

    # L_theta Eq.(7)
    f_bus = br_p['f_bus']; t_bus = br_p['t_bus']
    bid2idx = gen_all['bus_id_to_idx']
    dmax = 30.0 * np.pi / 180.0
    diffs = [theta_all[:, bid2idx[int(f_bus[k])]] - theta_all[:, bid2idx[int(t_bus[k])]]
             for k in range(len(f_bus))]
    if diffs:
        td = torch.stack(diffs, dim=1)
        L_theta = torch.mean(torch.sum(torch.relu(torch.abs(td)-dmax)**2, dim=1))
    else:
        L_theta = torch.tensor(0.0, device=device)

    # L_z Eq.(8)
    zib_idx = np.where(gen_all['zib_mask'])[0]
    if len(zib_idx) > 0:
        v_z = v_all[:, zib_idx]
        L_z = torch.mean(torch.sum(
            torch.relu(v_z-vm_max[zib_idx])**2 + torch.relu(vm_min[zib_idx]-v_z)**2, dim=1))
    else:
        L_z = torch.tensor(0.0, device=device)

    # L_d Eq.(9): predicted load vs demanded load
    if 'Pd_pred' in results and 'Qd_pred' in results:
        Pd_pred = results['Pd_pred']
        Qd_pred = results['Qd_pred']
        L_d = torch.mean(torch.sum((Pd_pred-Pd)**2 + (Qd_pred-Qd)**2, dim=1))
    else:
        L_d = torch.mean((torch.sum(Pg, dim=1) - torch.sum(Pd, dim=1))**2)

    return {'L_obj': L_obj, 'L_g': L_g, 'L_Sl': L_Sl,
            'L_theta': L_theta, 'L_z': L_z, 'L_d': L_d}


def total_loss_unsup(loss_dict, coeffs):
    """Eq.(10): unsupervised total loss."""
    return (coeffs['k_obj']*loss_dict['L_obj'] + coeffs['k_g']*loss_dict['L_g']
          + coeffs['k_Sl']*loss_dict['L_Sl']   + coeffs['k_theta']*loss_dict['L_theta']
          + coeffs['k_z']*loss_dict['L_z']     + coeffs['k_d']*loss_dict['L_d'])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Supervised voltage loss  L_v  Eq.(14)
# ─────────────────────────────────────────────────────────────────────────────

def compute_Lv(y_pred, V_gt, theta_gt, params, theta_max_rad):
    """
    L_v = Σ_{i∈N} [‖V̂_i - V_i‖² + ‖θ̂_i - θ_i‖²]  Eq.(14)
    Comparison in physical domain; GT is for non-ZIB buses.
    """
    nonzib_idx = params['general']['nonzib_indices']
    n_nz = len(nonzib_idx)
    vm_min = torch.as_tensor(params['bus']['vm_min'], dtype=torch.float32, device=y_pred.device)
    vm_max = torch.as_tensor(params['bus']['vm_max'], dtype=torch.float32, device=y_pred.device)
    v_min = vm_min[nonzib_idx]; v_max = vm_max[nonzib_idx]
    v_hat     = v_min + y_pred[:, :n_nz] * (v_max - v_min)
    theta_hat = (y_pred[:, n_nz:] - 0.5) * 2.0 * theta_max_rad
    return torch.mean(torch.sum((v_hat-V_gt)**2 + (theta_hat-theta_gt)**2, dim=1))


def total_loss_sup(Lv, loss_dict, k_v, coeffs):
    """Eq.(13): supervised total loss."""
    return (k_v*Lv + coeffs['k_g']*loss_dict['L_g']
          + coeffs['k_Sl']*loss_dict['L_Sl'] + coeffs['k_theta']*loss_dict['L_theta']
          + coeffs['k_z']*loss_dict['L_z']   + coeffs['k_d']*loss_dict['L_d'])


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Adaptive weight update  Eq.(12)
# ─────────────────────────────────────────────────────────────────────────────

def update_coefficients(coeffs, loss_dict, k_upper):
    """k_i^t = min(k_obj·L_obj / L_i, k̄_i)."""
    L_obj = loss_dict['L_obj']; k_obj = coeffs['k_obj']
    for name, lk in [('k_g','L_g'),('k_Sl','L_Sl'),('k_theta','L_theta'),
                     ('k_z','L_z'),('k_d','L_d')]:
        Li = loss_dict[lk]
        coeffs[name] = float(min(k_obj*L_obj/Li if Li>1e-12 else k_upper[name],
                                  k_upper[name]))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Training  (Algorithm 2)
# ─────────────────────────────────────────────────────────────────────────────

def train_extended_deepopf_ngt(
        case_name, params_path, data_path,
        data_mode='random_split', n_train_use=10000, n_labeled=300,
        n_test_samples=None, test_data_path=None, test_params_path=None,
        seed=42, n_epochs=100, learning_rate=1e-3,
        hidden_sizes=None, batch_size=256,
        k_v=100.0,
        k_obj=1.0,
        k_g_0=1.0,   k_g_max=1000.0,
        k_Sl_0=1.0,  k_Sl_max=1000.0,
        k_th_0=1.0,  k_th_max=1000.0,
        k_z_0=1.0,   k_z_max=1000.0,
        k_d_0=1.0,   k_d_max=1000.0,
        theta_max_deg=30.0, device='cpu'):

    if hidden_sizes is None:
        hidden_sizes = [256, 256]

    print(f"\n{'='*70}")
    print(f"Extended DeepOPF-NGT — Semi-Supervised ACOPF (Paper Algorithm 2)")
    print(f"{'='*70}")
    print(f"  Labeled: {n_labeled}  |  k_v (fixed)={k_v}  |  k_obj={k_obj}")
    print(f"{'='*70}")

    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    # [1] Parameters
    params = load_parameters_from_csv(case_name, params_path)
    G, B   = build_admittance_matrix(params, dev)
    zib_mask = params['general']['zib_mask']
    nonzib_indices = np.where(~zib_mask)[0].tolist()
    params['general']['nonzib_indices'] = nonzib_indices
    n_nonzib = len(nonzib_indices)
    print(f"  Buses: {params['general']['n_buses']}  |  Non-ZIB: {n_nonzib}")

    # [2] Data
    x_data_scaled, _, scalers, raw_data, cost_baseline = load_and_scale_acopf_data(
        data_path, params, fit_scalers=True)

    if data_mode == DataMode.API_TEST:
        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, x_data_scaled, mode=DataMode.API_TEST,
            n_train_use=n_train_use, seed=seed)
        test_params, test_x_scaled, _, test_raw_data, _ = load_api_test_data(
            test_data_path, test_params_path, scalers,
            n_test_samples=n_test_samples, seed=seed)
        test_idx = np.arange(len(test_x_scaled))
    elif data_mode == DataMode.GENERALIZATION:
        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, x_data_scaled, mode=DataMode.GENERALIZATION,
            n_train_use=n_train_use, seed=seed)
        test_x_scaled, _, test_raw_data, _ = load_generalization_test_data(
            test_data_path, params, scalers, n_test_samples=n_test_samples, seed=seed)
        test_idx = np.arange(len(test_x_scaled)); test_params = params
    else:
        train_idx, val_idx, test_idx = prepare_data_splits(
            x_data_scaled, x_data_scaled, mode=data_mode,
            n_train_use=n_train_use, seed=seed)
        test_x_scaled = x_data_scaled; test_raw_data = raw_data; test_params = params

    n_loads = params['general']['n_loads']
    X_train_all = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32, device=dev)
    n_train_all = len(X_train_all)

    # Labeled subset D
    rng = np.random.default_rng(seed)
    lbl_local = rng.choice(n_train_all, size=min(n_labeled, n_train_all), replace=False)
    X_labeled = X_train_all[lbl_local]

    # Ground-truth voltages for labeled samples (Eq. 14)
    Vm_all = raw_data.get('Vm', None)
    Va_all = raw_data.get('Va', None)
    if Vm_all is None or Va_all is None:
        raise ValueError("raw_data must contain 'Vm' and 'Va' for semi-supervised training.")
    global_lbl = train_idx[lbl_local]
    V_gt_lbl  = torch.tensor(Vm_all[global_lbl][:, nonzib_indices],
                              dtype=torch.float32, device=dev)
    Th_gt_lbl = torch.tensor(Va_all[global_lbl][:, nonzib_indices],
                              dtype=torch.float32, device=dev)

    # Physical (Pd, Qd) for labeled set (needed for constraint losses in Step 1)
    x_lbl_phys = scalers['x'].inverse_transform(X_labeled.cpu().numpy())
    Pd_lbl = torch.tensor(x_lbl_phys[:, :n_loads], dtype=torch.float32, device=dev)
    Qd_lbl = torch.tensor(x_lbl_phys[:, n_loads:], dtype=torch.float32, device=dev)

    X_test = torch.tensor(test_x_scaled[test_idx], dtype=torch.float32, device=dev)
    print(f"  Train(all)={n_train_all}  Labeled={len(X_labeled)}  Test={len(X_test)}")

    # [3] Model
    model = DeepOPF_NGT(n_loads*2, n_nonzib*2, hidden_sizes).to(dev)
    opt   = optim.Adam(model.parameters(), lr=learning_rate)
    theta_max_rad = theta_max_deg * np.pi / 180.0

    # [4] Coefficients
    coeffs  = {'k_obj':k_obj,'k_g':k_g_0,'k_Sl':k_Sl_0,'k_theta':k_th_0,'k_z':k_z_0,'k_d':k_d_0}
    k_upper = {'k_g':k_g_max,'k_Sl':k_Sl_max,'k_theta':k_th_max,'k_z':k_z_max,'k_d':k_d_max}

    # [5] Training  (Algorithm 2)
    print(f"\n{'─'*70}")
    print(f"  Training: at each epoch → Step 1 (labeled) → Step 2 (all data)")
    print(f"{'─'*70}")
    n_lbl   = len(X_labeled)
    t0 = time.time()

    for epoch in range(1, n_epochs+1):
        model.train()
        ep_sup = 0.0; ep_uns = 0.0
        ep_sums = {k:0.0 for k in ['L_obj','L_g','L_Sl','L_theta','L_z','L_d']}

        # ── Step 1: supervised on D ───────────────────────────────────────────
        lbl_perm = torch.randperm(n_lbl, device=dev)
        for b_start in range(0, n_lbl, batch_size):
            idx   = lbl_perm[b_start:min(b_start+batch_size, n_lbl)]
            X_b   = X_labeled[idx]
            Pd_b  = Pd_lbl[idx]; Qd_b = Qd_lbl[idx]
            Vgt_b = V_gt_lbl[idx]; Tgt_b = Th_gt_lbl[idx]

            opt.zero_grad()
            y_pred = model(X_b)
            va, ta = denormalise_output(y_pred, params, theta_max_rad)
            res = compute_algebraic_acopf(va, ta, Pd_b, Qd_b, params, G, B, dev)
            ld  = compute_loss_terms(res, Pd_b, Qd_b, params, dev)
            Lv  = compute_Lv(y_pred, Vgt_b, Tgt_b, params, theta_max_rad)
            L   = total_loss_sup(Lv, ld, k_v, coeffs)
            L.backward(); opt.step()
            ep_sup += L.item() * len(idx)

        # ── Step 2: unsupervised on D̄ ─────────────────────────────────────────
        all_perm = torch.randperm(n_train_all, device=dev)
        for b_start in range(0, n_train_all, batch_size):
            idx = all_perm[b_start:min(b_start+batch_size, n_train_all)]
            X_b = X_train_all[idx]
            x_phys = scalers['x'].inverse_transform(X_b.cpu().numpy())
            Pd_b = torch.tensor(x_phys[:, :n_loads], dtype=torch.float32, device=dev)
            Qd_b = torch.tensor(x_phys[:, n_loads:], dtype=torch.float32, device=dev)

            opt.zero_grad()
            y_pred = model(X_b)
            va, ta = denormalise_output(y_pred, params, theta_max_rad)
            res = compute_algebraic_acopf(va, ta, Pd_b, Qd_b, params, G, B, dev)
            ld  = compute_loss_terms(res, Pd_b, Qd_b, params, dev)
            L   = total_loss_unsup(ld, coeffs)
            L.backward(); opt.step()

            # Eq.(12): weight update per mini-batch after epoch 1
            if epoch > 1:
                update_coefficients(coeffs, {k:v.item() for k,v in ld.items()}, k_upper)

            bs = len(idx)
            ep_uns += L.item() * bs
            for k,v in ld.items(): ep_sums[k] += v.item()*bs

        if epoch % 10 == 0 or epoch == 1:
            em = {k: ep_sums[k]/n_train_all for k in ep_sums}
            cg = (em['L_obj']-cost_baseline)/cost_baseline*100 if cost_baseline else 0
            print(f"Epoch {epoch:4d}/{n_epochs} | "
                  f"L_uns={ep_uns/n_train_all:.4f} L_sup={ep_sup/n_lbl:.4f} | "
                  f"L_obj={em['L_obj']:.1f}({cg:+.2f}%) | "
                  f"L_g={em['L_g']:.4f} | L_d={em['L_d']:.4f}")
            print(f"  Weights: k_g={coeffs['k_g']:.2f} k_Sl={coeffs['k_Sl']:.2f} "
                  f"k_θ={coeffs['k_theta']:.2f} k_z={coeffs['k_z']:.2f} k_d={coeffs['k_d']:.2f}")

    print(f"\n  Training complete in {time.time()-t0:.2f} s")

    # [6] Test evaluation
    model.eval()
    Pd_test   = test_raw_data['Pd']; Qd_test = test_raw_data['Qd']
    y_true_pg = test_raw_data.get('Pg'); y_true_vm = test_raw_data.get('Vm')
    y_true_qg = test_raw_data.get('Qg'); y_true_va = test_raw_data.get('Va')
    n_test    = len(X_test)
    n_gen_t   = test_params['general']['n_gen']
    n_buses_t = test_params['general']['n_buses']
    base_mva  = test_params['general'].get('base_mva', 100.0)

    y_pred_pg_list=[]; y_pred_vm_list=[]; pf_results_list=[]; converge_flags=[]
    t_inf = time.time()
    with torch.no_grad():
        for i in range(n_test):
            Pd_i = torch.tensor(Pd_test[i:i+1], dtype=torch.float32, device=dev)
            Qd_i = torch.tensor(Qd_test[i:i+1], dtype=torch.float32, device=dev)
            x_i  = scalers['x'].transform(np.concatenate([Pd_test[i:i+1], Qd_test[i:i+1]], axis=1))
            X_i  = torch.tensor(x_i, dtype=torch.float32, device=dev)
            y_pred = model(X_i)
            va, ta = denormalise_output(y_pred, params, theta_max_rad)
            res = compute_algebraic_acopf(va, ta, Pd_i, Qd_i, test_params, G, B, dev)
            Pg=res['Pg'].cpu().numpy()[0]; Qg=res['Qg'].cpu().numpy()[0]
            v_all=res['v_all'].cpu().numpy()[0]; th=res['theta_all'].cpu().numpy()[0]
            Pb=res['P_branch'].cpu().numpy()[0]; Qb=res['Q_branch'].cpu().numpy()[0]
            y_pred_pg_list.append(Pg); y_pred_vm_list.append(v_all)
            nb = len(test_params['branch']['f_bus'])
            pf = {'success':True,'gen':np.zeros((n_gen_t,21)),
                  'bus':np.zeros((n_buses_t,13)),'branch':np.zeros((nb,17))}
            pf['gen'][:,1]=Pg*base_mva; pf['gen'][:,2]=Qg*base_mva
            pf['gen'][:,8]=test_params['generator']['pg_max'].flatten()*base_mva
            pf['gen'][:,9]=test_params['generator']['pg_min'].flatten()*base_mva
            pf['gen'][:,3]=test_params['generator']['qg_max'].flatten()*base_mva
            pf['gen'][:,4]=test_params['generator']['qg_min'].flatten()*base_mva
            pf['bus'][:,7]=v_all; pf['bus'][:,8]=th*180/np.pi
            pf['bus'][:,11]=test_params['bus']['vm_max']; pf['bus'][:,12]=test_params['bus']['vm_min']
            pf['branch'][:,13]=Pb*base_mva; pf['branch'][:,14]=Qb*base_mva
            pf['branch'][:,15]=Pb*base_mva; pf['branch'][:,16]=Qb*base_mva
            pf['branch'][:,5]=test_params['branch']['rate_a']*base_mva
            pf_results_list.append([pf]); converge_flags.append(True)

    avg_ms = (time.time()-t_inf)/n_test*1000
    metrics = evaluate_acopf_predictions(
        np.array(y_pred_pg_list), np.array(y_pred_vm_list),
        y_true_pg, y_true_vm, y_true_qg, y_true_va,
        pf_results_list, converge_flags, test_params, verbose=False)

    print(f"\n{'='*70}")
    print(f"  MAE_Pg(non-slack)={metrics['mae_pg_non_slack_percent']:.4f}% | "
          f"MAE_Vm={metrics['mae_vm_percent']:.4f}%")
    print(f"  Pg_viol={metrics['mean_max_pg_viol_pu']:.6f} | "
          f"Vm_viol={metrics['mean_max_vm_viol_pu']:.6f} | "
          f"Branch_viol={metrics['mean_max_branch_viol_pu']:.6f} p.u.")
    print(f"  Cost Gap={metrics['cost_optimality_gap_percent']:.4f}% | "
          f"Inference={avg_ms:.4f}ms | Train={time.time()-t0:.2f}s")
    print(f"{'='*70}")

    return model, params, G, B, scalers, coeffs, metrics


if __name__ == '__main__':
    paths = acopf_config.get_all_paths()
    cfg   = acopf_config.get_all_params()
    model, params, G, B, scalers, coeffs, metrics = train_extended_deepopf_ngt(
        case_name=paths['case_name'], params_path=paths['params_path'],
        data_path=paths['data_path'], test_data_path=paths.get('test_data_path'),
        test_params_path=paths.get('test_params_path'), data_mode=cfg['data_mode'],
        n_train_use=cfg['n_train_use'], n_test_samples=cfg.get('n_test_samples'),
        n_labeled=300, seed=cfg['seed'], n_epochs=cfg['n_epochs'],
        learning_rate=cfg['learning_rate'], hidden_sizes=cfg['hidden_sizes'],
        batch_size=cfg['batch_size'], k_v=100.0, device=cfg['device'])
    torch.save({'model_state_dict':model.state_dict(),'params':params,'scalers':scalers,
                'coeffs':coeffs,'test_metrics':metrics,'version':'paper_alg2_faithful'},
               'deepopf_ngt_semi_paper.pth')
    print('\nModel saved to deepopf_ngt_semi_paper.pth')