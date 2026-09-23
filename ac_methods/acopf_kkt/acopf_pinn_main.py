# -*- coding: utf-8 -*-
"""KKT-informed PINN for ACOPF: three branches trained on primal and dual labels
plus a KKT residual evaluated on unlabelled collocation points.

Metrics are reported twice: directly on the network output, and after a power flow has
been solved at the predicted setpoints.
"""

import numpy as np
import torch
import time
import os
import sys

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
        reconstruct_full_pg,
    )
    from ac_configuration.acopf_duals import load_dual_by_ids, load_dual_sorted
    from ac_configuration.acopf_evaluation_metrics import (
        compute_mae_absolute,
        compute_mae_percentage,
        evaluate_acopf_predictions,
    )
    from ac_configuration.acopf_pypower import load_case_from_csv, solve_pf_setpoints
except ImportError as e:
    print(f"Error: Unable to import from ac_configuration/ ({e})")
    sys.exit(1)

from acopf_pinnmodel import PinnModel

GLOBAL_CASE_DATA = None


def load_dual_variables(duals_dir, case_name, params, n_samples):
    """Load every dual variable this method supervises on, with columns realigned.

    Per-generator duals are ordered by generator id; per-bus and per-branch duals are
    reordered to CSV row order, which is what the rest of the pipeline indexes by.
    """
    bus_ids = params['general']['bus_ids']
    branch_ids = params['general']['branch_ids']

    duals = {}
    for key in ('mu_pg_min', 'mu_pg_max', 'mu_qg_min', 'mu_qg_max'):
        duals[key] = load_dual_sorted(duals_dir, case_name, key)[:n_samples]

    for key in ('mu_vm_min', 'mu_vm_max', 'lambda_kcl_r', 'lambda_kcl_i'):
        duals[key] = load_dual_by_ids(duals_dir, case_name, key, bus_ids)[:n_samples]

    for key in ('mu_sm_fr', 'mu_sm_to'):
        duals[key] = load_dual_by_ids(duals_dir, case_name, key, branch_ids)[:n_samples]

    print(f"  Dual variables loaded from: {duals_dir}")
    for k, v in duals.items():
        print(f"    {k}: shape={v.shape}, non-zero={int(np.sum(np.abs(v) > 1e-6))}")

    return duals


def compute_rect_voltage(vm, va_rad):
    """Convert polar voltage to the rectangular form the V branch predicts."""
    return np.hstack([vm * np.cos(va_rad), vm * np.sin(va_rad)]).astype('float32')


def prepare_pinn_targets(raw_data, indices, params, duals):
    """Build the target dict for a set of samples, in p.u. and unscaled.

    Passing duals=None yields zero dual targets, which is how collocation points are
    built: their mask is zero, so those entries never reach the loss.
    """
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_gen']
    n_gen_ns = params['general']['n_gen_non_slack']
    n_br = params['general']['n_branches']

    targets = {
        'pg_qg': np.hstack([
            raw_data['pg_non_slack'][indices],
            raw_data['qg'][indices],
        ]).astype('float32'),
        'v_rect': compute_rect_voltage(
            raw_data['vm'][indices], raw_data['va'][indices]),
    }

    if duals is None:
        n = len(indices)
        targets['lambda_p'] = np.zeros((n, 2 * n_buses), dtype='float32')
        targets['mu_g_u'] = np.zeros((n, n_gen_ns + n_gen), dtype='float32')
        targets['mu_g_d'] = np.zeros((n, n_gen_ns + n_gen), dtype='float32')
        targets['mu_v_u'] = np.zeros((n, n_buses), dtype='float32')
        targets['mu_v_d'] = np.zeros((n, n_buses), dtype='float32')
        targets['mu_sm_fr'] = np.zeros((n, n_br), dtype='float32')
        targets['mu_sm_to'] = np.zeros((n, n_br), dtype='float32')
        return targets

    targets['lambda_p'] = np.hstack([
        duals['lambda_kcl_r'][indices],
        duals['lambda_kcl_i'][indices],
    ]).astype('float32')

    # The G branch predicts non-slack Pg and all Qg, so its bound duals are sliced
    # to match that layout
    targets['mu_g_u'] = np.hstack([
        duals['mu_pg_max'][indices][:, non_slack_gen_idx],
        duals['mu_qg_max'][indices],
    ]).astype('float32')
    targets['mu_g_d'] = np.hstack([
        duals['mu_pg_min'][indices][:, non_slack_gen_idx],
        duals['mu_qg_min'][indices],
    ]).astype('float32')

    targets['mu_v_u'] = duals['mu_vm_max'][indices].astype('float32')
    targets['mu_v_d'] = duals['mu_vm_min'][indices].astype('float32')
    targets['mu_sm_fr'] = duals['mu_sm_fr'][indices].astype('float32')
    targets['mu_sm_to'] = duals['mu_sm_to'][indices].astype('float32')

    return targets


def compute_direct_metrics(pg_ns_np, vm_all_np, va_all_np, qg_all_np,
                           y_true_pg, y_true_vm, y_true_qg, y_true_va_rad,
                           pf_results_list, converge_flags, params, gen_bus_indices):
    """Metrics taken straight from the network output, with no power flow involved.

    These are the paper's Table 3 numbers. The only exceptions are slack Pg and the
    cost gap, which the network does not predict and which therefore must come from
    the power flow even here.
    """
    n_samples = len(pg_ns_np)
    slack_gen_mask = params['general']['slack_gen_mask']
    ns_idx = params['general']['non_slack_gen_idx']

    pg_min = params['generator']['pg_min'].flatten()
    pg_max = params['generator']['pg_max'].flatten()
    qg_min = params['generator']['qg_min'].flatten()
    qg_max = params['generator']['qg_max'].flatten()
    vm_min = params['bus']['vm_min']
    vm_max = params['bus']['vm_max']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    pg_viol = (np.maximum(0, pg_min[ns_idx] - pg_ns_np)
               + np.maximum(0, pg_ns_np - pg_max[ns_idx]))
    qg_viol = np.maximum(0, qg_min - qg_all_np) + np.maximum(0, qg_all_np - qg_max)
    vm_viol = np.maximum(0, vm_min - vm_all_np) + np.maximum(0, vm_all_np - vm_max)

    # Slack Pg only exists once the power flow has run, so its violation is read off
    # the converged results even though the rest of this block is prediction-only
    slack_gen_idx = np.where(slack_gen_mask)[0]
    slack_viols = []
    if len(slack_gen_idx) > 0:
        for i in range(n_samples):
            if converge_flags[i]:
                gen = pf_results_list[i][0]['gen']
                viols = (np.maximum(0, gen[:, 9] - gen[:, 1])
                         + np.maximum(0, gen[:, 1] - gen[:, 8]))
                slack_viols.append(
                    viols[slack_gen_idx[0]] / params['general']['BASE_MVA'])

    # Branch flow from the predicted voltages via the pi-model, in power form
    f_idx = np.array([bus_id_to_idx[int(b)] for b in params['branch']['f_bus']])
    t_idx = np.array([bus_id_to_idx[int(b)] for b in params['branch']['t_bus']])

    r_pu = params['branch']['r_pu'].astype(np.float64)
    x_pu = params['branch']['x_pu'].astype(np.float64)
    z_sq = np.maximum(r_pu ** 2 + x_pu ** 2, 1e-20)
    g_br = (r_pu / z_sq).astype(np.float32)
    b_br = (-x_pu / z_sq).astype(np.float32)

    # Unrated branches use the same threshold as the evaluation module, and NaN
    # ratings must be caught explicitly or every violation becomes NaN
    rate_a = params['branch']['rate_a'].astype(np.float64).copy()
    unlimited = ~np.isfinite(rate_a) | (rate_a <= 0) | (rate_a >= 9000)
    rate_a[unlimited] = np.inf

    vi = vm_all_np[:, f_idx]
    vj = vm_all_np[:, t_idx]
    theta_ij = va_all_np[:, f_idx] - va_all_np[:, t_idx]
    pf_flow = g_br * vi ** 2 - vi * vj * (b_br * np.sin(theta_ij)
                                          + g_br * np.cos(theta_ij))
    qf_flow = -b_br * vi ** 2 - vi * vj * (g_br * np.sin(theta_ij)
                                           - b_br * np.cos(theta_ij))
    branch_viol = np.maximum(0, pf_flow ** 2 + qf_flow ** 2 - rate_a ** 2)

    return {
        'direct_mae_pg_percent': compute_mae_percentage(
            y_true_pg[:, ~slack_gen_mask], pg_ns_np),
        'direct_mae_vm_percent': compute_mae_percentage(
            y_true_vm[:, gen_bus_indices], vm_all_np[:, gen_bus_indices]),
        'direct_mae_qg_percent': compute_mae_percentage(y_true_qg, qg_all_np),
        'direct_mae_va_deg': compute_mae_absolute(
            y_true_va_rad * (180.0 / np.pi), va_all_np * (180.0 / np.pi)),
        'direct_pg_viol_pu': float(np.mean(np.max(pg_viol, axis=1))),
        'direct_pg_slack_viol_pu': float(np.mean(slack_viols)) if slack_viols else 0.0,
        'direct_qg_viol_pu': float(np.mean(np.max(qg_viol, axis=1))),
        'direct_vm_viol_pu': float(np.mean(np.max(vm_viol, axis=1))),
        'direct_branch_viol_pu': float(np.mean(np.max(branch_viol, axis=1))),
    }


def evaluate_split(model, X, indices, raw_data, params, device, split_name, verbose=True):
    """Evaluate the network directly and again after a power flow at its setpoints."""
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()
    pg_ns_pred, vm_gen_pred, vm_all_pred, va_all_pred, qg_all_pred = \
        model.predict_for_evaluation(X.to(device))

    pg_ns_np = pg_ns_pred.cpu().numpy()
    vm_gen_np = vm_gen_pred.cpu().numpy()
    vm_all_np = vm_all_pred.cpu().numpy()
    va_all_np = va_all_pred.cpu().numpy()
    qg_all_np = qg_all_pred.cpu().numpy()

    n_gen = params['general']['n_gen']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_bus_indices = np.array(
        [bus_id_to_idx[int(gid)] for gid in params['general']['gen_bus_ids']])

    y_pred_pg_full = reconstruct_full_pg(pg_ns_np, params)
    y_pred_vm_for_pf = np.full((len(X), n_buses), 1.0, dtype=np.float32)
    y_pred_vm_for_pf[:, gen_bus_indices] = vm_gen_np

    y_true_pg = raw_data['pg'][indices]
    y_true_vm = raw_data['vm'][indices]
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

    x_raw = raw_data['x'][indices]
    pd_pu = x_raw[:, :n_loads]
    qd_pu = x_raw[:, n_loads:]

    n_samples = len(X)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_setpoints(
                pd_pu[i], qd_pu[i], pg_ns_np[i], vm_gen_np[i],
                params, GLOBAL_CASE_DATA)
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except Exception:
            pf_results_list.append((
                {'success': False,
                 'gen': np.zeros((n_gen, 21)),
                 'bus': np.zeros((n_buses, 13)),
                 'branch': np.zeros((1, 17))},
            ))
            converge_flags.append(False)

    if verbose:
        print(f"  Converged: {sum(converge_flags)}/{n_samples}")

    pf_metrics = evaluate_acopf_predictions(
        y_pred_pg_full, y_pred_vm_for_pf,
        y_true_pg, y_true_vm, y_true_qg, y_true_va_rad,
        pf_results_list, converge_flags, params, verbose=verbose
    )

    direct_metrics = compute_direct_metrics(
        pg_ns_np, vm_all_np, va_all_np, qg_all_np,
        y_true_pg, y_true_vm, y_true_qg, y_true_va_rad,
        pf_results_list, converge_flags, params, gen_bus_indices)

    # The network cannot produce slack Pg, so cost is only defined after the flow
    direct_metrics['direct_cost_gap_percent'] = \
        pf_metrics['cost_optimality_gap_percent']

    return {**pf_metrics, **direct_metrics}


def print_results(m, case_name, train_time, latency_ms):
    """Print the direct and post-power-flow metric blocks side by side."""
    print(f"\n{'=' * 70}")
    print(f"Final Results Summary")
    print(f"{'=' * 70}")
    print(f"\nCase: {case_name}")

    print(f"\n--- Network Output (no power flow) ---")
    print(f"MAE: Pg={m['direct_mae_pg_percent']:.4f}%  "
          f"Vm={m['direct_mae_vm_percent']:.4f}%  "
          f"Qg={m['direct_mae_qg_percent']:.4f}%  "
          f"Va={m['direct_mae_va_deg']:.4f} deg")
    print(f"Viol: Pg(non-slack)={m['direct_pg_viol_pu']:.6f}  "
          f"Pg(slack)={m['direct_pg_slack_viol_pu']:.6f}  "
          f"Qg={m['direct_qg_viol_pu']:.6f}  "
          f"Vm={m['direct_vm_viol_pu']:.6f}  "
          f"Branch={m['direct_branch_viol_pu']:.6f}")
    print(f"Cost Gap: {m['direct_cost_gap_percent']:.4f}%")

    print(f"\n--- After Power Flow ---")
    print(f"Convergence: {m['convergence_rate_percent']:.2f}% "
          f"({m['n_converged']}/{m['n_samples']})")
    print(f"MAE: Pg(non-slack)={m['mae_pg_non_slack_percent']:.4f}%  "
          f"Vm={m['mae_vm_percent']:.4f}%  "
          f"Qg={m['mae_qg_percent']:.4f}%  "
          f"Va={m['mae_va_deg']:.4f} deg")
    print(f"Viol: Pg(non-slack)={m['mean_pg_viol_non_slack_pu']:.6f}  "
          f"Pg(slack)={m['mean_pg_viol_slack_pu']:.6f}  "
          f"Qg={m['mean_max_qg_viol_pu']:.6f}  "
          f"Vm={m['mean_max_vm_viol_pu']:.6f}  "
          f"Branch={m['mean_max_branch_viol_pu']:.6f}")
    print(f"Cost Gap: {m['cost_optimality_gap_percent']:.4f}%")

    print(f"\n--- Performance ---")
    print(f"Inference Time: {latency_ms:.4f} ms/sample (forward pass only)")
    print(f"Training Time:  {train_time:.2f} s")
    print(f"{'=' * 70}")


def acopf_pinn_experiment(
        case_name,
        params_path,
        data_path,
        duals_dir,
        n_train_use=None,
        seed=42,
        n_epochs=1000,
        early_stop_patience=20,
        early_stop_min_delta=1e-6,
        learning_rate=1e-3,
        batch_size=None,
        device='cuda',
        hidden_sizes_V=(256, 128),
        hidden_sizes_G=(256, 128),
        hidden_sizes_Lg=(256, 128),
        lambda_P=10.0,
        lambda_V=10.0,
        lambda_L=1e-3,
        lambda_eps=1e-4,
        collocation_ratio=0.5,
        **kwargs  # Absorbs settings that do not apply, such as hidden_sizes
):
    """Train the KKT PINN on a random split and evaluate on the test indices."""
    global GLOBAL_CASE_DATA
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"ACOPF KKT PINN Experiment")
    print(f"{'=' * 70}")
    print(f"Device: {device_obj}")
    print(f"Case: {case_name}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # 1. Load network parameters and PyPower case data
    # ------------------------------------------------------------------
    params = load_parameters_from_csv(case_name, params_path)
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)

    # ------------------------------------------------------------------
    # 2. Load dataset. The scaled arrays only drive the split; this method feeds
    #    raw p.u. values to the network, because the KKT residual is a physical
    #    equation and is only meaningful in physical units.
    # ------------------------------------------------------------------
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']

    print(f"\n[Dataset Info]")
    print(f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_non_slack}), "
          f"Loads: {n_loads}")
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # ------------------------------------------------------------------
    # 3. Dual variables. These supervise the Lm branch, so unlike the other
    #    methods they are required rather than optional.
    # ------------------------------------------------------------------
    print(f"\n[Dual Variables]")
    duals = load_dual_variables(duals_dir, case_name, params, len(x_data_scaled))

    for key, arr in duals.items():
        if len(arr) != len(x_data_scaled):
            raise ValueError(
                f"Dual file {key} has {len(arr)} rows but the dataset has "
                f"{len(x_data_scaled)} samples; the two folders are out of sync")

    # ------------------------------------------------------------------
    # 4. Split
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = prepare_data_splits(
        x_data_scaled, y_data_scaled,
        n_train_use=n_train_use,
        seed=seed
    )

    # ------------------------------------------------------------------
    # 5. Hold back part of the training set as collocation points: their labels
    #    are discarded and only the KKT residual is applied to them.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed + 999)
    n_train_total = len(train_idx)
    n_collocation = int(n_train_total * collocation_ratio)
    n_supervised = n_train_total - n_collocation

    perm = rng.permutation(n_train_total)
    supervised_idx = train_idx[perm[:n_supervised]]
    collocation_idx = train_idx[perm[n_supervised:]]

    print(f"\n[Collocation Split]")
    print(f"  Total train: {n_train_total}")
    print(f"  Supervised: {n_supervised} ({100 * (1 - collocation_ratio):.0f}%)")
    print(f"  Collocation: {n_collocation} ({100 * collocation_ratio:.0f}%)")

    sup_targets = prepare_pinn_targets(raw_data, supervised_idx, params, duals)
    col_targets = prepare_pinn_targets(raw_data, collocation_idx, params, None)
    val_targets = prepare_pinn_targets(raw_data, val_idx, params, duals)

    X_train_raw = np.vstack([
        raw_data['x'][supervised_idx],
        raw_data['x'][collocation_idx],
    ]).astype('float32')
    mask_train = np.vstack([
        np.ones((n_supervised, 1), dtype='float32'),
        np.zeros((n_collocation, 1), dtype='float32'),
    ])
    all_targets = {k: np.vstack([sup_targets[k], col_targets[k]]) for k in sup_targets}

    X_train_t = torch.tensor(X_train_raw, dtype=torch.float32, device=device_obj)
    mask_t = torch.tensor(mask_train, dtype=torch.float32, device=device_obj)
    targets_t = {k: torch.tensor(v, dtype=torch.float32, device=device_obj)
                 for k, v in all_targets.items()}

    X_val_t = torch.tensor(raw_data['x'][val_idx], dtype=torch.float32, device=device_obj)
    val_targets_t = {k: torch.tensor(v, dtype=torch.float32, device=device_obj)
                     for k, v in val_targets.items()}
    val_mask_t = torch.ones(len(X_val_t), 1, dtype=torch.float32, device=device_obj)

    X_test = torch.tensor(raw_data['x'][test_idx], dtype=torch.float32)

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"V branch: {list(hidden_sizes_V)}")
    print(f"G branch: {list(hidden_sizes_G)}")
    print(f"Lm branch: {list(hidden_sizes_Lg)}")
    print(f"Input dim: {2 * n_loads} (raw p.u. pd + qd)")

    model = PinnModel(
        simulation_parameters=params,
        neurons_V=list(hidden_sizes_V),
        neurons_G=list(hidden_sizes_G),
        neurons_Lg=list(hidden_sizes_Lg),
        lambda_P=lambda_P,
        lambda_V=lambda_V,
        lambda_L=lambda_L,
        lambda_eps=lambda_eps,
        collocation_ratio=collocation_ratio,
    ).to(device_obj)

    # The optimizer is built after the move to the target device so that it holds
    # references to the parameters in their final location
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training params: max_epochs={n_epochs}, patience={early_stop_patience}, "
          f"lr={learning_rate}, batch_size={batch_size or 'full batch'}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # 7. Training with early stopping on validation loss
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Training Progress")
    print(f"{'=' * 70}")

    n_train = len(X_train_t)
    batch_size = batch_size or n_train
    n_batches = (n_train + batch_size - 1) // batch_size
    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = epoch_kkt = epoch_mae_g = epoch_mae_v = 0.0
        indices = torch.randperm(n_train, device=device_obj)

        for i in range(n_batches):
            batch_idx = indices[i * batch_size:min((i + 1) * batch_size, n_train)]
            tgt_batch = {k: v[batch_idx] for k, v in targets_t.items()}

            optimizer.zero_grad()
            outputs = model(X_train_t[batch_idx])
            total_loss, loss_dict = model.compute_loss(
                outputs, tgt_batch, mask_t[batch_idx])
            total_loss.backward()
            optimizer.step()

            bs = len(batch_idx)
            epoch_loss += total_loss.item() * bs
            epoch_kkt += loss_dict['mae_eps'] * bs
            epoch_mae_g += loss_dict['mae_g'] * bs
            epoch_mae_v += loss_dict['mae_v'] * bs

        train_loss = epoch_loss / n_train

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = model.compute_loss(val_out, val_targets_t, val_mask_t)[0].item()

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Epoch {epoch:4d}/{n_epochs} - Train: {train_loss:.6f} - "
                  f"Val: {val_loss:.6f} | mae_g: {epoch_mae_g / n_train:.6f} "
                  f"mae_v: {epoch_mae_v / n_train:.6f} "
                  f"KKT: {epoch_kkt / n_train:.4f}")

        if val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Epoch {epoch:4d}/{n_epochs} - Train: {train_loss:.6f} - "
                      f"Val: {val_loss:.6f}")
                print(f"Early stopping triggered at epoch {epoch} "
                      f"(patience={early_stop_patience})")
                break

    model.load_state_dict({k: v.to(device_obj) for k, v in best_state_dict.items()})
    train_time = time.perf_counter() - t0
    print(f"Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
    print(f"Training completed in {train_time:.2f} seconds")

    # ------------------------------------------------------------------
    # 8. Inference latency (forward pass only)
    # ------------------------------------------------------------------
    model.eval()
    speed_x = X_test[:1].to(device_obj)
    for _ in range(10):
        model.predict_for_evaluation(speed_x)

    times = []
    for _ in range(100):
        t_start = time.perf_counter()
        model.predict_for_evaluation(speed_x)
        if device_obj.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t_start)
    latency_ms = float(np.mean(times)) * 1000

    # ------------------------------------------------------------------
    # 9. Evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    test_metrics = evaluate_split(
        model, X_test, test_idx, raw_data, params, device_obj, "Test", verbose=True)

    print_results(test_metrics, case_name, train_time, latency_ms)

    return test_metrics


if __name__ == "__main__":
    # Per-branch architectures and loss weights, following the paper's Table II
    HIDDEN_SIZES_V = [256, 128]
    HIDDEN_SIZES_G = [256, 128]
    HIDDEN_SIZES_LG = [256, 128]
    LAMBDA_P = 10.0
    LAMBDA_V = 10.0
    LAMBDA_L = 1e-3
    LAMBDA_EPS = 1e-4
    COLLOCATION_RATIO = 0.5

    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)
    print(f"\n[PINN Configuration]")
    print(f"  V branch: {HIDDEN_SIZES_V}, G branch: {HIDDEN_SIZES_G}, "
          f"Lm branch: {HIDDEN_SIZES_LG}")
    print(f"  Loss weights: P={LAMBDA_P}, V={LAMBDA_V}, L={LAMBDA_L}, "
          f"eps={LAMBDA_EPS}")
    print(f"  Collocation ratio: {COLLOCATION_RATIO}")
    print("=" * 70)

    results = acopf_pinn_experiment(
        **acopf_config.get_all_paths(),
        **acopf_config.get_all_params(),
        duals_dir=acopf_config.get_duals_path(),
        hidden_sizes_V=HIDDEN_SIZES_V,
        hidden_sizes_G=HIDDEN_SIZES_G,
        hidden_sizes_Lg=HIDDEN_SIZES_LG,
        lambda_P=LAMBDA_P,
        lambda_V=LAMBDA_V,
        lambda_L=LAMBDA_L,
        lambda_eps=LAMBDA_EPS,
        collocation_ratio=COLLOCATION_RATIO,
    )

    print("\nExperiment completed successfully!")