# -*- coding: utf-8 -*-
"""Q-correction ACOPF network: predicts box-normalized Pg/Vm, then enforces Qg limits by re-solving."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from collections import defaultdict

from pypower.runpf import runpf

# The shared modules live in ac_configuration/, a subpackage of this script's
# directory, so they resolve regardless of the working directory.
try:
    from ac_configuration import acopf_config
    from ac_configuration.acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        prepare_data_splits,
        reconstruct_full_pg
    )
    from ac_configuration.acopf_evaluation_metrics import evaluate_acopf_predictions
    from ac_configuration.acopf_pypower import get_ppopt, load_case_from_csv
except ImportError as e:
    print(f"Error: Unable to import from ac_configuration/ ({e})")
    sys.exit(1)

GLOBAL_CASE_DATA = None


class QCorrectionNN_ACOPF(nn.Module):
    """Sigmoid network; the output sigmoid confines predictions to the [0, 1] box."""

    def __init__(self, input_size, output_size, hidden_sizes=[256, 256]):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Sigmoid())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_parametrization_bounds(params):
    """Return the Pg bounds of non-slack generators and the Vm bounds of generator buses."""
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_vm_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])

    return (
        params['generator']['pg_min'][:, non_slack_gen_idx],
        params['generator']['pg_max'][:, non_slack_gen_idx],
        params['bus']['vm_min'][gen_vm_indices],
        params['bus']['vm_max'][gen_vm_indices],
    )


def parametrize_pg_vm(pg_non_slack, vm_gen, params):
    """Map physical Pg/Vm onto [0, 1] box coordinates (alpha, beta)."""
    pg_min, pg_max, vm_min, vm_max = get_parametrization_bounds(params)

    alpha = (pg_non_slack - pg_min) / (pg_max - pg_min + 1e-8)
    beta = (vm_gen - vm_min) / (vm_max - vm_min + 1e-8)
    return np.hstack([alpha, beta])


def inverse_parametrize(alpha_beta_pred, params):
    """Map [0, 1] box coordinates back to physical Pg/Vm; feasibility is by construction."""
    pg_min, pg_max, vm_min, vm_max = get_parametrization_bounds(params)
    n_gen_non_slack = params['general']['n_gen_non_slack']

    alpha = alpha_beta_pred[:, :n_gen_non_slack]
    beta = alpha_beta_pred[:, n_gen_non_slack:]

    pg_non_slack = pg_min + alpha * (pg_max - pg_min)
    vm_gen = vm_min + beta * (vm_max - vm_min)
    return pg_non_slack, vm_gen


def build_mpc_for_sample(pd, qd, pg_non_slack, params):
    """Copy the base case and apply one sample's loads and non-slack Pg setpoints."""
    BASE_MVA = params['general']['BASE_MVA']

    mpc = {
        'version': GLOBAL_CASE_DATA['version'],
        'baseMVA': BASE_MVA,
        'bus': GLOBAL_CASE_DATA['bus'].copy(),
        'gen': GLOBAL_CASE_DATA['gen'].copy(),
        'branch': GLOBAL_CASE_DATA['branch'].copy(),
        'gencost': GLOBAL_CASE_DATA['gencost'],
    }

    bus_id_to_idx = params['general']['bus_id_to_idx']
    for i, bus_id in enumerate(params['general']['load_bus_ids']):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc['bus'][bus_idx, 2] = pd[i] * BASE_MVA
            mpc['bus'][bus_idx, 3] = qd[i] * BASE_MVA

    # Only non-slack Pg is set; slack Pg is left for the solver to determine
    for i, gen_idx in enumerate(params['general']['non_slack_gen_idx']):
        mpc['gen'][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    return mpc


def solve_pf_with_qg_correction(pd, qd, pg_non_slack, vm_gen, params):
    """Solve the power flow, then re-solve with Qg clipped at buses that exceeded their limits.

    Returns (pf_result, had_correction, t_pf1, t_pf2) with stage times in seconds.
    """
    global GLOBAL_CASE_DATA
    BASE_MVA = params['general']['BASE_MVA']
    n_gen = params['general']['n_gen']
    n_buses = params['general']['n_buses']
    qg_min = params['generator']['qg_min'].flatten()
    qg_max = params['generator']['qg_max'].flatten()
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Stage 1: power flow at the predicted setpoints
    t0 = time.perf_counter()
    mpc1 = build_mpc_for_sample(pd, qd, pg_non_slack, params)
    for i in range(n_gen):
        mpc1['gen'][i, 5] = vm_gen[i]

    try:
        r1 = runpf(mpc1, get_ppopt())
    except Exception:
        fail = {'success': False,
                'gen': np.zeros((n_gen, 21)),
                'bus': np.zeros((n_buses, 13)),
                'branch': np.zeros((1, 17))}
        return (fail,), False, time.perf_counter() - t0, 0.0
    t_pf1 = time.perf_counter() - t0

    if not r1[0]['success']:
        return r1, False, t_pf1, 0.0

    # Stage 2: any generator outside its Qg range triggers the correction
    qg_pf_pu = r1[0]['gen'][:n_gen, 2] / BASE_MVA
    violation_mask = (qg_pf_pu < qg_min) | (qg_pf_pu > qg_max)

    if not np.any(violation_mask):
        return r1, False, t_pf1, 0.0

    # Stage 3: pin the violating generators at their Qg limit and convert their bus
    # from PV to PQ, so the re-solve gives up the Vm setpoint to respect the Qg limit
    t0 = time.perf_counter()
    qg_clipped_pu = np.clip(qg_pf_pu, qg_min, qg_max)
    mpc2 = build_mpc_for_sample(pd, qd, pg_non_slack, params)

    for i in range(n_gen):
        if not violation_mask[i]:
            mpc2['gen'][i, 5] = vm_gen[i]

    bus_to_gens = defaultdict(list)
    for gi, bid in enumerate(params['general']['gen_bus_ids']):
        bus_to_gens[int(bid)].append(gi)

    for bus_id, gens_at in bus_to_gens.items():
        bus_idx = bus_id_to_idx.get(bus_id)
        if bus_idx is None:
            continue
        # The slack bus keeps its type, and a bus is only converted when every
        # generator on it violates; otherwise a healthy unit can still hold Vm
        if int(mpc2['bus'][bus_idx, 1]) == 3:
            continue
        if not all(violation_mask[g] for g in gens_at):
            continue
        for g in gens_at:
            mpc2['gen'][g, 2] = float(qg_clipped_pu[g]) * BASE_MVA
        mpc2['bus'][bus_idx, 1] = 1

    try:
        r2 = runpf(mpc2, get_ppopt())
    except Exception:
        return r1, True, t_pf1, time.perf_counter() - t0

    return r2, True, t_pf1, time.perf_counter() - t0


def evaluate_split_qcorrection(model, X, indices, raw_data, params, scalers, device,
                               split_name, verbose=True):
    """Predict, run the Q-correction power flow per sample, and return the metrics."""
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()
    with torch.no_grad():
        alpha_beta_pred = model(X.to(device)).cpu().numpy()

    n_gen = params['general']['n_gen']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    BASE_MVA = params['general']['BASE_MVA']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_vm_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])

    pg_non_slack_pred, vm_gen_pred = inverse_parametrize(alpha_beta_pred, params)

    y_true_pg = raw_data['pg'][indices]
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]
    y_true_vm = raw_data['vm'][indices]

    x_raw_data = scalers['x'].inverse_transform(X.cpu().numpy())
    pd_pu = x_raw_data[:, :n_loads]
    qd_pu = x_raw_data[:, n_loads:]

    n_samples = len(X)
    pf_results_list = []
    converge_flags = []
    qg_correction_count = 0

    if verbose:
        print(f"  Computing power flow with Q-correction for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r_pf, had_correction, _, _ = solve_pf_with_qg_correction(
                pd_pu[i], qd_pu[i], pg_non_slack_pred[i], vm_gen_pred[i], params
            )
            pf_results_list.append(r_pf)
            converge_flags.append(r_pf[0]['success'])
            if had_correction:
                qg_correction_count += 1
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
        print(f"  Q-Correction applied: {qg_correction_count}/{n_samples}")

    pg_full_pred = np.zeros((n_samples, n_gen))
    y_pred_vm_all = np.full((n_samples, n_buses), 1.0)

    for i in range(n_samples):
        if converge_flags[i]:
            pg_full_pred[i, :] = pf_results_list[i][0]['gen'][:n_gen, 1] / BASE_MVA
            # Vm comes from the power flow rather than the network setpoint: the
            # correction may have converted PV buses to PQ, so the setpoint no
            # longer holds at those buses
            y_pred_vm_all[i, :] = pf_results_list[i][0]['bus'][:n_buses, 7]
        else:
            pg_full_pred[i, :] = reconstruct_full_pg(pg_non_slack_pred[i], params)
            y_pred_vm_all[i, gen_vm_indices] = vm_gen_pred[i]

    return evaluate_acopf_predictions(
        pg_full_pred, y_pred_vm_all, y_true_pg, y_true_vm, y_true_qg, y_true_va_rad,
        pf_results_list, converge_flags, params, verbose=verbose
    )


def qcorrection_nn_acopf_experiment(
        case_name,
        params_path,
        data_path,
        n_train_use=None,
        seed=42,
        n_epochs=1000,
        early_stop_patience=20,
        early_stop_min_delta=1e-6,
        learning_rate=0.001,
        hidden_sizes=[256, 256],
        batch_size=None,
        device='cuda'
):
    """Train the Q-correction network on a random split and evaluate on the test indices."""
    global GLOBAL_CASE_DATA
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"Q-Correction ACOPF Experiment")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Case: {case_name}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # 1. Load network parameters and PyPower case data
    # ------------------------------------------------------------------
    params = load_parameters_from_csv(case_name, params_path)
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)

    # ------------------------------------------------------------------
    # 2. Load dataset and fit scalers
    # ------------------------------------------------------------------
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    baseMVA = params['general']['BASE_MVA']

    print(f"\n[Dataset Info]")
    print(f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_non_slack}), "
          f"Loads: {n_loads}, Base MVA: {baseMVA}")
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # ------------------------------------------------------------------
    # 3. Re-target the labels onto box coordinates. The shared loader scales
    #    with MinMax over the dataset; this method instead needs the position
    #    within the physical Pg/Vm limits, so the scaling is undone first.
    # ------------------------------------------------------------------
    y_pg_non_slack = scalers['pg'].inverse_transform(y_data_scaled[:, :n_gen_non_slack])
    y_vm_gen = scalers['vm'].inverse_transform(y_data_scaled[:, n_gen_non_slack:])
    y_alpha_beta = parametrize_pg_vm(y_pg_non_slack, y_vm_gen, params)

    print(f"\n[Parametrization]")
    print(f"  alpha (non-slack Pg): {n_gen_non_slack}, beta (generator Vm): {n_gen}")
    print(f"  alpha range: [{y_alpha_beta[:, :n_gen_non_slack].min():.4f}, "
          f"{y_alpha_beta[:, :n_gen_non_slack].max():.4f}]")
    print(f"  beta range: [{y_alpha_beta[:, n_gen_non_slack:].min():.4f}, "
          f"{y_alpha_beta[:, n_gen_non_slack:].max():.4f}]")

    # ------------------------------------------------------------------
    # 4. Split
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = prepare_data_splits(
        x_data_scaled, y_alpha_beta,
        n_train_use=n_train_use,
        seed=seed
    )

    X_train = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32, device=device)
    Y_train = torch.tensor(y_alpha_beta[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(x_data_scaled[val_idx], dtype=torch.float32, device=device)
    Y_val = torch.tensor(y_alpha_beta[val_idx], dtype=torch.float32, device=device)
    X_test = torch.tensor(x_data_scaled[test_idx], dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # 5. Model
    # ------------------------------------------------------------------
    input_dim = x_data_scaled.shape[1]
    output_dim = y_alpha_beta.shape[1]

    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"Input dim: {input_dim} (pd + qd)")
    print(f"Output dim: {output_dim} (alpha: {n_gen_non_slack} + beta: {n_gen})")
    print(f"Network: {input_dim} -> {' -> '.join(map(str, hidden_sizes))} -> {output_dim}")
    print(f"Activation: Sigmoid throughout, including the output layer")
    print(f"Training params: max_epochs={n_epochs}, patience={early_stop_patience}, "
          f"lr={learning_rate}, batch_size={batch_size or 'full batch'}")
    print(f"{'=' * 70}")

    model = QCorrectionNN_ACOPF(input_dim, output_dim, hidden_sizes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------
    # 6. Training with early stopping on validation loss
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Training Progress")
    print(f"{'=' * 70}")

    n_train = len(X_train)
    batch_size = batch_size or n_train
    n_batches = (n_train + batch_size - 1) // batch_size
    train_losses, val_losses = [], []
    t0 = time.perf_counter()

    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        indices = torch.randperm(n_train)
        for i in range(n_batches):
            batch_idx = indices[i * batch_size:min((i + 1) * batch_size, n_train)]
            optimizer.zero_grad()
            loss = criterion(model(X_train[batch_idx]), Y_train[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
        train_losses.append(epoch_loss / n_train)

        model.eval()
        with torch.no_grad():
            val_losses.append(float(criterion(model(X_val), Y_val).item()))

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {train_losses[-1]:.6f} - "
                  f"Val Loss: {val_losses[-1]:.6f}")

        if val_losses[-1] < best_val_loss - early_stop_min_delta:
            best_val_loss = val_losses[-1]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {train_losses[-1]:.6f} - "
                      f"Val Loss: {val_losses[-1]:.6f}")
                print(f"Early stopping triggered at epoch {epoch} (patience={early_stop_patience})")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    train_time = time.perf_counter() - t0
    print(f"Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
    print(f"Training completed in {train_time:.2f} seconds")

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    test_metrics = evaluate_split_qcorrection(
        model, X_test, test_idx, raw_data, params, scalers, device, "Test", verbose=True
    )

    # ------------------------------------------------------------------
    # 8. Inference latency, split across the two stages. This calls the same
    #    solver used for evaluation, so the timing cannot drift from the method.
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(X_test[:1])

    n_timing_samples = min(100, len(X_test))
    x_raw_timing = scalers['x'].inverse_transform(X_test[:n_timing_samples].cpu().numpy())
    nn_times, pf1_times, pf2_times = [], [], []
    n_stage2_applied = 0

    for i in range(n_timing_samples):
        t_nn = time.perf_counter()
        with torch.no_grad():
            alpha_beta_single = model(X_test[i:i + 1])
        if device.type == 'cuda':
            torch.cuda.synchronize()
        pg_single, vm_single = inverse_parametrize(alpha_beta_single.cpu().numpy(), params)
        nn_times.append(time.perf_counter() - t_nn)

        _, had_correction, t_pf1, t_pf2 = solve_pf_with_qg_correction(
            x_raw_timing[i, :n_loads], x_raw_timing[i, n_loads:],
            pg_single[0], vm_single[0], params
        )
        pf1_times.append(t_pf1)
        pf2_times.append(t_pf2)
        if had_correction:
            n_stage2_applied += 1

    stage1_mean = (np.mean(nn_times) + np.mean(pf1_times)) * 1000
    stage2_mean = np.mean(pf2_times) * 1000
    total_mean = stage1_mean + stage2_mean

    # ------------------------------------------------------------------
    # 9. Results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Final Results Summary")
    print(f"{'=' * 70}")
    print(f"\nCase: {case_name}")

    print(f"\n--- Accuracy Metrics ---")
    print(f"MAE_Pg (Non-Slack): {test_metrics['mae_pg_non_slack_percent']:.4f}%")
    print(f"MAE_Vm (Generator): {test_metrics['mae_vm_percent']:.4f}%")
    print(f"MAE_Qg (All Gens):  {test_metrics['mae_qg_percent']:.4f}%")
    print(f"MAE_Va (All Buses): {test_metrics['mae_va_deg']:.4f} degrees")

    print(f"\n--- Violations (p.u.) ---")
    print(f"Pg_viol (Non-Slack): {test_metrics['mean_pg_viol_non_slack_pu']:.6f} p.u.")
    print(f"Pg_viol (Slack):     {test_metrics['mean_pg_viol_slack_pu']:.6f} p.u.")
    print(f"Qg_viol (All Gens):  {test_metrics['mean_max_qg_viol_pu']:.6f} p.u.")
    print(f"Vm_viol (All Buses): {test_metrics['mean_max_vm_viol_pu']:.6f} p.u.")
    print(f"Branch_viol:         {test_metrics['mean_max_branch_viol_pu']:.6f} p.u. "
          f"(1.0 = 100% overload)")

    print(f"\n--- Cost Metrics ---")
    print(f"Cost Gap: {test_metrics['cost_optimality_gap_percent']:.4f}%")

    print(f"\n--- Performance ---")
    print(f"Total Inference: {total_mean:.4f} ms/sample")
    print(f"  Stage 1 (NN + 1st PF):  {stage1_mean:.4f} ms "
          f"({stage1_mean / total_mean * 100:.1f}%)")
    print(f"  Stage 2 (Q-correction): {stage2_mean:.4f} ms "
          f"({stage2_mean / total_mean * 100:.1f}%)")
    print(f"  Stage 2 applied: {n_stage2_applied}/{n_timing_samples} samples")
    print(f"Training Time: {train_time:.2f} s")
    print(f"Convergence Rate: {test_metrics['convergence_rate_percent']:.2f}%")
    print(f"{'=' * 70}")

    return test_metrics


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    results = qcorrection_nn_acopf_experiment(
        **acopf_config.get_all_paths(),
        **acopf_config.get_all_params()
    )

    print("\nExperiment completed successfully!")