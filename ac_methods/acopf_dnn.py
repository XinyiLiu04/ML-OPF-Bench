# -*- coding: utf-8 -*-
"""DNN baseline for ACOPF: predicts non-slack Pg and generator-bus Vm, then runs a power flow."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

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


class TraditionalNN_ACOPF(nn.Module):
    """Fully connected ReLU network mapping loads to OPF setpoints."""

    def __init__(self, input_size, output_size, hidden_sizes=[256, 256]):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def solve_pf_custom_optimized(pd, qd, pg_non_slack, vm, params):
    """Run a power flow with the predicted setpoints; the slack bus absorbs the imbalance."""
    global GLOBAL_CASE_DATA
    BASE_MVA = params['general']['BASE_MVA']

    mpc_pf = {
        'version': GLOBAL_CASE_DATA['version'],
        'baseMVA': GLOBAL_CASE_DATA['baseMVA'],
        'bus': GLOBAL_CASE_DATA['bus'].copy(),
        'gen': GLOBAL_CASE_DATA['gen'].copy(),
        'branch': GLOBAL_CASE_DATA['branch'],
        'gencost': GLOBAL_CASE_DATA['gencost']
    }

    load_bus_ids = params['general']['load_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc_pf["bus"][bus_idx, 2] = pd[i] * BASE_MVA
            mpc_pf["bus"][bus_idx, 3] = qd[i] * BASE_MVA

    # Only non-slack Pg is set; slack Pg is left for the solver to determine
    for i, gen_idx in enumerate(params['general']['non_slack_gen_idx']):
        mpc_pf["gen"][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    for i in range(params['general']['n_gen']):
        mpc_pf["gen"][i, 5] = vm[i]

    return runpf(mpc_pf, get_ppopt())


def evaluate_split(model, X, indices, raw_data, params, scalers, device, split_name, verbose=True):
    """Predict, run power flow per sample, and return the evaluation metrics."""
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()
    with torch.no_grad():
        y_pred_scaled_np = model(X.to(device)).cpu().numpy()

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Model output layout: [pg_non_slack, vm_gen]
    y_pred_pg_non_slack = scalers['pg'].inverse_transform(y_pred_scaled_np[:, :n_gen_non_slack])
    y_pred_vm_gen = scalers['vm'].inverse_transform(y_pred_scaled_np[:, n_gen_non_slack:])

    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)

    # Scatter generator Vm into an all-bus array; load buses are unpredicted, so they
    # are set to nominal voltage and excluded from MAE_Vm downstream
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_all = np.full((len(X), n_buses), 1.0, dtype=y_pred_vm_gen.dtype)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen

    y_true_pg = raw_data['pg'][indices]
    y_true_vm = raw_data['vm'][indices]
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

    x_raw_data = scalers['x'].inverse_transform(X.cpu().numpy())
    pd_pu = x_raw_data[:, :n_loads]
    qd_pu = x_raw_data[:, n_loads:]

    n_samples = len(X)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_custom_optimized(
                pd_pu[i], qd_pu[i], y_pred_pg_non_slack[i], y_pred_vm_gen[i], params
            )
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

    return evaluate_acopf_predictions(
        y_pred_pg_full,
        y_pred_vm_all,
        y_true_pg,
        y_true_vm,
        y_true_qg,
        y_true_va_rad,
        pf_results_list,
        converge_flags,
        params,
        verbose=verbose
    )


def traditional_nn_acopf_experiment(
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
    """Train the DNN on a random split and evaluate it on the held-out test indices."""
    global GLOBAL_CASE_DATA
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"ACOPF DNN Experiment")
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
    # 3. Split
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = prepare_data_splits(
        x_data_scaled, y_data_scaled,
        n_train_use=n_train_use,
        seed=seed
    )

    X_train = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32, device=device)
    Y_train = torch.tensor(y_data_scaled[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(x_data_scaled[val_idx], dtype=torch.float32, device=device)
    Y_val = torch.tensor(y_data_scaled[val_idx], dtype=torch.float32, device=device)
    X_test = torch.tensor(x_data_scaled[test_idx], dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    input_dim = x_data_scaled.shape[1]
    output_dim = y_data_scaled.shape[1]

    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"Input dim: {input_dim} (pd + qd)")
    print(f"Output dim: {output_dim} (pg_non_slack: {n_gen_non_slack} + "
          f"vm_gen: {output_dim - n_gen_non_slack})")
    print(f"Network: {input_dim} -> {' -> '.join(map(str, hidden_sizes))} -> {output_dim}")
    print(f"Training params: max_epochs={n_epochs}, patience={early_stop_patience}, "
          f"lr={learning_rate}, batch_size={batch_size or 'full batch'}")
    print(f"{'=' * 70}")

    model = TraditionalNN_ACOPF(input_dim, output_dim, hidden_sizes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------
    # 5. Training with early stopping on validation loss
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
    # 6. Evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    test_metrics = evaluate_split(
        model, X_test, test_idx, raw_data, params, scalers, device, "Test", verbose=True
    )

    # ------------------------------------------------------------------
    # 7. Inference latency (single sample, after warm-up)
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(X_test[:1])

    times = [time.perf_counter() for _ in range(101)]
    with torch.no_grad():
        for i in range(100):
            model(X_test[:1])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times[i + 1] = time.perf_counter()

    latency_ms = np.mean(np.diff(times)) * 1000

    # ------------------------------------------------------------------
    # 8. Results
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
    print(f"Inference Time: {latency_ms:.4f} ms/sample")
    print(f"Training Time:  {train_time:.2f} s")
    print(f"Convergence Rate: {test_metrics['convergence_rate_percent']:.2f}%")
    print(f"{'=' * 70}")

    return test_metrics


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    results = traditional_nn_acopf_experiment(
        **acopf_config.get_all_paths(),
        **acopf_config.get_all_params()
    )

    print("\nExperiment completed successfully!")