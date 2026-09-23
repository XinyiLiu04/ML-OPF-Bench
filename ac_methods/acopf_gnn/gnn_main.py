# -*- coding: utf-8 -*-
"""Spectral GNN for ACOPF: supervised regression from the sub-optimal state.

Run generate_subopt_state.py first to produce the _subopt_*.csv files next to the
sample data, then run this script.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

from sklearn.preprocessing import MinMaxScaler

# The shared modules live in ac_configuration/, a subpackage of this script's
# directory, so they resolve regardless of the working directory.
try:
    from ac_configuration import acopf_config
    from ac_configuration.acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        prepare_data_splits,
    )
    from ac_configuration.acopf_pypower import load_case_from_csv
except ImportError as e:
    print(f"Error: Unable to import from ac_configuration/ ({e})")
    sys.exit(1)

try:
    from gnn_model import SpectralGNN_ACOPF
    from gnn_utils import (
        build_adjacency_edge_weight,
        collate_graph_batch,
        evaluate_split,
        load_subopt_state,
    )
except ImportError as e:
    print(f"Error: Unable to import the GNN modules ({e})")
    sys.exit(1)


def spectral_gnn_acopf_experiment(
        case_name,
        params_path,
        data_path,
        n_train_use=None,
        seed=42,
        n_epochs=1000,
        early_stop_patience=20,
        early_stop_min_delta=1e-6,
        learning_rate=1e-3,
        batch_size=128,
        device='cuda',
        F1=128,
        F2=64,
        K=4,
        graph_kernel='gaussian',
        graph_scale_k=1.0,
        predict_vm=True,
        **kwargs  # Absorbs settings that do not apply, such as hidden_sizes
):
    """Train the spectral GNN on a random split and evaluate on the test indices."""
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'=' * 70}")
    print(f"Spectral GNN ACOPF Experiment")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Case: {case_name}")
    print(f"Predict: pg_non_slack" + (" + vm_gen" if predict_vm else " only"))
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # 1. Load network parameters and PyPower case data
    # ------------------------------------------------------------------
    params = load_parameters_from_csv(case_name, params_path)
    case_data = load_case_from_csv(case_name, params_path)

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    baseMVA = params['general']['BASE_MVA']

    print(f"\n[Dataset Info]")
    print(f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_non_slack}), "
          f"Loads: {n_loads}, Base MVA: {baseMVA}")

    # ------------------------------------------------------------------
    # 2. Graph structure
    # ------------------------------------------------------------------
    edge_index, edge_weight = build_adjacency_edge_weight(
        params, kernel=graph_kernel, scale_k=graph_scale_k)

    # ------------------------------------------------------------------
    # 3. ACOPF labels and scalers
    # ------------------------------------------------------------------
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_loads = params['general']['n_loads']
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    if not predict_vm:
        y_data_scaled = y_data_scaled[:, :n_gen_non_slack]
        print(f"  Output restricted to pg_non_slack, Y shape: {y_data_scaled.shape}")

    # ------------------------------------------------------------------
    # 4. Sub-optimal state, which is this method's input
    # ------------------------------------------------------------------
    data_dir = os.path.dirname(data_path)
    print(f"\n[Sub-optimal State]")
    print(f"  Loading from: {data_dir}")
    subopt_x_raw, subopt_converged = load_subopt_state(data_dir, case_name, params)

    if len(subopt_x_raw) != len(x_data_scaled):
        raise ValueError(
            f"Sub-optimal state has {len(subopt_x_raw)} rows but the dataset has "
            f"{len(x_data_scaled)} samples; regenerate the _subopt_*.csv files")

    n_total = len(subopt_x_raw)
    n_conv = int(subopt_converged.sum())
    print(f"  Samples: {n_total}, DCOPF power flow converged: {n_conv} "
          f"({n_conv / n_total * 100:.1f}%)")

    # The scaler is fitted on converged rows only: the remaining rows hold whatever
    # the generator wrote for a failed solve and would distort the MinMax range
    subopt_scaler = MinMaxScaler()
    subopt_scaler.fit(subopt_x_raw[subopt_converged])
    subopt_x_scaled = subopt_scaler.transform(subopt_x_raw).astype('float32')
    scalers['subopt_x'] = subopt_scaler
    print(f"  Scaled sub-optimal state: {subopt_x_scaled.shape} (= 4 x {n_buses} buses)")

    # ------------------------------------------------------------------
    # 5. Split first, then drop the samples whose sub-optimal state failed.
    #    Splitting on the full dataset with the shared seed keeps this method's
    #    test set a subset of the other methods', so results stay comparable.
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = prepare_data_splits(
        x_data_scaled, y_data_scaled,
        n_train_use=n_train_use,
        seed=seed
    )

    def _keep_converged(idx, name):
        kept = idx[subopt_converged[idx]]
        dropped = len(idx) - len(kept)
        if dropped:
            print(f"  {name}: dropped {dropped}/{len(idx)} samples without a "
                  f"converged sub-optimal state")
        if len(kept) == 0:
            raise ValueError(f"No usable {name} sample remains")
        return kept

    print(f"\n[Filtering to converged sub-optimal states]")
    train_idx = _keep_converged(train_idx, "Train")
    val_idx = _keep_converged(val_idx, "Val")
    test_idx = _keep_converged(test_idx, "Test")

    X_train = torch.tensor(subopt_x_scaled[train_idx], dtype=torch.float32, device=device)
    Y_train = torch.tensor(y_data_scaled[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(subopt_x_scaled[val_idx], dtype=torch.float32, device=device)
    Y_val = torch.tensor(y_data_scaled[val_idx], dtype=torch.float32, device=device)
    X_test = torch.tensor(subopt_x_scaled[test_idx], dtype=torch.float32)

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    model = SpectralGNN_ACOPF(
        params, F1=F1, F2=F2, K=K, predict_vm=predict_vm).to(device)
    out_dim = n_gen_non_slack + n_gen if predict_vm else n_gen_non_slack

    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"Node features: [vm, va, p_inj, q_inj] (4 per bus, sub-optimal state)")
    print(f"Graph kernel: {graph_kernel} (k={graph_scale_k}), ChebConv K={K}")
    print(f"Hidden dims: {F1} -> {F2}, local readout at generator nodes")
    print(f"Output dim: {out_dim}"
          + (f" (pg_non_slack: {n_gen_non_slack} + vm_gen: {n_gen})" if predict_vm
             else f" (pg_non_slack: {n_gen_non_slack})"))
    print(f"Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training params: max_epochs={n_epochs}, patience={early_stop_patience}, "
          f"lr={learning_rate}, batch_size={batch_size}")
    print(f"{'=' * 70}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ei_dev = edge_index.to(device)
    ew_dev = edge_weight.to(device)

    # ------------------------------------------------------------------
    # 7. Training with early stopping on validation loss
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Training Progress")
    print(f"{'=' * 70}")

    n_train = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size
    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(n_train)

        for i in range(n_batches):
            b = perm[i * batch_size:min((i + 1) * batch_size, n_train)]
            optimizer.zero_grad()
            nf, bei, bew, B = collate_graph_batch(
                X_train[b], ei_dev, ew_dev, n_buses, device)
            loss = criterion(model(nf, bei, bew, batch_size=B), Y_train[b])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b)

        train_loss = epoch_loss / n_train

        model.eval()
        with torch.no_grad():
            nf_v, bei_v, bew_v, B_v = collate_graph_batch(
                X_val, ei_dev, ew_dev, n_buses, device)
            val_loss = float(criterion(model(nf_v, bei_v, bew_v, batch_size=B_v), Y_val))

        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {train_loss:.6f} - "
                  f"Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {train_loss:.6f} - "
                      f"Val Loss: {val_loss:.6f}")
                print(f"Early stopping triggered at epoch {epoch} "
                      f"(patience={early_stop_patience})")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    train_time = time.perf_counter() - t0
    print(f"Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
    print(f"Training completed in {train_time:.2f} seconds")

    # ------------------------------------------------------------------
    # 8. Evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    test_metrics = evaluate_split(
        model=model,
        X_subopt_scaled=X_test,
        indices=test_idx,
        raw_data=raw_data,
        params=params,
        scalers=scalers,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
        case_data=case_data,
        split_name="Test",
        subopt_x_raw=subopt_x_raw[test_idx],
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 9. Inference latency (graph convolution and readout only)
    # ------------------------------------------------------------------
    nf1, bei1, bew1, B1 = collate_graph_batch(
        X_test[:1].to(device), ei_dev, ew_dev, n_buses, device)

    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(nf1, bei1, bew1, batch_size=B1)

        ts = []
        for _ in range(100):
            t_start = time.perf_counter()
            model(nf1, bei1, bew1, batch_size=B1)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t_start)

    latency_ms = float(np.mean(ts)) * 1000

    # ------------------------------------------------------------------
    # 10. Results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Final Results Summary")
    print(f"{'=' * 70}")
    print(f"\nCase: {case_name}")
    print(f"Input: sub-optimal state X = [vm, va, p, q]")

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
    GNN_F1 = 512  
    GNN_F2 = 256  
    GNN_K = 4
    PREDICT_VM = False  

    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)
    print(f"\n[Spectral GNN Configuration]")
    print(f"  F1: {GNN_F1}, F2: {GNN_F2}, ChebConv K: {GNN_K}")
    print(f"  Predict Vm: {PREDICT_VM}")
    print("=" * 70)

    results = spectral_gnn_acopf_experiment(
        **acopf_config.get_all_paths(),
        **acopf_config.get_all_params(),
        F1=GNN_F1,
        F2=GNN_F2,
        K=GNN_K,
        predict_vm=PREDICT_VM,
    )

    print("\nExperiment completed successfully!")