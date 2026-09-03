# -*- coding: utf-8 -*-
"""
Spectral GNN ACOPF Main Script — Paper-faithful edition
(Owerko et al., ICASSP 2020 — supervised, sub-optimal state input)

Key change:
    Input X is now the sub-optimal state [vm, va, p_inj, q_inj] per bus,
    pre-computed by generate_subopt_state.py (DCOPF + AC Power Flow).
    This matches the paper's formulation exactly.

Model  : SpectralGNN_ACOPF  —  2 x ChebConv (K=4) + local readout
Output : [pg_non_slack | vm_gen] (or pg_non_slack only)
Loss   : MSE (imitation learning)

Supported data modes:
    - random_split   : train/val/test from same dataset
    - fixed_valtest  : fixed val/test, variable training size
    - generalization : train on v=X, test on v=Y (same case)
    - api_test       : train on case, test on case__api (same topology)

Usage:
    1. Run generate_subopt_state.py first to create _subopt_*.csv files
       (for BOTH training and test datasets if using generalization/api_test)
    2. Set configuration in acopf_config.py
    3. python gnn_spectral_main_v2.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os

from sklearn.preprocessing import MinMaxScaler

try:
    import acopf_config
except ImportError:
    print("Error: Unable to import acopf_config.py"); sys.exit(1)

try:
    from acopf_data_setup import (
        load_parameters_from_csv, load_and_scale_acopf_data,
        DataMode, prepare_data_splits,
        load_generalization_test_data, load_api_test_data,
        reconstruct_full_pg,
    )
except ImportError:
    print("Error: Unable to import acopf_data_setup."); sys.exit(1)

try:
    import gnn_spectral_utils_v2 as spectral_utils
    from gnn_spectral_model_v2 import SpectralGNN_ACOPF
    from gnn_spectral_utils_v2 import (
        build_adjacency_edge_weight, init_pypower_options,
        load_case_from_csv, evaluate_split,
        load_subopt_state,
    )
except ImportError as e:
    print(f"Error: Unable to import v2 modules: {e}"); sys.exit(1)


# =====================================================================
# Main experiment
# =====================================================================
def spectral_gnn_acopf_experiment(
        # ── paths ─────────────────────────────────────────────────────
        case_name, params_path, data_path,
        test_data_path=None, test_params_path=None,
        log_path=None, results_path=None,
        # ── training / data params ────────────────────────────────────
        data_mode='random_split',
        n_train_use=None, n_test_samples=None, seed=42,
        n_epochs=1000, early_stop_patience=50,
        early_stop_min_delta=1e-6, learning_rate=1e-3,
        batch_size=128, device='cuda',
        hidden_sizes=None,
        # ── GNN hyper-params ─────────────────────────────────────────
        F1=128, F2=64, K=4,
        graph_kernel='gaussian', graph_scale_k=1.0,
        predict_vm=True,
        **kwargs,
):
    # ── device / seed ─────────────────────────────────────────────────
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA unavailable, falling back to CPU.")
        device = 'cpu'
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    init_pypower_options()

    print(f"\n{'=' * 70}")
    print(f"Spectral GNN ACOPF Experiment (Paper-faithful: sub-optimal state X)")
    print(f"{'=' * 70}")
    print(f"Device    : {device}")
    print(f"Case      : {case_name}")
    print(f"Data Mode : {data_mode}")
    print(f"Predict   : pg_non_slack" + (" + vm_gen" if predict_vm else " only"))
    print(f"{'=' * 70}")

    # ==================================================================
    # 1. Load training params + PyPower case
    # ==================================================================
    print("\n" + "=" * 70)
    print("Loading Network Parameters")
    print("=" * 70)
    params = load_parameters_from_csv(case_name, params_path)
    n_gen           = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses         = params['general']['n_buses']
    n_loads         = params['general']['n_loads']
    baseMVA         = params['general']['BASE_MVA']

    spectral_utils.GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)
    print(f"✓ PyPower case data loaded")
    print(f"\n[Training Data Info]")
    print(f"  Buses: {n_buses}, Generators: {n_gen}"
          f" (Non-Slack: {n_gen_non_slack}), Loads: {n_loads},"
          f" Base MVA: {baseMVA}")

    # ==================================================================
    # 2. Build graph structure
    # ==================================================================
    print("\n" + "=" * 70)
    print("Building Graph Structure")
    print("=" * 70)
    edge_index, edge_weight = build_adjacency_edge_weight(
        params, kernel=graph_kernel, scale_k=graph_scale_k)

    # ==================================================================
    # 3. Load ACOPF data (for labels Y and raw_data for evaluation)
    # ==================================================================
    print("\n" + "=" * 70)
    print("Loading ACOPF Labels (Y) and Sub-optimal State (X)")
    print("=" * 70)

    # Load standard ACOPF data (provides Y labels, scalers, raw_data)
    x_data_scaled_orig, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params)
    print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # ==================================================================
    # 3b. Load sub-optimal state X
    # ==================================================================
    data_dir = os.path.dirname(data_path)
    print(f"\n  Loading sub-optimal state from: {data_dir}")
    subopt_x_raw, subopt_converged = load_subopt_state(
        data_dir, case_name, params)

    n_total = len(subopt_x_raw)
    n_converged = subopt_converged.sum()
    print(f"  Sub-optimal samples: {n_total}, converged: {n_converged} "
          f"({n_converged / n_total * 100:.1f}%)")

    # Filter to converged samples only
    if n_converged < n_total:
        print(f"  ⚠️ Filtering to {n_converged} converged samples")
        conv_mask = subopt_converged
        subopt_x_raw = subopt_x_raw[conv_mask]
        y_data_scaled = y_data_scaled[conv_mask]
        x_data_scaled_orig = x_data_scaled_orig[conv_mask]
        for key in raw_data:
            raw_data[key] = raw_data[key][conv_mask]

    # Scale sub-optimal state X
    subopt_scaler = MinMaxScaler()
    subopt_x_scaled = subopt_scaler.fit_transform(subopt_x_raw).astype('float32')
    scalers['subopt_x'] = subopt_scaler
    print(f"  ✓ Sub-optimal state scaled (MinMax)")
    print(f"    X_subopt shape: {subopt_x_scaled.shape} "
          f"(= 4 × {n_buses} buses)")

    # Adjust Y if not predicting vm
    if not predict_vm:
        y_data_scaled = y_data_scaled[:, :n_gen_non_slack]
        print(f"  Output: pg_non_slack only, Y shape: {y_data_scaled.shape}")

    # ==================================================================
    # 4. Data splits
    # ==================================================================
    data_mode_lower = data_mode.lower().strip()

    if data_mode_lower == 'api_test':
        # ── API_TEST: same topology, different constraints / loads ─────
        print(f"\n{'=' * 70}")
        print(f"Data Mode: API_TEST")
        print(f"{'=' * 70}")

        if test_data_path is None or test_params_path is None:
            raise ValueError("API_TEST mode requires test_data_path and test_params_path")

        # Split training data (test split discarded, replaced by API data)
        train_idx, val_idx, _ = prepare_data_splits(
            subopt_x_scaled, y_data_scaled,
            mode=DataMode.API_TEST,
            n_train_use=n_train_use, seed=seed)

        # Load API test ACOPF labels + raw_data (Y side)
        test_params, test_x_scaled_orig, test_y_scaled, test_raw_data, _ = \
            load_api_test_data(
                test_data_path, test_params_path, scalers,
                n_test_samples=n_test_samples or 1000, seed=seed)

        # Derive test case name
        test_base = os.path.basename(test_data_path)
        test_case_name = test_base[:-7] if test_base.endswith('_pd.csv') else test_base.rsplit('_', 1)[0]

        # Load API test sub-optimal state X
        test_data_dir = os.path.dirname(test_data_path)
        print(f"\n  Loading API test sub-optimal state from: {test_data_dir}")
        test_subopt_raw, test_subopt_conv = load_subopt_state(
            test_data_dir, test_case_name, test_params)

        # Subsample to match test_raw_data (already subsampled by load_api_test_data)
        n_test_total = len(test_subopt_raw)
        n_test_used = len(test_raw_data['pg'])
        if n_test_used < n_test_total:
            # Reproduce the same random selection used by load_api_test_data
            rng_test = np.random.default_rng(seed)
            test_sel_idx = rng_test.choice(n_test_total, size=n_test_used, replace=False)
            test_subopt_raw = test_subopt_raw[test_sel_idx]
            test_subopt_conv = test_subopt_conv[test_sel_idx]

        # Filter to converged sub-optimal samples
        n_test_conv = test_subopt_conv.sum()
        print(f"  API test sub-optimal: {len(test_subopt_raw)} samples, "
              f"converged: {n_test_conv} ({n_test_conv / len(test_subopt_raw) * 100:.1f}%)")
        if n_test_conv < len(test_subopt_raw):
            print(f"  ⚠️ Filtering to {n_test_conv} converged samples")
            conv_mask_test = test_subopt_conv
            test_subopt_raw = test_subopt_raw[conv_mask_test]
            for key in test_raw_data:
                test_raw_data[key] = test_raw_data[key][conv_mask_test]

        # Scale test sub-optimal state using TRAINING scaler
        test_subopt_scaled = scalers['subopt_x'].transform(test_subopt_raw).astype('float32')
        print(f"  ✓ API test sub-optimal state scaled (using training scaler)")

        test_idx = np.arange(len(test_subopt_scaled))
        X_test = torch.tensor(test_subopt_scaled, dtype=torch.float32)
        test_subopt_x_raw = test_subopt_raw  # raw (unscaled) for Vm extraction

        # Same topology → reuse graph structure; use API params + case data for PF
        test_edge_index       = edge_index
        test_edge_weight      = edge_weight
        global_case_data_test = load_case_from_csv(test_case_name, test_params_path)
        test_split_name       = "API Test"

        print(f"\n[API Test Data Info]")
        print(f"  Buses: {test_params['general']['n_buses']}")
        print(f"  Generators: {test_params['general']['n_gen']}"
              f" (Non-Slack: {test_params['general']['n_gen_non_slack']})")
        print(f"  Loads: {test_params['general']['n_loads']}")
        print(f"  Base MVA: {test_params['general']['BASE_MVA']}")

    elif data_mode_lower == 'generalization':
        # ── GENERALIZATION: same case, different variance ─────────────
        print(f"\n{'=' * 70}")
        print(f"Data Mode: GENERALIZATION")
        print(f"{'=' * 70}")

        if test_data_path is None:
            raise ValueError("GENERALIZATION mode requires test_data_path")

        # Split training data (test split discarded, replaced by gen data)
        train_idx, val_idx, _ = prepare_data_splits(
            subopt_x_scaled, y_data_scaled,
            mode=DataMode.GENERALIZATION,
            n_train_use=n_train_use, seed=seed)

        # Load generalization test ACOPF labels + raw_data (Y side)
        test_x_scaled_orig, test_y_scaled, test_raw_data, _ = \
            load_generalization_test_data(
                test_data_path, params, scalers,
                n_test_samples=n_test_samples or 1000, seed=seed)

        # Derive test case name (same case, different variance folder)
        test_base = os.path.basename(test_data_path)
        test_case_name = test_base[:-7] if test_base.endswith('_pd.csv') else test_base.rsplit('_', 1)[0]

        # Load generalization test sub-optimal state X
        test_data_dir = os.path.dirname(test_data_path)
        print(f"\n  Loading generalization test sub-optimal state from: {test_data_dir}")
        test_subopt_raw, test_subopt_conv = load_subopt_state(
            test_data_dir, test_case_name, params)

        # Subsample to match test_raw_data (already subsampled by load_generalization_test_data)
        n_test_total = len(test_subopt_raw)
        n_test_used = len(test_raw_data['pg'])
        if n_test_used < n_test_total:
            rng_test = np.random.default_rng(seed)
            test_sel_idx = rng_test.choice(n_test_total, size=n_test_used, replace=False)
            test_subopt_raw = test_subopt_raw[test_sel_idx]
            test_subopt_conv = test_subopt_conv[test_sel_idx]

        # Filter to converged sub-optimal samples
        n_test_conv = test_subopt_conv.sum()
        print(f"  Generalization test sub-optimal: {len(test_subopt_raw)} samples, "
              f"converged: {n_test_conv} ({n_test_conv / len(test_subopt_raw) * 100:.1f}%)")
        if n_test_conv < len(test_subopt_raw):
            print(f"  ⚠️ Filtering to {n_test_conv} converged samples")
            conv_mask_test = test_subopt_conv
            test_subopt_raw = test_subopt_raw[conv_mask_test]
            for key in test_raw_data:
                test_raw_data[key] = test_raw_data[key][conv_mask_test]

        # Scale test sub-optimal state using TRAINING scaler
        test_subopt_scaled = scalers['subopt_x'].transform(test_subopt_raw).astype('float32')
        print(f"  ✓ Generalization test sub-optimal state scaled (using training scaler)")

        test_idx = np.arange(len(test_subopt_scaled))
        X_test = torch.tensor(test_subopt_scaled, dtype=torch.float32)
        test_subopt_x_raw = test_subopt_raw  # raw (unscaled) for Vm extraction

        # Same case → reuse everything except test data
        test_params           = params
        test_edge_index       = edge_index
        test_edge_weight      = edge_weight
        global_case_data_test = spectral_utils.GLOBAL_CASE_DATA
        test_split_name       = "Generalization Test"

    else:
        # ── RANDOM_SPLIT / FIXED_VALTEST ──────────────────────────────
        print(f"\n{'=' * 70}")
        print(f"Data Mode: {data_mode}")
        print(f"{'=' * 70}")

        _split_mode = {
            'random_split':  DataMode.RANDOM_SPLIT,
            'fixed_valtest': DataMode.FIXED_VALTEST,
        }.get(data_mode_lower)

        if _split_mode is None:
            raise ValueError(f"Unknown data mode: {data_mode}")

        train_idx, val_idx, test_idx = prepare_data_splits(
            subopt_x_scaled, y_data_scaled,
            mode=_split_mode,
            n_train_use=n_train_use, seed=seed)

        test_raw_data         = raw_data
        test_params           = params
        test_edge_index       = edge_index
        test_edge_weight      = edge_weight
        global_case_data_test = spectral_utils.GLOBAL_CASE_DATA
        test_split_name       = "Test"

        X_test = torch.tensor(subopt_x_scaled[test_idx], dtype=torch.float32)
        test_subopt_x_raw = subopt_x_raw[test_idx]  # raw (unscaled) for Vm extraction

    # ==================================================================
    # 5. Build training tensors
    # ==================================================================
    X_train = torch.tensor(subopt_x_scaled[train_idx],
                           dtype=torch.float32, device=device)
    Y_train = torch.tensor(y_data_scaled[train_idx],
                           dtype=torch.float32, device=device)
    X_val   = torch.tensor(subopt_x_scaled[val_idx],
                           dtype=torch.float32, device=device)
    Y_val   = torch.tensor(y_data_scaled[val_idx],
                           dtype=torch.float32, device=device)

    print(f"\n[Dataset Sizes]")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(test_idx)} samples")

    # ==================================================================
    # 6. Model
    # ==================================================================
    print("\n" + "=" * 70)
    print("Model Configuration")
    print("=" * 70)
    model = SpectralGNN_ACOPF(
        F1=F1, F2=F2, K=K, params=params, predict_vm=predict_vm
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out_dim = n_gen_non_slack + n_gen if predict_vm else n_gen_non_slack

    print(f"Architecture    : Spectral GNN (Owerko et al., ICASSP 2020)")
    print(f"Node features   : [vm, va, p_inj, q_inj]  (dim=4 per bus, sub-optimal state)")
    print(f"Graph kernel    : {graph_kernel}  (k={graph_scale_k})")
    print(f"ChebConv K      : {K}")
    print(f"Hidden dims     : {F1} → {F2}")
    print(f"Readout         : local (per generator node)")
    print(f"Output dim      : {out_dim}"
          + (f"  (pg_non_slack: {n_gen_non_slack} + vm_gen: {n_gen})"
             if predict_vm else f"  (pg_non_slack: {n_gen_non_slack})"))
    print(f"Trainable params: {n_params:,}")
    print(f"Training params : max_epochs={n_epochs},"
          f" patience={early_stop_patience},"
          f" lr={learning_rate}, batch_size={batch_size}")
    print("=" * 70)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ei_dev    = edge_index.to(device)
    ew_dev    = edge_weight.to(device)

    # ==================================================================
    # 7. Training loop
    # ==================================================================
    print("\n" + "=" * 70)
    print("Training Progress")
    print("=" * 70)
    n_train_samples = len(X_train)
    n_batches = (n_train_samples + batch_size - 1) // batch_size
    best_val_loss, best_epoch, best_state_dict = float('inf'), 0, None
    patience_counter = 0
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(n_train_samples)

        for i in range(n_batches):
            b = perm[i * batch_size: min((i + 1) * batch_size, n_train_samples)]
            optimizer.zero_grad()
            nf, bei, bew, B = spectral_utils.collate_graph_batch(
                X_train[b], ei_dev, ew_dev, n_buses, device)
            loss = criterion(
                model(nf, bei, bew, batch_size=B, params=params),
                Y_train[b])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b)

        train_loss = epoch_loss / n_train_samples

        model.eval()
        with torch.no_grad():
            nf_v, bei_v, bew_v, B_v = spectral_utils.collate_graph_batch(
                X_val, ei_dev, ew_dev, n_buses, device)
            val_loss = float(criterion(
                model(nf_v, bei_v, bew_v, batch_size=B_v, params=params),
                Y_val))

        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print(f"Epoch {epoch:5d}/{n_epochs} - "
                  f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss  = val_loss
            best_epoch     = epoch
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Epoch {epoch:5d}/{n_epochs} - "
                  f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            print(f"\n⚡ Early stopping triggered at epoch {epoch} "
                  f"(patience={early_stop_patience})")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"✓ Restored best model from epoch {best_epoch} "
              f"(val_loss={best_val_loss:.6f})")

    train_time = time.perf_counter() - t0
    print(f"✓ Training completed in {train_time:.2f} seconds")

    # ==================================================================
    # 8. Evaluation
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    global_case_data_backup         = spectral_utils.GLOBAL_CASE_DATA
    spectral_utils.GLOBAL_CASE_DATA = global_case_data_test

    test_metrics = evaluate_split(
        model        = model,
        X_subopt_scaled = X_test,
        indices      = test_idx,
        raw_data     = test_raw_data,
        params       = test_params,
        scalers      = scalers,
        edge_index   = test_edge_index,
        edge_weight  = test_edge_weight,
        device       = device,
        split_name   = test_split_name,
        subopt_x_raw = test_subopt_x_raw,
        verbose      = True,
    )

    spectral_utils.GLOBAL_CASE_DATA = global_case_data_backup

    # ==================================================================
    # 9. Inference speed
    # ==================================================================
    model.eval()
    ei_t = test_edge_index.to(device)
    ew_t = test_edge_weight.to(device)
    X1   = X_test[:1].to(device)

    nf1, bei1, bew1, B1 = spectral_utils.collate_graph_batch(
        X1, ei_t, ew_t, n_buses, device)

    with torch.no_grad():
        for _ in range(10):
            model(nf1, bei1, bew1, batch_size=B1, params=test_params)

    ts = []
    with torch.no_grad():
        for _ in range(100):
            t0_inf = time.perf_counter()
            model(nf1, bei1, bew1, batch_size=B1, params=test_params)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0_inf)
    latency_ms = float(np.mean(ts)) * 1000

    # ==================================================================
    # 10. Final summary
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"Final Results Summary")
    print(f"{'=' * 70}")

    print(f"\nData Mode : {data_mode}")
    print(f"Train Case: {case_name}")
    print(f"Input     : Sub-optimal state X = [vm, va, p, q] (paper-faithful)")

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
    print(f"Branch_viol:         {test_metrics['mean_max_branch_viol_pu']:.6f} p.u."
          f" (1.0 = 100% overload)")

    print(f"\n--- Cost Metrics ---")
    print(f"Cost Gap: {test_metrics['cost_optimality_gap_percent']:.4f}%")

    print(f"\n--- Performance ---")
    print(f"Inference Time  : {latency_ms:.4f} ms/sample")
    print(f"Training Time   : {train_time:.2f} s")
    print(f"Convergence Rate: {test_metrics['convergence_rate_percent']:.2f}%")

    print(f"{'=' * 70}")

    return test_metrics


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    paths  = acopf_config.get_all_paths()
    params = acopf_config.get_all_params()

    # ── GNN hyper-parameters (paper defaults) ─────────────────────────
    GNN_F1 = 512       # Paper: F1=128
    GNN_F2 = 256       # Paper: F2=64
    GNN_K  = 4         # Paper: K=4
    PREDICT_VM = False   # True: predict pg+vm; False: pg only (paper)

    print(f"\n[Spectral GNN Configuration]")
    print(f"  F1 (conv1 output) : {GNN_F1}")
    print(f"  F2 (conv2 output) : {GNN_F2}")
    print(f"  K  (ChebConv)     : {GNN_K}")
    print(f"  Predict Vm        : {PREDICT_VM}")
    print("=" * 70)

    results = spectral_gnn_acopf_experiment(
        **paths,
        **params,
        F1=GNN_F1,
        F2=GNN_F2,
        K=GNN_K,
        predict_vm=PREDICT_VM,
    )

    print("\n✓ Experiment completed successfully!")