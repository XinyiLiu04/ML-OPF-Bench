# -*- coding: utf-8 -*-
"""
Spectral GNN DCOPF Main Script  (Enhanced: susceptance kernel + 6-dim features)

Changes from v1:
  1. Node features expanded from 2-dim to 6-dim:
     [pd_i, is_gen_i, pg_min_i, pg_max_i, degree_i, neighbor_load_sum_i]
  2. Default graph kernel changed from 'gaussian' to 'susceptance'
  3. Model now receives edge_index at construction time for static feature computation
  4. collate_graph_batch_dc() updated to pass static features + single edge_index
  5. Default patience increased from 20 to 50

Learning paradigm : SUPERVISED (imitation learning)
Model  : SpectralGNN_DCOPF
Input  : node features [pd_i, is_gen_i, pg_min_i, pg_max_i, degree_i, neighbor_load_sum_i]
         (N × 6 per sample)
Output : pg_non_slack  [n_g_non_slack per sample]
"""

import os
import sys
import time
import gnn_dcopf_model
print("Model file:", gnn_dcopf_model.__file__)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# ── Make sure all sibling modules are importable
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# ── Project imports ───────────────────────────────────────────────────
from dcopf_data_setup import (
    load_parameters_from_csv,
    DataSplitMode,
    split_data_by_mode,
)
from dcopf_slack_utils import (
    identify_slack_bus_and_gens,
    update_params_with_slack_info,
)
from gnn_dcopf_model import SpectralGNN_DCOPF, NODE_FEAT_DIM
from gnn_dcopf_utils import (
    load_branch_info,
    build_graph_from_branch_info,
    load_and_prepare_dc,
    evaluate_split_dc,
    collate_graph_batch_dc,
)


# =====================================================================
# Metric printer
# =====================================================================
def print_metrics(label, metrics, latency_ms=None, train_time=None):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  --- Accuracy ---")
    print(f"  MAE Pg (Non-Slack) : {metrics['mae_pg_non_slack']:.4f}%")
    print(f"  --- Violation (p.u.) ---")
    print(f"  Pg_viol (Non-Slack): {metrics['viol_pg_non_slack']:.6f} p.u.")
    print(f"  Pg_viol (Slack)    : {metrics['viol_pg_slack']:.6f} p.u.")
    print(f"  Branch_viol        : {metrics['viol_branch']:.6f} p.u."
          f"  (1.0 = 100% overload)")
    print(f"  --- Cost ---")
    print(f"  Cost Gap           : {metrics['cost_gap_percent']:.4f}%")
    print(f"  --- Performance ---")
    if latency_ms is not None:
        print(f"  Inference Time     : {latency_ms:.4f} ms/sample")
    if train_time is not None:
        print(f"  Training Time      : {train_time:.2f} s")
    print(f"  Samples evaluated  : {metrics['n_samples']}")


# =====================================================================
# External test loader  (generalization / API)
# =====================================================================
def _load_external_test(data_path, params_ref, n_test_samples, seed,
                        params_path_ext=None, case_name_ext=None, is_api=False):
    """
    Load an external test set (generalization or API topology).

    Returns:
        x_raw       : np.ndarray [M, n_buses]
        y_pg_all    : np.ndarray [M, n_g]
        params_test : params dict for the evaluation topology
        branch_info : branch_info dict (None if same topology as training)
    """
    if params_path_ext is not None and case_name_ext is not None:
        params_test = load_parameters_from_csv(case_name_ext, params_path_ext,
                                               is_api=is_api)
        slack_info  = identify_slack_bus_and_gens(params_test)
        params_test = update_params_with_slack_info(params_test, slack_info)
        branch_info = load_branch_info(params_path_ext, case_name_ext, is_api=is_api)
    else:
        params_test = params_ref
        branch_info = None          # caller reuses training graph

    x_raw_full, _, y_pg_all_full = load_and_prepare_dc(data_path, params_test)

    n_available = len(x_raw_full)
    n_actual    = min(n_test_samples, n_available)
    rng         = np.random.default_rng(seed)
    if n_actual < n_available:
        idx           = rng.choice(n_available, n_actual, replace=False)
        x_raw_full    = x_raw_full[idx]
        y_pg_all_full = y_pg_all_full[idx]

    print(f"  External test: {n_actual} / {n_available} samples used")
    return x_raw_full, y_pg_all_full, params_test, branch_info


# =====================================================================
# Main experiment
# =====================================================================
def gnn_dcopf_experiment(
        case_name,
        params_path,
        dataset_path,
        split_mode           = DataSplitMode.RANDOM_SPLIT,
        gen_test_data_path   = None,
        api_test_data_path   = None,
        api_test_params_path = None,
        api_test_case_name   = None,
        column_names         = None,
        n_train_use          = None,
        n_test_samples       = 1000,
        seed                 = 42,
        n_epochs             = 1000,
        early_stop_patience  = 50,
        early_stop_min_delta = 1e-6,
        learning_rate        = 1e-3,
        batch_size           = 256,
        device               = 'cuda',
        gnn_hidden_sizes     = None,
        K                    = 4,
        graph_kernel         = 'susceptance',
        graph_scale_k        = 1.0,
):
    if gnn_hidden_sizes is None:
        gnn_hidden_sizes = [128, 64]
    if column_names is None:
        column_names = {
            'load_prefix'       : 'pd',
            'gen_prefix'        : 'pg',
            'lambda'            : 'lambda',
            'mu_g_min_prefix'   : 'mu_g_min_',
            'mu_g_max_prefix'   : 'mu_g_max_',
            'mu_line_pos_prefix': 'mu_line_max_',
            'mu_line_neg_prefix': 'mu_line_min_',
        }

    # ── device / seed ────────────────────────────────────────────────
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA unavailable, falling back to CPU.")
        device = 'cpu'
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # =================================================================
    # 1. Parameters & slack identification
    # =================================================================
    print("\n" + "=" * 70)
    print("Loading Network Parameters")
    print("=" * 70)

    params     = load_parameters_from_csv(case_name, params_path)
    slack_info = identify_slack_bus_and_gens(params)
    params     = update_params_with_slack_info(params, slack_info)

    n_buses       = params['general']['n_buses']
    n_g           = params['general']['n_g']
    n_g_non_slack = params['general']['n_g_non_slack']

    print(f"  Case          : {case_name}")
    print(f"  Buses         : {n_buses}  |  Generators: {n_g}"
          f"  (non-slack: {n_g_non_slack})")
    print(f"  Slack bus idx : {params['general']['slack_bus_idx']}")

    # =================================================================
    # 2. Graph construction
    # =================================================================
    print("\n" + "=" * 70)
    print("Building Graph Structure")
    print("=" * 70)

    branch_info             = load_branch_info(params_path, case_name)
    edge_index, edge_weight = build_graph_from_branch_info(
        branch_info, params, kernel=graph_kernel, scale_k=graph_scale_k
    )

    # =================================================================
    # 3. Load & scale data
    # =================================================================
    print("\n" + "=" * 70)
    print("Loading and Scaling Training Data")
    print("=" * 70)

    x_raw, y_pg_ns_raw, y_pg_all_raw = load_and_prepare_dc(dataset_path, params)

    x_scaler       = MinMaxScaler().fit(x_raw)
    y_pg_ns_scaler = MinMaxScaler().fit(y_pg_ns_raw)
    scalers        = {'x': x_scaler, 'y_pg_non_slack': y_pg_ns_scaler}

    x_scaled    = x_scaler.transform(x_raw)
    y_ns_scaled = y_pg_ns_scaler.transform(y_pg_ns_raw)
    raw_data    = {'x': x_raw, 'y_pg_all': y_pg_all_raw}

    # =================================================================
    # 4. Data split
    # =================================================================
    print("\n" + "=" * 70)
    print(f"Data Split  ({split_mode.value})")
    print("=" * 70)

    train_idx, val_idx, test_idx, _, _ = split_data_by_mode(
        x_data_raw     = x_raw,
        y_pg_raw       = y_pg_all_raw,
        mode           = split_mode,
        n_train_use    = n_train_use,
        seed           = seed,
        test_data_path = None,
        params         = params,
        column_names   = column_names,
        n_test_samples = n_test_samples,
    )

    X_train = torch.tensor(x_scaled[train_idx],   dtype=torch.float32, device=device)
    Y_train = torch.tensor(y_ns_scaled[train_idx], dtype=torch.float32, device=device)
    X_val   = torch.tensor(x_scaled[val_idx],     dtype=torch.float32, device=device)
    Y_val   = torch.tensor(y_ns_scaled[val_idx],  dtype=torch.float32, device=device)

    mode_label = split_mode.value
    print(f"\n[Dataset sizes]")
    print(f"  Train  : {len(train_idx)}")
    print(f"  Val    : {len(val_idx)}")
    print(f"  Test-1 : {len(test_idx)}  ({mode_label})")
    if split_mode == DataSplitMode.RANDOM_SPLIT:
        print(f"  Test-2 : {'up to ' + str(n_test_samples) + ' (generalization)' if gen_test_data_path else 'SKIPPED'}")
        print(f"  Test-3 : {'up to ' + str(n_test_samples) + ' (API test)' if api_test_data_path else 'SKIPPED'}")

    # =================================================================
    # 5. Model  (NOW receives edge_index for static feature computation)
    # =================================================================
    print("\n" + "=" * 70)
    print("Model Configuration")
    print("=" * 70)

    model    = SpectralGNN_DCOPF(
        F1=gnn_hidden_sizes[0], F2=gnn_hidden_sizes[1],
        K=K, params=params, edge_index=edge_index    # ← NEW: pass edge_index
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Architecture     : Spectral GNN (Owerko et al., ICASSP 2020) – DCOPF Enhanced")
    print(f"  Node features    : [pd, is_gen, pg_min, pg_max, degree, neighbor_load]"
          f"  (dim={NODE_FEAT_DIM} per bus)")
    print(f"  Graph kernel     : {graph_kernel}  (k={graph_scale_k})")
    print(f"  ChebConv K       : {K}")
    print(f"  Hidden dims      : {gnn_hidden_sizes[0]} → {gnn_hidden_sizes[1]}  (fixed 2-layer)")
    print(f"  Readout          : local MLP (per non-slack generator node)")
    print(f"  Output dim       : {n_g_non_slack}  (pg_non_slack)")
    print(f"  Trainable params : {n_params:,}")
    print(f"  Training params  : max_epochs={n_epochs}, patience={early_stop_patience},"
          f" lr={learning_rate}, batch={batch_size}")
    print("=" * 70)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ei_dev    = edge_index.to(device)
    ew_dev    = edge_weight.to(device)

    # =================================================================
    # 6. Training loop  (updated collate call signature)
    # =================================================================
    print("\n" + "=" * 70)
    print("Training Progress")
    print("=" * 70)

    n_train   = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size

    best_val_loss    = float('inf')
    best_epoch       = 0
    best_state_dict  = None
    patience_counter = 0
    t0               = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        # ── train ──
        model.train()
        epoch_loss = 0.0
        perm       = torch.randperm(n_train)
        for i in range(n_batches):
            idx_b = perm[i * batch_size: min((i + 1) * batch_size, n_train)]
            optimizer.zero_grad()
            nf, bei, bew, B = collate_graph_batch_dc(
                X_train[idx_b], ei_dev, ew_dev,
                model.static_node_feats, model.single_edge_index,  # ← NEW
                n_buses, device)
            loss  = criterion(
                model(nf, bei, bew, batch_size=B, params=params),
                Y_train[idx_b])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx_b)
        train_loss = epoch_loss / n_train

        # ── validate ──
        model.eval()
        with torch.no_grad():
            nf_v, bei_v, bew_v, B_v = collate_graph_batch_dc(
                X_val, ei_dev, ew_dev,
                model.static_node_feats, model.single_edge_index,  # ← NEW
                n_buses, device)
            val_loss = float(criterion(
                model(nf_v, bei_v, bew_v, batch_size=B_v, params=params),
                Y_val).item())

        # ── log every 10 epochs ──
        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Epoch {epoch:4d}/{n_epochs}"
                  f" - Train Loss: {train_loss:.6f}"
                  f" - Val Loss: {val_loss:.6f}")

        # ── early stopping ──
        if val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss    = val_loss
            best_epoch       = epoch
            best_state_dict  = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Epoch {epoch:4d}/{n_epochs}"
                      f" - Train Loss: {train_loss:.6f}"
                      f" - Val Loss: {val_loss:.6f}")
                print(f"\n⚡ Early stopping triggered at epoch {epoch}"
                      f"  (patience={early_stop_patience})")
                break

    model.load_state_dict(
        {k: v.to(device) for k, v in best_state_dict.items()})
    train_time = time.perf_counter() - t0
    print(f"✓ Restored best model from epoch {best_epoch}"
          f"  (val_loss={best_val_loss:.6f})")
    print(f"✓ Training completed in {train_time:.2f} s")

    # =================================================================
    # 7. Inference latency  (updated collate call)
    # =================================================================
    model.eval()
    x_single = torch.tensor(x_scaled[test_idx[:1]], dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(10):
            nf_s, bei_s, bew_s, B_s = collate_graph_batch_dc(
                x_single, ei_dev, ew_dev,
                model.static_node_feats, model.single_edge_index,
                n_buses, device)
            model(nf_s, bei_s, bew_s, batch_size=B_s, params=params)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(100):
            t_s = time.perf_counter()
            nf_s, bei_s, bew_s, B_s = collate_graph_batch_dc(
                x_single, ei_dev, ew_dev,
                model.static_node_feats, model.single_edge_index,
                n_buses, device)
            model(nf_s, bei_s, bew_s, batch_size=B_s, params=params)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t_s)
    latency_ms = float(np.mean(times)) * 1000

    # =================================================================
    # 8. Test-1  (in-distribution)
    # =================================================================
    print("\n" + "=" * 70)
    print(f"Test-1  :  {mode_label}  (in-distribution)")
    print("=" * 70)

    metrics_t1 = evaluate_split_dc(
        model         = model,
        edge_index    = edge_index,
        edge_weight   = edge_weight,
        scalers       = scalers,
        params        = params,
        device        = device,
        x_raw_eval    = raw_data['x'][test_idx],
        y_true_pg_all = raw_data['y_pg_all'][test_idx],
    )

    # =================================================================
    # 9. Test-2  (generalization, random_split only)
    # =================================================================
    metrics_t2 = None
    if split_mode == DataSplitMode.RANDOM_SPLIT:
        if gen_test_data_path is not None:
            print("\n" + "=" * 70)
            print("Test-2  :  Generalization  (same topology, different variance)")
            print("=" * 70)
            x_t2, y_t2, params_t2, _ = _load_external_test(
                data_path      = gen_test_data_path,
                params_ref     = params,
                n_test_samples = n_test_samples,
                seed           = seed,
            )
            metrics_t2 = evaluate_split_dc(
                model         = model,
                edge_index    = edge_index,
                edge_weight   = edge_weight,
                scalers       = scalers,
                params        = params,
                device        = device,
                x_raw_eval    = x_t2,
                y_true_pg_all = y_t2,
            )
        else:
            print("\n[Test-2 Generalization]  SKIPPED  (gen_test_data_path = None)")

    # =================================================================
    # 10. Test-3  (API test, random_split only)
    # =================================================================
    metrics_t3 = None
    if split_mode == DataSplitMode.RANDOM_SPLIT:
        if (api_test_data_path is not None
                and api_test_params_path is not None
                and api_test_case_name is not None):
            print("\n" + "=" * 70)
            print("Test-3  :  API Test  (different topology)")
            print("=" * 70)
            x_t3, y_t3, params_t3, branch_info_t3 = _load_external_test(
                data_path       = api_test_data_path,
                params_ref      = params,
                n_test_samples  = n_test_samples,
                seed            = seed,
                params_path_ext = api_test_params_path,
                case_name_ext   = api_test_case_name,
                is_api          = True,
            )
            ei_t3, ew_t3 = build_graph_from_branch_info(
                branch_info_t3, params_t3,
                kernel=graph_kernel, scale_k=graph_scale_k
            )
            print(f"  [API Topology]  buses={params_t3['general']['n_buses']}"
                  f"  gen={params_t3['general']['n_g']}"
                  f"  (non-slack={params_t3['general']['n_g_non_slack']})"
                  f"  baseMVA={params_t3['general']['BASE_MVA']}")
            metrics_t3 = evaluate_split_dc(
                model         = model,
                edge_index    = ei_t3,
                edge_weight   = ew_t3,
                scalers       = scalers,
                params        = params_t3,
                device        = device,
                x_raw_eval    = x_t3,
                y_true_pg_all = y_t3,
            )
        else:
            print("\n[Test-3 API Test]  SKIPPED  (paths not fully configured)")

    # =================================================================
    # 11. Final summary
    # =================================================================
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY  (GNN-DCOPF, supervised, enhanced)")
    print("=" * 70)
    print(f"  Case          : {case_name}")
    print(f"  Split mode    : {mode_label}")
    print(f"  GNN           : hidden={gnn_hidden_sizes}  K={K}  kernel={graph_kernel}")
    print(f"  Node features : dim={NODE_FEAT_DIM}"
          f"  [pd, is_gen, pg_min, pg_max, degree, neighbor_load]")
    print(f"  Training time : {train_time:.2f} s")
    print(f"  Inference time: {latency_ms:.4f} ms/sample")

    print_metrics(f"Test-1 : {mode_label} (in-distribution)",
                  metrics_t1, latency_ms=latency_ms, train_time=train_time)

    if metrics_t2 is not None:
        print_metrics("Test-2 : Generalization (different variance)", metrics_t2)
    elif split_mode == DataSplitMode.RANDOM_SPLIT:
        print(f"\n{'=' * 70}")
        print(f"  Test-2 : Generalization  –  SKIPPED")

    if metrics_t3 is not None:
        print_metrics("Test-3 : API Test (different topology)", metrics_t3)
    elif split_mode == DataSplitMode.RANDOM_SPLIT:
        print(f"\n{'=' * 70}")
        print(f"  Test-3 : API Test  –  SKIPPED")

    print("=" * 70)

    return {
        'test1'       : metrics_t1,
        'test2_gen'   : metrics_t2,
        'test3_api'   : metrics_t3,
        'train_time_s': train_time,
        'latency_ms'  : latency_ms,
    }


# =====================================================================
# ╔══════════════════════════════════════════════════════════════════╗
# ║                    USER CONFIGURATION                           ║
# ╚══════════════════════════════════════════════════════════════════╝
# =====================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 0. Root directory
    # ------------------------------------------------------------------
    ROOT_DIR = "/lambda/nfs/lxy/dcopf_project/data"   # ← modify

    # ------------------------------------------------------------------
    # 1. Split mode
    # ------------------------------------------------------------------
    SPLIT_MODE = DataSplitMode.RANDOM_SPLIT

    # ------------------------------------------------------------------
    # 2. Training case
    # ------------------------------------------------------------------
    CASE_NAME       = 'pglib_opf_case30_ieee'
    CASE_SHORT_NAME = 'case30'
    TRAIN_VARIANCE  = 'v=0.12'

    # ------------------------------------------------------------------
    # 3. Test-2 : Generalization  (None = skip)
    # ------------------------------------------------------------------
    GEN_TEST_VARIANCE = None     # e.g. 'v=0.25'

    # ------------------------------------------------------------------
    # 4. Test-3 : API test  (None = skip)
    # ------------------------------------------------------------------
    API_TEST_CASE_SHORT = None
    API_TEST_CASE_NAME  = None

    # ------------------------------------------------------------------
    # 5. Hyper-parameters
    # ------------------------------------------------------------------
    N_TRAIN_USE          = 12000
    N_TEST_SAMPLES       = 1000
    N_EPOCHS             = 1000
    EARLY_STOP_PATIENCE  = 20         # ← increased from 20
    EARLY_STOP_MIN_DELTA = 1e-6
    LEARNING_RATE        = 1e-3
    BATCH_SIZE           = 32
    SEED                 = 42
    DEVICE               = 'cuda'     # 'cuda' or 'cpu'

    # ------------------------------------------------------------------
    # 6. GNN architecture
    # ------------------------------------------------------------------
    GNN_HIDDEN_SIZES = [32, 16]
    GNN_K            = 4

    # ------------------------------------------------------------------
    # 7. Graph kernel  (CHANGED: susceptance is now default for DCOPF)
    # ------------------------------------------------------------------
    GRAPH_KERNEL  = 'susceptance'     # ← changed from 'gaussian'
    GRAPH_SCALE_K = 0.01                 # only used for gaussian kernel

    # ------------------------------------------------------------------
    # 8. Column names
    # ------------------------------------------------------------------
    COLUMN_NAMES = {
        'load_prefix'       : 'pd',
        'gen_prefix'        : 'pg',
        'lambda'            : 'lambda',
        'mu_g_min_prefix'   : 'mu_g_min_',
        'mu_g_max_prefix'   : 'mu_g_max_',
        'mu_line_pos_prefix': 'mu_line_max_',
        'mu_line_neg_prefix': 'mu_line_min_',
    }

    # ==================================================================
    # Path assembly
    # ==================================================================
    params_path     = os.path.join(ROOT_DIR, "DCOPF Constraints", CASE_SHORT_NAME)
    train_data_path = os.path.join(
        ROOT_DIR, "DCOPF dataset",
        f"{CASE_SHORT_NAME}({TRAIN_VARIANCE})",
        f"{CASE_NAME}_dataset_with_duals.csv",
    )

    gen_test_data_path = (
        os.path.join(ROOT_DIR, "DCOPF dataset",
                     f"{CASE_SHORT_NAME}({GEN_TEST_VARIANCE})",
                     f"{CASE_NAME}_dataset_with_duals.csv")
        if GEN_TEST_VARIANCE is not None else None
    )

    if API_TEST_CASE_SHORT is not None and API_TEST_CASE_NAME is not None:
        api_test_params_path = os.path.join(
            ROOT_DIR, "DCOPF Constraints", f"{API_TEST_CASE_SHORT}(api)")
        api_test_data_path   = os.path.join(
            ROOT_DIR, "DCOPF dataset",
            f"{API_TEST_CASE_SHORT}(v=api)",
            f"{API_TEST_CASE_NAME}__api_dataset_with_duals.csv")
    else:
        api_test_params_path = None
        api_test_data_path   = None

    # ==================================================================
    # Config summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("GNN-DCOPF Experiment Configuration  (Enhanced)")
    print("=" * 70)
    print(f"  Split mode  : {SPLIT_MODE.value}")
    print(f"  Case        : {CASE_NAME}  ({TRAIN_VARIANCE})")
    print(f"  Dataset     : {train_data_path}")
    print(f"  Constraints : {params_path}")
    if SPLIT_MODE == DataSplitMode.RANDOM_SPLIT:
        print(f"  Test-2      : {gen_test_data_path or 'SKIPPED'}")
        print(f"  Test-3      : {api_test_data_path or 'SKIPPED'}")
    else:
        print(f"  Test-2/3    : N/A  (valid_fixed mode)")
    print(f"  N_train     : {N_TRAIN_USE}  |  N_test(2&3): {N_TEST_SAMPLES}")
    print(f"  Epochs      : {N_EPOCHS}  patience={EARLY_STOP_PATIENCE}"
          f"  lr={LEARNING_RATE}  batch={BATCH_SIZE}  seed={SEED}")
    print(f"  Device      : {DEVICE}")
    print(f"  GNN         : hidden={GNN_HIDDEN_SIZES}  K={GNN_K}"
          f"  kernel={GRAPH_KERNEL}(k={GRAPH_SCALE_K})")
    print(f"  Node feats  : dim={NODE_FEAT_DIM}"
          f"  [pd, is_gen, pg_min, pg_max, degree, neighbor_load]")
    print("=" * 70)

    # ==================================================================
    # Run
    # ==================================================================
    results = gnn_dcopf_experiment(
        case_name            = CASE_NAME,
        params_path          = params_path,
        dataset_path         = train_data_path,
        split_mode           = SPLIT_MODE,
        gen_test_data_path   = gen_test_data_path,
        api_test_data_path   = api_test_data_path,
        api_test_params_path = api_test_params_path,
        api_test_case_name   = API_TEST_CASE_NAME,
        column_names         = COLUMN_NAMES,
        n_train_use          = N_TRAIN_USE,
        n_test_samples       = N_TEST_SAMPLES,
        seed                 = SEED,
        n_epochs             = N_EPOCHS,
        early_stop_patience  = EARLY_STOP_PATIENCE,
        early_stop_min_delta = EARLY_STOP_MIN_DELTA,
        learning_rate        = LEARNING_RATE,
        batch_size           = BATCH_SIZE,
        device               = DEVICE,
        gnn_hidden_sizes     = GNN_HIDDEN_SIZES,
        K                    = GNN_K,
        graph_kernel         = GRAPH_KERNEL,
        graph_scale_k        = GRAPH_SCALE_K,
    )

    print("\n✓ Experiment completed successfully!")