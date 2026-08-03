# -*- coding: utf-8 -*-
"""
PINN-KKT for DCOPF - Main Experiment Script
Physics-Informed Neural Network with Explicit KKT Conditions

Version: v6.1 - Fixed virtual bus handling for case300

Fixes in v6.1:
- BUG FIX 1: load_and_prepare_pinn_kkt_data now uses bus_id_to_idx mapping
  instead of assuming bus_id == array_index (bus_id - 1)
- BUG FIX 2: pg columns now explicitly follow g_bus order from params,
  instead of unsorted column scanning

Changes from v5.0:
- Added collocation_ratio parameter to split train_idx into supervised + collocation
- Collocation samples: only X (load), labels discarded, trained with MAE_ε only
- Supervised samples: full labels (Pg + duals + KKT), trained with MAE_p + MAE_l + MAE_ε
- Validation uses supervised loss only (with labels) for early stopping
- Two DataLoaders, merged per-batch via itertools.zip_longest
"""

import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from itertools import zip_longest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Import custom modules
sys.path.append(os.path.dirname(__file__))

from dcopf_data_setup import (
    load_parameters_from_csv,
    DataSplitMode,
    split_data_by_mode
)
from dcopf_violation_metrics import (
    feasibility,
    compute_branch_violation_pu,
    compute_cost,
    compute_cost_gap_percentage
)
from dcopf_slack_utils import (
    identify_slack_bus_and_gens,
    update_params_with_slack_info,
    reconstruct_full_pg,
    compute_detailed_mae,
    compute_detailed_pg_violations_pu
)

# Import Slack-aware PINN-KKT model (v3.0 with collocation support)
from PinnModel import PinnModel


# =====================================================================
# Data Loading Function (v6.1 FIXED)
# =====================================================================

def load_and_prepare_pinn_kkt_data(file_path, params, column_names):
    """
    Load and prepare PINN-KKT training data

    v6.1 FIXES:
    -----------
    1. Load demand mapping: uses bus_id_to_idx dict instead of bus_id-1 indexing.
       This is critical for case300 where bus IDs like 7001, 9533 exist.
    2. Generator columns: explicitly built from g_bus order in params,
       instead of unsorted column scanning which may produce wrong column order.

    PINN-KKT requires all outputs: Pg + dual variables
    Returns 3 Pg values: y_pg_raw_non_slack (for training), y_pg_raw_all (for evaluation)
    Dual variable dimensions keep n_g (all generators)
    """
    import pandas as pd

    full_df = pd.read_csv(file_path)
    n_samples = len(full_df)
    n_buses = params['general']['n_buses']

    # ------------------------------------------------------------------ #
    # FIX 1: Load demand data — use bus_id_to_idx mapping                #
    # ------------------------------------------------------------------ #
    load_prefix = column_names['load_prefix']
    load_cols = [col for col in full_df.columns if col.startswith(load_prefix)]

    bus_id_to_idx = params['general']['bus_id_to_idx']
    x_data_raw = np.zeros((n_samples, n_buses), dtype='float32')

    for col_name in load_cols:
        bus_id = int(col_name[len(load_prefix):])
        if bus_id in bus_id_to_idx:
            x_data_raw[:, bus_id_to_idx[bus_id]] = full_df[col_name].values.astype('float32')
        else:
            print(f"[WARNING] load bus {bus_id} not in bus_id_to_idx, skipping")

    # ------------------------------------------------------------------ #
    # FIX 2: Generator columns — MUST match g_bus order from params      #
    # ------------------------------------------------------------------ #
    g_bus = params['general']['g_bus']
    gen_prefix = column_names['gen_prefix']
    pg_cols = [f"{gen_prefix}{int(gen_id)}" for gen_id in g_bus]

    missing = [c for c in pg_cols if c not in full_df.columns]
    if missing:
        raise ValueError(
            f"[load_and_prepare_pinn_kkt_data] Missing pg columns in CSV: {missing}\n"
            f"Available pg columns (first 10): "
            f"{sorted([c for c in full_df.columns if c.startswith(gen_prefix)])[:10]}"
        )

    y_pg_raw_all = full_df[pg_cols].values.astype('float32')

    expected_n_gen = params['general']['n_g']
    if len(pg_cols) != expected_n_gen:
        print(f"[WARNING] Number of generators in CSV ({len(pg_cols)}) "
              f"does not match parameters file ({expected_n_gen})!")

    # Extract non-Slack generator data
    non_slack_indices = params['general']['non_slack_gen_indices']
    y_pg_raw_non_slack = y_pg_raw_all[:, non_slack_indices]

    # Load dual variables (PINN-KKT specific) - keep n_g dimension
    y_lambda_raw = full_df[column_names['lambda']].values.reshape(-1, 1).astype('float32')

    mu_g_min_cols = [f"{column_names['mu_g_min_prefix']}{i}" for i in g_bus]
    y_mu_g_min_raw = full_df[mu_g_min_cols].values.astype('float32')

    mu_g_max_cols = [f"{column_names['mu_g_max_prefix']}{i}" for i in g_bus]
    y_mu_g_max_raw = full_df[mu_g_max_cols].values.astype('float32')

    valid_branch_indices = np.where(params['constraints']['Pl_max'] < 1e10)[0]
    valid_branch_ids = params['general']['branch_ids'][valid_branch_indices]

    mu_line_pos_cols = [f"{column_names['mu_line_pos_prefix']}{i}" for i in valid_branch_ids]
    y_mu_line_pos_raw = full_df[mu_line_pos_cols].values.astype('float32')

    mu_line_neg_cols = [f"{column_names['mu_line_neg_prefix']}{i}" for i in valid_branch_ids]
    y_mu_line_neg_raw = full_df[mu_line_neg_cols].values.astype('float32')

    return (x_data_raw, y_pg_raw_non_slack, y_pg_raw_all,
            y_lambda_raw, y_mu_g_min_raw, y_mu_g_max_raw,
            y_mu_line_pos_raw, y_mu_line_neg_raw)


# =====================================================================
# Evaluation Function
# =====================================================================

def evaluate_model(
        model, X_tensor, indices, raw_data_dict, scalers, params, device,
        test_data_external=None, test_params=None):
    """Evaluate PINN-KKT model (aligned with DNN)"""

    eval_params = test_params if test_params is not None else params
    model.eval()

    if test_data_external is not None:
        x_raw = test_data_external['x']
        y_true_pg_all = test_data_external['y_pg_all']
        x_scaled = scalers['x'].transform(x_raw)
        X_eval = torch.tensor(x_scaled, dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = model(X_eval)
            y_pred_pg_non_slack_scaled = outputs[0]
    else:
        y_true_pg_all = raw_data_dict['y_pg_all'][indices]
        x_raw = raw_data_dict['x'][indices]
        with torch.no_grad():
            outputs = model(X_tensor)
            y_pred_pg_non_slack_scaled = outputs[0]

    y_pred_pg_non_slack_scaled_np = y_pred_pg_non_slack_scaled.cpu().numpy()
    y_pred_pg_non_slack = scalers['pg_non_slack'].inverse_transform(y_pred_pg_non_slack_scaled_np)

    pd_total = x_raw.sum(axis=1)
    y_pred_pg_all = reconstruct_full_pg(pg_non_slack=y_pred_pg_non_slack, pd_total=pd_total, params=eval_params)

    mae_dict = compute_detailed_mae(y_true_all=y_true_pg_all, y_pred_non_slack=y_pred_pg_non_slack,
                                     y_pred_all=y_pred_pg_all, params=eval_params)
    gen_up_viol, gen_lo_viol, line_viol, balance_err = feasibility(
        y_pred_pg=y_pred_pg_all, x_pd=x_raw, params=eval_params)
    viol_dict = compute_detailed_pg_violations_pu(gen_up_viol=gen_up_viol, gen_lo_viol=gen_lo_viol, params=eval_params)
    viol_branch_pu = compute_branch_violation_pu(line_viol=line_viol, Pl_max=eval_params['constraints']['Pl_max'])

    cost_coeffs = {
        'C2': eval_params['constraints'].get('C_Pg_c2', np.zeros(y_true_pg_all.shape[1])),
        'C1': eval_params['constraints']['C_Pg'],
        'C0': eval_params['constraints'].get('C_Pg_c0', np.zeros(y_true_pg_all.shape[1]))
    }
    cost_true = compute_cost(y_true_pg_all, cost_coeffs)
    cost_pred = compute_cost(y_pred_pg_all, cost_coeffs)
    cost_gap_pct = compute_cost_gap_percentage(cost_true, cost_pred)

    return {
        'mae_pg_non_slack': mae_dict['mae_non_slack'], 'mae_pg_slack': mae_dict['mae_slack'],
        'viol_pg_non_slack': viol_dict['viol_non_slack'], 'viol_pg_slack': viol_dict['viol_slack'],
        'viol_branch': viol_branch_pu, 'cost_gap_percent': cost_gap_pct,
    }


# =====================================================================
# Main Training Function
# =====================================================================

def train_pinn_kkt_dcopf(
        case_name, params_path, dataset_path, column_names,
        n_train_use=10000, neurons_pg=[128, 128], neurons_lm=[128, 128],
        n_epochs=1000, patience=20, min_delta=1e-6, batch_size=128,
        learning_rate=1e-3, weight1=0.005, weight2=0.005, seed=42,
        device='cuda', split_mode=DataSplitMode.RANDOM_SPLIT,
        test_data_path=None, scale_type='minmax', test_params_path=None,
        n_test_samples=1000, collocation_ratio=0.0):

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    assert 0.0 <= collocation_ratio < 1.0, f"collocation_ratio must be in [0.0, 1.0), got {collocation_ratio}"
    use_collocation = collocation_ratio > 0.0

    # Load parameters
    params = load_parameters_from_csv(case_name, params_path, is_api=False)
    slack_info = identify_slack_bus_and_gens(params)
    params = update_params_with_slack_info(params, slack_info)

    test_params = None
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError('API_TEST mode requires test_params_path')
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_slack_info = identify_slack_bus_and_gens(test_params)
        test_params = update_params_with_slack_info(test_params, test_slack_info)
    else:
        test_params = params

    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_g']
    n_gen_non_slack = params['general']['n_g_non_slack']
    n_line = params['general']['n_line']

    # Load data
    (x_data_raw, y_pg_raw_non_slack, y_pg_raw_all,
     y_lambda_raw, y_mu_g_min_raw, y_mu_g_max_raw,
     y_mu_line_pos_raw, y_mu_line_neg_raw) = load_and_prepare_pinn_kkt_data(
        dataset_path, params, column_names)

    raw_data_dict = {'x': x_data_raw, 'y_pg_non_slack': y_pg_raw_non_slack, 'y_pg_all': y_pg_raw_all}

    # Data split
    train_idx, val_idx, test_idx, x_test_external, y_test_external = split_data_by_mode(
        x_data_raw=x_data_raw, y_pg_raw=y_pg_raw_all, mode=split_mode,
        n_train_use=n_train_use, seed=seed, test_data_path=test_data_path,
        params=params, column_names=column_names, n_test_samples=n_test_samples)

    # Split train_idx into supervised_idx + collocation_idx
    if use_collocation:
        rng_split = np.random.default_rng(seed + 1000)
        n_train_total = len(train_idx)
        n_collocation = int(n_train_total * collocation_ratio)
        n_supervised = n_train_total - n_collocation
        shuffled_train_idx = rng_split.permutation(train_idx)
        supervised_idx = shuffled_train_idx[:n_supervised]
        collocation_idx = shuffled_train_idx[n_supervised:]
        print(f"\n[Collocation Split] Total: {n_train_total}  Supervised: {n_supervised}  Collocation: {n_collocation}")
    else:
        supervised_idx = train_idx
        collocation_idx = np.array([], dtype=int)

    # Data normalization
    if scale_type == 'minmax':
        x_scaler = MinMaxScaler().fit(x_data_raw[train_idx])
    elif scale_type == 'standard':
        x_scaler = StandardScaler().fit(x_data_raw[train_idx])
    else:
        x_scaler = MinMaxScaler().fit(x_data_raw[train_idx])

    pg_non_slack_scaler = MinMaxScaler().fit(y_pg_raw_non_slack[train_idx])
    lambda_scaler = MinMaxScaler().fit(y_lambda_raw[train_idx])
    mu_g_min_scaler = MinMaxScaler().fit(y_mu_g_min_raw[train_idx])
    mu_g_max_scaler = MinMaxScaler().fit(y_mu_g_max_raw[train_idx])
    mu_line_pos_scaler = MinMaxScaler().fit(y_mu_line_pos_raw[train_idx])
    mu_line_neg_scaler = MinMaxScaler().fit(y_mu_line_neg_raw[train_idx])

    scalers = {'x': x_scaler, 'pg_non_slack': pg_non_slack_scaler, 'lambda': lambda_scaler,
               'mu_g_min': mu_g_min_scaler, 'mu_g_max': mu_g_max_scaler,
               'mu_line_pos': mu_line_pos_scaler, 'mu_line_neg': mu_line_neg_scaler}

    # Supervised DataLoader
    x_sup_scaled = x_scaler.transform(x_data_raw[supervised_idx])
    X_sup = torch.from_numpy(x_sup_scaled).float().to(device)
    Y_sup_pg = torch.from_numpy(pg_non_slack_scaler.transform(y_pg_raw_non_slack[supervised_idx])).float().to(device)
    Y_sup_lambda = torch.from_numpy(lambda_scaler.transform(y_lambda_raw[supervised_idx])).float().to(device)
    Y_sup_mu_g_min = torch.from_numpy(mu_g_min_scaler.transform(y_mu_g_min_raw[supervised_idx])).float().to(device)
    Y_sup_mu_g_max = torch.from_numpy(mu_g_max_scaler.transform(y_mu_g_max_raw[supervised_idx])).float().to(device)
    Y_sup_mu_line_pos = torch.from_numpy(mu_line_pos_scaler.transform(y_mu_line_pos_raw[supervised_idx])).float().to(device)
    Y_sup_mu_line_neg = torch.from_numpy(mu_line_neg_scaler.transform(y_mu_line_neg_raw[supervised_idx])).float().to(device)
    Y_sup_physics = torch.zeros((len(X_sup), 1), dtype=torch.float32, device=device)

    sup_dataset = Data.TensorDataset(X_sup, Y_sup_pg, Y_sup_lambda, Y_sup_mu_g_min,
                                      Y_sup_mu_g_max, Y_sup_mu_line_pos, Y_sup_mu_line_neg, Y_sup_physics)
    sup_loader = Data.DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)

    # Collocation DataLoader
    if use_collocation:
        x_col_scaled = x_scaler.transform(x_data_raw[collocation_idx])
        X_col = torch.from_numpy(x_col_scaled).float().to(device)
        col_dataset = Data.TensorDataset(X_col)
        col_loader = Data.DataLoader(col_dataset, batch_size=batch_size, shuffle=True)
    else:
        col_loader = None

    # Validation data
    x_val_scaled = x_scaler.transform(x_data_raw[val_idx])
    X_val = torch.from_numpy(x_val_scaled).float().to(device)
    Y_val_pg = torch.from_numpy(pg_non_slack_scaler.transform(y_pg_raw_non_slack[val_idx])).float().to(device)
    Y_val_lambda = torch.from_numpy(lambda_scaler.transform(y_lambda_raw[val_idx])).float().to(device)
    Y_val_mu_g_min = torch.from_numpy(mu_g_min_scaler.transform(y_mu_g_min_raw[val_idx])).float().to(device)
    Y_val_mu_g_max = torch.from_numpy(mu_g_max_scaler.transform(y_mu_g_max_raw[val_idx])).float().to(device)
    Y_val_mu_line_pos = torch.from_numpy(mu_line_pos_scaler.transform(y_mu_line_pos_raw[val_idx])).float().to(device)
    Y_val_mu_line_neg = torch.from_numpy(mu_line_neg_scaler.transform(y_mu_line_neg_raw[val_idx])).float().to(device)
    Y_val_physics = torch.zeros((len(X_val), 1), dtype=torch.float32, device=device)

    # Test data
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        X_test = None
    else:
        x_test_scaled = x_scaler.transform(x_data_raw[test_idx])
        X_test = torch.tensor(x_test_scaled, dtype=torch.float32)

    # Simulation parameters for PINN model
    simulation_parameters = params.copy()
    simulation_parameters['training'] = {
        'neurons_in_hidden_layers_Pg': neurons_pg,
        'neurons_in_hidden_layers_Lm': neurons_lm
    }
    if scale_type == 'minmax':
        simulation_parameters['pd_scale_type'] = 'minmax'
        simulation_parameters['pd_min'] = x_scaler.data_min_
        simulation_parameters['pd_max'] = x_scaler.data_max_
    elif scale_type == 'standard':
        simulation_parameters['pd_scale_type'] = 'standard'
        simulation_parameters['pd_mean'] = x_scaler.mean_
        simulation_parameters['pd_std'] = x_scaler.scale_
    else:
        simulation_parameters['pd_scale_type'] = None

    Lg_Max = [np.max(np.abs(y_lambda_raw)), np.max(np.abs(y_mu_g_max_raw)),
              np.max(np.abs(y_mu_g_min_raw)), np.max(np.abs(y_mu_line_pos_raw)),
              np.max(np.abs(y_mu_line_neg_raw))]
    simulation_parameters['Lg_Max'] = Lg_Max

    # FIX #3 (PinnLayer v3.0): pass full MinMaxScaler params for strict
    # dual-variable de-scaling (physical = scaled * (max - min) + min).
    # Without this, PinnLayer falls back to legacy (scaled * Lg_Max), which is
    # only correct when data_min_ == 0.
    simulation_parameters['dual_scalers'] = {
        'lambda':      (lambda_scaler.data_min_,      lambda_scaler.data_max_),
        'mu_g_max':    (mu_g_max_scaler.data_min_,    mu_g_max_scaler.data_max_),
        'mu_g_min':    (mu_g_min_scaler.data_min_,    mu_g_min_scaler.data_max_),
        'mu_line_pos': (mu_line_pos_scaler.data_min_, mu_line_pos_scaler.data_max_),
        'mu_line_neg': (mu_line_neg_scaler.data_min_, mu_line_neg_scaler.data_max_),
    }

    # KKT 归一化分母: 对齐作者源码 np.max(L_max) = 系统最大单点负荷。
    # 直接取训练集原始物理负荷的最大值, 与 scale_type 无关。
    simulation_parameters['load_scale'] = float(x_data_raw[train_idx].max())

    model = PinnModel(weight1=weight1, weight2=weight2, simulation_parameters=simulation_parameters,
                       learning_rate=learning_rate, device=device)

    # Training Loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf'); patience_counter = 0; best_model_state = None
    n_supervised_total = len(supervised_idx)
    n_collocation_total = len(collocation_idx) if use_collocation else 0

    training_start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_sup_loss = 0.0; epoch_col_loss = 0.0; epoch_n_sup = 0; epoch_n_col = 0

        if use_collocation:
            col_scale = n_supervised_total / (n_supervised_total + n_collocation_total)
            for sup_batch, col_batch in zip_longest(sup_loader, col_loader):
                model.optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=device)

                if sup_batch is not None:
                    X_batch = sup_batch[0]; Y_batch = sup_batch[1:]
                    outputs = model(X_batch)
                    sup_loss, _ = model.compute_supervised_loss(outputs, Y_batch)
                    batch_loss = batch_loss + sup_loss
                    epoch_sup_loss += sup_loss.item() * len(X_batch); epoch_n_sup += len(X_batch)

                if col_batch is not None:
                    X_col_batch = col_batch[0]
                    col_outputs = model(X_col_batch)
                    col_loss, _ = model.compute_collocation_loss(col_outputs)
                    batch_loss = batch_loss + col_scale * col_loss
                    epoch_col_loss += col_loss.item() * len(X_col_batch); epoch_n_col += len(X_col_batch)

                batch_loss.backward(); model.optimizer.step()

            avg_sup_loss = epoch_sup_loss / max(epoch_n_sup, 1)
            avg_col_loss = epoch_col_loss / max(epoch_n_col, 1)
            train_loss = avg_sup_loss + avg_col_loss
        else:
            epoch_loss = 0.0
            for batch_data in sup_loader:
                X_batch = batch_data[0]; Y_batch = batch_data[1:]
                model.optimizer.zero_grad()
                outputs = model(X_batch)
                loss, _ = model.compute_loss(outputs, Y_batch)
                loss.backward(); model.optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            train_loss = epoch_loss / n_supervised_total
            avg_sup_loss = train_loss; avg_col_loss = 0.0

        train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_targets = (Y_val_pg, Y_val_lambda, Y_val_mu_g_min, Y_val_mu_g_max,
                           Y_val_mu_line_pos, Y_val_mu_line_neg, Y_val_physics)
            val_loss_tensor, _ = model.compute_loss(val_outputs, val_targets)
            val_loss = val_loss_tensor.item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss; patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if use_collocation:
            print(f"Epoch {epoch}/{n_epochs} - sup: {avg_sup_loss:.6f} col: {avg_col_loss:.6f} "
                  f"val: {val_loss:.6f} pat: {patience_counter}/{patience}")
        else:
            print(f"Epoch {epoch}/{n_epochs} - train: {train_loss:.6f} val: {val_loss:.6f} "
                  f"pat: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n[Early Stopping] Best val_loss: {best_val_loss:.6f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    train_time = time.time() - training_start

    # Evaluation
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_metrics = evaluate_model(model, None, None, raw_data_dict, scalers, params, device,
                                       test_data_external={'x': x_test_external, 'y_pg_all': y_test_external},
                                       test_params=test_params)
    else:
        test_metrics = evaluate_model(model, X_test.to(device), test_idx, raw_data_dict, scalers,
                                       params, device, test_params=test_params)

    # Speed test
    model.eval()
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_sample = torch.tensor(x_scaler.transform(x_test_external[:1]), dtype=torch.float32, device=device)
    else:
        test_sample = X_test[:1].to(device)

    times = []
    with torch.no_grad():
        for _ in range(10): _ = model(test_sample)
        if device.type == 'cuda': torch.cuda.synchronize()
        for _ in range(100):
            t_start = time.time(); _ = model(test_sample)
            if device.type == 'cuda': torch.cuda.synchronize()
            times.append(time.time() - t_start)
    inference_time_ms = np.mean(times) * 1000

    # Print results
    print("\n" + "=" * 70 + "\nTest Set Results\n" + "=" * 70)
    print(f"\nNon-Slack: MAE={test_metrics['mae_pg_non_slack']:.4f}%  Viol={test_metrics['viol_pg_non_slack']:.4f} p.u.")
    print(f"Slack:     MAE={test_metrics['mae_pg_slack']:.4f}%  Viol={test_metrics['viol_pg_slack']:.4f} p.u.")
    print(f"Branch:    Viol={test_metrics['viol_branch']:.4f} p.u.")
    print(f"Cost Gap:  {test_metrics['cost_gap_percent']:.4f}%")
    print(f"Training:  {train_time:.2f} s   Inference: {inference_time_ms:.4f} ms")
    print("=" * 70 + "\n")


# =====================================================================
# Main Program
# =====================================================================

if __name__ == '__main__':
    CASE_NAME = 'pglib_opf_case118_ieee'; CASE_SHORT_NAME = 'case118'
    SPLIT_MODE = DataSplitMode.VALID_FIXED
    N_TRAIN_USE = 35000; N_TEST_SAMPLES = 1000
    N_EPOCHS = 1000; PATIENCE = 20; LEARNING_RATE = 1e-3; BATCH_SIZE = 64
    NEURONS_PG = [128, 64]; NEURONS_LM = [128, 64]
    # v3.2 KKT_error ≈ 13 (vs v2.1 ≈ 0.3), so WEIGHT2 needs ~43x reduction
    # to maintain the same effective KKT contribution to total loss.
    WEIGHT1 = 0.05; WEIGHT2 = 0.05; SCALE_TYPE = 'minmax'; SEED = 42
    COLLOCATION_RATIO = 0.5

    ROOT_DIR = "/lambda/nfs/lxy/dcopf_project/data"; TRAIN_VARIANCE = "v=0.12"; TEST_VARIANCE = "v=0.25"
    COLUMN_NAMES = {'load_prefix': 'pd', 'gen_prefix': 'pg', 'lambda': 'lambda',
                    'mu_g_min_prefix': 'mu_g_min_', 'mu_g_max_prefix': 'mu_g_max_',
                    'mu_line_pos_prefix': 'mu_line_max_', 'mu_line_neg_prefix': 'mu_line_min_'}

    params_path = os.path.join(ROOT_DIR, "DCOPF Constraints", CASE_SHORT_NAME)
    train_data_path = os.path.join(ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TRAIN_VARIANCE})",
                                    f"{CASE_NAME}_dataset_with_duals.csv")

    if SPLIT_MODE == DataSplitMode.GENERALIZATION:
        test_data_path = os.path.join(ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TEST_VARIANCE})",
                                       f"{CASE_NAME}_dataset_with_duals.csv"); test_params_path = None
    elif SPLIT_MODE == DataSplitMode.API_TEST:
        test_data_path = os.path.join(ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}(v=api)",
                                       f"{CASE_NAME}__api_dataset_with_duals.csv")
        test_params_path = os.path.join(ROOT_DIR, "DCOPF Constraints", f"{CASE_SHORT_NAME}(api)")
    else:
        test_data_path = None; test_params_path = None

    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    train_pinn_kkt_dcopf(
        case_name=CASE_NAME, params_path=params_path, dataset_path=train_data_path,
        column_names=COLUMN_NAMES, n_train_use=N_TRAIN_USE, neurons_pg=NEURONS_PG, neurons_lm=NEURONS_LM,
        n_epochs=N_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        weight1=WEIGHT1, weight2=WEIGHT2, seed=SEED, device=device_name, split_mode=SPLIT_MODE,
        test_data_path=test_data_path, scale_type=SCALE_TYPE, test_params_path=test_params_path,
        n_test_samples=N_TEST_SAMPLES, collocation_ratio=COLLOCATION_RATIO)