# -*- coding: utf-8 -*-
"""
Lagrangian Dual DCOPF - Streamlined Version

Implements Lagrangian dual method for DC Optimal Power Flow:
- Predicts only non-slack generator outputs
- Uses PTDF matrix for constraint evaluation (no power flow needed)
- Supports 4 data modes: random_split, valid_fixed, generalization, api_test
- Prints only test set results
- All comments and outputs in English
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
import copy

# Import DCOPF modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dcopf_data_setup import (
    load_parameters_from_csv,
    DataSplitMode,
    split_data_by_mode,
    load_and_prepare_data_generalization
)
from dcopf_violation_metrics import (
    feasibility as dc_feasibility,
    compute_cost,
    compute_cost_gap_percentage,
    compute_branch_violation_pu,
    compute_mae_percentage
)
from dcopf_config import PathConfig

from dcopf_slack_utils import (
    identify_slack_bus_and_gens,
    update_params_with_slack_info,
    reconstruct_full_pg,
    compute_detailed_mae,
    compute_detailed_pg_violations_pu
)

from sklearn.preprocessing import MinMaxScaler


# =====================================================================
# Data Loading (independent, no dnn_dcopf_main dependency)
# =====================================================================
def load_and_prepare_data(full_dataset_path, params):
    """
    Load dataset CSV → (x_data_raw, y_pg_raw_non_slack, y_pg_raw_all).

    Uses bus_id_to_idx from params for correct column mapping
    (handles case300 non-contiguous bus IDs).
    Generator columns extracted in g_bus order (matching Pg_min/Pg_max/Map_g).
    """
    full_df = pd.read_csv(full_dataset_path)
    n_samples = len(full_df)
    n_buses = params['general']['n_buses']

    # ---- Load (Pd) columns ----
    pd_cols_available = [col for col in full_df.columns if col.startswith('pd')]
    pd_cols_available = sorted(pd_cols_available, key=lambda c: int(c[2:]))

    bus_id_to_idx = params['general']['bus_id_to_idx']
    x_data_raw_full = np.zeros((n_samples, n_buses), dtype='float32')
    for col in pd_cols_available:
        bus_id = int(col[2:])
        if bus_id in bus_id_to_idx:
            x_data_raw_full[:, bus_id_to_idx[bus_id]] = full_df[col].values.astype('float32')
        else:
            print(f"[WARNING] pd bus {bus_id} not in bus_id_to_idx, skipping")

    # ---- Generator (Pg) columns — MUST match g_bus order ----
    g_bus = params['general']['g_bus']
    pg_cols = [f'pg{int(gen_id)}' for gen_id in g_bus]

    missing = [c for c in pg_cols if c not in full_df.columns]
    if missing:
        raise ValueError(
            f"[load_and_prepare_data] Missing pg columns: {missing}\n"
            f"Available (first 10): "
            f"{sorted([c for c in full_df.columns if c.startswith('pg')])[:10]}"
        )

    y_pg_raw = full_df[pg_cols].values.astype('float32')

    # ---- Extract non-slack ----
    non_slack_indices = params['general']['non_slack_gen_indices']
    y_pg_raw_non_slack = y_pg_raw[:, non_slack_indices]

    return x_data_raw_full, y_pg_raw_non_slack, y_pg_raw


# =====================================================================
# Neural Network
# =====================================================================
class LagrangianDNN_DCOPF(nn.Module):
    """
    Lagrangian Dual Neural Network for DCOPF

    Input: Load demand (n_buses)
    Output: Non-slack generator outputs (n_g_non_slack)
    """

    def __init__(self, input_dim, n_g_non_slack, hidden_dims=[128, 128]):
        super().__init__()
        self.n_g_non_slack = n_g_non_slack
        output_dim = n_g_non_slack

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =====================================================================
# Lagrangian Constraint Violation Computation
# =====================================================================
def compute_constraint_violations(Y_pred_scaled, X_scaled, scalers, params):
    """
    Compute constraint violations for Lagrangian dual method.

    Follows Fioretto et al. AAAI-20 violation degree definition:
        nu_c = (1/b) * sum over batch of violation degree of constraint c

    Constraints:
        - Generator limits: Pg_min <= Pg <= Pg_max
        - Branch flow limits: |Pl| <= Pl_max (finite-capacity lines only)

    Args:
        Y_pred_scaled : Scaled predictions, shape (batch, n_g_non_slack)
        X_scaled      : Scaled load, shape (batch, n_buses)
        scalers       : Scalers dictionary
        params        : Parameters dictionary

    Returns:
        violations : dict with keys 'nu_pg' and 'nu_branch' (plain floats)
    """

    # --- 1. Denormalize ---
    pg_pred_non_slack = scalers['y_pg_non_slack'].inverse_transform(
        Y_pred_scaled.detach().cpu().numpy()
    )
    x_pd = scalers['x'].inverse_transform(
        X_scaled.detach().cpu().numpy()
    )

    # --- 2. Reconstruct full Pg via power balance (slack fills the gap) ---
    pd_total     = x_pd.sum(axis=1)
    pg_pred_full = reconstruct_full_pg(pg_pred_non_slack, pd_total, params)

    # --- 3. Raw constraint violations ---
    gen_up_viol, gen_lo_viol, line_viol, _ = dc_feasibility(
        pg_pred_full, x_pd, params
    )

    violations = {}

    # --- 4. nu_pg: mean over samples and all generators ---
    violations['nu_pg'] = float(np.mean(gen_up_viol + gen_lo_viol))

    # --- 5. nu_branch: mean over samples, finite-capacity branches only ---
    Pl_max    = params['constraints']['Pl_max']
    valid_idx = np.where(Pl_max < 1e10)[0]
    if len(valid_idx) > 0:
        violations['nu_branch'] = float(np.mean(line_viol[:, valid_idx]))
    else:
        violations['nu_branch'] = 0.0

    return violations


# =====================================================================
# Training Function
# =====================================================================
def train_lagrangian_dual(model, train_loader, val_loader, scalers, params,
                          n_epochs=1000, lr=1e-3, rho=1e-2, device='cpu',
                          patience=20, min_delta=1e-6):
    """
    Lagrangian Dual training (strictly follows Fioretto et al. AAAI-20, Algorithm 1)

    Algorithm 1:
      1  lambda^0 <- 0  for all c in C
      2  for epoch k = 0, 1, ... do
      3    foreach (x, y) <- minibatch(X, Y) of size b do
      4      y_hat <- model(x)
      5      Lo(y_hat, y) <- (1/b) sum MSE
      6      Lc(x, y_hat) <- (1/b) sum_c lambda^k_c * nu_c(x, y_hat)
      7      w <- w - alpha * grad_w(Lo + Lc)          [L1: update weights]
      8      foreach c in C do
      9        lambda^{k+1}_c <- max(0, lambda^k_c + rho * nu_c(x, y_hat))
                                                        [L2: update multipliers]

    Key: lambda is updated INSIDE the mini-batch loop, immediately after each
    weight update. This is the strict replication of the paper algorithm.

    Early stopping monitors pure val MSE loss (no Lagrangian penalty).
    Best model weights (lowest val_loss) are restored before returning.
    Lambda multipliers are NOT rolled back — they reflect the full training history.

    Args:
        model        : Neural network model
        train_loader : Training data loader
        val_loader   : Validation data loader
        scalers      : Scalers dictionary
        params       : Parameters dictionary
        n_epochs     : Maximum number of training epochs
        lr           : Learning rate for model weights alpha
        rho          : Step size for Lagrangian multiplier updates rho
        device       : Computing device
        patience     : Early stopping patience (epochs without improvement)
        min_delta    : Minimum improvement in val_loss to reset patience counter
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Algorithm 1, line 1: initialize all multipliers to 0
    lambda_multipliers = {
        'lambda_pg':     0.0,
        'lambda_branch': 0.0,
    }

    history = {'train_loss': [], 'val_loss': []}

    # --- Early stopping state ---
    best_val_loss    = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"\n{'=' * 70}")
    print(f"Lagrangian Dual Training  (strict Algorithm 1 replication)")
    print(f"{'=' * 70}")
    print(f"Optimizer learning rate alpha: {lr},  Lagrangian step size rho: {rho}")
    print(f"Early stopping — patience: {patience}, min_delta: {min_delta}")
    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  "
          f"{'lam_pg':>10}  {'lam_br':>10}  "
          f"{'nu_pg(avg)':>12}  {'nu_br(avg)':>12}  {'Pat':>5}")
    print(f"{'-' * 100}")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss      = 0.0
        n_batches       = 0

        # For logging only
        epoch_nu_pg     = 0.0
        epoch_nu_branch = 0.0

        # Algorithm 1, line 3: iterate over mini-batches
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()

            # Algorithm 1, line 4: forward pass
            Y_pred = model(X_batch)

            # Algorithm 1, line 5: supervised loss Lo
            mse_loss = criterion(Y_pred, Y_batch)

            # Compute violation degrees nu_c for current batch
            violations = compute_constraint_violations(
                Y_pred, X_batch, scalers, params
            )

            # Algorithm 1, line 6: constraint penalty Lc = sum lambda_c * nu_c
            constraint_loss = 0.0
            for key, nu_val in violations.items():
                lambda_key = f"lambda_{key.replace('nu_', '')}"
                if lambda_key in lambda_multipliers:
                    constraint_loss += lambda_multipliers[lambda_key] * float(nu_val)

            # Algorithm 1, line 7 (L1): update model weights w
            total_loss = mse_loss + constraint_loss
            total_loss.backward()
            optimizer.step()

            # Algorithm 1, lines 8-9 (L2): update lambda immediately after weight
            # update, using the SAME batch's violation degrees
            for key, nu_val in violations.items():
                lambda_key = f"lambda_{key.replace('nu_', '')}"
                if lambda_key in lambda_multipliers:
                    lambda_multipliers[lambda_key] = max(
                        0.0,
                        lambda_multipliers[lambda_key] + rho * float(nu_val)
                    )

            epoch_loss      += total_loss.item()
            epoch_nu_pg     += float(violations.get('nu_pg',     0.0))
            epoch_nu_branch += float(violations.get('nu_branch', 0.0))
            n_batches       += 1

        avg_train_loss  = epoch_loss      / n_batches
        avg_nu_pg       = epoch_nu_pg     / n_batches
        avg_nu_branch   = epoch_nu_branch / n_batches
        history['train_loss'].append(avg_train_loss)

        # Validation loss (pure MSE, no penalty) — used for early stopping
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                val_loss += criterion(model(X_val), Y_val).item()
                n_val    += 1

        avg_val_loss = val_loss / n_val if n_val > 0 else 0.0
        history['val_loss'].append(avg_val_loss)

        # --- Early stopping check ---
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        # Print every epoch
        print(
            f"{epoch + 1:6d}  "
            f"{avg_train_loss:12.6f}  "
            f"{avg_val_loss:12.6f}  "
            f"{lambda_multipliers['lambda_pg']:10.5f}  "
            f"{lambda_multipliers['lambda_branch']:10.5f}  "
            f"{avg_nu_pg:12.6f}  "
            f"{avg_nu_branch:12.6f}  "
            f"{patience_counter:>3d}/{patience}"
        )

        if patience_counter >= patience:
            print(f"\n[Early Stopping] No improvement for {patience} epochs. "
                  f"Best val_loss: {best_val_loss:.6f} (epoch {epoch + 1 - patience})")
            break

    # Restore best model weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[Early Stopping] Best model weights restored (val_loss={best_val_loss:.6f})")

    print(f"\nFinal Lagrangian Multipliers:")
    for key, val in lambda_multipliers.items():
        print(f"  {key}: {val:.6f}")

    return history, lambda_multipliers


# =====================================================================
# Evaluation Function
# =====================================================================
def evaluate_model(
        model, X_tensor, indices, raw_data, scalers, params, device,
        test_data_external=None, test_params=None
):
    """
    Evaluate model on test set

    Returns metrics:
    - MAE: non_slack, slack
    - Violations (p.u.): non_slack, slack, branch
    - Cost gap (%)
    """
    model.eval()
    eval_params = test_params if test_params is not None else params

    # Determine if using external test data
    if test_data_external is not None:
        x_raw_eval = test_data_external['x']
        y_true_pg_all = test_data_external['y_pg_all']

        # Scale test data
        x_scaled = scalers['x'].transform(x_raw_eval)
        X_tensor_eval = torch.tensor(x_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            y_pred_non_slack_scaled = model(X_tensor_eval).cpu().numpy()
    else:
        # Normal mode: use indices
        with torch.no_grad():
            y_pred_non_slack_scaled = model(X_tensor).cpu().numpy()

        x_raw_eval = raw_data['x'][indices]
        y_true_pg_all = raw_data['y_pg_all'][indices]

    # Inverse transform non-slack predictions
    y_pred_non_slack = scalers['y_pg_non_slack'].inverse_transform(
        y_pred_non_slack_scaled
    )

    # Reconstruct full Pg vector (including slack)
    pd_total = x_raw_eval.sum(axis=1)
    y_pred_pg_all = reconstruct_full_pg(y_pred_non_slack, pd_total, eval_params)

    # --- 1. Calculate detailed MAE (%) ---
    mae_dict = compute_detailed_mae(
        y_true_pg_all, y_pred_non_slack, y_pred_pg_all, eval_params
    )

    # --- 2. Calculate DCOPF violations (using full Pg) ---
    gen_up_viol_pu, gen_lo_viol_pu, line_viol_pu, _ = dc_feasibility(
        y_pred_pg_all, x_raw_eval, eval_params
    )

    # --- 3. Calculate detailed Pg violations (p.u.) ---
    viol_dict = compute_detailed_pg_violations_pu(
        gen_up_viol_pu, gen_lo_viol_pu, eval_params
    )

    # --- 4. Calculate Branch violations (p.u.) ---
    branch_violation_pu = compute_branch_violation_pu(
        line_viol_pu, eval_params['constraints']['Pl_max']
    )

    # --- 5. Calculate Cost metrics ---
    cost_coeffs = {
        'C2': eval_params['constraints'].get('C_Pg_c2', np.zeros(y_true_pg_all.shape[1])),
        'C1': eval_params['constraints']['C_Pg'],
        'C0': eval_params['constraints'].get('C_Pg_c0', np.zeros(y_true_pg_all.shape[1]))
    }

    cost_true = compute_cost(y_true_pg_all, cost_coeffs)
    cost_pred = compute_cost(y_pred_pg_all, cost_coeffs)
    cost_gap_pct = compute_cost_gap_percentage(cost_true, cost_pred)

    return {
        # MAE metrics
        'mae_pg_non_slack': mae_dict['mae_non_slack'],
        'mae_pg_slack': mae_dict['mae_slack'],
        # Violation metrics (p.u.)
        'viol_pg_non_slack': viol_dict['viol_non_slack'],
        'viol_pg_slack': viol_dict['viol_slack'],
        'viol_branch': branch_violation_pu,
        # Cost metrics
        'cost_gap_percent': cost_gap_pct,
    }


# =====================================================================
# Main Experiment Function
# =====================================================================
def lagrangian_dcopf_experiment(
        case_name,
        params_path,
        dataset_path,
        # Data mode parameters
        split_mode=DataSplitMode.RANDOM_SPLIT,
        n_train_use=10000,
        test_data_path=None,
        test_params_path=None,
        n_test_samples=1000,
        # Training parameters
        n_epochs=1000,
        patience=20,
        min_delta=1e-6,
        learning_rate=0.001,
        lagrangian_lr=0.01,
        hidden_layers=[128, 128],
        batch_size=128,
        seed=42,
        device='cuda',
        column_names=None
):
    """
    Lagrangian Dual DCOPF main experiment function

    Supports four data modes:
    - RANDOM_SPLIT: Random split (10:1:1)
    - VALID_FIXED: Fixed validation/test sets
    - GENERALIZATION: Cross-distribution test
    - API_TEST: Different topology test

    Early stopping monitors pure val MSE (no Lagrangian penalty).
    Best model weights are restored before evaluation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\nRunning: {split_mode.value} - {case_name}\n")

    # ========================================================================
    # 1. Load training parameters
    # ========================================================================
    params = load_parameters_from_csv(case_name, params_path, is_api=False)

    # Automatically identify slack bus and generators
    slack_info = identify_slack_bus_and_gens(params)
    params = update_params_with_slack_info(params, slack_info)

    # If API_TEST mode, load test parameters
    test_params = None
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError("API_TEST mode requires test_params_path")
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_slack_info = identify_slack_bus_and_gens(test_params)
        test_params = update_params_with_slack_info(test_params, test_slack_info)
    else:
        test_params = params

    # Default column names
    if column_names is None:
        column_names = {
            'load_prefix': 'pd',
            'gen_prefix': 'pg',
            'lambda': 'lambda',
            'mu_g_min_prefix': 'mu_g_min_',
            'mu_g_max_prefix': 'mu_g_max_',
            'mu_line_pos_prefix': 'mu_line_max_',
            'mu_line_neg_prefix': 'mu_line_min_',
        }

    # ========================================================================
    # 2. Load and prepare training data (independent, no dnn_dcopf_main dep)
    # ========================================================================
    x_data_raw, y_pg_raw_non_slack, y_pg_raw_all = load_and_prepare_data(
        dataset_path, params
    )

    raw_data = {
        'x': x_data_raw,
        'y_pg_non_slack': y_pg_raw_non_slack,
        'y_pg_all': y_pg_raw_all
    }

    # ========================================================================
    # 3. Data splitting
    # ========================================================================
    train_idx, val_idx, test_idx, x_test_external, y_test_external = split_data_by_mode(
        x_data_raw=x_data_raw,
        y_pg_raw=y_pg_raw_all,
        mode=split_mode,
        n_train_use=n_train_use,
        seed=seed,
        test_data_path=test_data_path,
        params=params,
        column_names=column_names,
        n_test_samples=n_test_samples
    )

    # ========================================================================
    # 4. Data normalization
    # ========================================================================
    x_scaler = MinMaxScaler().fit(x_data_raw[train_idx])
    y_pg_non_slack_scaler = MinMaxScaler().fit(y_pg_raw_non_slack[train_idx])

    scalers = {
        'x': x_scaler,
        'y_pg_non_slack': y_pg_non_slack_scaler,
    }

    x_train_scaled = x_scaler.transform(x_data_raw[train_idx])
    y_train_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[train_idx])
    x_val_scaled = x_scaler.transform(x_data_raw[val_idx])
    y_val_scaled = y_pg_non_slack_scaler.transform(y_pg_raw_non_slack[val_idx])

    # Handle test set
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        X_test = None  # Will be created in evaluate
    else:
        x_test_scaled = x_scaler.transform(x_data_raw[test_idx])
        X_test = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)

    # Convert to PyTorch tensors
    X_train = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
    Y_train = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
    X_val = torch.tensor(x_val_scaled, dtype=torch.float32, device=device)
    Y_val = torch.tensor(y_val_scaled, dtype=torch.float32, device=device)

    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

    # ========================================================================
    # 5. Create model and train
    # ========================================================================
    model = LagrangianDNN_DCOPF(
        input_dim=x_data_raw.shape[1],
        n_g_non_slack=params['general']['n_g_non_slack'],
        hidden_dims=hidden_layers
    ).to(device)

    # Training
    t0 = time.perf_counter()
    history, final_lambdas = train_lagrangian_dual(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        scalers=scalers,
        params=params,
        n_epochs=n_epochs,
        lr=learning_rate,
        rho=lagrangian_lr,
        device=device,
        patience=patience,
        min_delta=min_delta
    )
    train_time = time.perf_counter() - t0

    # ========================================================================
    # 6. Model evaluation (test set only)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"Test Set Results")
    print(f"{'=' * 70}")

    if split_mode == DataSplitMode.GENERALIZATION:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }
        test_metrics = evaluate_model(
            model, None, None, raw_data, scalers, params, device,
            test_data_external=test_data_external_dict,
            test_params=test_params
        )
    elif split_mode == DataSplitMode.API_TEST:
        test_data_external_dict = {
            'x': x_test_external,
            'y_pg_all': y_test_external
        }
        test_metrics = evaluate_model(
            model, None, None, raw_data, scalers, params, device,
            test_data_external=test_data_external_dict,
            test_params=test_params
        )
    else:
        test_metrics = evaluate_model(
            model, X_test, test_idx, raw_data, scalers, params, device,
            test_params=test_params
        )

    # ========================================================================
    # 7. Inference speed test
    # ========================================================================
    model.eval()

    # Prepare test sample
    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        test_sample = torch.tensor(
            x_scaler.transform(x_test_external[:1]),
            dtype=torch.float32,
            device=device
        )
    else:
        test_sample = X_test[:1]

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_sample)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Measure
    n_repeats = 100
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t_start = time.perf_counter()
            _ = model(test_sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t_start)

    latency_ms = np.mean(times) * 1000

    # ========================================================================
    # 8. Print final results
    # ========================================================================
    print(f"\nNon-Slack Generators:")
    print(f"  MAE:        {test_metrics['mae_pg_non_slack']:.4f}%")
    print(f"  Violation:  {test_metrics['viol_pg_non_slack']:.4f} p.u.")
    print(f"\nSlack-Only Generators:")
    print(f"  MAE:        {test_metrics['mae_pg_slack']:.4f}%")
    print(f"  Violation:  {test_metrics['viol_pg_slack']:.4f} p.u.")
    print(f"\nBranch:")
    print(f"  Violation:  {test_metrics['viol_branch']:.4f} p.u.")
    print(f"\nCost Gap:     {test_metrics['cost_gap_percent']:.4f}%")
    print(f"\nTraining Time:   {train_time:.2f} s")
    print(f"Inference Time:  {latency_ms:.4f} ms")
    print("\n" + "=" * 70 + "\n")

    return test_metrics


# =====================================================================
# Main Function
# =====================================================================
if __name__ == "__main__":
    # ===================================================================
    # Experiment Configuration
    # ===================================================================

    # --- 1. Case Configuration ---
    CASE_NAME = 'pglib_opf_case118_ieee'
    CASE_SHORT_NAME = 'case118'

    # --- 2. Data Split Mode ---
    SPLIT_MODE = DataSplitMode.VALID_FIXED

    # --- 3. Training & Test Sample Counts ---
    N_TRAIN_USE = 35000
    N_TEST_SAMPLES = 1000

    # --- 4. Training Hyperparameters ---
    N_EPOCHS = 1000
    PATIENCE = 20
    MIN_DELTA = 1e-6
    LEARNING_RATE = 0.001
    LAGRANGIAN_LR = 0.01  # Lagrangian step size rho
    BATCH_SIZE = 64
    HIDDEN_LAYERS = [128, 64]
    SEED = 42

    # --- 5. Path Configuration ---
    ROOT_DIR = PathConfig.ROOT_DIR
    TRAIN_VARIANCE = "v=0.12"
    TEST_VARIANCE = "v=0.25"

    # Column name mapping
    COLUMN_NAMES = {
        'load_prefix': 'pd',
        'gen_prefix': 'pg',
        'lambda': 'lambda',
        'mu_g_min_prefix': 'mu_g_min_',
        'mu_g_max_prefix': 'mu_g_max_',
        'mu_line_pos_prefix': 'mu_line_max_',
        'mu_line_neg_prefix': 'mu_line_min_',
    }

    # ===================================================================
    # Path Generation (based on split mode)
    # ===================================================================

    # Training constraints and data paths
    params_path = os.path.join(ROOT_DIR, "DCOPF Constraints", CASE_SHORT_NAME)
    train_data_path = os.path.join(
        ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TRAIN_VARIANCE})",
        f"{CASE_NAME}_dataset_with_duals.csv"
    )

    # Test data path (based on mode)
    if SPLIT_MODE == DataSplitMode.GENERALIZATION:
        test_data_path = os.path.join(
            ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}({TEST_VARIANCE})",
            f"{CASE_NAME}_dataset_with_duals.csv"
        )
        test_params_path = None
    elif SPLIT_MODE == DataSplitMode.API_TEST:
        test_data_path = os.path.join(
            ROOT_DIR, "DCOPF dataset", f"{CASE_SHORT_NAME}(v=api)",
            f"{CASE_NAME}__api_dataset_with_duals.csv"
        )
        test_params_path = os.path.join(
            ROOT_DIR, "DCOPF Constraints", f"{CASE_SHORT_NAME}(api)"
        )
    else:
        test_data_path = None
        test_params_path = None

    # ===================================================================
    # Device Detection
    # ===================================================================
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    # ===================================================================
    # Run Experiment
    # ===================================================================
    results = lagrangian_dcopf_experiment(
        case_name=CASE_NAME,
        params_path=params_path,
        dataset_path=train_data_path,
        split_mode=SPLIT_MODE,
        n_train_use=N_TRAIN_USE,
        test_data_path=test_data_path,
        test_params_path=test_params_path,
        n_test_samples=N_TEST_SAMPLES,
        n_epochs=N_EPOCHS,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        learning_rate=LEARNING_RATE,
        lagrangian_lr=LAGRANGIAN_LR,
        hidden_layers=HIDDEN_LAYERS,
        batch_size=BATCH_SIZE,
        seed=SEED,
        device=device_name,
        column_names=COLUMN_NAMES
    )