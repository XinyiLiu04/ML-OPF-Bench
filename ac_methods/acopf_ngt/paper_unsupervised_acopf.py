# -*- coding: utf-8 -*-
"""DeepOPF-NGT, unsupervised ACOPF, Algorithm 1 of Huang, Chen & Low (IEEE TPWRS 2024).

No ground-truth solutions are used. The network predicts voltage at the non-ZIB buses,
the algebraic power flow reconstructs everything else, and the loss is the generation
cost plus weighted constraint violations. 
"""

import numpy as np
import torch
import torch.optim as optim
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
    )
except ImportError as e:
    print(f"Error: Unable to import from ac_configuration/ ({e})")
    sys.exit(1)

from algebraic_power_flow import AlgebraicPowerFlow
from deepopf_ngt_common import (
    LOSS_KEYS,
    SOLVER_ALGEBRAIC,
    DeepOPFNGT,
    LossTerms,
    VoltageDenormaliser,
    evaluate_algebraic,
    print_metrics_block,
    unweighted_total,
    update_coefficients_eq12,
    weighted_total,
)


def train_deepopf_ngt(
        case_name,
        params_path,
        data_path,
        n_train_use=None,
        seed=42,
        n_epochs=100,
        learning_rate=1e-3,
        hidden_sizes=None,
        batch_size=256,
        device='cuda',
        k_obj=1.0,
        k_init=1.0,
        k_max=1000.0,
        theta_max_deg=30.0,
        **kwargs
):
    """Train by Algorithm 1 on a random split and evaluate on the test indices."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"DeepOPF-NGT, Unsupervised ACOPF (Algorithm 1)")
    print(f"{'=' * 70}")
    print(f"Case: {case_name}  |  Device: {device}")
    print(f"k_obj={k_obj}, initial k_i={k_init}, upper bound k_i={k_max}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # 1. Network parameters and the algebraic power flow engine
    # ------------------------------------------------------------------
    params = load_parameters_from_csv(case_name, params_path)
    pf_engine = AlgebraicPowerFlow(params, device)
    denorm = VoltageDenormaliser(params, device, theta_max_deg=theta_max_deg)

    print(f"\n[Reduction] {pf_engine.summary()}")

    # ------------------------------------------------------------------
    # 2. Dataset. Only the inputs are used for training; the labels are read
    #    at evaluation time only, which is what makes this unsupervised.
    # ------------------------------------------------------------------
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True,
                                  n_train_use=n_train_use, seed=seed)

    n_loads = params['general']['n_loads']
    if cost_baseline:
        print(f"  Reference cost: {cost_baseline:.2f} $/h")

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = prepare_data_splits(
        x_data_scaled, y_data_scaled, n_train_use=n_train_use, seed=seed)

    X_train = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32, device=device)
    Pd_train = torch.tensor(raw_data['x'][train_idx][:, :n_loads],
                            dtype=torch.float32, device=device)
    Qd_train = torch.tensor(raw_data['x'][train_idx][:, n_loads:],
                            dtype=torch.float32, device=device)

    X_val = torch.tensor(x_data_scaled[val_idx], dtype=torch.float32, device=device)
    Pd_val = torch.tensor(raw_data['x'][val_idx][:, :n_loads],
                          dtype=torch.float32, device=device)
    Qd_val = torch.tensor(raw_data['x'][val_idx][:, n_loads:],
                          dtype=torch.float32, device=device)

    X_test = torch.tensor(x_data_scaled[test_idx], dtype=torch.float32)

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model = DeepOPFNGT(2 * n_loads, denorm.output_dim, hidden_sizes).to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    loss_terms = LossTerms(params, pf_engine, device, theta_max_deg=theta_max_deg)

    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"Input dim: {2 * n_loads} (pd + qd)")
    print(f"Output dim: {denorm.output_dim} "
          f"(v and theta at {denorm.n_nonzib} non-ZIB buses)")
    print(f"Angle window: +/-{theta_max_deg} degrees")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training params: epochs={n_epochs}, lr={learning_rate}, "
          f"batch_size={batch_size}")
    print(f"{'=' * 70}")

    coeffs = {'k_obj': k_obj, 'k_g': k_init, 'k_Sl': k_init,
              'k_theta': k_init, 'k_z': k_init, 'k_d': k_init}
    k_upper = {'k_g': k_max, 'k_Sl': k_max, 'k_theta': k_max,
               'k_z': k_max, 'k_d': k_max}

    # ------------------------------------------------------------------
    # 5. Training (Algorithm 1)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Training Progress")
    print(f"{'=' * 70}")

    n_train = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_sums = {k: 0.0 for k in LOSS_KEYS}
        epoch_total = 0.0
        perm = torch.randperm(n_train, device=device)

        for b in range(n_batches):
            idx = perm[b * batch_size:min((b + 1) * batch_size, n_train)]
            optimiser.zero_grad()

            v_alpha, theta_alpha = denorm(model(X_train[idx]))
            results = pf_engine(v_alpha, theta_alpha, Pd_train[idx], Qd_train[idx])
            loss_dict = loss_terms(results)
            loss = weighted_total(loss_dict, coeffs)

            loss.backward()
            optimiser.step()

            scalars = {k: v.item() for k, v in loss_dict.items()}

            # Eq. (12) is applied per mini-batch, but only once a first epoch has
            # established loss magnitudes; the initial weights cover epoch one
            if epoch > 1:
                update_coefficients_eq12(coeffs, scalars, k_upper)

            bs = len(idx)
            epoch_total += loss.item() * bs
            for k in LOSS_KEYS:
                epoch_sums[k] += scalars[k] * bs

        means = {k: epoch_sums[k] / n_train for k in LOSS_KEYS}

        # Diagnostic only, not used for model selection. Unit weights are used so
        # the score means the same thing at every epoch, unlike the training
        # objective whose weights move; a score that climbs late in training means
        # the adaptive weights have pushed the solution somewhere worse.
        model.eval()
        with torch.no_grad():
            v_a, th_a = denorm(model(X_val))
            val_dict = loss_terms(pf_engine(v_a, th_a, Pd_val, Qd_val))
        val_score = unweighted_total(val_dict)

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            gap = ((means['L_obj'] - cost_baseline) / cost_baseline * 100
                   if cost_baseline else float('nan'))
            print(f"Epoch {epoch:4d}/{n_epochs} | L={epoch_total / n_train:.4f} | "
                  f"L_obj={means['L_obj']:.1f} ({gap:+.2f}%) | "
                  f"L_g={means['L_g']:.4f} L_Sl={means['L_Sl']:.4f} "
                  f"L_th={means['L_theta']:.4f} L_z={means['L_z']:.4f} "
                  f"L_d={means['L_d']:.4f} | val={val_score:.4f}")
            print(f"  Weights: k_g={coeffs['k_g']:.2f} k_Sl={coeffs['k_Sl']:.2f} "
                  f"k_th={coeffs['k_theta']:.2f} k_z={coeffs['k_z']:.2f} "
                  f"k_d={coeffs['k_d']:.2f}")

    train_time = time.perf_counter() - t0
    print(f"Training completed in {train_time:.2f} seconds, "
          f"returning the final epoch's model")

    # ------------------------------------------------------------------
    # 6. Inference latency (forward pass only)
    # ------------------------------------------------------------------
    sample = X_test[:1].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(sample)
        times = []
        for _ in range(100):
            t_start = time.perf_counter()
            model(sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t_start)
    latency_ms = float(np.mean(times)) * 1000

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")
    print(f"  Reconstructing {len(X_test)} samples algebraically...")

    metrics = evaluate_algebraic(model, denorm, pf_engine, X_test, test_idx,
                                 raw_data, params, device)
    metrics['solver'] = SOLVER_ALGEBRAIC

    print_metrics_block(metrics, SOLVER_ALGEBRAIC, case_name, train_time, latency_ms)

    return model, params, pf_engine, scalers, coeffs, metrics


if __name__ == '__main__':
    K_OBJ = 1.0
    K_INIT = 1.0
    K_MAX = 1000.0
    THETA_MAX_DEG = 30.0

    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)
    print("Note: this method trains for a fixed epoch budget and returns the final")
    print("      model, so EARLY_STOP_PATIENCE and EARLY_STOP_MIN_DELTA are ignored.")
    print(f"\n[DeepOPF-NGT Configuration]")
    print(f"  k_obj={K_OBJ}, initial k_i={K_INIT}, upper bound={K_MAX}")
    print(f"  Angle window: +/-{THETA_MAX_DEG} degrees")
    print("=" * 70)

    results = train_deepopf_ngt(
        **acopf_config.get_all_paths(),
        **acopf_config.get_all_params(),
        k_obj=K_OBJ,
        k_init=K_INIT,
        k_max=K_MAX,
        theta_max_deg=THETA_MAX_DEG,
    )

    print("\nExperiment completed successfully!")