# -*- coding: utf-8 -*-
"""Unsupervised ACOPF, DeepOPF-NGT with reshaped losses and EMA-scheduled weights.

Same method as paper_unsupervised_acopf.py, with three changes aimed at making training
behave on larger cases:

  - the cost term is evaluated on clamped Pg plus a marginal-cost term outside the
    bounds, so the optimizer cannot chase a low cost by driving Pg out of range
  - every loss is normalized by its epoch-one magnitude and the constraint weights are
    EMA smoothed with satisfied and violated regimes, instead of the plain Eq. (12) ratio
  - gradients are norm clipped

Predictions are also verified with a real power flow, so the convergence rate reported
here is a Newton-Raphson rate and is comparable with the other benchmark methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

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

from algebraic_power_flow import AlgebraicPowerFlow
from deepopf_ngt_common import (
    LOSS_KEYS,
    SOLVER_NEWTON,
    AdaptiveWeightScheduler,
    DeepOPFNGT,
    LossTermsClampedCost,
    VoltageDenormaliser,
    evaluate_with_power_flow,
    print_metrics_block,
    unweighted_total,
    weighted_total,
)

DEFAULT_INITIAL = {'k_g': 100.0, 'k_Sl': 10.0, 'k_theta': 100.0,
                   'k_z': 10.0, 'k_d': 10.0}
# The generator and angle terms keep a high ceiling because they bound physics the
# reduction cannot enforce; the rest are capped conservatively so they cannot swamp cost
DEFAULT_UPPER = {'k_g': 10000.0, 'k_Sl': 200.0, 'k_theta': 10000.0,
                 'k_z': 200.0, 'k_d': 200.0}
DEFAULT_LOWER = {'k_g': 1.0, 'k_Sl': 1.0, 'k_theta': 1.0,
                 'k_z': 1.0, 'k_d': 1.0}


def train_deepopf_ngt_smoothed(
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
        k_obj=0.1,
        grad_clip=1.0,
        theta_max_deg=30.0,
        **kwargs
):
    """Train on a random split and evaluate on the test indices."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"Unsupervised ACOPF, EMA-scheduled weights")
    print(f"{'=' * 70}")
    print(f"Case: {case_name}  |  Device: {device}")
    print(f"k_obj={k_obj} (fixed), gradient clip={grad_clip}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # 1. Network parameters, algebraic engine, and the case for verification
    # ------------------------------------------------------------------
    params = load_parameters_from_csv(case_name, params_path)
    case_data = load_case_from_csv(case_name, params_path)
    pf_engine = AlgebraicPowerFlow(params, device)
    denorm = VoltageDenormaliser(params, device, theta_max_deg=theta_max_deg)

    print(f"\n[Reduction] {pf_engine.summary()}")

    # ------------------------------------------------------------------
    # 2. Dataset. Labels are read only at evaluation time.
    # ------------------------------------------------------------------
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_loads = params['general']['n_loads']
    if cost_baseline:
        print(f"  Reference cost: {cost_baseline:.2f} $/h "
              f"(also the normalization scale for L_obj)")

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = prepare_data_splits(
        x_data_scaled, y_data_scaled, n_train_use=n_train_use, seed=seed)

    def _dev(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    X_train = _dev(x_data_scaled[train_idx])
    Pd_train = _dev(raw_data['x'][train_idx][:, :n_loads])
    Qd_train = _dev(raw_data['x'][train_idx][:, n_loads:])

    X_val = _dev(x_data_scaled[val_idx])
    Pd_val = _dev(raw_data['x'][val_idx][:, :n_loads])
    Qd_val = _dev(raw_data['x'][val_idx][:, n_loads:])

    X_test = torch.tensor(x_data_scaled[test_idx], dtype=torch.float32)

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model = DeepOPFNGT(2 * n_loads, denorm.output_dim, hidden_sizes).to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    loss_terms = LossTermsClampedCost(params, pf_engine, device,
                                      theta_max_deg=theta_max_deg)
    scheduler = AdaptiveWeightScheduler(
        k_obj, DEFAULT_INITIAL, DEFAULT_UPPER, DEFAULT_LOWER)

    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"Input dim: {2 * n_loads} (pd + qd)")
    print(f"Output dim: {denorm.output_dim} "
          f"(v and theta at {denorm.n_nonzib} non-ZIB buses)")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial weights: {scheduler.describe()}")
    print(f"Training params: epochs={n_epochs}, lr={learning_rate}, "
          f"batch_size={batch_size}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # 5. Training. Epoch one runs with the initial weights and fixes the
    #    normalization scales; the scheduler takes over from epoch two.
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

            # The weights are applied to normalized losses, so they stay comparable
            # across terms whose physical units differ by orders of magnitude
            loss = weighted_total(scheduler.normalise(loss_dict), scheduler.coeffs)

            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimiser.step()

            bs = len(idx)
            epoch_total += loss.item() * bs
            for k in LOSS_KEYS:
                epoch_sums[k] += loss_dict[k].item() * bs

        means = {k: epoch_sums[k] / n_train for k in LOSS_KEYS}

        if epoch == 1:
            scheduler.set_references(means, cost_baseline)
            print(f"  [Normalization] references fixed after epoch 1: "
                  + ", ".join(f"{k}={scheduler.refs[k]:.4g}" for k in LOSS_KEYS))
        else:
            scheduler.update(means)

        # Diagnostic only, no model selection. Unit weights on the raw losses keep
        # the score comparable across epochs while the weights move.
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
            print(f"  Weights: {scheduler.describe()}{scheduler.describe_ema()}")

    train_time = time.perf_counter() - t0
    print(f"\nTraining completed in {train_time:.2f} seconds, "
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
    # 7. Evaluation, verified with a real power flow
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation (power flow verification)")
    print(f"{'=' * 70}")

    metrics = evaluate_with_power_flow(
        model, denorm, pf_engine, X_test, test_idx, raw_data, params,
        case_data, device, verbose=True)
    metrics['solver'] = SOLVER_NEWTON

    print_metrics_block(metrics, SOLVER_NEWTON, case_name, train_time, latency_ms)

    return model, params, pf_engine, scalers, scheduler.coeffs, metrics


if __name__ == '__main__':
    K_OBJ = 0.1
    GRAD_CLIP = 1.0
    THETA_MAX_DEG = 30.0

    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)
    print("Note: this method trains for a fixed epoch budget and returns the final")
    print("      model, so EARLY_STOP_PATIENCE and EARLY_STOP_MIN_DELTA are ignored.")
    print(f"\n[EMA-Scheduled Configuration]")
    print(f"  k_obj={K_OBJ} (fixed), gradient clip={GRAD_CLIP}")
    print(f"  Initial weights: {DEFAULT_INITIAL}")
    print(f"  Upper bounds:    {DEFAULT_UPPER}")
    print("=" * 70)

    results = train_deepopf_ngt_smoothed(
        **acopf_config.get_all_paths(),
        **acopf_config.get_all_params(),
        k_obj=K_OBJ,
        grad_clip=GRAD_CLIP,
        theta_max_deg=THETA_MAX_DEG,
    )

    print("\nExperiment completed successfully!")