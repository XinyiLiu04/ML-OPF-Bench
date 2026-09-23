"""Lagrangian dual training (Fioretto et al., AAAI-20): MSE plus multiplier-weighted violation degrees."""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from dc_configuration import dcopf_config
from dc_configuration.dcopf_data_setup import (
    load_parameters_from_csv, load_samples, prepare_data_splits, reconstruct_full_pg,
)
from dc_configuration.dcopf_evaluation_metrics import evaluate_dispatch, print_metrics
from dc_configuration.dcopf_torch_utils import (
    DCTensors, MLP, measure_latency, minmax_inverse, train_with_early_stopping,
)


def mu_experiment(case_name, params_path, data_path,
                  n_train_use, seed, n_epochs, early_stop_patience, early_stop_min_delta,
                  learning_rate, hidden_sizes, batch_size, device, rho):
    torch.manual_seed(seed)
    device = torch.device(device)
    print(f"\nLagrangian dual on {case_name}, device {device}, rho {rho}\n")

    params = load_parameters_from_csv(case_name, params_path)
    pd_bus, pg = load_samples(data_path, params)
    train_idx, val_idx, test_idx = prepare_data_splits(len(pd_bus), n_train_use, seed)
    non_slack = params['general']['non_slack_gen_idx']
    net = DCTensors(params, device)

    x_scaler = MinMaxScaler().fit(pd_bus[train_idx])
    y_scaler = MinMaxScaler().fit(pg[train_idx][:, non_slack])
    decode_pg = minmax_inverse(y_scaler, device)

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    X_train, X_val, X_test = (to_tensor(x_scaler.transform(pd_bus[i])) for i in (train_idx, val_idx, test_idx))
    Y_train, Y_val = (to_tensor(y_scaler.transform(pg[i][:, non_slack])) for i in (train_idx, val_idx))
    Pd_train = to_tensor(pd_bus[train_idx])

    model = MLP(pd_bus.shape[1], len(non_slack), hidden_sizes, sigmoid_output=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()

    multipliers = {'pg': 0.0, 'branch': 0.0}
    last_nu = {}

    def batch_loss(m, batch):
        x, y, pd_b = batch
        pred = m(x)
        pg_full = net.full_pg(decode_pg(pred), pd_b)
        # Violation degrees stay in the graph: the multiplier term must reach the weights,
        # otherwise the method reduces to plain MSE training
        nu = {
            'pg': net.gen_violation(pg_full).mean(),
            'branch': net.line_violation(pg_full, pd_b).sum(dim=1).mean() / max(net.n_constrained, 1),
        }
        last_nu.update({k: v.item() for k, v in nu.items()})
        return mse(pred, y) + sum(multipliers[k] * nu[k] for k in nu)

    def update_multipliers():
        # Per-batch update with the violation degrees of the batch just stepped on
        for k in multipliers:
            multipliers[k] = max(0.0, multipliers[k] + rho * last_nu[k])

    t0 = time.perf_counter()
    train_with_early_stopping(
        model, optimizer, (X_train, Y_train, Pd_train),
        batch_loss=batch_loss,
        val_loss=lambda m: mse(m(X_val), Y_val).item(),  # pure MSE: the multipliers move during training
        n_epochs=n_epochs, batch_size=batch_size,
        patience=early_stop_patience, min_delta=early_stop_min_delta,
        after_step=update_multipliers,
        epoch_log=lambda: f"lambda_pg {multipliers['pg']:.4f}  lambda_branch {multipliers['branch']:.4f}")
    train_time = time.perf_counter() - t0
    print(f"Final multipliers: pg {multipliers['pg']:.6f}, branch {multipliers['branch']:.6f}")

    model.eval()
    with torch.no_grad():
        pg_non_slack = y_scaler.inverse_transform(model(X_test).cpu().numpy().astype(np.float64))
    pg_pred = reconstruct_full_pg(pg_non_slack, pd_bus[test_idx], params)

    metrics = evaluate_dispatch(pg_pred, pg[test_idx], pd_bus[test_idx], params)
    metrics['train_time_s'] = train_time
    metrics['inference_ms'] = measure_latency(lambda: model(X_test[:1]), device)
    metrics['inference_scope'] = 'forward pass only, batch of 1; excludes scaling and slack reconstruction'
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    RHO = 1e-2  # multiplier step size

    dcopf_config.print_config()
    mu_experiment(**dcopf_config.get_all_paths(), **dcopf_config.get_all_params(), rho=RHO)