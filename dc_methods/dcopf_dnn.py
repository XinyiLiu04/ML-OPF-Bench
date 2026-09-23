"""Supervised MLP baseline: predict non-slack dispatch from loads, close the balance on the slack."""

import copy
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


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        layers, prev = [], input_size
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def measure_latency(model, sample, device, n_warmup=10, n_repeats=100):
    """Mean forward-pass latency in ms for a single sample."""
    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample)
        dcopf_config.synchronize(device)
        times = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            model(sample)
            dcopf_config.synchronize(device)
            times.append(time.perf_counter() - start)
    return 1000.0 * float(np.mean(times))


def dnn_experiment(case_name, params_path, data_path,
                   n_train_use, seed, n_epochs, early_stop_patience, early_stop_min_delta,
                   learning_rate, hidden_sizes, batch_size, device):
    torch.manual_seed(seed)
    device = torch.device(device)
    print(f"\nDNN on {case_name}, device {device}\n")

    params = load_parameters_from_csv(case_name, params_path)
    pd_bus, pg = load_samples(data_path, params)
    train_idx, val_idx, test_idx = prepare_data_splits(len(pd_bus), n_train_use, seed)
    non_slack = params['general']['non_slack_gen_idx']

    x_scaler = MinMaxScaler().fit(pd_bus[train_idx])
    y_scaler = MinMaxScaler().fit(pg[train_idx][:, non_slack])

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    X_train, X_val, X_test = (to_tensor(x_scaler.transform(pd_bus[i])) for i in (train_idx, val_idx, test_idx))
    Y_train, Y_val = (to_tensor(y_scaler.transform(pg[i][:, non_slack])) for i in (train_idx, val_idx))

    model = MLP(pd_bus.shape[1], len(non_slack), hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    n_train = len(X_train)

    t0 = time.perf_counter()
    best_val_loss, best_epoch, best_state, patience_counter = float('inf'), 0, None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            batch = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train[batch]), Y_train[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), Y_val).item()

        if val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}/{n_epochs}  train {epoch_loss / n_train:.3e}  "
                  f"val {val_loss:.3e}  patience {patience_counter}/{early_stop_patience}")
        if patience_counter >= early_stop_patience:
            print(f"Early stop at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Validation loss never improved (non-finite loss?); no checkpoint to restore")
    model.load_state_dict(best_state)
    print(f"Restored best checkpoint: epoch {best_epoch}, val loss {best_val_loss:.3e}")
    train_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        pg_non_slack = y_scaler.inverse_transform(model(X_test).cpu().numpy().astype(np.float64))
    pg_pred = reconstruct_full_pg(pg_non_slack, pd_bus[test_idx], params)

    metrics = evaluate_dispatch(pg_pred, pg[test_idx], pd_bus[test_idx], params)
    metrics['train_time_s'] = train_time
    metrics['inference_ms'] = measure_latency(model, X_test[:1], device)
    metrics['inference_scope'] = 'forward pass only, batch of 1; excludes scaling and slack reconstruction'
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    dcopf_config.print_config()
    dnn_experiment(**dcopf_config.get_all_paths(), **dcopf_config.get_all_params())