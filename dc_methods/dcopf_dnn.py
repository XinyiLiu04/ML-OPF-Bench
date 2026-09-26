"""Supervised MLP baseline: predict non-slack dispatch from loads, close the balance on the slack."""

from ml_opf_bench.runtime import TrainingState, is_managed

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
from dc_configuration.dcopf_torch_utils import MLP, measure_latency, train_with_early_stopping


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
    mse = nn.MSELoss()

    t0 = time.perf_counter()
    train_with_early_stopping(
        model, optimizer, (X_train, Y_train),
        batch_loss=lambda m, b: mse(m(b[0]), b[1]),
        val_loss=lambda m: mse(m(X_val), Y_val).item(),
        n_epochs=n_epochs, batch_size=batch_size,
        patience=early_stop_patience, min_delta=early_stop_min_delta)
    train_time = time.perf_counter() - t0
    if is_managed():
        return TrainingState(model, params, train_time, dict(x_scaler=x_scaler, y_scaler=y_scaler))

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
    dcopf_config.print_config()
    dnn_experiment(**dcopf_config.get_all_paths(), **dcopf_config.get_all_params())