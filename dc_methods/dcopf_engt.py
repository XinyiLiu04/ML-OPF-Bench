"""Extended DeepOPF-NGT (Huang, Chen, Low, IEEE TPWRS 2024), semi-supervised.

Every epoch runs a supervised pass over a small labeled subset of the training split, then the
unsupervised NGT pass over the whole training split. The constraint weights adapt only from the
unsupervised pass; k_v stays fixed.
"""

from ml_opf_bench.runtime import TrainingState, is_managed, record_epoch

import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from dc_configuration import dcopf_config
from dc_configuration.dcopf_data_setup import load_parameters_from_csv, load_samples, prepare_data_splits
from dc_configuration.dcopf_evaluation_metrics import print_metrics
from dc_configuration.dcopf_torch_utils import DCTensors, MLP, measure_latency
from dcopf_ngt import (
    AdaptiveWeights, evaluate_sigmoid_model, log_epoch, non_slack_from_sigmoid, physics_losses,
    unsupervised_epoch, validation_score,
)


def engt_experiment(case_name, params_path, data_path,
                    n_train_use, seed, n_epochs, learning_rate, hidden_sizes, batch_size, device,
                    n_labeled, k_v, **ignored):
    dcopf_config.print_ignored('ENGT', **ignored)
    torch.manual_seed(seed)
    device = torch.device(device)
    print(f"\nExtended DeepOPF-NGT on {case_name}, device {device}, {n_labeled} labeled, k_v {k_v}\n")

    params = load_parameters_from_csv(case_name, params_path)
    pd_bus, pg = load_samples(data_path, params)
    train_idx, val_idx, test_idx = prepare_data_splits(len(pd_bus), n_train_use, seed)
    non_slack = params['general']['non_slack_gen_idx']
    net = DCTensors(params, device)

    if n_labeled > len(train_idx):
        raise ValueError(f"n_labeled = {n_labeled} exceeds the {len(train_idx)} training samples")
    # Labeled rows are drawn from the training split only, so no test label is ever seen
    labeled_idx = np.random.default_rng(seed).choice(train_idx, size=n_labeled, replace=False)

    x_scaler = MinMaxScaler().fit(pd_bus[train_idx])

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    X_train, X_val, X_test, X_lbl = (to_tensor(x_scaler.transform(pd_bus[i]))
                                     for i in (train_idx, val_idx, test_idx, labeled_idx))
    Pd_train, Pd_val, Pd_lbl = (to_tensor(pd_bus[i]) for i in (train_idx, val_idx, labeled_idx))
    Pg_lbl = to_tensor(pg[labeled_idx][:, non_slack])

    model = MLP(pd_bus.shape[1], len(non_slack), hidden_sizes, sigmoid_output=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = AdaptiveWeights()

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        record_epoch(epoch)
        model.train()
        sup_sum = 0.0
        perm = torch.randperm(n_labeled, device=device)
        for start in range(0, n_labeled, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            pg_ns = non_slack_from_sigmoid(model(X_lbl[idx]), net)
            l_v = ((pg_ns - Pg_lbl[idx]) ** 2).sum(dim=1).mean()
            (k_v * l_v + weights.constraint_loss(physics_losses(pg_ns, Pd_lbl[idx], net))).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            sup_sum += l_v.item() * len(idx)

        raw = unsupervised_epoch(model, optimizer, weights, X_train, Pd_train, batch_size, net)
        weights.end_epoch(epoch, raw)
        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            log_epoch(epoch, n_epochs, raw, weights, validation_score(model, weights, X_val, Pd_val, net))
            print(f"  L_v (labeled) {sup_sum / n_labeled:.3e}")
    train_time = time.perf_counter() - t0
    if is_managed():
        return TrainingState(model, params, train_time, dict(x_scaler=x_scaler))

    metrics = evaluate_sigmoid_model(model, X_test, pd_bus[test_idx], pg[test_idx], params, net)
    metrics['train_time_s'] = train_time
    metrics['inference_ms'] = measure_latency(lambda: model(X_test[:1]), device)
    metrics['inference_scope'] = 'forward pass only, batch of 1; excludes scaling and slack reconstruction'
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    N_LABELED = 5000
    K_V = 100.0  # weight of the supervised dispatch error

    dcopf_config.print_config()
    engt_experiment(**dcopf_config.get_all_paths(), **dcopf_config.get_all_params(),
                    n_labeled=N_LABELED, k_v=K_V)