"""DeepOPF (Pan et al.): supervised MLP with a line-flow penalty, then a QP projection onto the feasible set."""

import time

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

from dc_configuration import dcopf_config
from dc_configuration.dcopf_data_setup import (
    load_parameters_from_csv, load_samples, prepare_data_splits, reconstruct_full_pg,
)
from dc_configuration.dcopf_evaluation_metrics import evaluate_dispatch, print_metrics
from dc_configuration.dcopf_torch_utils import DCTensors, MLP, measure_latency, train_with_early_stopping


def line_penalty(pg_full, pd_bus, net):
    """Mean over branches of p(x) = x^2 - 1, with x the flow normalized by its rating (paper Eq. 8, 12).

    As written in the paper, p is negative inside the limits, so the term also rewards lightly
    loaded branches rather than only punishing violations; the paper's prose describes it as a
    penalty on infeasible solutions, but the equation has no max(., 0) and is kept as published.
    """
    x = net.flows(pg_full, pd_bus) / net.rate
    return ((x ** 2 - 1.0).sum(dim=1) / max(net.n_constrained, 1)).mean()


def project_qp(pg_pred, pd_bus, params):
    """Per sample, min 0.5 ||u - pg_pred||^2 over generator limits, power balance and line limits.

    Returns the projections and the number of samples where SLSQP did not converge; those keep
    the clipped prediction, so a failure shows up in the violation metrics instead of vanishing.
    """
    c = params['constraints']
    mask = c['constrained_branches']
    ptdf, rate = c['ptdf'][mask], c['rate_a'][mask]
    flow_per_gen = ptdf @ c['gen_bus_map']  # (n_constrained, n_gens)
    bounds = list(zip(c['pg_min'], c['pg_max']))
    ones = np.ones(pg_pred.shape[1])

    out = np.empty_like(pg_pred)
    n_failed = 0
    for i, (target, pd_i) in enumerate(zip(pg_pred, pd_bus)):
        offset, total = ptdf @ pd_i, pd_i.sum()
        constraints = [
            {'type': 'eq', 'fun': lambda u, total=total: ones @ u - total, 'jac': lambda u: ones},
            {'type': 'ineq', 'fun': lambda u, b=offset: rate - (flow_per_gen @ u - b), 'jac': lambda u: -flow_per_gen},
            {'type': 'ineq', 'fun': lambda u, b=offset: rate + (flow_per_gen @ u - b), 'jac': lambda u: flow_per_gen},
        ]
        start = np.clip(target, c['pg_min'], c['pg_max'])
        res = minimize(lambda u, t=target: 0.5 * np.dot(u - t, u - t), start,
                       jac=lambda u, t=target: u - t, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'ftol': 1e-9, 'maxiter': 1000})
        if res.success:
            out[i] = res.x
        else:
            out[i] = start
            n_failed += 1
    return out, n_failed


def cp_qp_experiment(case_name, params_path, data_path,
                     n_train_use, seed, n_epochs, early_stop_patience, early_stop_min_delta,
                     learning_rate, hidden_sizes, batch_size, device, penalty_weight):
    torch.manual_seed(seed)
    device = torch.device(device)
    print(f"\nDeepOPF + QP projection on {case_name}, device {device}, penalty weight {penalty_weight}\n")

    params = load_parameters_from_csv(case_name, params_path)
    pd_bus, pg = load_samples(data_path, params)
    train_idx, val_idx, test_idx = prepare_data_splits(len(pd_bus), n_train_use, seed)
    non_slack = params['general']['non_slack_gen_idx']
    net = DCTensors(params, device)

    x_scaler = MinMaxScaler().fit(pd_bus[train_idx])
    y_scaler = MinMaxScaler().fit(pg[train_idx][:, non_slack])
    y_min = torch.tensor(y_scaler.data_min_, dtype=torch.float32, device=device)
    y_range = torch.tensor(y_scaler.data_range_, dtype=torch.float32, device=device)

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    X_train, X_val, X_test = (to_tensor(x_scaler.transform(pd_bus[i])) for i in (train_idx, val_idx, test_idx))
    Y_train, Y_val = (to_tensor(y_scaler.transform(pg[i][:, non_slack])) for i in (train_idx, val_idx))
    Pd_train, Pd_val = to_tensor(pd_bus[train_idx]), to_tensor(pd_bus[val_idx])

    model = MLP(pd_bus.shape[1], len(non_slack), hidden_sizes, sigmoid_output=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    mse = nn.MSELoss()

    def loss(m, x, y, pd_b):
        pred = m(x)
        return mse(pred, y) + penalty_weight * line_penalty(net.full_pg(pred * y_range + y_min, pd_b), pd_b, net)

    t0 = time.perf_counter()
    train_with_early_stopping(
        model, optimizer, (X_train, Y_train, Pd_train),
        batch_loss=lambda m, b: loss(m, *b),
        val_loss=lambda m: loss(m, X_val, Y_val, Pd_val).item(),
        n_epochs=n_epochs, batch_size=batch_size,
        patience=early_stop_patience, min_delta=early_stop_min_delta)
    train_time = time.perf_counter() - t0

    def predict_full(X, pd_rows):
        with torch.no_grad():
            pg_ns = y_scaler.inverse_transform(model(X).cpu().numpy().astype(np.float64))
        return reconstruct_full_pg(pg_ns, pd_rows, params)

    model.eval()
    pd_test = pd_bus[test_idx]
    pg_nn = predict_full(X_test, pd_test)
    t_qp = time.perf_counter()
    pg_pp, n_failed = project_qp(pg_nn, pd_test, params)
    print(f"QP projection: {time.perf_counter() - t_qp:.2f} s for {len(test_idx)} samples, "
          f"{n_failed} did not converge")

    nn_ms = measure_latency(lambda: model(X_test[:1]), device)
    qp_ms = measure_latency(lambda: project_qp(pg_nn[:1], pd_test[:1], params), device, n_warmup=2, n_repeats=20)

    nn_metrics = evaluate_dispatch(pg_nn, pg[test_idx], pd_test, params)
    nn_metrics.update(train_time_s=train_time, inference_ms=nn_ms,
                      inference_scope='forward pass only, batch of 1')
    pp_metrics = evaluate_dispatch(pg_pp, pg[test_idx], pd_test, params)
    pp_metrics.update(train_time_s=train_time, inference_ms=nn_ms + qp_ms,
                      inference_scope='forward pass plus QP projection, batch of 1',
                      inference_ms_network=nn_ms, inference_ms_qp=qp_ms,
                      qp_failure_rate=n_failed / len(test_idx))
    print_metrics(nn_metrics, "Test Set Results - network only")
    print_metrics(pp_metrics, "Test Set Results - with QP projection")
    return {'nn_only': nn_metrics, 'post_processed': pp_metrics}


if __name__ == "__main__":
    PENALTY_WEIGHT = 1e-5

    dcopf_config.print_config()
    cp_qp_experiment(**dcopf_config.get_all_paths(), **dcopf_config.get_all_params(),
                     penalty_weight=PENALTY_WEIGHT)