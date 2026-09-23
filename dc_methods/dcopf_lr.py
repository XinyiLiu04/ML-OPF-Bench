"""Linear regression baseline: one least-squares map from bus loads to non-slack dispatch."""

import time

import numpy as np
from sklearn.linear_model import LinearRegression

from dc_configuration import dcopf_config
from dc_configuration.dcopf_data_setup import (
    load_parameters_from_csv, load_samples, prepare_data_splits, reconstruct_full_pg,
)
from dc_configuration.dcopf_evaluation_metrics import evaluate_dispatch, print_metrics


def lr_experiment(case_name, params_path, data_path, n_train_use, seed, **ignored):
    dcopf_config.print_ignored('LR', **ignored)
    print(f"\nLinear regression on {case_name}\n")

    params = load_parameters_from_csv(case_name, params_path)
    pd_bus, pg = load_samples(data_path, params)
    train_idx, _, test_idx = prepare_data_splits(len(pd_bus), n_train_use, seed)
    non_slack = params['general']['non_slack_gen_idx']

    # A multi-output fit solves the same per-column least-squares problems as one model per
    # generator; inputs stay unscaled since ordinary least squares is scale-equivariant
    model = LinearRegression()
    t0 = time.perf_counter()
    model.fit(pd_bus[train_idx], pg[train_idx][:, non_slack])
    train_time = time.perf_counter() - t0

    pg_pred = reconstruct_full_pg(model.predict(pd_bus[test_idx]), pd_bus[test_idx], params)
    metrics = evaluate_dispatch(pg_pred, pg[test_idx], pd_bus[test_idx], params)

    sample = pd_bus[test_idx[:1]]
    times = []
    for _ in range(100):
        start = time.perf_counter()
        model.predict(sample)
        times.append(time.perf_counter() - start)
    metrics['train_time_s'] = train_time
    metrics['inference_ms'] = 1000.0 * float(np.mean(times))
    metrics['inference_scope'] = 'predict only, batch of 1, cpu; excludes slack reconstruction'
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    dcopf_config.print_config()
    lr_experiment(**dcopf_config.get_all_paths(), **dcopf_config.get_all_params())