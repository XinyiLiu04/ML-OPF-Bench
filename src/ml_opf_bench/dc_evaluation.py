"""Evaluate a fixed DC model against each scenario's own constraints."""

import copy
import time

import numpy as np
import torch

from .registry import load_method


def evaluate_dc(spec, state, paths, indices=None):
    module, _, _ = load_method("dc", spec.method)
    from dc_configuration.dcopf_data_setup import load_parameters_from_csv, load_samples, reconstruct_full_pg
    from dc_configuration.dcopf_evaluation_metrics import evaluate_dispatch
    from dc_configuration.dcopf_config import synchronize

    params = load_parameters_from_csv(paths.case_name, str(paths.params_path))
    pd_all, pg_all = load_samples(str(paths.data_path), params)
    for key in ("bus_ids", "gen_ids", "branch_ids", "non_slack_gen_idx", "slack_gen_idx"):
        np.testing.assert_array_equal(params["general"][key], state.params["general"][key])
    idx = np.arange(len(pd_all)) if indices is None else indices
    if spec.eval_limit is not None:
        idx = idx[:spec.eval_limit]
    pd_test, pg_test = pd_all[idx], pg_all[idx]
    model, artifacts = state.model, state.artifacts
    device = torch.device(spec.device)
    if spec.method != "LR":
        model.eval()
    if spec.method == "GNN":
        model = copy.deepcopy(model)
        edge_index, _ = module.load_graph(paths.case_name, str(paths.params_path), params, "susceptance", 0.01)
        model.static.copy_(torch.as_tensor(module.static_node_features(params, edge_index), device=device, dtype=torch.float32))
    as_failures = 0

    def prediction(loads):
        nonlocal as_failures
        if spec.method == "LR":
            pred_ns = model.predict(loads)
        else:
            X = torch.as_tensor(artifacts["x_scaler"].transform(loads), dtype=torch.float32, device=device)
            with torch.no_grad():
                out = model.pg_net(X) if spec.method == "KKT" else model(X)
                values = out.cpu().numpy().astype(np.float64)
            if spec.method in ("NGT", "E-NGT"):
                c = state.params["constraints"]
                ns = state.params["general"]["non_slack_gen_idx"]
                pred_ns = c["pg_min"][ns] + values * (c["pg_max"][ns] - c["pg_min"][ns])
            elif spec.method == "AS":
                recovery = module.ActiveSetRecovery(params)
                topk = np.argsort(-values, axis=1)[:, :artifacts["top_k"]]
                result, failures = module.recover_topk(topk, artifacts["vocab"], loads, recovery, params)
                as_failures += failures
                return result
            else:
                pred_ns = artifacts["y_scaler"].inverse_transform(values)
        return reconstruct_full_pg(pred_ns, loads, params)

    parts = [prediction(pd_test[i:i + 512]) for i in range(0, len(idx), 512)]
    raw_pg = np.concatenate(parts)
    variants = {spec.method: raw_pg}
    failure_counts = {"AS": as_failures}
    if spec.method in ("CP", "QP"):
        repaired, failed = module.project_qp(raw_pg, pd_test, params)
        variants = {"CP": raw_pg, "QP": repaired}
        failure_counts["QP"] = failed
    records, arrays = {}, {"indices": idx, "pd": pd_test, "pg_true": pg_test}
    for name, predicted in variants.items():
        def complete_predict(loads):
            result = prediction(loads)
            return module.project_qp(result, loads, params)[0] if name == "QP" else result
        for _ in range(2):
            complete_predict(pd_test[:1])
        elapsed = []
        timing_indices = idx[:20]
        for position in range(len(timing_indices)):
            synchronize(device)
            start = time.perf_counter()
            complete_predict(pd_test[position:position + 1])
            synchronize(device)
            elapsed.append(time.perf_counter() - start)
        metrics = evaluate_dispatch(predicted, pg_test, pd_test, params)
        metrics.update(train_time_s=state.train_time_s, inference_ms=float(np.mean(elapsed) * 1000),
                       inference_sample_indices=timing_indices.tolist(),
                       inference_scope="load preprocessing, model, reconstruction, and method postprocessing; batch=1",
                       n_samples=len(idx), recovery_failed_samples=failure_counts.get(name, 0))
        records[name] = metrics
        arrays[f"{name}_pg_pred"] = predicted
    return records, arrays
