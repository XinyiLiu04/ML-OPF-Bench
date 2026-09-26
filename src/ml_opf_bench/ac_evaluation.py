"""Evaluate frozen AC models with scenario-specific power-flow constraints."""

import time

import numpy as np
import torch

from .registry import load_method


def evaluate_ac(spec, state, paths, indices=None):
    module, _, _ = load_method("ac", spec.method)
    from ac_configuration.acopf_data_setup import load_parameters_from_csv, load_and_scale_acopf_data
    from ac_configuration.acopf_evaluation_metrics import evaluate_acopf_predictions
    from ac_configuration.acopf_pypower import load_case_from_csv, solve_pf_setpoints

    params = load_parameters_from_csv(paths.case_name, str(paths.params_path))
    artifacts = state.artifacts
    scalers = artifacts["scalers"]
    X, _, _, raw, _ = load_and_scale_acopf_data(str(paths.data_path), params,
                                              fit_scalers=False, scalers=scalers)
    for key in ("bus_ids", "gen_bus_ids", "load_bus_ids", "non_slack_gen_idx"):
        np.testing.assert_array_equal(params["general"][key], state.params["general"][key])
    idx = np.arange(len(X)) if indices is None else np.asarray(indices)
    if spec.eval_limit is not None:
        idx = idx[:spec.eval_limit]
    general = params["general"]
    n_bus, n_gen, n_load = (general[k] for k in ("n_buses", "n_gen", "n_loads"))
    ns = general["non_slack_gen_idx"]
    gen_bus = np.array([general["bus_id_to_idx"][int(b)] for b in general["gen_bus_ids"]])
    device = torch.device(spec.device)
    model = state.model
    if spec.method not in ("LR", "RL"):
        model.eval()
    case = load_case_from_csv(paths.case_name, str(paths.params_path))
    module.GLOBAL_CASE_DATA = case
    valid_input = np.ones(len(X), dtype=bool)
    if spec.method == "GNN":
        import pandas as pd
        from gnn_utils import build_adjacency_edge_weight, collate_graph_batch
        from generate_subopt_state import load_dcopf_constraints, solve_dcopf, run_powerflow_with_dcopf
        from .gnn_features import load_gnn_features
        subopt, valid_input = load_gnn_features(paths.case_name, paths.params_path, paths.data_path,
                                               params, raw["x"], idx, spec.workers)
        if len(subopt) != len(X):
            raise ValueError("GNN features and load samples have different row counts")
        graph = build_adjacency_edge_weight(params, artifacts["graph_kernel"], artifacts["graph_scale_k"])
        dc_folder = paths.params_path.parents[2] / "dc_dataset" / "dcopf_constraints" / paths.params_path.name
        dc_params = load_dcopf_constraints(paths.case_name, dc_folder)
        dc_bus_ids = pd.read_csv(dc_folder / f"{paths.case_name}_bus_ids.csv")["bus_id"].to_numpy()
        np.testing.assert_array_equal(dc_bus_ids, general["bus_ids"])
    if spec.method in ("NGT", "E-NGT"):
        from algebraic_power_flow import AlgebraicPowerFlow
        pf_engine = AlgebraicPowerFlow(params, device)

    def predict(rows, subopt_override=None):
        if spec.method == "LR":
            return model.predict(scalers["x"].transform(raw["x"][rows]))
        if spec.method == "RL":
            actions, _ = model.predict(scalers["x"].transform(raw["x"][rows]), deterministic=True)
            pairs = [module.action_to_setpoints(a, artifacts["bounds"]) for a in actions]
            return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
        values = raw["x"][rows] if spec.method == "KKT" else scalers["x"].transform(raw["x"][rows])
        inputs = torch.as_tensor(values, device=device, dtype=torch.float32)
        with torch.no_grad():
            if spec.method == "KKT":
                pg, vm, *_ = model.predict_for_evaluation(inputs)
                return pg.cpu().numpy(), vm.cpu().numpy()
            if spec.method == "MU":
                vm_s, _, pg_s, _ = model(inputs)
                return (scalers["pg"].inverse_transform(pg_s.cpu().numpy()),
                        scalers["vm_all"].inverse_transform(vm_s.cpu().numpy())[:, gen_bus])
            if spec.method in ("NGT", "E-NGT"):
                voltage, angle = artifacts["denorm"](model(inputs))
                loads = torch.as_tensor(raw["x"][rows], device=device, dtype=torch.float32)
                result = pf_engine(voltage, angle, loads[:, :n_load], loads[:, n_load:])
                return result["Pg"].cpu().numpy()[:, ns], result["v_all"].cpu().numpy()[:, gen_bus]
            if spec.method == "GNN":
                subopt_input = subopt[rows] if subopt_override is None else subopt_override
                features = scalers["subopt_x"].transform(subopt_input)
                features = torch.as_tensor(features, device=device, dtype=torch.float32)
                nf, edges, weights, batch_size = collate_graph_batch(features, *graph, n_bus, device)
                output = model(nf, edges, weights, batch_size=batch_size).cpu().numpy()
            elif spec.method in ("CP", "FR"):
                output = model.net(inputs).cpu().numpy()
            else:
                output = model(inputs).cpu().numpy()
        if spec.method == "AS":
            topk = np.argsort(-output, axis=1)[:, :artifacts["top_k"]]
            return module.recover_batch_topk(topk, artifacts["label_to_as"], raw["x"][rows, :n_load],
                                             params, artifacts["meta"])
        if spec.method == "QC":
            return module.inverse_parametrize(output, state.params)
        pg = scalers["pg"].inverse_transform(output[:, :len(ns)])
        vm = (subopt_input[:, :n_bus][:, gen_bus] if spec.method == "GNN" and not artifacts["predict_vm"]
              else scalers["vm"].inverse_transform(output[:, len(ns):]))
        return pg, vm

    def solve(row, pg, vm, repair=False):
        pd, qd = raw["x"][row, :n_load], raw["x"][row, n_load:]
        if spec.method == "QC":
            return module.solve_pf_with_qg_correction(pd, qd, pg, vm, params)[0], False, True
        result = solve_pf_setpoints(pd, qd, pg, vm, params, case)
        if repair:
            return module.deepopf_post_process(result, pd, qd, pg, vm, params)
        return result, False, True

    pg = np.full((len(idx), len(ns)), np.nan)
    vm = np.full((len(idx), n_gen), np.nan)
    valid_positions = np.flatnonzero(valid_input[idx])
    for start in range(0, len(valid_positions), 512):
        positions = valid_positions[start:start + 512]
        pg[positions], vm[positions] = predict(idx[positions])
    records = {}
    arrays = {"indices": idx, "input_valid": valid_input[idx], "pg_setpoints": pg, "vm_setpoints": vm,
              **{f"{key}_true": value[idx] for key, value in raw.items()}}
    variants = ("CP", "FR") if spec.method in ("CP", "FR") else (spec.method,)
    for name in variants:
        solutions, converged = [], []
        attempts = failures = 0
        vm_all = np.full((len(idx), n_bus), np.nan)
        pg_all = np.full((len(idx), n_gen), np.nan)
        qg_all = np.full_like(pg_all, np.nan)
        for pos, row in enumerate(idx):
            if valid_input[row]:
                result, attempted, succeeded = solve(row, pg[pos], vm[pos], name == "FR")
                attempts += attempted
                failures += attempted and not succeeded
            else:
                result = ({"success": False},)
            solutions.append(result)
            success = bool(result[0]["success"])
            converged.append(success)
            if success:
                vm_all[pos] = result[0]["bus"][:, 7]
                pg_all[pos] = result[0]["gen"][:, 1] / general["BASE_MVA"]
                qg_all[pos] = result[0]["gen"][:, 2] / general["BASE_MVA"]
        metrics = evaluate_acopf_predictions(pg_all, vm_all, raw["pg"][idx], raw["vm"][idx],
                                             raw["qg"][idx], raw["va"][idx], solutions,
                                             converged, params, verbose=False)
        timings = []
        feature_deltas = []
        for row in idx[valid_input[idx]][:20]:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            generated_features = None
            if spec.method == "GNN":
                pd, qd = raw["x"][row, :n_load], raw["x"][row, n_load:]
                full_load = np.zeros(n_bus)
                for j, bus in enumerate(general["load_bus_ids"]):
                    full_load[general["bus_id_to_idx"][int(bus)]] = pd[j]
                dc_pg = solve_dcopf(full_load, dc_params)
                if dc_pg is None:
                    raise RuntimeError("GNN timing input DCOPF failed on a previously valid sample")
                success, *features = run_powerflow_with_dcopf(
                    pd, qd, dc_pg, case, general["load_bus_ids"], general["bus_id_to_idx"], general["BASE_MVA"])
                if not success:
                    raise RuntimeError("GNN timing input power flow failed on a previously valid sample")
                generated_features = np.concatenate(features)[None, :]
                feature_deltas.append(float(np.max(np.abs(generated_features[0] - subopt[row]))))
            pg_one, vm_one = predict(np.array([row]), generated_features)
            solve(row, pg_one[0], vm_one[0], name == "FR")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
        metrics.update(train_time_s=state.train_time_s,
                       inference_sample_indices=idx[valid_input[idx]][:20].tolist(),
                       inference_ms=float(np.mean(timings) * 1000) if timings else None,
                       inference_scope="preprocessing, model and power-flow pipeline; batch=1; includes GNN DCOPF+PF features",
                       recovery_attempted_samples=int(attempts) if name != "AS" else None,
                       recovery_failed_samples=int(failures) if name != "AS" else None,
                       invalid_input_samples=int((~valid_input[idx]).sum()))
        if feature_deltas:
            metrics["input_feature_reconstruction_max_abs"] = max(feature_deltas)
        records[name] = metrics
        arrays.update({f"{name}_pg_pred": pg_all, f"{name}_qg_pred": qg_all,
                       f"{name}_vm_pred": vm_all, f"{name}_converged": np.asarray(converged)})
    return records, arrays
