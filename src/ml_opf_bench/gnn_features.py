"""Derived GNN inputs; original OPF labels and feature CSVs are never overwritten."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from .io import sha256, write_json
from .registry import implementation_root, load_method

_WORKER = None


def _initialize(case_name, ac_folder, dc_folder, params):
    global _WORKER
    load_method("ac", "GNN")
    from generate_subopt_state import load_dcopf_constraints
    from ac_configuration.acopf_pypower import load_case_from_csv
    _WORKER = (load_dcopf_constraints(case_name, dc_folder),
               load_case_from_csv(case_name, ac_folder), params["general"])


def _sample(item):
    from generate_subopt_state import solve_dcopf, run_powerflow_with_dcopf
    row, pd, qd = item
    dc, case, g = _WORKER
    full = np.zeros(g["n_buses"])
    for j, bus in enumerate(g["load_bus_ids"]):
        full[g["bus_id_to_idx"][int(bus)]] = pd[j]
    pg = solve_dcopf(full, dc)
    if pg is None:
        return row, None
    success, *features = run_powerflow_with_dcopf(pd, qd, pg, case, g["load_bus_ids"],
                                                g["bus_id_to_idx"], g["BASE_MVA"])
    return row, np.concatenate(features) if success else None


def load_gnn_features(case_name, ac_folder, data_path, params, loads, rows, workers):
    ac_folder, data_path = Path(ac_folder).resolve(), Path(data_path).resolve()
    dc_folder = ac_folder.parents[2] / "dc_dataset" / "dcopf_constraints" / ac_folder.name
    rows = np.unique(rows)
    key = {
        "case": case_name,
        "rows": hashlib.sha256(rows.tobytes()).hexdigest(),
        "loads": hashlib.sha256(loads[rows].tobytes()).hexdigest(),
        "constraints": {str(p.relative_to(ac_folder.parents[2])): sha256(p)
                        for folder in (ac_folder, dc_folder) for p in folder.glob("*.csv")},
        "algorithm": {p.name: sha256(p) for p in (
            Path(__file__), implementation_root("ac") / "acopf_gnn/generate_subopt_state.py",
            implementation_root("ac") / "ac_configuration/acopf_pypower.py")},
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    cache_root = Path(os.environ.get("ML_OPF_FEATURE_CACHE", ac_folder.parents[2] / "runs/derived_features"))
    folder = cache_root / case_name / digest
    filename = folder / "features.npz"
    n_bus, n_load = params["general"]["n_buses"], params["general"]["n_loads"]
    features = np.full((len(loads), 4 * n_bus), np.nan, dtype=np.float32)
    valid = np.zeros(len(loads), dtype=bool)
    if filename.exists():
        with np.load(filename) as cached:
            np.testing.assert_array_equal(cached["rows"], rows)
            features[rows], valid[rows] = cached["features"], cached["valid"]
        return features, valid
    print(f"Preparing {len(rows)} GNN inputs with current constraints", flush=True)
    initargs = (case_name, str(ac_folder), str(dc_folder), params)
    items = ((int(i), loads[i, :n_load], loads[i, n_load:]) for i in rows)
    if workers == 1:
        _initialize(*initargs)
        results = map(_sample, items)
        for row, result in results:
            if result is not None:
                features[row], valid[row] = result, True
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize, initargs=initargs) as pool:
            for count, (row, result) in enumerate(pool.map(_sample, items, chunksize=16), 1):
                if result is not None:
                    features[row], valid[row] = result, True
                if count % 256 == 0:
                    print(f"GNN features {count}/{len(rows)}", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "manifest.json", key)
    with filename.open("xb") as out:
        np.savez_compressed(out, rows=rows, features=features[rows], valid=valid[rows])
    return features, valid
