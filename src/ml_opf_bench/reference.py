"""Benchmark the original numerical solver on held-out inputs without changing labels."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from .config import dataset_paths
from .io import code_version, data_signature, environment, sha256, write_json
from .registry import implementation_root, load_method
from .splits import make_splits


def ac_load_inputs(pd_path, qd_path):
    frames = [pd.read_csv(pd_path), pd.read_csv(qd_path)]
    columns = []
    bus_ids = []
    for frame, prefix in zip(frames, ("pd", "qd")):
        matched = [(int(match.group(1)), col) for col in frame.columns
                   if (match := re.fullmatch(rf"{prefix}_?(\d+)", col))]
        if not matched:
            raise ValueError(f"No {prefix} load columns found")
        matched.sort()
        ids, cols = zip(*matched)
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate {prefix} load bus IDs")
        bus_ids.append(list(ids))
        columns.append(list(cols))
    if bus_ids[0] != bus_ids[1] or len(frames[0]) != len(frames[1]):
        raise ValueError("Active and reactive loads have different buses or row counts")
    loads = [frame[cols].to_numpy(dtype=float) for frame, cols in zip(frames, columns)]
    if not all(np.isfinite(values).all() for values in loads):
        raise ValueError("Load inputs contain nonfinite values")
    return bus_ids[0], *loads


def reference_timing(formulation, case, data_root, output_root, julia="julia", seed=42, samples=20):
    paths = dataset_paths(data_root, formulation, case)
    load_method(formulation, "LR")
    if formulation == "dc":
        from dc_configuration.dcopf_data_setup import load_parameters_from_csv, load_samples
        params = load_parameters_from_csv(paths.case_name, str(paths.params_path))
        pd_load, pg_labels = load_samples(str(paths.data_path), params)
        costs = params["constraints"]
        payload = {"constraints": params["constraints"]}
    else:
        from ac_configuration.acopf_data_setup import load_parameters_from_csv
        params = load_parameters_from_csv(paths.case_name, str(paths.params_path))
        ids, pd_load, qd_load = ac_load_inputs(
            paths.data_path, paths.data_path.with_name(f"{paths.case_name}_qd.csv"))
        pg_frame = pd.read_csv(paths.data_path.with_name(f"{paths.case_name}_pg.csv"))
        pg_columns = sorted((c for c in pg_frame.columns if re.fullmatch(r"pg_\d+", c)),
                            key=lambda c: int(c.split("_")[1]))
        pg_labels = pg_frame[pg_columns].to_numpy(dtype=float)
        costs = params["generator"]
        payload = {"load_bus_ids": ids,
                   "case_path": implementation_root("ac").parent / "test_systems/typ" / f"{paths.case_name}.m"}
    idx = make_splits(len(pd_load), seed)[2][:samples]
    if pg_labels.shape != (len(pd_load), len(costs["cost_c2"])) or not np.isfinite(pg_labels).all():
        raise ValueError("Invalid generation label shape or values")
    label_costs = np.sum(pg_labels[idx] ** 2 * costs["cost_c2"]
                        + pg_labels[idx] * costs["cost_c1"] + costs["cost_c0"], axis=1)
    payload.update(formulation=formulation, samples=[{"index": int(i), "pd": pd_load[i],
                   **({"qd": qd_load[i]} if formulation == "ac" else {})} for i in idx])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    folder = Path(output_root).resolve() / f"reference-{formulation}-{case}-seed{seed}" / stamp
    folder.mkdir(parents=True)
    write_json(folder / "inputs.json", payload)
    project = implementation_root("ac").parent / "reference_solver"
    write_json(folder / "manifest.json", {"data_signature": data_signature(paths), "code": code_version(),
               "solver_sha256": sha256(project / "time_reference.jl"), "seed": seed,
               "case": case, "formulation": formulation, "sample_indices": idx})
    with (folder / "solver.log").open("x") as log:
        subprocess.run([julia, f"--project={project}", str(project / "time_reference.jl"),
                        str(folder / "inputs.json"), str(folder / "timings.json")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    records = json.loads((folder / "timings.json").read_text())["records"]
    if any(r["status"] not in ("LOCALLY_SOLVED", "OPTIMAL") for r in records):
        raise RuntimeError("Reference solver failed; see timings.json")
    np.testing.assert_array_equal([r["index"] for r in records], idx)
    reference_costs = np.asarray([r["cost"] for r in records])
    if not np.isfinite(reference_costs).all() or np.any(label_costs == 0):
        raise ValueError("Invalid costs in reference comparison")
    write_json(folder / "label_comparison.json", {"indices": idx, "label_costs": label_costs,
               "reference_costs": reference_costs,
               "signed_relative_gap_percent": 100 * (reference_costs - label_costs) / label_costs})
    write_json(folder / "completed.json", {"inference_ms": float(np.mean([r["milliseconds"] for r in records])),
                                           "n_samples": len(records), "environment": environment()})
    return folder
