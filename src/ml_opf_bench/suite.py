"""Run the manuscript matrix in isolated processes and retain every attempt."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import dataset_paths, paper_experiments
from .io import code_version, data_signature, write_json


def completed_scenarios_match(attempt, spec, data_root, signatures):
    scenarios = ["base"]
    if spec.case == "case118" and spec.mode == "cross-system" and spec.evaluate_shifts:
        scenarios += ["larger_variance", "heavier_loads"]
    marker = json.loads((attempt / "completed.json").read_text())
    if marker.get("scenarios") != scenarios:
        return False
    for scenario in scenarios:
        signature_path = attempt / f"{scenario}_data_manifest.json"
        if not all((attempt / name).is_file() for name in
                   (f"{scenario}.json", f"{scenario}_samples.npz", signature_path.name)):
            return False
        key = (spec.formulation, spec.case, scenario)
        if key not in signatures:
            signatures[key] = data_signature(dataset_paths(data_root, *key))
        if json.loads(signature_path.read_text()) != signatures[key]:
            return False
    return all((attempt / name).is_file() for name in ("checkpoint.pt", "splits.npz"))


def run_suite(data_root, output_root, seed=42, device="cuda", workers=4, formulation=None, mode=None):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    code = code_version()
    specs = [s for s in paper_experiments(seed, device)
             if (formulation is None or s.formulation == formulation) and (mode is None or s.mode == mode)]
    # DNN first makes the initial hardware timing available early.
    specs.sort(key=lambda s: (s.method != "DNN", s.mode != "cross-system", s.formulation, s.case))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    write_json(output_root / f"suite-{stamp}.json", {"experiments": [s.as_dict() for s in specs], "code": code})
    failed = []
    signatures = {}
    for i, spec in enumerate(specs, 1):
        if code_version()["source_sha256"] != code["source_sha256"]:
            raise RuntimeError("Source changed during the suite; start a separate versioned run")
        complete = False
        for marker in (output_root / spec.run_id).glob("*/completed.json"):
            manifest = json.loads((marker.parent / "manifest.json").read_text())
            expected = spec.as_dict() | {"workers": workers}
            data_key = (spec.formulation, spec.case)
            if data_key not in signatures:
                signatures[data_key] = data_signature(dataset_paths(data_root, *data_key))
            if (manifest["experiment"] == expected
                    and manifest["code"]["source_sha256"] == code["source_sha256"]
                    and manifest["data_signature"] == signatures[data_key]
                    and completed_scenarios_match(marker.parent, spec, data_root, signatures)):
                complete = True
                break
        if complete:
            print(f"[{i}/{len(specs)}] Already completed {spec.run_id}", flush=True)
            continue
        command = [sys.executable, "-m", "ml_opf_bench.cli", "run", "--formulation", spec.formulation,
                   "--method", spec.method, "--case", spec.case, "--mode", spec.mode,
                   "--seed", str(seed), "--device", device, "--workers", str(workers),
                   "--data-root", str(Path(data_root).resolve()), "--output-root", str(output_root)]
        if spec.train_size:
            command += ["--train-size", str(spec.train_size)]
        if not spec.evaluate_shifts:
            command += ["--no-shifts"]
        print(f"[{i}/{len(specs)}] Running {spec.run_id}", flush=True)
        environment = os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        result = subprocess.run(command, env=environment)
        if result.returncode:
            failed.append(spec.run_id)
            print(f"FAILED {spec.run_id}, exit={result.returncode}", flush=True)
    write_json(output_root / f"suite-outcome-{stamp}.json", {"failed": failed, "complete": not failed})
    return 1 if failed else 0
