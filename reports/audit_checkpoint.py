"""Replay trusted benchmark checkpoints in a fresh process against saved samples."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from ml_opf_bench.config import Experiment, dataset_paths
from ml_opf_bench.io import code_version, data_signature, write_json
from ml_opf_bench.registry import load_method


TIMING_FIELDS = {"train_time_s", "inference_ms"}


def compare_metrics(expected, actual, atol, rtol, path=""):
    """Compare every stored metric except measurements of elapsed time."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError(f"Metric schema mismatch: {path}")
        rows = []
        for key in expected:
            if key not in TIMING_FIELDS:
                rows.extend(compare_metrics(expected[key], actual[key], atol, rtol, f"{path}/{key}"))
        return rows
    if expected is None:
        matches = actual is None or (isinstance(actual, (float, np.floating)) and not np.isfinite(actual))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        matches = isinstance(actual, (int, float, np.number)) and bool(np.isclose(expected, actual, atol=atol, rtol=rtol, equal_nan=True))
    else:
        matches = bool(np.array_equal(expected, actual))
    return [{"metric": path, "matches": matches}]


def audit(attempt, data_root, samples, atol, rtol, device="cpu"):
    manifest = json.loads((attempt / "manifest.json").read_text())
    if not (attempt / "completed.json").is_file():
        raise ValueError("Only completed formal attempts can be audited")
    if code_version()["source_sha256"] != manifest["code"]["source_sha256"]:
        raise ValueError("Use the source version recorded by the attempt")
    original = Experiment(**manifest["experiment"])
    if original.epochs is not None or original.eval_limit is not None:
        raise ValueError("This audit requires a formal experiment, not smoke output")
    spec = replace(original, device=device, eval_limit=samples)
    if getattr(spec, "variant", "modified") != "modified":
        load_method(spec.formulation, spec.method, spec.variant)
    else:
        load_method(spec.formulation, spec.method)
    state = torch.load(attempt / "checkpoint.pt", map_location=device, weights_only=False)
    if spec.formulation == "ac":
        from ml_opf_bench.ac_evaluation import evaluate_ac as evaluate
    else:
        from ml_opf_bench.dc_evaluation import evaluate_dc as evaluate
    scenarios = json.loads((attempt / "completed.json").read_text())["scenarios"]
    results = []
    metric_results = []
    for scenario in scenarios:
        paths = dataset_paths(data_root, spec.formulation, spec.case, scenario)
        signature = json.loads((attempt / f"{scenario}_data_manifest.json").read_text())
        if signature != data_signature(paths):
            raise ValueError(f"Data signature changed: {scenario}")
        saved = np.load(attempt / f"{scenario}_samples.npz")
        metrics, replayed = evaluate(spec, state, paths, saved["indices"])
        if samples is None:
            saved_metrics = json.loads((attempt / f"{scenario}.json").read_text())
            metric_results.extend(compare_metrics(saved_metrics, metrics, atol, rtol, scenario))
        for key, values in replayed.items():
            expected = saved[key][:len(values)]
            if expected.shape != values.shape:
                raise ValueError(f"Replay shape mismatch: {scenario}/{key}")
            matches = bool(np.allclose(values, expected, atol=atol, rtol=rtol, equal_nan=True))
            finite = np.isfinite(values) & np.isfinite(expected)
            delta = float(np.max(np.abs(values[finite].astype(float) - expected[finite].astype(float)))) \
                if finite.any() else None
            results.append({"scenario": scenario, "array": key, "matches": matches, "max_abs_error": delta})
    return {"attempt": str(attempt), "source_sha256": manifest["code"]["source_sha256"],
            "device": device, "atol": atol, "rtol": rtol, "requested_samples": samples,
            "passed": all(r["matches"] for r in results + metric_results), "comparisons": results,
            "metrics_checked": samples is None, "metric_comparisons": metric_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="Replay every saved sample and compare non-timing metrics")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = audit(args.attempt.resolve(), args.data_root.resolve(), None if args.full else args.samples, args.atol, args.rtol, args.device)
    write_json(args.output, result)
    print(json.dumps({"passed": result["passed"], "output": str(args.output)}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
