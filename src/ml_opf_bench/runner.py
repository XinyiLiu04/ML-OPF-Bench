"""Run one experiment without overwriting earlier attempts."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import traceback
import inspect
import time

import numpy as np
import torch

from .config import dataset_paths
from .io import code_version, data_signature, environment, write_json
from .registry import training_call, create_method, is_custom_method
from .runtime import TrainingState, managed_run
from .datasets import load_dataset
from .methods import predict_method
from .evaluation import evaluate_baseline, evaluate_predictions, EvaluationResult


def run_experiment(spec, data_root, output_root):
    if spec.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_root = Path(output_root) / spec.run_id
    attempt = run_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt.mkdir(parents=True, exist_ok=False)
    paths = dataset_paths(data_root, spec.formulation, spec.case)
    metadata = dict(experiment=spec.as_dict(), data_root=str(Path(data_root).resolve()),
                    environment=environment(), code=code_version(),
                    data_signature=data_signature(paths))
    with (attempt / "run.log").open("x", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            custom = is_custom_method(spec.formulation, spec.method)
            if custom:
                method = create_method(spec.formulation, spec.method, **spec.method_options)
                data = load_dataset(data_root, spec.formulation, spec.case)
                split = data.split(seed=spec.seed, mode=spec.mode, pool_size=spec.pool_size,
                                   train_size=spec.train_size)
                kwargs = {"seed": spec.seed, "method_options": spec.method_options}
                source = inspect.getsourcefile(type(method))
                from .io import sha256
                metadata["custom_method"] = {
                    "class": f"{type(method).__module__}:{type(method).__qualname__}",
                    "source_sha256": sha256(source) if source is not None and Path(source).is_file() else None,
                }
            else:
                module, function, kwargs = training_call(spec, paths)
            metadata["training_protocol"] = spec.resolved_training_protocol
            metadata["training_parameters"] = kwargs
            write_json(attempt / "manifest.json", metadata)
            with managed_run(spec) as context:
                if custom:
                    context.indices = tuple(part.inputs.indices for part in (split.train, split.validation, split.test))
                    context.n_total = len(data.inputs.indices)
                    _synchronize(spec.device)
                    started = time.perf_counter()
                    fitted = method.fit(split.train, split.validation, seed=spec.seed)
                    _synchronize(spec.device)
                    if fitted is not None:
                        raise TypeError("fit must update the method instance and return None")
                    state = TrainingState(method, data.inputs.params, time.perf_counter() - started,
                                          {"method_options": spec.method_options})
                else:
                    state = function(**kwargs)
            if not isinstance(state, TrainingState):
                raise TypeError("Training function did not return a model checkpoint")
            if isinstance(state.model, torch.nn.Module):
                if any(not torch.isfinite(p).all() for p in state.model.parameters()):
                    raise FloatingPointError("Training produced nonfinite model parameters")
                architecture = str(state.model)
                n_parameters = sum(p.numel() for p in state.model.parameters())
            else:
                architecture, n_parameters = type(state.model).__name__, None
            write_json(attempt / "checkpoint_metadata.json", {
                "architecture": architecture, "n_parameters": n_parameters,
                "epochs_completed": context.epochs_completed,
                "environment_steps": state.artifacts.get("total_timesteps"),
                "train_time_s": state.train_time_s,
                "training_timing": state.artifacts.get("training_timing"), "artifacts": list(state.artifacts)})
            torch.save(state, attempt / "checkpoint.pt")
            if "candidate_selection" in state.artifacts:
                write_json(attempt / "candidate_selection.json", state.artifacts["candidate_selection"])
            if "validation_selection" in state.artifacts:
                write_json(attempt / "validation_selection.json", state.artifacts["validation_selection"])
            np.savez_compressed(attempt / "splits.npz", train=context.indices[0], val=context.indices[1], test=context.indices[2])
            scenarios = ["base"]
            if spec.case == "case118" and spec.mode == "cross-system" and spec.evaluate_shifts:
                scenarios += ["larger_variance", "heavier_loads"]
            for scenario in scenarios:
                evaluation_paths = dataset_paths(data_root, spec.formulation, spec.case, scenario)
                write_json(attempt / f"{scenario}_data_manifest.json", data_signature(evaluation_paths))
                indices = context.indices[2] if scenario == "base" else None
                if custom:
                    evaluation_data = data if scenario == "base" else load_dataset(
                        data_root, spec.formulation, spec.case, scenario)
                    keys = ("bus_ids", "gen_bus_ids", "load_bus_ids", "non_slack_gen_idx") if spec.formulation == "ac" else (
                        "bus_ids", "gen_ids", "non_slack_gen_idx", "slack_gen_idx")
                    for key in keys:
                        np.testing.assert_array_equal(data.inputs.params["general"][key],
                                                      evaluation_data.inputs.params["general"][key])
                    from .evaluation import evaluation_indices
                    rows = evaluation_indices(len(evaluation_data.inputs.indices), indices, spec.eval_limit)
                    result = _evaluate_custom(spec, state, evaluation_data.partition(rows))
                else:
                    result = evaluate_baseline(spec, state, evaluation_paths, indices)
                write_json(attempt / f"{scenario}.json", result.records)
                write_json(attempt / f"{scenario}_evaluation.json", result.to_dict())
                np.savez_compressed(attempt / f"{scenario}_samples.npz", **result.samples)
                print(f"Completed {scenario}: {result.records}", flush=True)
            write_json(attempt / "completed.json", {"status": "completed", "scenarios": scenarios})
        except Exception as error:
            traceback.print_exc()
            write_json(attempt / "failed.json", {"status": "failed", "type": type(error).__name__, "error": str(error)})
            raise
    return attempt


def _synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _evaluate_custom(spec, state, partition):
    _synchronize(spec.device)
    started = time.perf_counter()
    raw = predict_method(state.model, partition.inputs)
    result = evaluate_predictions(partition, raw, name=spec.method)
    _synchronize(spec.device)
    elapsed = time.perf_counter() - started
    result.records[spec.method].update(train_time_s=state.train_time_s,
                                      evaluation_time_s=elapsed,
                                      evaluation_scope="batch prediction and physical evaluation")
    if not spec.postprocess:
        return result
    from .methods import Postprocessor, validate_repair
    if not isinstance(state.model, Postprocessor):
        raise TypeError("method does not implement postprocess")
    repaired = state.model.postprocess(partition.inputs, raw)
    validate_repair(partition.inputs, raw, repaired)
    repaired_result = evaluate_predictions(partition, repaired, name=spec.method + "-repaired")
    samples = dict(result.samples)
    for key, value in repaired_result.samples.items():
        if key.startswith(spec.method + "-repaired_"):
            samples[key] = value
    samples[spec.method + "-repaired_valid"] = (np.ones(len(partition.inputs.indices), dtype=bool)
                                               if repaired.valid is None else repaired.valid.copy())
    samples[spec.method + "-repaired_pg_setpoints"] = repaired.pg.copy()
    if repaired.vm is not None:
        samples[spec.method + "-repaired_vm_setpoints"] = repaired.vm.copy()
    return EvaluationResult(spec.formulation, result.records | repaired_result.records, samples)
