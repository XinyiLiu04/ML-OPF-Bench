"""Run one experiment without overwriting earlier attempts."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import traceback

import numpy as np
import torch

from .config import dataset_paths
from .io import code_version, data_signature, environment, write_json
from .registry import training_call
from .runtime import TrainingState, managed_run


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
            module, function, kwargs = training_call(spec, paths)
            metadata["training_parameters"] = kwargs
            write_json(attempt / "manifest.json", metadata)
            with managed_run(spec) as context:
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
                "train_time_s": state.train_time_s, "artifacts": list(state.artifacts)})
            torch.save(state, attempt / "checkpoint.pt")
            np.savez_compressed(attempt / "splits.npz", train=context.indices[0], val=context.indices[1], test=context.indices[2])
            if spec.formulation == "dc":
                from .dc_evaluation import evaluate_dc as evaluate
            else:
                from .ac_evaluation import evaluate_ac as evaluate
            scenarios = ["base"]
            if spec.case == "case118" and spec.mode == "cross-system" and spec.evaluate_shifts:
                scenarios += ["larger_variance", "heavier_loads"]
            for scenario in scenarios:
                evaluation_paths = dataset_paths(data_root, spec.formulation, spec.case, scenario)
                write_json(attempt / f"{scenario}_data_manifest.json", data_signature(evaluation_paths))
                records, arrays = evaluate(spec, state, evaluation_paths,
                                           context.indices[2] if scenario == "base" else None)
                write_json(attempt / f"{scenario}.json", records)
                np.savez_compressed(attempt / f"{scenario}_samples.npz", **arrays)
                print(f"Completed {scenario}: {records}", flush=True)
            write_json(attempt / "completed.json", {"status": "completed", "scenarios": scenarios})
        except Exception as error:
            traceback.print_exc()
            write_json(attempt / "failed.json", {"status": "failed", "type": type(error).__name__, "error": str(error)})
            raise
    return attempt
