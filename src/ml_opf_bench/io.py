"""Versioned run metadata and JSON serialization."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(json_value(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def data_signature(paths):
    folders = {"constraints": paths.params_path, "samples": paths.data_path.parent}
    if paths.duals_path is not None:
        folders["duals"] = paths.duals_path
    return {name: {p.name: sha256(p) for p in sorted(folder.glob("*.csv"))}
            for name, folder in folders.items()}


def environment():
    packages = ("torch", "numpy", "pandas", "scipy", "scikit-learn", "PYPOWER", "torch-geometric", "stable-baselines3")
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in packages}}
    import torch
    result["cuda"] = torch.version.cuda
    result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return result


def code_version():
    from .registry import implementation_root
    root = implementation_root("ac").parent
    files = [p for folder in ("src/ml_opf_bench", "ac_methods", "dc_methods")
             for p in (root / folder).rglob("*.py")]
    digests = {str(p.relative_to(root)): sha256(p) for p in sorted(files)}
    digest = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=False)
    return {"source_sha256": digest, "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "files": digests}
