"""Explicit dataset selection and the manuscript's experiment settings."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import os

AC_METHODS = ("LR", "DNN", "GNN", "MU", "CP", "FR", "KKT", "QC", "NGT", "E-NGT", "AS", "RL")
DC_METHODS = ("RL", "LR", "DNN", "GNN", "MU", "CP", "QP", "KKT", "NGT", "E-NGT", "AS")
SIZES = (1000, 5000, 12000, 20000, 30000, 35000)
SCALING = {"ac": ("LR", "DNN", "MU", "QC", "KKT"), "dc": ("LR", "DNN", "MU", "QP", "KKT")}
WIDTHS = {"ac": {"case30": (64, 32), "case118": (256, 128), "case300": (512, 256)},
          "dc": {"case30": (32, 16), "case118": (128, 64), "case300": (256, 128)}}
BATCHES = {"case30": 32, "case118": 64, "case300": 128}


def default_data_root():
    return Path(os.environ.get("ML_OPF_BENCH_DATA", Path.cwd())).expanduser().resolve()


@dataclass(frozen=True)
class DatasetPaths:
    case_name: str
    params_path: Path
    data_path: Path
    duals_path: Path | None


def dataset_paths(root, formulation, case, scenario="base"):
    root = Path(root)
    if formulation not in ("ac", "dc") or case not in WIDTHS[formulation]:
        raise ValueError(f"Unsupported formulation/case: {formulation}/{case}")
    if scenario not in ("base", "larger_variance", "heavier_loads"):
        raise ValueError(f"Unknown scenario: {scenario}")
    if scenario != "base" and case != "case118":
        raise ValueError("Distribution shifts are defined only for case118")
    stem = f"pglib_opf_{case}_ieee"
    constraint_case = case
    sampling = "v=0.25" if scenario == "larger_variance" else "v=0.12"
    if scenario == "heavier_loads":
        stem += "__api"
        constraint_case += "(api)"
        sampling = "api" if formulation == "ac" else "v=api"
    base = root / f"{formulation}_dataset"
    folder = base / f"{formulation}opf_datasets" / f"{case}({sampling})"
    suffix = "pd" if formulation == "ac" else "dataset_with_duals"
    return DatasetPaths(stem, base / f"{formulation}opf_constraints" / constraint_case,
                        folder / f"{stem}_{suffix}.csv",
                        folder.with_name(folder.name + "_with_duals") if formulation == "ac" else None)


@dataclass(frozen=True)
class Experiment:
    formulation: str
    method: str
    case: str = "case118"
    mode: str = "cross-system"
    seed: int = 42
    train_size: int | None = None
    pool_size: int = 12000
    epochs: int | None = None
    device: str = "cuda"
    evaluate_shifts: bool = True
    eval_limit: int | None = None
    workers: int = 4
    variant: str | None = None
    method_options: dict = field(default_factory=dict)
    postprocess: bool = False

    def __post_init__(self):
        if self.variant is None:
            object.__setattr__(self, "variant", "ddpg-pgonly" if self.formulation == "ac" and self.method == "RL" else "modified")
        if self.variant not in ("modified", "paper", "ddpg-pgonly"):
            raise ValueError(self.variant)
        if self.variant == "ddpg-pgonly" and (self.formulation != "ac" or self.method != "RL"):
            raise ValueError("DDPG Pg-only is only defined for AC RL")
        if self.variant == "paper" and (self.formulation != "ac" or self.method not in ("NGT", "E-NGT")):
            raise ValueError("Paper variant is defined only for AC NGT/E-NGT")
        if self.formulation not in ("ac", "dc"):
            raise ValueError(self.formulation)
        from .registry import list_methods, is_custom_method
        if self.method not in list_methods(self.formulation):
            raise ValueError(self.method)
        if self.case not in WIDTHS[self.formulation]:
            raise ValueError(self.case)
        if self.mode not in ("cross-system", "scaling"):
            raise ValueError(self.mode)
        if self.mode == "scaling" and (self.case != "case118" or not self.train_size):
            raise ValueError("Scaling requires case118 and a positive training size")
        for name in ("epochs", "eval_limit", "workers", "pool_size", "train_size"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.method_options, dict):
            raise TypeError("method_options must be a dictionary")
        json.dumps(self.method_options, allow_nan=False)
        custom = is_custom_method(self.formulation, self.method)
        if not custom and (self.method_options or self.postprocess):
            raise ValueError("method_options and postprocess are for registered extensions")
        if custom and self.epochs is not None:
            raise ValueError("Set custom training budgets through method_options")

    @property
    def run_id(self):
        size = f"-n{self.train_size}" if self.mode == "scaling" else ""
        variant = "" if self.variant == "modified" else f"-{self.variant}"
        return f"{self.formulation}-{self.case}-{self.method.lower()}-{self.mode}{size}-seed{self.seed}{variant}"

    @property
    def resolved_training_protocol(self):
        if self.formulation == "ac" and self.method in ("NGT", "E-NGT") and self.variant == "modified":
            return "fixed-budget-final"
        return "implementation-default"

    def training_parameters(self):
        epochs = 1000
        if self.method == "MU":
            epochs = 80
        if self.method in ("NGT", "E-NGT"):
            epochs = 3000 if self.case == "case118" else 2500
        settings = dict(n_train_use=self.pool_size, seed=self.seed, n_epochs=self.epochs or epochs,
                    early_stop_patience=20, early_stop_min_delta=1e-6,
                    learning_rate=3e-4 if self.method == "RL" else 1e-3,
                    hidden_sizes=list(WIDTHS[self.formulation][self.case]),
                    batch_size=BATCHES[self.case], device=self.device)
        if self.resolved_training_protocol == "fixed-budget-final":
            settings.pop("early_stop_patience")
            settings.pop("early_stop_min_delta")
        return settings

    def as_dict(self):
        return asdict(self)


def paper_experiments(seed=42, device="cuda"):
    for form, methods in (("ac", AC_METHODS), ("dc", DC_METHODS)):
        for case in WIDTHS[form]:
            for method in methods:
                if method in ("FR", "QP"):
                    continue  # The CP run records both the raw and repaired dispatch.
                yield Experiment(form, method, case, seed=seed, device=device,
                                 variant="ddpg-pgonly" if form == "ac" and method == "RL" else "modified")
        for method in SCALING[form]:
            for size in SIZES:
                yield Experiment(form, method, mode="scaling", train_size=size, seed=seed,
                                 device=device, evaluate_shifts=False)
