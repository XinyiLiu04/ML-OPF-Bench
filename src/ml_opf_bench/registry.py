"""Adapters to the maintained method implementations."""

import importlib.util
import inspect
from pathlib import Path
import sys

AC = {
    "LR": ("acopf_lr.py", "linear_regression_experiment", {}),
    "DNN": ("acopf_dnn.py", "traditional_nn_acopf_experiment", {}),
    "CP": ("acopf_cp(fr).py", "train_pinn_acopf", {"penalty_weight": 0.1}),
    "FR": ("acopf_cp(fr).py", "train_pinn_acopf", {"penalty_weight": 0.1}),
    "QC": ("acopf_qc.py", "qcorrection_nn_acopf_experiment", {}),
    "MU": ("acopf_mu.py", "lagrangian_acopf_experiment", {}),
    "KKT": ("acopf_kkt/acopf_pinn_main.py", "acopf_pinn_experiment", {}),
    "GNN": ("acopf_gnn/gnn_main.py", "spectral_gnn_acopf_experiment", {"K": 4, "predict_vm": False}),
    "AS": ("acopf_as.py", "active_set_acopf_experiment", {"top_k": 3}),
    "RL": ("acopf_rl/acopf_rl.py", "acopf_rl_experiment", {"action_bounds": "physical"}),
    "NGT": ("acopf_ngt/unsupervised_learning_acopf.py", "train_deepopf_ngt_smoothed", {}),
    "E-NGT": ("acopf_ngt/semi_supervised_acopf.py", "train_extended_deepopf_ngt_smoothed", {"n_labeled": 300}),
}
DC = {
    "LR": ("dcopf_lr.py", "lr_experiment", {}),
    "DNN": ("dcopf_dnn.py", "dnn_experiment", {}),
    "CP": ("dcopf_cp(qp).py", "cp_qp_experiment", {"penalty_weight": 1e-5}),
    "QP": ("dcopf_cp(qp).py", "cp_qp_experiment", {"penalty_weight": 1e-5}),
    "MU": ("dcopf_mu.py", "mu_experiment", {"rho": 1e-2}),
    "KKT": ("dcopf_kkt_pinn.py", "kkt_pinn_experiment", {
        "dual_weight": 0.05, "kkt_weight": 5e-10, "collocation_ratio": 0.5}),
    "GNN": ("dcopf_gnn.py", "gnn_experiment", {"K": 4, "graph_kernel": "susceptance", "gaussian_scale": 0.01}),
    "AS": ("dcopf_as.py", "as_experiment", {"active_threshold": 1e-4, "top_k": 3, "dropout_rate": 0.1}),
    "NGT": ("dcopf_ngt.py", "ngt_experiment", {}),
    "E-NGT": ("dcopf_engt.py", "engt_experiment", {"n_labeled": 5000, "k_v": 100.0}),
}


def implementation_root(form):
    spec = importlib.util.find_spec(f"{form}_methods")
    return Path(next(iter(spec.submodule_search_locations)))


def load_method(form, method):
    filename, function, defaults = (AC if form == "ac" else DC)[method]
    root = implementation_root(form)
    path = root / filename
    for folder in (root, path.parent):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    name = path.stem
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    return module, getattr(module, function), defaults.copy()


def training_call(experiment, paths):
    module, function, options = load_method(experiment.formulation, experiment.method)
    settings = experiment.training_parameters()
    if experiment.method == "MU":
        settings["early_stop_patience"] = settings["n_epochs"] + 1
    if experiment.formulation == "ac":
        if experiment.method == "AS":
            settings["hidden_layers"] = settings.pop("hidden_sizes")
        elif experiment.method == "KKT":
            widths = settings.pop("hidden_sizes")
            settings.update({f"hidden_sizes_{part}": widths for part in ("V", "G", "Lg")})
        elif experiment.method == "GNN":
            settings["F1"], settings["F2"] = settings.pop("hidden_sizes")
        if experiment.method in ("CP", "FR"):
            settings["n_cores"] = experiment.workers
    # Small smoke tests retain the labeled fraction without sampling validation/test rows.
    if experiment.epochs is not None and "n_labeled" in options:
        available = experiment.train_size or (experiment.pool_size - 2 * (experiment.pool_size // 12))
        options["n_labeled"] = min(options["n_labeled"], available)
    if experiment.method == "RL" and experiment.epochs is not None:
        options.update(total_timesteps=128 * experiment.epochs, rollout_steps=64, n_scaling_probes=32)
    settings.update(options)
    settings.update(case_name=paths.case_name, params_path=str(paths.params_path), data_path=str(paths.data_path))
    signature = inspect.signature(function)
    if "duals_dir" in signature.parameters:
        settings["duals_dir"] = str(paths.duals_path)
    if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        settings = {k: v for k, v in settings.items() if k in signature.parameters}
    signature.bind(**settings)
    return module, function, settings
