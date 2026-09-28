"""Command-line entry point for reproducible experiments."""

import argparse
import json
import importlib

from .config import Experiment, default_data_root, paper_experiments


def main():
    parser = argparse.ArgumentParser(prog="ml-opf-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    methods = commands.add_parser("methods")
    methods.add_argument("--formulation", choices=("ac", "dc"), required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--seed", type=int, default=42)
    suite = commands.add_parser("suite")
    suite.add_argument("--seed", type=int, default=42)
    suite.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    suite.add_argument("--data-root", default=str(default_data_root()))
    suite.add_argument("--output-root", required=True)
    suite.add_argument("--workers", type=int, default=4)
    suite.add_argument("--formulation", choices=("ac", "dc"))
    suite.add_argument("--mode", choices=("cross-system", "scaling"))
    reference = commands.add_parser("reference")
    reference.add_argument("--formulation", choices=("ac", "dc"), required=True)
    reference.add_argument("--case", default="case118")
    reference.add_argument("--data-root", default=str(default_data_root()))
    reference.add_argument("--output-root", required=True)
    reference.add_argument("--julia", default="julia")
    reference.add_argument("--seed", type=int, default=42)
    reference.add_argument("--samples", type=int, default=20)
    run = commands.add_parser("run")
    run.add_argument("--formulation", choices=("ac", "dc"), required=True)
    run.add_argument("--method", required=True)
    run.add_argument("--plugin", help="Importable module:factory for a custom method")
    run.add_argument("--method-options", type=json.loads, default={})
    run.add_argument("--postprocess", action="store_true")
    run.add_argument("--variant", choices=("modified", "paper", "ddpg-pgonly"), default=None)
    run.add_argument("--case", default="case118")
    run.add_argument("--mode", choices=("cross-system", "scaling"), default="cross-system")
    run.add_argument("--train-size", type=int)
    run.add_argument("--pool-size", type=int, default=12000)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--epochs", type=int)
    run.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    run.add_argument("--data-root", default=str(default_data_root()))
    run.add_argument("--output-root", default="runs")
    run.add_argument("--eval-limit", type=int)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--no-shifts", action="store_true")
    args = parser.parse_args()
    if args.command == "methods":
        from .registry import list_methods
        print(json.dumps(list_methods(args.formulation)))
        return
    if args.command == "plan":
        print(json.dumps([spec.as_dict() for spec in paper_experiments(args.seed)], indent=2))
        return
    if args.command == "suite":
        from .suite import run_suite
        raise SystemExit(run_suite(args.data_root, args.output_root, args.seed, args.device,
                                   args.workers, args.formulation, args.mode))
    if args.command == "reference":
        from .reference import reference_timing
        print(reference_timing(args.formulation, args.case, args.data_root, args.output_root,
                               args.julia, args.seed, args.samples))
        return
    from .runner import run_experiment
    method = args.method.upper()
    if args.plugin:
        from .registry import register_method
        module_name, separator, attribute = args.plugin.partition(":")
        if not separator or not module_name or not attribute:
            parser.error("--plugin must be module:factory")
        register_method(args.formulation, method, getattr(importlib.import_module(module_name), attribute))
    spec = Experiment(
        formulation=args.formulation, method=method, case=args.case, mode=args.mode,
        seed=args.seed, train_size=args.train_size, pool_size=args.pool_size, epochs=args.epochs,
        device=args.device, evaluate_shifts=not args.no_shifts, eval_limit=args.eval_limit,
        workers=args.workers,
        variant=args.variant or ("ddpg-pgonly" if args.formulation == "ac" and method == "RL" else "modified"),
        method_options=args.method_options, postprocess=args.postprocess,
    )
    print(run_experiment(spec, args.data_root, args.output_root))


if __name__ == "__main__":
    main()
