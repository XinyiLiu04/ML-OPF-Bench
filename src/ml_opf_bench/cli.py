"""Command-line entry point for reproducible experiments."""

import argparse
import json

from .config import Experiment, default_data_root, paper_experiments


def main():
    parser = argparse.ArgumentParser(prog="ml-opf-bench")
    commands = parser.add_subparsers(dest="command", required=True)
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
    spec = Experiment(args.formulation, args.method.upper(), args.case, args.mode, args.seed,
                      args.train_size, args.pool_size, args.epochs, args.device, not args.no_shifts,
                      args.eval_limit, args.workers)
    print(run_experiment(spec, args.data_root, args.output_root))


if __name__ == "__main__":
    main()
