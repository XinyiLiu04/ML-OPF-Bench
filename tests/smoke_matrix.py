"""Explicit integration checks; these outputs are never paper results."""

import argparse
import os
import subprocess
import sys

from ml_opf_bench.config import AC_METHODS, DC_METHODS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cases", nargs="+", default=["case30", "case118", "case300"])
    args = parser.parse_args()
    failures = []
    for case in args.cases:
        for form, methods in (("dc", DC_METHODS), ("ac", AC_METHODS)):
            for method in methods:
                if method in ("QP", "FR"):
                    continue
                command = [sys.executable, "-m", "ml_opf_bench.cli", "run", "--formulation", form,
                           "--method", method, "--case", case, "--pool-size", "120", "--epochs", "1",
                           "--eval-limit", "4", "--device", args.device, "--workers", str(args.workers),
                           "--data-root", args.data_root, "--output-root", args.output_root]
                print(f"SMOKE {form} {case} {method}", flush=True)
                result = subprocess.run(command, env=os.environ | {"OMP_NUM_THREADS": "1",
                                         "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
                if result.returncode:
                    failures.append((form, case, method, result.returncode))
    print(f"Smoke failures: {failures}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
