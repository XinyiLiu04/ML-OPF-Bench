"""Check RL OOD observations and case300 AS with the actual training split."""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    for method, case, pool in (("RL", "case118", 120), ("AS", "case300", 12000),
                               ("DNN", "case30", 120)):
        command = [sys.executable, "-m", "ml_opf_bench.cli", "run", "--formulation", "ac",
                   "--method", method, "--case", case, "--pool-size", str(pool), "--epochs", "1",
                   "--eval-limit", "4", "--device", "cuda", "--workers", "8",
                   "--data-root", args.data_root, "--output-root", args.output_root]
        subprocess.run(command, check=True, env=os.environ | {"OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    print("Final preflight passed", flush=True)


if __name__ == "__main__":
    main()
