"""Check RL OOD observations and case300 AS with the actual training split."""

import argparse
import json
import os
from pathlib import Path
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
        result = subprocess.run(command, env=os.environ | {"OMP_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        if result.returncode:
            folder = Path(args.output_root) / f"ac-{case}-{method.lower()}-cross-system-seed42"
            failure = json.loads(sorted(folder.glob("*/failed.json"))[-1].read_text())
            if method == "AS" and failure["error"].startswith("No validation sample shares an active set with training;"):
                print(f"Known method limitation ({case} AS): {failure['error']}", flush=True)
            else:
                raise subprocess.CalledProcessError(result.returncode, command)
    print("Final preflight finished; review any explicitly reported method limitation", flush=True)


if __name__ == "__main__":
    main()
