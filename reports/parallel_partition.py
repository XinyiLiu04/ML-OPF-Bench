"""Continue a paused partition with bounded concurrency and no retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ml_opf_bench.config import Experiment
from ml_opf_bench.io import code_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--paused-pid', type=int, required=True)
    parser.add_argument('--adopt-pid', type=int, required=True)
    parser.add_argument('--adopt-run', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    out = args.root / 'runs/paper-seed42-2c1d52c'
    adopted = Experiment(**next(r for r in plan['experiments'] if Experiment(**r).run_id == args.adopt_run))
    pending = []
    for raw in plan['experiments']:
        spec = Experiment(**raw)
        attempts = list((out / spec.run_id).glob('*/manifest.json'))
        if spec.run_id == args.adopt_run:
            assert len(attempts) == 1
        elif attempts:
            assert len(attempts) == 1 and (attempts[0].parent / 'completed.json').exists(), spec.run_id
        else:
            pending.append(spec)
    active = {args.adopt_pid: (adopted, None)}
    with (args.root / 'logs/parallel-partition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while pending or active:
            assert Path(f'/proc/{args.paused_pid}/stat').read_text().split()[2] == 'T'
            for pid, (spec, child) in list(active.items()):
                if child is not None:
                    finished = child.poll() is not None
                else:
                    proc = Path(f'/proc/{pid}/stat')
                    finished = not proc.exists() or proc.read_text().split()[2] == 'Z'
                if not finished:
                    continue
                completed = list((out / spec.run_id).glob('*/completed.json'))
                failures = list((out / spec.run_id).glob('*/failed.json'))
                known = (spec.case == 'case300' and spec.method == 'AS' and len(failures) == 1
                         and 'No validation sample shares an active set' in failures[0].read_text())
                assert (len(completed) == 1 and not failures and (child is None or child.returncode == 0)) or known, spec.run_id
                print(json.dumps({'action': 'finish', 'run_id': spec.run_id, 'pid': pid, 'known_unavailable': known}), flush=True)
                del active[pid]
            while pending and len(active) < 3:
                spec = pending.pop(0)
                assert code_version()['source_sha256'] == plan['source_sha256']
                assert not list((out / spec.run_id).glob('*/manifest.json')), spec.run_id
                cmd = [sys.executable, '-m', 'ml_opf_bench.cli', 'run', '--formulation', spec.formulation,
                       '--method', spec.method, '--case', spec.case, '--mode', spec.mode, '--seed', str(spec.seed),
                       '--device', 'cuda', '--workers', '8', '--data-root', str(args.root / 'data'), '--output-root', str(out)]
                if spec.train_size:
                    cmd += ['--train-size', str(spec.train_size)]
                if not spec.evaluate_shifts:
                    cmd += ['--no-shifts']
                child = subprocess.Popen(cmd)
                active[child.pid] = (spec, child)
                print(json.dumps({'action': 'start', 'run_id': spec.run_id, 'pid': child.pid,
                                  'host': os.uname().nodename, 'concurrency_limit': 3}), flush=True)
            time.sleep(10)
        print('PARTITION_FINISHED', flush=True)


if __name__ == '__main__':
    main()
