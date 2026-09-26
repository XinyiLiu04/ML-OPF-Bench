"""Execute an explicit disjoint queue using an unchanged frozen implementation."""
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
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--wait-pid', type=int)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    with (a.root / 'logs/partition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.wait_pid:
            while (q := Path(f'/proc/{a.wait_pid}/stat')).exists() and q.read_text().split()[2] != 'Z':
                time.sleep(10)
        for raw in plan['experiments']:
            s = Experiment(**raw)
            assert code_version()['source_sha256'] == plan['source_sha256']
            out = a.root / 'runs/paper-seed42-2c1d52c'
            if list((out / s.run_id).glob('*/manifest.json')):
                raise RuntimeError(f'Existing attempt requires review: {s.run_id}')
            cmd = [sys.executable, '-m', 'ml_opf_bench.cli', 'run', '--formulation', s.formulation,
                   '--method', s.method, '--case', s.case, '--mode', s.mode, '--seed', str(s.seed),
                   '--device', 'cuda', '--workers', '8', '--data-root', str(a.root / 'data'),
                   '--output-root', str(out)]
            if s.train_size:
                cmd += ['--train-size', str(s.train_size)]
            if not s.evaluate_shifts:
                cmd += ['--no-shifts']
            print(json.dumps({'action': 'start', 'run_id': s.run_id, 'host': os.uname().nodename}), flush=True)
            result = subprocess.run(cmd)
            print(json.dumps({'action': 'finish', 'run_id': s.run_id, 'returncode': result.returncode}), flush=True)
            if result.returncode:
                failures = list((out / s.run_id).glob('*/failed.json'))
                known = (s.formulation == 'ac' and s.case == 'case300' and s.method == 'AS'
                         and len(failures) == 1 and 'No validation sample shares an active set' in failures[0].read_text())
                if not known:
                    raise RuntimeError('Unexpected failure; review before continuing')
        print('PARTITION_FINISHED', flush=True)


if __name__ == '__main__':
    main()
