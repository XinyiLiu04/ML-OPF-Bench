"""Use one spare execution slot while the existing AC child finishes."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone

from ml_opf_bench.config import paper_experiments
from ml_opf_bench.io import code_version
from ml_opf_bench.suite import completed_scenarios_match

ROOT = Path('/lambda/nfs/opf/lxy-ml-opf')
OUT = ROOT / 'runs/paper-seed42-2c1d52c'
PARENT, CHILD = 10180, 11167
SHA = '39caf5accb8e14f0ae2ae179955629ce16e50491db439714d42930542ca5c0c0'


def event(**fields):
    record = {'utc': datetime.now(timezone.utc).isoformat(), **fields}
    print(json.dumps(record), flush=True)


def live_child():
    p = Path(f'/proc/{CHILD}/stat')
    return p.exists() and p.read_text().split()[2] != 'Z'


def main():
    with (ROOT / 'logs/parallel-dc.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert code_version()['source_sha256'] == SHA
        parent_cmd = Path(f'/proc/{PARENT}/cmdline').read_bytes()
        assert b'ml_opf_bench.cli\x00suite' in parent_cmd
        assert live_child()
        assert int(Path(f'/proc/{CHILD}/stat').read_text().split()[3]) == PARENT
        os.kill(PARENT, signal.SIGSTOP)
        event(action='pause_dispatch_only', parent=PARENT, active_child=CHILD)
        try:
            signatures = {}
            for spec in paper_experiments(42, 'cuda'):
                if not live_child():
                    break
                if spec.formulation != 'dc':
                    continue
                attempts = OUT / spec.run_id
                if list(attempts.glob('*/failed.json')):
                    raise RuntimeError(f'Existing failure requires review: {spec.run_id}')
                complete = False
                for marker in attempts.glob('*/completed.json'):
                    manifest = json.loads((marker.parent / 'manifest.json').read_text())
                    if (manifest['experiment'] == spec.as_dict() | {'workers': 8}
                            and manifest['code']['source_sha256'] == SHA
                            and completed_scenarios_match(marker.parent, spec, ROOT / 'data', signatures)):
                        complete = True
                        break
                if complete:
                    continue
                assert code_version()['source_sha256'] == SHA
                cmd = [sys.executable, '-m', 'ml_opf_bench.cli', 'run', '--formulation', 'dc',
                       '--method', spec.method, '--case', spec.case, '--mode', spec.mode,
                       '--seed', '42', '--device', 'cuda', '--workers', '8',
                       '--data-root', str(ROOT / 'data'), '--output-root', str(OUT)]
                if spec.train_size:
                    cmd += ['--train-size', str(spec.train_size)]
                if not spec.evaluate_shifts:
                    cmd += ['--no-shifts']
                event(action='start', run_id=spec.run_id, timing_context='concurrent_with_ac_cp')
                result = subprocess.run(cmd, check=False)
                event(action='finish', run_id=spec.run_id, returncode=result.returncode)
                if result.returncode:
                    raise RuntimeError(f'New failure requires review: {spec.run_id}')
        except BaseException:
            event(action='review_required_dispatch_remains_paused', parent=PARENT)
            raise
        else:
            os.kill(PARENT, signal.SIGCONT)
            event(action='resume_dispatch', parent=PARENT)


if __name__ == '__main__':
    main()
