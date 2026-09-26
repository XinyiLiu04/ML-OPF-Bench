"""Run one bounded third-slot experiment without changing the frozen suite."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone
from ml_opf_bench.io import code_version

ROOT = Path('/lambda/nfs/opf/lxy-ml-opf')
OUT = ROOT / 'runs/paper-seed42-2c1d52c'
DISPATCH = 12964
RUN = 'ac-case30-qc-cross-system-seed42'


def event(action, **fields):
    print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'action': action, **fields}), flush=True)


def main():
    with (ROOT / 'logs/third-slot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert code_version()['source_sha256'] == '39caf5accb8e14f0ae2ae179955629ce16e50491db439714d42930542ca5c0c0'
        assert b'parallel_dc_v1.py' in Path(f'/proc/{DISPATCH}/cmdline').read_bytes()
        assert Path('/proc/10180/stat').read_text().split()[2] == 'T'
        assert not list((OUT / RUN).glob('*/manifest.json')), 'Existing attempt requires inspection'
        os.kill(DISPATCH, signal.SIGSTOP)
        event('pause_dc_dispatch_only', pid=DISPATCH)
        try:
            assert Path('/proc/10180/stat').read_text().split()[2] == 'T'
            cmd = [sys.executable, '-m', 'ml_opf_bench.cli', 'run', '--formulation', 'ac',
                   '--method', 'QC', '--case', 'case30', '--mode', 'cross-system',
                   '--seed', '42', '--device', 'cuda', '--workers', '8',
                   '--data-root', str(ROOT / 'data'), '--output-root', str(OUT)]
            event('start', run_id=RUN, timing_context='up_to_three_concurrent_experiments')
            result = subprocess.run(cmd, check=False)
            event('finish', run_id=RUN, returncode=result.returncode)
            if result.returncode:
                raise RuntimeError('Third-slot failure requires review; dispatch remains paused')
        except BaseException:
            event('review_required_dispatch_remains_paused')
            raise
        else:
            os.kill(DISPATCH, signal.SIGCONT)
            event('resume_dc_dispatch', pid=DISPATCH)


if __name__ == '__main__':
    main()
