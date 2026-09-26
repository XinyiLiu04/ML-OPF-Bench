"""Validate transferred inputs before starting the second host's explicit queue."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/ubuntu/lxy-west')
for pid in (2503, 2524):
    while (p := Path(f'/proc/{pid}/stat')).exists() and p.read_text().split()[2] != 'Z':
        time.sleep(10)
manifest = json.loads((root / 'environment/data_manifest.json').read_text())
for relative, expected in manifest['files'].items():
    p = root / 'data' / relative
    assert p.stat().st_size == expected['size'], relative
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    assert h.hexdigest() == expected['sha256'], relative
snapshot = root / 'snapshots/seed42-2c1d52c'
os.chdir(snapshot)
os.environ.update(PYTHONPATH=f'{snapshot}/src:{snapshot}', ML_OPF_FEATURE_CACHE=str(root / 'derived_features'),
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
python = '/home/ubuntu/lxy-ml-opf-env/bin/python'
check = '''import json,torch
from pathlib import Path
from ml_opf_bench.io import code_version,environment
assert code_version()['source_sha256']=='39caf5accb8e14f0ae2ae179955629ce16e50491db439714d42930542ca5c0c0'
e=environment()
old=json.loads(Path('/home/ubuntu/lxy-west/logs/expected-environment.json').read_text())
assert e['packages']==old['packages'], (e['packages'],old['packages'])
assert e['python']==old['python'] and e['cuda']==old['cuda']
assert torch.cuda.is_available() and 'A10' in e['gpu']
with Path('/home/ubuntu/lxy-west/logs/verified-environment.json').open('x') as f:json.dump(e,f,indent=2)
print('SOURCE_ENVIRONMENT_GPU_VERIFIED',flush=True)
'''
subprocess.run([python, '-c', check], check=True)
print(f'DATA_VERIFIED {len(manifest["files"])} files', flush=True)
with (root / 'logs/bootstrap-verified.json').open('x') as f:
    json.dump({'verified_files': len(manifest['files']), 'source_sha256': '39caf5accb8e14f0ae2ae179955629ce16e50491db439714d42930542ca5c0c0'}, f)
os.execv(python, [python, '-u', str(root / 'logs/partition_queue_v1.py'), '--root', str(root),
                  '--plan', str(root / 'logs/partition-west-v1.json')])
