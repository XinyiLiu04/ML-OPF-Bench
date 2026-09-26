"""Verify an explicitly transferred third-host bundle before dispatching its queue."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root
assert (root/'logs/transfer-complete.json').exists(), 'Transfer has not completed'
manifest = json.loads((root/'environment/data_manifest.json').read_text())
for relative, expected in manifest['files'].items():
    path = root/'data'/relative
    assert path.stat().st_size == expected['size'], relative
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256'], relative
snapshot = root/'snapshots/seed42-2c1d52c'
os.chdir(snapshot)
os.environ.update(PYTHONPATH=f'{snapshot}/src:{snapshot}', ML_OPF_FEATURE_CACHE=str(root/'derived_features'),
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
python = '/home/ubuntu/lxy-ml-opf-env/bin/python'
check = '''import json,sys,torch
from pathlib import Path
from ml_opf_bench.io import code_version,environment
root=Path(sys.argv[1]); e=environment()
old=json.loads((root/'logs/expected-environment.json').read_text())
assert code_version()['source_sha256']=='39caf5accb8e14f0ae2ae179955629ce16e50491db439714d42930542ca5c0c0'
assert e['packages']==old['packages'] and e['python']==old['python'] and e['cuda']==old['cuda']
assert torch.cuda.is_available() and 'A10' in e['gpu']
with (root/'logs/verified-environment.json').open('x') as f: json.dump(e,f,indent=2)
'''
subprocess.run([python,'-c',check,str(root)],check=True)
with (root/'logs/bootstrap-verified.json').open('x') as f:
    json.dump({'verified_files':len(manifest['files'])},f)
print('DATA_SOURCE_ENVIRONMENT_GPU_VERIFIED',flush=True)
os.execv(python,[python,'-u',str(root/'logs/correction_queue.py'),'--root',str(root),
                 '--plan',str(root/'logs/plan-west2-v1.json')])
