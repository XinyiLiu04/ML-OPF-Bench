"""Run the bounded, predeclared NGT variant diagnostic matrix once."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
snapshot=a.root/'snapshots/paper-comparison-cba535c'
out=a.root/'runs/ngt-variant-comparison-cba535c'
if out.exists():
    raise RuntimeError('Diagnostic output already exists; review before any continuation')
out.mkdir()
results=[]
for case in ('case30','case118','case300'):
    for method in ('NGT','E-NGT'):
        for variant in ('paper','modified'):
            print(json.dumps({'action':'start','case':case,'method':method,'variant':variant}),flush=True)
            result=subprocess.run([sys.executable,str(snapshot/'reports/compare_ngt_variants.py'),
                     '--data-root',str(a.root/'data'),'--output-root',str(out),'--case',case,
                     '--method',method,'--variant',variant,'--epochs','30','--pool','1200','--device','cuda'],cwd=snapshot)
            results.append({'case':case,'method':method,'variant':variant,'returncode':result.returncode})
            print(json.dumps({'action':'finish',**results[-1]}),flush=True)
with (out/'matrix.json').open('x') as f:json.dump(results,f,indent=2)
print('DIAGNOSTIC_MATRIX_FINISHED',flush=True)
