import json
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from ml_opf_bench.config import Experiment,dataset_paths
from ml_opf_bench.registry import load_method
from ml_opf_bench.runtime import managed_run,training_indices
parser=argparse.ArgumentParser()
parser.add_argument('--float64',action='store_true')
args=parser.parse_args()
spec=Experiment('ac','RL',case='case300',variant='ddpg-pgonly',device='cpu')
module,_,_=load_method('ac','RL',spec.variant)
root=Path('runs/corrections-v1/ddpg-pgonly-case300-pilot2048-v1')
state=torch.load(root.with_suffix('.pt'),weights_only=False)
p=dataset_paths(Path.cwd(),'ac','case300')
X,_,_,raw,_=module.load_and_scale_acopf_data(str(p.data_path),state.params,fit_scalers=False,scalers=state.artifacts['scalers'])
with managed_run(spec):
 _,val,test=training_indices(len(X),12000,42)
assert not np.intersect1d(val,test).size
case=module.load_case_from_csv(p.case_name,str(p.params_path))
from pgonly_reward import constraint_components,feasible
from ac_configuration.acopf_pypower import solve_pf_setpoints
if args.float64:
 from ac_configuration.acopf_data_setup import extract_id
 def read(kind):
  frame=pd.read_csv(p.data_path.parent/f'{p.case_name}_{kind}.csv')
  cols=sorted([c for c in frame.columns if c.startswith(kind)],key=extract_id)
  return frame[cols].to_numpy(dtype=np.float64)
 raw['x']=np.concatenate([read('pd'),read('qd')],axis=1)
 raw['pg_non_slack']=read('pg')[:,state.params['general']['non_slack_gen_idx']]
 frame=pd.read_csv(p.data_path.parent/f'{p.case_name}_vm.csv')
 raw['vm_gen']=frame[[f'vm_{b}' for b in state.params['general']['gen_bus_ids']]].to_numpy(dtype=np.float64)
val=val[:32]
g=state.params['general']
lookup={int(b):i for i,b in enumerate(case['bus'][:,0])}
genbus=np.array([lookup[int(b)] for b in case['gen'][:,0]])
fixed=case['gen'][:,5]
out={'scope':'validation-only oracle diagnostic; no label action used for training','raw_precision':'float64' if args.float64 else 'float32','fixed_vm_min':float(fixed.min()),'fixed_vm_max':float(fixed.max()),'fixed_vm_bound_violations':int(np.sum((fixed<case['bus'][genbus,12])|(fixed>case['bus'][genbus,11]))),'variants':{}}
for name in ('label_pg_fixed_vm','label_pg_label_vm'):
 rows=[]
 for row in val:
  vm=fixed if name=='label_pg_fixed_vm' else raw['vm_gen'][row]
  result=solve_pf_setpoints(raw['x'][row,:g['n_loads']],raw['x'][row,g['n_loads']:],raw['pg_non_slack'][row],vm,state.params,case)
  record={'index':int(row),'pf_converged':bool(result[0]['success']),'feasible':False}
  if record['pf_converged']:
   parts=constraint_components(result[0],g['BASE_MVA'])
   record.update(violations=parts,feasible=feasible(parts))
  rows.append(record)
 out['variants'][name]=rows
 print(name,'PF',sum(r['pf_converged'] for r in rows),'feasible',sum(r['feasible'] for r in rows),flush=True)
version='float64-v1' if args.float64 else 'v1'
path=Path(f'runs/corrections-v1/ddpg-pgonly-case300-fixed-vm-oracle-{version}.json')
with path.open('x') as f:json.dump(out,f,indent=2)
print({k:v for k,v in out.items() if k!='variants'})
