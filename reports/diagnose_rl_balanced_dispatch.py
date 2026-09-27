"""Bounded validation-only diagnostics for a trusted frozen RL checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ml_opf_bench.registry import load_method
from ml_opf_bench.config import Experiment, dataset_paths
from ml_opf_bench.io import code_version, data_signature, write_json

p=argparse.ArgumentParser()
p.add_argument('--attempt',type=Path,required=True)
p.add_argument('--data-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
m=json.loads((a.attempt/'manifest.json').read_text())
assert m['code']['source_sha256']==code_version()['source_sha256']
spec=Experiment(**m['experiment'])
assert spec.method=='RL' and (a.attempt/'completed.json').exists()
module,_,_=load_method('ac','RL')
state=torch.load(a.attempt/'checkpoint.pt',map_location='cpu',weights_only=False)
paths=dataset_paths(a.data_root,'ac',spec.case,'base')
assert data_signature(paths)==json.loads((a.attempt/'base_data_manifest.json').read_text())
from ac_configuration.acopf_data_setup import load_and_scale_acopf_data
from ac_configuration.acopf_pypower import load_case_from_csv,solve_pf_setpoints
X,_,_,raw,_=load_and_scale_acopf_data(str(paths.data_path),state.params,fit_scalers=False,scalers=state.artifacts['scalers'])
splits=np.load(a.attempt/'splits.npz');idx=np.concatenate([splits['train'][:32],splits['val'][:32]])
assert not np.intersect1d(idx,splits['test']).size
from ac_configuration.acopf_pypower import build_mpc_for_sample
bounds=state.artifacts['bounds'];g=state.params['general'];b=load_case_from_csv(paths.case_name,str(paths.params_path))
lo=b['gen'][:,9]/g['BASE_MVA'];hi=b['gen'][:,8]/g['BASE_MVA']
ns=g['non_slack_gen_idx']
out={'case':spec.case,'source_sha256':m['code']['source_sha256'],'scope':'first32 train and first32 validation; no labels used for actions','variants':{}}
for margin in (0.0,0.05):
 for voltage in ('csv','midpoint'):
  rows=[]
  for row in idx:
   pd=raw['x'][row,:g['n_loads']];qd=raw['x'][row,g['n_loads']:]
   base=build_mpc_for_sample(pd,qd,lo[ns],state.params,b)
   demand=float(base['bus'][:,2].sum()/g['BASE_MVA'])
   target=demand*(1+margin)
   alpha=(target-lo.sum())/(hi-lo).sum()
   if not 0<=alpha<=1:
    raise ValueError(f'Aggregate dispatch outside capacity: {alpha}')
   pg=(lo+alpha*(hi-lo))[ns]
   vm=b['gen'][:,5] if voltage=='csv' else (bounds['vm_min']+bounds['vm_max'])/2
   result=solve_pf_setpoints(pd,qd,pg,vm,state.params,b)
   ok=bool(result[0]['success'])
   record={'index':int(row),'split':'train' if row in splits['train'] else 'val','pf_converged':ok,'demand_pu':demand,'dispatch_fraction':float(alpha)}
   if ok:
    record['legacy_violation_sum']=float(-module.compute_penalty_from_pf(result,g['BASE_MVA']))
   rows.append(record)
  key=f'margin{margin}/{voltage}'
  out['variants'][key]=rows
  print(key,{split:sum(r['pf_converged'] for r in rows if r['split']==split) for split in ('train','val')},flush=True)
write_json(a.output,out)
