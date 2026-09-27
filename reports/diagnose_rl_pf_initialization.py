"""Validation-only PF initialization diagnostics for a trusted frozen RL checkpoint."""
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
splits=np.load(a.attempt/'splits.npz');idx=splits['val'][:32]
assert not np.intersect1d(idx,splits['test']).size
model=state.model;model.eval()
with torch.no_grad():
 dist=model.get_distribution(torch.as_tensor(X[idx],dtype=torch.float32))
 means=dist.distribution.mean.cpu().numpy()
 actions,_=model.predict(X[idx],deterministic=True)
bounds=state.artifacts['bounds'];n=len(bounds['pg_min']);g=state.params['general'];b=load_case_from_csv(paths.case_name,str(paths.params_path))
from ac_configuration.acopf_pypower import build_mpc_for_sample,get_ppopt
from pypower.runpf import runpf
from pypower.ppoption import ppoption
out={'case':spec.case,'indices':idx.tolist(),'source_sha256':m['code']['source_sha256'],'scope':'validation only; labels used solely as diagnostic controls','options':{k:get_ppopt()[k] for k in ('PF_ALG','PF_TOL','PF_MAX_IT','ENFORCE_Q_LIMS')},'variants':{}}
for control in ('label','policy','midpoint'):
 for initial in ('csv','flat','label'):
  for max_it in (10,50):
   rows=[]
   for pos,row in enumerate(idx):
    if control=='label':
     pg,vm=raw['pg_non_slack'][row],raw['vm_gen'][row]
    else:
     act=actions[pos] if control=='policy' else np.full_like(actions[pos],0.5)
     pg,vm=module.action_to_setpoints(act,bounds)
    mpc=build_mpc_for_sample(raw['x'][row,:g['n_loads']],raw['x'][row,g['n_loads']:],pg,state.params,b)
    mpc['gen'][:,5]=vm
    if initial=='flat':
     mpc['bus'][:,7]=1.0;mpc['bus'][:,8]=0.0
    elif initial=='label':
     mpc['bus'][:,7]=raw['vm'][row];mpc['bus'][:,8]=np.rad2deg(raw['va'][row])
    result,ok=runpf(mpc,ppoption(get_ppopt(),PF_MAX_IT=max_it))
    record={'index':int(row),'pf_converged':bool(ok)}
    if ok:
     record['vm_label_max_abs']=float(np.max(np.abs(result['bus'][:,7]-raw['vm'][row])))
    rows.append(record)
   key=f'{control}/{initial}/iterations{max_it}'
   out['variants'][key]=rows
   print(key,sum(r['pf_converged'] for r in rows),flush=True)
write_json(a.output,out)
