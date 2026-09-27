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
splits=np.load(a.attempt/'splits.npz');idx=splits['val'][:32]
assert not np.intersect1d(idx,splits['test']).size
model=state.model;model.eval()
with torch.no_grad():
 dist=model.get_distribution(torch.as_tensor(X[idx],dtype=torch.float32))
 means=dist.distribution.mean.cpu().numpy()
 actions,_=model.predict(X[idx],deterministic=True)
bounds=state.artifacts['bounds'];n=len(bounds['pg_min']);g=state.params['general'];b=load_case_from_csv(paths.case_name,str(paths.params_path))
variants={'policy':actions,'midpoint':np.full_like(actions,0.5),'pg_lower_vm_mid':np.concatenate([np.zeros_like(actions[:,:n]),np.full_like(actions[:,n:],0.5)],axis=1)}
out={'case':spec.case,'scope':'first32 validation samples only; diagnostic controls not trained methods','indices':idx.tolist(),'source_sha256':m['code']['source_sha256'],'raw_mean_min':float(means.min()),'raw_mean_max':float(means.max()),'raw_mean_outside_fraction':float(np.mean((means<0)|(means>1))),'policy_pg_lower_fraction':float(np.mean(actions[:,:n]<=0)),'policy_pg_upper_fraction':float(np.mean(actions[:,:n]>=1)),'variants':{}}
for name,acts in variants.items():
 rows=[]
 for row,act in zip(idx,acts):
  pg,vm=module.action_to_setpoints(act,bounds)
  result=solve_pf_setpoints(raw['x'][row,:g['n_loads']],raw['x'][row,g['n_loads']:],pg,vm,state.params,b)
  ok=bool(result[0]['success']);record={'index':int(row),'pf_converged':ok}
  if ok:
   gen=result[0]['gen'];bus=result[0]['bus'];br=result[0]['branch'];base=g['BASE_MVA']
   parts={'pg_pu':float(np.maximum(gen[:,9]-gen[:,1],0).sum()/base+np.maximum(gen[:,1]-gen[:,8],0).sum()/base),'qg_pu':float((np.maximum(gen[:,4]-gen[:,2],0)+np.maximum(gen[:,2]-gen[:,3],0)).sum()/base),'vm_pu':float((np.maximum(bus[:,12]-bus[:,7],0)+np.maximum(bus[:,7]-bus[:,11],0)).sum())}
   parts['legacy_total']=float(-module.compute_penalty_from_pf(result,base))
   parts['thermal_relative']=parts['legacy_total']-parts['pg_pu']-parts['qg_pu']-parts['vm_pu']
   record.update(violations=parts,cost=float(module.compute_cost_from_pf(result,base,state.params['generator']['cost_c2'],state.params['generator']['cost_c1'],state.params['generator']['cost_c0'])))
  rows.append(record)
 out['variants'][name]=rows
write_json(a.output,out)
print(json.dumps({k:sum(x['pf_converged'] for x in v) for k,v in out['variants'].items()}))
