import json
import argparse
from pathlib import Path
import numpy as np
import torch
from ml_opf_bench.config import Experiment,dataset_paths
from ml_opf_bench.registry import load_method
from ml_opf_bench.runtime import managed_run
parser=argparse.ArgumentParser()
parser.add_argument('--case',required=True,choices=['case30','case118','case300'])
args=parser.parse_args()
output=Path('runs/corrections-v1')/f'ddpg-pgonly-{args.case}-validation-smoke-v3.json'
checkpoint=output.with_suffix('.pt')
if output.exists() or checkpoint.exists():
 raise FileExistsError(output)
spec=Experiment('ac','RL',case=args.case,variant='ddpg-pgonly',device='cpu')
module,fn,_=load_method('ac','RL',spec.variant)
paths=dataset_paths(Path.cwd(),'ac',args.case)
with managed_run(spec) as context:
 state=fn(paths.case_name,str(paths.params_path),str(paths.data_path),n_train_use=12000,
          total_timesteps=128,learning_starts=16,hidden_sizes=[16,16],batch_size=16,device='cpu')
 val=context.indices[1][:16]
 assert not np.intersect1d(val,context.indices[2]).size
X,_,_,raw,_=module.load_and_scale_acopf_data(str(paths.data_path),state.params,fit_scalers=False,scalers=state.artifacts['scalers'])
case=module.load_case_from_csv(paths.case_name,str(paths.params_path))
env=module.AcopfEnv(X,raw['x'],val,state.params,case,state.artifacts['bounds'],module.BoundedSummation(state.params))

torch.save(state,checkpoint)
loaded=torch.load(checkpoint,weights_only=False)
a=state.model.predict(X[val],deterministic=True)[0]
np.testing.assert_array_equal(a,loaded.model.predict(X[val],deterministic=True)[0])
infos=[]
for i,act in enumerate(a):
 env._current_idx=i
 infos.append(env.step(act)[4])
out=dict(scope='128-step software smoke; validation only; not paper evidence',case=args.case,steps=state.artifacts['total_timesteps'],actions=list(a.shape),pf_converged=sum(i['pf_converged'] for i in infos),feasible=sum(i['feasible'] for i in infos),samples=len(infos),checkpoint_equal=True,details=infos)
output.write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='details'}))
