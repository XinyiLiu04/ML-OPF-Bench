"""Bounded validation-only DC RL execution and checkpoint checks."""
import json
from pathlib import Path
import numpy as np
import torch
from ml_opf_bench.config import Experiment,dataset_paths
from ml_opf_bench.registry import load_method
from ml_opf_bench.runtime import managed_run
from ml_opf_bench.dc_evaluation import evaluate_dc
module,fn,_=load_method('dc','RL')
for case in ('case30','case118','case300'):
    output=Path('runs/corrections-v1')/f'dc-rl-{case}-smoke-v1.json'
    if output.exists() or output.with_suffix('.pt').exists():raise FileExistsError(output)
    spec=Experiment('dc','RL',case,device='cpu',eval_limit=16)
    p=dataset_paths(Path.cwd(),'dc',case)
    with managed_run(spec) as context:
        state=fn(p.case_name,str(p.params_path),str(p.data_path),total_timesteps=128,
                 learning_starts=16,hidden_sizes=[16,16],batch_size=16)
        val=context.indices[1][:16]
        assert not np.intersect1d(val,context.indices[2]).size
    torch.save(state,output.with_suffix('.pt'))
    loaded=torch.load(output.with_suffix('.pt'),weights_only=False)
    records,arrays=evaluate_dc(spec,state,p,indices=val)
    _,reloaded=evaluate_dc(spec,loaded,p,indices=val)
    np.testing.assert_array_equal(arrays['RL_pg_pred'],reloaded['RL_pg_pred'])
    loads,_=module.load_samples(str(p.data_path),state.params)
    env=module.DcEnv(state.artifacts['x_scaler'].transform(loads),loads,val,state.params)
    actions=state.model.predict(env.x_scaled,deterministic=True)[0]
    info=[]
    for i,action in enumerate(actions):
        env._current_idx=i;info.append(env.step(action)[4])
    result=dict(case=case,scope='128-step validation-only software check, not paper result',
                checkpoint_equal=True,feasible=sum(r['feasible'] for r in info),samples=len(info),metrics=records)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
