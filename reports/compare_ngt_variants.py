"""Bounded validation-only comparison, separate from formal paper results."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone
import importlib
from pathlib import Path
import traceback
import numpy as np
import torch
from ml_opf_bench.config import Experiment,dataset_paths
from ml_opf_bench.registry import training_call
from ml_opf_bench.runtime import managed_run
from ml_opf_bench.io import code_version,data_signature,write_json,environment
from ml_opf_bench.ac_evaluation import evaluate_ac


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--method',choices=['NGT','E-NGT'],required=True)
    p.add_argument('--variant',choices=['paper','modified'],required=True)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--pool',type=int,default=1200)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    spec=Experiment('ac',a.method,case=a.case,seed=42,pool_size=a.pool,epochs=a.epochs,
                    device=a.device,evaluate_shifts=False,eval_limit=None)
    out=a.output_root/(a.case+'-'+a.method.lower()+'-'+a.variant)/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True,exist_ok=False)
    paths=dataset_paths(a.data_root,'ac',a.case)
    grad_norms=[]
    with (out/'run.log').open('x') as log,redirect_stdout(log),redirect_stderr(log):
        try:
            _,function,kwargs=training_call(spec,paths)
            if a.variant=='paper':
                module=importlib.import_module('paper_unsupervised_acopf' if a.method=='NGT' else 'paper_semi_supervised_acopf')
                function=getattr(module,'train_deepopf_ngt' if a.method=='NGT' else 'train_extended_deepopf_ngt')
            write_json(out/'manifest.json',{'purpose':'validation-only diagnostic; not formal results','experiment':spec.as_dict(),
                        'variant':a.variant,'source':code_version(),'data':data_signature(paths),'environment':environment(),
                        'kwargs':kwargs})
            original_step=torch.optim.Adam.step
            def checked_step(optimizer,*args,**kw):
                gradients=[v.grad for group in optimizer.param_groups for v in group['params'] if v.grad is not None]
                if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError('Nonfinite or missing training gradients')
                grad_norms.append(float(torch.sqrt(sum(g.detach().square().sum() for g in gradients))))
                result=original_step(optimizer,*args,**kw)
                if any(not torch.isfinite(v).all() for group in optimizer.param_groups for v in group['params']):
                    raise FloatingPointError('Nonfinite parameters')
                return result
            torch.optim.Adam.step=checked_step
            try:
                with managed_run(spec) as ctx:state=function(**kwargs)
            finally:
                torch.optim.Adam.step=original_step
            np.savez_compressed(out/'splits.npz',train=ctx.indices[0],val=ctx.indices[1],test=ctx.indices[2])
            torch.save(state,out/'checkpoint.pt')
            metrics,arrays=evaluate_ac(spec,state,paths,indices=ctx.indices[1])
            write_json(out/'validation.json',metrics)
            from algebraic_power_flow import AlgebraicPowerFlow
            from deepopf_ngt_common import LossTerms
            engine=AlgebraicPowerFlow(state.params,a.device)
            raw=arrays['x_true']
            x=torch.as_tensor(state.artifacts['scalers']['x'].transform(raw),dtype=torch.float32,device=a.device)
            demand=torch.as_tensor(raw,dtype=torch.float32,device=a.device)
            n=state.params['general']['n_loads']
            state.model.eval()
            with torch.no_grad():
                vm,va=state.artifacts['denorm'](state.model(x))
                direct=engine(vm,va,demand[:,:n],demand[:,n:])
                terms=LossTerms(state.params,engine,a.device)(direct)
            write_json(out/'validation_direct.json',{k:float(v) for k,v in terms.items()})
            np.savez_compressed(out/'validation_samples.npz',**arrays)
            write_json(out/'completed.json',{'epochs':ctx.epochs_completed,'steps':len(grad_norms),
                       'gradient_norm_max':max(grad_norms),'finite_gradients':True,'train_time_s':state.train_time_s})
        except Exception as error:
            write_json(out/'failed.json',{'error':str(error),'traceback':traceback.format_exc()})
            raise
    print(out,flush=True)

if __name__=='__main__':main()
