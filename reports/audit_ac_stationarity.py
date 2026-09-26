"""Independent decomposition of stationarity at existing primal/dual labels."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'ac_methods'),str(ROOT/'ac_methods/acopf_kkt')]
import reports.audit_ac_kkt_labels as audit
import torch,numpy as np
from scipy.optimize import nnls
from functools import partial
from ml_opf_bench.io import code_version
audit.load_parameters_from_csv=partial(audit.load_parameters_from_csv,dtype="float64")
Original=audit.PinnLayer
records=[]
class Probe(Original):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs,dtype=torch.float64)
 def residual_components(self,o,x):
  with torch.enable_grad():
   v=o['v_rect'].detach().requires_grad_(True); g=o['pg_qg'].detach().requires_grad_(True)
   bal,cons,cost=self.physical_terms(v,g,x)
   lag=cost+(o['lambda_p']*bal).sum(1)
   for k,c in cons.items():lag=lag+(o[k]*c).sum(1)
   dv,dg=torch.autograd.grad(lag.sum(),(v,g))
   vr,vi=v.detach().numpy()[:,:self.n_buses],v.detach().numpy()[:,self.n_buses:]
   polar_a=-vi*dv[:,:self.n_buses].numpy()+vr*dv[:,self.n_buses:].numpy()
   polar_m=(vr*dv[:,:self.n_buses].numpy()+vi*dv[:,self.n_buses:].numpy())/np.hypot(vr,vi)
   projected=[];active=[];mus=[]
   for row in range(len(v)):
    cols=[]
    for key,sign in [('mu_ang_u',1),('mu_ang_d',-1)]:
     for j in np.flatnonzero(np.abs(cons[key][row].detach().numpy())<1e-5):
      a=np.zeros(self.n_buses);a[int(self.f_idx[j])]=sign;a[int(self.t_idx[j])]-=sign;cols.append(a)
    keep=np.arange(self.n_buses)!=self.slack_bus_idx
    if cols:
     mat=np.array(cols).T; mu,_=nnls(mat[keep],-polar_a[row,keep]);res=polar_a[row]+mat@mu;mus.append(float(mu.max()))
    else:res=polar_a[row];mus.append(0.)
    projected.append(float(np.abs(res[keep]).max()));active.append(len(cols))
   records.append({'generation_max':float(dg.abs().max()),'voltage_magnitude_max':float(abs(polar_m).max()),'angle_max':float(abs(polar_a).max()),'angle_after_missing_duals_max':max(projected),'angle_after_missing_duals_median':float(np.median(projected)),'active_angle_max':max(active),'inferred_angle_mu_max':max(mus),'magnitude_largest_bus':int(abs(polar_m).max(0).argmax())})
  return super().residual_components(o,x)
def main():
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--count',type=int,default=200)
 args=parser.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 audit.PinnLayer=Probe
 evidence=[]
 for c,s,n in [('case30','base','case30'),('case118','base','case118'),('case300','base','case300'),('case118','heavier_loads','api')]:
  maps=json.loads((ROOT/'ac_methods/ac_configuration/thermal_dual_order.json').read_text())
  stem='pglib_opf_'+c+'_ieee'+('__api' if s=='heavier_loads' else '')
  result=audit.audit(c,s,args.count,maps[stem])
  evidence.append(result)
  records[-1].update(case=c,scenario=s,samples=result['samples'])
  print(json.dumps(records[-1]))
 args.output.write_text(json.dumps({'source':code_version(),'stationarity':records,'label_replay':evidence,
  'note':'Read-only float64 label audit. Missing angle multipliers fitted only on near-active bounds for diagnostic purposes, never written as labels. Missing reference multiplier cannot explain non-reference angle residuals. No OPF solves.'},indent=2))

if __name__=='__main__':main()
