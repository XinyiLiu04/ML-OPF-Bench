"""Compare four authorized diagnostic OPF solutions with untouched existing labels."""
import json,sys,re
from functools import partial
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'ac_methods'),str(ROOT/'ac_methods/acopf_kkt')]
import reports.audit_ac_kkt_labels as audit
from ml_opf_bench.config import dataset_paths
from ml_opf_bench.io import code_version
Original=audit.PinnLayer
captured={}
class Capture(Original):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs,dtype=torch.float64)
    def residual_components(self,o,x):
        captured.update(layer=self,o=o,x=x)
        return super().residual_components(o,x)

def stationarity(layer,o,x,flow_bounds=None):
    v=o['v_rect'].detach().requires_grad_(True);g=o['pg_qg'].detach().requires_grad_(True)
    bal,cons,cost=layer.physical_terms(v,g,x)
    lag=cost+(o['lambda_p']*bal).sum(1)
    for k,c in cons.items():lag=lag+(o[k]*c).sum(1)
    if flow_bounds is not None:
        voltage=torch.complex(v[:,:layer.n_buses],v[:,layer.n_buses:])
        sf=voltage[:,layer.f_idx]*torch.conj(voltage@torch.complex(layer.yf_r,layer.yf_i).T)
        st=voltage[:,layer.t_idx]*torch.conj(voltage@torch.complex(layer.yt_r,layer.yt_i).T)
        params=flow_bounds['params'];pos={int(b):j for j,b in enumerate(params['general']['branch_ids'])}
        for bound in flow_bounds['bounds']:
            bid,f,t=bound['arc'];j=pos[bid]
            power=sf if f==params['branch']['f_bus'][j] else st
            value=power.real[:,j] if bound['kind']=='p' else power.imag[:,j]
            lag=lag+(-bound['dual_upper']-bound['dual_lower'])*value
    dv,dg=torch.autograd.grad(lag.sum(),(v,g))
    n=layer.n_buses;vr=v[:,:n];vi=v[:,n:];vm=torch.hypot(vr,vi)
    mag=(dv[:,:n]*vr+dv[:,n:]*vi)/vm
    ang=-dv[:,:n]*vi+dv[:,n:]*vr
    keep=np.arange(n)!=layer.slack_bus_idx
    return {'generation_max':float(dg.abs().max()),'magnitude_max':float(mag.abs().max()),
            'nonreference_angle_max':float(ang[:,keep].abs().max()),'balance_max':float(bal.detach().abs().max()),
            'thermal_complementarity':float(sum((o[k]*cons[k]).abs().sum() for k in ('mu_sm_fr','mu_sm_to')).detach()),
            'active_angle_count':int(sum((cons[k].detach().abs()<1e-5).sum() for k in ('mu_ang_u','mu_ang_d')))}

def main():
    audit.PinnLayer=Capture
    audit.load_parameters_from_csv=partial(audit.load_parameters_from_csv,dtype='float64')
    maps=json.loads((ROOT/'ac_methods/ac_configuration/thermal_dual_order.json').read_text())
    records=json.loads((ROOT/'runs/corrections-v1/diagnostic-pairs-output-v2.json').read_text())['records']
    inputs=json.loads((ROOT/'runs/corrections-v1/diagnostic-pairs-input.json').read_text())
    results=[]
    for rec,inp in zip(records,inputs):
        c,s,i=rec['case'],rec['scenario'],rec['index'];paths=dataset_paths(ROOT,'ac',c,s)
        audit.audit(c,s,i+1,maps[paths.case_name])
        layer=captured['layer'];old={k:v[i:i+1] for k,v in captured['o'].items()};x=captured['x'][i:i+1]
        params=audit.load_parameters_from_csv(paths.case_name,paths.params_path)
        bus=[str(i) for i in params['general']['bus_ids']];gen=sorted(rec['solution']['gen'],key=int)
        sol=rec['solution']
        bounds={name:{re.fullmatch(r'JuMP\.Containers\.DenseAxisArrayKey\{Tuple\{Int64\}\}\(\((\d+),\)\)',k).group(1):v for k,v in values.items()} for name,values in rec['bounds'].items()}
        t=lambda v:torch.as_tensor(np.asarray(v)[None,:],dtype=torch.float64)
        vm=np.array([sol['bus'][b]['vm'] for b in bus]);va=np.array([sol['bus'][b]['va'] for b in bus])
        new={k:v.clone() for k,v in old.items()}
        new['v_rect']=t(np.r_[vm*np.cos(va),vm*np.sin(va)])
        new['pg_qg']=t([sol['gen'][g][key] for key in ('pg','qg') for g in gen])
        new['lambda_p']=t([-sol['bus'][b][key] for key in ('lam_kcl_r','lam_kcl_i') for b in bus])
        for side,j,sign in [('u',1,-1),('d',0,1)]:
            new['mu_g_'+side]=t([sign*bounds[k][g][j] for k in ('pg','qg') for g in gen])
            new['mu_v_'+side]=t([sign*bounds['vm'][b][j] for b in bus])
        pos={int(b):j for j,b in enumerate(params['general']['branch_ids'])}
        for row in rec['thermal']:
            bid,f,to=row['arc'];j=pos[bid]
            key='mu_sm_fr' if f==params['branch']['f_bus'][j] else 'mu_sm_to'
            new[key][0,j]=-row['dual']
        primal=('v_rect','pg_qg')
        item={'case':c,'scenario':s,'index':i,'status':rec['status'],'comparisons':{},
              'primal_max_change':{k:float((old[k]-new[k]).abs().max()) for k in primal},
              'dual_max_change':{k:float((old[k]-new[k]).abs().max()) for k in old if k not in primal}}
        for pname,p in [('old',old),('new',new)]:
            for dname,d in [('old',old),('new',new)]:
                combo=d|{k:p[k] for k in primal}
                item['comparisons'][pname+'_primal_'+dname+'_dual']=stationarity(layer,combo,x)
        item['comparisons']['new_pair_with_flow_bound_duals']=stationarity(layer,new,x,{'params':params,'bounds':rec['flow_bounds']})
        item['comparisons']['old_pair_with_flow_bound_duals']=stationarity(layer,old,x,{'params':params,'bounds':rec['flow_bounds']})
        complete={k:v.clone() for k,v in new.items()}
        for bound in rec['flow_bounds']:
            bid,f,to=bound['arc'];j=pos[bid]
            end='f' if f==params['branch']['f_bus'][j] else 't'
            rate=float(params['branch']['rate_a'][j])
            np.testing.assert_allclose([bound['lower'],bound['upper']],[-rate,rate],rtol=1e-12)
            complete[f"mu_{bound['kind']}{end}_u"][0,j]=-bound['dual_upper']
            complete[f"mu_{bound['kind']}{end}_d"][0,j]=bound['dual_lower']
        item['production_full_stationarity']=stationarity(layer,complete,x)
        with torch.no_grad():
            item['production_full_components']={k:float(v.item()) for k,v in Original.residual_components(layer,complete,x).items()}
        for key in ('generation_max','magnitude_max','nonreference_angle_max'):
            assert item['production_full_stationarity'][key] < 1e-5, (c,s,key)
        results.append(item);print(json.dumps(item))
    output=ROOT/'runs/corrections-v1/diagnostic-pairs-comparison-v3.json'
    with output.open('x') as f:json.dump({'source':code_version(),'records':results},f,indent=2)
if __name__=='__main__':main()
