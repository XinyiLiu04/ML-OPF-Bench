"""Read-only KKT diagnostics on existing labels; never solve or regenerate labels."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from ml_opf_bench.config import dataset_paths
from ml_opf_bench.io import code_version, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ac_methods'), str(ROOT/'ac_methods/acopf_kkt')]
from ac_configuration.acopf_data_setup import load_parameters_from_csv
from acopf_pinnlayer import PinnLayer


def audit(case, scenario, count):
    paths = dataset_paths(ROOT, 'ac', case, scenario)
    params = load_parameters_from_csv(paths.case_name, paths.params_path)
    files = {}
    def read(path):
        files[str(path.relative_to(ROOT))] = sha256(path)
        return pd.read_csv(path, nrows=count)
    raw = {k: read(paths.data_path.with_name(paths.case_name+'_'+k+'.csv'))
           for k in ('pg','qg','pd','qd','vm','va')}
    ids = [int(c.replace('pd','').replace('_','')) for c in raw['pd'].columns]
    params['general']['load_bus_ids'] = np.array(ids)
    params['general']['n_loads'] = len(ids)
    busids = params['general']['bus_ids']
    vm = raw['vm'][[f'vm_{i}' for i in busids]].to_numpy()
    va = raw['va'][[f'va_{i}' for i in busids]].to_numpy()
    layer = PinnLayer(params,[2],[2],[2]).double()
    tensor = lambda a: torch.as_tensor(a,dtype=torch.double)
    def dual(name, target):
        df=read(paths.duals_path/(paths.case_name+'_'+name+'.csv'))
        return df[[name+'_'+str(i) for i in target]].to_numpy()
    gids = sorted(int(c.rsplit('_',1)[1]) for c in raw['pg'].columns)
    bids = params['general']['branch_ids']
    o = {'v_rect':tensor(np.hstack((vm*np.cos(va),vm*np.sin(va)))),
         'pg_qg':tensor(np.hstack((raw['pg'].to_numpy(),raw['qg'].to_numpy()))),
         'lambda_p':tensor(-np.hstack((dual('lambda_kcl_r',busids),dual('lambda_kcl_i',busids)))),
         'lambda_ref':tensor(np.zeros((len(vm),1))),
         'mu_g_u':tensor(-np.hstack((dual('mu_pg_max',gids),dual('mu_qg_max',gids)))),
         'mu_g_d':tensor(np.hstack((dual('mu_pg_min',gids),dual('mu_qg_min',gids))))}
    for out,name,sign in [('mu_v_u','mu_vm_max',-1),('mu_v_d','mu_vm_min',1),
                          ('mu_sm_fr','mu_sm_fr',-1),('mu_sm_to','mu_sm_to',-1)]:
        o[out]=tensor(sign*dual(name,bids if name.startswith('mu_sm') else busids))
    for key in ('mu_ang_u','mu_ang_d'):
        o[key]=tensor(np.zeros((len(vm),len(bids))))
    x=tensor(np.hstack((raw['pd'].to_numpy(),raw['qd'].to_numpy())))
    with torch.no_grad():
        parts=layer.residual_components(o,x)
        _,cons,_=layer.physical_terms(o['v_rect'],o['pg_qg'],x)
    result={'case':case,'scenario':scenario,'samples':len(vm),'files':files,
            'note':'Angle and reference dual labels are unavailable; set to zero for this diagnostic. Not an optimality certificate.',
            'components':{k:{'mean':float(v.mean()),'max':float(v.max())} for k,v in parts.items()},
            'complementarity_by_family':{k:float((o[k]*v).abs().sum(1).mean()) for k,v in cons.items()}}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    results=[audit(case,scenario,20) for case,scenario in [('case30','base'),('case118','base'),('case300','base'),('case118','heavier_loads')]]
    write_json(args.output,{'source':code_version(),'results':results})
    print(json.dumps([{k:v for k,v in r.items() if k not in ('files',)} for r in results],indent=2))

if __name__=='__main__':
    main()
