"""Float64 physical inputs; neural observations remain float32 at policy entry."""
from pathlib import Path
import numpy as np
import pandas as pd
from ac_configuration.acopf_data_setup import load_and_scale_acopf_data as original_loader, extract_id, parse_case_name


def load_and_scale_acopf_data(data_path, params, **kwargs):
    X,Y,scalers,raw,cost=original_loader(data_path,params,**kwargs)
    folder=Path(data_path).parent;name=parse_case_name(data_path)
    def read(kind):
        frame=pd.read_csv(folder/f'{name}_{kind}.csv')
        columns=sorted((c for c in frame if c.startswith(kind)),key=extract_id)
        return frame[columns].to_numpy(dtype=np.float64)
    raw['x']=np.concatenate([read('pd'),read('qd')],axis=1)
    raw['pg']=read('pg');raw['qg']=read('qg')
    general=params['general']
    for kind in ('vm','va'):
        frame=pd.read_csv(folder/f'{name}_{kind}.csv')
        raw[kind]=frame[[f'{kind}_{b}' for b in general['bus_ids']]].to_numpy(dtype=np.float64)
    raw['pg_non_slack']=raw['pg'][:,general['non_slack_gen_idx']]
    raw['vm_gen']=raw['vm'][:,[general['bus_id_to_idx'][int(b)] for b in general['gen_bus_ids']]]
    return scalers['x'].transform(raw['x']),Y,scalers,raw,cost
