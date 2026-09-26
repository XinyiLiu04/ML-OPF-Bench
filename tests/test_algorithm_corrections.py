from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ac_methods'))
sys.path.insert(0, str(ROOT / 'ac_methods/acopf_ngt'))
sys.path.insert(0, str(ROOT / 'dc_methods'))
sys.path.insert(0, str(ROOT / 'ac_methods/acopf_kkt'))


def test_ngt_reference_and_unbounded_angles():
    from deepopf_ngt_common import DeepOPFNGT, VoltageDenormaliser
    params = {'general': {'zib_mask': [False, False, False], 'bus_types': [1, 3, 1]},
              'bus': {'vm_min': [.9] * 3, 'vm_max': [1.1] * 3}}
    denorm = VoltageDenormaliser(params, 'cpu')
    net = DeepOPFNGT(2, 6, [4], reference_position=denorm.reference_position).double()
    with torch.no_grad():
        net.net[-1].weight.zero_()
        net.net[-1].bias.copy_(torch.tensor([0., 0., 0., -1.2, .2, 1.5]))
    vm, va = denorm(net(torch.zeros(1, 2, dtype=torch.double)))
    torch.testing.assert_close(va, torch.tensor([[-1.4, 0., 1.3]], dtype=torch.double))
    assert bool((vm >= .9).all() and (vm <= 1.1).all())
    encoded = denorm.encode(np.ones((1, 3)), np.array([[-1.2, .2, 1.5]]))
    np.testing.assert_allclose(encoded[:, 3:], va.detach().numpy(), atol=1e-6)
    va.square().sum().backward()
    assert torch.isfinite(net.net[-1].bias.grad).all()


def test_dc_kkt_affine_scaling_preserves_physical_zero_and_sign():
    from dcopf_kkt_pinn import kkt_residual, DUALS
    k = SimpleNamespace(safe_max=np.ones(1), p_min_norm=np.zeros(1), p_max_norm=np.ones(1),
                        gen_bus_map=np.ones((1, 1)), ptdf=np.zeros((1, 1)), rate=np.ones(1),
                        load_scale=1., n_c=1, n_g=1, c1=np.zeros(1), c2=np.zeros(1),
                        flow_per_gen=np.zeros((1, 1)),
                        shift={n: np.array([.4]) for n in DUALS},
                        scale={n: np.array([2.]) for n in DUALS},
                        penalty_scale={n: np.array([1.]) for n in DUALS})
    duals = {n: np.array([[.4]]) for n in DUALS}
    np.testing.assert_allclose(kkt_residual(np.array([[.5]]), duals, np.array([[.5]]), k), 0.)
    duals['mu_g_min'] = np.array([[.2]])  # scaled positive, physical negative
    assert kkt_residual(np.array([[.5]]), duals, np.array([[.5]]), k)[0] > 0


def test_ac_kkt_exact_one_bus_solution_and_backward():
    sys.path.insert(0, str(ROOT / 'ac_methods/acopf_kkt'))
    from acopf_pinnlayer import PinnLayer
    params = {'general': {'n_buses':1,'n_gen':1,'n_gen_non_slack':0,'n_branches':0,'n_loads':1,
                          'bus_id_to_idx':{1:0},'bus_types':np.array([3]),'non_slack_gen_idx':[],
                          'gen_bus_ids':[1],'load_bus_ids':[1]},
              'generator': {k:np.array(v) for k,v in {'pg_min':[0.],'pg_max':[2.],
                  'qg_min':[-1.],'qg_max':[1.],'cost_c1':[3.],'cost_c2':[2.]}.items()},
              'bus':{k:np.array(v) for k,v in {'vm_min':[.9],'vm_max':[1.1],'gs':[0.],'bs':[0.],'pd_base':[.5],'qd_base':[.2]}.items()},
              'branch':{k:np.array([]) for k in ('f_bus','t_bus','r_pu','x_pu','b_pu','tap_ratio','shift_rad',
                                                'rate_a','angmin_rad','angmax_rad')}}
    layer=PinnLayer(params,[2],[2],[2]).double()
    t=lambda v:torch.tensor(v,dtype=torch.double,requires_grad=True)
    o={'v_rect':t([[1.,0.]]),'pg_qg':t([[.5,.2]]),'lambda_p':t([[5.,0.]]),'lambda_ref':t([[0.]])}
    for k,size in [('mu_g_u',2),('mu_g_d',2),('mu_v_u',1),('mu_v_d',1),
                   ('mu_sm_fr',0),('mu_sm_to',0),('mu_ang_u',0),('mu_ang_d',0)]:
        o[k]=t(np.zeros((1,size)))
    parts=layer.residual_components(o,t([[.5,.2]]))
    for value in parts.values():
        torch.testing.assert_close(value,torch.zeros(1,dtype=torch.double),atol=1e-12,rtol=0)
    o['v_rect']=t([[-1.,0.]])
    assert layer.residual_components(o,t([[.5,.2]]))['primal'].item() >= np.pi
    o['v_rect']=t([[1.,0.]])
    o['pg_qg']=t([[.6,.2]])
    sum(layer.residual_components(o,t([[.5,.2]])).values()).sum().backward()
    assert torch.isfinite(o['pg_qg'].grad).all()
    assert o['pg_qg'].grad[0,0] > 0


def test_ngt_asymmetric_and_unlimited_branch_angles():
    from deepopf_ngt_common import LossTerms
    params={'generator':{k:np.array([v]) for k,v in {'pg_min':0.,'pg_max':2.,'qg_min':-2.,
            'qg_max':2.,'cost_c0':0.,'cost_c1':1.,'cost_c2':0.}.items()},
            'general':{'zib_mask':[False,False]},'bus':{'vm_min':[.9,.9],'vm_max':[1.1,1.1]},
            'branch':{'rate_a':[0.],'angmin_rad':[-.1],'angmax_rad':[.3]}}
    engine=SimpleNamespace(f_idx=torch.tensor([0]),t_idx=torch.tensor([1]))
    loss=LossTerms(params,engine,'cpu')
    z=torch.zeros((1,1)); results={k:z for k in ['Pg','Qg','Pd_pred','Pd_demanded','Qd_pred','Qd_demanded']}
    results['v_all']=torch.ones((1,2)); results['theta_all']=torch.tensor([[-.2,0.]],requires_grad=True)
    value=loss(results)['L_theta'];torch.testing.assert_close(value,torch.tensor(.01))
    value.backward();assert results['theta_all'].grad[0,0]<0
    params['branch'].update(angmin_rad=[-2*np.pi],angmax_rad=[2*np.pi])
    results['theta_all']=torch.tensor([[8.,0.]])
    torch.testing.assert_close(LossTerms(params,engine,'cpu')(results)['L_theta'],torch.tensor(0.))


def test_ac_branch_operators_match_pypower():
    sys.path.insert(0, str(ROOT / 'ac_methods/acopf_kkt'))
    from acopf_pinnlayer import PinnLayer
    from ac_configuration.acopf_data_setup import load_parameters_from_csv
    from ac_configuration.acopf_pypower import load_case_from_csv
    from ml_opf_bench.config import dataset_paths
    from pypower.ext2int import ext2int
    from pypower.makeYbus import makeYbus
    for case in ('case30','case118','case300'):
        paths=dataset_paths(ROOT,'ac',case)
        params=load_parameters_from_csv(paths.case_name,paths.params_path)
        layer=PinnLayer(params,[2],[2],[2])
        pp=ext2int(load_case_from_csv(paths.case_name,paths.params_path))
        _,yf,yt=makeYbus(pp['baseMVA'],pp['bus'],pp['branch'])
        np.testing.assert_allclose(layer.yf_r.numpy()+1j*layer.yf_i.numpy(),yf.toarray(),rtol=2e-6,atol=2e-5)
        np.testing.assert_allclose(layer.yt_r.numpy()+1j*layer.yt_i.numpy(),yt.toarray(),rtol=2e-6,atol=2e-5)


def test_ngt_true_voltage_preserves_shunts_and_fixed_loads():
    import pandas as pd
    from algebraic_power_flow import AlgebraicPowerFlow
    from ac_configuration.acopf_data_setup import load_parameters_from_csv
    from ml_opf_bench.config import dataset_paths
    for case,scenario in [('case30','base'),('case118','base'),('case300','base'),('case118','heavier_loads')]:
        paths=dataset_paths(ROOT,'ac',case,scenario)
        params=load_parameters_from_csv(paths.case_name,paths.params_path)
        frames={k:pd.read_csv(paths.data_path.with_name(paths.case_name+'_'+k+'.csv'),nrows=4)
                for k in ('pd','qd','vm','va','pg','qg')}
        ids=[int(c.replace('pd','').replace('_','')) for c in frames['pd'].columns]
        params['general']['load_bus_ids']=np.array(ids)
        params['general']['n_loads']=len(ids)
        engine=AlgebraicPowerFlow(params,'cpu')
        bids=params['general']['bus_ids']
        t=lambda v:torch.tensor(v,dtype=torch.float32)
        vm=t(frames['vm'][[f'vm_{i}' for i in bids]].to_numpy())
        va=t(frames['va'][[f'va_{i}' for i in bids]].to_numpy())
        result=engine(vm[:,engine.nonzib_idx],va[:,engine.nonzib_idx],t(frames['pd'].to_numpy()),t(frames['qd'].to_numpy()))
        torch.testing.assert_close(result['Pd_pred'],result['Pd_demanded'],atol=3e-3,rtol=1e-4)
        torch.testing.assert_close(result['Qd_pred'],result['Qd_demanded'],atol=3e-3,rtol=1e-4)
        torch.testing.assert_close(result['Pg'],t(frames['pg'].to_numpy()),atol=3e-3,rtol=1e-4)
        torch.testing.assert_close(result['Qg'],t(frames['qg'].to_numpy()),atol=3e-3,rtol=1e-4)


def test_direct_branch_metric_matches_pypower_both_ends():
    from ac_configuration.acopf_branch import direct_branch_relative_violation
    from ac_configuration.acopf_data_setup import load_parameters_from_csv
    from ac_configuration.acopf_pypower import load_case_from_csv
    from ml_opf_bench.config import dataset_paths
    from pypower.ext2int import ext2int
    from pypower.makeYbus import makeYbus
    rng = np.random.default_rng(42)
    for case, scenario in [('case30','base'),('case118','base'),('case300','base'),('case118','heavier_loads')]:
        paths = dataset_paths(ROOT,'ac',case,scenario)
        params = load_parameters_from_csv(paths.case_name,paths.params_path)
        pp = ext2int(load_case_from_csv(paths.case_name,paths.params_path))
        _, yf, yt = makeYbus(pp['baseMVA'],pp['bus'],pp['branch'])
        n = len(pp['bus'])
        vm = rng.uniform(.85,1.15,(5,n)); va = rng.uniform(-.5,.5,(5,n))
        v = vm*np.exp(1j*va)
        sf = v[:,pp['branch'][:,0].astype(int)]*np.conj((yf@v.T).T)
        st = v[:,pp['branch'][:,1].astype(int)]*np.conj((yt@v.T).T)
        rate = pp['branch'][:,5]/pp['baseMVA']
        expected = np.maximum(0,np.maximum(abs(sf),abs(st))/rate-1).max(1)
        np.testing.assert_allclose(direct_branch_relative_violation(vm,va,params),expected,rtol=2e-6,atol=2e-6)


def test_thermal_loader_preserves_values_and_corrects_columns():
    import json
    from ac_configuration.acopf_duals import load_thermal_duals, load_dual_by_ids
    from ac_configuration.acopf_data_setup import load_parameters_from_csv
    from ml_opf_bench.config import dataset_paths
    mappings=json.loads((ROOT/'ac_methods/ac_configuration/thermal_dual_order.json').read_text())
    for case,scenario in [('case30','base'),('case118','base'),('case300','base'),('case118','heavier_loads')]:
        paths=dataset_paths(ROOT,'ac',case,scenario)
        params=load_parameters_from_csv(paths.case_name,paths.params_path)
        fixed=load_thermal_duals(paths.duals_path,paths.case_name,params)
        ids=params['general']['branch_ids']; pos={int(b):i for i,b in enumerate(ids)}
        old=[load_dual_by_ids(paths.duals_path,paths.case_name,k,ids) for k in ('mu_sm_fr','mu_sm_to')]
        for j,(bid,f,t) in enumerate(mappings[paths.case_name]['constraint_arcs']):
            end='mu_sm_fr' if f==params['branch']['f_bus'][pos[bid]] else 'mu_sm_to'
            np.testing.assert_array_equal(fixed[end][:,pos[bid]],old[j%2][:,pos[sorted(pos)[j//2]]])


def test_ac_kkt_stationarity_against_independent_powerflow_finite_difference():
    from acopf_pinnlayer import PinnLayer
    from ac_configuration.acopf_data_setup import load_parameters_from_csv
    from ac_configuration.acopf_pypower import load_case_from_csv
    from ml_opf_bench.config import dataset_paths
    from pypower.ext2int import ext2int
    from pypower.makeYbus import makeYbus
    rng=np.random.default_rng(42)
    for case in ('case30','case118','case300'):
        paths=dataset_paths(ROOT,'ac',case)
        p=load_parameters_from_csv(paths.case_name,paths.params_path,dtype='float64')
        layer=PinnLayer(p,[2],[2],[2],dtype=torch.float64).double()
        pp=ext2int(load_case_from_csv(paths.case_name,paths.params_path))
        y,yf,yt=makeYbus(pp['baseMVA'],pp['bus'],pp['branch'])
        n,ng=layer.n_buses,layer.n_gen
        v=rng.uniform(.95,1.05,n)*np.exp(1j*rng.uniform(-.1,.1,n))
        z=np.r_[v.real,v.imag,rng.uniform(.2,.8,2*ng)]
        x=np.r_[p['bus']['pd_base'],p['bus']['qd_base']][None,:]
        o={'v_rect':torch.tensor(z[:2*n][None,:],requires_grad=True),
           'pg_qg':torch.tensor(z[2*n:][None,:],requires_grad=True),
           'lambda_p':torch.tensor(rng.normal(size=(1,2*n))),
           'lambda_ref':torch.tensor([[.7]],dtype=torch.double)}
        balance,cons,cost=layer.physical_terms(o['v_rect'],o['pg_qg'],torch.tensor(x))
        for k,c in cons.items():o[k]=torch.tensor(rng.uniform(.1,1,c.shape))
        lag=cost+(o['lambda_p']*balance).sum(1)
        lag=lag+.7*torch.atan2(o['v_rect'][:,n+layer.slack_bus_idx],o['v_rect'][:,layer.slack_bus_idx])
        for k,c in cons.items():lag=lag+(o[k]*c).sum(1)
        dv,dg=torch.autograd.grad(lag.sum(),(o['v_rect'],o['pg_qg']))
        gradient=np.r_[dv.detach().numpy().ravel(),dg.detach().numpy().ravel()]
        def independent(a):
            voltage=a[:n]+1j*a[n:2*n];pg=a[2*n:2*n+ng];qg=a[2*n+ng:]
            s=voltage*np.conj(y@voltage)
            generation=layer.cg.numpy()@(pg+1j*qg)
            bal=s-generation+p['bus']['pd_base']+1j*p['bus']['qd_base']
            sf=voltage[pp['branch'][:,0].astype(int)]*np.conj(yf@voltage)
            st=voltage[pp['branch'][:,1].astype(int)]*np.conj(yt@voltage)
            vm=abs(voltage);angle=np.angle(voltage)
            delta=angle[layer.f_idx.numpy()]-angle[layer.t_idx.numpy()]
            c={'mu_g_u':np.r_[pg-p['generator']['pg_max'].ravel(),qg-p['generator']['qg_max'].ravel()],
               'mu_g_d':np.r_[p['generator']['pg_min'].ravel()-pg,p['generator']['qg_min'].ravel()-qg],
               'mu_v_u':vm-p['bus']['vm_max'],'mu_v_d':p['bus']['vm_min']-vm,
               'mu_sm_fr':abs(sf)**2-p['branch']['rate_a']**2,
               'mu_sm_to':abs(st)**2-p['branch']['rate_a']**2,
               'mu_ang_u':delta-p['branch']['angmax_rad'],'mu_ang_d':p['branch']['angmin_rad']-delta}
            cost=(p['generator']['cost_c2'].ravel()*pg**2+p['generator']['cost_c1'].ravel()*pg).sum()
            return cost+o['lambda_p'].numpy().ravel()@np.r_[bal.real,bal.imag]+.7*angle[layer.slack_bus_idx]+sum(o[k].numpy().ravel()@b for k,b in c.items())
        for _ in range(5):
            direction=rng.normal(size=len(z));direction/=np.linalg.norm(direction)
            step=1e-6
            numerical=(independent(z+step*direction)-independent(z-step*direction))/(2*step)
            np.testing.assert_allclose(gradient@direction,numerical,rtol=2e-5,atol=.01)
        actual=layer.residual_components(o,torch.tensor(x))['stationarity'].item()
        np.testing.assert_allclose(actual,abs(gradient).sum(),rtol=1e-12)
