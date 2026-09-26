from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ac_methods'))
sys.path.insert(0, str(ROOT / 'ac_methods/acopf_ngt'))
sys.path.insert(0, str(ROOT / 'dc_methods'))


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
              'bus':{k:np.array(v) for k,v in {'vm_min':[.9],'vm_max':[1.1],'gs':[0.],'bs':[0.]}.items()},
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
