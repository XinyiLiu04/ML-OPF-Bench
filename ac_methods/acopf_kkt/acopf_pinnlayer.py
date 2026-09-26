"""Physical AC KKT residual in rectangular voltages and full generation."""
import numpy as np
import torch
from torch import nn
from acopf_densecorenetwork import DenseCoreNetwork


class PinnLayer(nn.Module):
    def __init__(self, simulation_parameters, neurons_V, neurons_G, neurons_Lg):
        super().__init__()
        p = simulation_parameters
        g = p['general']
        self.n_buses, self.n_gen = g['n_buses'], g['n_gen']
        self.n_gen_non_slack, self.n_branches = g['n_gen_non_slack'], g['n_branches']
        self.n_loads = g['n_loads']
        lookup = g['bus_id_to_idx']
        refs = np.flatnonzero(g['bus_types'] == 3)
        if len(refs) != 1:
            raise ValueError('Exactly one reference bus is required')
        self.slack_bus_idx = int(refs[0])
        def buf(name, value, integer=False):
            self.register_buffer(name, torch.as_tensor(np.asarray(value), dtype=torch.long if integer else torch.float32))
        buf('non_slack_gen_idx', g['non_slack_gen_idx'], True)
        buf('gen_to_bus_idx', [lookup[int(b)] for b in g['gen_bus_ids']], True)
        buf('load_to_bus_idx', [lookup[int(b)] for b in g['load_bus_ids']], True)
        self.core_network = DenseCoreNetwork(2*self.n_loads, self.n_buses, self.n_gen,
                                            self.n_gen_non_slack, self.n_branches,
                                            neurons_V, neurons_G, neurons_Lg)
        gen, bus, br = p['generator'], p['bus'], p['branch']
        for key in ('pg_min', 'pg_max', 'qg_min', 'qg_max', 'cost_c1', 'cost_c2'):
            buf(key, gen[key].flatten())
        for key in ('vm_min', 'vm_max', 'gs', 'bs', 'pd_base', 'qd_base'):
            buf(key, bus[key].flatten())
        fi = np.array([lookup[int(b)] for b in br['f_bus']], dtype=int)
        ti = np.array([lookup[int(b)] for b in br['t_bus']], dtype=int)
        buf('f_idx', fi, True); buf('t_idx', ti, True)
        for key in ('angmin_rad', 'angmax_rad'):
            buf(key, br[key])
        rate = np.asarray(br['rate_a'])
        buf('rate_sq', np.where(np.isfinite(rate) & (rate > 0), rate**2, 0))
        buf('rated', (np.isfinite(rate) & (rate > 0)).astype(float))
        # MATPOWER pi model: charging at both ends; complex tap at from end.
        series = 1 / (np.asarray(br['r_pu']) + 1j*np.asarray(br['x_pu']))
        tap = np.where(np.asarray(br['tap_ratio']) == 0, 1, br['tap_ratio']) * np.exp(1j*np.asarray(br['shift_rad']))
        charging = 1j*np.asarray(br['b_pu'])/2
        yf = np.zeros((self.n_branches, self.n_buses), complex)
        yt = np.zeros_like(yf)
        idx = np.arange(self.n_branches)
        yf[idx, fi] += (series+charging)/(abs(tap)**2)
        yf[idx, ti] -= series/np.conj(tap)
        yt[idx, fi] -= series/tap
        yt[idx, ti] += series+charging
        for name, a in [('yf', yf), ('yt', yt)]:
            buf(name+'_r', a.real); buf(name+'_i', a.imag)
        cg = np.zeros((self.n_buses, self.n_gen))
        cg[[lookup[int(b)] for b in g['gen_bus_ids']], np.arange(self.n_gen)] = 1
        buf('cg', cg)

    def physical_terms(self, v, generation, inputs):
        n = self.n_buses
        vr, vi = v[:, :n], v[:, n:]
        pg, qg = generation[:, :self.n_gen], generation[:, self.n_gen:]
        vm = torch.sqrt(vr.square()+vi.square()+1e-16)
        def flows(yr, yi, indices):
            ir, ii = vr@yr.T-vi@yi.T, vr@yi.T+vi@yr.T
            return vr[:, indices]*ir+vi[:, indices]*ii, vi[:, indices]*ir-vr[:, indices]*ii
        pf,qf = flows(self.yf_r,self.yf_i,self.f_idx)
        pt,qt = flows(self.yt_r,self.yt_i,self.t_idx)
        pinj,qinj = vr.new_zeros(vr.shape), vr.new_zeros(vr.shape)
        for indices,power,reactive in [(self.f_idx,pf,qf),(self.t_idx,pt,qt)]:
            pinj = pinj.index_add(1,indices,power)
            qinj = qinj.index_add(1,indices,reactive)
        pd = self.pd_base.unsqueeze(0).expand(vr.shape[0],-1).index_copy(1,self.load_to_bus_idx,inputs[:,:self.n_loads])
        qd = self.qd_base.unsqueeze(0).expand(vr.shape[0],-1).index_copy(1,self.load_to_bus_idx,inputs[:,self.n_loads:])
        # Net branch export plus shunts equals generation minus demand.
        balance = torch.cat((pinj+self.gs*vm.square()-pg@self.cg.T+pd,
                             qinj-self.bs*vm.square()-qg@self.cg.T+qd),dim=1)
        angle = torch.atan2(vi,vr)
        delta = angle[:,self.f_idx]-angle[:,self.t_idx]
        inequalities = {
            'mu_g_u': torch.cat((pg-self.pg_max,qg-self.qg_max),dim=1),
            'mu_g_d': torch.cat((self.pg_min-pg,self.qg_min-qg),dim=1),
            'mu_v_u': vm-self.vm_max, 'mu_v_d': self.vm_min-vm,
            'mu_sm_fr': (pf.square()+qf.square()-self.rate_sq)*self.rated,
            'mu_sm_to': (pt.square()+qt.square()-self.rate_sq)*self.rated,
            'mu_ang_u': torch.where(self.angmax_rad < 2*np.pi-1e-6,delta-self.angmax_rad,torch.zeros_like(delta)),
            'mu_ang_d': torch.where(self.angmin_rad > -2*np.pi+1e-6,self.angmin_rad-delta,torch.zeros_like(delta)),
        }
        cost = (self.cost_c2*pg.square()+self.cost_c1*pg).sum(1)
        return balance, inequalities, cost

    def residual_components(self, outputs, inputs):
        training_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            v, gen = outputs['v_rect'], outputs['pg_qg']
            if not v.requires_grad:
                v = v.detach().requires_grad_(True)
            if not gen.requires_grad:
                gen = gen.detach().requires_grad_(True)
            balance, constraints, cost = self.physical_terms(v,gen,inputs)
            ref = v[:,self.n_buses+self.slack_bus_idx]
            lagrangian = cost+(outputs['lambda_p']*balance).sum(1)+outputs['lambda_ref'].flatten()*ref
            primal = balance.abs().sum(1)+ref.abs()
            complementarity = torch.zeros_like(cost)
            dual_feasibility = torch.zeros_like(cost)
            for name,c in constraints.items():
                mu = outputs[name]
                lagrangian = lagrangian+(mu*c).sum(1)
                primal = primal+c.relu().sum(1)
                complementarity = complementarity+(mu*c).abs().sum(1)
                dual_feasibility = dual_feasibility+(-mu).relu().sum(1)
            dv,dg = torch.autograd.grad(lagrangian.sum(),(v,gen),create_graph=training_graph)
            result = {'primal':primal,'stationarity':dv.abs().sum(1)+dg.abs().sum(1),
                      'complementarity':complementarity,'dual_feasibility':dual_feasibility}
        return result if training_graph else {k:a.detach() for k,a in result.items()}

    def forward(self, inputs):
        outputs = self.core_network(inputs)
        outputs['kkt_error'] = sum(self.residual_components(outputs,inputs).values())
        return outputs
