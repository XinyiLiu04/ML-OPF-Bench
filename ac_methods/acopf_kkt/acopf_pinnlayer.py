# -*- coding: utf-8 -*-
"""
ACOPF PINN Layer - PyTorch (Faithful to Paper Eqs. 23a-23d)
============================================================
Core responsibilities:
1. Wrap DenseCoreNetwork
2. Build admittance matrices (Y, Ybr, IM) from branch data
3. Compute KKT error using RECTANGULAR voltage formulation

KKT error components (paper Eq. 23a-23d + primal violations):
  ε_prim  — Power flow equation error  (Eq. 23d adapted)
  ε_gen   — Generator limit primal violations
  ε_vm    — Voltage magnitude primal violations
  ε_line  — Line flow primal violations
  ε_comp  — Complementary slackness (Eq. 23b)
  ε_dual  — Dual feasibility (Eq. 23c)
  ε_stat  — Stationarity condition (Eq. 23a)

Differences from original TF code (Pinn_layer.py):
  - PyTorch tensors instead of TF
  - Admittance matrix built from CSV params (not from create_example_parameters)
  - Adapted for non-slack Pg
  - Includes line flow constraints with mu_sm_fr/to
  - Supports sparse bus numbering
"""

import torch
import torch.nn as nn
import numpy as np
from acopf_densecorenetwork import DenseCoreNetwork


class PinnLayer(nn.Module):
    """
    PINN physics layer: wraps DenseCoreNetwork + KKT error computation.
    """

    def __init__(self, simulation_parameters, device='cpu'):
        super().__init__()
        self.device = device

        # ============ Extract topology ============
        gen = simulation_parameters['general']
        self.n_buses = gen['n_buses']
        self.n_gen = gen['n_gen']
        self.n_gen_non_slack = gen['n_gen_non_slack']
        self.n_branches = gen['n_branches']
        self.n_loads = gen['n_loads']
        self.BASE_MVA = gen['BASE_MVA']

        bus_ids = gen['bus_ids']
        bus_id_to_idx = gen['bus_id_to_idx']
        gen_bus_ids = gen['gen_bus_ids']
        load_bus_ids = gen['load_bus_ids']
        non_slack_gen_idx = gen['non_slack_gen_idx']

        # Slack bus info
        slack_bus_ids = bus_ids[gen['bus_types'] == 3]
        slack_bus_indices = [bus_id_to_idx[int(bid)] for bid in slack_bus_ids]
        self.slack_bus_idx = slack_bus_indices[0] if len(slack_bus_indices) > 0 else 0

        # Index mappings (as tensors for scatter operations)
        self.register_buffer('load_to_bus_idx', torch.tensor(
            [bus_id_to_idx[int(bid)] for bid in load_bus_ids], dtype=torch.long))
        self.register_buffer('gen_to_bus_idx', torch.tensor(
            [bus_id_to_idx[int(bid)] for bid in gen_bus_ids], dtype=torch.long))
        self.register_buffer('non_slack_gen_idx', torch.tensor(
            non_slack_gen_idx, dtype=torch.long))

        # ============ Build core network ============
        input_dim = 2 * self.n_loads
        neurons_V = simulation_parameters['training']['neurons_in_hidden_layers_V']
        neurons_G = simulation_parameters['training']['neurons_in_hidden_layers_G']
        neurons_Lg = simulation_parameters['training']['neurons_in_hidden_layers_Lg']

        self.core_network = DenseCoreNetwork(
            input_dim=input_dim,
            n_buses=self.n_buses,
            n_gen=self.n_gen,
            n_gen_non_slack=self.n_gen_non_slack,
            n_branches=self.n_branches,
            neurons_V=neurons_V,
            neurons_G=neurons_G,
            neurons_Lg=neurons_Lg,
        )

        # ============ Constraint bounds (as buffers) ============
        # Generator bounds — all generators
        pg_min_all = simulation_parameters['generator']['pg_min'].flatten()
        pg_max_all = simulation_parameters['generator']['pg_max'].flatten()
        qg_min_all = simulation_parameters['generator']['qg_min'].flatten()
        qg_max_all = simulation_parameters['generator']['qg_max'].flatten()

        # Non-slack Pg bounds
        self.register_buffer('pg_min_ns', torch.tensor(
            pg_min_all[non_slack_gen_idx], dtype=torch.float32).unsqueeze(0))
        self.register_buffer('pg_max_ns', torch.tensor(
            pg_max_all[non_slack_gen_idx], dtype=torch.float32).unsqueeze(0))
        # Qg bounds (all generators)
        self.register_buffer('qg_min', torch.tensor(
            qg_min_all, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('qg_max', torch.tensor(
            qg_max_all, dtype=torch.float32).unsqueeze(0))

        # Voltage bounds (all buses, squared for rectangular comparison)
        vm_min = simulation_parameters['bus']['vm_min'].astype(np.float32)
        vm_max = simulation_parameters['bus']['vm_max'].astype(np.float32)
        self.register_buffer('vm_min_sq', torch.tensor(vm_min ** 2).unsqueeze(0))
        self.register_buffer('vm_max_sq', torch.tensor(vm_max ** 2).unsqueeze(0))

        # Generation cost (linear term c1 for non-slack Pg, zero for Qg)
        cost_c1 = simulation_parameters['generator']['cost_c1'].astype(np.float32)
        # Stationarity: dL/dPg = c1 + lambda terms + mu terms = 0
        # For non-slack Pg
        cost_c1_ns = cost_c1[non_slack_gen_idx]
        # Combined cost vector: [c1_pg_non_slack, 0_qg]
        cost_vec = np.concatenate([cost_c1_ns, np.zeros(self.n_gen, dtype=np.float32)])
        self.register_buffer('cost_vec', torch.tensor(cost_vec).unsqueeze(0))

        # ============ Build admittance matrices ============
        self._build_admittance_matrices(simulation_parameters, bus_id_to_idx)

        # ============ Build generator-bus mapping matrix ============
        # Map_g: (n_buses, n_gen_non_slack + n_gen) for [pg_non_slack, qg]
        # Used in power balance: P_inj = Map_g @ G - Map_L @ D
        self._build_mapping_matrices(simulation_parameters, bus_id_to_idx)

    def _build_admittance_matrices(self, params, bus_id_to_idx):
        """
        Build Y (bus admittance), Ybr (branch admittance), IM (incidence matrix)
        in rectangular block form for use with v = [Vr; Vi].

        Paper uses: v^T M_p v = p_n  (Eq. 6)
        We compute S = V * conj(Y @ V) in rectangular form.
        """
        n = self.n_buses
        br = params['branch']
        f_bus = br['f_bus']
        t_bus = br['t_bus']
        r_pu = br['r_pu'].astype(np.float64)
        x_pu = br['x_pu'].astype(np.float64)
        b_pu = br['b_pu'].astype(np.float64)
        tap_ratio = br['tap_ratio'].astype(np.float64)
        shift_deg = br['shift_deg'].astype(np.float64)
        rate_a = br['rate_a'].astype(np.float64)
        n_br = len(f_bus)

        # Bus admittance matrix Y (complex)
        Y = np.zeros((n, n), dtype=np.complex128)
        # Branch admittance for line flow: Ybr (2*n_br x 2*n_br diagonal)
        # Incidence matrix IM: (2*n_br x 2*n_buses)
        ybr_diag = np.zeros(n_br, dtype=np.complex128)
        IM_complex = np.zeros((n_br, n), dtype=np.complex128)

        for k in range(n_br):
            i = bus_id_to_idx[int(f_bus[k])]
            j = bus_id_to_idx[int(t_bus[k])]

            z = complex(r_pu[k], x_pu[k])
            y_series = 1.0 / z if abs(z) > 1e-10 else 0.0
            y_shunt = complex(0, b_pu[k])

            tap = tap_ratio[k] if tap_ratio[k] != 0 else 1.0
            shift = shift_deg[k] * np.pi / 180.0
            tap_c = tap * np.exp(1j * shift)

            # Bus admittance matrix
            Y[i, i] += y_series / (tap * np.conj(tap_c)) + y_shunt / 2.0
            Y[j, j] += y_series + y_shunt / 2.0
            Y[i, j] -= y_series / np.conj(tap_c)
            Y[j, i] -= y_series / tap_c

            # Branch admittance and incidence for line flow
            ybr_diag[k] = y_series / tap_c
            IM_complex[k, i] = 1.0
            IM_complex[k, j] = -1.0

        # Store Y as real block form: Y_real (n x n), Y_imag (n x n)
        self.register_buffer('Y_real', torch.tensor(Y.real, dtype=torch.float32))
        self.register_buffer('Y_imag', torch.tensor(Y.imag, dtype=torch.float32))

        # Line flow: Ibr = Ybr @ IM @ V  (in rectangular block form)
        # Ybr_block: (2*n_br x 2*n_br), IM_block: (2*n_br x 2*n)
        # Ybr is diagonal: [[Ybr_r, -Ybr_i], [Ybr_i, Ybr_r]]
        Ybr_r = np.diag(ybr_diag.real)
        Ybr_i = np.diag(ybr_diag.imag)
        Ybr_block = np.block([
            [Ybr_r, -Ybr_i],
            [Ybr_i,  Ybr_r]
        ])
        IM_r = IM_complex.real
        IM_i = IM_complex.imag
        IM_block = np.block([
            [IM_r, -IM_i],
            [IM_i,  IM_r]
        ])

        # Combined: Ybr_IM = Ybr_block @ IM_block  (2*n_br x 2*n)
        Ybr_IM = Ybr_block @ IM_block
        self.register_buffer('Ybr_IM', torch.tensor(Ybr_IM, dtype=torch.float32))

        # Line flow limits (squared): rate_a^2 in p.u.
        # For branches without limits (rate_a = 0 or very large), use a large number
        line_limit_sq = np.zeros(n_br, dtype=np.float32)
        self.branch_has_limit = np.zeros(n_br, dtype=bool)
        for k in range(n_br):
            if rate_a[k] > 0 and rate_a[k] < 9000:
                line_limit_sq[k] = rate_a[k] ** 2
                self.branch_has_limit[k] = True
            else:
                line_limit_sq[k] = 1e10  # effectively no limit
        self.register_buffer('line_limit_sq', torch.tensor(line_limit_sq).unsqueeze(0))
        self.register_buffer('branch_has_limit_t', torch.tensor(
            self.branch_has_limit, dtype=torch.bool))

    def _build_mapping_matrices(self, params, bus_id_to_idx):
        """
        Build generator-to-bus and load-to-bus mapping matrices.

        Map_g_P: (n_buses, n_gen_non_slack) — non-slack Pg → bus (P side)
        Map_g_Q: (n_buses, n_gen) — Qg → bus (Q side)
        Map_g_full: (2*n_buses, n_gen_non_slack + n_gen) — block form for stationarity
            [[Map_g_P,   0     ],    ← P rows (lambda_P side)
             [  0    , Map_g_Q ]]    ← Q rows (lambda_Q side)
        """
        n = self.n_buses
        gen_bus_ids = params['general']['gen_bus_ids']
        load_bus_ids = params['general']['load_bus_ids']
        non_slack_gen_idx = params['general']['non_slack_gen_idx']

        # P-side mapping: non-slack Pg → bus
        Map_g_P = np.zeros((n, self.n_gen_non_slack), dtype=np.float32)
        for col_idx, gen_global_idx in enumerate(non_slack_gen_idx):
            bus_id = int(gen_bus_ids[gen_global_idx])
            bus_idx = bus_id_to_idx[bus_id]
            Map_g_P[bus_idx, col_idx] += 1.0

        # Q-side mapping: all Qg → bus
        Map_g_Q = np.zeros((n, self.n_gen), dtype=np.float32)
        for gen_local_idx in range(self.n_gen):
            bus_id = int(gen_bus_ids[gen_local_idx])
            bus_idx = bus_id_to_idx[bus_id]
            Map_g_Q[bus_idx, gen_local_idx] += 1.0

        # Full block form: (2*n_buses, n_gen_non_slack + n_gen)
        # lambda_p = [lambda_P (n), lambda_Q (n)]
        # G = [pg_non_slack (n_ns), qg (n_gen)]
        # stationarity: lambda_p^T @ Map_g_full = [lambda_P^T @ Map_g_P, lambda_Q^T @ Map_g_Q]
        n_g_out = self.n_gen_non_slack + self.n_gen
        Map_g_full = np.zeros((2 * n, n_g_out), dtype=np.float32)
        Map_g_full[:n, :self.n_gen_non_slack] = Map_g_P       # P rows, Pg columns
        Map_g_full[n:, self.n_gen_non_slack:] = Map_g_Q       # Q rows, Qg columns

        self.register_buffer('Map_g', torch.tensor(Map_g_full))

        # Also store individual maps for power injection computation
        self.register_buffer('Map_g_P', torch.tensor(Map_g_P))
        self.register_buffer('Map_g_Q', torch.tensor(Map_g_Q))

    def compute_kkt_error(self, v_rect, pg_qg, inputs,
                          lambda_p, mu_g_u, mu_g_d, mu_v_u, mu_v_d,
                          mu_sm_fr, mu_sm_to):
        """
        Compute KKT error (physics loss) using rectangular voltage.

        Args:
            v_rect:   (batch, 2*n_buses) — [Vr, Vi]
            pg_qg:    (batch, n_gen_non_slack + n_gen) — [pg_ns, qg]
            inputs:   (batch, 2*n_loads) — [pd, qd]
            lambda_p: (batch, 2*n_buses) — KCL duals
            mu_g_u/d: (batch, n_gen_non_slack + n_gen) — gen bound duals
            mu_v_u/d: (batch, n_buses) — voltage bound duals
            mu_sm_fr/to: (batch, n_branches) — line flow duals

        Returns:
            kkt_error: (batch,) — total KKT error per sample
        """
        batch = v_rect.shape[0]
        n = self.n_buses
        n_br = self.n_branches

        Vr = v_rect[:, :n]          # (batch, n)
        Vi = v_rect[:, n:]          # (batch, n)

        pg_ns = pg_qg[:, :self.n_gen_non_slack]
        qg = pg_qg[:, self.n_gen_non_slack:]

        kkt_error = torch.zeros(batch, device=v_rect.device)

        # ============ 1. Reference bus constraint ============
        # Vi at slack bus should be 0 (angle reference)
        kkt_error = kkt_error + torch.abs(Vi[:, self.slack_bus_idx])

        # ============ 2. Power flow equation error (ε_prim) ============
        # P_calc = Vr * (Y_r @ Vr - Y_i @ Vi) + Vi * (Y_i @ Vr + Y_r @ Vi)
        # Q_calc = Vi * (Y_r @ Vr - Y_i @ Vi) - Vr * (Y_i @ Vr + Y_r @ Vi)
        YrVr = torch.matmul(Vr, self.Y_real.t())  # (batch, n)
        YiVi = torch.matmul(Vi, self.Y_imag.t())
        YiVr = torch.matmul(Vr, self.Y_imag.t())
        YrVi = torch.matmul(Vi, self.Y_real.t())

        P_calc = Vr * (YrVr - YiVi) + Vi * (YiVr + YrVi)  # (batch, n)
        Q_calc = Vi * (YrVr - YiVi) - Vr * (YiVr + YrVi)  # (batch, n)

        # Injection: P_inj = Map_g_P @ pg_ns - Pd_bus,  Q_inj = Map_g_Q @ qg - Qd_bus
        # Map_g_P: (n_buses, n_gen_non_slack), Map_g_Q: (n_buses, n_gen)
        P_gen = torch.matmul(pg_ns, self.Map_g_P.t())   # (batch, n)
        Q_gen = torch.matmul(qg, self.Map_g_Q.t())      # (batch, n)

        # Load injection
        pd = inputs[:, :self.n_loads]
        qd = inputs[:, self.n_loads:]
        P_load = torch.zeros(batch, n, device=v_rect.device)
        Q_load = torch.zeros(batch, n, device=v_rect.device)
        P_load.scatter_add_(1, self.load_to_bus_idx.unsqueeze(0).expand(batch, -1), pd)
        Q_load.scatter_add_(1, self.load_to_bus_idx.unsqueeze(0).expand(batch, -1), qd)

        P_inj = P_gen - P_load
        Q_inj = Q_gen - Q_load

        # Power balance error (skip slack bus for P — slack Pg is unknown)
        P_error = torch.abs(P_calc - P_inj)
        Q_error = torch.abs(Q_calc - Q_inj)

        kkt_error = kkt_error + torch.sum(P_error, dim=1) + torch.sum(Q_error, dim=1)

        # ============ 3. Generator limit violations ============
        # Pg (non-slack) violations
        kkt_error = kkt_error + torch.sum(torch.relu(pg_ns - self.pg_max_ns), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(self.pg_min_ns - pg_ns), dim=1)
        # Qg violations
        kkt_error = kkt_error + torch.sum(torch.relu(qg - self.qg_max), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(self.qg_min - qg), dim=1)

        # ============ 4. Voltage limit violations ============
        Vm_sq = Vr ** 2 + Vi ** 2  # (batch, n)
        kkt_error = kkt_error + torch.sum(torch.relu(Vm_sq - self.vm_max_sq), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(self.vm_min_sq - Vm_sq), dim=1)

        # ============ 5. Line flow violations ============
        # Ibr = Ybr_IM @ V, where V = [Vr; Vi]
        Ibr = torch.matmul(v_rect, self.Ybr_IM.t())  # (batch, 2*n_br)
        Ibr_sq = Ibr[:, :n_br] ** 2 + Ibr[:, n_br:] ** 2  # |I|^2 per branch
        line_viol = torch.relu(Ibr_sq - self.line_limit_sq)
        # Only count branches with limits
        kkt_error = kkt_error + torch.sum(
            line_viol * self.branch_has_limit_t.unsqueeze(0).float(), dim=1)

        # ============ 6. Complementary slackness (ε_comp) ============
        # mu_g_u * (G - G_max) = 0,  mu_g_d * (G_min - G) = 0
        gen_max = torch.cat([self.pg_max_ns, self.qg_max], dim=1)  # (1, n_ns+n_gen)
        gen_min = torch.cat([self.pg_min_ns, self.qg_min], dim=1)
        comp_g_u = torch.abs(mu_g_u * (pg_qg - gen_max))
        comp_g_d = torch.abs(mu_g_d * (gen_min - pg_qg))
        kkt_error = kkt_error + torch.sum(comp_g_u + comp_g_d, dim=1)

        # mu_v_u * (Vm^2 - Vm_max^2) = 0,  mu_v_d * (Vm_min^2 - Vm^2) = 0
        comp_v_u = torch.abs(mu_v_u * (Vm_sq - self.vm_max_sq))
        comp_v_d = torch.abs(mu_v_d * (self.vm_min_sq - Vm_sq))
        kkt_error = kkt_error + torch.sum(comp_v_u + comp_v_d, dim=1)

        # mu_sm * (|Ibr|^2 - limit^2) = 0
        comp_sm_fr = torch.abs(mu_sm_fr * (Ibr_sq - self.line_limit_sq))
        comp_sm_to = torch.abs(mu_sm_to * (Ibr_sq - self.line_limit_sq))
        kkt_error = kkt_error + torch.sum(
            (comp_sm_fr + comp_sm_to) * self.branch_has_limit_t.unsqueeze(0).float(), dim=1)

        # ============ 7. Dual feasibility (ε_dual) ============
        # All mu >= 0
        kkt_error = kkt_error + torch.sum(torch.relu(-mu_g_u), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(-mu_g_d), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(-mu_v_u), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(-mu_v_d), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(-mu_sm_fr), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(-mu_sm_to), dim=1)

        # ============ 8. Stationarity w.r.t. G (ε_stat partial) ============
        # dL/dG = c - lambda^T @ Map_g + mu_g_u - mu_g_d = 0
        # lambda_p has 2*n_buses entries: [lambda_P (n), lambda_Q (n)]
        lambda_mapped = torch.matmul(lambda_p, self.Map_g)  # (batch, n_ns+n_gen)
        stat_g = torch.abs(self.cost_vec - lambda_mapped + mu_g_u - mu_g_d)
        kkt_error = kkt_error + torch.sum(stat_g, dim=1)

        return kkt_error

    def forward(self, inputs):
        """
        Full forward pass: network prediction + KKT error computation.

        Args:
            inputs: (batch, 2*n_loads) — [pd, qd]

        Returns:
            outputs dict with all predictions + 'kkt_error'
        """
        outputs = self.core_network(inputs)

        kkt_error = self.compute_kkt_error(
            v_rect=outputs['v_rect'],
            pg_qg=outputs['pg_qg'],
            inputs=inputs,
            lambda_p=outputs['lambda_p'],
            mu_g_u=outputs['mu_g_u'],
            mu_g_d=outputs['mu_g_d'],
            mu_v_u=outputs['mu_v_u'],
            mu_v_d=outputs['mu_v_d'],
            mu_sm_fr=outputs['mu_sm_fr'],
            mu_sm_to=outputs['mu_sm_to'],
        )
        outputs['kkt_error'] = kkt_error
        return outputs