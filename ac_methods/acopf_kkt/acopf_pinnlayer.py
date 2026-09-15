# -*- coding: utf-8 -*-
"""PINN physics layer: wraps the core network and scores the KKT conditions.

The KKT error sums, per sample: the angle reference, the power flow residual, primal
violations of the generator, voltage and line limits, complementary slackness, dual
feasibility, and stationarity with respect to the generation variables. Everything is
computed in rectangular voltage coordinates, which keeps the power flow polynomial.
"""

import numpy as np
import torch
import torch.nn as nn

from acopf_densecorenetwork import DenseCoreNetwork


class PinnLayer(nn.Module):
    """Core network plus the KKT residual it is penalized on."""

    def __init__(self, simulation_parameters, neurons_V, neurons_G, neurons_Lg):
        super().__init__()

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

        slack_bus_ids = bus_ids[gen['bus_types'] == 3]
        slack_bus_indices = [bus_id_to_idx[int(bid)] for bid in slack_bus_ids]
        self.slack_bus_idx = slack_bus_indices[0] if slack_bus_indices else 0

        self.register_buffer('load_to_bus_idx', torch.tensor(
            [bus_id_to_idx[int(bid)] for bid in load_bus_ids], dtype=torch.long))
        self.register_buffer('gen_to_bus_idx', torch.tensor(
            [bus_id_to_idx[int(bid)] for bid in gen_bus_ids], dtype=torch.long))
        self.register_buffer('non_slack_gen_idx', torch.tensor(
            non_slack_gen_idx, dtype=torch.long))

        self.core_network = DenseCoreNetwork(
            input_dim=2 * self.n_loads,
            n_buses=self.n_buses,
            n_gen=self.n_gen,
            n_gen_non_slack=self.n_gen_non_slack,
            n_branches=self.n_branches,
            neurons_V=neurons_V,
            neurons_G=neurons_G,
            neurons_Lg=neurons_Lg,
        )

        pg_min_all = simulation_parameters['generator']['pg_min'].flatten()
        pg_max_all = simulation_parameters['generator']['pg_max'].flatten()
        qg_min_all = simulation_parameters['generator']['qg_min'].flatten()
        qg_max_all = simulation_parameters['generator']['qg_max'].flatten()

        self.register_buffer('pg_min_ns', torch.tensor(
            pg_min_all[non_slack_gen_idx], dtype=torch.float32).unsqueeze(0))
        self.register_buffer('pg_max_ns', torch.tensor(
            pg_max_all[non_slack_gen_idx], dtype=torch.float32).unsqueeze(0))
        self.register_buffer('qg_min', torch.tensor(
            qg_min_all, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('qg_max', torch.tensor(
            qg_max_all, dtype=torch.float32).unsqueeze(0))

        # Voltage limits are stored squared, so they compare directly against
        # Vr^2 + Vi^2 without taking a square root in the loss
        vm_min = simulation_parameters['bus']['vm_min'].astype(np.float32)
        vm_max = simulation_parameters['bus']['vm_max'].astype(np.float32)
        self.register_buffer('vm_min_sq', torch.tensor(vm_min ** 2).unsqueeze(0))
        self.register_buffer('vm_max_sq', torch.tensor(vm_max ** 2).unsqueeze(0))

        # Stationarity uses only the linear cost term, and Qg carries no cost, so the
        # gradient vector is [c1 over non-slack generators, zeros over Qg]
        cost_c1 = simulation_parameters['generator']['cost_c1'].astype(np.float32)
        cost_vec = np.concatenate([
            cost_c1[non_slack_gen_idx],
            np.zeros(self.n_gen, dtype=np.float32)
        ])
        self.register_buffer('cost_vec', torch.tensor(cost_vec).unsqueeze(0))

        self._build_admittance_matrices(simulation_parameters, bus_id_to_idx)
        self._build_mapping_matrices(simulation_parameters, bus_id_to_idx)

    def _build_admittance_matrices(self, params, bus_id_to_idx):
        """Build the bus admittance matrix and the branch current operator.

        Both are stored as real blocks so they act on v = [Vr, Vi] directly.
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

        Y = np.zeros((n, n), dtype=np.complex128)
        ybr_diag = np.zeros(n_br, dtype=np.complex128)
        IM_complex = np.zeros((n_br, n), dtype=np.complex128)

        for k in range(n_br):
            i = bus_id_to_idx[int(f_bus[k])]
            j = bus_id_to_idx[int(t_bus[k])]

            z = complex(r_pu[k], x_pu[k])
            y_series = 1.0 / z if abs(z) > 1e-10 else 0.0
            y_shunt = complex(0, b_pu[k])

            tap = tap_ratio[k] if tap_ratio[k] != 0 else 1.0
            tap_c = tap * np.exp(1j * shift_deg[k] * np.pi / 180.0)

            Y[i, i] += y_series / (tap * np.conj(tap_c)) + y_shunt / 2.0
            Y[j, j] += y_series + y_shunt / 2.0
            Y[i, j] -= y_series / np.conj(tap_c)
            Y[j, i] -= y_series / tap_c

            ybr_diag[k] = y_series / tap_c
            IM_complex[k, i] = 1.0
            IM_complex[k, j] = -1.0

        self.register_buffer('Y_real', torch.tensor(Y.real, dtype=torch.float32))
        self.register_buffer('Y_imag', torch.tensor(Y.imag, dtype=torch.float32))

        # Branch current: I = Ybr @ IM @ v, with each complex operator written as the
        # real block [[re, -im], [im, re]] so the product stays a real matmul
        Ybr_block = np.block([
            [np.diag(ybr_diag.real), -np.diag(ybr_diag.imag)],
            [np.diag(ybr_diag.imag), np.diag(ybr_diag.real)],
        ])
        IM_block = np.block([
            [IM_complex.real, -IM_complex.imag],
            [IM_complex.imag, IM_complex.real],
        ])
        self.register_buffer('Ybr_IM', torch.tensor(
            Ybr_block @ IM_block, dtype=torch.float32))

        # Branches without a usable rating get an unreachable limit and are masked
        # out of the loss, matching the 9900 sentinel used elsewhere
        line_limit_sq = np.full(n_br, 1e10, dtype=np.float32)
        has_limit = np.zeros(n_br, dtype=bool)
        for k in range(n_br):
            if np.isfinite(rate_a[k]) and 0 < rate_a[k] < 9000:
                line_limit_sq[k] = rate_a[k] ** 2
                has_limit[k] = True

        self.register_buffer('line_limit_sq', torch.tensor(line_limit_sq).unsqueeze(0))
        self.register_buffer('branch_has_limit', torch.tensor(
            has_limit.astype(np.float32)).unsqueeze(0))

    def _build_mapping_matrices(self, params, bus_id_to_idx):
        """Build the generator-to-bus incidence matrices used by the power balance.

        Map_g is the block form used by stationarity: the P rows only see non-slack Pg
        and the Q rows only see Qg, because slack Pg is not a decision variable here.
        """
        n = self.n_buses
        gen_bus_ids = params['general']['gen_bus_ids']
        non_slack_gen_idx = params['general']['non_slack_gen_idx']

        Map_g_P = np.zeros((n, self.n_gen_non_slack), dtype=np.float32)
        for col_idx, gen_global_idx in enumerate(non_slack_gen_idx):
            Map_g_P[bus_id_to_idx[int(gen_bus_ids[gen_global_idx])], col_idx] += 1.0

        Map_g_Q = np.zeros((n, self.n_gen), dtype=np.float32)
        for gen_local_idx in range(self.n_gen):
            Map_g_Q[bus_id_to_idx[int(gen_bus_ids[gen_local_idx])], gen_local_idx] += 1.0

        Map_g_full = np.zeros((2 * n, self.n_gen_non_slack + self.n_gen), dtype=np.float32)
        Map_g_full[:n, :self.n_gen_non_slack] = Map_g_P
        Map_g_full[n:, self.n_gen_non_slack:] = Map_g_Q

        self.register_buffer('Map_g', torch.tensor(Map_g_full))
        self.register_buffer('Map_g_P', torch.tensor(Map_g_P))
        self.register_buffer('Map_g_Q', torch.tensor(Map_g_Q))

    def compute_kkt_error(self, v_rect, pg_qg, inputs, lambda_p,
                          mu_g_u, mu_g_d, mu_v_u, mu_v_d, mu_sm_fr, mu_sm_to):
        """Return the summed KKT residual per sample, shape (batch,)."""
        batch = v_rect.shape[0]
        n = self.n_buses
        n_br = self.n_branches

        Vr = v_rect[:, :n]
        Vi = v_rect[:, n:]
        pg_ns = pg_qg[:, :self.n_gen_non_slack]
        qg = pg_qg[:, self.n_gen_non_slack:]

        kkt_error = torch.zeros(batch, device=v_rect.device)

        # Angle reference: the slack bus must have zero imaginary voltage
        kkt_error = kkt_error + torch.abs(Vi[:, self.slack_bus_idx])

        # Power flow residual from S = V * conj(Y @ V), expanded into real terms
        YrVr = torch.matmul(Vr, self.Y_real.t())
        YiVi = torch.matmul(Vi, self.Y_imag.t())
        YiVr = torch.matmul(Vr, self.Y_imag.t())
        YrVi = torch.matmul(Vi, self.Y_real.t())

        P_calc = Vr * (YrVr - YiVi) + Vi * (YiVr + YrVi)
        Q_calc = Vi * (YrVr - YiVi) - Vr * (YiVr + YrVi)

        P_gen = torch.matmul(pg_ns, self.Map_g_P.t())
        Q_gen = torch.matmul(qg, self.Map_g_Q.t())

        pd = inputs[:, :self.n_loads]
        qd = inputs[:, self.n_loads:]
        P_load = torch.zeros(batch, n, device=v_rect.device)
        Q_load = torch.zeros(batch, n, device=v_rect.device)
        load_idx = self.load_to_bus_idx.unsqueeze(0).expand(batch, -1)
        P_load.scatter_add_(1, load_idx, pd)
        Q_load.scatter_add_(1, load_idx, qd)

        kkt_error = kkt_error + torch.sum(torch.abs(P_calc - (P_gen - P_load)), dim=1)
        kkt_error = kkt_error + torch.sum(torch.abs(Q_calc - (Q_gen - Q_load)), dim=1)

        # Primal violations of the generator limits
        kkt_error = kkt_error + torch.sum(torch.relu(pg_ns - self.pg_max_ns), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(self.pg_min_ns - pg_ns), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(qg - self.qg_max), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(self.qg_min - qg), dim=1)

        # Primal violations of the voltage limits
        Vm_sq = Vr ** 2 + Vi ** 2
        kkt_error = kkt_error + torch.sum(torch.relu(Vm_sq - self.vm_max_sq), dim=1)
        kkt_error = kkt_error + torch.sum(torch.relu(self.vm_min_sq - Vm_sq), dim=1)

        # Primal violations of the line current limits
        Ibr = torch.matmul(v_rect, self.Ybr_IM.t())
        Ibr_sq = Ibr[:, :n_br] ** 2 + Ibr[:, n_br:] ** 2
        line_slack = Ibr_sq - self.line_limit_sq
        kkt_error = kkt_error + torch.sum(
            torch.relu(line_slack) * self.branch_has_limit, dim=1)

        # Complementary slackness on every inequality
        gen_max = torch.cat([self.pg_max_ns, self.qg_max], dim=1)
        gen_min = torch.cat([self.pg_min_ns, self.qg_min], dim=1)
        kkt_error = kkt_error + torch.sum(
            torch.abs(mu_g_u * (pg_qg - gen_max))
            + torch.abs(mu_g_d * (gen_min - pg_qg)), dim=1)
        kkt_error = kkt_error + torch.sum(
            torch.abs(mu_v_u * (Vm_sq - self.vm_max_sq))
            + torch.abs(mu_v_d * (self.vm_min_sq - Vm_sq)), dim=1)
        kkt_error = kkt_error + torch.sum(
            (torch.abs(mu_sm_fr * line_slack) + torch.abs(mu_sm_to * line_slack))
            * self.branch_has_limit, dim=1)

        # Dual feasibility: every inequality multiplier must be non-negative
        for mu in (mu_g_u, mu_g_d, mu_v_u, mu_v_d, mu_sm_fr, mu_sm_to):
            kkt_error = kkt_error + torch.sum(torch.relu(-mu), dim=1)

        # Stationarity with respect to the generation variables
        lambda_mapped = torch.matmul(lambda_p, self.Map_g)
        kkt_error = kkt_error + torch.sum(
            torch.abs(self.cost_vec - lambda_mapped + mu_g_u - mu_g_d), dim=1)

        return kkt_error

    def forward(self, inputs):
        """Predict, then attach the KKT residual under the 'kkt_error' key."""
        outputs = self.core_network(inputs)

        outputs['kkt_error'] = self.compute_kkt_error(
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
        return outputs