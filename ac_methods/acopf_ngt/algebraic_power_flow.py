# -*- coding: utf-8 -*-
"""Differentiable algebraic power flow with Kron reduction, for the DeepOPF-NGT family.

The network predicts voltage at the non-zero-injection buses. The zero-injection buses
follow linearly from the network equations, after which every other quantity (nodal
injections, generation, branch flows, achieved load) is an algebraic function of voltage.
Nothing here calls a power flow solver, so the whole pipeline stays differentiable.

The admittance matrix and the branch flow equations are both assembled from the pi-model
coefficients precomputed in acopf_data_setup, so the Y-bus and the branch flows cannot
disagree about line charging or transformer taps.
"""

import numpy as np
import torch


class AlgebraicPowerFlow:
    """Topology-dependent tensors, built once, then applied to every batch.

    The Kron reduction system is constant for a fixed topology, so it is LU factorized
    in the constructor instead of being re-solved from scratch on every batch.
    """

    def __init__(self, params, device):
        self.device = device
        self.params = params

        n = params['general']['n_buses']
        self.n_buses = n
        self.n_branches = params['general']['n_branches']
        self.n_loads = params['general']['n_loads']

        br = params['branch']
        f_idx = np.asarray(br['f_bus_idx'], dtype=np.int64)
        t_idx = np.asarray(br['t_bus_idx'], dtype=np.int64)

        # Y-bus from the pi-model coefficients: Y[f,f] += Yff, Y[f,t] += Yft,
        # Y[t,f] += Ytf, Y[t,t] += Ytt. Parallel branches accumulate.
        G = np.zeros((n, n), dtype=np.float64)
        B = np.zeros((n, n), dtype=np.float64)
        for k in range(self.n_branches):
            i, j = f_idx[k], t_idx[k]
            G[i, i] += br['Yff_g'][k]
            B[i, i] += br['Yff_b'][k]
            G[j, j] += br['Ytt_g'][k]
            B[j, j] += br['Ytt_b'][k]
            G[i, j] += br['Yft_g'][k]
            B[i, j] += br['Yft_b'][k]
            G[j, i] += br['Ytf_g'][k]
            B[j, i] += br['Ytf_b'][k]

        gs = np.asarray(params['bus']['gs'], dtype=np.float64)
        bs = np.asarray(params['bus']['bs'], dtype=np.float64)
        G[np.diag_indices(n)] += gs
        B[np.diag_indices(n)] += bs

        self.G = torch.tensor(G, dtype=torch.float32, device=device)
        self.B = torch.tensor(B, dtype=torch.float32, device=device)
        self.gs = torch.tensor(gs, dtype=torch.float32, device=device).unsqueeze(0)
        self.bs = torch.tensor(bs, dtype=torch.float32, device=device).unsqueeze(0)

        self.f_idx = torch.tensor(f_idx, dtype=torch.long, device=device)
        self.t_idx = torch.tensor(t_idx, dtype=torch.long, device=device)
        for key in ('Yff_g', 'Yff_b', 'Yft_g', 'Yft_b',
                    'Ytf_g', 'Ytf_b', 'Ytt_g', 'Ytt_b'):
            setattr(self, key, torch.tensor(
                np.asarray(br[key], dtype=np.float32), device=device).unsqueeze(0))

        # Zero-injection buses are solved from the network; the rest are predicted
        zib_mask = np.asarray(params['general']['zib_mask'], dtype=bool)
        self.zib_indices = np.where(zib_mask)[0]
        self.nonzib_indices = np.where(~zib_mask)[0]
        self.n_zib = len(self.zib_indices)
        self.n_nonzib = len(self.nonzib_indices)

        self.zib_idx = torch.tensor(self.zib_indices, dtype=torch.long, device=device)
        self.nonzib_idx = torch.tensor(self.nonzib_indices, dtype=torch.long, device=device)

        if self.n_zib > 0:
            G_bb = self.G[self.zib_idx][:, self.zib_idx]
            B_bb = self.B[self.zib_idx][:, self.zib_idx]
            self.G_ba = self.G[self.zib_idx][:, self.nonzib_idx]
            self.B_ba = self.B[self.zib_idx][:, self.nonzib_idx]

            # Zero injection at a bus with non-zero voltage means (Y V) = 0 there,
            # which is linear in V. In real blocks with V = e + jf:
            #   [G_bb  -B_bb] [e_b]     [G_ba e_a - B_ba f_a]
            #   [B_bb   G_bb] [f_b] = - [B_ba e_a + G_ba f_a]
            A = torch.cat([
                torch.cat([G_bb, -B_bb], dim=1),
                torch.cat([B_bb, G_bb], dim=1),
            ], dim=0)
            self._lu, self._piv = torch.linalg.lu_factor(A)
            self._A_cond = float(torch.linalg.cond(A))
        else:
            self._lu = self._piv = None
            self._A_cond = float('nan')

        # Buses carrying load but hosting no generator: their injection is fully
        # determined by the voltage solution, so the achieved load there is what the
        # load satisfaction term has to match. Where a generator sits on the same bus
        # it absorbs any mismatch, so those buses carry no such constraint.
        load_bus_ids = np.asarray(params['general']['load_bus_ids'])
        gen_bus_ids = set(int(g) for g in params['general']['gen_bus_ids'])
        bus_id_to_idx = params['general']['bus_id_to_idx']

        load_only_pos = [i for i, bid in enumerate(load_bus_ids)
                         if int(bid) not in gen_bus_ids]
        self.load_only_pos = torch.tensor(load_only_pos, dtype=torch.long, device=device)
        self.load_only_bus_idx = torch.tensor(
            [bus_id_to_idx[int(load_bus_ids[i])] for i in load_only_pos],
            dtype=torch.long, device=device)

        self.load_bus_idx = torch.tensor(
            [bus_id_to_idx[int(bid)] for bid in load_bus_ids],
            dtype=torch.long, device=device)
        self.gen_bus_idx = torch.tensor(
            [bus_id_to_idx[int(g)] for g in params['general']['gen_bus_ids']],
            dtype=torch.long, device=device)

    def summary(self):
        """One-line description of the reduction this object performs."""
        return (f"buses={self.n_buses}, predicted (non-ZIB)={self.n_nonzib}, "
                f"solved (ZIB)={self.n_zib}"
                + (f", Kron system cond={self._A_cond:.2e}" if self.n_zib else ""))

    def solve_zib(self, v_alpha, theta_alpha):
        """Recover the zero-injection bus voltages from the predicted ones."""
        batch = v_alpha.shape[0]
        if self.n_zib == 0:
            empty = torch.empty(batch, 0, device=self.device)
            return empty, empty

        e_a = v_alpha * torch.cos(theta_alpha)
        f_a = v_alpha * torch.sin(theta_alpha)

        rhs_e = -(e_a @ self.G_ba.t() - f_a @ self.B_ba.t())
        rhs_f = -(e_a @ self.B_ba.t() + f_a @ self.G_ba.t())
        rhs = torch.cat([rhs_e, rhs_f], dim=1)

        ef_b = torch.linalg.lu_solve(self._lu, self._piv, rhs.t()).t()
        e_b = ef_b[:, :self.n_zib]
        f_b = ef_b[:, self.n_zib:]

        return torch.sqrt(e_b ** 2 + f_b ** 2 + 1e-12), torch.atan2(f_b, e_b)

    def injections(self, v_all, theta_all):
        """Nodal active and reactive injections implied by a voltage profile.

        From S = V * conj(Y V); written as two real matmuls rather than a dense
        bus-by-bus tensor, which keeps memory flat in the number of buses.
        """
        e = v_all * torch.cos(theta_all)
        f = v_all * torch.sin(theta_all)

        Ir = e @ self.G.t() - f @ self.B.t()
        Ii = e @ self.B.t() + f @ self.G.t()

        P = e * Ir + f * Ii
        Q = f * Ir - e * Ii
        return P, Q

    def branch_flows(self, v_all, theta_all):
        """Branch flows at both ends, from the full pi-model."""
        vi = v_all[:, self.f_idx]
        vj = v_all[:, self.t_idx]
        theta_ij = theta_all[:, self.f_idx] - theta_all[:, self.t_idx]
        cos_t = torch.cos(theta_ij)
        sin_t = torch.sin(theta_ij)
        vivj = vi * vj

        pf = vi ** 2 * self.Yff_g + vivj * (self.Yft_g * cos_t + self.Yft_b * sin_t)
        qf = -vi ** 2 * self.Yff_b + vivj * (self.Yft_g * sin_t - self.Yft_b * cos_t)
        pt = vj ** 2 * self.Ytt_g + vivj * (self.Ytf_g * cos_t - self.Ytf_b * sin_t)
        qt = -vj ** 2 * self.Ytt_b + vivj * (-self.Ytf_g * sin_t - self.Ytf_b * cos_t)
        return pf, qf, pt, qt

    def __call__(self, v_alpha, theta_alpha, Pd, Qd):
        """Run the full reduction for one batch.

        Returns a dict with the reconstructed voltages, nodal injections, generation at
        the generator buses, branch flows at both ends, and the achieved load at the
        buses where load satisfaction is actually a constraint.
        """
        batch = v_alpha.shape[0]
        n = self.n_buses

        v_beta, theta_beta = self.solve_zib(v_alpha, theta_alpha)

        v_all = torch.zeros(batch, n, device=self.device)
        theta_all = torch.zeros(batch, n, device=self.device)
        v_all = v_all.index_copy(1, self.nonzib_idx, v_alpha)
        theta_all = theta_all.index_copy(1, self.nonzib_idx, theta_alpha)
        if self.n_zib > 0:
            v_all = v_all.index_copy(1, self.zib_idx, v_beta)
            theta_all = theta_all.index_copy(1, self.zib_idx, theta_beta)

        P_inject, Q_inject = self.injections(v_all, theta_all)
        pf, qf, pt, qt = self.branch_flows(v_all, theta_all)

        # Generation is what is left once the demanded load is added back
        Pd_full = torch.zeros(batch, n, device=self.device)
        Qd_full = torch.zeros(batch, n, device=self.device)
        Pd_full = Pd_full.index_copy(1, self.load_bus_idx, Pd)
        Qd_full = Qd_full.index_copy(1, self.load_bus_idx, Qd)

        Pg = (P_inject + Pd_full)[:, self.gen_bus_idx]
        Qg = (Q_inject + Qd_full)[:, self.gen_bus_idx]

        # At a load bus with no generator the injection is minus the delivered load
        # minus what the bus shunt draws, so the shunt term is removed to recover the
        # load the voltage profile actually serves
        vm_sq_lo = v_all[:, self.load_only_bus_idx] ** 2
        Pd_pred = -P_inject[:, self.load_only_bus_idx] \
            - self.gs[:, self.load_only_bus_idx] * vm_sq_lo
        Qd_pred = -Q_inject[:, self.load_only_bus_idx] \
            + self.bs[:, self.load_only_bus_idx] * vm_sq_lo

        return {
            'Pg': Pg,
            'Qg': Qg,
            'v_all': v_all,
            'theta_all': theta_all,
            'P_branch': pf,
            'Q_branch': qf,
            'P_branch_to': pt,
            'Q_branch_to': qt,
            'P_inject': P_inject,
            'Q_inject': Q_inject,
            'Pd_pred': Pd_pred,
            'Qd_pred': Qd_pred,
            'Pd_demanded': Pd[:, self.load_only_pos],
            'Qd_demanded': Qd[:, self.load_only_pos],
            'v_beta': v_beta,
            'theta_beta': theta_beta,
        }