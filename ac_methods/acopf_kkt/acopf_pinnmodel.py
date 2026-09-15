# -*- coding: utf-8 -*-
"""PINN loss assembly.

    L = (Nt+Nc)/Nt * [Lambda_P * MAE_g + Lambda_V * MAE_v + Lambda_L * MAE_l]
        + Lambda_eps * MAE_eps

The three supervised terms are evaluated only where the mask is 1; the KKT residual is
evaluated everywhere, which is what lets unlabelled collocation points contribute. The
leading factor undoes the dilution those extra points cause in the supervised averages.
"""

import torch
import torch.nn as nn

from acopf_pinnlayer import PinnLayer

DUAL_KEYS = ('lambda_p', 'mu_g_u', 'mu_g_d', 'mu_v_u', 'mu_v_d', 'mu_sm_fr', 'mu_sm_to')


class PinnModel(nn.Module):
    """Wraps PinnLayer with the weighted, collocation-aware loss."""

    def __init__(self, simulation_parameters, neurons_V, neurons_G, neurons_Lg,
                 lambda_P=1.0, lambda_V=1.0, lambda_L=1e-3, lambda_eps=1e-2,
                 collocation_ratio=0.5):
        super().__init__()

        self.pinn_layer = PinnLayer(
            simulation_parameters,
            neurons_V=neurons_V,
            neurons_G=neurons_G,
            neurons_Lg=neurons_Lg,
        )

        self.lambda_P = lambda_P
        self.lambda_V = lambda_V
        self.lambda_L = lambda_L
        self.lambda_eps = lambda_eps

        # (Nt + Nc) / Nt for a collocation fraction of the training set
        self.ratio_factor = (1.0 / (1.0 - collocation_ratio)
                             if collocation_ratio < 1.0 else 1.0)

        self.criterion = nn.L1Loss(reduction='none')

        self.n_gen_non_slack = simulation_parameters['general']['n_gen_non_slack']
        self.n_gen = simulation_parameters['general']['n_gen']
        self.n_buses = simulation_parameters['general']['n_buses']
        self.n_branches = simulation_parameters['general']['n_branches']

        n_duals = (2 * self.n_buses
                   + 2 * (self.n_gen_non_slack + self.n_gen)
                   + 2 * self.n_buses
                   + 2 * self.n_branches)

        print(f"  PinnModel initialized:")
        print(f"    Loss weights: P={lambda_P}, V={lambda_V}, L={lambda_L}, "
              f"eps={lambda_eps}")
        print(f"    Collocation ratio factor: {self.ratio_factor:.2f}")
        print(f"    G branch: {self.n_gen_non_slack} (pg_ns) + {self.n_gen} (qg)")
        print(f"    V branch: {2 * self.n_buses} (Vr + Vi)")
        print(f"    Lm branch: {n_duals} dual variables")

    def forward(self, inputs):
        return self.pinn_layer(inputs)

    def compute_loss(self, outputs, targets, mask):
        """Return (total_loss, per-term dict) for one batch.

        mask is (batch, 1), 1 for supervised samples and 0 for collocation points.
        """
        n_supervised = mask.sum().item()

        def _masked_mae(key):
            elem = self.criterion(outputs[key], targets[key])
            return (elem * mask).sum(), elem.shape[1]

        g_sum, g_dim = _masked_mae('pg_qg')
        mae_g = g_sum / max(n_supervised * g_dim, 1)

        v_sum, v_dim = _masked_mae('v_rect')
        mae_v = v_sum / max(n_supervised * v_dim, 1)

        # The dual terms are pooled into a single average over all dual entries
        l_sum = torch.tensor(0.0, device=mask.device)
        l_dim = 0
        for key in DUAL_KEYS:
            s, d = _masked_mae(key)
            l_sum = l_sum + s
            l_dim += d
        mae_l = l_sum / max(n_supervised * l_dim, 1)

        mae_eps = outputs['kkt_error'].mean()

        total_loss = (
            self.ratio_factor * self.lambda_P * mae_g
            + self.ratio_factor * self.lambda_V * mae_v
            + self.ratio_factor * self.lambda_L * mae_l
            + self.lambda_eps * mae_eps
        )

        loss_dict = {
            'mae_g': mae_g.item(),
            'mae_v': mae_v.item(),
            'mae_l': mae_l.item(),
            'mae_eps': mae_eps.item(),
            'total': total_loss.item(),
        }
        return total_loss, loss_dict

    def predict_for_evaluation(self, x):
        """Return (pg_non_slack, vm_gen, vm_all, va_all_rad, qg_all) for a batch.

        The caller is responsible for putting the module in eval mode.
        """
        with torch.no_grad():
            outputs = self.forward(x)

        pg_non_slack = outputs['pg_qg'][:, :self.n_gen_non_slack]
        qg_all = outputs['pg_qg'][:, self.n_gen_non_slack:]

        n = self.n_buses
        Vr = outputs['v_rect'][:, :n]
        Vi = outputs['v_rect'][:, n:]
        vm_all = torch.sqrt(Vr ** 2 + Vi ** 2)
        va_all = torch.atan2(Vi, Vr)

        vm_gen = vm_all[:, self.pinn_layer.gen_to_bus_idx]

        return pg_non_slack, vm_gen, vm_all, va_all, qg_all