# -*- coding: utf-8 -*-
"""
ACOPF PINN Model - PyTorch (Faithful to Paper Eq. 24)
======================================================
Wraps PinnLayer with:
1. Loss function computation with collocation mask
2. Loss weighting scheme (Λ_P, Λ_V, Λ_L, Λ_ε)
3. Training optimizer

Loss function (Paper Eq. 24):
    L = (Nt+Nc)/Nt * [Λ_P * MAE_g + Λ_V * MAE_v + Λ_L * MAE_l]
        + Λ_ε * MAE_ε

  - MAE_g/v/l are only computed on supervised (training) points (mask=1)
  - MAE_ε (KKT error) is computed on ALL points (training + collocation)
  - The (Nt+Nc)/Nt factor compensates for collocation dilution of supervised loss

Collocation mechanism:
  - Supervised samples: mask = 1 → all losses active
  - Collocation samples: mask = 0 → only KKT error active
"""

import torch
import torch.nn as nn
from acopf_pinnlayer import PinnLayer


class PinnModel(nn.Module):
    """
    Complete PINN model for ACOPF.

    Args:
        simulation_parameters: dict with system params (from load_parameters_from_csv)
        lambda_P:  weight for generation prediction loss (MAE_g)
        lambda_V:  weight for voltage prediction loss (MAE_v)
        lambda_L:  weight for dual variable prediction loss (MAE_l)
        lambda_eps: weight for KKT physics loss (MAE_ε)
        collocation_ratio: fraction of training data used as collocation
        learning_rate: Adam learning rate
        device: 'cuda' or 'cpu'
    """

    def __init__(self, simulation_parameters,
                 lambda_P=1.0, lambda_V=1.0, lambda_L=1e-3, lambda_eps=1e-2,
                 collocation_ratio=0.5,
                 learning_rate=1e-3, device='cpu'):
        super().__init__()

        self.device = device
        self.pinn_layer = PinnLayer(simulation_parameters, device=device)

        # Loss weights (paper: Λ_P, Λ_V, Λ_L, Λ_ε)
        self.lambda_P = lambda_P
        self.lambda_V = lambda_V
        self.lambda_L = lambda_L
        self.lambda_eps = lambda_eps

        # Collocation ratio for supervised loss compensation
        # ratio_factor = (Nt + Nc) / Nt = 1 / (1 - collocation_ratio)
        if collocation_ratio < 1.0:
            self.ratio_factor = 1.0 / (1.0 - collocation_ratio)
        else:
            self.ratio_factor = 1.0

        # Loss function
        self.criterion = nn.L1Loss(reduction='none')  # element-wise MAE

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Store dimensions for convenience
        self.n_gen_non_slack = simulation_parameters['general']['n_gen_non_slack']
        self.n_gen = simulation_parameters['general']['n_gen']
        self.n_buses = simulation_parameters['general']['n_buses']
        self.n_branches = simulation_parameters['general']['n_branches']

        print(f"  PinnModel initialized:")
        print(f"    Loss weights: Λ_P={lambda_P}, Λ_V={lambda_V}, Λ_L={lambda_L}, Λ_ε={lambda_eps}")
        print(f"    Collocation ratio factor: {self.ratio_factor:.2f}")
        print(f"    Output dimensions:")
        print(f"      G branch: {self.n_gen_non_slack} (pg_ns) + {self.n_gen} (qg)")
        print(f"      V branch: {2 * self.n_buses} (Vr + Vi)")
        print(f"      Lm branch: {2*self.n_buses + 2*(self.n_gen_non_slack+self.n_gen) + 2*self.n_buses + 2*self.n_branches} dual vars")

    def forward(self, inputs):
        """Forward pass through PinnLayer."""
        return self.pinn_layer(inputs)

    def compute_loss(self, outputs, targets, mask):
        """
        Compute weighted loss with collocation mask.

        Args:
            outputs: dict from PinnLayer forward
            targets: dict with keys:
                'pg_qg':    (batch, n_gen_non_slack + n_gen) — [pg_ns, qg] true
                'v_rect':   (batch, 2 * n_buses) — [Vr, Vi] true
                'lambda_p': (batch, 2 * n_buses)
                'mu_g_u':   (batch, n_gen_non_slack + n_gen)
                'mu_g_d':   (batch, n_gen_non_slack + n_gen)
                'mu_v_u':   (batch, n_buses)
                'mu_v_d':   (batch, n_buses)
                'mu_sm_fr': (batch, n_branches)
                'mu_sm_to': (batch, n_branches)
            mask: (batch, 1) — 1 for supervised, 0 for collocation

        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss components (for logging)
        """
        batch = mask.shape[0]
        n_supervised = mask.sum().item()

        # ============ 1. Generation prediction loss (MAE_g) ============
        # Only on supervised points
        mae_g_elem = self.criterion(outputs['pg_qg'], targets['pg_qg'])  # (batch, dim)
        mae_g = (mae_g_elem * mask).sum() / max(n_supervised * mae_g_elem.shape[1], 1)

        # ============ 2. Voltage prediction loss (MAE_v) ============
        mae_v_elem = self.criterion(outputs['v_rect'], targets['v_rect'])
        mae_v = (mae_v_elem * mask).sum() / max(n_supervised * mae_v_elem.shape[1], 1)

        # ============ 3. Dual variable prediction loss (MAE_l) ============
        dual_keys = ['lambda_p', 'mu_g_u', 'mu_g_d', 'mu_v_u', 'mu_v_d', 'mu_sm_fr', 'mu_sm_to']
        mae_l = torch.tensor(0.0, device=mask.device)
        total_dual_dim = 0
        for key in dual_keys:
            elem = self.criterion(outputs[key], targets[key])
            mae_l = mae_l + (elem * mask).sum()
            total_dual_dim += elem.shape[1]
        mae_l = mae_l / max(n_supervised * total_dual_dim, 1)

        # ============ 4. KKT physics loss (MAE_ε) ============
        # On ALL points (training + collocation)
        kkt_error = outputs['kkt_error']  # (batch,)
        mae_eps = kkt_error.mean()

        # ============ 5. Total loss (Eq. 24) ============
        # Supervised terms scaled by ratio_factor to compensate collocation dilution
        total_loss = (
            self.ratio_factor * self.lambda_P * mae_g +
            self.ratio_factor * self.lambda_V * mae_v +
            self.ratio_factor * self.lambda_L * mae_l +
            self.lambda_eps * mae_eps
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
        """
        Extract all predicted quantities from model output for evaluation.

        Returns:
            pg_non_slack: (batch, n_gen_non_slack) — active power (non-slack)
            vm_gen:       (batch, n_gen) — voltage magnitudes at generator buses
            vm_all:       (batch, n_buses) — voltage magnitudes at all buses
            va_all:       (batch, n_buses) — voltage angles (radians) at all buses
            qg_all:       (batch, n_gen) — reactive power for all generators
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

        # pg_non_slack and qg from pg_qg
        pg_non_slack = outputs['pg_qg'][:, :self.n_gen_non_slack]
        qg_all = outputs['pg_qg'][:, self.n_gen_non_slack:]

        # vm, va from rectangular voltage
        n = self.n_buses
        Vr = outputs['v_rect'][:, :n]
        Vi = outputs['v_rect'][:, n:]
        vm_all = torch.sqrt(Vr ** 2 + Vi ** 2)
        va_all = torch.atan2(Vi, Vr)  # radians

        # Extract generator bus Vm
        vm_gen = vm_all[:, self.pinn_layer.gen_to_bus_idx]

        return pg_non_slack, vm_gen, vm_all, va_all, qg_all