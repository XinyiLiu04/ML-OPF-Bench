# -*- coding: utf-8 -*-
"""Three-branch dense core network for the KKT-informed PINN.

All three branches take the same input D = [pd, qd] and are otherwise independent:
  G  -> [Pg_non_slack, Qg_all]
  V  -> [Vr, Vi] over all buses, i.e. rectangular voltage
  Lm -> every dual variable of the ACOPF
"""

import torch
import torch.nn as nn


def _build_hidden_layers(input_size, hidden_sizes):
    """Stack Linear and ReLU layers with Glorot normal initialization."""
    layers = []
    prev = input_size
    for h in hidden_sizes:
        linear = nn.Linear(prev, h)
        nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.ReLU())
        prev = h
    return nn.Sequential(*layers), prev


def _build_output_layer(in_features, out_features):
    """Linear output layer with Glorot normal initialization."""
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_normal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class DenseCoreNetwork(nn.Module):
    """The three branches of the PINN, sharing an input and nothing else."""

    def __init__(self, input_dim, n_buses, n_gen, n_gen_non_slack, n_branches,
                 neurons_V, neurons_G, neurons_Lg):
        super().__init__()

        self.n_buses = n_buses
        self.n_gen = n_gen
        self.n_gen_non_slack = n_gen_non_slack
        self.n_branches = n_branches

        self.g_hidden, g_last = _build_hidden_layers(input_dim, neurons_G)
        self.g_output = _build_output_layer(g_last, n_gen_non_slack + n_gen)

        self.v_hidden, v_last = _build_hidden_layers(input_dim, neurons_V)
        self.v_output = _build_output_layer(v_last, 2 * n_buses)

        # Voltage starts at a flat profile, Vr = 1 and Vi = 0, so the first KKT
        # evaluations are near a physically sensible operating point
        with torch.no_grad():
            self.v_output.bias[:n_buses].fill_(1.0)
            self.v_output.bias[n_buses:].fill_(0.0)

        self.lg_hidden, lg_last = _build_hidden_layers(input_dim, neurons_Lg)

        n_dual_g = n_gen_non_slack + n_gen
        self.lg_lambda_p = _build_output_layer(lg_last, 2 * n_buses)
        self.lg_mu_g_u = _build_output_layer(lg_last, n_dual_g)
        self.lg_mu_g_d = _build_output_layer(lg_last, n_dual_g)
        self.lg_mu_v_u = _build_output_layer(lg_last, n_buses)
        self.lg_mu_v_d = _build_output_layer(lg_last, n_buses)
        self.lg_mu_sm_fr = _build_output_layer(lg_last, n_branches)
        self.lg_mu_sm_to = _build_output_layer(lg_last, n_branches)

    def forward(self, x):
        """Return a dict with the primal predictions and every dual variable."""
        g_feat = self.g_hidden(x)
        v_feat = self.v_hidden(x)
        lg_feat = self.lg_hidden(x)

        return {
            'pg_qg': self.g_output(g_feat),
            'v_rect': self.v_output(v_feat),
            'lambda_p': self.lg_lambda_p(lg_feat),
            'mu_g_u': self.lg_mu_g_u(lg_feat),
            'mu_g_d': self.lg_mu_g_d(lg_feat),
            'mu_v_u': self.lg_mu_v_u(lg_feat),
            'mu_v_d': self.lg_mu_v_d(lg_feat),
            'mu_sm_fr': self.lg_mu_sm_fr(lg_feat),
            'mu_sm_to': self.lg_mu_sm_to(lg_feat),
        }