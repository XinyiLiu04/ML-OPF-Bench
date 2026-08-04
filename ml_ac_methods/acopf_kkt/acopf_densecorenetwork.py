# -*- coding: utf-8 -*-
"""
ACOPF Dense Core Network - PyTorch (Faithful to Paper Fig.2)
=============================================================
Three independent sub-networks sharing the same input D = [pd, qd]:

  G  branch → [Pg_non_slack, Qg_all]           (n_gen_non_slack + n_gen)
  V  branch → [Vr, Vi]  (rectangular voltage)   (2 * n_buses)
  Lm branch → all dual variables                (lambda_p, mu_g_u/d, mu_qg_u/d,
                                                  mu_v_u/d, mu_sm_fr/to)

Each branch: input → hidden_0 → ReLU → hidden_1 → ReLU → ... → output (linear)

Differences from original TF code (DenseCoreNetwork.py):
  - PyTorch instead of TensorFlow
  - G outputs [pg_non_slack, qg] instead of [pg_all, qg_all]
  - V outputs [Vr, Vi] for all buses (same as paper)
  - Lm outputs include mu_sm_fr/to (line flow duals)
  - Dynamic layer count (not hardcoded to 3 hidden layers)
  - Glorot normal initialization preserved
"""

import torch
import torch.nn as nn


def _build_hidden_layers(input_size, hidden_sizes):
    """Build a sequence of Linear + ReLU layers with Glorot normal init."""
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
    """Build a linear output layer with Glorot normal init."""
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_normal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class DenseCoreNetwork(nn.Module):
    """
    Three-branch core network faithful to paper Fig.2.

    Args:
        input_dim:  input feature dimension (2 * n_loads)
        n_buses:    number of buses
        n_gen:      number of generators (all)
        n_gen_non_slack: number of non-slack generators
        n_branches: number of branches
        neurons_V:  list of hidden layer sizes for V branch
        neurons_G:  list of hidden layer sizes for G branch
        neurons_Lg: list of hidden layer sizes for Lm branch
    """

    def __init__(self, input_dim, n_buses, n_gen, n_gen_non_slack, n_branches,
                 neurons_V, neurons_G, neurons_Lg):
        super().__init__()

        self.n_buses = n_buses
        self.n_gen = n_gen
        self.n_gen_non_slack = n_gen_non_slack
        self.n_branches = n_branches

        # ==================== G branch ====================
        # Output: [pg_non_slack (n_gen_non_slack), qg (n_gen)]
        self.g_hidden, g_last = _build_hidden_layers(input_dim, neurons_G)
        self.g_output = _build_output_layer(g_last, n_gen_non_slack + n_gen)

        # ==================== V branch ====================
        # Output: [Vr (n_buses), Vi (n_buses)]  (rectangular voltage)
        self.v_hidden, v_last = _build_hidden_layers(input_dim, neurons_V)
        self.v_output = _build_output_layer(v_last, 2 * n_buses)
        with torch.no_grad():
            self.v_output.bias[:n_buses].fill_(1.0)    # Vr
            self.v_output.bias[n_buses:].fill_(0.0)    # Vi

        # ==================== Lm branch ====================
        # Output dual variables:
        #   lambda_p:   2 * n_buses  (lambda for P and Q power balance)
        #   mu_g_u:     n_gen_non_slack + n_gen  (upper bound: pg_non_slack + qg)
        #   mu_g_d:     n_gen_non_slack + n_gen  (lower bound: pg_non_slack + qg)
        #   mu_v_u:     n_buses  (Vm upper bound)
        #   mu_v_d:     n_buses  (Vm lower bound)
        #   mu_sm_fr:   n_branches  (line flow from side)
        #   mu_sm_to:   n_branches  (line flow to side)
        self.lg_hidden, lg_last = _build_hidden_layers(input_dim, neurons_Lg)

        n_dual_g = n_gen_non_slack + n_gen  # pg_non_slack + qg
        self.lg_lambda_p  = _build_output_layer(lg_last, 2 * n_buses)
        self.lg_mu_g_u    = _build_output_layer(lg_last, n_dual_g)
        self.lg_mu_g_d    = _build_output_layer(lg_last, n_dual_g)
        self.lg_mu_v_u    = _build_output_layer(lg_last, n_buses)
        self.lg_mu_v_d    = _build_output_layer(lg_last, n_buses)
        self.lg_mu_sm_fr  = _build_output_layer(lg_last, n_branches)
        self.lg_mu_sm_to  = _build_output_layer(lg_last, n_branches)

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) — [pd, qd]

        Returns:
            dict with keys:
                'pg_qg':     (batch, n_gen_non_slack + n_gen)
                'v_rect':    (batch, 2 * n_buses)   — [Vr, Vi]
                'lambda_p':  (batch, 2 * n_buses)
                'mu_g_u':    (batch, n_gen_non_slack + n_gen)
                'mu_g_d':    (batch, n_gen_non_slack + n_gen)
                'mu_v_u':    (batch, n_buses)
                'mu_v_d':    (batch, n_buses)
                'mu_sm_fr':  (batch, n_branches)
                'mu_sm_to':  (batch, n_branches)
        """
        # G branch
        g_feat = self.g_hidden(x)
        pg_qg = self.g_output(g_feat)

        # V branch
        v_feat = self.v_hidden(x)
        v_rect = self.v_output(v_feat)

        # Lm branch
        lg_feat = self.lg_hidden(x)
        lambda_p = self.lg_lambda_p(lg_feat)
        mu_g_u   = self.lg_mu_g_u(lg_feat)
        mu_g_d   = self.lg_mu_g_d(lg_feat)
        mu_v_u   = self.lg_mu_v_u(lg_feat)
        mu_v_d   = self.lg_mu_v_d(lg_feat)
        mu_sm_fr = self.lg_mu_sm_fr(lg_feat)
        mu_sm_to = self.lg_mu_sm_to(lg_feat)

        return {
            'pg_qg':    pg_qg,
            'v_rect':   v_rect,
            'lambda_p': lambda_p,
            'mu_g_u':   mu_g_u,
            'mu_g_d':   mu_g_d,
            'mu_v_u':   mu_v_u,
            'mu_v_d':   mu_v_d,
            'mu_sm_fr': mu_sm_fr,
            'mu_sm_to': mu_sm_to,
        }