# -*- coding: utf-8 -*-
"""
Spectral GNN Model for ACOPF (Paper: Owerko et al., ICASSP 2020)
— Paper-faithful node features edition —

Key change vs. previous version:
    Node features now follow the paper exactly:
        X_n = [vm_n, va_n, p_n, q_n]  (sub-optimal state from DCOPF + PF)
    instead of [pd_i, qd_i, 1.0, 0.0].

    The sub-optimal state X is pre-computed by generate_subopt_state.py
    and loaded as a [n_samples, 4*n_buses] flat vector (already scaled).

Architecture:
- Node features: [vm, va, p_inj, q_inj] per bus (4-dim)
- Graph convolution: 2x ChebConv (K=4 filter taps)
- Local readout: per-node Linear(F2 -> 1), select generator nodes
- Output: pg_non_slack only (following paper: predict p* only)
          OR [pg_non_slack | vm_gen] (extended version)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv


# =====================================================================
# Node Feature Builder (sub-optimal state X)
# =====================================================================
def build_node_features_subopt(x_scaled_batch: torch.Tensor,
                               n_buses: int,
                               device: torch.device) -> torch.Tensor:
    """
    Build per-node feature matrix from pre-scaled sub-optimal state.

    The input x_scaled_batch is a flat vector of shape [B, 4*N] where:
        x[:, 0:N]     = vm_scaled   (voltage magnitude, all buses)
        x[:, N:2N]    = va_scaled   (voltage angle, all buses)
        x[:, 2N:3N]   = pinj_scaled (active power injection, all buses)
        x[:, 3N:4N]   = qinj_scaled (reactive power injection, all buses)

    This matches the paper's X = [v, δ, p, q] ∈ R^{N×4}.

    Args:
        x_scaled_batch : Tensor [B, 4*N]  pre-scaled sub-optimal state
        n_buses        : int, number of buses N
        device         : torch.device

    Returns:
        Z : Tensor [B*N, 4]  flattened for disjoint-graph conv
    """
    B = x_scaled_batch.shape[0]
    N = n_buses

    # Reshape: [B, 4*N] → [B, N, 4]
    # Columns are stacked as [vm_all | va_all | pinj_all | qinj_all]
    vm   = x_scaled_batch[:, 0:N]       # [B, N]
    va   = x_scaled_batch[:, N:2*N]     # [B, N]
    pinj = x_scaled_batch[:, 2*N:3*N]   # [B, N]
    qinj = x_scaled_batch[:, 3*N:4*N]   # [B, N]

    # Stack to [B, N, 4] then flatten to [B*N, 4]
    Z = torch.stack([vm, va, pinj, qinj], dim=-1)  # [B, N, 4]
    return Z.reshape(B * N, 4).to(device)


# =====================================================================
# Spectral GNN (ChebConv-based, Batched-Graph, Local Readout)
# =====================================================================
class SpectralGNN_ACOPF(nn.Module):
    """
    Local Spectral GNN for ACOPF — paper-faithful edition.

    forward() expects a PRE-COLLATED disjoint graph:
        node_feats  [B*N, 4]
        edge_index  [2, B*E]
        edge_weight [B*E]
        batch_size  int

    Args:
        F1     : hidden features after layer 1 (default 128)
        F2     : hidden features after layer 2 (default 64)
        K      : Chebyshev filter order (default 4)
        params : training-time params dict (used as default in forward)
        predict_vm : if True, output [pg_non_slack | vm_gen];
                     if False, output pg_non_slack only (paper default)
    """

    def __init__(self, F1=128, F2=64, K=4, params=None, predict_vm=True):
        super().__init__()

        self.F1     = F1
        self.F2     = F2
        self.K      = K
        self.params = params
        self.predict_vm = predict_vm

        # ── Graph convolution layers ──────────────────────────────────
        self.conv1 = ChebConv(in_channels=4,  out_channels=F1, K=K)
        self.conv2 = ChebConv(in_channels=F1, out_channels=F2, K=K)

        # ── Non-linearities ──────────────────────────────────────────
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        # ── Local readout heads ───────────────────────────────────────
        self.pg_head = nn.Linear(F2, 1)
        if predict_vm:
            self.vm_head = nn.Linear(F2, 1)

    # ── Index helpers ─────────────────────────────────────────────────
    @staticmethod
    def _gen_node_indices(params, device):
        """Graph-local generator node indices [G]."""
        bus_id_to_idx = params['general']['bus_id_to_idx']
        gen_bus_ids   = params['general']['gen_bus_ids']
        return torch.tensor(
            [bus_id_to_idx[int(g)] for g in gen_bus_ids],
            dtype=torch.long, device=device)

    @staticmethod
    def _non_slack_gen_node_indices(params, device):
        """Graph-local non-slack generator node indices [G_ns]."""
        bus_id_to_idx     = params['general']['bus_id_to_idx']
        gen_bus_ids       = params['general']['gen_bus_ids']
        non_slack_idx     = params['general']['non_slack_gen_idx']
        non_slack_bus_ids = gen_bus_ids[non_slack_idx]
        return torch.tensor(
            [bus_id_to_idx[int(g)] for g in non_slack_bus_ids],
            dtype=torch.long, device=device)

    # ── Forward pass ──────────────────────────────────────────────────
    def forward(self, node_feats, edge_index, edge_weight=None,
                batch_size=None, params=None):
        """
        Args:
            node_feats  : Tensor [B*N, 4]    pre-collated node features
            edge_index  : Tensor [2, B*E]    pre-collated edge list
            edge_weight : Tensor [B*E]       pre-collated edge weights
            batch_size  : int                number of graphs in batch
            params      : params dict        (defaults to self.params)

        Returns:
            if predict_vm:
                output : Tensor [B, n_gen_non_slack + n_gen]
            else:
                output : Tensor [B, n_gen_non_slack]
        """
        if params is None:
            params = self.params
        device = node_feats.device
        N = params['general']['n_buses']

        if batch_size is None:
            batch_size = node_feats.shape[0] // N

        # ── Graph convolution ─────────────────────────────────────────
        h = self.act1(self.conv1(node_feats, edge_index, edge_weight))
        h = self.act2(self.conv2(h, edge_index, edge_weight))
        # h : [B*N, F2]

        # ── Batched readout ───────────────────────────────────────────
        ns_gen_idx = self._non_slack_gen_node_indices(params, device)  # [G_ns]
        offsets    = torch.arange(batch_size, device=device) * N       # [B]

        # pg: non-slack generators
        batch_ns_idx = (ns_gen_idx.unsqueeze(0)
                        + offsets.unsqueeze(1)).reshape(-1)        # [B*G_ns]
        pg = self.pg_head(h[batch_ns_idx]).squeeze(-1)             # [B*G_ns]
        pg = pg.reshape(batch_size, -1)                            # [B, G_ns]

        if self.predict_vm:
            gen_idx = self._gen_node_indices(params, device)           # [G]
            batch_gen_idx = (gen_idx.unsqueeze(0)
                             + offsets.unsqueeze(1)).reshape(-1)       # [B*G]
            vm = self.vm_head(h[batch_gen_idx]).squeeze(-1)            # [B*G]
            vm = vm.reshape(batch_size, -1)                            # [B, G]
            return torch.cat([pg, vm], dim=-1)   # [B, G_ns + G]
        else:
            return pg  # [B, G_ns]