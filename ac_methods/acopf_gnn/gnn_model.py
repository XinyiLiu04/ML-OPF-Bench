# -*- coding: utf-8 -*-
"""Spectral GNN for ACOPF (Owerko et al., ICASSP 2020).

Node features are the sub-optimal state [vm, va, p_inj, q_inj] per bus, precomputed by
generate_subopt_state.py from a DCOPF solve followed by an AC power flow. Two ChebConv
layers are followed by a per-node linear readout taken at the generator buses.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv


def build_node_features_subopt(x_scaled_batch, n_buses, device):
    """Reshape a flat sub-optimal state batch [B, 4*N] into node features [B*N, 4].

    The flat layout is [vm_all | va_all | pinj_all | qinj_all], matching the paper's
    X = [v, delta, p, q] in R^{N x 4}.
    """
    B = x_scaled_batch.shape[0]
    N = n_buses

    Z = torch.stack([
        x_scaled_batch[:, 0:N],
        x_scaled_batch[:, N:2 * N],
        x_scaled_batch[:, 2 * N:3 * N],
        x_scaled_batch[:, 3 * N:4 * N],
    ], dim=-1)

    return Z.reshape(B * N, 4).to(device)


class SpectralGNN_ACOPF(nn.Module):
    """Two ChebConv layers with a local readout at the generator nodes.

    forward() expects a pre-collated disjoint graph: node features [B*N, 4], an edge
    index [2, B*E] and edge weights [B*E]. With predict_vm the output is
    [pg_non_slack | vm_gen]; otherwise it is pg_non_slack only, as in the paper.
    """

    def __init__(self, params, F1=128, F2=64, K=4, predict_vm=True):
        super().__init__()

        self.n_buses = params['general']['n_buses']
        self.predict_vm = predict_vm

        self.conv1 = ChebConv(in_channels=4, out_channels=F1, K=K)
        self.conv2 = ChebConv(in_channels=F1, out_channels=F2, K=K)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.pg_head = nn.Linear(F2, 1)
        if predict_vm:
            self.vm_head = nn.Linear(F2, 1)

        # The readout positions follow from the topology, which never changes during
        # a run, so they are built once here instead of on every forward pass
        bus_id_to_idx = params['general']['bus_id_to_idx']
        gen_bus_ids = params['general']['gen_bus_ids']
        non_slack_idx = params['general']['non_slack_gen_idx']

        self.register_buffer('gen_node_idx', torch.tensor(
            [bus_id_to_idx[int(g)] for g in gen_bus_ids], dtype=torch.long))
        self.register_buffer('ns_gen_node_idx', torch.tensor(
            [bus_id_to_idx[int(g)] for g in gen_bus_ids[non_slack_idx]],
            dtype=torch.long))

    def forward(self, node_feats, edge_index, edge_weight=None, batch_size=None):
        """Convolve over the batched graph and read out at the generator nodes."""
        N = self.n_buses
        if batch_size is None:
            batch_size = node_feats.shape[0] // N

        h = self.act1(self.conv1(node_feats, edge_index, edge_weight))
        h = self.act2(self.conv2(h, edge_index, edge_weight))

        # Node i of graph b sits at row b*N + i in the disjoint batch
        offsets = (torch.arange(batch_size, device=node_feats.device) * N).unsqueeze(1)

        batch_ns_idx = (self.ns_gen_node_idx.unsqueeze(0) + offsets).reshape(-1)
        pg = self.pg_head(h[batch_ns_idx]).squeeze(-1).reshape(batch_size, -1)

        if not self.predict_vm:
            return pg

        batch_gen_idx = (self.gen_node_idx.unsqueeze(0) + offsets).reshape(-1)
        vm = self.vm_head(h[batch_gen_idx]).squeeze(-1).reshape(batch_size, -1)
        return torch.cat([pg, vm], dim=-1)