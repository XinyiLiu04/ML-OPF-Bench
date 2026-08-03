# -*- coding: utf-8 -*-
"""
Spectral GNN Model for DCOPF  (Enhanced Node Features + Susceptance Kernel)

Version: v2.2 — Fixed generator-to-bus mapping for case300

Fixes in v2.2 (supersedes v2.1):
- ROOT CAUSE: params['general']['g_bus'] stores GENERATOR IDs (gen_id),
  NOT bus IDs. For case30/case118, gen_id happens to equal bus_id, so
  `bid - 1` worked by coincidence. For case300, gen_id=18 does not
  correspond to bus 18 (which doesn't even exist in case300).

- FIX: Use Map_g matrix to find each generator's connected bus index.
  Map_g shape is (n_gen, n_buses) [stored transposed in params].
  Map_g[gen_i, bus_idx] = 1 means generator i is connected to bus bus_idx.
  This is the ONLY reliable way to map generators to bus positions.

Architecture  (fixed 2-layer, matching ACOPF GNN)
------------
- Node features : [pd_i, is_gen_i, pg_min_i, pg_max_i, degree_i, neighbor_load_sum_i]
                   per bus  (6-dim)
- Graph convolution : 2 × ChebConv(K)
- Local readout     : per-node MLP(F2 → F2//2 → 1), non-slack generator nodes only
- Output            : pg_non_slack  [B, n_g_non_slack]
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

NODE_FEAT_DIM = 6


# =====================================================================
# Helper: extract generator → bus_idx mapping from Map_g
# =====================================================================
def _get_gen_bus_indices(params):
    """
    Extract 0-based bus index for each generator from the Map_g matrix.

    Map_g in params['constraints']['Map_g'] has shape (n_gen, n_buses):
        Map_g[gen_i, bus_idx] = 1  means generator i is at bus bus_idx.

    This is the ONLY correct way to find generator bus positions.
    Using g_bus (which stores gen_id, NOT bus_id) is WRONG for case300.

    Returns:
        gen_bus_idx : np.ndarray of int, shape (n_gen,)
            gen_bus_idx[i] = 0-based bus index for generator i
    """
    Map_g = params['constraints']['Map_g']  # (n_gen, n_buses)
    n_gen = Map_g.shape[0]
    gen_bus_idx = np.zeros(n_gen, dtype=int)
    for i in range(n_gen):
        bus_positions = np.where(Map_g[i, :] > 0)[0]
        if len(bus_positions) > 0:
            gen_bus_idx[i] = bus_positions[0]
        else:
            print(f"[WARNING] Generator {i} has no bus mapping in Map_g!")
            gen_bus_idx[i] = 0
    return gen_bus_idx


# =====================================================================
# Static Feature Builder  (v2.2 FIXED)
# =====================================================================
def build_static_node_features(params, edge_index):
    """
    Build per-node STATIC feature channels.

    v2.2 FIX:
    ---------
    Uses Map_g matrix to find each generator's bus index, instead of
    treating g_bus (gen_id) as bus_id. gen_id != bus_id in case300.

    Static features (per bus i):
      ch1: is_gen_i
      ch2: pg_min_i (scaled)
      ch3: pg_max_i (scaled)
      ch4: degree_i (normalized)
    """
    n_buses = params['general']['n_buses']
    Pg_min  = params['constraints']['Pg_min']
    Pg_max  = params['constraints']['Pg_max']

    if Pg_min.ndim == 2: Pg_min = Pg_min.ravel()
    if Pg_max.ndim == 2: Pg_max = Pg_max.ravel()

    # ------------------------------------------------------------------ #
    # FIX: Use Map_g to find generator bus positions                     #
    #                                                                     #
    # OLD (BROKEN):                                                       #
    #   idx = int(bid) - 1      # bid is gen_id, NOT bus_id!             #
    #   or: idx = bus_id_to_idx[int(bid)]  # bid is gen_id, not bus_id!  #
    #                                                                     #
    # NEW (CORRECT):                                                      #
    #   gen_bus_idx = extracted from Map_g matrix                        #
    # ------------------------------------------------------------------ #
    gen_bus_idx = _get_gen_bus_indices(params)  # (n_gen,)
    n_gen = len(gen_bus_idx)

    is_gen = np.zeros(n_buses, dtype='float32')
    pg_min_bus = np.zeros(n_buses, dtype='float32')
    pg_max_bus = np.zeros(n_buses, dtype='float32')

    for i in range(n_gen):
        bus_idx = gen_bus_idx[i]
        if 0 <= bus_idx < n_buses:
            is_gen[bus_idx] = 1.0
            pg_min_bus[bus_idx] += float(Pg_min[i])
            pg_max_bus[bus_idx] += float(Pg_max[i])

    pg_max_global = max(pg_max_bus.max(), 1e-8)
    pg_min_scaled = pg_min_bus / pg_max_global
    pg_max_scaled = pg_max_bus / pg_max_global

    # ch4: normalized node degree
    degree = np.zeros(n_buses, dtype='float32')
    ei_np = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
    for node_id in ei_np[0]:
        if node_id < n_buses:
            degree[node_id] += 1.0
    max_degree = max(degree.max(), 1.0)
    degree_norm = degree / max_degree

    static_feats = np.stack([is_gen, pg_min_scaled, pg_max_scaled, degree_norm], axis=-1)
    return torch.tensor(static_feats, dtype=torch.float32)


# =====================================================================
# Node Feature Builder  (unchanged — operates on matrix indices)
# =====================================================================
def build_node_features_dc(x_pd_scaled_batch, static_node_feats, edge_index, device):
    B, N = x_pd_scaled_batch.shape
    ch0 = x_pd_scaled_batch
    sf = static_node_feats.to(device)
    ch1 = sf[:, 0].unsqueeze(0).expand(B, -1)
    ch2 = sf[:, 1].unsqueeze(0).expand(B, -1)
    ch3 = sf[:, 2].unsqueeze(0).expand(B, -1)
    ch4 = sf[:, 3].unsqueeze(0).expand(B, -1)

    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)
    pd_at_src = x_pd_scaled_batch[:, src]
    neighbor_sum = torch.zeros(B, N, device=device, dtype=x_pd_scaled_batch.dtype)
    dst_expanded = dst.unsqueeze(0).expand(B, -1)
    neighbor_sum.scatter_add_(1, dst_expanded, pd_at_src)

    degree_count = torch.zeros(N, device=device)
    degree_count.scatter_add_(0, dst, torch.ones(dst.shape[0], device=device))
    degree_count = degree_count.clamp(min=1.0)
    ch5 = neighbor_sum / degree_count.unsqueeze(0)

    Z = torch.stack([ch0, ch1, ch2, ch3, ch4, ch5], dim=-1)
    return Z.reshape(B * N, NODE_FEAT_DIM).to(device)


# =====================================================================
# SpectralGNN_DCOPF  (v2.2 FIXED)
# =====================================================================
class SpectralGNN_DCOPF(nn.Module):
    """
    Local Spectral GNN for DCOPF.

    v2.2 FIX: ns_bus_idx now derived from Map_g matrix instead of
    treating g_bus (gen_id) as bus_id.
    """

    def __init__(self, F1=128, F2=64, K=4, params=None, edge_index=None):
        super().__init__()
        self.F1 = F1; self.F2 = F2; self.K = K; self.params = params
        n_buses = params['general']['n_buses']

        # Static node features
        assert edge_index is not None
        static_feats = build_static_node_features(params, edge_index)
        self.register_buffer('static_node_feats', static_feats)
        self.register_buffer('is_gen_mask', static_feats[:, 0])

        # ------------------------------------------------------------------ #
        # FIX: Non-slack generator bus indices — derive from Map_g           #
        #                                                                     #
        # OLD (BROKEN):                                                       #
        #   ns_bus_ids = g_bus[ns_idx]  # g_bus is gen_id, NOT bus_id!       #
        #   ns_bus_idx = [int(bid) - 1 ...]  or  [bus_id_to_idx[bid] ...]   #
        #                                                                     #
        # NEW (CORRECT):                                                      #
        #   gen_bus_idx[i] = bus matrix index for generator i (from Map_g)   #
        #   ns_bus_idx = gen_bus_idx[non_slack_gen_indices]                   #
        # ------------------------------------------------------------------ #
        gen_bus_idx = _get_gen_bus_indices(params)  # (n_gen,) 0-based bus indices
        ns_gen_idx  = params['general']['non_slack_gen_indices']
        ns_bus_idx  = torch.tensor(gen_bus_idx[ns_gen_idx], dtype=torch.long)
        self.register_buffer('ns_bus_idx', ns_bus_idx)

        # Store single-graph edge_index
        self.register_buffer('single_edge_index', edge_index.clone())

        # Graph convolution layers
        self.conv1 = ChebConv(in_channels=NODE_FEAT_DIM, out_channels=F1, K=K)
        self.conv2 = ChebConv(in_channels=F1, out_channels=F2, K=K)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        # Local readout head
        self.pg_head = nn.Sequential(
            nn.Linear(F2, F2 // 2), nn.ReLU(), nn.Linear(F2 // 2, 1))

    def forward(self, node_feats, edge_index, edge_weight=None,
                batch_size=None, params=None):
        if params is None: params = self.params
        device = node_feats.device
        N = params['general']['n_buses']
        if batch_size is None: batch_size = node_feats.shape[0] // N

        h = self.act1(self.conv1(node_feats, edge_index, edge_weight))
        h = self.act2(self.conv2(h, edge_index, edge_weight))

        ns_gen_idx = self.ns_bus_idx
        offsets = torch.arange(batch_size, device=device) * N
        batch_ns_idx = (ns_gen_idx.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        pg = self.pg_head(h[batch_ns_idx]).squeeze(-1)
        pg = pg.reshape(batch_size, -1)
        return pg