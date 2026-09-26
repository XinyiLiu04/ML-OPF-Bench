"""Spectral GNN (Owerko et al., ICASSP 2020), supervised: ChebConv over the bus graph, per-generator readout."""

from ml_opf_bench.runtime import TrainingState, is_managed

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import ChebConv

from dc_configuration import dcopf_config
from dc_configuration.dcopf_data_setup import (
    load_parameters_from_csv, load_samples, prepare_data_splits, reconstruct_full_pg,
)
from dc_configuration.dcopf_evaluation_metrics import evaluate_dispatch, print_metrics
from dc_configuration.dcopf_torch_utils import measure_latency, train_with_early_stopping

NODE_FEAT_DIM = 6  # [pd, is_gen, pg_min, pg_max, degree, neighbor mean pd]


def load_graph(case_name, params_path, params, kernel, gaussian_scale):
    """Bidirectional edges (2, 2 n_branches) and weights, with nodes indexed like every other per-bus array.

    Node positions come from bus_ids.csv through params, the same lookup the samples are loaded
    with; a lookup rebuilt from branch endpoints only agrees with it while every bus has a branch.
    """
    df = pd.read_csv(os.path.join(params_path, f"{case_name}_branch_info.csv"))
    lookup = params['general']['bus_id_to_idx']
    unknown = sorted((set(df['f_bus']) | set(df['t_bus'])) - set(lookup))
    if unknown:
        raise ValueError(f"branch_info.csv references buses absent from bus_ids.csv: {unknown[:10]}")
    if sorted(df['branch_id']) != sorted(params['general']['branch_ids']):
        raise ValueError("branch_info.csv and branch_limits.csv list different branches")

    f = df['f_bus'].map(lookup).to_numpy()
    t = df['t_bus'].map(lookup).to_numpy()
    r, x = df['r_pu'].to_numpy(dtype=np.float64), df['x_pu'].to_numpy(dtype=np.float64)
    if kernel == 'susceptance':
        if np.any(np.abs(x) < 1e-12):
            raise ValueError("Zero-reactance branch; its susceptance weight is undefined")
        w = 1.0 / np.abs(x)
    elif kernel == 'gaussian':
        w = np.exp(-gaussian_scale * (r ** 2 + x ** 2))
    elif kernel == 'uniform':
        w = np.ones(len(df))
    else:
        raise ValueError(f"Unknown graph kernel '{kernel}', expected susceptance/gaussian/uniform")
    w = w / w.max()

    edge_index = np.stack([np.concatenate([f, t]), np.concatenate([t, f])])
    return edge_index, np.concatenate([w, w])


def static_node_features(params, edge_index):
    """(n_buses, 4): is_gen, pg_min and pg_max summed per bus (scaled by the largest bus pg_max), degree."""
    c = params['constraints']
    n_buses = params['general']['n_buses']
    pg_min_bus = c['gen_bus_map'] @ c['pg_min']
    pg_max_bus = c['gen_bus_map'] @ c['pg_max']
    scale = max(pg_max_bus.max(), 1e-8)
    degree = np.bincount(edge_index[0], minlength=n_buses).astype(np.float64)
    return np.stack([(c['gen_bus_map'].sum(axis=1) > 0).astype(np.float64),
                     pg_min_bus / scale, pg_max_bus / scale, degree / max(degree.max(), 1.0)], axis=1)


def neighbor_mean_matrix(edge_index, n_buses):
    """M with pd @ M the mean load over each bus's incoming neighbors, precomputed once."""
    M = np.zeros((n_buses, n_buses))
    np.add.at(M, (edge_index[0], edge_index[1]), 1.0)
    return M / np.maximum(M.sum(axis=0), 1.0)


class SpectralGNN(nn.Module):
    """Two ChebConv layers over disjoint copies of the bus graph, then a shared MLP read at generator buses.

    Generators that share a bus read the same node embedding and therefore receive identical
    predictions; the readout cannot tell co-located units apart.
    """

    def __init__(self, params, edge_index, edge_weight, hidden_sizes, K):
        super().__init__()
        if len(hidden_sizes) != 2:
            raise ValueError(f"SpectralGNN has exactly two graph layers, got hidden_sizes={hidden_sizes}")
        f1, f2 = hidden_sizes
        n_buses = params['general']['n_buses']
        gen_bus = np.argmax(params['constraints']['gen_bus_map'], axis=0)  # bus position of each generator

        def buf(name, a, dtype=torch.float32):
            self.register_buffer(name, torch.as_tensor(np.asarray(a), dtype=dtype))

        buf('static', static_node_features(params, edge_index))
        buf('neighbor_mean', neighbor_mean_matrix(edge_index, n_buses))
        buf('edge_index', edge_index, torch.long)
        buf('edge_weight', edge_weight)
        buf('ns_bus', gen_bus[params['general']['non_slack_gen_idx']], torch.long)
        self.n_buses = n_buses
        self._batched = {}

        self.conv1 = ChebConv(NODE_FEAT_DIM, f1, K=K)
        self.conv2 = ChebConv(f1, f2, K=K)
        self.head = nn.Sequential(nn.Linear(f2, f2 // 2), nn.ReLU(), nn.Linear(f2 // 2, 1))

    def batched_graph(self, B):
        """Edges of B disjoint graph copies, cached per batch size and device."""
        key = (B, self.edge_index.device)
        if key not in self._batched:
            offsets = torch.arange(B, device=self.edge_index.device).repeat_interleave(self.edge_index.shape[1]) * self.n_buses
            self._batched[key] = (self.edge_index.repeat(1, B) + offsets, self.edge_weight.repeat(B))
        return self._batched[key]

    def forward(self, pd_scaled):
        B, N = pd_scaled.shape
        feats = torch.cat([pd_scaled.unsqueeze(-1), self.static.expand(B, N, -1),
                           (pd_scaled @ self.neighbor_mean).unsqueeze(-1)], dim=-1)
        ei, ew = self.batched_graph(B)
        h = torch.relu(self.conv1(feats.reshape(B * N, NODE_FEAT_DIM), ei, ew))
        h = torch.relu(self.conv2(h, ei, ew))
        return self.head(h.view(B, N, -1)[:, self.ns_bus]).squeeze(-1)


def predict_chunked(model, X, chunk=512):
    """Forward pass in chunks; one batch of a whole split can be B * n_buses nodes wide."""
    return torch.cat([model(X[s:s + chunk]) for s in range(0, len(X), chunk)])


def gnn_experiment(case_name, params_path, data_path,
                   n_train_use, seed, n_epochs, early_stop_patience, early_stop_min_delta,
                   learning_rate, hidden_sizes, batch_size, device, K, graph_kernel, gaussian_scale):
    torch.manual_seed(seed)
    device = torch.device(device)
    print(f"\nSpectral GNN on {case_name}, device {device}, K {K}, kernel {graph_kernel}\n")

    params = load_parameters_from_csv(case_name, params_path)
    pd_bus, pg = load_samples(data_path, params)
    train_idx, val_idx, test_idx = prepare_data_splits(len(pd_bus), n_train_use, seed)
    non_slack = params['general']['non_slack_gen_idx']
    edge_index, edge_weight = load_graph(case_name, params_path, params, graph_kernel, gaussian_scale)
    print(f"Graph: {params['general']['n_buses']} nodes, {edge_index.shape[1]} directed edges")

    x_scaler = MinMaxScaler().fit(pd_bus[train_idx])
    y_scaler = MinMaxScaler().fit(pg[train_idx][:, non_slack])

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    X_train, X_val, X_test = (to_tensor(x_scaler.transform(pd_bus[i])) for i in (train_idx, val_idx, test_idx))
    Y_train, Y_val = (to_tensor(y_scaler.transform(pg[i][:, non_slack])) for i in (train_idx, val_idx))

    model = SpectralGNN(params, edge_index, edge_weight, hidden_sizes, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.perf_counter()
    train_with_early_stopping(
        model, optimizer, (X_train, Y_train),
        batch_loss=lambda m, b: mse(m(b[0]), b[1]),
        val_loss=lambda m: mse(predict_chunked(m, X_val), Y_val).item(),
        n_epochs=n_epochs, batch_size=batch_size,
        patience=early_stop_patience, min_delta=early_stop_min_delta)
    train_time = time.perf_counter() - t0
    if is_managed():
        return TrainingState(model, params, train_time, dict(x_scaler=x_scaler, y_scaler=y_scaler))

    model.eval()
    with torch.no_grad():
        pg_non_slack = y_scaler.inverse_transform(predict_chunked(model, X_test).cpu().numpy().astype(np.float64))
    pg_pred = reconstruct_full_pg(pg_non_slack, pd_bus[test_idx], params)

    metrics = evaluate_dispatch(pg_pred, pg[test_idx], pd_bus[test_idx], params)
    metrics['train_time_s'] = train_time
    metrics['inference_ms'] = measure_latency(lambda: model(X_test[:1]), device)
    metrics['inference_scope'] = ('forward pass incl. node-feature assembly, batch of 1; '
                                  'excludes scaling and slack reconstruction')
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    K = 4                       # Chebyshev polynomial order
    GRAPH_KERNEL = 'susceptance'
    GAUSSIAN_SCALE = 0.01       # used only by the gaussian kernel

    dcopf_config.print_config()
    gnn_experiment(**dcopf_config.get_all_paths(), **dcopf_config.get_all_params(),
                   K=K, graph_kernel=GRAPH_KERNEL, gaussian_scale=GAUSSIAN_SCALE)