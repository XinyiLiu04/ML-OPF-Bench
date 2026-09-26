"""KKT-informed neural network (Nellikkath & Chatzivasileiadis) with collocation points.

Two networks share the input: one predicts non-slack dispatch, the other the multipliers. The loss
is the supervised L1 error on both plus a weighted KKT residual; collocation rows keep only their
inputs and train on the residual alone.

Stationarity is imposed per generator with the dataset's JuMP signs,
    c1 + 2 c2 pg - lambda + mu_g_max - mu_g_min + (PTDF Cg)^T (mu_line_max - mu_line_min) = 0,
and vanishes on exact ground truth. The complementarity and dual-feasibility terms use the
scaled multipliers, as in the reference implementation, which keeps them O(1).
"""

from ml_opf_bench.runtime import TrainingState, is_managed, record_epoch

import time
from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from dc_configuration import dcopf_config
from dc_configuration.dcopf_data_setup import (
    load_duals, load_parameters_from_csv, load_samples, prepare_data_splits, reconstruct_full_pg,
)
from dc_configuration.dcopf_evaluation_metrics import evaluate_dispatch, print_metrics
from dc_configuration.dcopf_torch_utils import DCTensors, MLP, measure_latency, minmax_inverse

DUALS = ('lambda', 'mu_g_min', 'mu_g_max', 'mu_line_max', 'mu_line_min')
INEQUALITY_DUALS = DUALS[1:]


class KKTConstants:
    """Constants of the residual, as numpy arrays or device tensors depending on `array`."""

    def __init__(self, params, dual_scalers, load_scale, array):
        c = params['constraints']
        mask = c['constrained_branches']
        safe_max = np.where(c['pg_max'] > 0, c['pg_max'], 1.0)  # units with pg_max = 0 stay at 0 either way
        self.c1, self.c2 = array(c['cost_c1']), array(c['cost_c2'])
        self.p_max_norm, self.p_min_norm = array(c['pg_max'] / safe_max), array(c['pg_min'] / safe_max)
        self.safe_max = array(safe_max)
        self.ptdf, self.rate = array(c['ptdf'][mask]), array(c['rate_a'][mask])
        self.gen_bus_map = array(c['gen_bus_map'])
        self.flow_per_gen = array(c['ptdf'][mask] @ c['gen_bus_map'])
        self.shift = {n: array(dual_scalers[n].min_) for n in DUALS}
        self.scale = {n: array(dual_scalers[n].scale_) for n in DUALS}
        self.n_g, self.n_c, self.load_scale = len(c['pg_max']), max(int(mask.sum()), 1), load_scale


def _relu(v):
    return (v + abs(v)) / 2  # numpy- and torch-compatible


def kkt_residual(pg_full, duals_scaled, pd_bus, k):
    """Per-sample KKT error (primal + stationarity + complementarity + dual feasibility)."""
    p_norm = pg_full / k.safe_max
    flows = (pg_full @ k.gen_bus_map.T - pd_bus) @ k.ptdf.T
    over, under = flows - k.rate, -flows - k.rate
    line_scale = k.load_scale * k.n_c

    # The power-balance term of the reference is omitted: slack reconstruction zeroes it identically
    primal = ((_relu(p_norm - k.p_max_norm) + _relu(k.p_min_norm - p_norm)).sum(-1) / k.n_g
              + (_relu(over) + _relu(under)).sum(-1) / line_scale)

    phys = {n: (duals_scaled[n] - k.shift[n]) / k.scale[n] for n in DUALS}
    stationarity = (k.c1 + 2 * k.c2 * pg_full - phys['lambda'] + phys['mu_g_max'] - phys['mu_g_min']
                    + (phys['mu_line_max'] - phys['mu_line_min']) @ k.flow_per_gen)

    s = duals_scaled
    complementarity = ((abs(s['mu_g_max'] * (p_norm - k.p_max_norm))
                        + abs(s['mu_g_min'] * (k.p_min_norm - p_norm))).sum(-1) / k.n_g
                       + (abs(s['mu_line_max'] * over) + abs(s['mu_line_min'] * under)).sum(-1) / line_scale)
    dual_feasibility = sum(_relu(-s[n]).sum(-1) for n in INEQUALITY_DUALS)
    # /100 is the reference implementation's normalization of the stationarity term
    return primal + abs(stationarity).sum(-1) / 100.0 + complementarity + dual_feasibility


class KKTPinnNet(nn.Module):
    def __init__(self, n_inputs, n_non_slack, n_gens, n_constrained, hidden_sizes):
        super().__init__()
        self.pg_net = MLP(n_inputs, n_non_slack, hidden_sizes)
        layers, prev = [], n_inputs
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        self.dual_trunk = nn.Sequential(*layers)
        sizes = {'lambda': 1, 'mu_g_min': n_gens, 'mu_g_max': n_gens,
                 'mu_line_max': n_constrained, 'mu_line_min': n_constrained}
        self.dual_heads = nn.ModuleDict({n: nn.Linear(prev, sizes[n]) for n in DUALS})

    def forward(self, x):
        h = self.dual_trunk(x)
        return self.pg_net(x), {n: head(h) for n, head in self.dual_heads.items()}


def kkt_pinn_experiment(case_name, params_path, data_path,
                        n_train_use, seed, n_epochs, early_stop_patience, early_stop_min_delta,
                        learning_rate, hidden_sizes, batch_size, device,
                        dual_weight, kkt_weight, collocation_ratio):
    if not 0.0 <= collocation_ratio < 1.0:
        raise ValueError(f"collocation_ratio must lie in [0, 1), got {collocation_ratio}")
    torch.manual_seed(seed)
    device = torch.device(device)
    print(f"\nKKT-PINN on {case_name}, device {device}, dual weight {dual_weight}, "
          f"KKT weight {kkt_weight:.1e}, collocation ratio {collocation_ratio}\n")

    params = load_parameters_from_csv(case_name, params_path)
    pd_bus, pg = load_samples(data_path, params)
    duals = load_duals(data_path, params)
    train_idx, val_idx, test_idx = prepare_data_splits(len(pd_bus), n_train_use, seed)
    ns = params['general']['non_slack_gen_idx']

    shuffled = np.random.default_rng(seed + 1000).permutation(train_idx)
    n_col = int(len(train_idx) * collocation_ratio)
    sup_idx, col_idx = shuffled[:len(train_idx) - n_col], shuffled[len(train_idx) - n_col:]
    print(f"Training split: {len(sup_idx)} supervised, {len(col_idx)} collocation (labels unused)")

    # Inputs are known for every training row; label scalers see supervised rows only, since the
    # collocation rows' labels are by definition unavailable
    x_scaler = MinMaxScaler().fit(pd_bus[train_idx])
    y_scaler = MinMaxScaler().fit(pg[sup_idx][:, ns])
    dual_scalers = {n: MinMaxScaler().fit(duals[n][sup_idx]) for n in DUALS}

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    def targets(rows):
        return {'pg': to_tensor(y_scaler.transform(pg[rows][:, ns])),
                **{n: to_tensor(dual_scalers[n].transform(duals[n][rows])) for n in DUALS}}

    X = {name: to_tensor(x_scaler.transform(pd_bus[rows]))
         for name, rows in (('sup', sup_idx), ('col', col_idx), ('val', val_idx), ('test', test_idx))}
    Pd = {name: to_tensor(pd_bus[rows]) for name, rows in (('sup', sup_idx), ('col', col_idx), ('val', val_idx))}
    T_sup, T_val = targets(sup_idx), targets(val_idx)

    net = DCTensors(params, device)
    load_scale = float(pd_bus[train_idx].max())  # reference normalizer: largest single-bus load
    k_torch = KKTConstants(params, dual_scalers, load_scale,
                           lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device))
    decode_pg = minmax_inverse(y_scaler, device)

    k_np = KKTConstants(params, dual_scalers, load_scale, np.asarray)
    floor = kkt_residual(pg[sup_idx], {n: dual_scalers[n].transform(duals[n][sup_idx]) for n in DUALS},
                         pd_bus[sup_idx], k_np)
    print(f"KKT residual of the ground-truth supervised labels: mean {floor.mean():.3e}, max {floor.max():.3e}")

    model = KKTPinnNet(pd_bus.shape[1], len(ns), params['general']['n_gens'],
                       int(params['constraints']['constrained_branches'].sum()), hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    l1 = nn.L1Loss()

    def residual(pg_scaled, duals_scaled, pd_b):
        return kkt_residual(net.full_pg(decode_pg(pg_scaled), pd_b), duals_scaled, pd_b, k_torch)

    def supervised_loss(x, t, pd_b):
        pg_s, duals_s = model(x)
        return (l1(pg_s, t['pg']) + dual_weight * sum(l1(duals_s[n], t[n]) for n in DUALS)
                + kkt_weight * residual(pg_s, duals_s, pd_b).mean())

    def collocation_loss(x, pd_b):
        return kkt_weight * residual(*model(x), pd_b).mean()

    col_share = len(sup_idx) / len(train_idx)
    best_val, best_epoch, best_state, stale = float('inf'), 0, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        record_epoch(epoch)
        model.train()
        sup_batches = torch.randperm(len(sup_idx), device=device).split(batch_size)
        col_batches = torch.randperm(len(col_idx), device=device).split(batch_size) if len(col_idx) else ()
        sums = {'sup': 0.0, 'col': 0.0}
        # Paired supervised/collocation steps; the longer stream continues alone once the other ends
        for s_b, c_b in zip_longest(sup_batches, col_batches):
            optimizer.zero_grad()
            loss = 0.0
            if s_b is not None:
                sup = supervised_loss(X['sup'][s_b], {n: v[s_b] for n, v in T_sup.items()}, Pd['sup'][s_b])
                loss, sums['sup'] = loss + sup, sums['sup'] + sup.item() * len(s_b)
            if c_b is not None:
                col = collocation_loss(X['col'][c_b], Pd['col'][c_b])
                loss, sums['col'] = loss + col_share * col, sums['col'] + col.item() * len(c_b)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val = supervised_loss(X['val'], T_val, Pd['val']).item()
        if val < best_val - early_stop_min_delta:
            best_val, best_epoch, stale = val, epoch, 0
            best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}/{n_epochs}  sup {sums['sup'] / len(sup_idx):.3e}  "
                  f"col {sums['col'] / max(len(col_idx), 1):.3e}  val {val:.3e}  patience {stale}/{early_stop_patience}")
        if stale >= early_stop_patience:
            print(f"Early stop at epoch {epoch}")
            break
    if best_state is None:
        raise RuntimeError("Validation loss never improved (non-finite loss?); no checkpoint to restore")
    model.load_state_dict(best_state)
    train_time = time.perf_counter() - t0
    if is_managed():
        return TrainingState(model, params, train_time, dict(x_scaler=x_scaler, y_scaler=y_scaler))
    print(f"Restored best checkpoint: epoch {best_epoch}, val {best_val:.3e}")

    model.eval()
    with torch.no_grad():
        pg_ns = y_scaler.inverse_transform(model.pg_net(X['test']).cpu().numpy().astype(np.float64))
        test_residual = residual(*model(X['test']), to_tensor(pd_bus[test_idx])).mean().item()
    pg_pred = reconstruct_full_pg(pg_ns, pd_bus[test_idx], params)

    metrics = evaluate_dispatch(pg_pred, pg[test_idx], pd_bus[test_idx], params)
    metrics['train_time_s'] = train_time
    metrics['inference_ms'] = measure_latency(lambda: model.pg_net(X['test'][:1]), device)
    metrics['inference_scope'] = 'dispatch network forward pass only, batch of 1; the multiplier network is not needed'
    metrics['kkt_residual_test'] = test_residual
    metrics['kkt_residual_ground_truth'] = float(floor.mean())
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    DUAL_WEIGHT = 0.05       # on each multiplier's L1 error, relative to the dispatch error
    KKT_WEIGHT = 5e-10       # the effective value the previous version ran with (0.05 x a hidden 1e-8)
    COLLOCATION_RATIO = 0.5  # share of training rows whose labels are discarded

    dcopf_config.print_config()
    kkt_pinn_experiment(**dcopf_config.get_all_paths(), **dcopf_config.get_all_params(),
                        dual_weight=DUAL_WEIGHT, kkt_weight=KKT_WEIGHT, collocation_ratio=COLLOCATION_RATIO)