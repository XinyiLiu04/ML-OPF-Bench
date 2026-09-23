"""DeepOPF-NGT (Huang, Chen, Low, IEEE TPWRS 2024), unsupervised: minimize cost plus adaptively weighted violations."""

import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from dc_configuration import dcopf_config
from dc_configuration.dcopf_data_setup import (
    load_parameters_from_csv, load_samples, prepare_data_splits, reconstruct_full_pg,
)
from dc_configuration.dcopf_evaluation_metrics import evaluate_dispatch, print_metrics
from dc_configuration.dcopf_torch_utils import DCTensors, MLP, measure_latency

LOSS_TERMS = ('L_obj', 'L_slack', 'L_line')


def non_slack_from_sigmoid(s, net):
    """Map sigmoid outputs onto [pg_min, pg_max]; the non-slack bounds then hold by construction."""
    return net.pg_min_ns + s * (net.pg_max_ns - net.pg_min_ns)


def physics_losses(pg_non_slack, pd_bus, net):
    """Cost and squared violations, averaged over the batch.

    No non-slack bound term: the sigmoid mapping makes it identically zero, so only the slack
    generators, whose output is whatever closes the balance, can leave their limits.
    """
    pg_full = net.full_pg(pg_non_slack, pd_bus)
    slack_viol = net.gen_violation(pg_full)[:, net.slack_idx]
    return {
        'L_obj': net.cost(pg_full).mean(),
        'L_slack': (slack_viol ** 2).sum(dim=1).mean(),
        'L_line': (net.line_violation(pg_full, pd_bus) ** 2).sum(dim=1).mean(),
    }


class AdaptiveWeights:
    """Loss references fixed after epoch 1, EMA of the normalized terms, three-regime weight update.

    Per constraint term, with ema the EMA of its normalized loss:
      ema < 0.02          satisfied: decay the weight toward its initial value, never above it
      0.02 <= ema < 0.30  balance against the objective, k = k_obj * ema_obj / ema
      ema >= 0.30         boost in proportion to the violation, k = 50 * ema
    """

    K_OBJ = 0.1
    K_INIT = {'L_slack': 100.0, 'L_line': 10.0}
    K_MAX = {'L_slack': 1e4, 'L_line': 200.0}
    K_MIN = 1.0
    EMA_ALPHA, SATISFIED, BOOST, DECAY, BOOST_BASE = 0.3, 0.02, 0.30, 0.95, 50.0

    def __init__(self):
        self.k = dict(self.K_INIT)
        self.refs = {name: 1.0 for name in LOSS_TERMS}
        self.ema = None

    def normalized(self, losses):
        return {name: losses[name] / self.refs[name] for name in LOSS_TERMS}

    def constraint_loss(self, losses):
        n = self.normalized(losses)
        return sum(self.k[name] * n[name] for name in self.K_INIT)

    def total_loss(self, losses):
        return self.K_OBJ * self.normalized(losses)['L_obj'] + self.constraint_loss(losses)

    def unit_score(self, losses):
        """Unit-weight sum of normalized terms; comparable across epochs because refs are fixed."""
        return float(sum(self.normalized(losses).values()))

    def end_epoch(self, epoch, raw_means):
        if epoch == 1:
            self.refs = {name: v if v > 1e-8 else 1.0 for name, v in raw_means.items()}
            self.ema = {name: 1.0 for name in LOSS_TERMS}
            return
        norm = {name: raw_means[name] / self.refs[name] for name in LOSS_TERMS}
        self.ema = {name: self.EMA_ALPHA * norm[name] + (1 - self.EMA_ALPHA) * self.ema[name]
                    for name in LOSS_TERMS}
        for name, k_init in self.K_INIT.items():
            ema = self.ema[name]
            if ema < self.SATISFIED:
                # max(k * decay, k_init) alone would snap a weight that is already below its
                # initial value back up to it, the opposite of a decay
                if self.k[name] > k_init:
                    self.k[name] = max(self.k[name] * self.DECAY, k_init)
            elif ema >= self.BOOST:
                self.k[name] = float(np.clip(self.BOOST_BASE * ema, self.K_MIN, self.K_MAX[name]))
            elif ema > 1e-8:
                self.k[name] = float(np.clip(self.K_OBJ * self.ema['L_obj'] / ema, self.K_MIN, self.K_MAX[name]))


def unsupervised_epoch(model, optimizer, weights, X, Pd, batch_size, net):
    """One pass over (X, Pd) on the weighted physics loss; returns sample-weighted raw loss means."""
    model.train()
    sums = {name: 0.0 for name in LOSS_TERMS}
    perm = torch.randperm(len(X), device=X.device)
    for start in range(0, len(X), batch_size):
        idx = perm[start:start + batch_size]
        optimizer.zero_grad()
        losses = physics_losses(non_slack_from_sigmoid(model(X[idx]), net), Pd[idx], net)
        weights.total_loss(losses).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for name in LOSS_TERMS:
            sums[name] += losses[name].item() * len(idx)
    return {name: sums[name] / len(X) for name in LOSS_TERMS}


def validation_score(model, weights, X_val, Pd_val, net):
    model.eval()
    with torch.no_grad():
        return weights.unit_score(physics_losses(non_slack_from_sigmoid(model(X_val), net), Pd_val, net))


def log_epoch(epoch, n_epochs, raw, weights, val_score):
    print(f"Epoch {epoch}/{n_epochs}  L_obj {raw['L_obj']:.4e}  L_slack {raw['L_slack']:.3e}  "
          f"L_line {raw['L_line']:.3e}  k_slack {weights.k['L_slack']:.1f}  k_line {weights.k['L_line']:.1f}  "
          f"val(unit) {val_score:.4f}")


def evaluate_sigmoid_model(model, X_test, pd_test, pg_test, params, net):
    model.eval()
    with torch.no_grad():
        pg_non_slack = non_slack_from_sigmoid(model(X_test), net).cpu().numpy().astype(np.float64)
    return evaluate_dispatch(reconstruct_full_pg(pg_non_slack, pd_test, params), pg_test, pd_test, params)


def ngt_experiment(case_name, params_path, data_path,
                   n_train_use, seed, n_epochs, learning_rate, hidden_sizes, batch_size, device, **ignored):
    # Unsupervised: no label to overfit, so no early stopping and no checkpoint restore; the final
    # epoch is returned and the unit-weight validation score is printed for inspection only
    dcopf_config.print_ignored('NGT', **ignored)
    torch.manual_seed(seed)
    device = torch.device(device)
    print(f"\nDeepOPF-NGT on {case_name}, device {device}\n")

    params = load_parameters_from_csv(case_name, params_path)
    pd_bus, pg = load_samples(data_path, params)
    train_idx, val_idx, test_idx = prepare_data_splits(len(pd_bus), n_train_use, seed)
    net = DCTensors(params, device)

    x_scaler = MinMaxScaler().fit(pd_bus[train_idx])

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    X_train, X_val, X_test = (to_tensor(x_scaler.transform(pd_bus[i])) for i in (train_idx, val_idx, test_idx))
    Pd_train, Pd_val = to_tensor(pd_bus[train_idx]), to_tensor(pd_bus[val_idx])

    model = MLP(pd_bus.shape[1], len(params['general']['non_slack_gen_idx']), hidden_sizes,
                sigmoid_output=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = AdaptiveWeights()

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        raw = unsupervised_epoch(model, optimizer, weights, X_train, Pd_train, batch_size, net)
        weights.end_epoch(epoch, raw)
        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            log_epoch(epoch, n_epochs, raw, weights, validation_score(model, weights, X_val, Pd_val, net))
    train_time = time.perf_counter() - t0

    metrics = evaluate_sigmoid_model(model, X_test, pd_bus[test_idx], pg[test_idx], params, net)
    metrics['train_time_s'] = train_time
    metrics['inference_ms'] = measure_latency(lambda: model(X_test[:1]), device)
    metrics['inference_scope'] = 'forward pass only, batch of 1; excludes scaling and slack reconstruction'
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    dcopf_config.print_config()
    ngt_experiment(**dcopf_config.get_all_paths(), **dcopf_config.get_all_params())