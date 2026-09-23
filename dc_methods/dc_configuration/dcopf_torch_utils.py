"""Torch counterparts of the shared DC physics, plus the model, training-loop and timing helpers."""

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from dc_configuration import dcopf_config


class MLP(nn.Module):
    """ReLU hidden layers; an optional sigmoid output bounds predictions to (0, 1)."""

    def __init__(self, input_size, output_size, hidden_sizes, sigmoid_output=False):
        super().__init__()
        layers, prev = [], input_size
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, output_size))
        if sigmoid_output:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DCTensors:
    """Constant network data on the device, built once instead of on every batch.

    Branch quantities cover constrained branches only, using the same mask as the evaluation,
    so a loss term and the reported violation always refer to the same set of branches.
    """

    def __init__(self, params, device):
        c, g = params['constraints'], params['general']

        def t(a):
            return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

        mask = c['constrained_branches']
        self.pg_min, self.pg_max = t(c['pg_min']), t(c['pg_max'])
        self.cost_c2, self.cost_c1, self.cost_c0 = t(c['cost_c2']), t(c['cost_c1']), t(c['cost_c0'])
        self.ptdf = t(c['ptdf'][mask])                 # (n_constrained, n_buses)
        self.rate = t(c['rate_a'][mask])
        self.gen_bus_map = t(c['gen_bus_map'])         # (n_buses, n_gens)
        self.n_constrained = int(mask.sum())

        # Selection matrices place non-slack outputs and the shared slack residual into the
        # full dispatch without in-place indexing, so gradients flow through both paths
        n_gens = g['n_gens']
        slack, non_slack = g['slack_gen_idx'], g['non_slack_gen_idx']
        place = np.zeros((len(non_slack), n_gens))
        place[np.arange(len(non_slack)), non_slack] = 1.0
        share = np.zeros(n_gens)
        share[slack] = 1.0 / len(slack)
        self.place_non_slack, self.slack_share = t(place), t(share)
        self.pg_min_ns, self.pg_max_ns = t(c['pg_min'][non_slack]), t(c['pg_max'][non_slack])
        self.slack_idx = torch.as_tensor(slack, dtype=torch.long, device=device)

    def full_pg(self, pg_non_slack, pd_bus):
        """Torch version of reconstruct_full_pg."""
        residual = pd_bus.sum(dim=1, keepdim=True) - pg_non_slack.sum(dim=1, keepdim=True)
        return pg_non_slack @ self.place_non_slack + residual * self.slack_share

    def flows(self, pg, pd_bus):
        return (pg @ self.gen_bus_map.T - pd_bus) @ self.ptdf.T

    def cost(self, pg):
        return (self.cost_c2 * pg ** 2 + self.cost_c1 * pg + self.cost_c0).sum(dim=1)

    def gen_violation(self, pg):
        return torch.relu(self.pg_min - pg) + torch.relu(pg - self.pg_max)

    def line_violation(self, pg, pd_bus):
        return torch.relu(self.flows(pg, pd_bus).abs() - self.rate)


def train_with_early_stopping(model, optimizer, train_tensors, batch_loss, val_loss,
                              n_epochs, batch_size, patience, min_delta,
                              after_step=None, epoch_log=None, min_batch=1):
    """Supervised training loop: restore the best validation checkpoint, stop on patience.

    batch_loss(model, batch) returns the training loss for a tuple of batch tensors, val_loss(model)
    the scalar used for model selection. after_step runs after every optimizer step (Lagrangian
    multiplier updates), epoch_log returns extra text for the progress line. Trailing batches
    smaller than min_batch are skipped, which BatchNorm needs when n_train % batch_size == 1.
    """
    n_train = len(train_tensors[0])
    device = train_tensors[0].device
    best_val, best_epoch, best_state, stale = float('inf'), 0, None, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < min_batch:
                continue
            optimizer.zero_grad()
            loss = batch_loss(model, tuple(x[idx] for x in train_tensors))
            loss.backward()
            optimizer.step()
            if after_step is not None:
                after_step()
            epoch_loss += loss.item() * len(idx)
            n_seen += len(idx)

        model.eval()
        with torch.no_grad():
            current = float(val_loss(model))
        if current < best_val - min_delta:
            best_val, best_epoch, best_state, stale = current, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0:
            extra = f"  {epoch_log()}" if epoch_log is not None else ""
            print(f"Epoch {epoch}/{n_epochs}  train {epoch_loss / n_seen:.3e}  val {current:.3e}  "
                  f"patience {stale}/{patience}{extra}")
        if stale >= patience:
            print(f"Early stop at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Validation loss never improved (non-finite loss?); no checkpoint to restore")
    model.load_state_dict(best_state)
    print(f"Restored best checkpoint: epoch {best_epoch}, val {best_val:.3e}")


def measure_latency(fn, device, n_warmup=10, n_repeats=100):
    """Mean wall-clock time of fn() in ms, synchronizing the device so queued kernels are counted."""
    with torch.no_grad():
        for _ in range(n_warmup):
            fn()
        dcopf_config.synchronize(device)
        times = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            fn()
            dcopf_config.synchronize(device)
            times.append(time.perf_counter() - start)
    return 1000.0 * float(np.mean(times))