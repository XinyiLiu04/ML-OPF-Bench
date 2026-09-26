"""Active-set classification (Deka & Misra, arXiv:1902.05607): classify the binding constraints, then solve for Pg.

The class vocabulary is built from the training split only. A test sample whose active set never
occurs in training cannot be predicted correctly; it counts as misclassified, and test_unseen_rate
reports the resulting ceiling on accuracy.
"""

from ml_opf_bench.runtime import TrainingState, is_managed

import time

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linprog
from sklearn.preprocessing import MinMaxScaler

from dc_configuration import dcopf_config
from dc_configuration.dcopf_data_setup import load_duals, load_parameters_from_csv, load_samples, prepare_data_splits
from dc_configuration.dcopf_evaluation_metrics import compute_cost, evaluate_dispatch, print_metrics
from dc_configuration.dcopf_torch_utils import measure_latency, train_with_early_stopping

UNSEEN = -1


class ActiveSetClassifier(nn.Module):
    """[Linear -> ReLU -> BatchNorm -> Dropout] per hidden layer, then logits over the vocabulary."""

    def __init__(self, input_size, n_classes, hidden_sizes, dropout_rate):
        super().__init__()
        layers, prev = [], input_size
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU(), nn.BatchNorm1d(size), nn.Dropout(dropout_rate)]
            prev = size
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def active_matrix(duals, threshold):
    """Binary (n_samples, 2 n_gens + 2 n_constrained) matrix: [g_min | g_max | line_max | line_min]."""
    stacked = np.hstack([duals['mu_g_min'], duals['mu_g_max'], duals['mu_line_max'], duals['mu_line_min']])
    return (stacked > threshold).astype(np.int8)


def build_labels(active, train_idx):
    """Vocabulary from training rows only; rows with an active set absent from training get UNSEEN."""
    vocab, train_labels = np.unique(active[train_idx], axis=0, return_inverse=True)
    lookup = {row.tobytes(): i for i, row in enumerate(vocab)}
    labels = np.array([lookup.get(row.tobytes(), UNSEEN) for row in active], dtype=np.int64)
    assert np.array_equal(labels[train_idx], train_labels.ravel())
    return vocab, labels


class ActiveSetRecovery:
    """Solve for Pg with the predicted active constraints held as equalities.

    Active generator bounds fix those units; the remaining units solve
        min c1 . pg_free  s.t. power balance, active line limits as equalities, generator bounds
    with the linear cost of the paper. Returns None when the active set is infeasible at this load
    rather than relaxing it, since a relaxed solution is cheap, infeasible and would win a Top-K
    comparison on cost.
    """

    def __init__(self, params):
        c = params['constraints']
        mask = c['constrained_branches']
        self.pg_min, self.pg_max, self.c1 = c['pg_min'], c['pg_max'], c['cost_c1']
        self.ptdf, self.rate = c['ptdf'][mask], c['rate_a'][mask]
        self.flow_per_gen = self.ptdf @ c['gen_bus_map']
        self.n_gens, self.n_lines = len(self.pg_min), len(self.rate)
        if np.any(c['cost_c2'] != 0):
            print("[AS] Warning: quadratic cost terms present; recovery solves the linear-cost LP")

    def recover(self, active_row, pd_i, balance_tol=1e-6):
        g, n = self.n_gens, self.n_lines
        at_min, at_max = active_row[:g].astype(bool), active_row[g:2 * g].astype(bool)
        line_max, line_min = active_row[2 * g:2 * g + n].astype(bool), active_row[2 * g + n:].astype(bool)

        pg = np.full(g, np.nan)
        pg[at_min], pg[at_max] = self.pg_min[at_min], self.pg_max[at_max]
        both = at_min & at_max  # a unit reported at both bounds, typically pg_min == pg_max
        pg[both] = 0.5 * (self.pg_min[both] + self.pg_max[both])
        free = np.isnan(pg)
        fixed = ~free
        residual = pd_i.sum() - pg[fixed].sum()

        if not free.any():
            return pg if abs(residual) <= balance_tol else None

        offset = self.ptdf @ pd_i - self.flow_per_gen[:, fixed] @ pg[fixed]
        A_eq = np.vstack([np.ones(free.sum()), self.flow_per_gen[line_max][:, free],
                          self.flow_per_gen[line_min][:, free]])
        b_eq = np.concatenate([[residual], self.rate[line_max] + offset[line_max],
                               -self.rate[line_min] + offset[line_min]])
        res = linprog(self.c1[free], A_eq=A_eq, b_eq=b_eq,
                      bounds=list(zip(self.pg_min[free], self.pg_max[free])), method='highs')
        if not res.success:
            return None
        pg[free] = res.x
        return pg

    def feasible(self, pg, pd_i, tol=1e-3):
        within_box = np.all(pg >= self.pg_min - tol) and np.all(pg <= self.pg_max + tol)
        flows = self.flow_per_gen @ pg - self.ptdf @ pd_i
        return within_box and np.all(np.abs(flows) <= self.rate + tol)


def fallback_dispatch(pd_i, params):
    """Uniform split clipped to bounds: deliberately poor, so recovery failures degrade the metrics visibly."""
    c = params['constraints']
    n = len(c['pg_min'])
    return np.clip(np.full(n, pd_i.sum() / n), c['pg_min'], c['pg_max'])


def recover_top1(pred, vocab, pd_rows, recovery, params):
    out, n_failed = np.empty((len(pred), recovery.n_gens)), 0
    for i, (lbl, pd_i) in enumerate(zip(pred, pd_rows)):
        pg = recovery.recover(vocab[lbl], pd_i)
        if pg is None:
            pg, n_failed = fallback_dispatch(pd_i, params), n_failed + 1
        out[i] = pg
    return out, n_failed


def recover_topk(topk, vocab, pd_rows, recovery, params):
    """Among the K candidates, keep those that recover to a feasible dispatch and take the cheapest."""
    out, n_failed = np.empty((len(topk), recovery.n_gens)), 0
    for i, (labels, pd_i) in enumerate(zip(topk, pd_rows)):
        candidates = [pg for pg in (recovery.recover(vocab[lbl], pd_i) for lbl in labels)
                      if pg is not None and recovery.feasible(pg, pd_i)]
        if candidates:
            out[i] = min(candidates, key=lambda pg: compute_cost(pg[None, :], params)[0])
        else:
            out[i], n_failed = fallback_dispatch(pd_i, params), n_failed + 1
    return out, n_failed


def as_experiment(case_name, params_path, data_path,
                  n_train_use, seed, n_epochs, early_stop_patience, early_stop_min_delta,
                  learning_rate, hidden_sizes, batch_size, device,
                  active_threshold, top_k, dropout_rate):
    torch.manual_seed(seed)
    device = torch.device(device)
    print(f"\nActive-set classification on {case_name}, device {device}\n")

    params = load_parameters_from_csv(case_name, params_path)
    pd_bus, pg = load_samples(data_path, params)
    train_idx, val_idx, test_idx = prepare_data_splits(len(pd_bus), n_train_use, seed)
    vocab, labels = build_labels(active_matrix(load_duals(data_path, params), active_threshold), train_idx)
    test_labels = labels[test_idx]
    test_unseen_rate = float(np.mean(test_labels == UNSEEN))
    print(f"{len(vocab)} active sets in the training split; "
          f"{100 * np.mean(labels[val_idx] == UNSEEN):.2f}% of val and "
          f"{100 * test_unseen_rate:.2f}% of test samples have an unseen active set")

    x_scaler = MinMaxScaler().fit(pd_bus[train_idx])

    def to_tensor(a, dtype=torch.float32):
        return torch.tensor(a, dtype=dtype, device=device)

    X_train, X_val, X_test = (to_tensor(x_scaler.transform(pd_bus[i])) for i in (train_idx, val_idx, test_idx))
    Y_train, Y_val = (to_tensor(labels[i], torch.long) for i in (train_idx, val_idx))

    model = ActiveSetClassifier(pd_bus.shape[1], len(vocab), hidden_sizes, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ce = nn.CrossEntropyLoss(ignore_index=UNSEEN)  # unseen validation rows have no class to score

    t0 = time.perf_counter()
    train_with_early_stopping(
        model, optimizer, (X_train, Y_train),
        batch_loss=lambda m, b: ce(m(b[0]), b[1]),
        val_loss=lambda m: ce(m(X_val), Y_val).item(),
        n_epochs=n_epochs, batch_size=batch_size,
        patience=early_stop_patience, min_delta=early_stop_min_delta,
        min_batch=2)  # BatchNorm cannot normalize a batch of one
    train_time = time.perf_counter() - t0
    if is_managed():
        return TrainingState(model, params, train_time, dict(x_scaler=x_scaler, vocab=vocab, top_k=top_k))

    model.eval()
    k = min(top_k, len(vocab))
    with torch.no_grad():
        logits = model(X_test)
        top1 = logits.argmax(dim=1).cpu().numpy()
        topk = torch.topk(logits, k, dim=1).indices.cpu().numpy()

    recovery = ActiveSetRecovery(params)
    pd_test = pd_bus[test_idx]
    seen = test_labels != UNSEEN
    true_ok = sum(1 for lbl, pd_i in zip(test_labels[seen], pd_test[seen])
                  if (p := recovery.recover(vocab[lbl], pd_i)) is not None and recovery.feasible(p, pd_i))

    pg_top1, fail_top1 = recover_top1(top1, vocab, pd_test, recovery, params)
    pg_topk, fail_topk = recover_topk(topk, vocab, pd_test, recovery, params)

    nn_ms = measure_latency(lambda: model(X_test[:1]), device)
    top1_ms = measure_latency(lambda: recover_top1(top1[:1], vocab, pd_test[:1], recovery, params), device)
    topk_ms = measure_latency(lambda: recover_topk(topk[:1], vocab, pd_test[:1], recovery, params), device)

    shared = {
        'top1_accuracy': float(np.mean(top1 == test_labels)),
        f'top{k}_accuracy': float(np.mean((topk == test_labels[:, None]).any(axis=1))),
        'test_unseen_rate': test_unseen_rate,
        # Ceiling set by the extraction threshold and the recovery LP, independent of the classifier
        'true_label_recovery_rate': true_ok / max(int(seen.sum()), 1),
        'n_classes': len(vocab),
    }
    m1 = evaluate_dispatch(pg_top1, pg[test_idx], pd_test, params)
    m1.update(train_time_s=train_time, inference_ms=nn_ms + top1_ms,
              inference_scope='forward pass plus one recovery LP, batch of 1',
              recovery_failure_rate=fail_top1 / len(test_idx), **shared)
    mk = evaluate_dispatch(pg_topk, pg[test_idx], pd_test, params)
    mk.update(train_time_s=train_time, inference_ms=nn_ms + topk_ms,
              inference_scope=f'forward pass plus {k} recovery LPs, batch of 1',
              recovery_failure_rate=fail_topk / len(test_idx), **shared)
    print_metrics(m1, "Test Set Results - Top-1")
    print_metrics(mk, f"Test Set Results - Top-{k} ensemble")
    return {'top1': m1, f'top{k}': mk}


if __name__ == "__main__":
    ACTIVE_THRESHOLD = 1e-4  # on the sign-normalized multipliers
    TOP_K = 3
    DROPOUT_RATE = 0.1

    dcopf_config.print_config()
    as_experiment(**dcopf_config.get_all_paths(), **dcopf_config.get_all_params(),
                  active_threshold=ACTIVE_THRESHOLD, top_k=TOP_K, dropout_rate=DROPOUT_RATE)