# -*- coding: utf-8 -*-
"""Active set classification for ACOPF, extending Deka & Misra (arXiv:1902.05607) to AC.

A classifier maps loads to the active constraint set; Pg is then recovered by an LP over
the free generators and Vm is read off the active voltage bounds, followed by a power flow.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from collections import Counter

from scipy.optimize import linprog

import os

# ac_configuration/ sits in ac_methods/. Appending the parent of this script's own
# directory makes it importable whether this file is directly in ac_methods/ or one
# level down in a grouped method folder, and from any working directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ac_configuration import acopf_config
    from ac_configuration.acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        prepare_data_splits,
        reconstruct_full_pg,
    )
    from ac_configuration.acopf_duals import load_dual_by_ids, load_dual_sorted
    from ac_configuration.acopf_evaluation_metrics import evaluate_acopf_predictions
    from ac_configuration.acopf_pypower import load_case_from_csv, solve_pf_setpoints
except ImportError as e:
    print(f"Error: Unable to import from ac_configuration/ ({e})")
    sys.exit(1)

GLOBAL_CASE_DATA = None


def load_active_sets(duals_dir, case_name, params, threshold=1e-4,
                     active_set_type='full'):
    """Build a per-sample binary active-set vector from the dual variable CSVs.

    active_set_type 'full' uses pg/qg/vm bounds, giving 4*n_gen + 2*n_bus entries.
    'pg_only' uses just the Pg bounds (2*n_gen); on large cases the full set is so
    fragmented that nearly every sample gets its own class, which cannot be learned.

    Returns (active_matrix, meta).
    """
    if active_set_type not in ('full', 'pg_only'):
        raise ValueError(
            f"active_set_type must be 'full' or 'pg_only', got '{active_set_type}'")

    n_g = params['general']['n_gen']
    n_bus = params['general']['n_buses']
    bus_ids = params['general']['bus_ids']

    dim_str = f"2*n_g={2 * n_g}" if active_set_type == 'pg_only' \
        else f"4*n_g+2*n_bus={4 * n_g + 2 * n_bus}"
    print(f"  Active set type: {active_set_type.upper()} (dim = {dim_str})")

    def _load_and_reorder_gen(suffix):
        return load_dual_sorted(duals_dir, case_name, suffix)

    def _load_and_reorder_bus(suffix):
        return load_dual_by_ids(duals_dir, case_name, suffix, bus_ids)

    # JuMP minimization sign convention: a lower-bound dual is positive when the
    # bound binds, an upper-bound dual is negative, so upper bounds are negated
    # to make "active" mean "> threshold" for every constraint type.
    mu_pg_min = _load_and_reorder_gen('mu_pg_min')
    mu_pg_max = _load_and_reorder_gen('mu_pg_max')
    a_pg_min = (mu_pg_min > threshold).astype(np.int8)
    a_pg_max = (-mu_pg_max > threshold).astype(np.int8)

    if active_set_type == 'pg_only':
        active_matrix = np.hstack([a_pg_min, a_pg_max])
        slices = {
            'pg_min': (0, n_g),
            'pg_max': (n_g, 2 * n_g),
        }
    else:
        mu_qg_min = _load_and_reorder_gen('mu_qg_min')
        mu_qg_max = _load_and_reorder_gen('mu_qg_max')
        mu_vm_min = _load_and_reorder_bus('mu_vm_min')
        mu_vm_max = _load_and_reorder_bus('mu_vm_max')

        active_matrix = np.hstack([
            a_pg_min,
            a_pg_max,
            (mu_qg_min > threshold).astype(np.int8),
            (-mu_qg_max > threshold).astype(np.int8),
            (mu_vm_min > threshold).astype(np.int8),
            (-mu_vm_max > threshold).astype(np.int8),
        ])
        slices = {
            'pg_min': (0, n_g),
            'pg_max': (n_g, 2 * n_g),
            'qg_min': (2 * n_g, 3 * n_g),
            'qg_max': (3 * n_g, 4 * n_g),
            'vm_min': (4 * n_g, 4 * n_g + n_bus),
            'vm_max': (4 * n_g + n_bus, 4 * n_g + 2 * n_bus),
        }

    meta = {
        'n_g': n_g,
        'n_bus': n_bus,
        'dim': active_matrix.shape[1],
        'slices': slices,
        'active_set_type': active_set_type,
    }
    return active_matrix, meta


def build_label_space(active_matrix, train_idx, meta):
    """Enumerate the active sets that occur in training; these are the classifier's classes.

    The label space deliberately comes from the training rows alone. Enumerating over
    the whole dataset would let held-out samples decide the size of the output layer,
    and would hide the fact that active sets never seen in training are an intrinsic
    ceiling on this method rather than a detail of the implementation.
    """
    active_set_to_label = {}
    label_to_active_set = {}

    for row in active_matrix[train_idx]:
        key = tuple(row.tolist())
        if key not in active_set_to_label:
            lbl = len(active_set_to_label)
            active_set_to_label[key] = lbl
            label_to_active_set[lbl] = np.array(row, dtype=np.int8)

    n_classes = len(label_to_active_set)
    avg_spc = len(train_idx) / max(n_classes, 1)
    if avg_spc < 2.0:
        print(f"  [WARNING] Active sets are highly fragmented: {n_classes} unique sets "
              f"across {len(train_idx)} training samples ({avg_spc:.2f} samples/class). "
              f"Classification will fit the training split and generalize poorly.")
        if meta['active_set_type'] == 'full':
            print(f"  [WARNING] Consider active_set_type='pg_only' to reduce the "
                  f"active set dimension.")

    return active_set_to_label, label_to_active_set


def assign_labels(active_matrix, active_set_to_label):
    """Label every sample, marking active sets absent from the training vocabulary as -1."""
    labels = np.full(len(active_matrix), -1, dtype=np.int64)
    for i, row in enumerate(active_matrix):
        lbl = active_set_to_label.get(tuple(row.tolist()))
        if lbl is not None:
            labels[i] = lbl
    return labels


def print_active_set_statistics(labels, split_name="Dataset"):
    """Print the number of distinct active sets and the most frequent ones."""
    counter = Counter(labels.tolist())
    n_total = len(labels)
    print(f"\n[Active Set Statistics - {split_name}]")
    print(f"  Total samples: {n_total}")
    print(f"  Unique active sets: {len(counter)}")
    print(f"  Top-5 most common:")
    for lbl, cnt in counter.most_common(5):
        print(f"    Label {lbl:5d}: {cnt:6d} samples ({100. * cnt / n_total:.2f}%)")


def print_active_constraint_count_distribution(label_to_active_set, meta, labels=None):
    """Print how many constraints are active per label, overall and by category."""
    sl = meta['slices']
    is_full = (meta['active_set_type'] == 'full')

    lbl_ids = sorted(label_to_active_set.keys())
    counts = {}
    breakdown = {}

    for lbl in lbl_ids:
        v = label_to_active_set[lbl].astype(int)

        def _sum(key):
            if key not in sl:
                return 0
            s, e = sl[key]
            return int(v[s:e].sum())

        parts = (_sum('pg_min'), _sum('pg_max'), _sum('qg_min'),
                 _sum('qg_max'), _sum('vm_min'), _sum('vm_max'))
        counts[lbl] = sum(parts)
        breakdown[lbl] = parts

    all_counts = np.array(list(counts.values()))

    if labels is not None:
        weighted = []
        for lbl, cnt in Counter(labels.tolist()).items():
            if lbl in counts:
                weighted.extend([counts[lbl]] * cnt)
        weighted = np.array(weighted)
    else:
        weighted = all_counts

    print(f"\n[Active Constraint Count Diagnostics]")
    print(f"  System: n_g={meta['n_g']}, n_bus={meta['n_bus']}")
    print(f"  Active set type: {meta['active_set_type'].upper()}, "
          f"total dims: {meta['dim']}")
    print(f"  Per unique active set ({len(lbl_ids)} sets): "
          f"min={all_counts.min()}, max={all_counts.max()}, "
          f"mean={all_counts.mean():.2f}, median={np.median(all_counts):.1f}")
    print(f"  Sample-weighted: mean={weighted.mean():.2f}, std={weighted.std():.2f}")

    top5 = (Counter(labels.tolist()).most_common(5) if labels is not None
            else [(lbl, 1) for lbl in lbl_ids[:5]])

    print(f"  Top-5 active sets, constraint breakdown:")
    if is_full:
        print(f"    {'Label':>6}  {'Freq':>6}  {'Total':>5}  {'pgMn':>4}  {'pgMx':>4}"
              f"  {'qgMn':>4}  {'qgMx':>4}  {'vmMn':>4}  {'vmMx':>4}")
    else:
        print(f"    {'Label':>6}  {'Freq':>6}  {'Total':>5}  {'pgMn':>4}  {'pgMx':>4}")

    for lbl, cnt in top5:
        if lbl not in breakdown:
            continue
        pgn, pgx, qgn, qgx, vmn, vmx = breakdown[lbl]
        if is_full:
            print(f"    {lbl:6d}  {cnt:6d}  {counts[lbl]:5d}  {pgn:4d}  {pgx:4d}"
                  f"  {qgn:4d}  {qgx:4d}  {vmn:4d}  {vmx:4d}")
        else:
            print(f"    {lbl:6d}  {cnt:6d}  {counts[lbl]:5d}  {pgn:4d}  {pgx:4d}")


def _decode_active_set(active_set_vec, meta):
    """Turn a binary active-set vector into index arrays, one per constraint type."""
    sl = meta['slices']

    def _idx(key):
        if key not in sl:
            return np.array([], dtype=np.int64)
        s, e = sl[key]
        return np.where(active_set_vec[s:e] == 1)[0]

    return (_idx('pg_min'), _idx('pg_max'),
            _idx('qg_min'), _idx('qg_max'),
            _idx('vm_min'), _idx('vm_max'))


def recover_pg_vm_linprog(active_set_vec, x_pd_sample, params, meta):
    """Recover non-slack Pg and generator Vm from a predicted active set.

    Generators whose Pg bound is active are fixed at that bound; the rest are set by
    an LP that minimizes linear cost subject to meeting the remaining load. Qg bounds
    label the active set but never enter the LP, because Qg comes from the power flow.
    """
    non_slack = params['general']['non_slack_gen_idx']
    n_ns = params['general']['n_gen_non_slack']
    pg_min_all = params['generator']['pg_min'].ravel().astype(np.float64)
    pg_max_all = params['generator']['pg_max'].ravel().astype(np.float64)
    vm_min_all = params['bus']['vm_min'].ravel().astype(np.float64)
    vm_max_all = params['bus']['vm_max'].ravel().astype(np.float64)
    c1_all = params['generator']['cost_c1'].ravel().astype(np.float64)

    # The dataset mean of generator Vm is a better default operating point than the
    # case file's vg_pu, which on some cases is uniformly 1.0 and makes the power
    # flow diverge from the very first iteration
    if 'vm_gen_mean' in params['general']:
        vg_nominal = params['general']['vm_gen_mean'].astype(np.float64)
    else:
        vg_nominal = GLOBAL_CASE_DATA['gen'][:, 5].astype(np.float64)

    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_bus_indices = np.array(
        [bus_id_to_idx[int(gid)] for gid in params['general']['gen_bus_ids']])

    ag_pg_min, ag_pg_max, _, _, av_vm_min, av_vm_max = \
        _decode_active_set(active_set_vec, meta)

    # Vm: active voltage bounds pin the generators on that bus; in pg_only mode the
    # vm index arrays are always empty, so every generator keeps the nominal value
    vm_gen = vg_nominal.copy()
    for pos in av_vm_min:
        vm_gen[gen_bus_indices == int(pos)] = vm_min_all[int(pos)]
    for pos in av_vm_max:
        vm_gen[gen_bus_indices == int(pos)] = vm_max_all[int(pos)]

    pg_min_ns = pg_min_all[non_slack]
    pg_max_ns = pg_max_all[non_slack]
    c1_ns = c1_all[non_slack]
    ns_global_to_local = {g: i for i, g in enumerate(non_slack)}

    fixed_ns = np.full(n_ns, np.nan)

    # Units with pg_min == pg_max == 0 are committed but offline: they must output
    # zero whatever the predicted active set says, and must not become LP variables
    for i in range(n_ns):
        if pg_min_ns[i] == 0.0 and pg_max_ns[i] == 0.0:
            fixed_ns[i] = 0.0

    for g in ag_pg_min:
        if g in ns_global_to_local:
            loc = ns_global_to_local[g]
            if np.isnan(fixed_ns[loc]):
                fixed_ns[loc] = pg_min_all[g]
    for g in ag_pg_max:
        if g in ns_global_to_local:
            loc = ns_global_to_local[g]
            if np.isnan(fixed_ns[loc]):
                fixed_ns[loc] = pg_max_all[g]
            elif pg_max_all[g] != 0.0:
                # The prediction claims both bounds are active, which cannot hold;
                # the upper bound wins, except for offline units fixed at zero
                fixed_ns[loc] = pg_max_all[g]

    free_idx = np.where(np.isnan(fixed_ns))[0]

    if len(free_idx) == 0:
        return (np.nan_to_num(fixed_ns, nan=0.0).astype(np.float32),
                vm_gen.astype(np.float32))

    # Losses are ignored here; the slack generator picks up the resulting mismatch
    residual = float(np.sum(x_pd_sample)) - float(np.nansum(fixed_ns))
    n_free = len(free_idx)
    lb_free = pg_min_ns[free_idx]
    ub_free = pg_max_ns[free_idx]

    res = linprog(
        c1_ns[free_idx],
        A_eq=np.ones((1, n_free), dtype=np.float64),
        b_eq=np.array([residual], dtype=np.float64),
        bounds=list(zip(lb_free.tolist(), ub_free.tolist())),
        method='highs'
    )

    # An infeasible LP means the fixed generators cannot be reconciled with the
    # load, so the free ones share the residual evenly instead
    pg_free_val = res.x if res.success else np.clip(
        np.full(n_free, residual / n_free), lb_free, ub_free)

    pg_ns = fixed_ns.copy()
    pg_ns[free_idx] = pg_free_val
    return (np.nan_to_num(pg_ns, nan=0.0).astype(np.float32),
            vm_gen.astype(np.float32))


def _single_cost_ac(pg_non_slack, params):
    """Quadratic cost of the non-slack generators, used to rank Top-K candidates."""
    ns = params['general']['non_slack_gen_idx']
    c2 = params['generator']['cost_c2'].ravel()[ns]
    c1 = params['generator']['cost_c1'].ravel()[ns]
    c0 = params['generator']['cost_c0'].ravel()[ns]
    return float(np.sum(c2 * pg_non_slack ** 2 + c1 * pg_non_slack + c0))


def _fallback_pg_vm(x_pd_sample, params):
    """Dispatch used when no predicted active set yields a usable solution."""
    ns = params['general']['non_slack_gen_idx']
    pg_min_ns = params['generator']['pg_min'].ravel()[ns]
    pg_max_ns = params['generator']['pg_max'].ravel()[ns]

    pg_ns = np.clip(
        np.full(len(ns), float(np.sum(x_pd_sample)) / len(ns)),
        pg_min_ns, pg_max_ns
    ).astype(np.float32)

    if 'vm_gen_mean' in params['general']:
        vm_gen = params['general']['vm_gen_mean'].astype(np.float32)
    else:
        vm_gen = GLOBAL_CASE_DATA['gen'][:, 5].astype(np.float32)

    return pg_ns, vm_gen


def recover_batch_top1(top1_labels, label_to_as, x_pd_raw, params, meta):
    """Recover Pg and Vm from the single highest-probability active set per sample."""
    n = len(top1_labels)
    pg_out = np.zeros((n, params['general']['n_gen_non_slack']), dtype=np.float32)
    vm_out = np.zeros((n, params['general']['n_gen']), dtype=np.float32)
    n_fail = 0

    for i in range(n):
        lbl = top1_labels[i]
        if lbl not in label_to_as:
            pg_out[i], vm_out[i] = _fallback_pg_vm(x_pd_raw[i], params)
            n_fail += 1
        else:
            pg_out[i], vm_out[i] = recover_pg_vm_linprog(
                label_to_as[lbl], x_pd_raw[i], params, meta)

    if n_fail > 0:
        print(f"  [Top-1] Recovery fallback: {n_fail}/{n} samples")
    return pg_out, vm_out


def recover_batch_topk(topk_labels, label_to_as, x_pd_raw, params, meta):
    """Recover each of the K candidate active sets and keep the cheapest in-bounds one."""
    n, K = topk_labels.shape
    ns = params['general']['non_slack_gen_idx']
    pg_min = params['generator']['pg_min'].ravel()[ns]
    pg_max = params['generator']['pg_max'].ravel()[ns]

    pg_out = np.zeros((n, params['general']['n_gen_non_slack']), dtype=np.float32)
    vm_out = np.zeros((n, params['general']['n_gen']), dtype=np.float32)

    for i in range(n):
        best_pg = best_vm = None
        best_cost = np.inf

        for k in range(K):
            lbl = topk_labels[i, k]
            if lbl not in label_to_as:
                continue
            pg, vm = recover_pg_vm_linprog(
                label_to_as[lbl], x_pd_raw[i], params, meta)
            if np.any(pg < pg_min - 1e-3) or np.any(pg > pg_max + 1e-3):
                continue
            cost = _single_cost_ac(pg, params)
            if cost < best_cost:
                best_cost, best_pg, best_vm = cost, pg, vm

        if best_pg is None:
            pg_out[i], vm_out[i] = _fallback_pg_vm(x_pd_raw[i], params)
        else:
            pg_out[i], vm_out[i] = best_pg, best_vm

    return pg_out, vm_out


def run_power_flow_batch(pg_ns_batch, vm_gen_batch, x_raw, params, verbose=True):
    """Run the power flow for a batch of recovered setpoints.

    Returns (pf_results_list, converge_flags, y_pred_pg_full, y_pred_vm_all).
    """
    n_samples = len(pg_ns_batch)
    n_gen = params['general']['n_gen']
    n_bus = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_bus_indices = np.array(
        [bus_id_to_idx[int(gid)] for gid in params['general']['gen_bus_ids']])

    pd_pu = x_raw[:, :n_loads]
    qd_pu = x_raw[:, n_loads:]

    pf_results_list = []
    converge_flags = []
    y_pred_pg_full = np.zeros((n_samples, n_gen), dtype=np.float32)
    y_pred_vm_all = np.full((n_samples, n_bus), 1.0, dtype=np.float32)

    if verbose:
        print(f"  Running power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_setpoints(
                pd_pu[i], qd_pu[i], pg_ns_batch[i], vm_gen_batch[i],
                params, GLOBAL_CASE_DATA)
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except Exception:
            pf_results_list.append((
                {'success': False,
                 'gen': np.zeros((n_gen, 21)),
                 'bus': np.zeros((n_bus, 13)),
                 'branch': np.zeros((1, 17))},
            ))
            converge_flags.append(False)

        y_pred_pg_full[i] = reconstruct_full_pg(pg_ns_batch[i], params)
        y_pred_vm_all[i, gen_bus_indices] = vm_gen_batch[i]

    if verbose:
        print(f"  Converged: {sum(converge_flags)}/{n_samples}")

    return pf_results_list, converge_flags, y_pred_pg_full, y_pred_vm_all


class ActiveSetClassifierAC(nn.Module):
    """Classifier over observed active sets: Linear, ReLU, BatchNorm and Dropout per layer."""

    def __init__(self, input_size, n_classes, hidden_layers=None, dropout_rate=0.1):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 256, 128, 128, 64]

        layers = []
        prev = input_size
        for h in hidden_layers:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout_rate),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def active_set_acopf_experiment(
        case_name,
        params_path,
        data_path,
        duals_dir,
        n_train_use=None,
        seed=42,
        n_epochs=20,
        learning_rate=0.001,
        batch_size=32,
        hidden_layers=None,
        dropout_rate=0.1,
        device='cuda',
        early_stop_patience=20,
        early_stop_min_delta=1e-6,
        active_threshold=1e-4,
        active_set_type='full',
        top_k=3,
):
    """Train the active set classifier on a random split and evaluate Top-1 and Top-K."""
    global GLOBAL_CASE_DATA

    if hidden_layers is None:
        hidden_layers = [256, 256, 128, 128, 64]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 70}")
    print(f"ACOPF Active Set Classification Experiment")
    print(f"{'=' * 70}")
    print(f"Device: {device_obj}")
    print(f"Case: {case_name}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # 1. Load network parameters and PyPower case data
    # ------------------------------------------------------------------
    params = load_parameters_from_csv(case_name, params_path)
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)

    n_gen = params['general']['n_gen']
    n_gen_ns = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    baseMVA = params['general']['BASE_MVA']

    print(f"\n[System Info]")
    print(f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_ns}), "
          f"Base MVA: {baseMVA}")

    # ------------------------------------------------------------------
    # 2. Load dataset and fit scalers
    # ------------------------------------------------------------------
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    # n_loads is only known after loading, because the loader corrects it from the
    # CSV columns: a case may have far fewer load buses than buses
    n_loads = params['general']['n_loads']
    print(f"  Loads (from CSV columns): {n_loads}")
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    params['general']['vm_gen_mean'] = raw_data['vm_gen'].mean(axis=0).astype(np.float64)
    print(f"  vm_gen_mean range: "
          f"[{params['general']['vm_gen_mean'].min():.4f}, "
          f"{params['general']['vm_gen_mean'].max():.4f}] p.u.")

    # ------------------------------------------------------------------
    # 3. Active sets from the dual variables
    # ------------------------------------------------------------------
    print(f"\n[Active Set Labels]")
    active_matrix, meta = load_active_sets(
        duals_dir, case_name, params,
        threshold=active_threshold,
        active_set_type=active_set_type,
    )

    # A row-count mismatch means the duals folder and the sample folder came from
    # different runs, which would silently mislabel every sample
    if len(active_matrix) != len(x_data_scaled):
        raise ValueError(
            f"Dual CSVs have {len(active_matrix)} rows but the dataset has "
            f"{len(x_data_scaled)} samples; the two folders are out of sync")

    # ------------------------------------------------------------------
    # 4. Split, then derive the label space from the training rows only
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = prepare_data_splits(
        x_data_scaled, y_data_scaled,
        n_train_use=n_train_use,
        seed=seed
    )

    as_to_label, label_to_as = build_label_space(active_matrix, train_idx, meta)
    labels_all = assign_labels(active_matrix, as_to_label)
    n_classes = len(label_to_as)

    print(f"  Active sets in the training vocabulary: {n_classes}")
    print_active_set_statistics(labels_all[train_idx], "Train Split")
    print_active_constraint_count_distribution(
        label_to_as, meta, labels=labels_all[train_idx])

    # Samples whose active set never occurs in training have no valid target. They
    # are dropped from the validation loss (it drives early stopping and needs a
    # target) but kept in the test set, where they count as misclassified.
    val_known = labels_all[val_idx] >= 0
    n_val_unseen = int((~val_known).sum())
    if n_val_unseen > 0:
        print(f"  Val samples with an unseen active set: {n_val_unseen}/{len(val_idx)} "
              f"(excluded from validation loss)")
    if val_known.sum() == 0:
        raise ValueError(
            "No validation sample shares an active set with training; the active set "
            "space is too fragmented for this case and split to be learnable")

    val_idx_known = val_idx[val_known]

    X_train = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32, device=device_obj)
    Y_train = torch.tensor(labels_all[train_idx], dtype=torch.long, device=device_obj)
    X_val = torch.tensor(x_data_scaled[val_idx_known], dtype=torch.float32, device=device_obj)
    Y_val = torch.tensor(labels_all[val_idx_known], dtype=torch.long, device=device_obj)
    X_test = torch.tensor(x_data_scaled[test_idx], dtype=torch.float32, device=device_obj)

    # ------------------------------------------------------------------
    # 5. Classifier
    # ------------------------------------------------------------------
    input_dim = x_data_scaled.shape[1]
    model = ActiveSetClassifierAC(
        input_size=input_dim,
        n_classes=n_classes,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
    ).to(device_obj)

    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"Input dim: {input_dim} (pd + qd = 2 x {n_loads})")
    print(f"Num classes: {n_classes}")
    print(f"Hidden: {hidden_layers}, dropout={dropout_rate}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training params: max_epochs={n_epochs}, patience={early_stop_patience}, "
          f"lr={learning_rate}, batch_size={batch_size}")
    print(f"{'=' * 70}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    n_train = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size

    # ------------------------------------------------------------------
    # 6. Training with early stopping on validation loss
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Training Progress")
    print(f"{'=' * 70}")

    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        seen = 0
        perm = torch.randperm(n_train, device=device_obj)

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            # BatchNorm cannot compute statistics from a single sample, so a
            # trailing batch of one is skipped rather than crashing the epoch
            if len(idx) < 2:
                continue
            Xb, Yb = X_train[idx], Y_train[idx]
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, Yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)
            correct += (logits.argmax(-1) == Yb).sum().item()
            seen += len(Xb)

        t_loss = epoch_loss / seen
        t_acc = 100. * correct / seen

        model.eval()
        with torch.no_grad():
            vl = model(X_val)
            v_loss = float(criterion(vl, Y_val))
            v_acc = 100. * (vl.argmax(-1) == Y_val).sum().item() / len(Y_val)

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {t_loss:.6f}  "
                  f"Train Acc: {t_acc:.2f}%  |  Val Loss: {v_loss:.6f}  "
                  f"Val Acc: {v_acc:.2f}%")

        if v_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = v_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {t_loss:.6f}  "
                      f"Train Acc: {t_acc:.2f}%  |  Val Loss: {v_loss:.6f}  "
                      f"Val Acc: {v_acc:.2f}%")
                print(f"Early stopping triggered at epoch {epoch} "
                      f"(patience={early_stop_patience})")
                break

    model.load_state_dict({k: v.to(device_obj) for k, v in best_state_dict.items()})
    train_time = time.perf_counter() - t0
    print(f"Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
    print(f"Training completed in {train_time:.2f} seconds")

    # ------------------------------------------------------------------
    # 7. Classify the test split
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    x_raw_test = raw_data['x'][test_idx]
    y_true_pg = raw_data['pg'][test_idx]
    y_true_vm = raw_data['vm'][test_idx]
    y_true_qg = raw_data['qg'][test_idx]
    y_true_va = raw_data['va'][test_idx]
    true_labels = labels_all[test_idx]
    n_test = len(X_test)

    model.eval()
    with torch.no_grad():
        probs_test = torch.softmax(model(X_test), dim=-1)
        top1_pred = probs_test.argmax(-1).cpu().numpy()
        K = min(top_k, probs_test.shape[-1])
        topk_labels = torch.topk(probs_test, k=K, dim=-1).indices.cpu().numpy()

    # true_labels is -1 where the active set is outside the training vocabulary, and
    # predictions are always >= 0, so those samples are counted wrong automatically
    top1_acc = 100. * np.mean(top1_pred == true_labels)
    topk_acc = 100. * np.mean(
        [true_labels[i] in topk_labels[i] for i in range(n_test)])
    test_unseen_rate = 100. * np.mean(true_labels < 0)

    # ------------------------------------------------------------------
    # 8. Recover setpoints and run the power flow
    # ------------------------------------------------------------------
    x_pd_raw = x_raw_test[:, :n_loads]

    print(f"\n  Recovering Pg and Vm via linprog for {n_test} test samples...")
    t_r0 = time.perf_counter()
    pg_top1, vm_top1 = recover_batch_top1(
        top1_pred, label_to_as, x_pd_raw, params, meta)
    t_r1 = time.perf_counter()
    pg_topk, vm_topk = recover_batch_topk(
        topk_labels, label_to_as, x_pd_raw, params, meta)
    t_r2 = time.perf_counter()
    print(f"  Top-1 recovery: {t_r1 - t_r0:.1f}s  |  Top-{K} recovery: "
          f"{t_r2 - t_r1:.1f}s")

    print(f"\n[Top-1 Power Flow]")
    pf1, cf1, pg_full1, vm_all1 = run_power_flow_batch(
        pg_top1, vm_top1, x_raw_test, params, verbose=True)

    print(f"\n[Top-{K} Power Flow]")
    pfK, cfK, pg_fullK, vm_allK = run_power_flow_batch(
        pg_topk, vm_topk, x_raw_test, params, verbose=True)

    metrics1 = evaluate_acopf_predictions(
        pg_full1, vm_all1, y_true_pg, y_true_vm, y_true_qg, y_true_va,
        pf1, cf1, params, verbose=False)
    metricsK = evaluate_acopf_predictions(
        pg_fullK, vm_allK, y_true_pg, y_true_vm, y_true_qg, y_true_va,
        pfK, cfK, params, verbose=False)

    # ------------------------------------------------------------------
    # 9. Inference latency (classifier forward pass only)
    # ------------------------------------------------------------------
    sample_t = X_test[:1]
    with torch.no_grad():
        for _ in range(10):
            model(sample_t)
        if device_obj.type == 'cuda':
            torch.cuda.synchronize()

        times_lat = []
        for _ in range(100):
            ts = time.perf_counter()
            model(sample_t)
            if device_obj.type == 'cuda':
                torch.cuda.synchronize()
            times_lat.append(time.perf_counter() - ts)

    latency_ms = np.mean(times_lat) * 1000

    # ------------------------------------------------------------------
    # 10. Results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Final Results Summary")
    print(f"{'=' * 70}")
    print(f"\nCase: {case_name}")

    print(f"\n--- Active Set Classification ---")
    print(f"Active Set Type: {active_set_type.upper()}"
          + (" (Pg bounds only)" if active_set_type == 'pg_only'
             else " (Pg + Qg + Vm bounds)"))
    print(f"Active sets in training vocabulary: {n_classes}")
    print(f"Top-1 Classification Accuracy: {top1_acc:.2f}%")
    print(f"Top-{K} Classification Accuracy: {topk_acc:.2f}%")
    print(f"Test samples with unseen active set: {test_unseen_rate:.2f}% "
          f"(upper bound on achievable accuracy: {100.0 - test_unseen_rate:.2f}%)")

    conv1 = metrics1['convergence_rate_percent']
    if conv1 < 50.0:
        print(f"\n  [NOTE] Power flow converged for only {conv1:.1f}% of samples. "
              f"Recovered dispatches that fix many generators at their bounds sit "
              f"far from the true optimum, which the power flow may not solve from.")

    def _fmt(val, fmt=".4f"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:{fmt}}"

    def _print_metrics(m, label):
        n_conv = m['n_converged']
        n_tot = m['n_samples']

        print(f"\n--- {label} ---")
        print(f"  [computed on {n_conv}/{n_tot} converged samples]")
        print(f"MAE_Pg (Non-Slack): {_fmt(m['mae_pg_non_slack_percent'])}%")
        print(f"MAE_Vm (Generator): {_fmt(m['mae_vm_percent'])}%")
        print(f"MAE_Qg (All Gens):  {_fmt(m['mae_qg_percent'])}%")
        print(f"MAE_Va (All Buses): {_fmt(m['mae_va_deg'])} degrees")
        print(f"Pg_viol (Non-Slack): {_fmt(m['mean_pg_viol_non_slack_pu'], '.6f')} p.u.")
        print(f"Pg_viol (Slack):     {_fmt(m['mean_pg_viol_slack_pu'], '.6f')} p.u.")
        print(f"Qg_viol (All Gens):  {_fmt(m['mean_max_qg_viol_pu'], '.6f')} p.u.")
        print(f"Vm_viol (All Buses): {_fmt(m['mean_max_vm_viol_pu'], '.6f')} p.u.")
        print(f"Branch_viol:         {_fmt(m['mean_max_branch_viol_pu'], '.6f')} p.u. "
              f"(1.0 = 100% overload)")
        print(f"Cost Gap: {_fmt(m['cost_optimality_gap_percent'])}%")
        print(f"Convergence Rate: {m['convergence_rate_percent']:.2f}%")

    _print_metrics(metrics1, "Top-1 Prediction")
    _print_metrics(metricsK, f"Top-{K} Ensemble Prediction")

    print(f"\n--- Performance ---")
    print(f"Inference Time: {latency_ms:.4f} ms/sample (classifier forward only)")
    print(f"Training Time:  {train_time:.2f} s")
    print(f"{'=' * 70}")

    return {
        'top1': metrics1,
        'topk': metricsK,
        'top1_acc': top1_acc,
        'topk_acc': topk_acc,
        'test_unseen_rate': test_unseen_rate,
        'n_classes': n_classes,
        'train_time': train_time,
        'latency_ms': latency_ms,
    }


if __name__ == "__main__":
    # The classifier keeps its own architecture rather than reading HIDDEN_SIZES
    # from the config, since its output layer is a label space, not a setpoint vector
    HIDDEN_LAYERS = [256, 128]
    DROPOUT_RATE = 0.1
    ACTIVE_THRESHOLD = 1e-4
    TOP_K = 3
    ACTIVE_SET_TYPE = 'pg_only'

    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    paths = acopf_config.get_all_paths()
    params = acopf_config.get_all_params()

    results = active_set_acopf_experiment(
        case_name=paths['case_name'],
        params_path=paths['params_path'],
        data_path=paths['data_path'],
        duals_dir=acopf_config.get_duals_path(),
        n_train_use=params['n_train_use'],
        seed=params['seed'],
        n_epochs=params['n_epochs'],
        learning_rate=params['learning_rate'],
        batch_size=params['batch_size'] or 32,
        hidden_layers=HIDDEN_LAYERS,
        dropout_rate=DROPOUT_RATE,
        device=params['device'],
        early_stop_patience=params['early_stop_patience'],
        early_stop_min_delta=params['early_stop_min_delta'],
        active_threshold=ACTIVE_THRESHOLD,
        active_set_type=ACTIVE_SET_TYPE,
        top_k=TOP_K,
    )

    print("\nExperiment completed successfully!")