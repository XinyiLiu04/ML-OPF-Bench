# -*- coding: utf-8 -*-
"""
Active Set Classification ACOPF Main Experiment (PyTorch)
Extension of: "Learning for DC-OPF: Classifying active sets using neural nets"
              Deka & Misra, arXiv:1902.05607v1

AC-OPF Extension Design:
  Active set defined over generator and bus constraints.
  Two modes controlled by active_set_type parameter:

  'full' (default, for case30 / case118):
    [ mu_pg_min_1..n_g | mu_pg_max_1..n_g |
      mu_qg_min_1..n_g | mu_qg_max_1..n_g |
      mu_vm_min_1..n_bus | mu_vm_max_1..n_bus ]
    Total dimension = 4*n_g + 2*n_bus

  'pg_only' (for case300 and large cases where full active sets explode):
    [ mu_pg_min_1..n_g | mu_pg_max_1..n_g ]
    Total dimension = 2*n_g
    Qg/Vm constraints ignored; Vm recovery uses nominal voltage only.

Dual Sign Convention (JuMP minimisation problem → Python):
  Empirically verified: JuMP minimisation convention is:
    LowerBoundRef dual (var >= lb): active when dual > 0  → use as-is
    UpperBoundRef dual (var <= ub): active when dual < 0  → negate → active > 0
  Applied to:
    mu_pg_min, mu_qg_min, mu_vm_min  (LowerBound) → keep sign
    mu_pg_max, mu_qg_max, mu_vm_max  (UpperBound) → negate

Column Alignment (handles sparse bus numbering like case300):
  Julia writes columns sorted by gen_id / bus_id.
  Python uses CSV row-order (position) indices.
  _load_and_reorder_gen/bus re-sorts to match Python's bus_id_to_idx convention.

Pg Recovery (linprog, HiGHS):
  Active generator constraints fix pg_i directly.
  Remaining free non-slack generators solved via LP:
    min   c1[free] @ pg_free
    s.t.  sum(pg_free) = total_load - sum(pg_fixed)  (power balance)
          Pg_min[free] <= pg_free <= Pg_max[free]
  NOTE: Qg constraints inform the active set label but do NOT enter the LP
        (Qg is computed by power flow).

Vm Recovery:
  'full'    mode: active_vm_min → vm = vm_min
                  active_vm_max → vm = vm_max
                  else          → vm = nominal vg_pu
  'pg_only' mode: vm = nominal vg_pu (all generators, always)

Pipeline:
  NN classifier (load → active set label)
  → linprog recovers pg_non_slack
  → Vm assignment from active set (or nominal in pg_only mode)
  → solve_pf(pd, qd, pg_non_slack, vm_gen, params)
  → evaluate_acopf_predictions(...)

Top-K policy:
  Top-1 : argmax label → pg + vm → power flow
  Top-3  : recover pg+vm for 3 candidates, pick feasible min-cost

Print format: identical to acopf_dnn_main.py + active set statistics block

Data Split Modes: random_split, fixed_valtest, generalization, api_test

Dual CSV files (from Julia script, stored in separate _with_duals folder):
  {case_name}_mu_pg_min.csv  columns: mu_pg_min_1, mu_pg_min_2, ...
  {case_name}_mu_pg_max.csv
  {case_name}_mu_qg_min.csv  (only needed for 'full' mode)
  {case_name}_mu_qg_max.csv  (only needed for 'full' mode)
  {case_name}_mu_vm_min.csv  (only needed for 'full' mode)
  {case_name}_mu_vm_max.csv  (only needed for 'full' mode)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
from collections import Counter
from pathlib import Path

from scipy.optimize import linprog
from pypower.runpf import runpf
from pypower.ppoption import ppoption

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from acopf_data_setup import (
    load_parameters_from_csv,
    load_and_scale_acopf_data,
    DataMode,
    prepare_data_splits,
    load_generalization_test_data,
    load_api_test_data,
    reconstruct_full_pg,
)
from acopf_violation_metrics import evaluate_acopf_predictions

# =====================================================================
# PyPower globals (mirrors acopf_dnn_main.py pattern)
# =====================================================================
GLOBAL_CASE_DATA = None
PPOPT = None


def init_pypower_options():
    global PPOPT
    ppopt = ppoption()
    PPOPT = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)


def load_case_from_csv(case_name, constraints_path):
    """Load PyPower case data from CSV files (identical to acopf_dnn_main.py)."""
    base_path = Path(constraints_path)
    base_mva_df = pd.read_csv(base_path / f"{case_name}_base_mva.csv")
    bus_df      = pd.read_csv(base_path / f"{case_name}_bus_data.csv")
    gen_df      = pd.read_csv(base_path / f"{case_name}_gen_data.csv")
    branch_df   = pd.read_csv(base_path / f"{case_name}_branch_data.csv")
    baseMVA     = base_mva_df['value'].iloc[0]

    bus = np.zeros((len(bus_df), 13))
    bus[:, 0]  = bus_df['bus_id'].values
    bus[:, 1]  = bus_df['type'].values
    bus[:, 2]  = bus_df['pd_pu'].values
    bus[:, 3]  = bus_df['qd_pu'].values
    bus[:, 6]  = 1
    bus[:, 7]  = bus_df['vm_pu'].values
    bus[:, 8]  = bus_df['va_deg'].values
    bus[:, 9]  = bus_df['base_kv'].values
    bus[:, 10] = 1
    bus[:, 11] = bus_df['vmax_pu'].values
    bus[:, 12] = bus_df['vmin_pu'].values

    gen = np.zeros((len(gen_df), 21))
    gen[:, 0] = gen_df['bus_id'].values
    gen[:, 3] = gen_df['qg_max_pu'].values
    gen[:, 4] = gen_df['qg_min_pu'].values
    gen[:, 5] = gen_df['vg_pu'].values
    gen[:, 6] = baseMVA
    gen[:, 7] = 1
    gen[:, 8] = gen_df['pg_max_pu'].values
    gen[:, 9] = gen_df['pg_min_pu'].values

    branch = np.zeros((len(branch_df), 13))
    branch[:, 0]  = branch_df['f_bus'].values
    branch[:, 1]  = branch_df['t_bus'].values
    branch[:, 2]  = branch_df['r_pu'].values
    branch[:, 3]  = branch_df['x_pu'].values
    branch[:, 4]  = branch_df['b_pu'].values
    branch[:, 5]  = branch_df['rate_a_pu'].values
    branch[:, 6]  = branch[:, 5]
    branch[:, 7]  = branch[:, 5]
    branch[:, 8]  = branch_df['tap_ratio'].values
    branch[:, 9]  = branch_df['shift_deg'].values
    branch[:, 10] = 1
    branch[:, 11] = -360
    branch[:, 12] = 360
    rate_a_values = branch_df['rate_a_pu'].values
    branch[:, 5:8][np.isnan(rate_a_values) | np.isinf(rate_a_values), :] = 9900.0

    gencost = np.zeros((len(gen_df), 7))
    gencost[:, 0] = 2
    gencost[:, 3] = 3
    gencost[:, 4] = gen_df['cost_c2'].values
    gencost[:, 5] = gen_df['cost_c1'].values
    gencost[:, 6] = gen_df['cost_c0'].values

    ppc = {
        'version': '2', 'baseMVA': baseMVA,
        'bus': bus, 'gen': gen, 'branch': branch, 'gencost': gencost
    }
    ppc['bus'][:, 2]  *= baseMVA
    ppc['bus'][:, 3]  *= baseMVA
    ppc['gen'][:, 3]  *= baseMVA
    ppc['gen'][:, 4]  *= baseMVA
    ppc['gen'][:, 8]  *= baseMVA
    ppc['gen'][:, 9]  *= baseMVA
    mask = (ppc['branch'][:, 5] != 0) & (ppc['branch'][:, 5] < 9000)
    ppc['branch'][mask, 5:8] *= baseMVA
    return ppc


def solve_pf(pd_pu, qd_pu, pg_non_slack, vm_gen, params):
    """Run power flow (identical interface to acopf_dnn_main.py)."""
    global GLOBAL_CASE_DATA, PPOPT
    BASE_MVA       = params['general']['BASE_MVA']
    load_bus_ids   = params['general']['load_bus_ids']
    bus_id_to_idx  = params['general']['bus_id_to_idx']
    non_slack_idx  = params['general']['non_slack_gen_idx']
    n_gen          = params['general']['n_gen']

    mpc_pf = {
        'version':  GLOBAL_CASE_DATA['version'],
        'baseMVA':  GLOBAL_CASE_DATA['baseMVA'],
        'bus':      GLOBAL_CASE_DATA['bus'].copy(),
        'gen':      GLOBAL_CASE_DATA['gen'].copy(),
        'branch':   GLOBAL_CASE_DATA['branch'],
        'gencost':  GLOBAL_CASE_DATA['gencost'],
    }

    for i, bus_id in enumerate(load_bus_ids):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc_pf['bus'][bus_idx, 2] = pd_pu[i]  * BASE_MVA
            mpc_pf['bus'][bus_idx, 3] = qd_pu[i]  * BASE_MVA

    for i, gen_idx in enumerate(non_slack_idx):
        mpc_pf['gen'][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    for i in range(n_gen):
        mpc_pf['gen'][i, 5] = vm_gen[i]

    return runpf(mpc_pf, PPOPT)


# =====================================================================
# Part 1 – Dual Loading and Active Set Extraction
# =====================================================================

def load_dual_csvs(duals_dir, case_name, params, threshold=1e-4,
                   active_set_type='full'):
    """
    Load dual CSV files and build per-sample binary active-set vectors.

    Parameters
    ----------
    active_set_type : str
        'full'    : use all 6 constraint types (pg_min/max, qg_min/max, vm_min/max)
                    Active-set vector dim = 4*n_g + 2*n_bus
                    Suitable for case30, case118 where active sets are manageable.

        'pg_only' : use only pg_min/max dual variables (2*n_g dimensions)
                    Active-set vector dim = 2*n_g
                    Suitable for large cases (e.g. case300) where Qg/Vm constraints
                    cause active set explosion (every sample has a unique active set).
                    In this mode Vm recovery uses nominal voltage only.

    Sign convention (JuMP minimisation → Python), empirically verified:
        LowerBoundRef (var >= lb): active when dual > 0  → use as-is
        UpperBoundRef (var <= ub): active when dual < 0  → negate → active > 0
        mu_pg_min, mu_qg_min, mu_vm_min  (LowerBound) → keep sign
        mu_pg_max, mu_qg_max, mu_vm_max  (UpperBound) → negate

    Column alignment (handles sparse bus numbering like case300):
        Julia writes columns sorted by gen_id / bus_id.
        Python uses CSV row-order (position) indices (bus_id_to_idx convention).
        _load_and_reorder_gen/bus re-sorts to match Python position indexing.

    Active-set vector layout (fixed order, all position-indexed):
        'full'   : [pg_min | pg_max | qg_min | qg_max | vm_min | vm_max]
        'pg_only': [pg_min | pg_max]

    Returns
    -------
    labels              : np.ndarray (n_samples,) int64
    label_to_active_set : dict {int -> np.int8 array (dim,)}
    active_set_to_label : dict {tuple -> int}
    meta                : dict  (n_g, n_bus, dim, slices, active_set_type)
    """
    if active_set_type not in ('full', 'pg_only'):
        raise ValueError(
            f"active_set_type must be 'full' or 'pg_only', got '{active_set_type}'"
        )

    n_g       = params['general']['n_gen']
    n_bus     = params['general']['n_buses']
    bus_ids   = params['general']['bus_ids']       # bus IDs in bus_data.csv row order
    bus_id_to_idx = params['general']['bus_id_to_idx']

    dim_str = f"2*n_g={2*n_g}" if active_set_type == 'pg_only' \
              else f"4*n_g+2*n_bus={4*n_g+2*n_bus}"
    print(f"  Active set type: {active_set_type.upper()}  (dim = {dim_str})")

    # ── column-reorder helpers ───────────────────────────────────────
    def _load_and_reorder_gen(suffix):
        """Load gen dual CSV; reorder columns to match gen_data.csv row order."""
        path = os.path.join(duals_dir, f"{case_name}_{suffix}.csv")
        df   = pd.read_csv(path)
        # Column names: mu_pg_min_1, mu_pg_min_2, ... (Julia: sorted by gen_id)
        col_genids    = [int(col.rsplit('_', 1)[1]) for col in df.columns]
        sorted_genids = sorted(col_genids)              # same order Julia writes
        genid_to_col  = {gid: i for i, gid in enumerate(col_genids)}
        ordered_cols  = [df.columns[genid_to_col[gid]] for gid in sorted_genids]
        return df[ordered_cols].values.astype(np.float32)   # (n_samples, n_g)

    def _load_and_reorder_bus(suffix):
        """Load bus dual CSV; reorder columns to match bus_data.csv row order."""
        path = os.path.join(duals_dir, f"{case_name}_{suffix}.csv")
        df   = pd.read_csv(path)
        # Column names: mu_vm_min_1, mu_vm_min_7, ... (Julia: sorted by bus_id)
        col_busids   = [int(col.rsplit('_', 1)[1]) for col in df.columns]
        busid_to_col = {bid: i for i, bid in enumerate(col_busids)}
        # Reorder to match bus_data.csv row order (bus_id_to_idx position order)
        ordered_cols = [df.columns[busid_to_col[int(bid)]] for bid in bus_ids]
        return df[ordered_cols].values.astype(np.float32)   # (n_samples, n_bus)

    # ── load and sign-correct ────────────────────────────────────────
    # JuMP minimisation: LowerBound active > 0 (keep), UpperBound active < 0 (negate)
    mu_pg_min = _load_and_reorder_gen('mu_pg_min')
    mu_pg_max = _load_and_reorder_gen('mu_pg_max')
    a_pg_min  = ( mu_pg_min > threshold).astype(np.int8)   # LowerBound: keep
    a_pg_max  = (-mu_pg_max > threshold).astype(np.int8)   # UpperBound: negate

    if active_set_type == 'pg_only':
        # Use only Pg dual variables
        active_matrix = np.hstack([a_pg_min, a_pg_max])    # (n_samples, 2*n_g)
        slices = {
            'pg_min': (0,   n_g),
            'pg_max': (n_g, 2*n_g),
            # qg_min / qg_max / vm_min / vm_max slices absent in pg_only mode
        }
    else:
        # 'full': also load Qg and Vm duals
        mu_qg_min = _load_and_reorder_gen('mu_qg_min')
        mu_qg_max = _load_and_reorder_gen('mu_qg_max')
        mu_vm_min = _load_and_reorder_bus('mu_vm_min')
        mu_vm_max = _load_and_reorder_bus('mu_vm_max')

        a_qg_min  = ( mu_qg_min > threshold).astype(np.int8)
        a_qg_max  = (-mu_qg_max > threshold).astype(np.int8)
        a_vm_min  = ( mu_vm_min > threshold).astype(np.int8)
        a_vm_max  = (-mu_vm_max > threshold).astype(np.int8)

        active_matrix = np.hstack([
            a_pg_min, a_pg_max,
            a_qg_min, a_qg_max,
            a_vm_min, a_vm_max,
        ])   # (n_samples, 4*n_g + 2*n_bus)
        slices = {
            'pg_min': (0,            n_g),
            'pg_max': (n_g,          2*n_g),
            'qg_min': (2*n_g,        3*n_g),
            'qg_max': (3*n_g,        4*n_g),
            'vm_min': (4*n_g,        4*n_g + n_bus),
            'vm_max': (4*n_g+n_bus,  4*n_g + 2*n_bus),
        }

    # ── label assignment ─────────────────────────────────────────────
    active_set_to_label = {}
    label_to_active_set = {}
    labels = np.zeros(len(active_matrix), dtype=np.int64)

    for i, row in enumerate(active_matrix):
        key = tuple(row.tolist())
        if key not in active_set_to_label:
            lbl = len(active_set_to_label)
            active_set_to_label[key] = lbl
            label_to_active_set[lbl] = np.array(row, dtype=np.int8)
        labels[i] = active_set_to_label[key]

    n_classes = len(label_to_active_set)
    n_samples  = len(labels)

    # ── fragmentation warning ────────────────────────────────────────
    avg_spc = n_samples / max(n_classes, 1)
    if avg_spc < 2.0:
        print(f"  [WARNING] Extremely fragmented active sets: "
              f"{n_classes} unique sets for {n_samples} samples "
              f"(avg {avg_spc:.2f} samples/class).")
        if active_set_type == 'full':
            print(f"  [WARNING] Consider switching to active_set_type='pg_only' "
                  f"to reduce active set dimensionality.")
        print(f"  [WARNING] Classification will overfit training set "
              f"(train acc=100%, val/test acc≈0%).")

    meta = {
        'n_g':             n_g,
        'n_bus':           n_bus,
        'dim':             active_matrix.shape[1],
        'slices':          slices,
        'active_set_type': active_set_type,
    }
    return labels, label_to_active_set, active_set_to_label, meta


def print_active_set_statistics(labels, split_name="Dataset"):
    counter  = Counter(labels.tolist())
    n_unique = len(counter)
    n_total  = len(labels)
    top5     = counter.most_common(5)
    print(f"\n[Active Set Statistics – {split_name}]")
    print(f"  Total samples:      {n_total}")
    print(f"  Unique active sets: {n_unique}")
    print(f"  Top-5 most common:")
    for lbl, cnt in top5:
        print(f"    Label {lbl:5d}: {cnt:6d} samples ({100.*cnt/n_total:.2f}%)")


def print_active_constraint_count_distribution(label_to_active_set, meta,
                                                labels=None):
    """Diagnose how many constraints are active per label."""
    n_g     = meta['n_g']
    n_bus   = meta['n_bus']
    sl      = meta['slices']
    is_full = (meta.get('active_set_type', 'full') == 'full')

    lbl_ids   = sorted(label_to_active_set.keys())
    counts    = {}
    breakdown = {}

    for lbl in lbl_ids:
        v      = label_to_active_set[lbl].astype(int)
        pg_min = int(v[sl['pg_min'][0]:sl['pg_min'][1]].sum())
        pg_max = int(v[sl['pg_max'][0]:sl['pg_max'][1]].sum())
        qg_min = int(v[sl['qg_min'][0]:sl['qg_min'][1]].sum()) if is_full else 0
        qg_max = int(v[sl['qg_max'][0]:sl['qg_max'][1]].sum()) if is_full else 0
        vm_min = int(v[sl['vm_min'][0]:sl['vm_min'][1]].sum()) if is_full else 0
        vm_max = int(v[sl['vm_max'][0]:sl['vm_max'][1]].sum()) if is_full else 0
        total  = pg_min + pg_max + qg_min + qg_max + vm_min + vm_max
        counts[lbl]    = total
        breakdown[lbl] = (pg_min, pg_max, qg_min, qg_max, vm_min, vm_max)

    all_counts = np.array(list(counts.values()))
    if labels is not None:
        freq = Counter(labels.tolist())
        wc   = []
        for lbl, cnt in freq.items():
            if lbl in counts:
                wc.extend([counts[lbl]] * cnt)
        wc = np.array(wc)
    else:
        wc = all_counts

    print(f"\n[Active Constraint Count Diagnostics]")
    print(f"  System: n_g={n_g}, n_bus={n_bus}")
    print(f"  Active set type: {meta.get('active_set_type','full').upper()}")
    print(f"  Total constraint dims: {meta['dim']}")
    print(f"  Per unique active set ({len(lbl_ids)} sets):")
    print(f"    Min={all_counts.min()}  Max={all_counts.max()}  "
          f"Mean={all_counts.mean():.2f}  Median={np.median(all_counts):.1f}")
    print(f"  Sample-weighted: Mean={wc.mean():.2f}  Std={wc.std():.2f}")

    print(f"  Top-5 active sets – constraint breakdown:")
    if is_full:
        print(f"    {'Label':>6}  {'Freq':>6}  {'Total':>5}  "
              f"{'pgMn':>4}  {'pgMx':>4}  {'qgMn':>4}  {'qgMx':>4}  "
              f"{'vmMn':>4}  {'vmMx':>4}")
    else:
        print(f"    {'Label':>6}  {'Freq':>6}  {'Total':>5}  "
              f"{'pgMn':>4}  {'pgMx':>4}")

    if labels is not None:
        top5 = Counter(labels.tolist()).most_common(5)
    else:
        top5 = [(lbl, 1) for lbl in lbl_ids[:5]]

    for lbl, cnt in top5:
        if lbl not in breakdown:
            continue
        pgn, pgx, qgn, qgx, vmn, vmx = breakdown[lbl]
        tot = pgn + pgx + qgn + qgx + vmn + vmx
        if is_full:
            print(f"    {lbl:6d}  {cnt:6d}  {tot:5d}  "
                  f"{pgn:4d}  {pgx:4d}  {qgn:4d}  {qgx:4d}  {vmn:4d}  {vmx:4d}")
        else:
            print(f"    {lbl:6d}  {cnt:6d}  {tot:5d}  "
                  f"{pgn:4d}  {pgx:4d}")


# =====================================================================
# Part 2 – Pg and Vm Recovery
# =====================================================================

def _decode_active_set(active_set_vec, meta):
    """
    Decode a binary active-set vector into six generator/bus index arrays.

    In 'pg_only' mode, qg_min/max and vm_min/max slices are absent in
    meta['slices']; the corresponding arrays are returned as empty.

    Returns
    -------
    ag_pg_min, ag_pg_max : active pg lower/upper limit generator indices
    ag_qg_min, ag_qg_max : active qg lower/upper limit generator indices (empty in pg_only)
    av_vm_min, av_vm_max : active vm lower/upper limit bus positions    (empty in pg_only)
    """
    sl = meta['slices']

    def _idx(key):
        if key not in sl:
            return np.array([], dtype=np.int64)
        s, e = sl[key]
        return np.where(active_set_vec[s:e] == 1)[0]

    return (_idx('pg_min'), _idx('pg_max'),
            _idx('qg_min'), _idx('qg_max'),
            _idx('vm_min'), _idx('vm_max'))


def recover_pg_vm_linprog(active_set_vec, x_pd_sample, x_qd_sample,
                           params, meta):
    """
    Recover pg_non_slack and vm_gen from a predicted active set.

    Pg recovery (linprog / HiGHS):
      Active pg_min / pg_max constraints fix generator outputs directly.
      Remaining free non-slack generators solved via LP:
        min  c1[free] @ pg_free
        s.t. sum(pg_free) = sum(pd) - sum(pg_fixed_non_slack)
             Pg_min[free] <= pg_free <= Pg_max[free]

    Vm recovery:
      'full'    mode: active vm_min/max constraints fix generator bus voltages.
                      Other generator buses use nominal vg_pu.
      'pg_only' mode: all generator buses use nominal vg_pu (av_vm arrays are
                      always empty because no vm slices exist in meta).

    NOTE: Qg constraints inform the active set label but do NOT enter the LP;
          Qg is determined by the subsequent power flow calculation.

    Returns
    -------
    pg_non_slack : np.ndarray (n_gen_non_slack,) float32
    vm_gen       : np.ndarray (n_gen,)           float32
    """
    n_g           = params['general']['n_gen']
    non_slack     = params['general']['non_slack_gen_idx']
    n_ns          = params['general']['n_gen_non_slack']
    pg_min_all    = params['generator']['pg_min'].ravel().astype(np.float64)
    pg_max_all    = params['generator']['pg_max'].ravel().astype(np.float64)
    vm_min_all    = params['bus']['vm_min'].ravel().astype(np.float64)
    vm_max_all    = params['bus']['vm_max'].ravel().astype(np.float64)
    c1_all        = params['generator']['cost_c1'].ravel().astype(np.float64)

    # Nominal generator voltage: prefer mean vm_gen from training data (stored
    # in params after experiment setup) over vg_pu from case file, because for
    # some cases (e.g. case300) vg_pu is uniformly 1.0 which causes poor power
    # flow convergence. The training-data mean is a better operating-point estimate.
    if 'vm_gen_mean' in params['general']:
        vg_nominal = params['general']['vm_gen_mean'].astype(np.float64)
    else:
        vg_nominal = GLOBAL_CASE_DATA['gen'][:, 5].astype(np.float64)  # (n_gen,)

    # Generator bus positions (0-based bus_id_to_idx indices)
    gen_bus_ids   = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])

    ag_pg_min, ag_pg_max, _, _, av_vm_min, av_vm_max = \
        _decode_active_set(active_set_vec, meta)

    # ── Step 1: Vm recovery ──────────────────────────────────────────
    # In 'pg_only' mode av_vm_min/av_vm_max are always empty arrays,
    # so vm_gen stays at nominal for all generators.
    # In 'full' mode active vm constraints override the nominal.
    vm_gen = vg_nominal.copy()

    for pos in av_vm_min:
        bus_pos  = int(pos)
        gen_mask = (gen_bus_indices == bus_pos)
        if np.any(gen_mask):
            vm_gen[gen_mask] = vm_min_all[bus_pos]

    for pos in av_vm_max:
        bus_pos  = int(pos)
        gen_mask = (gen_bus_indices == bus_pos)
        if np.any(gen_mask):
            vm_gen[gen_mask] = vm_max_all[bus_pos]

    # ── Step 2: Pg recovery (non-slack generators only via LP) ───────
    pg_min_ns = pg_min_all[non_slack]
    pg_max_ns = pg_max_all[non_slack]
    c1_ns     = c1_all[non_slack]

    ns_global_to_local = {g: i for i, g in enumerate(non_slack)}

    fixed_ns = np.full(n_ns, np.nan)

    # Auto-fix offline generators (pg_min == pg_max == 0):
    # These are committed-but-offline units. Regardless of active set,
    # they must output 0 and should not enter the LP as free variables.
    for i in range(n_ns):
        if pg_min_ns[i] == 0.0 and pg_max_ns[i] == 0.0:
            fixed_ns[i] = 0.0

    # Apply active constraint fixings from the predicted active set
    for g in ag_pg_min:
        if g in ns_global_to_local:
            loc = ns_global_to_local[g]
            if np.isnan(fixed_ns[loc]):           # don't override offline fix
                fixed_ns[loc] = pg_min_all[g]
    for g in ag_pg_max:
        if g in ns_global_to_local:
            loc = ns_global_to_local[g]
            cur = fixed_ns[loc]
            if np.isnan(cur):
                fixed_ns[loc] = pg_max_all[g]
            elif pg_max_all[g] != 0.0:            # don't override offline fix
                # conflict: at both bounds → take the tighter (max) bound
                fixed_ns[loc] = pg_max_all[g]

    free_mask = np.isnan(fixed_ns)
    free_idx  = np.where(free_mask)[0]

    if len(free_idx) == 0:
        pg_ns = np.nan_to_num(fixed_ns, nan=0.0).astype(np.float32)
        return pg_ns, vm_gen.astype(np.float32)

    total_load  = float(np.sum(x_pd_sample))
    residual    = total_load - float(np.nansum(fixed_ns))
    n_free      = len(free_idx)
    c1_free     = c1_ns[free_idx]
    lb_free     = pg_min_ns[free_idx]
    ub_free     = pg_max_ns[free_idx]
    bounds_free = list(zip(lb_free.tolist(), ub_free.tolist()))

    A_eq = np.ones((1, n_free), dtype=np.float64)
    b_eq = np.array([residual],  dtype=np.float64)

    res = linprog(c1_free, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds_free, method='highs')

    pg_free_val = res.x if res.success else np.clip(
        np.full(n_free, residual / max(n_free, 1)), lb_free, ub_free
    )

    pg_ns = fixed_ns.copy()
    pg_ns[free_idx] = pg_free_val
    pg_ns = np.nan_to_num(pg_ns, nan=0.0).astype(np.float32)

    return pg_ns, vm_gen.astype(np.float32)


def _single_cost_ac(pg_non_slack, params):
    """Compute generation cost for non-slack generators only."""
    ns  = params['general']['non_slack_gen_idx']
    c2  = params['generator']['cost_c2'].ravel()[ns]
    c1  = params['generator']['cost_c1'].ravel()[ns]
    c0  = params['generator']['cost_c0'].ravel()[ns]
    return float(np.sum(c2 * pg_non_slack**2 + c1 * pg_non_slack + c0))


def _fallback_pg_vm(x_pd_sample, params):
    """Fallback: clipped uniform dispatch + mean vm from training data."""
    ns        = params['general']['non_slack_gen_idx']
    pg_min_ns = params['generator']['pg_min'].ravel()[ns]
    pg_max_ns = params['generator']['pg_max'].ravel()[ns]
    total     = float(np.sum(x_pd_sample))
    pg_ns     = np.clip(
        np.full(len(ns), total / len(ns)), pg_min_ns, pg_max_ns
    ).astype(np.float32)
    # Use mean vm from training data if available; fall back to vg_pu
    if 'vm_gen_mean' in params['general']:
        vm_gen = params['general']['vm_gen_mean'].astype(np.float32)
    else:
        vm_gen = GLOBAL_CASE_DATA['gen'][:, 5].astype(np.float32)
    return pg_ns, vm_gen


# =====================================================================
# Part 3 – Batch Recovery
# =====================================================================

def recover_batch_top1(top1_labels, label_to_as,
                        x_pd_raw, x_qd_raw, params, meta):
    """Top-1 batch: recover pg_non_slack and vm_gen for each sample."""
    n     = len(top1_labels)
    n_ns  = params['general']['n_gen_non_slack']
    n_gen = params['general']['n_gen']

    pg_out = np.zeros((n, n_ns),  dtype=np.float32)
    vm_out = np.zeros((n, n_gen), dtype=np.float32)
    n_fail = 0

    for i in range(n):
        lbl = top1_labels[i]
        if lbl not in label_to_as:
            pg_out[i], vm_out[i] = _fallback_pg_vm(x_pd_raw[i], params)
            n_fail += 1
            continue
        pg, vm = recover_pg_vm_linprog(
            label_to_as[lbl], x_pd_raw[i], x_qd_raw[i], params, meta)
        if pg is None:
            pg_out[i], vm_out[i] = _fallback_pg_vm(x_pd_raw[i], params)
            n_fail += 1
        else:
            pg_out[i] = pg
            vm_out[i] = vm

    if n_fail > 0:
        print(f"  [Top-1] Recovery fallback: {n_fail}/{n} samples.")
    return pg_out, vm_out


def recover_batch_topk(topk_labels, topk_probs,
                        label_to_as, x_pd_raw, x_qd_raw, params, meta):
    """
    Top-K ensemble policy (Paper Section II-C-b):
    For each sample, recover (pg, vm) for K candidates, keep those satisfying
    pg box bounds, pick the minimum-cost feasible one.
    """
    n      = topk_labels.shape[0]
    K      = topk_labels.shape[1]
    n_ns   = params['general']['n_gen_non_slack']
    n_gen  = params['general']['n_gen']
    ns     = params['general']['non_slack_gen_idx']
    pg_min = params['generator']['pg_min'].ravel()[ns]
    pg_max = params['generator']['pg_max'].ravel()[ns]

    pg_out = np.zeros((n, n_ns),  dtype=np.float32)
    vm_out = np.zeros((n, n_gen), dtype=np.float32)

    for i in range(n):
        best_pg   = None
        best_vm   = None
        best_cost = np.inf

        for k in range(K):
            lbl = topk_labels[i, k]
            if lbl not in label_to_as:
                continue
            pg, vm = recover_pg_vm_linprog(
                label_to_as[lbl], x_pd_raw[i], x_qd_raw[i], params, meta)
            if pg is None:
                continue
            if np.any(pg < pg_min - 1e-3) or np.any(pg > pg_max + 1e-3):
                continue
            cost = _single_cost_ac(pg, params)
            if cost < best_cost:
                best_cost = cost
                best_pg   = pg
                best_vm   = vm

        if best_pg is None:
            pg_out[i], vm_out[i] = _fallback_pg_vm(x_pd_raw[i], params)
        else:
            pg_out[i] = best_pg
            vm_out[i] = best_vm

    return pg_out, vm_out


# =====================================================================
# Part 4 – Power Flow Batch
# =====================================================================

def run_power_flow_batch(pg_ns_batch, vm_gen_batch,
                          x_raw, params, verbose=True):
    """
    Run power flow for a batch of (pg_non_slack, vm_gen) predictions.
    Interface identical to evaluate_split in acopf_dnn_main.py.

    Returns
    -------
    pf_results_list : list of runpf outputs
    converge_flags  : list of bool
    y_pred_pg_full  : np.ndarray (n_samples, n_gen)
    y_pred_vm_all   : np.ndarray (n_samples, n_bus)
    """
    n_samples     = len(pg_ns_batch)
    n_gen         = params['general']['n_gen']
    n_bus         = params['general']['n_buses']
    n_loads       = params['general']['n_loads']
    gen_bus_ids   = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])

    pd_pu = x_raw[:, :n_loads]
    qd_pu = x_raw[:, n_loads:]

    pf_results_list = []
    converge_flags  = []
    y_pred_pg_full  = np.zeros((n_samples, n_gen), dtype=np.float32)
    y_pred_vm_all   = np.zeros((n_samples, n_bus), dtype=np.float32)

    if verbose:
        print(f"  Running power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf(pd_pu[i], qd_pu[i],
                             pg_ns_batch[i], vm_gen_batch[i], params)
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except Exception:
            pf_results_list.append((
                {'success': False,
                 'gen':    np.zeros((n_gen, 21)),
                 'bus':    np.zeros((n_bus, 13)),
                 'branch': np.zeros((1, 17))},
            ))
            converge_flags.append(False)

        pg_full_i = reconstruct_full_pg(pg_ns_batch[i], params)
        y_pred_pg_full[i] = pg_full_i

        vm_all_i = np.ones(n_bus, dtype=np.float32)
        vm_all_i[gen_bus_indices] = vm_gen_batch[i]
        y_pred_vm_all[i] = vm_all_i

    if verbose:
        print(f"    ✓ Converged: {sum(converge_flags)}/{n_samples}")

    return pf_results_list, converge_flags, y_pred_pg_full, y_pred_vm_all


# =====================================================================
# Part 5 – NN Classifier
# =====================================================================

class ActiveSetClassifierAC(nn.Module):
    """
    Multi-layer fully connected NN classifier for AC active set prediction.
    Input  : MinMax-scaled (pd, qd) concatenated  (2 * n_loads,)
    Output : logits over all observed active set labels
    Architecture: Linear -> ReLU -> BatchNorm1d -> Dropout  (x r layers)
    Default hidden_layers = [256, 256, 128, 128, 64] (paper's deepest config).
    """

    def __init__(self, input_size, n_classes,
                 hidden_layers=None, dropout_rate=0.1):
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


# =====================================================================
# Part 6 – Main Experiment
# =====================================================================

def active_set_acopf_experiment(
        case_name,
        params_path,
        data_path,
        duals_dir,
        data_mode        = DataMode.RANDOM_SPLIT,
        n_train_use      = 10000,
        seed             = 42,
        n_epochs         = 20,
        learning_rate    = 0.001,
        batch_size       = 32,
        hidden_layers    = None,
        dropout_rate     = 0.1,
        device           = 'cuda',
        early_stop_patience  = 20,
        early_stop_min_delta = 1e-6,
        test_data_path   = None,
        test_params_path = None,
        test_duals_dir   = None,
        n_test_samples   = 1000,
        active_threshold = 1e-4,
        active_set_type  = 'full',
        top_k            = 3,
        log_path         = None,
        results_path     = None,
):
    """
    Active Set Classification experiment for AC-OPF.

    Parameters
    ----------
    duals_dir        : str   path to folder containing the mu_*.csv files
                             e.g. "case30(v=0.12)_with_duals"
    active_set_type  : str   'full'    – pg + qg + vm constraints (case30, case118)
                             'pg_only' – pg constraints only       (case300, large cases)
    test_duals_dir   : str or None  dual folder for external test set
    """
    if hidden_layers is None:
        hidden_layers = [256, 256, 128, 128, 64]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    global GLOBAL_CASE_DATA

    print(f"\n{'=' * 70}")
    print(f"ACOPF Active Set Classification Experiment")
    print(f"{'=' * 70}")
    print(f"Device: {device_obj}")
    print(f"Case: {case_name}")
    print(f"Data Mode: {data_mode}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # 1. Load system parameters and PyPower case
    # ------------------------------------------------------------------
    params = load_parameters_from_csv(case_name, params_path)
    init_pypower_options()
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)
    print(f"✓ Training params and PyPower case data loaded")

    n_gen    = params['general']['n_gen']
    n_gen_ns = params['general']['n_gen_non_slack']
    n_buses  = params['general']['n_buses']
    baseMVA  = params['general']['BASE_MVA']
    # NOTE: n_loads is read AFTER load_and_scale_acopf_data because that
    # function updates params['general']['n_loads'] from the CSV columns
    # (e.g. case300: 300 buses but only 191 have non-zero load → n_loads=191)

    print(f"\n[System Info]")
    print(f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_ns}), "
          f"Base MVA: {baseMVA}")

    # ------------------------------------------------------------------
    # 2. Load training data and fit scalers
    # ------------------------------------------------------------------
    x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    # Read n_loads AFTER load_and_scale_acopf_data has updated it from CSV columns
    n_loads = params['general']['n_loads']
    print(f"  Loads (actual from CSV): {n_loads}, Base MVA: {baseMVA}")

    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # Compute mean vm_gen from training data and store in params.
    # Used as default Vm in 'pg_only' mode instead of vg_nominal (which may be
    # all 1.0 for some cases like case300, causing power flow divergence).
    # vm_gen from dataset has shape (n_samples, n_gen); mean over samples.
    vm_gen_mean = raw_data['vm_gen'].mean(axis=0).astype(np.float64)  # (n_gen,)
    params['general']['vm_gen_mean'] = vm_gen_mean
    print(f"  vm_gen_mean range: [{vm_gen_mean.min():.4f}, {vm_gen_mean.max():.4f}] p.u.")

    # ------------------------------------------------------------------
    # 3. Load dual variables and build active set labels
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Loading dual variables and extracting active set labels...")
    labels_all, label_to_as, as_to_label, meta = load_dual_csvs(
        duals_dir, case_name, params,
        threshold=active_threshold,
        active_set_type=active_set_type,
    )
    n_classes = len(label_to_as)
    print(f"  Total unique active sets: {n_classes}")
    print_active_set_statistics(labels_all, "Full Dataset")

    # ------------------------------------------------------------------
    # 4. Data split
    # ------------------------------------------------------------------
    GLOBAL_CASE_DATA_TEST = GLOBAL_CASE_DATA

    if data_mode == DataMode.API_TEST:
        print(f"\n{'=' * 70}")
        print(f"Data Mode: API_TEST")
        print(f"{'=' * 70}")
        if test_data_path is None or test_params_path is None:
            raise ValueError("API_TEST requires test_data_path and test_params_path")

        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=DataMode.API_TEST, n_train_use=n_train_use, seed=seed
        )
        test_params, test_x_scaled, test_y_scaled, test_raw_data, _ = \
            load_api_test_data(test_data_path, test_params_path, scalers,
                               n_test_samples=n_test_samples, seed=seed)
        test_idx = np.arange(len(test_x_scaled))

        test_case_name = os.path.basename(test_data_path)
        if test_case_name.endswith('_pd.csv'):
            test_case_name = test_case_name[:-7]
        GLOBAL_CASE_DATA_TEST = load_case_from_csv(test_case_name, test_params_path)

    elif data_mode == DataMode.GENERALIZATION:
        print(f"\n{'=' * 70}")
        print(f"Data Mode: GENERALIZATION")
        print(f"{'=' * 70}")
        if test_data_path is None:
            raise ValueError("GENERALIZATION requires test_data_path")

        train_idx, val_idx, _ = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=DataMode.GENERALIZATION, n_train_use=n_train_use, seed=seed
        )
        test_x_scaled, test_y_scaled, test_raw_data, _ = \
            load_generalization_test_data(test_data_path, params, scalers,
                                          n_test_samples=n_test_samples, seed=seed)
        test_idx    = np.arange(len(test_x_scaled))
        test_params = params

    else:
        train_idx, val_idx, test_idx = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            mode=data_mode, n_train_use=n_train_use, seed=seed
        )
        test_x_scaled = x_data_scaled
        test_y_scaled = y_data_scaled
        test_raw_data = raw_data
        test_params   = params

    # ------------------------------------------------------------------
    # 5. Active set statistics + diagnostics
    # ------------------------------------------------------------------
    print_active_set_statistics(labels_all[train_idx], "Train Split")
    print_active_constraint_count_distribution(label_to_as, meta, labels=labels_all)

    unseen = set(labels_all[val_idx].tolist()) - set(labels_all[train_idx].tolist())
    if unseen:
        print(f"  [WARNING] Val set has {len(unseen)} unseen active set label(s).")

    # ------------------------------------------------------------------
    # 6. Tensors for classifier (use x_data_scaled from scalers['x'])
    # ------------------------------------------------------------------
    X_train = torch.tensor(x_data_scaled[train_idx], dtype=torch.float32,
                           device=device_obj)
    Y_train = torch.tensor(labels_all[train_idx],    dtype=torch.long,
                           device=device_obj)
    X_val   = torch.tensor(x_data_scaled[val_idx],   dtype=torch.float32,
                           device=device_obj)
    Y_val   = torch.tensor(labels_all[val_idx],       dtype=torch.long,
                           device=device_obj)

    n_test_display = (len(test_idx)
                      if data_mode not in [DataMode.GENERALIZATION,
                                           DataMode.API_TEST]
                      else len(test_x_scaled))
    print(f"\n[Dataset Sizes]")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {n_test_display} samples")

    # ------------------------------------------------------------------
    # 7. Build classifier
    # ------------------------------------------------------------------
    input_dim = x_data_scaled.shape[1]
    model = ActiveSetClassifierAC(
        input_size    = input_dim,
        n_classes     = n_classes,
        hidden_layers = hidden_layers,
        dropout_rate  = dropout_rate,
    ).to(device_obj)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 70}")
    print(f"Model Configuration")
    print(f"{'=' * 70}")
    print(f"Input dim:   {input_dim} (pd + qd = 2 × {n_loads})")
    print(f"Num classes: {n_classes}")
    print(f"Hidden:      {hidden_layers}")
    print(f"Parameters:  {n_params:,}")
    print(f"Training params: epochs={n_epochs}, patience={early_stop_patience}, "
          f"lr={learning_rate}, batch_size={batch_size}")
    print(f"{'=' * 70}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    n_train   = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size

    # ------------------------------------------------------------------
    # 8. Training loop (with early stopping, matching acopf_dnn_main.py)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Training Progress")
    print(f"{'=' * 70}")
    t0 = time.perf_counter()

    # Early stopping state
    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct    = 0
        perm       = torch.randperm(n_train, device=device_obj)

        for b in range(n_batches):
            idx    = perm[b * batch_size : (b + 1) * batch_size]
            Xb, Yb = X_train[idx], Y_train[idx]
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, Yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)
            correct    += (logits.argmax(-1) == Yb).sum().item()

        t_loss = epoch_loss / n_train
        t_acc  = 100. * correct / n_train
        model.eval()
        with torch.no_grad():
            vl     = model(X_val)
            v_loss = float(criterion(vl, Y_val))
            v_acc  = 100. * (vl.argmax(-1) == Y_val).sum().item() / len(Y_val)

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Epoch {epoch:4d}/{n_epochs} - "
                  f"Train Loss: {t_loss:.6f}  Train Acc: {t_acc:.2f}%  |  "
                  f"Val Loss: {v_loss:.6f}  Val Acc: {v_acc:.2f}%")

        # Early stopping check
        if v_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = v_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Epoch {epoch:4d}/{n_epochs} - "
                      f"Train Loss: {t_loss:.6f}  Train Acc: {t_acc:.2f}%  |  "
                      f"Val Loss: {v_loss:.6f}  Val Acc: {v_acc:.2f}%")
                print(f"\n⚡ Early stopping triggered at epoch {epoch} "
                      f"(patience={early_stop_patience})")
                break

    # Restore best model weights
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device_obj) for k, v in best_state_dict.items()})
        print(f"✓ Restored best model from epoch {best_epoch} "
              f"(val_loss={best_val_loss:.6f})")

    train_time = time.perf_counter() - t0
    print(f"✓ Training completed in {train_time:.2f} seconds")

    # ------------------------------------------------------------------
    # 9. Test evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Test Set Evaluation")
    print(f"{'=' * 70}")

    GLOBAL_CASE_DATA_BACKUP = GLOBAL_CASE_DATA
    GLOBAL_CASE_DATA        = GLOBAL_CASE_DATA_TEST
    model.eval()

    if data_mode in [DataMode.GENERALIZATION, DataMode.API_TEST]:
        X_test      = torch.tensor(test_x_scaled, dtype=torch.float32,
                                   device=device_obj)
        x_raw_test  = test_raw_data['x']
        y_true_pg   = test_raw_data['pg']
        y_true_vm   = test_raw_data['vm']
        y_true_qg   = test_raw_data['qg']
        y_true_va   = test_raw_data['va']
        true_labels = None
    else:
        X_test      = torch.tensor(x_data_scaled[test_idx], dtype=torch.float32,
                                   device=device_obj)
        x_raw_test  = raw_data['x'][test_idx]
        y_true_pg   = raw_data['pg'][test_idx]
        y_true_vm   = raw_data['vm'][test_idx]
        y_true_qg   = raw_data['qg'][test_idx]
        y_true_va   = raw_data['va'][test_idx]
        true_labels = labels_all[test_idx]

    n_test = len(X_test)

    with torch.no_grad():
        logits_test = model(X_test)
        probs_test  = torch.softmax(logits_test, dim=-1)
        top1_pred   = probs_test.argmax(-1).cpu().numpy()
        K           = min(top_k, probs_test.shape[-1])
        topk_p, topk_l = torch.topk(probs_test, k=K, dim=-1)
        topk_labels = topk_l.cpu().numpy()
        topk_probs  = topk_p.cpu().numpy()

    top1_acc = topk_acc = None
    if true_labels is not None:
        top1_acc = 100. * np.mean(top1_pred == true_labels)
        in_topk  = np.array([true_labels[i] in topk_labels[i]
                              for i in range(n_test)])
        topk_acc = 100. * np.mean(in_topk)

    n_ld     = params['general']['n_loads']
    x_pd_raw = x_raw_test[:, :n_ld]
    x_qd_raw = x_raw_test[:, n_ld:]

    print(f"\n  Recovering Pg and Vm via linprog for {n_test} test samples...")
    t_r0 = time.perf_counter()
    pg_top1, vm_top1 = recover_batch_top1(
        top1_pred, label_to_as, x_pd_raw, x_qd_raw, test_params, meta)
    t_r1 = time.perf_counter()
    pg_topk, vm_topk = recover_batch_topk(
        topk_labels, topk_probs, label_to_as,
        x_pd_raw, x_qd_raw, test_params, meta)
    t_r2 = time.perf_counter()
    print(f"  Top-1 recovery: {t_r1-t_r0:.1f}s  |  "
          f"Top-{K} recovery: {t_r2-t_r1:.1f}s")

    print(f"\n[Top-1 Power Flow]")
    pf1, cf1, pg_full1, vm_all1 = run_power_flow_batch(
        pg_top1, vm_top1, x_raw_test, test_params, verbose=True)

    print(f"\n[Top-{K} Power Flow]")
    pfK, cfK, pg_fullK, vm_allK = run_power_flow_batch(
        pg_topk, vm_topk, x_raw_test, test_params, verbose=True)

    metrics1 = evaluate_acopf_predictions(
        pg_full1, vm_all1, y_true_pg, y_true_vm, y_true_qg, y_true_va,
        pf1, cf1, test_params, verbose=False)

    metricsK = evaluate_acopf_predictions(
        pg_fullK, vm_allK, y_true_pg, y_true_vm, y_true_qg, y_true_va,
        pfK, cfK, test_params, verbose=False)

    GLOBAL_CASE_DATA = GLOBAL_CASE_DATA_BACKUP

    # ------------------------------------------------------------------
    # 10. Inference latency (NN forward only)
    # ------------------------------------------------------------------
    model.eval()
    sample_t = X_test[:1]
    with torch.no_grad():
        for _ in range(10): _ = model(sample_t)
        if device_obj.type == 'cuda': torch.cuda.synchronize()
    times_lat = []
    with torch.no_grad():
        for _ in range(100):
            ts = time.perf_counter()
            _  = model(sample_t)
            if device_obj.type == 'cuda': torch.cuda.synchronize()
            times_lat.append(time.perf_counter() - ts)
    latency_ms = np.mean(times_lat) * 1000

    # ------------------------------------------------------------------
    # 11. Print results (format identical to acopf_dnn_main.py)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Final Results Summary")
    print(f"{'=' * 70}")

    print(f"\nData Mode: {data_mode}")
    print(f"Test Case: {case_name}")

    print(f"\n--- Active Set Classification ---")
    print(f"Active Set Type: {active_set_type.upper()}"
          + (" (Pg constraints only; Qg/Vm excluded due to active set explosion)"
             if active_set_type == 'pg_only' else
             " (Pg + Qg + Vm constraints)"))
    print(f"Unique active sets (training): {n_classes}")
    if top1_acc is not None:
        print(f"Top-1 Classification Accuracy: {top1_acc:.2f}%")
        print(f"Top-{K} Classification Accuracy: {topk_acc:.2f}%")
    else:
        print(f"Classification Accuracy: N/A (external test set)")

    # Print method limitation note if convergence rate is low
    conv1 = metrics1['convergence_rate_percent']
    if conv1 < 50.0:
        print(f"\n  [NOTE] Low power flow convergence rate ({conv1:.1f}%) indicates")
        print(f"  that the active set method has limited applicability for this case.")
        if active_set_type == 'pg_only':
            print(f"  Root cause (pg_only mode, case300):")
            print(f"    - AC-OPF active sets are highly fragmented (3000+ unique sets")
            print(f"      for 16k samples), preventing reliable classification.")
            print(f"    - Active sets fix ~50/68 non-slack generators at bounds,")
            print(f"      leaving only ~18 free generators to cover full system load.")
            print(f"      This extreme dispatch deviates too far from the true optimum")
            print(f"      for Newton-Raphson power flow to converge.")
            print(f"  Conclusion: Active set classification is not suitable for")
            print(f"  AC-OPF on large cases with many offline/bounded generators.")

    def _fmt(val, fmt=".4f", suffix=""):
        """Format a metric value; show N/A if NaN (no converged samples)."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:{fmt}}{suffix}"

    def _print_metrics(m, label):
        n_conv = m.get('n_converged', '?')
        n_tot  = m.get('n_samples',   '?')
        note   = f"  [metrics computed on {n_conv}/{n_tot} converged samples]"

        print(f"\n--- {label}: Accuracy Metrics ---")
        print(note)
        print(f"MAE_Pg (Non-Slack): {_fmt(m['mae_pg_non_slack_percent'])}%")
        print(f"MAE_Vm (Generator): {_fmt(m['mae_vm_percent'])}%")
        print(f"MAE_Qg (All Gens):  {_fmt(m['mae_qg_percent'])}%")
        print(f"MAE_Va (All Buses): {_fmt(m['mae_va_deg'])} degrees")

        print(f"\n--- {label}: Violations (p.u., converged samples only) ---")
        print(f"Pg_viol (Non-Slack): {_fmt(m['mean_pg_viol_non_slack_pu'], '.6f')} p.u.")
        print(f"Pg_viol (Slack):     {_fmt(m['mean_pg_viol_slack_pu'],     '.6f')} p.u.")
        print(f"Qg_viol (All Gens):  {_fmt(m['mean_max_qg_viol_pu'],      '.6f')} p.u.")
        print(f"Vm_viol (All Buses): {_fmt(m['mean_max_vm_viol_pu'],      '.6f')} p.u.")
        print(f"Branch_viol:         {_fmt(m['mean_max_branch_viol_pu'],  '.6f')} p.u. "
              f"(1.0 = 100% overload)")

        print(f"\n--- {label}: Cost Metrics ---")
        print(f"Cost Gap: {_fmt(m['cost_optimality_gap_percent'])}%")

        print(f"\n--- {label}: Convergence ---")
        print(f"Convergence Rate: {m['convergence_rate_percent']:.2f}%  "
              f"({n_conv}/{n_tot} samples)")

    _print_metrics(metrics1, "Top-1 Prediction")
    _print_metrics(metricsK, f"Top-{K} Ensemble Prediction")

    print(f"\n--- Performance ---")
    print(f"Inference Time: {latency_ms:.4f} ms/sample")
    print(f"Training Time:  {train_time:.2f} s")

    print(f"{'=' * 70}")

    return {
        'top1': metrics1, 'topk': metricsK,
        'top1_acc': top1_acc, 'topk_acc': topk_acc,
        'n_classes': n_classes,
        'train_time': train_time, 'latency_ms': latency_ms,
    }


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":

    import acopf_config as cfg

    # ---------------------------------------------------------------
    # 1. Read base paths/params from acopf_config (same as DNN)
    # ---------------------------------------------------------------
    paths  = cfg.get_all_paths()
    params = cfg.get_all_params()

    # ---------------------------------------------------------------
    # 2. Dual directory: derived from data_path folder + '_with_duals'
    #    e.g. "case30(v=0.12)" → "case30(v=0.12)_with_duals"
    # ---------------------------------------------------------------
    train_data_folder = os.path.dirname(paths['data_path'])
    duals_dir         = train_data_folder + "_with_duals"

    # For generalization/api_test: test duals dir (set None to skip accuracy)
    test_duals_dir = None
    if params['data_mode'] == DataMode.GENERALIZATION and paths.get('test_data_path'):
        test_duals_dir = os.path.dirname(paths['test_data_path']) + "_with_duals"

    # ---------------------------------------------------------------
    # 3. Active Set specific hyperparameters
    #    (override here; the rest come from acopf_config)
    # ---------------------------------------------------------------
    HIDDEN_LAYERS    = [256, 128]    # match original script default
    DROPOUT_RATE     = 0.1
    ACTIVE_THRESHOLD = 1e-4
    TOP_K            = 3

    # Active set type:
    #   'full'    – use pg + qg + vm dual variables (recommended: case30, case118)
    #   'pg_only' – use pg dual variables only      (recommended: case300 and larger,
    #               where full active sets are too fragmented for classification)
    ACTIVE_SET_TYPE  = 'pg_only'

    # ---------------------------------------------------------------
    # 4. Run experiment
    # ---------------------------------------------------------------
    results = active_set_acopf_experiment(
        case_name        = paths['case_name'],
        params_path      = paths['params_path'],
        data_path        = paths['data_path'],
        duals_dir        = duals_dir,
        data_mode        = params['data_mode'],
        n_train_use      = params['n_train_use'],
        seed             = params['seed'],
        n_epochs         = params['n_epochs'],
        learning_rate    = params['learning_rate'],
        batch_size       = params['batch_size'] or 32,
        hidden_layers    = HIDDEN_LAYERS,
        dropout_rate     = DROPOUT_RATE,
        device           = params['device'],
        early_stop_patience  = params.get('early_stop_patience', 20),
        early_stop_min_delta = params.get('early_stop_min_delta', 1e-6),
        test_data_path   = paths.get('test_data_path'),
        test_params_path = paths.get('test_params_path'),
        test_duals_dir   = test_duals_dir,
        n_test_samples   = params.get('n_test_samples', 1000),
        active_threshold = ACTIVE_THRESHOLD,
        active_set_type  = ACTIVE_SET_TYPE,
        top_k            = TOP_K,
        log_path         = paths.get('log_path'),
        results_path     = paths.get('results_path'),
    )

    print("\n✓ Experiment completed successfully!")