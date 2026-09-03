# -*- coding: utf-8 -*-
"""
Active Set Classification DCOPF Main Experiment (PyTorch)
Based on: "Learning for DC-OPF: Classifying active sets using neural nets"
          Deka & Misra, arXiv:1902.05607v1

Version: v2.1 - Honest failure handling (no silent fallback)

Changes in v2.1 (vs v2.0):
- CHANGE 1: recover_pg_linprog no longer silently degenerates to a
  power-balance-only LP when line-flow equalities are infeasible.
  It now returns None instead, honestly reporting that the predicted
  active set is infeasible at the current load. This prevents the
  "low-cost but infeasible decoy" solution that was corrupting the
  Top-K ensemble.
- CHANGE 2: recover_pg_topk_batch now (a) discards any candidate that
  returns None, (b) additionally checks branch (line) violation, not
  just generator box bounds, before accepting a candidate. Only if ALL
  K candidates fail does it fall back to _fallback_pg.
- CHANGE 3: recover_pg_top1_batch keeps _fallback_pg (evaluation must
  produce a number), but failures are now VISIBLE: a failed Top-1 falls
  to a uniform-dispatch fallback that intentionally worsens metrics,
  so errors surface instead of being hidden.

Fixes inherited from v2.0:
- BUG FIX 1: load_data_for_active_set now uses bus_id_to_idx mapping
  instead of assuming bus_id == array_index (bus_id - 1)
- BUG FIX 2: pg columns now explicitly follow g_bus order from params,
  instead of unsorted column scanning

Method Overview:
  1. Extract active set labels from dual variables in the dataset
     - active constraint: dual > threshold (1e-4)
     - each unique binary vector of active constraints = one integer label

  2. Train a NN multi-class classifier (cross-entropy loss, Adam):
        input : MinMax-scaled bus loads  (n_buses,)
        output: probability over all observed active set labels

  3. Recover Pg from predicted active set (Paper Eq.3 spirit):
     Given active set A, the active constraints become EQUALITY constraints.
     Together with the power-balance equality, we solve a constrained LP:
        min   c1 @ pg                  (linear cost, paper Eq.1a)
        s.t.  pg_i = Pg_min_i        for i in active_g_min
              pg_i = Pg_max_i        for i in active_g_max
              PTDF[:,k]^T(Map_g pg - pd) =  Pl_max_k  for k in active_line_pos
              PTDF[:,k]^T(Map_g pg - pd) = -Pl_max_k  for k in active_line_neg
              Pg_min <= pg <= Pg_max  (always)
              sum(pg) = sum(pd)       (power balance, always)
     Solver: scipy.optimize.linprog (method='highs')

     IMPORTANT (v2.1): If the active-set equalities make the LP infeasible,
     recover_pg_linprog returns None. We do NOT silently drop the line
     equalities, because that produces a feasible-looking but constraint-
     violating solution whose (artificially low) cost corrupts the Top-K
     ensemble selection.

  4. Top-1  : directly use argmax predicted label -> recover Pg via LP.
              If recovery fails, fall back to uniform dispatch (visible,
              metric-worsening) so the failure is not hidden.
     Top-K  : recover Pg for top-K labels, DISCARD any that fail recovery
              or violate generator/line limits, pick the feasible candidate
              with minimum cost (ensemble policy, Paper Section II-C-b).

  5. Evaluate with the SAME strict metrics as dnn_dcopf_main.py

Evaluation Metrics (identical print format to dnn_dcopf_main.py):
  - MAE Pg (%)             - Non-Slack & Slack-Only
  - Pg Violation (p.u.)    - Non-Slack & Slack-Only, Mean of Max
  - Branch Violation (p.u.)- Mean of Max
  - Cost Gap (%)
  - Training Time (s)
  - Inference Time (ms)

Additional output:
  - Unique active sets count
  - Top-1 / Top-K Classification Accuracy (%) [when labels available]

Data Split Modes supported:
  RANDOM_SPLIT, VALID_FIXED, GENERALIZATION, API_TEST
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
import copy
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linprog

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dcopf_data_setup import load_parameters_from_csv, DataSplitMode, split_data_by_mode
from dcopf_violation_metrics import (
    feasibility as dc_feasibility,
    compute_cost,
    compute_cost_gap_percentage,
    compute_branch_violation_pu,
)
from dcopf_slack_utils import (
    identify_slack_bus_and_gens,
    update_params_with_slack_info,
    compute_detailed_mae,
    compute_detailed_pg_violations_pu,
)


# =====================================================================
# Part 1 - Active Set Extraction
# =====================================================================

def extract_active_sets(full_df, params, column_names, threshold=1e-4):
    """
    Build integer labels from dual variables.

    For each sample, a binary vector is formed:
        [ mu_g_min_1..n_g | mu_g_max_1..n_g |
          mu_line_pos_valid | mu_line_neg_valid ]
    Each unique binary vector -> one integer label.

    Returns
    -------
    labels              : np.ndarray (n_samples,) int64
    label_to_active_set : dict  {int -> np.int8 array (n_constraints,)}
    active_set_to_label : dict  {tuple -> int}
    active_set_cols     : list[str]  (column order used to build the vector)
    """
    g_bus               = params['general']['g_bus']
    valid_branch_indices = np.where(params['constraints']['Pl_max'] < 1e10)[0]
    valid_branch_ids    = params['general']['branch_ids'][valid_branch_indices]

    mu_g_min_cols = [f"{column_names['mu_g_min_prefix']}{i}"   for i in g_bus]
    mu_g_max_cols = [f"{column_names['mu_g_max_prefix']}{i}"   for i in g_bus]
    mu_lpos_cols  = [f"{column_names['mu_line_pos_prefix']}{i}" for i in valid_branch_ids]
    mu_lneg_cols  = [f"{column_names['mu_line_neg_prefix']}{i}" for i in valid_branch_ids]
    active_set_cols = mu_g_min_cols + mu_g_max_cols + mu_lpos_cols + mu_lneg_cols

    dual_matrix   = full_df[active_set_cols].values.astype('float32')
    active_matrix = (dual_matrix > threshold).astype(np.int8)   # 0 or 1

    active_set_to_label = {}
    label_to_active_set = {}
    labels = np.zeros(len(full_df), dtype=np.int64)

    for i, row in enumerate(active_matrix):
        key = tuple(row.tolist())
        if key not in active_set_to_label:
            lbl = len(active_set_to_label)
            active_set_to_label[key] = lbl
            label_to_active_set[lbl] = np.array(row, dtype=np.int8)
        labels[i] = active_set_to_label[key]

    return labels, label_to_active_set, active_set_to_label, active_set_cols


def print_active_set_statistics(labels, split_name="Dataset"):
    counter  = Counter(labels.tolist())
    n_unique = len(counter)
    n_total  = len(labels)
    top5     = counter.most_common(5)
    print(f"\n[Active Set Statistics - {split_name}]")
    print(f"  Total samples:      {n_total}")
    print(f"  Unique active sets: {n_unique}")
    print(f"  Top-5 most common:")
    for lbl, cnt in top5:
        print(f"    Label {lbl:5d}: {cnt:6d} samples ({100.*cnt/n_total:.2f}%)")


def print_active_constraint_count_distribution(label_to_active_set, params, labels=None):
    """
    Diagnose how many constraints are active per active set.

    This is critical for understanding whether Paper Eq.3 (matrix inversion)
    strictly applies:
      - Paper assumption: exactly n_g - 1 active constraints per sample
        (LP optimal solution lies at a vertex defined by n_g-1 tight constraints
         plus the power balance equality -> square system, unique solution)
      - If actual count << n_g: system is underdetermined -> linprog gives a
        valid but non-unique solution (still correct for our LP)
      - If actual count >> n_g: system is overdetermined -> constraints may
        conflict, linprog falls back to power-balance-only

    Breakdown by constraint type:
      - g_min  : generator lower limit active  (pg_i = Pg_min_i)
      - g_max  : generator upper limit active  (pg_i = Pg_max_i)
      - l_pos  : line positive flow limit active
      - l_neg  : line negative flow limit active

    Parameters
    ----------
    label_to_active_set : dict  {int -> np.int8 array}
    params              : dict  (system parameters, used to get n_g, n_valid)
    labels              : np.ndarray or None
        If provided, weights statistics by sample frequency.
        If None, each unique active set is counted once.
    """
    n_g     = params['general']['n_g']
    Pl_max  = params['constraints']['Pl_max']
    n_valid = int(np.sum(Pl_max < 1e10))
    n_total_constraints = 2 * n_g + 2 * n_valid

    # Per-label active count
    lbl_ids   = sorted(label_to_active_set.keys())
    counts_per_label = {}          # {lbl: total active count}
    breakdown_per_label = {}       # {lbl: (g_min, g_max, l_pos, l_neg)}

    for lbl in lbl_ids:
        vec   = label_to_active_set[lbl].astype(int)
        g_min = int(vec[:n_g].sum())
        g_max = int(vec[n_g:2*n_g].sum())
        l_pos = int(vec[2*n_g:2*n_g+n_valid].sum())
        l_neg = int(vec[2*n_g+n_valid:].sum())
        total = g_min + g_max + l_pos + l_neg
        counts_per_label[lbl]    = total
        breakdown_per_label[lbl] = (g_min, g_max, l_pos, l_neg)

    all_counts = np.array(list(counts_per_label.values()))

    # Frequency-weighted statistics (if labels provided)
    if labels is not None:
        freq = Counter(labels.tolist())
        weighted_counts = []
        for lbl, cnt in freq.items():
            if lbl in counts_per_label:
                weighted_counts.extend([counts_per_label[lbl]] * cnt)
        weighted_counts = np.array(weighted_counts)
    else:
        weighted_counts = all_counts

    print(f"\n[Active Constraint Count Diagnostics]")
    print(f"  System dimensions:")
    print(f"    n_g (generators)        : {n_g}")
    print(f"    n_valid (constrained lines): {n_valid}")
    print(f"    Total constraint dims   : {n_total_constraints}")
    print(f"    Paper Eq.3 target count : n_g - 1 = {n_g - 1}  (for square KKT system)")
    print(f"")
    print(f"  Per unique active set (unweighted, {len(lbl_ids)} sets):")
    print(f"    Min active constraints  : {all_counts.min()}")
    print(f"    Max active constraints  : {all_counts.max()}")
    print(f"    Mean                    : {all_counts.mean():.2f}")
    print(f"    Median                  : {np.median(all_counts):.1f}")
    print(f"")
    print(f"  Sample-frequency weighted (reflects actual distribution):")
    print(f"    Mean active constraints : {weighted_counts.mean():.2f}")
    print(f"    Median                  : {np.median(weighted_counts):.1f}")
    print(f"    Std                     : {weighted_counts.std():.2f}")
    print(f"")

    # Distribution histogram (bucket by count)
    max_c = int(all_counts.max())
    buckets = {}
    for c in weighted_counts:
        b = int(c)
        buckets[b] = buckets.get(b, 0) + 1
    total_w = len(weighted_counts)
    print(f"  Distribution of active constraint count (sample-weighted):")
    for b in sorted(buckets.keys()):
        pct  = 100. * buckets[b] / total_w
        bar  = '#' * int(pct / 2)
        print(f"    count={b:3d}: {buckets[b]:6d} samples ({pct:5.1f}%)  {bar}")

    # Show top-5 most common active sets with breakdown
    if labels is not None:
        freq = Counter(labels.tolist())
        top5 = freq.most_common(5)
    else:
        top5 = [(lbl, 1) for lbl in lbl_ids[:5]]

    print(f"")
    print(f"  Top-5 active sets (by frequency) - constraint breakdown:")
    print(f"    {'Label':>6}  {'Freq':>6}  {'Total':>5}  "
          f"{'g_min':>5}  {'g_max':>5}  {'l_pos':>5}  {'l_neg':>5}")
    for lbl, cnt in top5:
        if lbl not in breakdown_per_label:
            continue
        gn, gx, lp, ln = breakdown_per_label[lbl]
        tot = gn + gx + lp + ln
        print(f"    {lbl:6d}  {cnt:6d}  {tot:5d}  "
              f"{gn:5d}  {gx:5d}  {lp:5d}  {ln:5d}")

    # Verdict
    mean_w = weighted_counts.mean()
    print(f"")
    print(f"  Verdict:")
    if mean_w < n_g * 0.5:
        print(f"    => Mean active count ({mean_w:.1f}) << n_g ({n_g})")
        print(f"       System is HIGHLY UNDERDETERMINED.")
        print(f"       Paper Eq.3 (matrix inversion) does NOT strictly apply.")
        print(f"       linprog is the correct approach: finds minimum-cost")
        print(f"       feasible solution satisfying active constraints.")
    elif abs(mean_w - (n_g - 1)) < 5:
        print(f"    => Mean active count ({mean_w:.1f}) ≈ n_g-1 ({n_g-1})")
        print(f"       System is approximately square.")
        print(f"       Paper Eq.3 (matrix inversion) applies well.")
    else:
        print(f"    => Mean active count ({mean_w:.1f}), n_g-1 = {n_g-1}")
        print(f"       Partially underdetermined. linprog handles this correctly.")


# =====================================================================
# Part 2 - Pg Recovery via Constrained LP (linprog)
# =====================================================================

def _build_active_index_sets(active_set_vec, n_g, n_valid):
    """
    Decode binary active-set vector into four generator/line index sets.

    active_set_vec layout (matches active_set_cols order):
        [0      : n_g)               -> active generator lower limits
        [n_g    : 2*n_g)             -> active generator upper limits
        [2*n_g  : 2*n_g + n_valid)   -> active line positive limits
        [2*n_g+n_valid : end)        -> active line negative limits
    """
    ag_min = np.where(active_set_vec[            : n_g           ] == 1)[0]
    ag_max = np.where(active_set_vec[n_g         : 2*n_g         ] == 1)[0]
    al_pos = np.where(active_set_vec[2*n_g       : 2*n_g+n_valid ] == 1)[0]
    al_neg = np.where(active_set_vec[2*n_g+n_valid:               ] == 1)[0]
    return ag_min, ag_max, al_pos, al_neg


def recover_pg_linprog(active_set_vec, x_pd_sample, params):
    """
    Recover Pg by solving a constrained LP given a predicted active set.

    Strategy (equivalent to Paper Eq.3):
    ─────────────────────────────────────
    Active generator constraints (ag_min, ag_max) fix pg values directly.
    The remaining FREE generators are solved via LP:

        min   c1[free] @ pg_free          (linear cost, matches paper Eq.1a)
        s.t.  sum(pg_free) = total_load - sum(pg_fixed)   (power balance)
              (line flow equalities from al_pos, al_neg)
              Pg_min[free] <= pg_free <= Pg_max[free]

    HONEST FAILURE (v2.1):
    ──────────────────────
    If the active-set equalities (power balance + line-flow equalities)
    make the LP infeasible, this function returns None. It does NOT drop
    the line equalities to force a solution.

    Rationale: a predicted active set that yields an infeasible LP is simply
    WRONG for this load. Silently relaxing it to a power-balance-only LP
    produces a solution that satisfies neither the true optimum nor the line
    limits, yet has an artificially LOW cost (fewer binding constraints =>
    cheaper). In the Top-K ensemble, that cheap-but-infeasible "decoy"
    outranks the correct Top-1 solution and corrupts the result. Returning
    None lets the caller discard this candidate.

    Note on quadratic cost (c2 terms):
        DC-OPF in the paper uses LINEAR cost (Eq.1a: c^T p).
        For cases with small c2 coefficients the LP solution is near-identical
        to the true QP solution (verified numerically).

    Returns
    -------
    pg : np.ndarray (n_g,) float32, or None if the LP defined by the active
         set is infeasible.
    """
    n_g    = params['general']['n_g']
    Pg_min = params['constraints']['Pg_min'].ravel().astype(np.float64)
    Pg_max = params['constraints']['Pg_max'].ravel().astype(np.float64)
    Pl_max = params['constraints']['Pl_max']
    PTDF   = params['constraints']['PTDF']      # (n_buses, n_branches)
    Map_g  = params['constraints']['Map_g'].astype(np.float64)  # (n_g, n_buses)
    c1     = params['constraints']['C_Pg'].ravel().astype(np.float64)

    valid_idx    = np.where(Pl_max < 1e10)[0]
    Pl_max_valid = Pl_max[valid_idx].astype(np.float64)
    PTDF_valid   = PTDF[:, valid_idx].astype(np.float64)  # (n_buses, n_valid)
    n_valid      = len(valid_idx)

    x_pd = x_pd_sample.astype(np.float64)

    ag_min, ag_max, al_pos, al_neg = _build_active_index_sets(
        active_set_vec, n_g, n_valid
    )

    # ── Step 1: fix active-constraint generators ──────────────────────
    pg_fixed = np.full(n_g, np.nan)
    for i in ag_min: pg_fixed[i] = Pg_min[i]
    for i in ag_max: pg_fixed[i] = Pg_max[i]
    # conflict: same generator at both limits → use midpoint (rare edge case)
    for i in ag_min:
        if not np.isnan(pg_fixed[i]) and i in ag_max:
            pg_fixed[i] = (Pg_min[i] + Pg_max[i]) / 2.0

    free_mask = np.isnan(pg_fixed)
    free_idx  = np.where(free_mask)[0]

    # ── All generators fixed → return directly ────────────────────────
    if len(free_idx) == 0:
        pg_fixed = np.nan_to_num(pg_fixed, nan=0.0)
        return pg_fixed.astype(np.float32)

    # ── Step 2: build LP for free generators ─────────────────────────
    total_load   = float(np.sum(x_pd))
    pg_fixed_sum = float(np.nansum(pg_fixed))
    residual     = total_load - pg_fixed_sum

    n_free      = len(free_idx)
    c1_free     = c1[free_idx]
    lb_free     = Pg_min[free_idx]
    ub_free     = Pg_max[free_idx]
    bounds_free = list(zip(lb_free.tolist(), ub_free.tolist()))

    # ── Step 3: equality constraints ─────────────────────────────────
    rows_A = []
    rows_b = []

    # power balance (always)
    rows_A.append(np.ones(n_free))
    rows_b.append(residual)

    # active line positive limits
    for k in al_pos:
        ptdf_col   = PTDF_valid[:, k]                    # (n_buses,)
        coeff_all  = Map_g @ ptdf_col                    # (n_g,)
        fixed_contrib = sum(
            float(pg_fixed[j]) * float(coeff_all[j])
            for j in range(n_g) if not free_mask[j]
        )
        coeff_free = coeff_all[free_idx]
        rhs = float(Pl_max_valid[k]) + float(ptdf_col @ x_pd) - fixed_contrib
        rows_A.append(coeff_free)
        rows_b.append(rhs)

    # active line negative limits
    for k in al_neg:
        ptdf_col   = PTDF_valid[:, k]
        coeff_all  = Map_g @ ptdf_col
        fixed_contrib = sum(
            float(pg_fixed[j]) * float(coeff_all[j])
            for j in range(n_g) if not free_mask[j]
        )
        coeff_free = coeff_all[free_idx]
        rhs = -float(Pl_max_valid[k]) + float(ptdf_col @ x_pd) - fixed_contrib
        rows_A.append(coeff_free)
        rows_b.append(rhs)

    A_eq = np.array(rows_A, dtype=np.float64)
    b_eq = np.array(rows_b, dtype=np.float64)

    # ── Step 4: solve LP ─────────────────────────────────────────────
    # v2.1: NO power-balance-only fallback. If the active set's equalities
    # are infeasible, this active set is wrong for this load -> return None
    # so the caller can discard it. Hiding the failure behind a degenerate
    # LP produces a cheap, infeasible decoy that poisons the Top-K ensemble.
    res = linprog(c1_free, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds_free, method='highs')

    if not res.success:
        return None

    # ── Step 5: assemble full pg vector ──────────────────────────────
    pg_out = pg_fixed.copy()
    pg_out[free_idx] = res.x
    return pg_out.astype(np.float32)


def _fallback_pg(x_pd_sample, params):
    """
    Last-resort dispatch: clipped uniform allocation.

    Used ONLY when no valid active-set solution is available:
      - Top-1: the single candidate failed recovery.
      - Top-K: ALL K candidates failed recovery / feasibility.

    This is intentionally a poor solution (uniform split, clipped to box
    bounds). It does NOT respect line limits, so it will WORSEN the reported
    MAE / violation metrics. That is by design: a failure should be visible
    in the metrics, not hidden behind a plausible-looking number.
    """
    n_g    = params['general']['n_g']
    Pg_min = params['constraints']['Pg_min'].ravel()
    Pg_max = params['constraints']['Pg_max'].ravel()
    total  = float(np.sum(x_pd_sample))
    return np.clip(np.full(n_g, total / n_g, dtype=np.float32), Pg_min, Pg_max)


def _single_cost(pg, params):
    c2 = params['constraints'].get('C_Pg_c2', np.zeros(len(pg))).ravel()
    c1 = params['constraints']['C_Pg'].ravel()
    return float(np.sum(c2 * pg**2 + c1 * pg))


def _branch_violation_single(pg, x_pd_sample, params):
    """
    Max branch (line) flow violation for a single dispatch, in p.u.

    Uses the shared dc_feasibility metric so the gate matches the final
    evaluation exactly. Returns a non-negative scalar (0 = no violation).
    """
    gu, gl, lv, _ = dc_feasibility(
        pg[None, :], x_pd_sample[None, :], params
    )
    return float(np.max(lv))


# =====================================================================
# Part 2b - Degeneracy Diagnostics
# =====================================================================
#
# These two functions answer the two questions that decide whether the
# poor case300 result is a RECOVERY bug or genuine LP DEGENERACY:
#
#   Q1 (diagnose_recovery_on_true_labels):
#       If we feed the TRUE optimal active set (not the NN prediction)
#       into recover_pg_linprog, does it still fail / violate limits?
#       - High failure on TRUE labels  => the extracted active sets are
#         themselves over-determined / inconsistent. This is the signature
#         of threshold-based extraction picking up degenerate (>n-1) tight
#         constraints. The problem is UPSTREAM of the model.
#       - Low failure on TRUE labels    => extraction is fine; the failures
#         in the real run come from the model predicting wrong labels.
#
#   Q2 (diagnose_label_merging_by_cost):
#       If we MERGE active-set labels that yield the same recovered Pg
#       (same OPF cost), how many distinct labels remain? If 1753 collapses
#       toward the paper's ~78, degeneracy is confirmed as the cause of the
#       class-count explosion.
#
# Neither function changes the experiment; they only print diagnostics.
# =====================================================================

def diagnose_recovery_on_true_labels(true_labels, label_to_as, x_eval, params,
                                     n_samples=300, branch_tol=1e-3,
                                     box_tol=1e-3):
    """
    Test recover_pg_linprog using the TRUE optimal active set of each sample,
    bypassing the neural network entirely. This isolates recovery/extraction
    problems from model-prediction problems.

    Prints:
      - fraction where label is unknown (not in label_to_as)
      - fraction where linprog returns None (active set infeasible)
      - fraction solved but violating generator box bounds
      - fraction solved but violating branch (line) limits
      - fraction "clean" (solved, no violation)

    A high None / violation rate on TRUE labels is strong evidence that the
    threshold-based active-set extraction is producing over-determined,
    degenerate constraint sets rather than the paper's exact (n-1) basis.
    """
    if true_labels is None:
        print("\n[Recovery Diagnostic] Skipped: no true labels "
              "(external test set).")
        return

    n_g    = params['general']['n_g']
    Pg_min = params['constraints']['Pg_min'].ravel()
    Pg_max = params['constraints']['Pg_max'].ravel()

    n = min(n_samples, len(true_labels))
    n_unknown = n_none = n_box = n_branch = n_clean = 0

    for i in range(n):
        lbl = true_labels[i]
        if lbl not in label_to_as:
            n_unknown += 1
            continue
        pg = recover_pg_linprog(label_to_as[lbl], x_eval[i], params)
        if pg is None:
            n_none += 1
            continue
        box_viol    = bool(np.any(pg < Pg_min - box_tol) or
                           np.any(pg > Pg_max + box_tol))
        branch_viol = _branch_violation_single(pg, x_eval[i], params) > branch_tol
        if box_viol:
            n_box += 1
        elif branch_viol:
            n_branch += 1
        else:
            n_clean += 1

    print(f"\n[Recovery Diagnostic on TRUE labels] ({n} samples)")
    print(f"  This bypasses the NN: it tests whether the EXTRACTED active")
    print(f"  sets can themselves be recovered into a feasible dispatch.")
    print(f"  Unknown label (not in train) : {n_unknown:4d} "
          f"({100.*n_unknown/n:5.1f}%)")
    print(f"  linprog infeasible (None)    : {n_none:4d} "
          f"({100.*n_none/n:5.1f}%)")
    print(f"  solved, box-bound violation  : {n_box:4d} "
          f"({100.*n_box/n:5.1f}%)")
    print(f"  solved, branch violation     : {n_branch:4d} "
          f"({100.*n_branch/n:5.1f}%)")
    print(f"  clean (feasible, no violation): {n_clean:4d} "
          f"({100.*n_clean/n:5.1f}%)")

    bad = n_none + n_box + n_branch
    print(f"")
    if bad > 0.20 * n:
        print(f"  => {100.*bad/n:.1f}% of TRUE active sets fail to recover")
        print(f"     cleanly. The extracted active sets are over-determined")
        print(f"     / degenerate (threshold method picks up >n-1 tight")
        print(f"     constraints). This is an UPSTREAM extraction issue,")
        print(f"     not a model-accuracy issue.")
    else:
        print(f"  => TRUE active sets recover cleanly "
              f"({100.*n_clean/n:.1f}%). Recovery/extraction is sound;")
        print(f"     run-time failures stem from the model predicting the")
        print(f"     wrong label, i.e. a classification-accuracy problem.")


def diagnose_label_merging_by_cost(labels_all, label_to_as, x_raw, params,
                                   sample_idx=None, n_probe=2000,
                                   cost_tol_rel=1e-4):
    """
    Estimate how much the class count shrinks if active sets that produce the
    SAME optimal dispatch (same OPF cost) are merged into one class.

    Method
    ------
    For a probe set of samples, recover Pg from each sample's TRUE active set,
    compute its cost, and group (label -> representative cost). Two labels are
    considered degenerate-equivalent if, for samples carrying them, the
    recovered cost matches within cost_tol_rel (relative). We approximate the
    merge by clustering labels whose median recovered cost coincides AND whose
    recovered dispatch is numerically close.

    This is a HEURISTIC estimate (full equivalence requires checking every
    sample), but it is enough to see whether 1753 collapses toward ~78.

    Prints the original vs estimated-merged class count.
    """
    if sample_idx is None:
        sample_idx = np.arange(min(n_probe, len(labels_all)))
    else:
        sample_idx = np.asarray(sample_idx)[:n_probe]

    # For each label seen in the probe, recover a representative Pg + cost
    # using the FIRST sample that carries that label.
    label_repr_pg   = {}   # lbl -> recovered pg vector
    label_repr_cost = {}   # lbl -> recovered cost

    for i in sample_idx:
        lbl = labels_all[i]
        if lbl in label_repr_pg:
            continue
        if lbl not in label_to_as:
            continue
        pg = recover_pg_linprog(label_to_as[lbl], x_raw[i], params)
        if pg is None:
            continue
        label_repr_pg[lbl]   = pg
        label_repr_cost[lbl] = _single_cost(pg, params)

    probed_labels = sorted(label_repr_pg.keys())
    n_probed = len(probed_labels)
    if n_probed == 0:
        print("\n[Label-Merging Diagnostic] No labels could be recovered; "
              "skipped.")
        return

    # Greedy clustering: a label joins an existing cluster if its
    # representative cost AND dispatch match the cluster representative.
    clusters = []   # list of dicts: {'cost':..., 'pg':..., 'members':[...]}
    cost_scale = max(1.0, np.median(np.abs(list(label_repr_cost.values()))))

    for lbl in probed_labels:
        pg   = label_repr_pg[lbl]
        cost = label_repr_cost[lbl]
        placed = False
        for cl in clusters:
            if abs(cost - cl['cost']) <= cost_tol_rel * cost_scale:
                # cost matches; confirm dispatch is also close
                if np.max(np.abs(pg - cl['pg'])) <= 1e-2:
                    cl['members'].append(lbl)
                    placed = True
                    break
        if not placed:
            clusters.append({'cost': cost, 'pg': pg, 'members': [lbl]})

    n_merged = len(clusters)
    # Report distribution of cluster sizes (how many labels collapse together)
    sizes = sorted((len(cl['members']) for cl in clusters), reverse=True)

    print(f"\n[Label-Merging Diagnostic] (probe = {len(sample_idx)} samples, "
          f"{n_probed} distinct labels recovered)")
    print(f"  Labels BEFORE merging (recovered subset): {n_probed}")
    print(f"  Labels AFTER merging by equal cost+pg   : {n_merged}")
    if n_probed > 0:
        print(f"  Collapse ratio                          : "
              f"{n_probed / max(1, n_merged):.1f}x")
    print(f"  Largest merged clusters (label counts)  : "
          f"{sizes[:10]}")
    print(f"")
    if n_merged < 0.5 * n_probed:
        print(f"  => Strong degeneracy: many distinct active-set labels map")
        print(f"     to the SAME optimal dispatch. The class-count explosion")
        print(f"     is an artifact of representing one optimum with multiple")
        print(f"     degenerate active sets. Merging recovers a class count")
        print(f"     much closer to the paper's reported value.")
    else:
        print(f"  => Limited merging: most labels give distinct dispatches.")
        print(f"     The high class count is NOT primarily a degeneracy")
        print(f"     artifact; the data genuinely spans many active sets.")


def recover_pg_top1_batch(top1_labels, label_to_as, x_pd_raw, params):
    """
    Top-1 batch: recover Pg for each sample using the argmax active set.

    If recovery fails (unknown label, or infeasible LP -> None), fall back
    to _fallback_pg. The fallback is a deliberately poor dispatch that
    worsens metrics, so a high fallback rate is visible both in the printed
    count and in the degraded MAE / violation numbers.
    """
    n     = len(top1_labels)
    n_g   = params['general']['n_g']
    pg_out = np.zeros((n, n_g), dtype=np.float32)
    n_fail = 0

    for i in range(n):
        lbl = top1_labels[i]
        if lbl not in label_to_as:
            pg_out[i] = _fallback_pg(x_pd_raw[i], params)
            n_fail += 1
            continue
        pg = recover_pg_linprog(label_to_as[lbl], x_pd_raw[i], params)
        if pg is None:
            pg_out[i] = _fallback_pg(x_pd_raw[i], params)
            n_fail += 1
        else:
            pg_out[i] = pg

    if n_fail > 0:
        print(f"  [Top-1] Recovery fallback (infeasible/unknown active set): "
              f"{n_fail}/{n} samples ({100.*n_fail/n:.1f}%).")
    return pg_out


def recover_pg_topk_batch(topk_labels, topk_probs,
                           label_to_as, x_pd_raw, params,
                           branch_tol=1e-3, box_tol=1e-3):
    """
    Top-K ensemble policy (Paper Section II-C-b), v2.1.

    For each sample, recover Pg for each of the K candidate active sets.
    A candidate is DISCARDED if any of the following holds:
        - label is unknown
        - recover_pg_linprog returns None (active set infeasible at this load)
        - generator box bounds violated   (> box_tol)
        - branch (line) flow violated      (> branch_tol)   <-- NEW in v2.1
    Among the surviving feasible candidates, the minimum-cost one is chosen.

    Why the branch check matters:
        Previously only generator box bounds were checked. A wrong active set
        could yield a dispatch that satisfies box bounds but violates line
        limits, while having a lower cost (fewer binding constraints). Such a
        "decoy" would beat the correct Top-1 candidate on cost and get picked,
        inflating branch violation and slack-generator MAE. With both the
        None-return from recover_pg_linprog AND this branch-violation gate,
        such decoys are rejected.

    Only if ALL K candidates are discarded does this fall back to _fallback_pg
    (a deliberately poor, metric-worsening dispatch), keeping failures visible.
    """
    n     = topk_labels.shape[0]
    K     = topk_labels.shape[1]
    n_g   = params['general']['n_g']
    Pg_min = params['constraints']['Pg_min'].ravel()
    Pg_max = params['constraints']['Pg_max'].ravel()
    pg_out = np.zeros((n, n_g), dtype=np.float32)

    n_all_fail = 0   # samples where every candidate was discarded

    for i in range(n):
        best_pg   = None
        best_cost = np.inf

        for k in range(K):
            lbl = topk_labels[i, k]
            if lbl not in label_to_as:
                continue
            pg = recover_pg_linprog(label_to_as[lbl], x_pd_raw[i], params)
            if pg is None:
                continue
            # gate 1: generator box bounds
            if np.any(pg < Pg_min - box_tol) or np.any(pg > Pg_max + box_tol):
                continue
            # gate 2 (NEW): branch / line flow violation
            if _branch_violation_single(pg, x_pd_raw[i], params) > branch_tol:
                continue
            cost = _single_cost(pg, params)
            if cost < best_cost:
                best_cost = cost
                best_pg   = pg

        if best_pg is not None:
            pg_out[i] = best_pg
        else:
            pg_out[i] = _fallback_pg(x_pd_raw[i], params)
            n_all_fail += 1

    if n_all_fail > 0:
        print(f"  [Top-K] All-candidate fallback (no feasible active set): "
              f"{n_all_fail}/{n} samples ({100.*n_all_fail/n:.1f}%).")
    return pg_out


# =====================================================================
# Part 3 - NN Classifier  (Paper Fig.2, Section III-A/B)
# =====================================================================

class ActiveSetClassifier(nn.Module):
    """
    Multi-layer fully connected NN:
      Input -> [Linear -> ReLU -> BatchNorm1d -> Dropout] x r -> Linear -> logits

    CrossEntropyLoss (includes implicit Softmax) is used during training.
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
        return self.net(x)      # returns raw logits


# =====================================================================
# Part 4 - Data Loading (v2.0 FIXED)
# =====================================================================

def load_data_for_active_set(dataset_path, params, column_names):
    """
    Load CSV dataset and return:
        x_raw    (n_samples, n_buses) float32  - unscaled bus loads
        y_pg_raw (n_samples, n_g)    float32  - unscaled generator output
        full_df  pd.DataFrame                  - needed for dual extraction

    v2.0 FIXES:
    -----------
    1. Load demand mapping: uses bus_id_to_idx dict instead of bus_id-1 indexing.
       This is critical for case300 where bus IDs like 7001, 9533 exist.
    2. Generator columns: explicitly built from g_bus order in params,
       instead of unsorted column scanning which may produce wrong column order.
    """
    df      = pd.read_csv(dataset_path)
    n_buses = params['general']['n_buses']
    n_samp  = len(df)

    # ------------------------------------------------------------------ #
    # FIX 1: Load demand data — use bus_id_to_idx mapping                #
    #                                                                     #
    # OLD (BROKEN for case300):                                           #
    #   if bid <= n_buses:                                                #
    #       x_raw[:, bid - 1] = df[load_cols[idx]].values                #
    # This assumes bus_id == position+1, which fails for virtual buses   #
    # like 7001, 9533 in case300.                                        #
    #                                                                     #
    # NEW (CORRECT):                                                      #
    #   Uses bus_id_to_idx dict from params (loaded from bus_ids.csv)    #
    #   to map any bus_id to its correct 0-based matrix position.        #
    # ------------------------------------------------------------------ #
    load_prefix  = column_names['load_prefix']
    load_cols    = [c for c in df.columns if c.startswith(load_prefix)]

    bus_id_to_idx = params['general']['bus_id_to_idx']
    x_raw = np.zeros((n_samp, n_buses), dtype='float32')

    for col_name in load_cols:
        bus_id = int(col_name[len(load_prefix):])
        if bus_id in bus_id_to_idx:
            x_raw[:, bus_id_to_idx[bus_id]] = df[col_name].values.astype('float32')
        else:
            print(f"[WARNING] load bus {bus_id} not in bus_id_to_idx, skipping")

    # ------------------------------------------------------------------ #
    # FIX 2: Generator columns — MUST match g_bus order from params      #
    #                                                                     #
    # OLD (BROKEN):                                                       #
    #   pg_cols = [c for c in df.columns if c.startswith(gen_prefix)]     #
    # This produces UNSORTED columns (DataFrame iteration order) and     #
    # may not match the order of Pg_min/Pg_max/Map_g in params.          #
    #                                                                     #
    # NEW (CORRECT):                                                      #
    #   Build pg_cols explicitly from g_bus (gen_id order), which is      #
    #   the same order as gen_limits.csv / Pg_min / Pg_max / Map_g.      #
    # ------------------------------------------------------------------ #
    g_bus      = params['general']['g_bus']
    gen_prefix = column_names['gen_prefix']
    pg_cols    = [f"{gen_prefix}{int(gen_id)}" for gen_id in g_bus]

    missing = [c for c in pg_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[load_data_for_active_set] Missing pg columns in CSV: {missing}\n"
            f"Available pg columns (first 10): "
            f"{sorted([c for c in df.columns if c.startswith(gen_prefix)])[:10]}"
        )

    y_pg_raw = df[pg_cols].values.astype('float32')

    expected_n_gen = params['general']['n_g']
    if len(pg_cols) != expected_n_gen:
        print(f"[WARNING] Number of generators in CSV ({len(pg_cols)}) "
              f"does not match parameters file ({expected_n_gen})!")

    return x_raw, y_pg_raw, df


# =====================================================================
# Part 5 - Main Experiment
# =====================================================================

def active_set_classification_experiment(
        case_name,
        params_path,
        dataset_path,
        n_train_use      = 10000,
        seed             = 42,
        n_epochs         = 20,
        patience         = 20,
        min_delta        = 1e-6,
        learning_rate    = 0.001,
        batch_size       = 32,
        hidden_layers    = None,
        dropout_rate     = 0.1,
        device           = 'cuda',
        split_mode       = DataSplitMode.RANDOM_SPLIT,
        test_data_path   = None,
        test_params_path = None,
        column_names     = None,
        n_test_samples   = 1000,
        active_threshold = 1e-4,
        top_k            = 3,
):
    if hidden_layers is None:
        hidden_layers = [256, 256, 128, 128, 64]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\nRunning Active Set Classification: {split_mode.value} - {case_name}")
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # Load system parameters
    # ------------------------------------------------------------------
    params     = load_parameters_from_csv(case_name, params_path, is_api=False)
    slack_info = identify_slack_bus_and_gens(params)
    params     = update_params_with_slack_info(params, slack_info)

    test_params = params
    if split_mode == DataSplitMode.API_TEST:
        if test_params_path is None:
            raise ValueError("API_TEST mode requires test_params_path")
        test_params = load_parameters_from_csv(case_name, test_params_path, is_api=True)
        test_params = update_params_with_slack_info(
            test_params, identify_slack_bus_and_gens(test_params)
        )

    if column_names is None:
        column_names = {
            'load_prefix':        'pd',
            'gen_prefix':         'pg',
            'lambda':             'lambda',
            'mu_g_min_prefix':    'mu_g_min_',
            'mu_g_max_prefix':    'mu_g_max_',
            'mu_line_pos_prefix': 'mu_line_max_',
            'mu_line_neg_prefix': 'mu_line_min_',
        }

    # ------------------------------------------------------------------
    # Load data + extract active-set labels
    # ------------------------------------------------------------------
    x_raw, y_pg_raw, full_df = load_data_for_active_set(
        dataset_path, params, column_names
    )

    print("[Step 1] Extracting active set labels from dual variables...")
    labels_all, label_to_as, as_to_label, as_cols = extract_active_sets(
        full_df, params, column_names, threshold=active_threshold
    )
    n_classes = len(label_to_as)
    print(f"  Total unique active sets in training data: {n_classes}")
    print_active_set_statistics(labels_all, "Full Dataset")

    # ------------------------------------------------------------------
    # Data split  (reuses existing split_data_by_mode)
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx, x_test_ext, y_test_ext = split_data_by_mode(
        x_data_raw     = x_raw,
        y_pg_raw       = y_pg_raw,
        mode           = split_mode,
        n_train_use    = n_train_use,
        seed           = seed,
        test_data_path = test_data_path,
        params         = params,
        column_names   = column_names,
        n_test_samples = n_test_samples,
    )

    print_active_set_statistics(labels_all[train_idx], "Train Split")

    # Diagnose active constraint count distribution (key for Paper Eq.3 validity)
    print_active_constraint_count_distribution(
        label_to_as, params, labels=labels_all
    )

    # Diagnostic Q2: how many labels collapse if we merge degenerate active
    # sets that yield the same optimal dispatch? Probes the train split so it
    # only touches labels the model could actually learn.
    diagnose_label_merging_by_cost(
        labels_all, label_to_as, x_raw, params,
        sample_idx=train_idx, n_probe=2000
    )

    unseen = set(labels_all[val_idx].tolist()) - set(labels_all[train_idx].tolist())
    if unseen:
        print(f"  [WARNING] Val set has {len(unseen)} unseen active set label(s).")

    # ------------------------------------------------------------------
    # Scalers and tensors
    # ------------------------------------------------------------------
    x_scaler = MinMaxScaler().fit(x_raw[train_idx])

    X_train = torch.tensor(
        x_scaler.transform(x_raw[train_idx]),
        dtype=torch.float32, device=device)
    Y_train = torch.tensor(labels_all[train_idx], dtype=torch.long, device=device)
    X_val   = torch.tensor(
        x_scaler.transform(x_raw[val_idx]),
        dtype=torch.float32, device=device)
    Y_val   = torch.tensor(labels_all[val_idx], dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = ActiveSetClassifier(
        input_size    = x_raw.shape[1],
        n_classes     = n_classes,
        hidden_layers = hidden_layers,
        dropout_rate  = dropout_rate,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] ActiveSetClassifier")
    print(f"  Input size:  {x_raw.shape[1]}")
    print(f"  Num classes: {n_classes}")
    print(f"  Hidden:      {hidden_layers}")
    print(f"  Parameters:  {n_params:,}")

    criterion = nn.CrossEntropyLoss()           # paper: cross-entropy (Eq.11)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # paper: Adam

    n_train   = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n[Training] Epochs={n_epochs}, BatchSize={batch_size}, LR={learning_rate}")
    print(f"  Early Stopping: patience={patience}, min_delta={min_delta}")
    t0 = time.perf_counter()

    # --- Early stopping state ---
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct    = 0
        perm       = torch.randperm(n_train, device=device)

        for b in range(n_batches):
            idx    = perm[b * batch_size : (b + 1) * batch_size]
            Xb, Yb = X_train[idx], Y_train[idx]
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, Yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)
            correct    += (logits.argmax(dim=-1) == Yb).sum().item()

        t_loss = epoch_loss / n_train
        t_acc  = 100. * correct / n_train

        model.eval()
        with torch.no_grad():
            vl     = model(X_val)
            v_loss = float(criterion(vl, Y_val))
            v_acc  = 100. * (vl.argmax(-1) == Y_val).sum().item() / len(Y_val)

        # --- Early stopping check ---
        if v_loss < best_val_loss - min_delta:
            best_val_loss = v_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d}/{n_epochs} - "
              f"train_loss: {t_loss:.6f}  train_acc: {t_acc:.2f}%  |  "
              f"val_loss: {v_loss:.6f}  val_acc: {v_acc:.2f}%"
              f"  - patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n[Early Stopping] No improvement for {patience} epochs. "
                  f"Best val_loss: {best_val_loss:.6f} (epoch {epoch - patience})")
            break

    # Restore best model weights before evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[Early Stopping] Best model weights restored (val_loss={best_val_loss:.6f})")

    train_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print("\n[Evaluation] Computing test metrics (linprog Pg recovery)...")

    model.eval()

    if split_mode in [DataSplitMode.GENERALIZATION, DataSplitMode.API_TEST]:
        x_eval      = x_test_ext
        y_eval_pg   = y_test_ext
        eval_params = test_params
        true_labels = None           # no dual labels for external test set
    else:
        x_eval      = x_raw[test_idx]
        y_eval_pg   = y_pg_raw[test_idx]
        eval_params = params
        true_labels = labels_all[test_idx]

    # NN forward: Top-1 and Top-K predictions
    X_eval_t = torch.tensor(
        x_scaler.transform(x_eval), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits_eval = model(X_eval_t)
        probs_eval  = torch.softmax(logits_eval, dim=-1)
        top1_pred   = probs_eval.argmax(dim=-1).cpu().numpy()
        K           = min(top_k, probs_eval.shape[-1])
        topk_p, topk_l = torch.topk(probs_eval, k=K, dim=-1)
        topk_labels = topk_l.cpu().numpy()
        topk_probs  = topk_p.cpu().numpy()

    # Classification accuracy (only available when true labels exist)
    top1_acc = topk_acc = None
    if true_labels is not None:
        top1_acc = 100. * np.mean(top1_pred == true_labels)
        in_topk  = np.array([
            true_labels[i] in topk_labels[i]
            for i in range(len(true_labels))
        ])
        topk_acc = 100. * np.mean(in_topk)

    # --- Diagnostic Q1: can the TRUE active sets even be recovered? ---
    # Bypasses the NN. High failure here => extraction/degeneracy problem,
    # not a model problem.
    diagnose_recovery_on_true_labels(
        true_labels, label_to_as, x_eval, eval_params, n_samples=300
    )

    # Pg recovery via linprog
    print(f"  Recovering Pg via linprog for {len(x_eval)} test samples...")
    t_r0 = time.perf_counter()
    pg_top1 = recover_pg_top1_batch(
        top1_pred, label_to_as, x_eval, eval_params)
    t_r1 = time.perf_counter()
    pg_topk = recover_pg_topk_batch(
        topk_labels, topk_probs, label_to_as, x_eval, eval_params)
    t_r2 = time.perf_counter()
    print(f"  Top-1 recovery: {t_r1-t_r0:.1f}s  |  "
          f"Top-{K} recovery: {t_r2-t_r1:.1f}s")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    cost_coeffs = {
        'C2': eval_params['constraints'].get(
                  'C_Pg_c2', np.zeros(eval_params['general']['n_g'])),
        'C1': eval_params['constraints']['C_Pg'],
        'C0': eval_params['constraints'].get(
                  'C_Pg_c0', np.zeros(eval_params['general']['n_g'])),
    }

    def compute_all_metrics(pg_pred, x_pd, y_true, ep):
        ns_idx = ep['general']['non_slack_gen_indices']
        mae_d  = compute_detailed_mae(y_true, pg_pred[:, ns_idx], pg_pred, ep)
        gu, gl, lv, _ = dc_feasibility(pg_pred, x_pd, ep)
        vd    = compute_detailed_pg_violations_pu(gu, gl, ep)
        bv    = compute_branch_violation_pu(lv, ep['constraints']['Pl_max'])
        ct    = compute_cost(y_true,  cost_coeffs)
        cp    = compute_cost(pg_pred, cost_coeffs)
        cg    = compute_cost_gap_percentage(ct, cp)
        return {
            'mae_pg_non_slack':  mae_d['mae_non_slack'],
            'mae_pg_slack':      mae_d['mae_slack'],
            'viol_pg_non_slack': vd['viol_non_slack'],
            'viol_pg_slack':     vd['viol_slack'],
            'viol_branch':       bv,
            'cost_gap_percent':  cg,
        }

    m1 = compute_all_metrics(pg_top1, x_eval, y_eval_pg, eval_params)
    mK = compute_all_metrics(pg_topk, x_eval, y_eval_pg, eval_params)

    # ------------------------------------------------------------------
    # Inference latency (NN forward pass only; paper measures NN speed)
    # ------------------------------------------------------------------
    sample_t = torch.tensor(
        x_scaler.transform(x_eval[:1]),
        dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(10): _ = model(sample_t)
        if device.type == 'cuda': torch.cuda.synchronize()
    times_lat = []
    with torch.no_grad():
        for _ in range(100):
            ts = time.perf_counter()
            _  = model(sample_t)
            if device.type == 'cuda': torch.cuda.synchronize()
            times_lat.append(time.perf_counter() - ts)
    latency_ms = np.mean(times_lat) * 1000

    # ------------------------------------------------------------------
    # Print results  (same section/field format as dnn_dcopf_main.py)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Active Set Classification - Test Set Results")
    print("=" * 70)

    print(f"\nActive Set Statistics:")
    print(f"  Unique active sets (training data): {n_classes}")
    if top1_acc is not None:
        print(f"  Top-1 Classification Accuracy:      {top1_acc:.2f}%")
        print(f"  Top-{K} Classification Accuracy:     {topk_acc:.2f}%")
    else:
        print(f"  Classification Accuracy: N/A (external test set)")

    print(f"\n--- Top-1 Prediction ---")
    print(f"\nNon-Slack Generators:")
    print(f"  MAE:        {m1['mae_pg_non_slack']:.4f}%")
    print(f"  Violation:  {m1['viol_pg_non_slack']:.4f} p.u.")
    print(f"\nSlack-Only Generators:")
    print(f"  MAE:        {m1['mae_pg_slack']:.4f}%")
    print(f"  Violation:  {m1['viol_pg_slack']:.4f} p.u.")
    print(f"\nBranch:")
    print(f"  Violation:  {m1['viol_branch']:.4f} p.u.")
    print(f"\nCost Gap:     {m1['cost_gap_percent']:.4f}%")

    print(f"\n--- Top-{K} Ensemble Prediction ---")
    print(f"\nNon-Slack Generators:")
    print(f"  MAE:        {mK['mae_pg_non_slack']:.4f}%")
    print(f"  Violation:  {mK['viol_pg_non_slack']:.4f} p.u.")
    print(f"\nSlack-Only Generators:")
    print(f"  MAE:        {mK['mae_pg_slack']:.4f}%")
    print(f"  Violation:  {mK['viol_pg_slack']:.4f} p.u.")
    print(f"\nBranch:")
    print(f"  Violation:  {mK['viol_branch']:.4f} p.u.")
    print(f"\nCost Gap:     {mK['cost_gap_percent']:.4f}%")

    print(f"\nTraining Time:   {train_time:.2f} s")
    print(f"Inference Time:  {latency_ms:.4f} ms")
    print("\n" + "=" * 70 + "\n")

    return {
        'top1': m1,  'topk': mK,
        'top1_acc': top1_acc, 'topk_acc': topk_acc,
        'n_classes': n_classes,
        'train_time': train_time, 'latency_ms': latency_ms,
    }


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":

    from dcopf_config import PathConfig

    # ---------------------------------------------------------------
    # 1. Case
    # ---------------------------------------------------------------
    CASE_NAME       = 'pglib_opf_case300_ieee'
    CASE_SHORT_NAME = 'case300'

    # ---------------------------------------------------------------
    # 2. Split mode
    # ---------------------------------------------------------------
    SPLIT_MODE = DataSplitMode.RANDOM_SPLIT

    # ---------------------------------------------------------------
    # 3. Sample counts
    #    10000 total  ->  10:1:1  -> train~8334, val~833, test~833
    # ---------------------------------------------------------------
    N_TRAIN_USE    = 6000
    N_TEST_SAMPLES = 1000       # for GENERALIZATION / API_TEST only

    # ---------------------------------------------------------------
    # 4. Hyperparameters  (paper defaults)
    #    epochs=20, batch=32, hidden=[256,256,128,128,64]
    # ---------------------------------------------------------------
    N_EPOCHS      = 1000
    PATIENCE      = 20
    LEARNING_RATE = 0.001
    BATCH_SIZE    = 128
    HIDDEN_LAYERS = [256,128]
    DROPOUT_RATE  = 0.1
    SEED          = 42
    TOP_K         = 3
    ACTIVE_THR    = 1e-4

    # ---------------------------------------------------------------
    # 5. Paths (via PathConfig)
    # ---------------------------------------------------------------
    TRAIN_VARIANCE = "v=0.12"
    TEST_VARIANCE  = "v=0.25"

    COLUMN_NAMES = {
        'load_prefix':        'pd',
        'gen_prefix':         'pg',
        'lambda':             'lambda',
        'mu_g_min_prefix':    'mu_g_min_',
        'mu_g_max_prefix':    'mu_g_max_',
        'mu_line_pos_prefix': 'mu_line_max_',
        'mu_line_neg_prefix': 'mu_line_min_',
    }

    params_path     = PathConfig.get_constraints_path(CASE_SHORT_NAME)
    train_data_path = PathConfig.get_dataset_path(
        CASE_NAME, CASE_SHORT_NAME, variance=TRAIN_VARIANCE
    )

    if SPLIT_MODE == DataSplitMode.GENERALIZATION:
        test_data_path   = PathConfig.get_dataset_path(
            CASE_NAME, CASE_SHORT_NAME, variance=TEST_VARIANCE
        )
        test_params_path = None
    elif SPLIT_MODE == DataSplitMode.API_TEST:
        test_data_path   = PathConfig.get_dataset_path(
            CASE_NAME, CASE_SHORT_NAME, is_api=True
        )
        test_params_path = PathConfig.get_constraints_path(
            CASE_SHORT_NAME, is_api=True
        )
    else:
        test_data_path   = None
        test_params_path = None

    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    results = active_set_classification_experiment(
        case_name        = CASE_NAME,
        params_path      = params_path,
        dataset_path     = train_data_path,
        n_train_use      = N_TRAIN_USE,
        seed             = SEED,
        n_epochs         = N_EPOCHS,
        patience         = PATIENCE,
        learning_rate    = LEARNING_RATE,
        batch_size       = BATCH_SIZE,
        hidden_layers    = HIDDEN_LAYERS,
        dropout_rate     = DROPOUT_RATE,
        device           = device_name,
        split_mode       = SPLIT_MODE,
        test_data_path   = test_data_path,
        test_params_path = test_params_path,
        column_names     = COLUMN_NAMES,
        n_test_samples   = N_TEST_SAMPLES,
        active_threshold = ACTIVE_THR,
        top_k            = TOP_K,
    )