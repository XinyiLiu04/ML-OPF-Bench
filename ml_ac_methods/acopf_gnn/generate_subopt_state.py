# -*- coding: utf-8 -*-
"""
generate_subopt_state.py

Generate sub-optimal state X = [vm, va, p, q] for each sample in an existing
ACOPF dataset, following the Owerko et al. (ICASSP 2020) methodology:

    For each sample (pd, qd) in the ACOPF dataset:
        1. Solve DCOPF  →  pg_dcopf  (sub-optimal generator dispatch)
        2. Run AC Power Flow with (pd, qd, pg_dcopf)  →  (vm, va, p_inject, q_inject)
        3. Save the sub-optimal state X alongside the existing ACOPF optimal labels

The DCOPF is solved in Python via scipy (QP with PTDF constraints).
The AC Power Flow uses PyPower's runpf.

Usage:
    1. Set configuration at the bottom of this file
    2. python generate_subopt_state.py

Output:
    {case_name}_subopt_vm.csv   — sub-optimal voltage magnitudes (all buses)
    {case_name}_subopt_va.csv   — sub-optimal voltage angles in rad (all buses)
    {case_name}_subopt_pinj.csv — sub-optimal active power injection (all buses)
    {case_name}_subopt_qinj.csv — sub-optimal reactive power injection (all buses)
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog, minimize
from pypower.runpf import runpf
from pypower.ppoption import ppoption


# =====================================================================
# 1. Load ACOPF Constraints (PyPower case data) — reuse existing logic
# =====================================================================
def load_case_from_csv(case_name, constraints_path):
    """Load PyPower ppc dict from ACOPF Constraints CSV files."""
    base_path = Path(constraints_path)

    base_mva_df = pd.read_csv(base_path / f"{case_name}_base_mva.csv")
    bus_df = pd.read_csv(base_path / f"{case_name}_bus_data.csv")
    gen_df = pd.read_csv(base_path / f"{case_name}_gen_data.csv")
    branch_df = pd.read_csv(base_path / f"{case_name}_branch_data.csv")

    baseMVA = base_mva_df['value'].iloc[0]

    bus = np.zeros((len(bus_df), 13))
    bus[:, 0] = bus_df['bus_id'].values
    bus[:, 1] = bus_df['type'].values
    bus[:, 2] = bus_df['pd_pu'].values
    bus[:, 3] = bus_df['qd_pu'].values
    bus[:, 6] = 1
    bus[:, 7] = bus_df['vm_pu'].values
    bus[:, 8] = bus_df['va_deg'].values
    bus[:, 9] = bus_df['base_kv'].values
    bus[:, 10] = 1
    bus[:, 11] = bus_df['vmax_pu'].values
    bus[:, 12] = bus_df['vmin_pu'].values

    gen = np.zeros((len(gen_df), 21))
    gen[:, 0] = gen_df['bus_id'].values
    gen[:, 3] = gen_df['qg_max_pu'].values
    gen[:, 4] = gen_df['qg_min_pu'].values
    gen[:, 5] = gen_df['vg_pu'].values
    gen[:, 6] = baseMVA
    gen[:, 7] = 1  # status ON
    gen[:, 8] = gen_df['pg_max_pu'].values
    gen[:, 9] = gen_df['pg_min_pu'].values

    branch = np.zeros((len(branch_df), 13))
    branch[:, 0] = branch_df['f_bus'].values
    branch[:, 1] = branch_df['t_bus'].values
    branch[:, 2] = branch_df['r_pu'].values
    branch[:, 3] = branch_df['x_pu'].values
    branch[:, 4] = branch_df['b_pu'].values
    branch[:, 5] = branch_df['rate_a_pu'].values
    branch[:, 6] = branch[:, 5]
    branch[:, 7] = branch[:, 5]
    branch[:, 8] = branch_df['tap_ratio'].values
    branch[:, 9] = branch_df['shift_deg'].values
    branch[:, 10] = 1
    branch[:, 11] = -360
    branch[:, 12] = 360
    rate_a = branch_df['rate_a_pu'].values
    branch[:, 5:8][np.isnan(rate_a) | np.isinf(rate_a), :] = 9900.0

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

    # Convert p.u. → MVA for PyPower
    ppc['bus'][:, 2] *= baseMVA   # Pd
    ppc['bus'][:, 3] *= baseMVA   # Qd
    ppc['gen'][:, 3] *= baseMVA   # Qg_max
    ppc['gen'][:, 4] *= baseMVA   # Qg_min
    ppc['gen'][:, 8] *= baseMVA   # Pg_max
    ppc['gen'][:, 9] *= baseMVA   # Pg_min
    mask = (ppc['branch'][:, 5] != 0) & (ppc['branch'][:, 5] < 9000)
    ppc['branch'][mask, 5:8] *= baseMVA  # rate_a

    return ppc


# =====================================================================
# 2. Load DCOPF Constraints (PTDF-based)
# =====================================================================
def load_dcopf_constraints(case_name, dcopf_constraints_path):
    """
    Load DCOPF constraint data exported by dcopf_constraints.jl.

    Returns dict with keys:
        ptdf, bus_gen_map, pg_min, pg_max, f_max,
        cost_c2, cost_c1, cost_c0, n_buses, n_gen, n_branch,
        gen_ids, bus_ids
    """
    base = Path(dcopf_constraints_path)

    gen_limits = pd.read_csv(base / f"{case_name}_gen_limits.csv")
    gen_costs = pd.read_csv(base / f"{case_name}_gen_costs.csv")
    branch_limits = pd.read_csv(base / f"{case_name}_branch_limits.csv")
    ptdf = pd.read_csv(base / f"{case_name}_ptdf_matrix.csv").values.astype('float64')
    bus_gen_map = pd.read_csv(base / f"{case_name}_bus_gen_map.csv").values.astype('float64')
    base_mva_df = pd.read_csv(base / f"{case_name}_base_mva.csv")

    n_gen = len(gen_limits)
    n_buses = bus_gen_map.shape[0]
    n_branch = ptdf.shape[0]

    return {
        'ptdf': ptdf,                    # [n_branch, n_buses]
        'bus_gen_map': bus_gen_map,       # [n_buses, n_gen]
        'pg_min': gen_limits['pgmin'].values,
        'pg_max': gen_limits['pgmax'].values,
        'f_max': branch_limits['rate_a'].values,
        'cost_c2': gen_costs['cost_c2'].values,
        'cost_c1': gen_costs['cost_c1'].values,
        'cost_c0': gen_costs['cost_c0'].values,
        'gen_ids': gen_limits['gen_id'].values,
        'n_buses': n_buses,
        'n_gen': n_gen,
        'n_branch': n_branch,
        'baseMVA': base_mva_df['value'].iloc[0],
    }


# =====================================================================
# 3. Solve DCOPF (Python, scipy)
# =====================================================================
def solve_dcopf(pd_vector, dcopf_params):
    """
    Solve DCOPF for a single load sample using scipy.

    DCOPF formulation:
        min  sum( c2_i * pg_i^2 + c1_i * pg_i + c0_i )
        s.t. sum(pg) = sum(pd)                              (power balance)
             pg_min <= pg <= pg_max                          (gen limits)
             -f_max <= PTDF @ (bus_gen_map @ pg - pd) <= f_max  (line limits)

    Args:
        pd_vector: active load at each bus [n_buses], in p.u. (MW/baseMVA)
        dcopf_params: dict from load_dcopf_constraints()

    Returns:
        pg_dcopf: generator outputs [n_gen] in p.u., or None if infeasible
    """
    ptdf = dcopf_params['ptdf']
    bgm = dcopf_params['bus_gen_map']
    pg_min = dcopf_params['pg_min']
    pg_max = dcopf_params['pg_max']
    f_max = dcopf_params['f_max']
    c2 = dcopf_params['cost_c2']
    c1 = dcopf_params['cost_c1']
    n_gen = dcopf_params['n_gen']

    # Filter branches with finite limits
    valid = f_max < 1e10
    ptdf_v = ptdf[valid, :]
    f_max_v = f_max[valid]
    n_valid = ptdf_v.shape[0]

    total_load = pd_vector.sum()

    # Precompute PTDF @ pd (constant term)
    ptdf_pd = ptdf_v @ pd_vector  # [n_valid]

    # PTDF @ bus_gen_map gives mapping from pg to branch flows
    ptdf_bgm = ptdf_v @ bgm  # [n_valid, n_gen]

    if np.all(c2 == 0):
        # ── Linear cost: use linprog ──────────────────────────────────
        # Inequality: ptdf_bgm @ pg <= f_max + ptdf_pd  (pos direction)
        #            -ptdf_bgm @ pg <= f_max - ptdf_pd  (neg direction)
        A_ub = np.vstack([ptdf_bgm, -ptdf_bgm])
        b_ub = np.concatenate([f_max_v + ptdf_pd, f_max_v - ptdf_pd])

        # Equality: sum(pg) = total_load
        A_eq = np.ones((1, n_gen))
        b_eq = np.array([total_load])

        bounds = [(pg_min[i], pg_max[i]) for i in range(n_gen)]

        res = linprog(c1, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return None
    else:
        # ── Quadratic cost: use scipy.optimize.minimize (SLSQP) ──────
        def objective(pg):
            return np.sum(c2 * pg ** 2 + c1 * pg + dcopf_params['cost_c0'])

        def jac(pg):
            return 2 * c2 * pg + c1

        constraints = []
        # Power balance
        constraints.append({
            'type': 'eq',
            'fun': lambda pg: np.sum(pg) - total_load,
            'jac': lambda pg: np.ones(n_gen)
        })
        # Line limits (positive direction): ptdf_bgm @ pg - ptdf_pd <= f_max_v
        if n_valid > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda pg: f_max_v - (ptdf_bgm @ pg - ptdf_pd),
                'jac': lambda pg: -ptdf_bgm
            })
            # Line limits (negative direction): -(ptdf_bgm @ pg - ptdf_pd) <= f_max_v
            constraints.append({
                'type': 'ineq',
                'fun': lambda pg: f_max_v + (ptdf_bgm @ pg - ptdf_pd),
                'jac': lambda pg: ptdf_bgm
            })

        bounds = [(pg_min[i], pg_max[i]) for i in range(n_gen)]

        # Initial guess: proportional to max capacity
        pg0 = pg_max * (total_load / pg_max.sum()) if pg_max.sum() > 0 else np.full(n_gen, total_load / n_gen)
        pg0 = np.clip(pg0, pg_min, pg_max)

        res = minimize(objective, pg0, jac=jac, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            return res.x
        else:
            return None


# =====================================================================
# 4. Run AC Power Flow with DCOPF dispatch
# =====================================================================
def run_powerflow_with_dcopf(pd_pu, qd_pu, pg_dcopf, ppc_template,
                             load_bus_ids, bus_id_to_idx, gen_bus_ids,
                             baseMVA, ppopt):
    """
    Run AC power flow using DCOPF generator dispatch as initial condition.

    Sets:
        - Bus loads to (pd_pu, qd_pu)
        - Generator Pg to pg_dcopf
        - Generator Vm to reference values (from ppc template)

    Returns:
        success : bool
        vm      : voltage magnitude [n_buses] in p.u.
        va      : voltage angle [n_buses] in radians
        p_inj   : net active power injection [n_buses] in p.u.
        q_inj   : net reactive power injection [n_buses] in p.u.
    """
    mpc = {
        'version': ppc_template['version'],
        'baseMVA': ppc_template['baseMVA'],
        'bus': ppc_template['bus'].copy(),
        'gen': ppc_template['gen'].copy(),
        'branch': ppc_template['branch'].copy(),
        'gencost': ppc_template['gencost'],
    }

    # Set loads (pd, qd are in p.u., PyPower expects MVA)
    for i, bus_id in enumerate(load_bus_ids):
        idx = bus_id_to_idx.get(int(bus_id))
        if idx is not None:
            mpc['bus'][idx, 2] = pd_pu[i] * baseMVA
            mpc['bus'][idx, 3] = qd_pu[i] * baseMVA

    # Set generator active power from DCOPF (all generators, in p.u. → MVA)
    n_gen = mpc['gen'].shape[0]
    for i in range(n_gen):
        mpc['gen'][i, 1] = pg_dcopf[i] * baseMVA

    # Run power flow
    result, success = runpf(mpc, ppopt)

    n_buses = mpc['bus'].shape[0]

    if success:
        vm = result['bus'][:, 7]                       # voltage magnitude p.u.
        va = result['bus'][:, 8] * np.pi / 180.0       # voltage angle → rad

        # Net power injection at each bus (Pg - Pd, Qg - Qd) in p.u.
        p_inj = np.zeros(n_buses)
        q_inj = np.zeros(n_buses)

        # Load contribution (negative injection)
        p_inj -= result['bus'][:, 2] / baseMVA
        q_inj -= result['bus'][:, 3] / baseMVA

        # Generator contribution (positive injection)
        for g in range(n_gen):
            g_bus_id = int(mpc['gen'][g, 0])
            g_bus_idx = bus_id_to_idx[g_bus_id]
            p_inj[g_bus_idx] += result['gen'][g, 1] / baseMVA
            q_inj[g_bus_idx] += result['gen'][g, 2] / baseMVA

        return True, vm, va, p_inj, q_inj
    else:
        return False, None, None, None, None


# =====================================================================
# 5. Load existing ACOPF dataset (pd, qd)
# =====================================================================
def load_acopf_loads(data_dir, case_name):
    """
    Load pd and qd from existing ACOPF dataset CSVs.

    Returns:
        pd_raw : [n_samples, n_loads] in p.u.
        qd_raw : [n_samples, n_loads] in p.u.
        load_bus_ids : list of int, bus IDs with loads
    """
    pd_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_pd.csv"))
    qd_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_qd.csv"))

    def extract_id(col):
        match = re.search(r'(\d+)$', col)
        return int(match.group(1)) if match else -1

    pd_cols = sorted([c for c in pd_df.columns if c.startswith('pd')], key=extract_id)
    qd_cols = sorted([c for c in qd_df.columns if c.startswith('qd')], key=extract_id)
    load_bus_ids = [extract_id(c) for c in pd_cols]

    return (pd_df[pd_cols].values.astype('float64'),
            qd_df[qd_cols].values.astype('float64'),
            load_bus_ids)


# =====================================================================
# 6. Build bus mapping for ACOPF Constraints
# =====================================================================
def build_bus_mapping(ppc):
    """Build bus_id → row_index mapping from PyPower case."""
    bus_ids = ppc['bus'][:, 0].astype(int)
    return {int(bid): i for i, bid in enumerate(bus_ids)}, bus_ids


# =====================================================================
# 7. Build full-bus load vector for DCOPF
# =====================================================================
def build_full_bus_load(pd_sample, load_bus_ids, dcopf_bus_lookup, n_buses_dcopf):
    """
    Map per-load-bus pd values to full bus vector for DCOPF.

    dcopf_bus_lookup maps bus_id → 0-indexed position used by DCOPF
    (from dcopf_constraints.jl's bus_lookup).
    """
    pd_full = np.zeros(n_buses_dcopf)
    for i, bus_id in enumerate(load_bus_ids):
        if bus_id in dcopf_bus_lookup:
            pd_full[dcopf_bus_lookup[bus_id]] = pd_sample[i]
    return pd_full


# =====================================================================
# 8. Main pipeline
# =====================================================================
def generate_suboptimal_states(
        case_name,
        acopf_data_dir,
        acopf_constraints_path,
        dcopf_constraints_path,
        output_dir=None,
):
    """
    Main function: for each sample in ACOPF dataset, solve DCOPF then
    run AC power flow to produce sub-optimal state X = [vm, va, p, q].
    """
    if output_dir is None:
        output_dir = acopf_data_dir

    print("=" * 70)
    print("Generate Sub-optimal State X (Owerko et al. methodology)")
    print("=" * 70)

    # ── 1. Load ACOPF loads ───────────────────────────────────────────
    print("\n[1] Loading ACOPF dataset loads...")
    pd_raw, qd_raw, load_bus_ids = load_acopf_loads(acopf_data_dir, case_name)
    n_samples, n_loads = pd_raw.shape
    print(f"    Samples: {n_samples}, Load buses: {n_loads}")

    # ── 2. Load PyPower case (ACOPF constraints) ──────────────────────
    print("\n[2] Loading PyPower case data...")
    ppc = load_case_from_csv(case_name, acopf_constraints_path)
    bus_id_to_idx, bus_ids = build_bus_mapping(ppc)
    n_buses = len(bus_ids)
    baseMVA = ppc['baseMVA']
    gen_bus_ids = ppc['gen'][:, 0].astype(int)
    n_gen = len(gen_bus_ids)
    print(f"    Buses: {n_buses}, Generators: {n_gen}, BaseMVA: {baseMVA}")

    # ── 3. Load DCOPF constraints ─────────────────────────────────────
    print("\n[3] Loading DCOPF constraints...")
    dcopf_params = load_dcopf_constraints(case_name, dcopf_constraints_path)
    n_buses_dcopf = dcopf_params['n_buses']
    print(f"    DCOPF buses: {n_buses_dcopf}, generators: {dcopf_params['n_gen']}")

    # Build DCOPF bus lookup: bus_id → position (matching Julia's bus_lookup)
    # The DCOPF constraints use sorted bus IDs, 0-indexed
    dcopf_bus_ids = sorted(bus_ids)
    dcopf_bus_lookup = {int(bid): i for i, bid in enumerate(dcopf_bus_ids)}

    # Map DCOPF gen order → PyPower gen order
    dcopf_gen_ids = dcopf_params['gen_ids']  # sorted gen IDs from Julia
    # PyPower gen order: by row in ppc['gen'], gen bus IDs
    # We need a mapping: dcopf_gen_idx → pypower_gen_idx
    # Both are sorted by gen_id, so they should align if gen_ids match
    pypower_gen_bus = ppc['gen'][:, 0].astype(int)

    # ── 4. Setup PyPower ──────────────────────────────────────────────
    ppopt = ppoption()
    ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)

    # ── 5. Process each sample ────────────────────────────────────────
    print(f"\n[4] Processing {n_samples} samples (DCOPF + Power Flow)...")

    subopt_vm = np.zeros((n_samples, n_buses))
    subopt_va = np.zeros((n_samples, n_buses))
    subopt_pinj = np.zeros((n_samples, n_buses))
    subopt_qinj = np.zeros((n_samples, n_buses))
    success_flags = np.zeros(n_samples, dtype=bool)

    dcopf_fail = 0
    pf_fail = 0

    # ── Per-sample timing arrays (for benchmark reporting) ────────────
    from time import time, perf_counter
    dcopf_times = np.zeros(n_samples)    # DCOPF solve time per sample (s)
    pf_times    = np.zeros(n_samples)    # AC Power Flow time per sample (s)

    t0 = time()

    for i in range(n_samples):
        # Progress
        if (i + 1) % 1000 == 0 or i == 0:
            elapsed = time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    Sample {i + 1}/{n_samples}  "
                  f"({rate:.0f} samples/s, "
                  f"DCOPF fail: {dcopf_fail}, PF fail: {pf_fail})")

        # Build full-bus load vector for DCOPF
        pd_full = build_full_bus_load(
            pd_raw[i], load_bus_ids, dcopf_bus_lookup, n_buses_dcopf)

        # Solve DCOPF (timed)
        t_dcopf_start = perf_counter()
        pg_dcopf = solve_dcopf(pd_full, dcopf_params)
        t_dcopf_end = perf_counter()
        dcopf_times[i] = t_dcopf_end - t_dcopf_start

        if pg_dcopf is None:
            dcopf_fail += 1
            # Fallback: proportional dispatch
            total_load = pd_full.sum()
            pg_max = dcopf_params['pg_max']
            pg_dcopf = pg_max * (total_load / pg_max.sum())
            pg_dcopf = np.clip(pg_dcopf, dcopf_params['pg_min'], pg_max)

        # Map DCOPF gen output to PyPower gen order
        pg_pypower = np.zeros(n_gen)
        dcopf_gen_id_to_pg = {int(gid): pg_dcopf[j]
                              for j, gid in enumerate(dcopf_gen_ids)}
        if n_gen == dcopf_params['n_gen']:
            pg_pypower = pg_dcopf.copy()
        else:
            print(f"  ⚠️ Gen count mismatch: PyPower={n_gen}, DCOPF={dcopf_params['n_gen']}")
            pg_pypower[:dcopf_params['n_gen']] = pg_dcopf

        # Run AC power flow (timed)
        t_pf_start = perf_counter()
        ok, vm, va, p_inj, q_inj = run_powerflow_with_dcopf(
            pd_raw[i], qd_raw[i], pg_pypower,
            ppc, load_bus_ids, bus_id_to_idx, gen_bus_ids,
            baseMVA, ppopt)
        t_pf_end = perf_counter()
        pf_times[i] = t_pf_end - t_pf_start

        if ok:
            subopt_vm[i] = vm
            subopt_va[i] = va
            subopt_pinj[i] = p_inj
            subopt_qinj[i] = q_inj
            success_flags[i] = True
        else:
            pf_fail += 1

    elapsed = time() - t0
    n_ok = success_flags.sum()
    print(f"\n[5] Completed in {elapsed:.1f}s")
    print(f"    Converged: {n_ok}/{n_samples} "
          f"({n_ok / n_samples * 100:.1f}%)")
    print(f"    DCOPF failures: {dcopf_fail}, PF failures: {pf_fail}")

    # ── Timing summary (for benchmark) ────────────────────────────────
    mean_dcopf_ms  = np.mean(dcopf_times) * 1000
    mean_pf_ms     = np.mean(pf_times) * 1000
    mean_total_ms  = mean_dcopf_ms + mean_pf_ms
    std_dcopf_ms   = np.std(dcopf_times) * 1000
    std_pf_ms      = np.std(pf_times) * 1000

    print(f"\n{'=' * 70}")
    print(f"Sub-optimal State Generation — Timing Summary (Benchmark)")
    print(f"{'=' * 70}")
    print(f"  Total wall time       : {elapsed:.2f} s  ({n_samples} samples)")
    print(f"  Throughput             : {n_samples / elapsed:.1f} samples/s")
    print(f"")
    print(f"  --- Per-sample Timing (ms/sample) ---")
    print(f"  DCOPF solve            : {mean_dcopf_ms:.4f} ± {std_dcopf_ms:.4f} ms")
    print(f"  AC Power Flow          : {mean_pf_ms:.4f} ± {std_pf_ms:.4f} ms")
    print(f"  DCOPF + PF (total)     : {mean_total_ms:.4f} ms")
    print(f"")
    print(f"  Note: For GNN inference benchmark, end-to-end inference time")
    print(f"        = DCOPF + PF + GNN forward pass")
    print(f"{'=' * 70}")

    # ── 6. Save results ───────────────────────────────────────────────
    print(f"\n[6] Saving sub-optimal state to {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)

    bus_id_cols = [f"bus_{int(bid)}" for bid in bus_ids]

    pd.DataFrame(subopt_vm, columns=[f"vm_{int(bid)}" for bid in bus_ids]) \
        .to_csv(os.path.join(output_dir, f"{case_name}_subopt_vm.csv"), index=False)

    pd.DataFrame(subopt_va, columns=[f"va_{int(bid)}" for bid in bus_ids]) \
        .to_csv(os.path.join(output_dir, f"{case_name}_subopt_va.csv"), index=False)

    pd.DataFrame(subopt_pinj, columns=[f"pinj_{int(bid)}" for bid in bus_ids]) \
        .to_csv(os.path.join(output_dir, f"{case_name}_subopt_pinj.csv"), index=False)

    pd.DataFrame(subopt_qinj, columns=[f"qinj_{int(bid)}" for bid in bus_ids]) \
        .to_csv(os.path.join(output_dir, f"{case_name}_subopt_qinj.csv"), index=False)

    # Save convergence flags
    pd.DataFrame({'converged': success_flags.astype(int)}) \
        .to_csv(os.path.join(output_dir, f"{case_name}_subopt_converged.csv"), index=False)

    print("    ✓ Done!")
    print(f"\n    Files saved:")
    print(f"      {case_name}_subopt_vm.csv       — voltage magnitudes")
    print(f"      {case_name}_subopt_va.csv       — voltage angles (rad)")
    print(f"      {case_name}_subopt_pinj.csv     — active power injection")
    print(f"      {case_name}_subopt_qinj.csv     — reactive power injection")
    print(f"      {case_name}_subopt_converged.csv — convergence flags")

    # Return timing info alongside convergence flags
    timing_info = {
        'total_wall_time_s':        elapsed,
        'mean_dcopf_ms_per_sample': mean_dcopf_ms,
        'mean_pf_ms_per_sample':    mean_pf_ms,
        'mean_total_ms_per_sample': mean_total_ms,
        'std_dcopf_ms':             std_dcopf_ms,
        'std_pf_ms':                std_pf_ms,
        'dcopf_times':              dcopf_times,
        'pf_times':                 pf_times,
    }

    return success_flags, timing_info


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":

    def main():
        CASE_FILE              = r"/lambda/nfs/lxy/acopf_project/PGlib/standard/pglib_opf_case300_ieee.m"
        ACOPF_DATA_DIR         = r"/lambda/nfs/lxy/acopf_project/data/ACOPF dataset/case300(v=0.12)"
        ACOPF_CONSTRAINTS_PATH = r"/lambda/nfs/lxy/acopf_project/data/ACOPF Constraints/case300"
        DCOPF_CONSTRAINTS_PATH = r"/lambda/nfs/lxy/dcopf_project/data/DCOPF Constraints/case300"
        OUTPUT_DIR             = r"/lambda/nfs/lxy/acopf_project/data/ACOPF dataset/case300(v=0.12)"

        CASE_NAME = "pglib_opf_case300_ieee"

        success_flags, timing_info = generate_suboptimal_states(
            case_name              = CASE_NAME,
            acopf_data_dir         = ACOPF_DATA_DIR,
            acopf_constraints_path = ACOPF_CONSTRAINTS_PATH,
            dcopf_constraints_path = DCOPF_CONSTRAINTS_PATH,
            output_dir             = OUTPUT_DIR,
        )

    main()