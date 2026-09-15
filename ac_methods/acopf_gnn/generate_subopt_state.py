# -*- coding: utf-8 -*-
"""Generate the sub-optimal state X = [vm, va, p_inj, q_inj] used as GNN input.

For every load sample in the ACOPF dataset this solves a DCOPF, runs an AC power flow
at that dispatch, and records the resulting bus state, following Owerko et al. (ICASSP
2020). Writes {case_name}_subopt_{vm,va,pinj,qinj,converged}.csv next to the samples.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import perf_counter

from scipy.optimize import linprog, minimize
from pypower.runpf import runpf

# The shared modules live in ac_configuration/, a subpackage of this script's
# directory, so they resolve regardless of the working directory.
try:
    from ac_configuration import acopf_config
    from ac_configuration.acopf_pypower import get_ppopt, load_case_from_csv
except ImportError as e:
    print(f"Error: Unable to import from ac_configuration/ ({e})")
    sys.exit(1)


def load_dcopf_constraints(case_name, dcopf_constraints_path):
    """Load the PTDF matrix, generator limits and costs exported for the DCOPF."""
    base = Path(dcopf_constraints_path)

    gen_limits = pd.read_csv(base / f"{case_name}_gen_limits.csv")
    gen_costs = pd.read_csv(base / f"{case_name}_gen_costs.csv")
    branch_limits = pd.read_csv(base / f"{case_name}_branch_limits.csv")
    ptdf = pd.read_csv(base / f"{case_name}_ptdf_matrix.csv").values.astype('float64')
    bus_gen_map = pd.read_csv(base / f"{case_name}_bus_gen_map.csv").values.astype('float64')
    base_mva_df = pd.read_csv(base / f"{case_name}_base_mva.csv")

    return {
        'ptdf': ptdf,
        'bus_gen_map': bus_gen_map,
        'pg_min': gen_limits['pgmin'].values,
        'pg_max': gen_limits['pgmax'].values,
        'f_max': branch_limits['rate_a'].values,
        'cost_c2': gen_costs['cost_c2'].values,
        'cost_c1': gen_costs['cost_c1'].values,
        'cost_c0': gen_costs['cost_c0'].values,
        'gen_ids': gen_limits['gen_id'].values,
        'n_buses': bus_gen_map.shape[0],
        'n_gen': len(gen_limits),
        'n_branch': ptdf.shape[0],
        'baseMVA': base_mva_df['value'].iloc[0],
    }


def solve_dcopf(pd_vector, dcopf_params):
    """Solve the DCOPF for one load sample; returns Pg in p.u. or None if infeasible.

        min  sum(c2 * pg^2 + c1 * pg + c0)
        s.t. sum(pg) = sum(pd)
             pg_min <= pg <= pg_max
             |PTDF @ (bus_gen_map @ pg - pd)| <= f_max
    """
    ptdf = dcopf_params['ptdf']
    bgm = dcopf_params['bus_gen_map']
    pg_min = dcopf_params['pg_min']
    pg_max = dcopf_params['pg_max']
    f_max = dcopf_params['f_max']
    c2 = dcopf_params['cost_c2']
    c1 = dcopf_params['cost_c1']
    n_gen = dcopf_params['n_gen']

    # Unlimited branches are dropped rather than given a huge bound, which would
    # leave rows of near-zero slack in the constraint matrix
    valid = f_max < 1e10
    ptdf_v = ptdf[valid, :]
    f_max_v = f_max[valid]
    n_valid = ptdf_v.shape[0]

    total_load = pd_vector.sum()
    ptdf_pd = ptdf_v @ pd_vector
    ptdf_bgm = ptdf_v @ bgm

    bounds = [(pg_min[i], pg_max[i]) for i in range(n_gen)]

    if np.all(c2 == 0):
        res = linprog(
            c1,
            A_ub=np.vstack([ptdf_bgm, -ptdf_bgm]),
            b_ub=np.concatenate([f_max_v + ptdf_pd, f_max_v - ptdf_pd]),
            A_eq=np.ones((1, n_gen)),
            b_eq=np.array([total_load]),
            bounds=bounds,
            method='highs'
        )
        return res.x if res.success else None

    constraints = [{
        'type': 'eq',
        'fun': lambda pg: np.sum(pg) - total_load,
        'jac': lambda pg: np.ones(n_gen)
    }]
    if n_valid > 0:
        constraints.append({
            'type': 'ineq',
            'fun': lambda pg: f_max_v - (ptdf_bgm @ pg - ptdf_pd),
            'jac': lambda pg: -ptdf_bgm
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda pg: f_max_v + (ptdf_bgm @ pg - ptdf_pd),
            'jac': lambda pg: ptdf_bgm
        })

    pg0 = (pg_max * (total_load / pg_max.sum()) if pg_max.sum() > 0
           else np.full(n_gen, total_load / n_gen))
    pg0 = np.clip(pg0, pg_min, pg_max)

    res = minimize(
        lambda pg: np.sum(c2 * pg ** 2 + c1 * pg + dcopf_params['cost_c0']),
        pg0,
        jac=lambda pg: 2 * c2 * pg + c1,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12}
    )
    return res.x if res.success else None


def run_powerflow_with_dcopf(pd_pu, qd_pu, pg_dcopf, ppc_template,
                             load_bus_ids, bus_id_to_idx, baseMVA):
    """Run an AC power flow at the DCOPF dispatch and return the resulting bus state.

    Unlike the method scripts, Pg is set on every generator including the slack unit,
    because the DCOPF already dispatched all of them, and generator Vm is left at the
    case template value. That is why this does not reuse the shared solver.

    Returns (success, vm, va_rad, p_inj, q_inj); the arrays are None on failure.
    """
    mpc = {
        'version': ppc_template['version'],
        'baseMVA': ppc_template['baseMVA'],
        'bus': ppc_template['bus'].copy(),
        'gen': ppc_template['gen'].copy(),
        'branch': ppc_template['branch'].copy(),
        'gencost': ppc_template['gencost'],
    }

    for i, bus_id in enumerate(load_bus_ids):
        idx = bus_id_to_idx.get(int(bus_id))
        if idx is not None:
            mpc['bus'][idx, 2] = pd_pu[i] * baseMVA
            mpc['bus'][idx, 3] = qd_pu[i] * baseMVA

    n_gen = mpc['gen'].shape[0]
    for i in range(n_gen):
        mpc['gen'][i, 1] = pg_dcopf[i] * baseMVA

    result, success = runpf(mpc, get_ppopt())

    if not success:
        return False, None, None, None, None

    vm = result['bus'][:, 7]
    va = result['bus'][:, 8] * np.pi / 180.0

    # Net injection per bus: generation minus load, both converted back to p.u.
    p_inj = -result['bus'][:, 2] / baseMVA
    q_inj = -result['bus'][:, 3] / baseMVA
    for g in range(n_gen):
        g_bus_idx = bus_id_to_idx[int(mpc['gen'][g, 0])]
        p_inj[g_bus_idx] += result['gen'][g, 1] / baseMVA
        q_inj[g_bus_idx] += result['gen'][g, 2] / baseMVA

    return True, vm, va, p_inj, q_inj


def load_acopf_loads(data_dir, case_name):
    """Read pd and qd from the sample CSVs, ordered by bus id."""
    pd_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_pd.csv"))
    qd_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_qd.csv"))

    def extract_id(col):
        match = re.search(r'(\d+)$', col)
        return int(match.group(1)) if match else -1

    pd_cols = sorted([c for c in pd_df.columns if c.startswith('pd')], key=extract_id)
    qd_cols = sorted([c for c in qd_df.columns if c.startswith('qd')], key=extract_id)

    return (pd_df[pd_cols].values.astype('float64'),
            qd_df[qd_cols].values.astype('float64'),
            [extract_id(c) for c in pd_cols])


def build_full_bus_load(pd_sample, load_bus_ids, dcopf_bus_lookup, n_buses_dcopf):
    """Scatter per-load-bus Pd values into the full bus vector the PTDF expects."""
    pd_full = np.zeros(n_buses_dcopf)
    for i, bus_id in enumerate(load_bus_ids):
        if bus_id in dcopf_bus_lookup:
            pd_full[dcopf_bus_lookup[bus_id]] = pd_sample[i]
    return pd_full


def generate_suboptimal_states(case_name, acopf_data_dir, acopf_constraints_path,
                               dcopf_constraints_path, output_dir=None):
    """Produce the sub-optimal state for every sample and write it beside the data."""
    if output_dir is None:
        output_dir = acopf_data_dir

    print("=" * 70)
    print("Generate Sub-optimal State X = [vm, va, p_inj, q_inj]")
    print("=" * 70)

    print(f"\n[1] Loading sample loads...")
    pd_raw, qd_raw, load_bus_ids = load_acopf_loads(acopf_data_dir, case_name)
    n_samples, n_loads = pd_raw.shape
    print(f"    Samples: {n_samples}, Load buses: {n_loads}")

    print(f"\n[2] Loading PyPower case data...")
    ppc = load_case_from_csv(case_name, acopf_constraints_path)
    bus_ids = ppc['bus'][:, 0].astype(int)
    bus_id_to_idx = {int(bid): i for i, bid in enumerate(bus_ids)}
    n_buses = len(bus_ids)
    baseMVA = ppc['baseMVA']
    n_gen = ppc['gen'].shape[0]
    print(f"    Buses: {n_buses}, Generators: {n_gen}, Base MVA: {baseMVA}")

    print(f"\n[3] Loading DCOPF constraints...")
    dcopf_params = load_dcopf_constraints(case_name, dcopf_constraints_path)
    print(f"    DCOPF buses: {dcopf_params['n_buses']}, "
          f"generators: {dcopf_params['n_gen']}, branches: {dcopf_params['n_branch']}")

    # A generator count mismatch means the AC and DC exports describe different
    # systems, so every dispatch would be mapped onto the wrong units
    if n_gen != dcopf_params['n_gen']:
        raise ValueError(
            f"Generator count mismatch: PyPower case has {n_gen}, DCOPF constraints "
            f"have {dcopf_params['n_gen']}; the two exports are out of sync")

    # The PTDF columns follow the Julia exporter's bus_lookup, which is the sorted
    # bus ids, whereas everything else here is indexed by CSV row position
    dcopf_bus_lookup = {int(bid): i for i, bid in enumerate(sorted(bus_ids))}

    print(f"\n[4] Solving DCOPF and power flow for {n_samples} samples...")

    subopt_vm = np.zeros((n_samples, n_buses))
    subopt_va = np.zeros((n_samples, n_buses))
    subopt_pinj = np.zeros((n_samples, n_buses))
    subopt_qinj = np.zeros((n_samples, n_buses))
    pf_converged = np.zeros(n_samples, dtype=bool)
    dcopf_solved = np.zeros(n_samples, dtype=bool)

    dcopf_times = np.zeros(n_samples)
    pf_times = np.zeros(n_samples)
    dcopf_fail = 0
    pf_fail = 0

    t0 = perf_counter()

    for i in range(n_samples):
        if (i + 1) % 1000 == 0 or i == 0:
            elapsed = perf_counter() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    Sample {i + 1}/{n_samples} ({rate:.0f} samples/s, "
                  f"DCOPF fail: {dcopf_fail}, PF fail: {pf_fail})")

        pd_full = build_full_bus_load(
            pd_raw[i], load_bus_ids, dcopf_bus_lookup, dcopf_params['n_buses'])

        t_start = perf_counter()
        pg_dcopf = solve_dcopf(pd_full, dcopf_params)
        dcopf_times[i] = perf_counter() - t_start

        if pg_dcopf is None:
            # Falling back to a proportional dispatch keeps the pipeline running, but
            # the resulting state is not a DCOPF solution, so it is flagged separately
            dcopf_fail += 1
            pg_max = dcopf_params['pg_max']
            pg_dcopf = np.clip(
                pg_max * (pd_full.sum() / pg_max.sum()),
                dcopf_params['pg_min'], pg_max)
        else:
            dcopf_solved[i] = True

        t_start = perf_counter()
        ok, vm, va, p_inj, q_inj = run_powerflow_with_dcopf(
            pd_raw[i], qd_raw[i], pg_dcopf, ppc, load_bus_ids, bus_id_to_idx, baseMVA)
        pf_times[i] = perf_counter() - t_start

        if ok:
            subopt_vm[i] = vm
            subopt_va[i] = va
            subopt_pinj[i] = p_inj
            subopt_qinj[i] = q_inj
            pf_converged[i] = True
        else:
            # Rows for failed samples stay zero; consumers must filter on the
            # convergence flags rather than treat them as data
            pf_fail += 1

    elapsed = perf_counter() - t0
    n_ok = int(pf_converged.sum())

    print(f"\n[5] Completed in {elapsed:.1f}s")
    print(f"    Power flow converged: {n_ok}/{n_samples} ({n_ok / n_samples * 100:.1f}%)")
    print(f"    DCOPF solved: {int(dcopf_solved.sum())}/{n_samples} "
          f"({dcopf_solved.mean() * 100:.1f}%)")
    print(f"    Usable (DCOPF solved and power flow converged): "
          f"{int((dcopf_solved & pf_converged).sum())}/{n_samples}")

    mean_dcopf_ms = np.mean(dcopf_times) * 1000
    mean_pf_ms = np.mean(pf_times) * 1000

    print(f"\n{'=' * 70}")
    print(f"Timing Summary")
    print(f"{'=' * 70}")
    print(f"  Total wall time: {elapsed:.2f} s for {n_samples} samples "
          f"({n_samples / elapsed:.1f} samples/s)")
    print(f"  DCOPF solve:    {mean_dcopf_ms:.4f} +/- {np.std(dcopf_times) * 1000:.4f} "
          f"ms/sample")
    print(f"  AC power flow:  {mean_pf_ms:.4f} +/- {np.std(pf_times) * 1000:.4f} "
          f"ms/sample")
    print(f"  Combined:       {mean_dcopf_ms + mean_pf_ms:.4f} ms/sample")
    print(f"  This is the cost of preparing the GNN's input, which the GNN's own")
    print(f"  reported inference time does not include.")
    print(f"{'=' * 70}")

    print(f"\n[6] Saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for name, array, prefix in [
        ('vm', subopt_vm, 'vm'),
        ('va', subopt_va, 'va'),
        ('pinj', subopt_pinj, 'pinj'),
        ('qinj', subopt_qinj, 'qinj'),
    ]:
        pd.DataFrame(array, columns=[f"{prefix}_{int(bid)}" for bid in bus_ids]) \
            .to_csv(os.path.join(output_dir, f"{case_name}_subopt_{name}.csv"),
                    index=False)

    pd.DataFrame({
        'converged': pf_converged.astype(int),
        'dcopf_solved': dcopf_solved.astype(int),
    }).to_csv(os.path.join(output_dir, f"{case_name}_subopt_converged.csv"), index=False)

    print(f"    Wrote {case_name}_subopt_{{vm,va,pinj,qinj,converged}}.csv")

    return {
        'pf_converged': pf_converged,
        'dcopf_solved': dcopf_solved,
        'total_wall_time_s': elapsed,
        'mean_dcopf_ms_per_sample': mean_dcopf_ms,
        'mean_pf_ms_per_sample': mean_pf_ms,
        'mean_total_ms_per_sample': mean_dcopf_ms + mean_pf_ms,
    }


if __name__ == "__main__":
    paths = acopf_config.get_all_paths()

    results = generate_suboptimal_states(
        case_name=paths['case_name'],
        acopf_data_dir=os.path.dirname(paths['data_path']),
        acopf_constraints_path=paths['params_path'],
        dcopf_constraints_path=acopf_config.get_dc_params_path(),
    )

    print("\nSub-optimal state generation completed successfully!")