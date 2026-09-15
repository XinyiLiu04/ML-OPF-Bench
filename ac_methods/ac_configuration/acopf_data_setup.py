# -*- coding: utf-8 -*-
"""ACOPF data loading and preprocessing."""

import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def reconstruct_full_pg(pg_non_slack, params):
    """Scatter non-slack Pg back into a full-length Pg array; slack slots are left at 0."""
    n_gen = params['general']['n_gen']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']

    if pg_non_slack.ndim == 1:
        pg_full = np.zeros(n_gen, dtype=pg_non_slack.dtype)
        pg_full[non_slack_gen_idx] = pg_non_slack
    else:
        pg_full = np.zeros((pg_non_slack.shape[0], n_gen), dtype=pg_non_slack.dtype)
        pg_full[:, non_slack_gen_idx] = pg_non_slack

    return pg_full


def prepare_data_splits(
        x_data_scaled,
        y_data_scaled,
        n_train_use=None,
        seed=42,
        val_ratio=1 / 12,
        test_ratio=1 / 12
):
    """Draw a random train/val/test split from the first n_train_use shuffled samples."""
    rng = np.random.default_rng(seed)
    n_total = len(x_data_scaled)

    n_use = n_total if (n_train_use is None or n_train_use > n_total) else n_train_use
    pool_indices = rng.permutation(n_total)[:n_use]

    n_val = max(1, int(n_use * val_ratio))
    n_test = max(1, int(n_use * test_ratio))
    n_train = n_use - n_val - n_test

    if n_train <= 0:
        raise ValueError(f"Sample size {n_use} too small for splitting")

    train_idx = pool_indices[:n_train]
    val_idx = pool_indices[n_train:n_train + n_val]
    test_idx = pool_indices[n_train + n_val:]

    print(f"\n[Data Split]")
    print(f"  Total samples: {n_total}, Used samples: {n_use}")
    print(f"  Train: {len(train_idx)} ({len(train_idx) / n_use * 100:.1f}%)")
    print(f"  Val: {len(val_idx)} ({len(val_idx) / n_use * 100:.1f}%)")
    print(f"  Test: {len(test_idx)} ({len(test_idx) / n_use * 100:.1f}%)")

    return train_idx, val_idx, test_idx


def extract_id(col):
    """Parse the trailing integer id from a column name such as 'pd_12'."""
    match = re.search(r'(\d+)$', col)
    return int(match.group(1)) if match else -1


def parse_case_name(data_path):
    """Strip the variable suffix from a data filename to recover the case name."""
    base_filename = os.path.basename(data_path)
    for suffix in ('_pd.csv', '_qd.csv', '_pg.csv'):
        if base_filename.endswith(suffix):
            return base_filename[:-len(suffix)]
    return base_filename.rsplit('_', 1)[0]


def compute_cost_baseline_from_data(data_dir, case_name, params):
    """Mean generation cost of the ground-truth OPF solutions, used as the cost reference."""
    try:
        pg_csv_path = os.path.join(data_dir, f"{case_name}_pg.csv")
        if not os.path.exists(pg_csv_path):
            print(f"Warning: {pg_csv_path} not found")
            return None

        pg_df = pd.read_csv(pg_csv_path)
        pg_cols = sorted([col for col in pg_df.columns if col.startswith('pg_')],
                         key=lambda x: int(x.split('_')[1]))

        if len(pg_cols) == 0:
            print(f"Warning: No pg_ columns found in pg.csv")
            return None

        pg_pu = pg_df[pg_cols].values

        cost_per_gen = (params['generator']['cost_c2'].reshape(1, -1) * pg_pu ** 2 +
                        params['generator']['cost_c1'].reshape(1, -1) * pg_pu +
                        params['generator']['cost_c0'].reshape(1, -1))
        cost_per_sample = np.sum(cost_per_gen, axis=1)

        print(f"(data_setup) Cost baseline computed from dataset:")
        print(f"  Mean cost: {np.mean(cost_per_sample):.2f} $/h")
        print(f"  Cost range: [{np.min(cost_per_sample):.2f}, {np.max(cost_per_sample):.2f}] $/h")
        print(f"  Std dev: {np.std(cost_per_sample):.2f} $/h")
        print(f"  Based on {len(cost_per_sample)} samples")

        return np.mean(cost_per_sample)

    except Exception as e:
        print(f"Warning: Error computing cost baseline: {e}")
        return None


def load_parameters_from_csv(case_name, params_path):
    """Load network constraints and classify generators into slack / non-slack."""
    DTYPE = 'float32'

    bus_data = pd.read_csv(os.path.join(params_path, f"{case_name}_bus_data.csv"))
    gen_data = pd.read_csv(os.path.join(params_path, f"{case_name}_gen_data.csv"))
    branch_data = pd.read_csv(os.path.join(params_path, f"{case_name}_branch_data.csv"))

    try:
        bus_gen_map = pd.read_csv(os.path.join(params_path, f"{case_name}_bus_gen_map.csv"), header=0)
    except FileNotFoundError:
        print("Warning: bus_gen_map.csv not found. Using placeholder.")
        bus_gen_map = None

    base_mva_df = pd.read_csv(os.path.join(params_path, f"{case_name}_base_mva.csv"))
    BASE_MVA = base_mva_df['value'].iloc[0]

    bus_ids = bus_data['bus_id'].values
    bus_types = bus_data['type'].values
    bus_id_to_idx = {int(bid): idx for idx, bid in enumerate(bus_ids)}

    # Bus ids may be non-consecutive, so all indexing goes through bus_id_to_idx
    is_sparse = (len(bus_ids) != bus_ids.max())

    print(f"(data_setup) Bus mapping created: {len(bus_id_to_idx)} buses")
    print(f"  Bus ID range: [{bus_ids.min()}, {bus_ids.max()}]")
    if is_sparse:
        print(f"  Detected sparse bus numbering (non-consecutive)")

    slack_bus_ids = bus_ids[bus_types == 3]

    n_buses = len(bus_data)
    n_gen = len(gen_data)
    n_branches = len(branch_data)

    load_buses = bus_data['bus_id'].values
    n_loads = len(load_buses)

    gen_bus_ids = gen_data['bus_id'].values

    # Slack Pg is determined by the power flow solution, so it is not a DNN target
    slack_gen_mask = np.array([int(gid) in slack_bus_ids for gid in gen_bus_ids], dtype=bool)
    non_slack_gen_idx = np.where(~slack_gen_mask)[0]
    n_gen_non_slack = len(non_slack_gen_idx)

    print(f"\n(data_setup) Generator classification:")
    print(f"  Slack Bus ID: {slack_bus_ids}")
    print(f"  Total generators: {n_gen}")
    print(f"  Slack generators: {np.sum(slack_gen_mask)} (excluded from DNN prediction)")
    print(f"  Non-Slack generators: {n_gen_non_slack} (DNN prediction target)")

    pg_min = gen_data['pg_min_pu'].values.astype(DTYPE)
    pg_max = gen_data['pg_max_pu'].values.astype(DTYPE)
    qg_min = gen_data['qg_min_pu'].values.astype(DTYPE)
    qg_max = gen_data['qg_max_pu'].values.astype(DTYPE)

    print(f"\n(data_setup) Generator constraints loaded (p.u., BASE_MVA={BASE_MVA})")
    print(f"  Pg range: [{pg_min.min():.4f}, {pg_max.max():.4f}] p.u.")

    cost_c2 = gen_data['cost_c2'].values.astype(DTYPE)
    cost_c1 = gen_data['cost_c1'].values.astype(DTYPE)
    cost_c0 = gen_data['cost_c0'].values.astype(DTYPE)

    vm_min = bus_data['vmin_pu'].values.astype(DTYPE)
    vm_max = bus_data['vmax_pu'].values.astype(DTYPE)

    branch_ids = branch_data['branch_id'].values
    f_bus = branch_data['f_bus'].values
    t_bus = branch_data['t_bus'].values
    r_pu = branch_data['r_pu'].values.astype(DTYPE)
    x_pu = branch_data['x_pu'].values.astype(DTYPE)
    b_pu = branch_data['b_pu'].values.astype(DTYPE)
    rate_a = branch_data['rate_a_pu'].values.astype(DTYPE)
    tap_ratio = branch_data['tap_ratio'].values.astype(DTYPE)
    shift_deg = branch_data['shift_deg'].values.astype(DTYPE)

    # Derived branch quantities, for methods that evaluate the power flow equations
    # inside the loss instead of calling a solver
    f_bus_idx = np.array([bus_id_to_idx[int(fb)] for fb in f_bus])
    t_bus_idx = np.array([bus_id_to_idx[int(tb)] for tb in t_bus])

    z_sq = np.maximum(r_pu ** 2 + x_pu ** 2, 1e-20).astype(DTYPE)
    g_br = (r_pu / z_sq).astype(DTYPE)
    b_br = (-x_pu / z_sq).astype(DTYPE)

    # Branch admittance coefficients for the full pi-model, including line charging
    # and the transformer tap. They are precomputed here so the flow equations reduce
    # to indexing these arrays, rather than each method redoing the derivation:
    #   Pf =  vi^2*Yff_g + vi*vj*( Yft_g*cos(t_ij) + Yft_b*sin(t_ij))
    #   Qf = -vi^2*Yff_b + vi*vj*( Yft_g*sin(t_ij) - Yft_b*cos(t_ij))
    #   Pt =  vj^2*Ytt_g + vi*vj*( Ytf_g*cos(t_ij) - Ytf_b*sin(t_ij))
    #   Qt = -vj^2*Ytt_b + vi*vj*(-Ytf_g*sin(t_ij) - Ytf_b*cos(t_ij))
    # With b_pu = 0, tap_ratio = 1 and shift_deg = 0 these reduce to the series-only
    # formulas exactly.
    tau = np.where(tap_ratio == 0, 1.0, tap_ratio).astype(np.float64)
    shift_rad = np.deg2rad(shift_deg.astype(np.float64))
    cos_sh = np.cos(shift_rad)
    sin_sh = np.sin(shift_rad)
    g_s = g_br.astype(np.float64)
    b_s = b_br.astype(np.float64)
    # b_pu is the total line charging susceptance, split evenly across both ends
    b_tot = b_s + b_pu.astype(np.float64) / 2.0

    Yff_g = (g_s / tau ** 2).astype(DTYPE)
    Yff_b = (b_tot / tau ** 2).astype(DTYPE)
    Ytt_g = g_s.astype(DTYPE)
    Ytt_b = b_tot.astype(DTYPE)
    Yft_g = ((-g_s * cos_sh + b_s * sin_sh) / tau).astype(DTYPE)
    Yft_b = ((-g_s * sin_sh - b_s * cos_sh) / tau).astype(DTYPE)
    Ytf_g = ((-g_s * cos_sh - b_s * sin_sh) / tau).astype(DTYPE)
    Ytf_b = ((g_s * sin_sh - b_s * cos_sh) / tau).astype(DTYPE)

    bus_gen_map_matrix = np.zeros((n_buses, n_gen), dtype=DTYPE)
    if bus_gen_map is not None:
        gen_cols = [col for col in bus_gen_map.columns if col.startswith('gen_')]
        for i, gen_col in enumerate(gen_cols):
            if i < n_gen:
                bus_gen_map_matrix[:, i] = bus_gen_map[gen_col].values

    simulation_parameters = {
        'general': {
            'n_buses': n_buses,
            'n_gen': n_gen,
            'n_gen_non_slack': n_gen_non_slack,
            'n_branches': n_branches,
            'n_loads': n_loads,
            'gen_bus_ids': gen_bus_ids,
            'load_bus_ids': load_buses,
            'branch_ids': branch_ids,
            'BASE_MVA': BASE_MVA,
            'bus_ids': bus_ids,
            'bus_types': bus_types,
            'bus_id_to_idx': bus_id_to_idx,
            'slack_gen_mask': slack_gen_mask,
            'non_slack_gen_idx': non_slack_gen_idx,
        },
        'generator': {
            'pg_min': pg_min.reshape(1, -1),
            'pg_max': pg_max.reshape(1, -1),
            'qg_min': qg_min.reshape(1, -1),
            'qg_max': qg_max.reshape(1, -1),
            'cost_c2': cost_c2,
            'cost_c1': cost_c1,
            'cost_c0': cost_c0,
        },
        'bus': {
            'vm_min': vm_min,
            'vm_max': vm_max,
        },
        'branch': {
            'f_bus': f_bus,
            't_bus': t_bus,
            'f_bus_idx': f_bus_idx,
            't_bus_idx': t_bus_idx,
            'r_pu': r_pu,
            'x_pu': x_pu,
            'b_pu': b_pu,
            'g_br': g_br,
            'b_br': b_br,
            'rate_a': rate_a,
            'tap_ratio': tap_ratio,
            'shift_deg': shift_deg,
            'Yff_g': Yff_g, 'Yff_b': Yff_b,
            'Yft_g': Yft_g, 'Yft_b': Yft_b,
            'Ytf_g': Ytf_g, 'Ytf_b': Ytf_b,
            'Ytt_g': Ytt_g, 'Ytt_b': Ytt_b,
        },
        'topology': {
            'bus_gen_map': bus_gen_map_matrix,
        }
    }
    return simulation_parameters


def load_and_scale_acopf_data(data_path, params, fit_scalers=True, scalers=None):
    """Load the dataset and scale it; Y contains only non-slack Pg and generator-bus Vm."""
    case_name = parse_case_name(data_path)
    data_dir = os.path.dirname(data_path)

    pd_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_pd.csv"))
    qd_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_qd.csv"))
    pg_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_pg.csv"))
    qg_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_qg.csv"))
    vm_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_vm.csv"))
    va_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_va.csv"))

    cost_baseline = compute_cost_baseline_from_data(data_dir, case_name, params)

    bus_ids = params['general']['bus_ids']
    gen_bus_ids = params['general']['gen_bus_ids']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    # Inputs: load active and reactive power, ordered by bus id
    sorted_pd_cols = sorted([c for c in pd_df.columns if c.startswith('pd')], key=extract_id)
    sorted_qd_cols = sorted([c for c in qd_df.columns if c.startswith('qd')], key=extract_id)
    load_bus_ids = [extract_id(col) for col in sorted_pd_cols]
    n_loads = len(load_bus_ids)

    if params['general']['n_loads'] != n_loads:
        print(f"(data_setup) Updating n_loads: {params['general']['n_loads']} -> {n_loads}")
        params['general']['n_loads'] = n_loads
        params['general']['load_bus_ids'] = np.array(load_bus_ids)

    x_pd_raw = pd_df[sorted_pd_cols].values.astype('float32')
    x_qd_raw = qd_df[sorted_qd_cols].values.astype('float32')
    x_data_raw = np.hstack([x_pd_raw, x_qd_raw])

    print(f"(data_setup) Input feature dimension: {x_data_raw.shape} (pd: {n_loads}, qd: {n_loads})")
    print(f"  Pd range: [{x_pd_raw.min():.4f}, {x_pd_raw.max():.4f}] p.u.")

    pg_cols = sorted([c for c in pg_df.columns if c.startswith('pg')], key=extract_id)
    y_pg_raw_all = pg_df[pg_cols].values.astype('float32')
    y_pg_raw_non_slack = y_pg_raw_all[:, non_slack_gen_idx]

    qg_cols = sorted([c for c in qg_df.columns if c.startswith('qg')], key=extract_id)
    y_qg_raw = qg_df[qg_cols].values.astype('float32')

    # Vm targets are the generator-bus setpoints; all-bus Vm is kept for evaluation only
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_vm_raw_all = vm_df[[f"vm_{bid}" for bid in bus_ids]].values.astype('float32')
    y_vm_raw_gen = y_vm_raw_all[:, gen_bus_indices]

    y_va_raw = va_df[[f"va_{bid}" for bid in bus_ids]].values.astype('float32')

    print(f"(data_setup) Target dimensions: pg_non_slack={y_pg_raw_non_slack.shape[1]} "
          f"(of {y_pg_raw_all.shape[1]} generators), vm_gen={y_vm_raw_gen.shape[1]} "
          f"(of {y_vm_raw_all.shape[1]} buses)")
    print(f"  Non-Slack Pg range: [{y_pg_raw_non_slack.min():.4f}, {y_pg_raw_non_slack.max():.4f}] p.u.")
    print(f"  Generator Vm range: [{y_vm_raw_gen.min():.4f}, {y_vm_raw_gen.max():.4f}] p.u.")

    if fit_scalers:
        scalers = {
            'x': MinMaxScaler(),
            'pg': MinMaxScaler(),
            'qg': MinMaxScaler(),
            'vm': MinMaxScaler(),
            'va': MinMaxScaler(),
        }
        x_data_scaled = scalers['x'].fit_transform(x_data_raw)
        y_pg_scaled = scalers['pg'].fit_transform(y_pg_raw_non_slack)
        y_qg_scaled = scalers['qg'].fit_transform(y_qg_raw)
        y_vm_scaled = scalers['vm'].fit_transform(y_vm_raw_gen)
        y_va_scaled = scalers['va'].fit_transform(y_va_raw)
        print(f"(data_setup) Scalers fitted")
    else:
        if scalers is None:
            raise ValueError("scalers parameter required when fit_scalers=False")
        x_data_scaled = scalers['x'].transform(x_data_raw)
        y_pg_scaled = scalers['pg'].transform(y_pg_raw_non_slack)
        y_qg_scaled = scalers['qg'].transform(y_qg_raw)
        y_vm_scaled = scalers['vm'].transform(y_vm_raw_gen)
        y_va_scaled = scalers['va'].transform(y_va_raw)
        print(f"(data_setup) Using existing scalers")

    y_data_scaled = np.hstack([y_pg_scaled, y_vm_scaled])
    print(f"(data_setup) Final Y dimension (pg_non_slack, vm_gen): {y_data_scaled.shape}")

    raw_data = {
        'x': x_data_raw,
        'pg': y_pg_raw_all,
        'pg_non_slack': y_pg_raw_non_slack,
        'qg': y_qg_raw,
        'vm': y_vm_raw_all,
        'vm_gen': y_vm_raw_gen,
        'va': y_va_raw,
    }

    return x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline