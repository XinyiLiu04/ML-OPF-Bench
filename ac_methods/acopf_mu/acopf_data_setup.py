"""
ACOPF Data Loading and Preprocessing Module (V10-Lagrangian-Full)

For strict replication of Fioretto et al. 2019 (M_C^D model).

Key Changes from V9:
- DNN now predicts 4 output groups: vm_all (n), va_all (n), pg_non_slack (g), qg_all (g_total)
- Separate scalers for each output group
- raw_data includes all quantities needed for constraint computation
- Slack bus Pg still excluded from DNN prediction (restored by power flow at evaluation)
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# =====================================================================
# Data Mode Enumeration
# =====================================================================
class DataMode:
    """Data splitting modes"""
    RANDOM_SPLIT = 'random_split'
    FIXED_VALTEST = 'fixed_valtest'
    GENERALIZATION = 'generalization'
    API_TEST = 'api_test'


# =====================================================================
# Reconstruct Full Pg Array (for power flow calculation)
# =====================================================================
def reconstruct_full_pg(pg_non_slack, params):
    """
    Reconstruct full Pg array from non-slack generator Pg.

    Args:
        pg_non_slack: shape (n_samples, n_gen_non_slack) or (n_gen_non_slack,)
        params: Parameters dictionary
    Returns:
        pg_full: shape (n_samples, n_gen) or (n_gen,)
                 Slack positions filled with 0 (adjusted by power flow)
    """
    n_gen = params['general']['n_gen']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']

    if pg_non_slack.ndim == 1:
        pg_full = np.zeros(n_gen, dtype=pg_non_slack.dtype)
        pg_full[non_slack_gen_idx] = pg_non_slack
    else:
        n_samples = pg_non_slack.shape[0]
        pg_full = np.zeros((n_samples, n_gen), dtype=pg_non_slack.dtype)
        pg_full[:, non_slack_gen_idx] = pg_non_slack

    return pg_full


# =====================================================================
# Data Splitting Functions
# =====================================================================
def prepare_data_splits(
        x_data_scaled,
        y_data_scaled,
        mode='random_split',
        n_train_use=None,
        seed=42,
        val_ratio=1 / 12,
        test_ratio=1 / 12
):
    """Prepare data split indices"""
    rng = np.random.default_rng(seed)
    n_total = len(x_data_scaled)

    if mode in [DataMode.RANDOM_SPLIT, DataMode.GENERALIZATION, DataMode.API_TEST]:
        n_use = n_total if (n_train_use is None or n_train_use > n_total) else n_train_use
        all_indices = rng.permutation(n_total)
        pool_indices = all_indices[:n_use]

        n_val = max(1, int(n_use * val_ratio))
        n_test = max(1, int(n_use * test_ratio))
        n_train = n_use - n_val - n_test

        if n_train <= 0:
            raise ValueError(f"Sample size {n_use} too small for splitting")

        train_idx = pool_indices[:n_train]
        val_idx = pool_indices[n_train:n_train + n_val]
        test_idx = pool_indices[n_train + n_val:]

        if mode == DataMode.RANDOM_SPLIT:
            mode_name = "RANDOM_SPLIT"
        elif mode == DataMode.GENERALIZATION:
            mode_name = "GENERALIZATION (using RANDOM_SPLIT split)"
        else:
            mode_name = "API_TEST (using RANDOM_SPLIT split)"

        print(f"\n[Data Split] Mode: {mode_name}")
        print(f"  Total samples: {n_total}, Used samples: {n_use}")
        print(f"  Train: {len(train_idx)} ({len(train_idx) / n_use * 100:.1f}%)")
        print(f"  Val: {len(val_idx)} ({len(val_idx) / n_use * 100:.1f}%)")
        print(f"  Test: {len(test_idx)} ({len(test_idx) / n_use * 100:.1f}%)")
        if mode in [DataMode.GENERALIZATION, DataMode.API_TEST]:
            print(f"  Note: Test indices will be replaced by external data")

    elif mode == DataMode.FIXED_VALTEST:
        fixed_rng = np.random.default_rng(seed)
        all_indices = fixed_rng.permutation(n_total)

        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, int(n_total * test_ratio))
        n_pool = n_total - n_val - n_test

        pool_indices = all_indices[:n_pool]
        val_idx = all_indices[n_pool:n_pool + n_val]
        test_idx = all_indices[n_pool + n_val:]

        n_train_actual = n_pool if (n_train_use is None or n_train_use > n_pool) else n_train_use
        train_rng = np.random.default_rng(seed + 1000)
        train_idx = train_rng.permutation(pool_indices)[:n_train_actual]

        print(f"\n[Data Split] Mode: FIXED_VALTEST")
        print(f"  Total samples: {n_total}")
        print(f"  Train pool: {n_pool} ({n_pool / n_total * 100:.1f}%)")
        print(f"  Val set: {len(val_idx)} (fixed, {len(val_idx) / n_total * 100:.1f}%)")
        print(f"  Test set: {len(test_idx)} (fixed, {len(test_idx) / n_total * 100:.1f}%)")
        print(f"  Actual training samples: {len(train_idx)} (sampled from train pool)")

    else:
        raise ValueError(f"Unknown data mode: {mode}")

    return train_idx, val_idx, test_idx


def load_generalization_test_data(test_data_path, params, scalers, n_test_samples=None, seed=42):
    """Load generalization test data"""
    print(f"\n[Generalization Test] Loading test data: {test_data_path}")

    test_x_scaled, test_y_scaled, _, test_raw_data, test_cost_baseline = \
        load_and_scale_acopf_data(test_data_path, params, fit_scalers=False, scalers=scalers)

    n_total = len(test_x_scaled)
    if n_test_samples is None or n_test_samples >= n_total:
        print(f"  Using all {n_total} test samples")
        test_idx = np.arange(n_total)
    else:
        rng = np.random.default_rng(seed)
        test_idx = rng.choice(n_total, size=n_test_samples, replace=False)
        print(f"  Randomly selecting {n_test_samples} from {n_total} samples")

        test_x_scaled = test_x_scaled[test_idx]
        test_y_scaled = test_y_scaled[test_idx]
        for key in test_raw_data:
            test_raw_data[key] = test_raw_data[key][test_idx]

    print(f"  Final test samples: {len(test_x_scaled)}")

    return test_x_scaled, test_y_scaled, test_raw_data, test_cost_baseline


def load_api_test_data(test_data_path, test_params_path, train_scalers, n_test_samples=None, seed=42):
    """Load API test data"""
    print(f"\n[API Test] Loading API parameters and data")
    print(f"  Data path: {test_data_path}")
    print(f"  Params path: {test_params_path}")

    base_filename = os.path.basename(test_data_path)
    if base_filename.endswith('_pd.csv'):
        case_name = base_filename[:-7]
    else:
        case_name = base_filename.rsplit('_', 1)[0]

    print(f"  Case name: {case_name}")

    test_params = load_parameters_from_csv(case_name, test_params_path)
    print(f"  ✓ API params loaded")
    print(f"    Buses: {test_params['general']['n_buses']}")
    print(f"    Generators: {test_params['general']['n_gen']}")
    print(f"    Non-Slack Generators: {test_params['general']['n_gen_non_slack']}")
    print(f"    Loads: {test_params['general']['n_loads']}")
    print(f"    Base MVA: {test_params['general']['BASE_MVA']}")

    test_x_scaled, test_y_scaled, _, test_raw_data, test_cost_baseline = \
        load_and_scale_acopf_data(test_data_path, test_params, fit_scalers=False, scalers=train_scalers)

    n_total = len(test_x_scaled)
    if n_test_samples is None or n_test_samples >= n_total:
        print(f"  Using all {n_total} test samples")
        test_idx = np.arange(n_total)
    else:
        rng = np.random.default_rng(seed)
        test_idx = rng.choice(n_total, size=n_test_samples, replace=False)
        print(f"  Randomly selecting {n_test_samples} from {n_total} samples")

        test_x_scaled = test_x_scaled[test_idx]
        test_y_scaled = test_y_scaled[test_idx]
        for key in test_raw_data:
            test_raw_data[key] = test_raw_data[key][test_idx]

    print(f"  ✓ Final test samples: {len(test_x_scaled)}")

    return test_params, test_x_scaled, test_y_scaled, test_raw_data, test_cost_baseline


def compute_cost_baseline_from_data(data_dir, case_name, params):
    """Compute cost baseline from dataset"""
    try:
        pg_csv_path = os.path.join(data_dir, f"{case_name}_pg.csv")
        if not os.path.exists(pg_csv_path):
            return None

        pg_df = pd.read_csv(pg_csv_path)
        pg_cols = sorted([col for col in pg_df.columns if col.startswith('pg_')],
                         key=lambda x: int(x.split('_')[1]))
        if len(pg_cols) == 0:
            return None

        pg_pu = pg_df[pg_cols].values
        cost_c2 = params['generator']['cost_c2']
        cost_c1 = params['generator']['cost_c1']
        cost_c0 = params['generator']['cost_c0']

        cost_per_gen = (cost_c2.reshape(1, -1) * pg_pu ** 2 +
                        cost_c1.reshape(1, -1) * pg_pu +
                        cost_c0.reshape(1, -1))
        cost_per_sample = np.sum(cost_per_gen, axis=1)
        cost_mean = np.mean(cost_per_sample)

        print(f"✓ (data_setup) Cost baseline: mean={cost_mean:.2f} $/h "
              f"[{np.min(cost_per_sample):.2f}, {np.max(cost_per_sample):.2f}] "
              f"over {len(cost_per_sample)} samples")

        return cost_mean
    except Exception as e:
        print(f"⚠️ Warning: Error computing cost baseline: {e}")
        return None


# =====================================================================
# Load Parameters: Identify Slack Bus Generators
# =====================================================================
def load_parameters_from_csv(case_name, params_path):
    """
    Load ACOPF parameters from CSV files and identify Slack Bus generators.
    """
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

    # Bus information
    bus_ids = bus_data['bus_id'].values
    bus_types = bus_data['type'].values
    bus_id_to_idx = {int(bid): idx for idx, bid in enumerate(bus_ids)}

    is_sparse = (len(bus_ids) != bus_ids.max())
    print(f"✓ (data_setup) Bus mapping: {len(bus_id_to_idx)} buses, "
          f"ID range [{bus_ids.min()}, {bus_ids.max()}]"
          f"{' (sparse)' if is_sparse else ''}")

    # Slack Bus
    slack_bus_mask = (bus_types == 3)
    slack_bus_ids = bus_ids[slack_bus_mask]
    print(f"  Slack Bus ID: {slack_bus_ids}")

    # Generator information
    n_buses = len(bus_data)
    n_gen = len(gen_data)
    n_branches = len(branch_data)

    load_buses = bus_data['bus_id'].values
    n_loads = len(load_buses)

    gen_bus_ids = gen_data['bus_id'].values

    # Identify slack generators
    slack_gen_mask = np.array([int(gid) in slack_bus_ids for gid in gen_bus_ids], dtype=bool)
    non_slack_gen_idx = np.where(~slack_gen_mask)[0]
    n_gen_non_slack = len(non_slack_gen_idx)

    print(f"  Generators: {n_gen} total, {n_gen_non_slack} non-slack, "
          f"{np.sum(slack_gen_mask)} slack")

    # Constraint parameters
    pg_min = gen_data['pg_min_pu'].values.astype(DTYPE)
    pg_max = gen_data['pg_max_pu'].values.astype(DTYPE)
    qg_min = gen_data['qg_min_pu'].values.astype(DTYPE)
    qg_max = gen_data['qg_max_pu'].values.astype(DTYPE)

    cost_c2 = gen_data['cost_c2'].values.astype(DTYPE)
    cost_c1 = gen_data['cost_c1'].values.astype(DTYPE)
    cost_c0 = gen_data['cost_c0'].values.astype(DTYPE)

    vm_min = bus_data['vmin_pu'].values.astype(DTYPE)
    vm_max = bus_data['vmax_pu'].values.astype(DTYPE)

    # Branch parameters
    branch_ids = branch_data['branch_id'].values
    f_bus = branch_data['f_bus'].values
    t_bus = branch_data['t_bus'].values
    r_pu = branch_data['r_pu'].values.astype(DTYPE)
    x_pu = branch_data['x_pu'].values.astype(DTYPE)
    b_pu = branch_data['b_pu'].values.astype(DTYPE)
    rate_a = branch_data['rate_a_pu'].values.astype(DTYPE)
    tap_ratio = branch_data['tap_ratio'].values.astype(DTYPE)
    shift_deg = branch_data['shift_deg'].values.astype(DTYPE)

    # Bus-generator mapping
    n_gen_actual = n_gen
    bus_gen_map_matrix = np.zeros((n_buses, n_gen_actual), dtype=DTYPE)
    if bus_gen_map is not None:
        gen_cols = [col for col in bus_gen_map.columns if col.startswith('gen_')]
        for i, gen_col in enumerate(gen_cols):
            if i < n_gen_actual:
                bus_gen_map_matrix[:, i] = bus_gen_map[gen_col].values

    # Compute branch admittance (g_ij, b_ij) from r, x
    z_sq = r_pu ** 2 + x_pu ** 2
    z_sq = np.where(z_sq < 1e-12, 1e-12, z_sq)  # avoid division by zero
    g_br = r_pu / z_sq  # branch conductance
    b_br = -x_pu / z_sq  # branch susceptance

    # Branch endpoint indices (bus_id -> matrix index)
    f_bus_idx = np.array([bus_id_to_idx[int(fb)] for fb in f_bus])
    t_bus_idx = np.array([bus_id_to_idx[int(tb)] for tb in t_bus])

    # Angle difference limits (from branch data, default 360 deg -> essentially no limit)
    theta_diff_max = np.full(n_branches, np.pi, dtype=DTYPE)  # default: pi radians

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
            'theta_diff_max': theta_diff_max,
        },
        'topology': {
            'bus_gen_map': bus_gen_map_matrix,
        }
    }
    return simulation_parameters


# =====================================================================
# Column ID extraction helper
# =====================================================================
def _extract_id(col):
    match = re.search(r'(\d+)$', col)
    return int(match.group(1)) if match else -1


# =====================================================================
# Main Data Loading: Y = (vm_all, va_all, pg_non_slack, qg_all)
# =====================================================================
def load_and_scale_acopf_data(data_path, params, fit_scalers=True, scalers=None):
    """
    Load and normalize ACOPF data for M_C^D replication.

    Output Y structure (concatenated):
        [vm_all (n), va_all (n), pg_non_slack (g), qg_all (g_total)]

    Scalers:
        'x'  : MinMaxScaler for input (pd, qd)
        'vm' : MinMaxScaler for vm_all (all buses)
        'va' : MinMaxScaler for va_all (all buses)
        'pg' : MinMaxScaler for pg_non_slack
        'qg' : MinMaxScaler for qg_all (all generators)

    raw_data: unscaled arrays for evaluation
    """
    # Extract case_name
    base_filename = os.path.basename(data_path)
    if base_filename.endswith('_pd.csv'):
        case_name = base_filename[:-7]
    elif base_filename.endswith('_qd.csv'):
        case_name = base_filename[:-7]
    elif base_filename.endswith('_pg.csv'):
        case_name = base_filename[:-7]
    else:
        case_name = base_filename.rsplit('_', 1)[0]

    data_dir = os.path.dirname(data_path)

    # Load all CSV files
    pd_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_pd.csv"))
    qd_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_qd.csv"))
    pg_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_pg.csv"))
    qg_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_qg.csv"))
    vm_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_vm.csv"))
    va_df = pd.read_csv(os.path.join(data_dir, f"{case_name}_va.csv"))

    cost_baseline = compute_cost_baseline_from_data(data_dir, case_name, params)

    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    bus_ids = params['general']['bus_ids']
    gen_bus_ids = params['general']['gen_bus_ids']
    non_slack_gen_idx = params['general']['non_slack_gen_idx']

    # ---- Input X: (pd, qd) ----
    sorted_pd_cols = sorted([c for c in pd_df.columns if c.startswith('pd')], key=_extract_id)
    sorted_qd_cols = sorted([c for c in qd_df.columns if c.startswith('qd')], key=_extract_id)
    load_bus_ids = [_extract_id(col) for col in sorted_pd_cols]
    n_loads = len(load_bus_ids)

    if params['general']['n_loads'] != n_loads:
        print(f"⚠️ Updating n_loads: {params['general']['n_loads']} → {n_loads}")
        params['general']['n_loads'] = n_loads
        params['general']['load_bus_ids'] = np.array(load_bus_ids)

    x_pd_raw = pd_df[sorted_pd_cols].values.astype('float32')
    x_qd_raw = qd_df[sorted_qd_cols].values.astype('float32')
    x_data_raw = np.hstack([x_pd_raw, x_qd_raw])

    print(f"(data_setup) Input X: {x_data_raw.shape} (pd:{n_loads}, qd:{n_loads})")

    # ---- Output Y components ----
    # Vm for all buses
    vm_cols_ordered = [f"vm_{bid}" for bid in bus_ids]
    y_vm_raw = vm_df[vm_cols_ordered].values.astype('float32')

    # Va for all buses (radians in dataset)
    va_cols_ordered = [f"va_{bid}" for bid in bus_ids]
    y_va_raw = va_df[va_cols_ordered].values.astype('float32')

    # Pg for all generators, then extract non-slack
    pg_cols = sorted([c for c in pg_df.columns if c.startswith('pg')], key=_extract_id)
    y_pg_raw_all = pg_df[pg_cols].values.astype('float32')
    y_pg_raw_non_slack = y_pg_raw_all[:, non_slack_gen_idx]

    # Qg for all generators
    qg_cols = sorted([c for c in qg_df.columns if c.startswith('qg')], key=_extract_id)
    y_qg_raw = qg_df[qg_cols].values.astype('float32')

    print(f"(data_setup) Output Y components: vm_all={y_vm_raw.shape[1]}, "
          f"va_all={y_va_raw.shape[1]}, pg_non_slack={y_pg_raw_non_slack.shape[1]}, "
          f"qg_all={y_qg_raw.shape[1]}")

    # ---- Scaling ----
    if fit_scalers:
        scalers = {
            'x': MinMaxScaler(),
            'vm': MinMaxScaler(),
            'va': MinMaxScaler(),
            'pg': MinMaxScaler(),
            'qg': MinMaxScaler(),
        }
        x_scaled = scalers['x'].fit_transform(x_data_raw)
        vm_scaled = scalers['vm'].fit_transform(y_vm_raw)
        va_scaled = scalers['va'].fit_transform(y_va_raw)
        pg_scaled = scalers['pg'].fit_transform(y_pg_raw_non_slack)
        qg_scaled = scalers['qg'].fit_transform(y_qg_raw)
        print(f"(data_setup) ✓ Scalers fitted")
    else:
        if scalers is None:
            raise ValueError("scalers required when fit_scalers=False")
        x_scaled = scalers['x'].transform(x_data_raw)
        vm_scaled = scalers['vm'].transform(y_vm_raw)
        va_scaled = scalers['va'].transform(y_va_raw)
        pg_scaled = scalers['pg'].transform(y_pg_raw_non_slack)
        qg_scaled = scalers['qg'].transform(y_qg_raw)
        print(f"(data_setup) ✓ Using existing scalers")

    # Concatenate Y: [vm_all, va_all, pg_non_slack, qg_all]
    y_scaled = np.hstack([vm_scaled, va_scaled, pg_scaled, qg_scaled])

    print(f"(data_setup) Final Y dimension: {y_scaled.shape}  "
          f"(vm:{n_buses} + va:{n_buses} + pg:{n_gen_non_slack} + qg:{n_gen})")

    # Also store the Vm for generator buses only (for evaluation interface compatibility)
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])

    raw_data = {
        'x': x_data_raw,
        'pg': y_pg_raw_all,             # All gen Pg (for evaluation)
        'pg_non_slack': y_pg_raw_non_slack,
        'qg': y_qg_raw,                 # All gen Qg
        'vm': y_vm_raw,                 # All bus Vm
        'vm_gen': y_vm_raw[:, gen_bus_indices],  # Gen bus Vm (for eval compatibility)
        'va': y_va_raw,                 # All bus Va (radians)
    }

    return x_scaled, y_scaled, scalers, raw_data, cost_baseline