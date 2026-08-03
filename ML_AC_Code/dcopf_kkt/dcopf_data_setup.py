"""
dcopf_data_setup.py

"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from enum import Enum


class DataSplitMode(Enum):
    RANDOM_SPLIT = "random_split"
    VALID_FIXED = "valid_fixed"
    GENERALIZATION = "generalization"
    API_TEST = "api_test"


def load_parameters_from_csv(case_name, params_path, is_api=False):
    DTYPE = 'float32'

    suffix = "__api" if is_api else ""

    gen_limits = pd.read_csv(os.path.join(params_path, f"{case_name}{suffix}_gen_limits.csv"))
    gen_costs = pd.read_csv(os.path.join(params_path, f"{case_name}{suffix}_gen_costs.csv"))
    branch_limits = pd.read_csv(os.path.join(params_path, f"{case_name}{suffix}_branch_limits.csv"))
    ptdf_matrix = pd.read_csv(os.path.join(params_path, f"{case_name}{suffix}_ptdf_matrix.csv"),
                              header=0).values.astype(DTYPE)
    bus_gen_map = pd.read_csv(os.path.join(params_path, f"{case_name}{suffix}_bus_gen_map.csv"),
                              header=0).values.astype(DTYPE)
    base_mva_df = pd.read_csv(os.path.join(params_path, f"{case_name}{suffix}_base_mva.csv"))
    BASE_MVA = base_mva_df['value'].iloc[0]

    ptdf_T = ptdf_matrix.T
    bus_gen_map_T = bus_gen_map.T

    # ---- Load bus_id mapping (critical for case300 with non-contiguous bus IDs) ----
    bus_ids_path = os.path.join(params_path, f"{case_name}{suffix}_bus_ids.csv")
    bus_ids_df = pd.read_csv(bus_ids_path)
    bus_ids = bus_ids_df['bus_id'].values.astype(int)  # e.g. [1,2,...,250,281,...,9533]
    bus_id_to_idx = {int(bid): i for i, bid in enumerate(bus_ids)}

    simulation_parameters = {
        'general': {
            'n_buses': ptdf_T.shape[0],
            'n_line': ptdf_T.shape[1],
            'n_g': len(gen_limits),
            'n_d': ptdf_T.shape[0],
            'g_bus': gen_limits['gen_id'].values,
            'l_bus': bus_ids,
            'bus_ids': bus_ids,
            'bus_id_to_idx': bus_id_to_idx,
            'branch_ids': branch_limits['branch_id'].values,
            'BASE_MVA': BASE_MVA,
        },
        'constraints': {
            'Pg_min': gen_limits['pgmin'].values.reshape(1, -1).astype(DTYPE),
            'Pg_max': gen_limits['pgmax'].values.reshape(1, -1).astype(DTYPE),
            'Pg_max_real': gen_limits['pgmax'].values.reshape(1, -1).astype(DTYPE),
            'C_Pg': gen_costs['cost_c1'].values.astype(DTYPE),
            'C_Pg_c2': gen_costs['cost_c2'].values.astype(DTYPE),
            'C_Pg_c0': gen_costs['cost_c0'].values.astype(DTYPE),
            'Pl_max': branch_limits['rate_a'].values.astype(DTYPE),
            'PTDF': ptdf_T,
            'Map_g': bus_gen_map_T,
            'Map_L': np.eye(ptdf_T.shape[0], dtype=DTYPE)
        }
    }
    return simulation_parameters


def load_and_scale_data(file_path, params, column_names):

    full_df = pd.read_csv(file_path)

    n_buses = params['general']['n_buses']
    x_data_raw = np.zeros((len(full_df), n_buses), dtype='float32')

    load_prefix = column_names['load_prefix']
    load_cols = [col for col in full_df.columns if col.startswith(load_prefix)]
    bus_id_to_idx = params['general']['bus_id_to_idx']
    for col_name in load_cols:
        load_id = int(col_name[len(load_prefix):])
        if load_id in bus_id_to_idx:
            x_data_raw[:, bus_id_to_idx[load_id]] = full_df[col_name].values
        else:
            print(f"[WARNING] load bus {load_id} not in bus_id_to_idx, skipping")

    pg_cols = [f"{column_names['gen_prefix']}{i}" for i in params['general']['g_bus']]
    y_pg_raw = full_df[pg_cols].values

    y_lambda_raw = full_df[column_names['lambda']].values.reshape(-1, 1)

    mu_g_min_cols = [f"{column_names['mu_g_min_prefix']}{i}" for i in params['general']['g_bus']]
    y_mu_g_min_raw = full_df[mu_g_min_cols].values

    mu_g_max_cols = [f"{column_names['mu_g_max_prefix']}{i}" for i in params['general']['g_bus']]
    y_mu_g_max_raw = full_df[mu_g_max_cols].values

    valid_branch_indices = np.where(params['constraints']['Pl_max'] < 1e10)[0]
    valid_branch_ids = params['general']['branch_ids'][valid_branch_indices]

    # Use the configured names for line limit duals
    mu_line_pos_cols = [f"{column_names['mu_line_pos_prefix']}{i}" for i in valid_branch_ids]
    y_mu_line_pos_raw = full_df[mu_line_pos_cols].values

    mu_line_neg_cols = [f"{column_names['mu_line_neg_prefix']}{i}" for i in valid_branch_ids]
    y_mu_line_neg_raw = full_df[mu_line_neg_cols].values

    # Initialize and apply scalers
    x_scaler = MinMaxScaler()
    scalers = {
        "pg": MinMaxScaler(), "lambda": MinMaxScaler(), "mu_g_min": MinMaxScaler(),
        "mu_g_max": MinMaxScaler(), "mu_line_pos": MinMaxScaler(),
        "mu_line_neg": MinMaxScaler(), "x": x_scaler
    }

    x_data_scaled = x_scaler.fit_transform(x_data_raw)
    y_pg_scaled = scalers['pg'].fit_transform(y_pg_raw)
    y_lambda_scaled = scalers['lambda'].fit_transform(y_lambda_raw)
    y_mu_g_min_scaled = scalers['mu_g_min'].fit_transform(y_mu_g_min_raw)
    y_mu_g_max_scaled = scalers['mu_g_max'].fit_transform(y_mu_g_max_raw)
    y_mu_line_pos_scaled = scalers['mu_line_pos'].fit_transform(y_mu_line_pos_raw)
    y_mu_line_neg_scaled = scalers['mu_line_neg'].fit_transform(y_mu_line_neg_raw)

    y_data_scaled = (
        y_pg_scaled, y_lambda_scaled, y_mu_g_min_scaled,
        y_mu_g_max_scaled, y_mu_line_pos_scaled, y_mu_line_neg_scaled
    )

    y_physics = np.zeros((len(x_data_scaled), 1), dtype='float32')
    y_training = y_data_scaled + (y_physics,)

    Lg_Max = [np.max(np.abs(y)) for y in (
        y_lambda_raw, y_mu_g_max_raw, y_mu_g_min_raw,
        y_mu_line_pos_raw, y_mu_line_neg_raw
    )]

    return x_data_scaled, y_training, Lg_Max, scalers, y_pg_raw, x_data_raw


def split_data_by_mode(
        x_data_raw: np.ndarray,
        y_pg_raw: np.ndarray,
        mode: DataSplitMode,
        n_train_use: int = None,
        seed: int = 42,
        test_data_path: str = None,
        params: dict = None,
        column_names: dict = None,
        n_test_samples: int = 1000
):

    rng = np.random.default_rng(seed)
    n_total = len(x_data_raw)

    if mode == DataSplitMode.RANDOM_SPLIT:
        n_use = n_train_use
        print(f"[random_split]")

        all_indices = rng.permutation(n_total)
        pool_indices = all_indices[:n_use]

        n_val = int(n_use * 1 / 12)
        n_test = int(n_use * 1 / 12)
        if n_val == 0 and n_use >= 12:
            n_val = 1
        if n_test == 0 and n_use >= 12:
            n_test = 1
        n_train = n_use - n_val - n_test

        if (n_train + n_val + n_test) < n_use:
            n_test = n_use - n_train - n_val

        train_idx = pool_indices[:n_train]
        val_idx = pool_indices[n_train: n_train + n_val]
        test_idx = pool_indices[n_train + n_val:]

        print(f"  Train: {len(train_idx)} ({len(train_idx) / n_use * 100:.1f}%)")
        print(f"  Val:   {len(val_idx)} ({len(val_idx) / n_use * 100:.1f}%)")
        print(f"  Test:  {len(test_idx)} ({len(test_idx) / n_use * 100:.1f}%)")

        return train_idx, val_idx, test_idx, None, None

    elif mode == DataSplitMode.VALID_FIXED:
        print(f"[valid_fixed]")

        fixed_rng = np.random.default_rng(seed)
        all_indices = fixed_rng.permutation(n_total)

        n_val_fixed = int(n_total * 0.1)
        n_test_fixed = int(n_total * 0.1)
        n_train_pool = n_total - n_val_fixed - n_test_fixed

        val_idx = all_indices[:n_val_fixed]
        test_idx = all_indices[n_val_fixed: n_val_fixed + n_test_fixed]
        train_pool_idx = all_indices[n_val_fixed + n_test_fixed:]

        print(f" Val:  {len(val_idx)} (10%)")
        print(f" Test: {len(test_idx)} (10%)")
        print(f" Training pool:    {len(train_pool_idx)} (80%)")

        n_use_train = n_train_use
        train_idx = rng.choice(train_pool_idx, size=n_use_train, replace=False)

        return train_idx, val_idx, test_idx, None, None

    elif mode == DataSplitMode.GENERALIZATION:
        print(f"[generalization]")

        n_use = n_train_use

        all_indices = rng.permutation(n_total)
        pool_indices = all_indices[:n_use]

        n_val = int(n_use * 1 / 12)
        n_test_internal = int(n_use * 1 / 12)
        if n_val == 0 and n_use >= 12:
            n_val = 1

        n_train = n_use - n_val - n_test_internal
        if (n_train + n_val + n_test_internal) < n_use:
            n_test_internal = n_use - n_train - n_val

        train_idx = pool_indices[:n_train]
        val_idx = pool_indices[n_train: n_train + n_val]

        print(f"  Train (v=0.12): {len(train_idx)} ({len(train_idx) / n_use * 100:.1f}%)")
        print(f"  Val (v=0.12):   {len(val_idx)} ({len(val_idx) / n_use * 100:.1f}%)")

        x_test_raw_full, y_test_pg_raw_full = load_and_prepare_data_generalization(
            test_data_path, params, column_names
        )

        n_available = len(x_test_raw_full)
        n_test_actual = min(n_test_samples, n_available)

        if n_test_actual < n_available:
            test_sample_indices = rng.choice(n_available, n_test_actual, replace=False)
            x_test_raw = x_test_raw_full[test_sample_indices]
            y_test_pg_raw = y_test_pg_raw_full[test_sample_indices]
        else:
            x_test_raw = x_test_raw_full
            y_test_pg_raw = y_test_pg_raw_full

        print(f"  Test (v=0.25):  {len(x_test_raw)}")

        test_idx = None

        return train_idx, val_idx, test_idx, x_test_raw, y_test_pg_raw

    elif mode == DataSplitMode.API_TEST:

        print(f"[api_test]")

        if n_train_use is None or n_train_use > n_total:
            n_use = n_total
        else:
            n_use = n_train_use

        all_indices = rng.permutation(n_total)
        pool_indices = all_indices[:n_use]

        n_val = int(n_use * 1 / 12)
        n_test_internal = int(n_use * 1 / 12)
        if n_val == 0 and n_use >= 12:
            n_val = 1

        n_train = n_use - n_val - n_test_internal
        if (n_train + n_val + n_test_internal) < n_use:
            n_test_internal = n_use - n_train - n_val

        train_idx = pool_indices[:n_train]
        val_idx = pool_indices[n_train: n_train + n_val]

        print(f"  Train (v=0.12): {len(train_idx)} ({len(train_idx) / n_use * 100:.1f}%)")
        print(f"  Val (v=0.12):   {len(val_idx)} ({len(val_idx) / n_use * 100:.1f}%)")

        x_test_raw_full, y_test_pg_raw_full = load_and_prepare_data_generalization(
            test_data_path, params, column_names
        )
        n_available = len(x_test_raw_full)
        n_test_actual = min(n_test_samples, n_available)

        if n_test_actual < n_available:
            test_sample_indices = rng.choice(n_available, n_test_actual, replace=False)
            x_test_raw = x_test_raw_full[test_sample_indices]
            y_test_pg_raw = y_test_pg_raw_full[test_sample_indices]
        else:
            x_test_raw = x_test_raw_full
            y_test_pg_raw = y_test_pg_raw_full

        print(f"  Test (API):     {len(x_test_raw)}")
        test_idx = None

        return train_idx, val_idx, test_idx, x_test_raw, y_test_pg_raw

    else:
        raise ValueError(f"Unknown: {mode}")


def load_and_prepare_data_generalization(file_path, params, column_names):

    full_df = pd.read_csv(file_path)
    n_samples = len(full_df)
    n_buses = params['general']['n_buses']

    load_prefix = column_names['load_prefix']
    load_cols = [col for col in full_df.columns if col.startswith(load_prefix)]

    load_bus_ids = []
    for col in load_cols:
        bus_id = int(col[len(load_prefix):])
        load_bus_ids.append(bus_id)

    x_data_raw_loads = full_df[load_cols].values.astype('float32')

    x_data_raw_full = np.zeros((n_samples, n_buses), dtype='float32')
    bus_id_to_idx = params['general']['bus_id_to_idx']
    for i, bus_id in enumerate(load_bus_ids):
        if bus_id in bus_id_to_idx:
            x_data_raw_full[:, bus_id_to_idx[bus_id]] = x_data_raw_loads[:, i]
        else:
            print(f"[WARNING] load bus {bus_id} not in bus_id_to_idx, skipping")

    g_bus = params['general']['g_bus']
    pg_cols = [f"{column_names['gen_prefix']}{int(gen_id)}" for gen_id in g_bus]
    y_pg_raw = full_df[pg_cols].values.astype('float32')

    return x_data_raw_full, y_pg_raw


if __name__ == "__main__":


    # 1. Case and Path Configuration
    CASE_NAME = 'pglib_opf_case14_ieee'
    CASE_SHORT_NAME = 'case14'
    ROOT_DIR = r"C:\Users\Aloha\Desktop\dataset"

    PARAMS_PATH = os.path.join(ROOT_DIR, "DCOPF Constraints", "case14")
    DATA_PATH = os.path.join(ROOT_DIR, "DCOPF dataset", 'case14(v=0.12)', f"{CASE_NAME}_dataset_with_duals.csv")

    # 2. Dataset Column Name Mapping
    COLUMN_NAMES = {
        'load_prefix': 'pd',
        'gen_prefix': 'pg',
        'lambda': 'lambda',
        'mu_g_min_prefix': 'mu_g_min_',
        'mu_g_max_prefix': 'mu_g_max_',
        'mu_line_pos_prefix': 'mu_line_max_',
        'mu_line_neg_prefix': 'mu_line_min_',
    }


    # Load parameters
    simulation_params = load_parameters_from_csv(
        case_name=CASE_NAME,
        params_path=PARAMS_PATH
    )

    # Load and scale data using the configurations
    x_train, y_train, Lg_Max, scalers, y_pg_raw, x_raw = load_and_scale_data(
        file_path=DATA_PATH,
        params=simulation_params,
        column_names=COLUMN_NAMES
    )