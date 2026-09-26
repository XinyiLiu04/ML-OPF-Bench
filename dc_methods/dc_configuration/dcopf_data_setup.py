"""Constraint and sample loading, the shared train/val/test split, and slack reconstruction."""

from ml_opf_bench.runtime import training_indices

import os
import re

import numpy as np
import pandas as pd

# Must match the generator's `f_max .< 1e10`, or evaluation and the reference solutions
# disagree about which branches are constrained. NaN compares False and is left unconstrained,
# exactly as in Julia.
UNRATED_THRESHOLD = 1e10

_LOAD_COLUMN = re.compile(r"^pd(\d+)$")
_GEN_COLUMN = re.compile(r"^pg(\d+)$")


def _read(params_path, case_name, name, **kwargs):
    return pd.read_csv(os.path.join(params_path, f"{case_name}_{name}.csv"), **kwargs)


def load_parameters_from_csv(case_name, params_path):
    """Load the exported DC network and derive the slack bus and slack generators."""
    gen_limits = _read(params_path, case_name, 'gen_limits')
    gen_costs = _read(params_path, case_name, 'gen_costs')
    branch_limits = _read(params_path, case_name, 'branch_limits')
    bus_ids = _read(params_path, case_name, 'bus_ids')['bus_id'].to_numpy(dtype=int)
    base_mva = float(_read(params_path, case_name, 'base_mva')['value'].iloc[0])
    # Both dense matrices carry auto-generated x1..xN headers; their columns follow bus_ids.csv
    # (resp. gen_limits.csv) order, not the CSV row position of any per-bus file.
    ptdf = _read(params_path, case_name, 'ptdf_matrix').to_numpy(dtype=np.float64)
    gen_bus_map = _read(params_path, case_name, 'bus_gen_map').to_numpy(dtype=np.float64)

    gen_ids = gen_limits['gen_id'].to_numpy(dtype=int)
    branch_ids = branch_limits['branch_id'].to_numpy(dtype=int)
    n_buses, n_gens, n_branches = len(bus_ids), len(gen_ids), len(branch_ids)

    if ptdf.shape != (n_branches, n_buses):
        raise ValueError(f"PTDF shape {ptdf.shape} != (n_branches, n_buses) = ({n_branches}, {n_buses})")
    if gen_bus_map.shape != (n_buses, n_gens):
        raise ValueError(f"bus_gen_map shape {gen_bus_map.shape} != (n_buses, n_gens) = ({n_buses}, {n_gens})")
    if not np.array_equal(gen_costs['gen_id'].to_numpy(dtype=int), gen_ids):
        raise ValueError("gen_costs and gen_limits list generators in a different order")
    if not np.array_equal(gen_bus_map.sum(axis=0), np.ones(n_gens)):
        raise ValueError("Every bus_gen_map column must contain exactly one 1")

    rate_a = branch_limits['rate_a'].to_numpy(dtype=np.float64)
    constrained = rate_a < UNRATED_THRESHOLD
    if np.any(rate_a[constrained] <= 0):
        raise ValueError("Constrained branches with a nonpositive rate_a would make relative violations undefined")

    # The exporter writes the slack column of the PTDF as exact zeros, and in a connected network
    # no other bus has an all-zero column
    zero_columns = np.flatnonzero(~ptdf.any(axis=0))
    if len(zero_columns) != 1:
        raise ValueError(f"Expected exactly one all-zero PTDF column (the slack bus), found {len(zero_columns)}")
    slack_bus_idx = int(zero_columns[0])
    slack_gen_idx = np.flatnonzero(gen_bus_map[slack_bus_idx] > 0)
    if len(slack_gen_idx) == 0:
        raise ValueError(f"No generator at slack bus {bus_ids[slack_bus_idx]}; power balance cannot be closed")
    non_slack_gen_idx = np.flatnonzero(gen_bus_map[slack_bus_idx] == 0)

    print(f"Loaded {case_name}: {n_buses} buses, {n_gens} generators, "
          f"{n_branches} branches ({int(constrained.sum())} constrained)")
    print(f"Slack bus {bus_ids[slack_bus_idx]}, slack generators {gen_ids[slack_gen_idx].tolist()}")

    return {
        'general': {
            'case_name': case_name,
            'n_buses': n_buses,
            'n_gens': n_gens,
            'n_branches': n_branches,
            'bus_ids': bus_ids,
            'bus_id_to_idx': {int(b): i for i, b in enumerate(bus_ids)},
            'gen_ids': gen_ids,
            'branch_ids': branch_ids,
            'base_mva': base_mva,
            'slack_bus_idx': slack_bus_idx,
            'slack_gen_idx': slack_gen_idx,
            'non_slack_gen_idx': non_slack_gen_idx,
        },
        'constraints': {
            'pg_min': gen_limits['pgmin'].to_numpy(dtype=np.float64),
            'pg_max': gen_limits['pgmax'].to_numpy(dtype=np.float64),
            'cost_c2': gen_costs['cost_c2'].to_numpy(dtype=np.float64),
            'cost_c1': gen_costs['cost_c1'].to_numpy(dtype=np.float64),
            'cost_c0': gen_costs['cost_c0'].to_numpy(dtype=np.float64),
            'rate_a': rate_a,
            'constrained_branches': constrained,
            'ptdf': ptdf,                # (n_branches, n_buses)
            'gen_bus_map': gen_bus_map,  # (n_buses, n_gens)
        },
    }


def load_samples(data_path, params):
    """Return per-bus loads (n_samples, n_buses) and dispatch (n_samples, n_gens), both in p.u."""
    df = pd.read_csv(data_path)
    general = params['general']

    pd_bus = np.zeros((len(df), general['n_buses']))
    load_columns = [c for c in df.columns if _LOAD_COLUMN.match(c)]
    if not load_columns:
        raise ValueError(f"No pd<bus_id> columns in {data_path}")
    for col in load_columns:
        bus_id = int(_LOAD_COLUMN.match(col).group(1))
        if bus_id not in general['bus_id_to_idx']:
            raise ValueError(f"Load column {col} refers to a bus absent from bus_ids.csv; "
                             f"dataset and constraints are probably from different cases")
        pd_bus[:, general['bus_id_to_idx'][bus_id]] = df[col].to_numpy(dtype=np.float64)

    # Columns are named by gen_id and selected in gen_limits order, never by sorting the names
    # (lexicographic order puts pg10 before pg2)
    csv_gen_ids = {int(_GEN_COLUMN.match(c).group(1)) for c in df.columns if _GEN_COLUMN.match(c)}
    if csv_gen_ids != set(general['gen_ids'].tolist()):
        raise ValueError(f"Generator ids in {data_path} do not match gen_limits.csv: "
                         f"only in dataset {sorted(csv_gen_ids - set(general['gen_ids']))}, "
                         f"only in constraints {sorted(set(general['gen_ids']) - csv_gen_ids)}")
    pg = df[[f"pg{g}" for g in general['gen_ids']]].to_numpy(dtype=np.float64)

    print(f"Loaded {len(df)} samples, {len(load_columns)} load buses")
    return pd_bus, pg


def prepare_data_splits(n_total, n_train_use, seed):
    """Use the run-scoped cross-system or fixed-heldout scaling split."""
    return training_indices(n_total, n_train_use, seed)


def reconstruct_full_pg(pg_non_slack, pd_bus, params):
    """Close the power balance on the slack generators, splitting the residual evenly among them."""
    general = params['general']
    pg = np.zeros((len(pg_non_slack), general['n_gens']))
    pg[:, general['non_slack_gen_idx']] = pg_non_slack
    residual = pd_bus.sum(axis=1) - pg_non_slack.sum(axis=1)
    pg[:, general['slack_gen_idx']] = (residual / len(general['slack_gen_idx']))[:, None]
    return pg


def load_duals(data_path, params):
    """Power-balance, generator-bound and line-limit multipliers, aligned with gen_ids and constrained branches.

    The generator writes the inequality multipliers already sign-normalized, so each is nonnegative
    and "active" means "> threshold" for all four families; no negation is needed, unlike the AC
    duals. lambda follows JuMP's convention, +d(cost)/d(load). Line multipliers exist only for
    constrained branches, in branch_limits order.
    """
    general, c = params['general'], params['constraints']
    gen_ids = general['gen_ids']
    branch_ids = general['branch_ids'][c['constrained_branches']]
    columns = {
        'lambda': ['lambda'],
        'mu_g_min': [f"mu_g_min_{g}" for g in gen_ids],
        'mu_g_max': [f"mu_g_max_{g}" for g in gen_ids],
        'mu_line_max': [f"mu_line_max_{b}" for b in branch_ids],
        'mu_line_min': [f"mu_line_min_{b}" for b in branch_ids],
    }
    wanted = [col for cols in columns.values() for col in cols]
    available = set(pd.read_csv(data_path, nrows=0).columns)
    missing = [col for col in wanted if col not in available]
    if missing:
        raise ValueError(f"{len(missing)} dual columns missing from {data_path}, e.g. {missing[:5]}")
    df = pd.read_csv(data_path, usecols=wanted)
    return {name: df[cols].to_numpy(dtype=np.float64) for name, cols in columns.items()}