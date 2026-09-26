# -*- coding: utf-8 -*-
"""Shared PyPower helpers: solver options and case construction from the constraint CSVs."""

import numpy as np
import pandas as pd
from pathlib import Path

from pypower.ppoption import ppoption
from pypower.runpf import runpf

_PPOPT = None
_PPOPT_OPF = None


def get_ppopt():
    """PyPower options for silent power flow runs without Q-limit enforcement."""
    global _PPOPT
    if _PPOPT is None:
        _PPOPT = ppoption(ppoption(), OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=0)
    return _PPOPT


def get_ppopt_opf():
    """PyPower options for OPF solves; Q limits and flow limits are enforced here."""
    global _PPOPT_OPF
    if _PPOPT_OPF is None:
        _PPOPT_OPF = ppoption(ppoption(), OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=1,
                              OPF_FLOW_LIM=0)
    return _PPOPT_OPF


def load_case_from_csv(case_name, constraints_path):
    """Assemble a PyPower case dict from the constraint CSVs.

    Matrices are filled in p.u. and converted to MW/MVAr at the end, so the
    unrated-branch sentinel is applied in a single unit domain.
    """
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
    bus[:, 8] = np.rad2deg(bus_df['va_rad'].values)
    bus[:, 9] = bus_df['base_kv'].values
    bus[:, 10] = 1
    bus[:, 11] = bus_df['vmax_pu'].values
    bus[:, 12] = bus_df['vmin_pu'].values
    bus[:, 4] = bus_df['gs_pu'].values
    bus[:, 5] = bus_df['bs_pu'].values

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
    branch[:, 0] = branch_df['f_bus'].values
    branch[:, 1] = branch_df['t_bus'].values
    branch[:, 2] = branch_df['r_pu'].values
    branch[:, 3] = branch_df['x_pu'].values
    branch[:, 4] = branch_df['b_pu'].values
    branch[:, 5] = branch_df['rate_a_pu'].values
    branch[:, 6] = branch[:, 5]
    branch[:, 7] = branch[:, 5]
    branch[:, 8] = branch_df['tap_ratio'].values
    branch[:, 9] = np.rad2deg(branch_df['shift_rad'].values)
    branch[:, 10] = 1
    branch[:, 11] = np.rad2deg(branch_df['angmin_rad'].values)
    branch[:, 12] = np.rad2deg(branch_df['angmax_rad'].values)

    # PyPower uses a zero rating for unconstrained branches.
    rate_a_values = branch_df['rate_a_pu'].values
    branch[:, 5:8][~np.isfinite(rate_a_values), :] = 0.0

    gencost = np.zeros((len(gen_df), 7))
    gencost[:, 0] = 2
    gencost[:, 3] = 3
    gencost[:, 4] = gen_df['cost_c2'].values / baseMVA ** 2
    gencost[:, 5] = gen_df['cost_c1'].values / baseMVA
    gencost[:, 6] = gen_df['cost_c0'].values

    ppc = {'version': '2', 'baseMVA': baseMVA, 'bus': bus, 'gen': gen,
           'branch': branch, 'gencost': gencost}

    ppc['bus'][:, 2] *= baseMVA
    ppc['bus'][:, 3] *= baseMVA
    ppc['bus'][:, 4] *= baseMVA
    ppc['bus'][:, 5] *= baseMVA
    ppc['gen'][:, 3] *= baseMVA
    ppc['gen'][:, 4] *= baseMVA
    ppc['gen'][:, 8] *= baseMVA
    ppc['gen'][:, 9] *= baseMVA

    ppc['branch'][:, 5:8] *= baseMVA
    return ppc


def build_mpc_for_sample(pd, qd, pg_non_slack, params, case_data):
    """Copy the base case and apply one sample's loads and non-slack Pg setpoints.

    Slack Pg is deliberately left untouched: the power flow determines it, which is
    how the slack generator absorbs whatever imbalance the prediction leaves behind.
    """
    BASE_MVA = params['general']['BASE_MVA']

    mpc = {
        'version': case_data['version'],
        'baseMVA': case_data['baseMVA'],
        'bus': case_data['bus'].copy(),
        'gen': case_data['gen'].copy(),
        'branch': case_data['branch'].copy(),
        'gencost': case_data['gencost'],
    }

    bus_id_to_idx = params['general']['bus_id_to_idx']
    for i, bus_id in enumerate(params['general']['load_bus_ids']):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc['bus'][bus_idx, 2] = pd[i] * BASE_MVA
            mpc['bus'][bus_idx, 3] = qd[i] * BASE_MVA

    for i, gen_idx in enumerate(params['general']['non_slack_gen_idx']):
        mpc['gen'][gen_idx, 1] = pg_non_slack[i] * BASE_MVA

    return mpc


def solve_pf_setpoints(pd, qd, pg_non_slack, vm_gen, params, case_data):
    """Run a power flow at the predicted non-slack Pg and generator Vm setpoints."""
    mpc = build_mpc_for_sample(pd, qd, pg_non_slack, params, case_data)

    for i in range(params['general']['n_gen']):
        mpc['gen'][i, 5] = vm_gen[i]

    return runpf(mpc, get_ppopt())