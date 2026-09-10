# -*- coding: utf-8 -*-
"""DeepOPF for ACOPF: MSE loss plus a zero-order estimated constraint penalty,
with optional warm-start OPF recovery for infeasible predictions."""

import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Function
import multiprocessing as mp

from pypower.runopf import runopf
from pypower.idx_bus import VMAX, VMIN, VM
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, PMAX, PMIN, GEN_STATUS
from pypower.idx_brch import RATE_A, PF, QF, PT, QT

# The shared modules live in ac_configuration/, a subpackage of this script's
# directory, so they resolve regardless of the working directory.
try:
    from ac_configuration import acopf_config
    from ac_configuration.acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        prepare_data_splits,
        reconstruct_full_pg
    )
    from ac_configuration.acopf_evaluation_metrics import evaluate_acopf_predictions
    from ac_configuration.acopf_pypower import (
        get_ppopt,
        get_ppopt_opf,
        load_case_from_csv,
        solve_pf_setpoints
    )
except ImportError as e:
    print(f"Error: Unable to import from ac_configuration/ ({e})")
    sys.exit(1)

GLOBAL_PARAMS = {}
GLOBAL_SCALERS = {}
GLOBAL_CASE_DATA = None
GLOBAL_POOL = None

# The penalty layer runs a power flow per sample, so it is only evaluated on
# every penalty_freq-th step; this flag tells the layer whether this is one.
COMPUTE_PENALTY_THIS_STEP = False

WORKER_PARAMS = None
WORKER_CASE_DATA = None


def init_worker_globals(final_params, case_name, params_path):
    """Rebuild the case data inside each worker; PyPower cases are not picklable cheaply."""
    global WORKER_PARAMS, WORKER_CASE_DATA
    WORKER_PARAMS = final_params
    WORKER_CASE_DATA = load_case_from_csv(case_name, params_path)


def init_global_pool(n_cores, final_params, case_name, params_path):
    """Start the worker pool used to parallelize the penalty power flows."""
    global GLOBAL_POOL
    if GLOBAL_POOL is None:
        try:
            ctx = mp.get_context('spawn')
        except RuntimeError:
            ctx = mp.get_context()

        GLOBAL_POOL = ctx.Pool(
            n_cores,
            initializer=init_worker_globals,
            initargs=(final_params, case_name, params_path)
        )
        print(f"Initialized process pool with {n_cores} workers")


def close_global_pool():
    """Shut the worker pool down; leaving it open keeps the interpreter alive."""
    global GLOBAL_POOL
    if GLOBAL_POOL is not None:
        GLOBAL_POOL.close()
        GLOBAL_POOL.join()
        GLOBAL_POOL = None


def check_acopf_feasibility(pf_result, params, tol=1e-4):
    """Return (is_feasible, max_violation) against all ACOPF inequality constraints."""
    if not pf_result[0]['success']:
        return False, float('inf')

    res = pf_result[0]
    BASE_MVA = params['general']['BASE_MVA']
    max_viol = 0.0

    on_idx = np.where(res['gen'][:, GEN_STATUS] == 1)[0]
    if len(on_idx):
        pg_pu = res['gen'][on_idx, PG] / BASE_MVA
        max_viol = max(
            max_viol,
            np.maximum(0, res['gen'][on_idx, PMIN] / BASE_MVA - pg_pu - tol).max(),
            np.maximum(0, pg_pu - res['gen'][on_idx, PMAX] / BASE_MVA - tol).max()
        )

        qg_pu = res['gen'][on_idx, QG] / BASE_MVA
        max_viol = max(
            max_viol,
            np.maximum(0, res['gen'][on_idx, QMIN] / BASE_MVA - qg_pu - tol).max(),
            np.maximum(0, qg_pu - res['gen'][on_idx, QMAX] / BASE_MVA - tol).max()
        )

    vm = res['bus'][:, VM]
    max_viol = max(
        max_viol,
        np.maximum(0, res['bus'][:, VMIN] - vm - tol).max(),
        np.maximum(0, vm - res['bus'][:, VMAX] - tol).max()
    )

    # The 9900 sentinel marks unrated branches, matching the threshold the
    # evaluation module uses so both agree on which branches are constrained
    rate_a = res['branch'][:, RATE_A]
    active_br = np.where((rate_a > 0) & (rate_a < 9000))[0]
    if len(active_br):
        sf = np.abs(res['branch'][active_br, PF] + 1j * res['branch'][active_br, QF])
        st = np.abs(res['branch'][active_br, PT] + 1j * res['branch'][active_br, QT])
        br_viol = np.maximum(
            0, np.maximum(sf, st) - rate_a[active_br] - tol * BASE_MVA).max() / BASE_MVA
        max_viol = max(max_viol, br_viol)

    return max_viol <= 0, max_viol


def run_acopf_warm_start(pd_pu, qd_pu, pg_non_slack_warm, vm_gen_warm, params):
    """Solve the ACOPF from the predicted setpoints as an initial point."""
    global GLOBAL_CASE_DATA
    BASE_MVA = params['general']['BASE_MVA']
    n_gen = params['general']['n_gen']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    mpc = {
        'version': GLOBAL_CASE_DATA['version'],
        'baseMVA': GLOBAL_CASE_DATA['baseMVA'],
        'bus': GLOBAL_CASE_DATA['bus'].copy(),
        'gen': GLOBAL_CASE_DATA['gen'].copy(),
        'branch': GLOBAL_CASE_DATA['branch'].copy(),
        'gencost': GLOBAL_CASE_DATA['gencost']
    }

    for i, bus_id in enumerate(params['general']['load_bus_ids']):
        bus_idx = bus_id_to_idx.get(int(bus_id))
        if bus_idx is not None:
            mpc['bus'][bus_idx, 2] = pd_pu[i] * BASE_MVA
            mpc['bus'][bus_idx, 3] = qd_pu[i] * BASE_MVA

    # The initial point is clipped into the feasible box; an out-of-bounds
    # start would just be a worse starting guess for the solver
    for i, gen_idx in enumerate(params['general']['non_slack_gen_idx']):
        mpc['gen'][gen_idx, PG] = float(np.clip(
            pg_non_slack_warm[i] * BASE_MVA,
            mpc['gen'][gen_idx, PMIN],
            mpc['gen'][gen_idx, PMAX]
        ))

    gen_bus_indices = np.array(
        [bus_id_to_idx[int(gid)] for gid in params['general']['gen_bus_ids']])
    for i in range(n_gen):
        bus_idx = gen_bus_indices[i]
        mpc['gen'][i, VG] = float(np.clip(
            vm_gen_warm[i],
            mpc['bus'][bus_idx, VMIN],
            mpc['bus'][bus_idx, VMAX]
        ))

    try:
        return (runopf(mpc, get_ppopt_opf()),)
    except Exception:
        return ({'success': False},)


def deepopf_post_process(pf_result, pd_pu, qd_pu, pg_non_slack, vm_gen, params, tol=1e-4):
    """Keep a feasible power flow as is; otherwise try to recover it with a warm-started OPF.

    Returns (final_pf_result, was_recovery_attempted, recovery_succeeded).
    """
    is_feasible, _ = check_acopf_feasibility(pf_result, params, tol)
    if is_feasible:
        return pf_result, False, True

    try:
        opf_result = run_acopf_warm_start(pd_pu, qd_pu, pg_non_slack, vm_gen, params)
        if opf_result[0].get('success', False):
            return opf_result, True, True
        return pf_result, True, False
    except Exception:
        return pf_result, True, False


def zero_order_penalty_abs(pf_results):
    """Constraint penalty of one power flow, following formula (14) of Pan et al. 2023.

    Each category is averaged over its own count and the categories are then summed:
    branch flows over |E|, PQ-bus voltages over |D|, PV-generator Qg over |G|, plus
    the slack generator's Pg and Qg. Only reconstructed quantities are penalized;
    non-slack Pg is a direct network output and is bounded by the output layer.
    """
    standval = pf_results[0]["baseMVA"]
    ctol = 1e-4

    res = pf_results[0]
    gen = res["gen"]
    bus = res["bus"]
    branch = res["branch"]

    slack_bus_ids = set(bus[bus[:, 1] == 3, 0].astype(int))
    pq_bus_ids = set(bus[bus[:, 1] == 1, 0].astype(int))

    On_index = np.where(gen[:, 7] == 1)[0]
    gen_bus_ids_on = gen[On_index, 0].astype(int)

    slack_gen_mask = np.array([int(bid) in slack_bus_ids for bid in gen_bus_ids_on])
    pq_gen_mask = np.array([int(bid) in pq_bus_ids for bid in gen_bus_ids_on])

    # Slack generator Pg and Qg
    slack_gen_idx = On_index[slack_gen_mask]
    Pg_slack_penalty = 0.0
    Qg_slack_penalty = 0.0
    for gi in slack_gen_idx:
        pg_lo = (gen[gi, 9] - gen[gi, 1]) / standval
        pg_hi = (gen[gi, 1] - gen[gi, 8]) / standval
        Pg_slack_penalty += (pg_lo if pg_lo >= ctol else 0.0) + (pg_hi if pg_hi >= ctol else 0.0)

        qg_lo = (gen[gi, 4] - gen[gi, 2]) / standval
        qg_hi = (gen[gi, 2] - gen[gi, 3]) / standval
        Qg_slack_penalty += (qg_lo if qg_lo >= ctol else 0.0) + (qg_hi if qg_hi >= ctol else 0.0)

    # PV generator Qg, averaged over |G|
    pv_gen_idx = On_index[~pq_gen_mask & ~slack_gen_mask]
    Qg_pv_penalty = 0.0
    if len(pv_gen_idx) > 0:
        Qg_lo = (gen[pv_gen_idx, 4] - gen[pv_gen_idx, 2]) / standval
        Qg_lo[Qg_lo < ctol] = 0
        Qg_hi = (gen[pv_gen_idx, 2] - gen[pv_gen_idx, 3]) / standval
        Qg_hi[Qg_hi < ctol] = 0
        Qg_pv_penalty = np.sum(Qg_lo + Qg_hi) / len(pv_gen_idx)

    # PQ bus voltages, averaged over |D|
    PQ_index = np.where(bus[:, 1] == 1)[0]
    V_penalty = 0.0
    if len(PQ_index) > 0:
        V_lo = bus[PQ_index, 12] - bus[PQ_index, 7]
        V_lo[V_lo < ctol] = 0
        V_hi = bus[PQ_index, 7] - bus[PQ_index, 11]
        V_hi[V_hi < ctol] = 0
        V_penalty = np.sum(V_lo + V_hi) / len(PQ_index)

    # Branch flows, averaged over |E|
    Ff = np.abs(branch[:, 13] + 1j * branch[:, 14])
    Ft = np.abs(branch[:, 15] + 1j * branch[:, 16])
    Branch_index = np.where(branch[:, 5] != 0)[0]
    Branch_penalty = 0.0
    if len(Branch_index) > 0:
        Branch_bound = branch[Branch_index, 5]
        Ff_temp = (Ff[Branch_index] - Branch_bound) / standval
        Ff_temp[Ff_temp < ctol] = 0
        Ft_temp = (Ft[Branch_index] - Branch_bound) / standval
        Ft_temp[Ft_temp < ctol] = 0
        Branch_penalty = np.sum(Ff_temp + Ft_temp) / len(Branch_index)

    return (Branch_penalty + V_penalty + Qg_pv_penalty
            + Pg_slack_penalty + Qg_slack_penalty)


def compute_penalty_worker_mp(args):
    """Worker-side penalty for one sample; a failed power flow gets the maximum penalty."""
    pd, qd, pg_non_slack, vm_gen = args

    if WORKER_CASE_DATA is None or WORKER_PARAMS is None:
        return 1.0

    try:
        r1_pf = solve_pf_setpoints(pd, qd, pg_non_slack, vm_gen,
                                   WORKER_PARAMS, WORKER_CASE_DATA)
        return zero_order_penalty_abs(r1_pf) if r1_pf[0]['success'] else 1.0
    except Exception:
        return 1.0


def penalties_for_batch(pd, qd, pg_non_slack, vm_gen, params):
    """Penalty per sample, run in the worker pool when one is available."""
    batch_size = len(pd)

    if GLOBAL_POOL is None:
        penalties = np.zeros(batch_size)
        for i in range(batch_size):
            r1_pf = solve_pf_setpoints(pd[i], qd[i], pg_non_slack[i], vm_gen[i],
                                       params, GLOBAL_CASE_DATA)
            penalties[i] = zero_order_penalty_abs(r1_pf) if r1_pf[0]['success'] else 1.0
        return penalties

    args_list = [(pd[i], qd[i], pg_non_slack[i], vm_gen[i]) for i in range(batch_size)]
    return np.array(GLOBAL_POOL.map(compute_penalty_worker_mp, args_list))


class Penalty_ACPF_Optimized(Function):
    """Constraint penalty whose gradient is estimated by a two-point directional probe."""

    @staticmethod
    def forward(ctx, nn_output_scaled, x_input_scaled):
        ctx.save_for_backward(nn_output_scaled, x_input_scaled)

        global COMPUTE_PENALTY_THIS_STEP
        if not COMPUTE_PENALTY_THIS_STEP:
            ctx.skip_penalty = True
            return torch.tensor(0.0, dtype=torch.float32, device=nn_output_scaled.device)
        ctx.skip_penalty = False

        nn_output_np = nn_output_scaled.cpu().detach().numpy()
        x_input_np = x_input_scaled.cpu().detach().numpy()

        params = GLOBAL_PARAMS
        scalers = GLOBAL_SCALERS
        n_gen_non_slack = params['general']['n_gen_non_slack']
        n_loads = params['general']['n_loads']

        y_pred_pg_non_slack = scalers['pg'].inverse_transform(nn_output_np[:, :n_gen_non_slack])
        y_pred_vm_gen = scalers['vm'].inverse_transform(nn_output_np[:, n_gen_non_slack:])

        x_raw = scalers['x'].inverse_transform(x_input_np)
        penalties = penalties_for_batch(
            x_raw[:, :n_loads], x_raw[:, n_loads:],
            y_pred_pg_non_slack, y_pred_vm_gen, params
        )

        return torch.tensor(float(np.mean(penalties)), dtype=torch.float32,
                            device=nn_output_scaled.device)

    @staticmethod
    def backward(ctx, grad_output):
        nn_output_scaled, x_input_scaled = ctx.saved_tensors

        if ctx.skip_penalty:
            return torch.zeros_like(nn_output_scaled), None

        nn_output_np = nn_output_scaled.cpu().detach().numpy()
        x_input_np = x_input_scaled.cpu().detach().numpy()

        batch_size, output_dim = nn_output_np.shape
        params = GLOBAL_PARAMS
        scalers = GLOBAL_SCALERS
        n_gen_non_slack = params['general']['n_gen_non_slack']
        n_loads = params['general']['n_loads']

        # One random unit direction per sample; the penalty is probed at +-h along
        # it and the directional derivative is scaled by output_dim to form an
        # unbiased estimate of the full gradient
        vec = np.random.randn(batch_size, output_dim)
        vector_h = vec / (np.linalg.norm(vec, axis=1).reshape(-1, 1) + 1e-10)
        h = 1e-4

        # Probes are clipped to [0, 1] because the network output is a sigmoid
        nn_output_plus_h = np.clip(nn_output_np + vector_h * h, 0, 1)
        nn_output_minus_h = np.clip(nn_output_np - vector_h * h, 0, 1)

        x_raw = scalers['x'].inverse_transform(x_input_np)
        pd, qd = x_raw[:, :n_loads], x_raw[:, n_loads:]

        penalty_plus = penalties_for_batch(
            pd, qd,
            scalers['pg'].inverse_transform(nn_output_plus_h[:, :n_gen_non_slack]),
            scalers['vm'].inverse_transform(nn_output_plus_h[:, n_gen_non_slack:]),
            params
        )
        penalty_minus = penalties_for_batch(
            pd, qd,
            scalers['pg'].inverse_transform(nn_output_minus_h[:, :n_gen_non_slack]),
            scalers['vm'].inverse_transform(nn_output_minus_h[:, n_gen_non_slack:]),
            params
        )

        directional = ((penalty_plus - penalty_minus) / (2 * h)).reshape(-1, 1)
        gradient_estimate = (directional * vector_h * output_dim).astype('float32')
        final_gradient = gradient_estimate / batch_size

        return torch.from_numpy(final_gradient).to(nn_output_scaled.device) * grad_output, None


class PINN_ACOPF(nn.Module):
    """ReLU network with a sigmoid output, returning both the prediction and its penalty."""

    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self.penalty_layer = Penalty_ACPF_Optimized.apply

    def forward(self, x):
        x_sol = self.net(x)
        x_penalty = self.penalty_layer(x_sol, x)
        return x_sol, x_penalty.to(x_sol.device)


def evaluate_model(model, X, indices, raw_data, params, scalers, device,
                   split_name, verbose=True, apply_post_processing=True):
    """Evaluate the model before and, optionally, after warm-start OPF recovery.

    Returns {'before': metrics, 'after': metrics or None, 'pp_stats': counters,
    'pipeline_latency_ms': float}.
    """
    if verbose:
        print(f"\n{split_name} Evaluation:")

    model.eval()
    with torch.no_grad():
        y_pred_scaled, _ = model(X.to(device))
    y_pred_scaled_np = y_pred_scaled.cpu().numpy()

    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']
    n_buses = params['general']['n_buses']
    n_loads = params['general']['n_loads']
    BASE_MVA = params['general']['BASE_MVA']
    gen_bus_ids = params['general']['gen_bus_ids']
    bus_id_to_idx = params['general']['bus_id_to_idx']

    y_pred_pg_non_slack = scalers['pg'].inverse_transform(
        y_pred_scaled_np[:, :n_gen_non_slack])
    y_pred_vm_gen = scalers['vm'].inverse_transform(
        y_pred_scaled_np[:, n_gen_non_slack:])

    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_non_slack, params)

    # Load buses are unpredicted, so they sit at nominal voltage and are
    # excluded from MAE_Vm downstream
    gen_bus_indices = np.array([bus_id_to_idx[int(gid)] for gid in gen_bus_ids])
    y_pred_vm_all = np.full((len(X), n_buses), 1.0, dtype=y_pred_vm_gen.dtype)
    y_pred_vm_all[:, gen_bus_indices] = y_pred_vm_gen

    y_true_pg = raw_data['pg'][indices]
    y_true_vm = raw_data['vm'][indices]
    y_true_qg = raw_data['qg'][indices]
    y_true_va_rad = raw_data['va'][indices]

    x_raw_data = scalers['x'].inverse_transform(X.cpu().numpy())
    pd_pu = x_raw_data[:, :n_loads]
    qd_pu = x_raw_data[:, n_loads:]

    n_samples = len(X)

    def _dummy():
        return ({'success': False,
                 'gen': np.zeros((n_gen, 21)),
                 'bus': np.zeros((n_buses, 13)),
                 'branch': np.zeros((1, 17))},)

    raw_pf_list, raw_conv_flags = [], []
    final_pf_list, final_conv_flags = [], []
    n_pre_feasible = n_recovered = n_unrecoverable = 0
    pipeline_times = []

    if verbose:
        print(f"  Computing power flow for {n_samples} samples...")
        if apply_post_processing:
            print(f"  Post-processing enabled (feasibility check + warm-start OPF)")

    for i in range(n_samples):
        t0 = time.perf_counter()
        try:
            r1_pf = solve_pf_setpoints(
                pd_pu[i], qd_pu[i], y_pred_pg_non_slack[i], y_pred_vm_gen[i],
                params, GLOBAL_CASE_DATA)
            raw_pf_list.append(r1_pf)
            raw_conv_flags.append(r1_pf[0]['success'])

            if apply_post_processing:
                final_pf, attempted, recovery_ok = deepopf_post_process(
                    r1_pf, pd_pu[i], qd_pu[i],
                    y_pred_pg_non_slack[i], y_pred_vm_gen[i], params)

                if not attempted:
                    n_pre_feasible += 1
                elif recovery_ok:
                    n_recovered += 1
                else:
                    n_unrecoverable += 1

                final_pf_list.append(final_pf)
                final_conv_flags.append(final_pf[0]['success'])
            else:
                final_pf_list.append(r1_pf)
                final_conv_flags.append(r1_pf[0]['success'])

        except Exception:
            raw_pf_list.append(_dummy())
            raw_conv_flags.append(False)
            final_pf_list.append(_dummy())
            final_conv_flags.append(False)
            if apply_post_processing:
                n_unrecoverable += 1

        pipeline_times.append(time.perf_counter() - t0)

    if verbose:
        print(f"  Converged (raw): {sum(raw_conv_flags)}/{n_samples}")
        if apply_post_processing:
            print(f"  Converged (after recovery): {sum(final_conv_flags)}/{n_samples}")
            print(f"  Pre-recovery feasible: {n_pre_feasible}/{n_samples} "
                  f"({n_pre_feasible / n_samples * 100:.1f}%)")
            print(f"  Recovered: {n_recovered}/{n_samples} "
                  f"({n_recovered / n_samples * 100:.1f}%)")
            print(f"  Unrecoverable: {n_unrecoverable}/{n_samples} "
                  f"({n_unrecoverable / n_samples * 100:.1f}%)")

    metrics_before = evaluate_acopf_predictions(
        y_pred_pg=y_pred_pg_full,
        y_pred_vm=y_pred_vm_all,
        y_true_pg=y_true_pg,
        y_true_vm=y_true_vm,
        y_true_qg=y_true_qg,
        y_true_va_rad=y_true_va_rad,
        pf_results_list=raw_pf_list,
        converge_flags=raw_conv_flags,
        params=params,
        verbose=False
    )

    if apply_post_processing:
        pp_pg_full = y_pred_pg_full.copy()
        pp_vm_all = y_pred_vm_all.copy()
        for i in range(n_samples):
            if final_conv_flags[i]:
                pp_pg_full[i, :] = final_pf_list[i][0]['gen'][:, 1] / BASE_MVA
                pp_vm_all[i, :] = final_pf_list[i][0]['bus'][:, 7]

        metrics_after = evaluate_acopf_predictions(
            y_pred_pg=pp_pg_full,
            y_pred_vm=pp_vm_all,
            y_true_pg=y_true_pg,
            y_true_vm=y_true_vm,
            y_true_qg=y_true_qg,
            y_true_va_rad=y_true_va_rad,
            pf_results_list=final_pf_list,
            converge_flags=final_conv_flags,
            params=params,
            verbose=False
        )
    else:
        metrics_after = None

    return {
        'before': metrics_before,
        'after': metrics_after,
        'pp_stats': {
            'pre_recovery_feasibility_rate': n_pre_feasible / n_samples * 100,
            'recovery_success_rate': n_recovered / n_samples * 100,
            'post_recovery_feasibility_rate': (n_pre_feasible + n_recovered) / n_samples * 100,
            'unrecoverable_rate': n_unrecoverable / n_samples * 100,
        },
        'pipeline_latency_ms': float(np.mean(pipeline_times)) * 1000,
    }


def train_pinn_acopf(
        case_name,
        params_path,
        data_path,
        n_train_use=None,
        hidden_sizes=[256, 256],
        n_epochs=100,
        early_stop_patience=20,
        early_stop_min_delta=1e-6,
        batch_size=256,
        learning_rate=1e-3,
        penalty_weight=0.1,
        penalty_freq=1,
        apply_post_processing=True,
        seed=42,
        device='cuda',
        n_cores=30
):
    """Train the PINN on a random split and evaluate it on the held-out test indices."""
    global GLOBAL_PARAMS, GLOBAL_SCALERS, GLOBAL_CASE_DATA, COMPUTE_PENALTY_THIS_STEP

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    device_obj = torch.device(device)

    print(f"\n{'=' * 70}")
    print(f"DeepOPF (PINN) ACOPF Training")
    print(f"{'=' * 70}")
    print(f"Case: {case_name}")
    print(f"Device: {device_obj}")
    print(f"Penalty weight: {penalty_weight}, evaluated every {penalty_freq} steps")
    print(f"Post-processing: {'Enabled' if apply_post_processing else 'Disabled'}")
    print(f"Process pool cores: {n_cores}")
    print(f"{'=' * 70}")

    try:
        # --------------------------------------------------------------
        # 1. Load network parameters and PyPower case data
        # --------------------------------------------------------------
        params = load_parameters_from_csv(case_name, params_path)
        GLOBAL_PARAMS = params
        GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)
        get_ppopt()

        # --------------------------------------------------------------
        # 2. Load dataset and fit scalers
        # --------------------------------------------------------------
        x_data_scaled, y_data_scaled, scalers, raw_data, cost_baseline = \
            load_and_scale_acopf_data(data_path, params, fit_scalers=True)
        GLOBAL_SCALERS = scalers

        n_gen = params['general']['n_gen']
        n_gen_non_slack = params['general']['n_gen_non_slack']
        n_buses = params['general']['n_buses']
        n_loads = params['general']['n_loads']
        baseMVA = params['general']['BASE_MVA']

        print(f"\n[Dataset Info]")
        print(f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_non_slack}), "
              f"Loads: {n_loads}, Base MVA: {baseMVA}")
        if cost_baseline:
            print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

        # --------------------------------------------------------------
        # 3. Worker pool for the penalty power flows
        # --------------------------------------------------------------
        init_global_pool(n_cores, GLOBAL_PARAMS, case_name, params_path)

        # --------------------------------------------------------------
        # 4. Split
        # --------------------------------------------------------------
        train_idx, val_idx, test_idx = prepare_data_splits(
            x_data_scaled, y_data_scaled,
            n_train_use=n_train_use,
            seed=seed
        )

        X_train = torch.from_numpy(x_data_scaled[train_idx]).float().to(device_obj)
        Y_train = torch.from_numpy(y_data_scaled[train_idx]).float().to(device_obj)
        X_val = torch.from_numpy(x_data_scaled[val_idx]).float().to(device_obj)
        Y_val = torch.from_numpy(y_data_scaled[val_idx]).float().to(device_obj)
        X_test = torch.from_numpy(x_data_scaled[test_idx]).float().to(device_obj)

        train_loader = Data.DataLoader(
            dataset=Data.TensorDataset(X_train, Y_train),
            batch_size=batch_size,
            shuffle=True
        )

        # --------------------------------------------------------------
        # 5. Model
        # --------------------------------------------------------------
        input_dim = x_data_scaled.shape[1]
        output_dim = y_data_scaled.shape[1]
        model = PINN_ACOPF(input_dim, output_dim, hidden_sizes).to(device_obj)

        print(f"\n{'=' * 70}")
        print(f"Model Configuration")
        print(f"{'=' * 70}")
        print(f"Input dim: {input_dim} (pd + qd)")
        print(f"Output dim: {output_dim} (pg_non_slack: {n_gen_non_slack} + "
              f"vm_gen: {output_dim - n_gen_non_slack})")
        print(f"Network: {input_dim} -> {' -> '.join(map(str, hidden_sizes))} -> {output_dim}")
        print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Training params: max_epochs={n_epochs}, patience={early_stop_patience}, "
              f"lr={learning_rate}, batch_size={batch_size}")
        print(f"{'=' * 70}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

        # --------------------------------------------------------------
        # 6. Training with early stopping on validation loss
        # --------------------------------------------------------------
        print(f"\n{'=' * 70}")
        print(f"Training Progress")
        print(f"{'=' * 70}")
        print(f"Loss = MSE + {penalty_weight} * Penalty")

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_epoch = 0
        best_state_dict = None
        patience_counter = 0
        global_step = 0
        t0 = time.perf_counter()

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_total = 0.0

            for batch_x, batch_y in train_loader:
                global_step += 1
                COMPUTE_PENALTY_THIS_STEP = (global_step % penalty_freq == 0)

                optimizer.zero_grad()
                pred, penalty = model(batch_x)
                total_loss = criterion(pred, batch_y) + penalty_weight * penalty
                total_loss.backward()
                optimizer.step()
                epoch_total += total_loss.item() * len(batch_x)

            train_losses.append(epoch_total / len(X_train))

            # The penalty is always evaluated on the validation set, so validation
            # loss stays comparable across epochs regardless of penalty_freq
            COMPUTE_PENALTY_THIS_STEP = True
            model.eval()
            with torch.no_grad():
                val_pred, val_penalty = model(X_val)
                val_loss = criterion(val_pred, Y_val).item() + \
                    penalty_weight * val_penalty.item()
            val_losses.append(val_loss)

            print(f"Epoch {epoch:4d}/{n_epochs} - Train Loss: {train_losses[-1]:.6f} - "
                  f"Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss - early_stop_min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch} "
                          f"(patience={early_stop_patience})")
                    break

        train_time = time.perf_counter() - t0
        model.load_state_dict({k: v.to(device_obj) for k, v in best_state_dict.items()})
        print(f"Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
        print(f"Training completed in {train_time:.2f} seconds")

        # --------------------------------------------------------------
        # 7. Evaluation
        # --------------------------------------------------------------
        print(f"\n{'=' * 70}")
        print(f"Test Set Evaluation")
        print(f"{'=' * 70}")

        test_results = evaluate_model(
            model, X_test, test_idx, raw_data, params, scalers, device_obj, "Test",
            verbose=True, apply_post_processing=apply_post_processing
        )

        # --------------------------------------------------------------
        # 8. Inference latency. The penalty layer is skipped so this measures
        #    the network alone; the power flow shows up in pipeline latency.
        # --------------------------------------------------------------
        COMPUTE_PENALTY_THIS_STEP = False
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                model(X_test[:1])

            times = []
            for _ in range(100):
                t_start = time.perf_counter()
                model(X_test[:1])
                if device_obj.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t_start)

        latency_ms = np.mean(times) * 1000

        # --------------------------------------------------------------
        # 9. Results
        # --------------------------------------------------------------
        mb = test_results['before']
        ma = test_results['after']
        pp = test_results['pp_stats']
        pipeline_ms = test_results['pipeline_latency_ms']

        print(f"\n{'=' * 70}")
        print(f"Final Results Summary")
        print(f"{'=' * 70}")
        print(f"\nCase: {case_name}")

        def _fmt(v):
            return f"{v:.6f}" if not np.isnan(v) else "N/A"

        if ma is not None:
            rows = [
                ("MAE_Pg Non-Slack (%)", 'mae_pg_non_slack_percent'),
                ("MAE_Vm Generator (%)", 'mae_vm_percent'),
                ("MAE_Qg All Gens  (%)", 'mae_qg_percent'),
                ("MAE_Va All Buses (deg)", 'mae_va_deg'),
                ("Pg_viol Non-Slack (p.u.)", 'mean_pg_viol_non_slack_pu'),
                ("Pg_viol Slack     (p.u.)", 'mean_pg_viol_slack_pu'),
                ("Qg_viol All Gens  (p.u.)", 'mean_max_qg_viol_pu'),
                ("Vm_viol All Buses (p.u.)", 'mean_max_vm_viol_pu'),
                ("Branch_viol       (p.u.)", 'mean_max_branch_viol_pu'),
                ("Cost Gap          (%)", 'cost_optimality_gap_percent'),
                ("Convergence Rate  (%)", 'convergence_rate_percent'),
            ]
            print(f"\n  {'':28s}  {'Before':>12}   {'After':>12}")
            print(f"  {'-' * 58}")
            for label, key in rows:
                print(f"  {label:<28s}  {_fmt(mb[key]):>12}   {_fmt(ma[key]):>12}")
            print(f"  {'Feasibility Rate  (%)':<28s}"
                  f"  {pp['pre_recovery_feasibility_rate']:12.2f}"
                  f"   {pp['post_recovery_feasibility_rate']:12.2f}")

            print(f"\n--- Post-processing Summary ---")
            print(f"  Pre-recovery feasible:  {pp['pre_recovery_feasibility_rate']:.2f}%")
            print(f"  Successfully recovered: {pp['recovery_success_rate']:.2f}%")
            print(f"  Post-recovery feasible: {pp['post_recovery_feasibility_rate']:.2f}%")
            print(f"  Unrecoverable:          {pp['unrecoverable_rate']:.2f}%")
        else:
            print(f"\n--- Accuracy Metrics ---")
            print(f"MAE_Pg (Non-Slack): {mb['mae_pg_non_slack_percent']:.4f}%")
            print(f"MAE_Vm (Generator): {mb['mae_vm_percent']:.4f}%")
            print(f"MAE_Qg (All Gens):  {mb['mae_qg_percent']:.4f}%")
            print(f"MAE_Va (All Buses): {mb['mae_va_deg']:.4f} degrees")

            print(f"\n--- Violations (p.u.) ---")
            print(f"Pg_viol (Non-Slack): {_fmt(mb['mean_pg_viol_non_slack_pu'])} p.u.")
            print(f"Pg_viol (Slack):     {_fmt(mb['mean_pg_viol_slack_pu'])} p.u.")
            print(f"Qg_viol (All Gens):  {_fmt(mb['mean_max_qg_viol_pu'])} p.u.")
            print(f"Vm_viol (All Buses): {_fmt(mb['mean_max_vm_viol_pu'])} p.u.")
            print(f"Branch_viol:         {_fmt(mb['mean_max_branch_viol_pu'])} p.u. "
                  f"(1.0 = 100% overload)")

            print(f"\n--- Cost Metrics ---")
            print(f"Cost Gap: {mb['cost_optimality_gap_percent']:.4f}%")
            print(f"Convergence Rate: {mb['convergence_rate_percent']:.2f}%")

        print(f"\n--- Performance ---")
        print(f"NN Inference:  {latency_ms:.4f} ms/sample (forward pass only)")
        print(f"Pipeline:      {pipeline_ms:.4f} ms/sample "
              f"(power flow{' + post-processing' if apply_post_processing else ''})")
        print(f"Training Time: {train_time:.2f} s")
        print(f"{'=' * 70}")

        return test_results

    finally:
        # The pool must be closed even if training or evaluation raises, or the
        # worker processes keep the interpreter alive
        close_global_pool()


if __name__ == '__main__':
    PENALTY_WEIGHT = 0.1
    PENALTY_FREQ = 1
    N_CORES = 30
    APPLY_POST_PROCESSING = True

    print("\n" + "=" * 70)
    print("Loading Configuration")
    print("=" * 70)

    params = acopf_config.get_all_params()
    params['penalty_weight'] = PENALTY_WEIGHT
    params['penalty_freq'] = PENALTY_FREQ
    params['apply_post_processing'] = APPLY_POST_PROCESSING
    params['n_cores'] = N_CORES

    results = train_pinn_acopf(**acopf_config.get_all_paths(), **params)

    print("\nExperiment completed successfully!")