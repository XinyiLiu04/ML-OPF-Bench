# -*- coding: utf-8 -*-
"""Shared pieces of the DeepOPF-NGT family (Huang, Chen & Low, IEEE TPWRS 2024).

All four variants use the same network and the same map from the sigmoid output to
physical voltage; they differ only in how the loss terms are weighted and whether any
labelled data is used.
"""

import numpy as np
import torch
import torch.nn as nn
import os
import sys

# ac_configuration/ sits in ac_methods/. Appending the parent of this script's own
# directory makes it importable whether this file is directly in ac_methods/ or one
# level down in a grouped method folder, and from any working directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ac_configuration.acopf_evaluation_metrics import evaluate_acopf_predictions
from ac_configuration.acopf_pypower import solve_pf_setpoints

# How the metrics in a returned dict were obtained. The algebraic variants never run an
# iterative power flow, so a convergence rate is not defined for them and must not be
# compared against the Newton-Raphson rates the other benchmark methods report.
SOLVER_ALGEBRAIC = 'algebraic'
SOLVER_NEWTON = 'newton_raphson'

LOSS_KEYS = ('L_obj', 'L_g', 'L_Sl', 'L_theta', 'L_z', 'L_d')


class DeepOPFNGT(nn.Module):
    """Loads to voltage: ReLU hidden layers and a sigmoid output in (0, 1)."""

    def __init__(self, input_size, output_size, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, output_size), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VoltageDenormaliser:
    """Maps the sigmoid output onto physical voltage at the non-ZIB buses, and back.

    Magnitudes are mapped linearly onto each bus's own [vm_min, vm_max], so the network
    cannot violate a voltage bound at a predicted bus by construction. Angles are mapped
    onto a symmetric window; the paper uses 30 degrees. Note the window bounds each bus
    angle, not the branch angle difference, which can still reach twice the window and
    is what the angle loss term penalizes.
    """

    def __init__(self, params, device, theta_max_deg=30.0):
        nonzib_indices = np.where(~np.asarray(params['general']['zib_mask'], dtype=bool))[0]
        self.nonzib_indices = nonzib_indices
        self.n_nonzib = len(nonzib_indices)
        self.theta_max_rad = float(theta_max_deg) * np.pi / 180.0

        vm_min = np.asarray(params['bus']['vm_min'], dtype=np.float32)[nonzib_indices]
        vm_max = np.asarray(params['bus']['vm_max'], dtype=np.float32)[nonzib_indices]
        self.v_min = torch.tensor(vm_min, device=device).unsqueeze(0)
        self.v_max = torch.tensor(vm_max, device=device).unsqueeze(0)
        self._v_min_np = vm_min
        self._v_max_np = vm_max

    @property
    def output_dim(self):
        return 2 * self.n_nonzib

    def __call__(self, y_norm):
        """Return (v_alpha, theta_alpha) in p.u. and radians."""
        n = self.n_nonzib
        v_alpha = self.v_min + y_norm[:, :n] * (self.v_max - self.v_min)
        theta_alpha = (y_norm[:, n:] - 0.5) * 2.0 * self.theta_max_rad
        return v_alpha, theta_alpha

    def encode(self, vm_nonzib, va_nonzib):
        """Inverse map, turning ground-truth voltage into a target in (0, 1).

        Used by the semi-supervised variants when the supervised loss is taken in the
        normalized domain rather than the physical one.
        """
        v_scaled = ((vm_nonzib - self._v_min_np)
                    / (self._v_max_np - self._v_min_np + 1e-8))
        theta_scaled = va_nonzib / (2.0 * self.theta_max_rad) + 0.5
        return np.hstack([v_scaled, theta_scaled]).astype('float32')


class LossTerms:
    """Evaluates Eqs. (3) to (9) for a batch, holding the limit tensors on device."""

    def __init__(self, params, pf_engine, device, theta_max_deg=30.0):
        self.device = device
        self.pf = pf_engine
        self.theta_diff_max = float(theta_max_deg) * np.pi / 180.0

        def _t(a):
            return torch.tensor(np.asarray(a, dtype=np.float32), device=device)

        gen_p = params['generator']
        self.pg_min = _t(gen_p['pg_min'].flatten()).unsqueeze(0)
        self.pg_max = _t(gen_p['pg_max'].flatten()).unsqueeze(0)
        self.qg_min = _t(gen_p['qg_min'].flatten()).unsqueeze(0)
        self.qg_max = _t(gen_p['qg_max'].flatten()).unsqueeze(0)
        self.c2 = _t(gen_p['cost_c2']).unsqueeze(0)
        self.c1 = _t(gen_p['cost_c1']).unsqueeze(0)
        self.c0 = _t(gen_p['cost_c0']).unsqueeze(0)

        zib_mask = np.asarray(params['general']['zib_mask'], dtype=bool)
        self.zib_idx = torch.tensor(np.where(zib_mask)[0], dtype=torch.long, device=device)
        vm_min = np.asarray(params['bus']['vm_min'], dtype=np.float32)
        vm_max = np.asarray(params['bus']['vm_max'], dtype=np.float32)
        self.vm_min_zib = _t(vm_min[zib_mask]).unsqueeze(0)
        self.vm_max_zib = _t(vm_max[zib_mask]).unsqueeze(0)
        self.vm_min_all = _t(vm_min).unsqueeze(0)
        self.vm_max_all = _t(vm_max).unsqueeze(0)

        # Branches with no usable rating are excluded from the thermal term
        rate_a = np.asarray(params['branch']['rate_a'], dtype=np.float64)
        usable = np.isfinite(rate_a) & (rate_a > 0)
        self.br_mask = torch.tensor(usable, device=device)
        self.rate_a = _t(rate_a[usable]).unsqueeze(0)
        self.any_branch_limit = bool(usable.any())

    def __call__(self, results):
        """Return the six loss terms as scalars on the graph."""
        Pg = results['Pg']
        Qg = results['Qg']
        v_all = results['v_all']
        theta_all = results['theta_all']

        # Eq. (3): generation cost of the reconstructed dispatch
        L_obj = torch.mean(torch.sum(
            self.c2 * Pg ** 2 + self.c1 * Pg + self.c0, dim=1))

        # Eq. (5): generator capacity
        L_g = torch.mean(torch.sum(
            torch.relu(Pg - self.pg_max) ** 2 + torch.relu(self.pg_min - Pg) ** 2
            + torch.relu(Qg - self.qg_max) ** 2 + torch.relu(self.qg_min - Qg) ** 2,
            dim=1))

        # Eq. (6): thermal limits, taken at the more heavily loaded end
        if self.any_branch_limit:
            s_from = results['P_branch'] ** 2 + results['Q_branch'] ** 2
            s_to = results['P_branch_to'] ** 2 + results['Q_branch_to'] ** 2
            s_mag = torch.sqrt(torch.maximum(s_from, s_to) + 1e-12)
            L_Sl = torch.mean(torch.sum(
                torch.relu(s_mag[:, self.br_mask] - self.rate_a) ** 2, dim=1))
        else:
            L_Sl = torch.zeros((), device=self.device)

        # Eq. (7): branch angle differences. Each bus angle is already inside the
        # denormalisation window, but a difference can reach twice that window.
        theta_diff = (theta_all[:, self.pf.f_idx] - theta_all[:, self.pf.t_idx])
        L_theta = torch.mean(torch.sum(
            torch.relu(torch.abs(theta_diff) - self.theta_diff_max) ** 2, dim=1))

        # Eq. (8): voltage magnitude at the ZIBs. The predicted buses are inside their
        # bounds by construction, so only the solved buses can violate them.
        if len(self.zib_idx) > 0:
            v_zib = v_all[:, self.zib_idx]
            L_z = torch.mean(torch.sum(
                torch.relu(v_zib - self.vm_max_zib) ** 2
                + torch.relu(self.vm_min_zib - v_zib) ** 2, dim=1))
        else:
            L_z = torch.zeros((), device=self.device)

        # Eq. (9): load satisfaction at the buses where the voltage solution alone
        # fixes the delivered load
        L_d = torch.mean(torch.sum(
            (results['Pd_pred'] - results['Pd_demanded']) ** 2
            + (results['Qd_pred'] - results['Qd_demanded']) ** 2, dim=1))

        return {'L_obj': L_obj, 'L_g': L_g, 'L_Sl': L_Sl,
                'L_theta': L_theta, 'L_z': L_z, 'L_d': L_d}


def evaluate_algebraic(model, denorm, pf_engine, X, indices, raw_data, params, device):
    """Predict, reconstruct algebraically, and score against the ground truth."""
    n_loads = params['general']['n_loads']

    model.eval()
    with torch.no_grad():
        v_alpha, theta_alpha = denorm(model(X.to(device)))

        x_raw = raw_data['x'][indices]
        Pd = torch.tensor(x_raw[:, :n_loads], dtype=torch.float32, device=device)
        Qd = torch.tensor(x_raw[:, n_loads:], dtype=torch.float32, device=device)
        results = pf_engine(v_alpha, theta_alpha, Pd, Qd)

    Pg = results['Pg'].cpu().numpy()
    Qg = results['Qg'].cpu().numpy()
    v_all = results['v_all'].cpu().numpy()
    theta_all = results['theta_all'].cpu().numpy()
    pf = results['P_branch'].cpu().numpy()
    qf = results['Q_branch'].cpu().numpy()
    pt = results['P_branch_to'].cpu().numpy()
    qt = results['Q_branch_to'].cpu().numpy()

    pf_results_list = [
        build_algebraic_pf_result(Pg[i], Qg[i], v_all[i], theta_all[i],
                                  pf[i], qf[i], pt[i], qt[i], params)
        for i in range(len(Pg))
    ]

    return evaluate_acopf_predictions(
        Pg, v_all,
        raw_data['pg'][indices], raw_data['vm'][indices],
        raw_data['qg'][indices], raw_data['va'][indices],
        pf_results_list, [True] * len(Pg), params, verbose=False)


def update_coefficients_eq12(coeffs, loss_vals, k_upper):
    """Paper Eq. (12): k_i = min(k_obj * L_obj / L_i, k_bar_i), modified in place.

    abs(L_obj) is used because a cost curve with a large negative constant term would
    otherwise flip the sign of every weight. A constraint whose loss has reached zero
    is already satisfied, so its weight is parked at the upper bound rather than
    dividing by zero.
    """
    L_obj_abs = abs(loss_vals['L_obj'])
    k_obj = coeffs['k_obj']

    for name, loss_key in (('k_g', 'L_g'), ('k_Sl', 'L_Sl'), ('k_theta', 'L_theta'),
                           ('k_z', 'L_z'), ('k_d', 'L_d')):
        Li = loss_vals[loss_key]
        new_k = k_obj * L_obj_abs / Li if Li > 1e-12 else k_upper[name]
        coeffs[name] = float(min(new_k, k_upper[name]))


def weighted_total(loss_dict, coeffs):
    """Paper Eq. (10): the weighted sum of the objective and constraint terms."""
    return (coeffs['k_obj'] * loss_dict['L_obj']
            + coeffs['k_g'] * loss_dict['L_g']
            + coeffs['k_Sl'] * loss_dict['L_Sl']
            + coeffs['k_theta'] * loss_dict['L_theta']
            + coeffs['k_z'] * loss_dict['L_z']
            + coeffs['k_d'] * loss_dict['L_d'])


def unweighted_total(loss_dict):
    """Sum of every loss term with unit weights.

    Model selection needs a score that means the same thing at every epoch. The training
    objective does not qualify, because the adaptive weights change underneath it, so a
    later epoch can score lower purely because its weights shrank. This fixed-weight sum
    is used for the validation score instead.
    """
    return sum(float(loss_dict[k]) for k in LOSS_KEYS)


def build_algebraic_pf_result(Pg, Qg, v_all, theta_all, pf, qf, pt, qt, params):
    """Pack an algebraic solution into the result layout the metrics module reads.

    The metrics module is written against PyPower output, so the algebraic solution is
    placed into the same matrix positions. Only the fields the module actually reads are
    filled. The convergence flag has no meaning here and callers must report the solver
    as algebraic rather than treating this as a converged power flow.
    """
    base_mva = params['general']['BASE_MVA']
    n_gen = params['general']['n_gen']
    n_buses = params['general']['n_buses']
    n_branches = params['general']['n_branches']

    gen = np.zeros((n_gen, 21))
    gen[:, 1] = Pg * base_mva
    gen[:, 2] = Qg * base_mva
    gen[:, 3] = params['generator']['qg_max'].flatten() * base_mva
    gen[:, 4] = params['generator']['qg_min'].flatten() * base_mva
    gen[:, 8] = params['generator']['pg_max'].flatten() * base_mva
    gen[:, 9] = params['generator']['pg_min'].flatten() * base_mva

    bus = np.zeros((n_buses, 13))
    bus[:, 7] = v_all
    bus[:, 8] = theta_all * 180.0 / np.pi
    bus[:, 11] = params['bus']['vm_max']
    bus[:, 12] = params['bus']['vm_min']

    # Both ends are filled with their own flows, so the metrics module sees the same
    # asymmetry a real power flow would produce
    branch = np.zeros((n_branches, 17))
    branch[:, 5] = np.asarray(params['branch']['rate_a']) * base_mva
    branch[:, 13] = pf * base_mva
    branch[:, 14] = qf * base_mva
    branch[:, 15] = pt * base_mva
    branch[:, 16] = qt * base_mva

    return ({'success': True, 'gen': gen, 'bus': bus, 'branch': branch},)


def print_metrics_block(metrics, solver, case_name, train_time, latency_ms,
                        extra_lines=()):
    """Print the metric block, stating which solver produced the numbers."""
    print(f"\n{'=' * 70}")
    print(f"Final Results Summary")
    print(f"{'=' * 70}")
    print(f"\nCase: {case_name}")

    print(f"\n--- Accuracy Metrics ---")
    print(f"MAE_Pg (Non-Slack): {metrics['mae_pg_non_slack_percent']:.4f}%")
    print(f"MAE_Pg (All Gens):  {metrics['mae_pg_all_percent']:.4f}%")
    print(f"MAE_Vm (Generator): {metrics['mae_vm_percent']:.4f}%")
    print(f"MAE_Qg (All Gens):  {metrics['mae_qg_percent']:.4f}%")
    print(f"MAE_Va (All Buses): {metrics['mae_va_deg']:.4f} degrees")

    print(f"\n--- Violations (p.u.) ---")
    print(f"Pg_viol (Non-Slack): {metrics['mean_pg_viol_non_slack_pu']:.6f} p.u.")
    print(f"Pg_viol (Slack):     {metrics['mean_pg_viol_slack_pu']:.6f} p.u.")
    print(f"Qg_viol (All Gens):  {metrics['mean_max_qg_viol_pu']:.6f} p.u.")
    print(f"Vm_viol (All Buses): {metrics['mean_max_vm_viol_pu']:.6f} p.u.")
    print(f"Branch_viol:         {metrics['mean_max_branch_viol_pu']:.6f} p.u. "
          f"(1.0 = 100% overload)")

    print(f"\n--- Cost Metrics ---")
    print(f"Cost Gap: {metrics['cost_optimality_gap_percent']:.4f}%")

    print(f"\n--- Solver ---")
    if solver == SOLVER_ALGEBRAIC:
        print(f"Convergence Rate: n/a (algebraic solution, no iterative power flow;")
        print(f"                  not comparable with Newton-Raphson rates)")
    else:
        print(f"Convergence Rate: {metrics['convergence_rate_percent']:.2f}% "
              f"({metrics['n_converged']}/{metrics['n_samples']}) via Newton-Raphson")

    print(f"\n--- Performance ---")
    print(f"Inference Time: {latency_ms:.4f} ms/sample")
    print(f"Training Time:  {train_time:.2f} s")
    for line in extra_lines:
        print(line)
    print(f"{'=' * 70}")


class LossTermsClampedCost(LossTerms):
    """Loss variant that reshapes two terms to keep gradients pointing at feasibility.

    L_obj evaluates the cost on Pg clamped into its bounds, plus a linear term outside
    them using the marginal cost at the boundary. Evaluating the raw quadratic instead
    lets the optimizer discover that a strongly negative Pg has a low cost, so it drifts
    into an infeasible low-cost regime; clamping alone removes the gradient out there,
    and the linear term restores a push back toward the feasible range.

    L_z covers every bus rather than only the zero-injection ones. The predicted buses
    sit inside their bounds by construction, so they contribute nothing, but including
    them makes the term independent of how the reduction splits the network.
    """

    def _cost_and_bound_terms(self, Pg):
        Pg_clipped = torch.clamp(Pg, self.pg_min, self.pg_max)
        cost = torch.sum(
            self.c2 * Pg_clipped ** 2 + self.c1 * Pg_clipped + self.c0, dim=1)

        mc_min = 2.0 * self.c2 * self.pg_min + self.c1
        mc_max = 2.0 * self.c2 * self.pg_max + self.c1
        extra = torch.sum(
            mc_min * torch.relu(self.pg_min - Pg)
            + mc_max * torch.relu(Pg - self.pg_max), dim=1)

        return torch.mean(cost + extra)

    def __call__(self, results):
        loss_dict = super().__call__(results)
        loss_dict['L_obj'] = self._cost_and_bound_terms(results['Pg'])

        v_all = results['v_all']
        vm_viol = (torch.relu(self.vm_min_all - v_all)
                   + torch.relu(v_all - self.vm_max_all))
        loss_dict['L_z'] = torch.mean(torch.sum(vm_viol ** 2, dim=1))
        return loss_dict


class AdaptiveWeightScheduler:
    """EMA-smoothed constraint weights over auto-normalized losses, in three regimes.

    Every loss is divided by its own epoch-one magnitude, so the weights compare terms
    that are all near 1.0 at the start regardless of physical units. The normalized
    values are then smoothed, which stops one bad epoch from ratcheting a weight up
    permanently. Each weight then falls into one of three regimes:

      satisfied  (ema < satisfied_thresh): decay back toward the initial weight, so the
                 cost term can regain influence once a constraint stops binding
      normal:    the Eq. (12) ratio k_i = k_obj * ema_obj / ema_i
      violated   (ema >= boost_thresh): a boost proportional to the violation, so a
                 constraint that stays far from satisfied cannot sit at the floor

    The two thresholds exist to stop the symmetric failure modes of the plain ratio: a
    weight locking at its ceiling once its loss approaches zero, and a weight stuck at
    its floor while its loss stays large.
    """

    def __init__(self, k_obj, initial, upper, lower, ema_alpha=0.3,
                 satisfied_thresh=0.02, boost_thresh=0.30,
                 decay=0.95, boost_base=50.0):
        self.coeffs = {'k_obj': k_obj}
        self.coeffs.update(initial)
        self.initial = dict(initial)
        self.upper = dict(upper)
        self.lower = dict(lower)

        self.ema_alpha = ema_alpha
        self.satisfied_thresh = satisfied_thresh
        self.boost_thresh = boost_thresh
        self.decay = decay
        self.boost_base = boost_base

        self.refs = {k: 1.0 for k in LOSS_KEYS}
        self.ema = {k: None for k in LOSS_KEYS}
        self._weight_to_loss = {'k_g': 'L_g', 'k_Sl': 'L_Sl', 'k_theta': 'L_theta',
                                'k_z': 'L_z', 'k_d': 'L_d'}

    def normalise(self, loss_dict):
        """Divide each term by its reference magnitude."""
        return {k: loss_dict[k] / self.refs[k] for k in LOSS_KEYS}

    def set_references(self, epoch_means, cost_baseline=None):
        """Fix the normalization scales from the first epoch; never revisited.

        The cost reference comes from the dataset's mean optimal cost when available,
        not from epoch one: at random initialization the voltages are meaningless, so
        the reconstructed cost is not a sensible scale to normalize against.
        """
        if cost_baseline and cost_baseline > 1e-3:
            self.refs['L_obj'] = float(cost_baseline)
        else:
            raw = epoch_means['L_obj']
            self.refs['L_obj'] = float(raw) if abs(raw) > 1e-8 else 1.0

        for k in LOSS_KEYS:
            if k == 'L_obj':
                continue
            raw = epoch_means[k]
            self.refs[k] = float(raw) if raw > 1e-8 else 1.0

        for k in LOSS_KEYS:
            self.ema[k] = 1.0

    def update(self, epoch_means):
        """Advance the EMA and reassign every constraint weight."""
        for k in LOSS_KEYS:
            norm = epoch_means[k] / self.refs[k]
            self.ema[k] = self.ema_alpha * norm + (1.0 - self.ema_alpha) * self.ema[k]

        ema_obj = self.ema['L_obj']
        for wk, lk in self._weight_to_loss.items():
            ema_i = self.ema[lk]
            lo, hi = self.lower[wk], self.upper[wk]

            if ema_i < self.satisfied_thresh:
                self.coeffs[wk] = float(max(self.coeffs[wk] * self.decay,
                                            self.initial[wk]))
            elif ema_i >= self.boost_thresh:
                self.coeffs[wk] = float(np.clip(self.boost_base * ema_i, lo, hi))
            elif ema_i > 1e-8:
                self.coeffs[wk] = float(
                    np.clip(self.coeffs['k_obj'] * ema_obj / ema_i, lo, hi))

    def describe(self):
        return (f"k_g={self.coeffs['k_g']:.1f}, k_Sl={self.coeffs['k_Sl']:.1f}, "
                f"k_th={self.coeffs['k_theta']:.1f}, k_z={self.coeffs['k_z']:.1f}, "
                f"k_d={self.coeffs['k_d']:.1f}")

    def describe_ema(self):
        if self.ema['L_obj'] is None:
            return ""
        return ("  EMA obj:{:.4f} g:{:.4f} Sl:{:.4f} th:{:.4f} z:{:.4f} d:{:.4f}"
                .format(*(self.ema[k] for k in LOSS_KEYS)))


def evaluate_with_power_flow(model, denorm, pf_engine, X, indices, raw_data, params,
                             case_data, device, verbose=True):
    """Reconstruct algebraically, then verify each sample with a real power flow.

    The algebraic solution supplies the setpoints, non-slack Pg and generator Vm, and
    PyPower solves the network from there. The convergence rate this produces is a real
    Newton-Raphson rate and is comparable with the other benchmark methods.
    """
    n_loads = params['general']['n_loads']
    n_gen = params['general']['n_gen']
    n_buses = params['general']['n_buses']
    non_slack = params['general']['non_slack_gen_idx']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    gen_bus_indices = np.array(
        [bus_id_to_idx[int(g)] for g in params['general']['gen_bus_ids']])

    model.eval()
    with torch.no_grad():
        v_alpha, theta_alpha = denorm(model(X.to(device)))
        x_raw = raw_data['x'][indices]
        Pd = torch.tensor(x_raw[:, :n_loads], dtype=torch.float32, device=device)
        Qd = torch.tensor(x_raw[:, n_loads:], dtype=torch.float32, device=device)
        results = pf_engine(v_alpha, theta_alpha, Pd, Qd)

    Pg_alg = results['Pg'].cpu().numpy()
    v_all = results['v_all'].cpu().numpy()
    pd_pu = x_raw[:, :n_loads]
    qd_pu = x_raw[:, n_loads:]

    n_samples = len(Pg_alg)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"  Verifying {n_samples} samples with a power flow...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_setpoints(
                pd_pu[i], qd_pu[i], Pg_alg[i][non_slack],
                v_all[i][gen_bus_indices], params, case_data)
            pf_results_list.append(r1_pf)
            converge_flags.append(r1_pf[0]['success'])
        except Exception:
            pf_results_list.append((
                {'success': False,
                 'gen': np.zeros((n_gen, 21)),
                 'bus': np.zeros((n_buses, 13)),
                 'branch': np.zeros((1, 17))},
            ))
            converge_flags.append(False)

    if verbose:
        print(f"  Converged: {sum(converge_flags)}/{n_samples}")

    return evaluate_acopf_predictions(
        Pg_alg, v_all,
        raw_data['pg'][indices], raw_data['vm'][indices],
        raw_data['qg'][indices], raw_data['va'][indices],
        pf_results_list, converge_flags, params, verbose=False)