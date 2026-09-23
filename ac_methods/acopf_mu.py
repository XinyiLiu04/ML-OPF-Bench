# -*- coding: utf-8 -*-
"""Lagrangian dual ACOPF, model M_C^D of Fioretto et al. 2019.

Four heads predict v, theta, non-slack Pg and Qg. The loss is the sum of their MSEs plus
a weighted sum of nine constraint violation degrees, and the weights are Lagrange
multipliers updated by dual ascent inside the mini-batch loop (Algorithm 1). Slack Pg is
not predicted; it is restored by a power flow at evaluation time.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import sys

from sklearn.preprocessing import MinMaxScaler

# The shared modules live in ac_configuration/, a subpackage of this script's
# directory, so they resolve regardless of the working directory.
try:
    from ac_configuration import acopf_config
    from ac_configuration.acopf_data_setup import (
        load_parameters_from_csv,
        load_and_scale_acopf_data,
        prepare_data_splits,
        reconstruct_full_pg,
    )
    from ac_configuration.acopf_evaluation_metrics import (
        compute_mae_absolute,
        compute_mae_percentage,
        evaluate_acopf_predictions,
    )
    from ac_configuration.acopf_pypower import load_case_from_csv, solve_pf_setpoints
except ImportError as e:
    print(f"Error: Unable to import from ac_configuration/ ({e})")
    sys.exit(1)

GLOBAL_CASE_DATA = None

VIOLATION_NAMES = ('nu_2a', 'nu_2b', 'nu_3a', 'nu_3b', 'nu_4',
                   'nu_5a', 'nu_5b', 'nu_6a', 'nu_6b')


def unlimited_branch_mask(rate_a):
    """Branches with no usable thermal rating, using the project-wide sentinel rules."""
    rate_a = np.asarray(rate_a, dtype=np.float64)
    return ~np.isfinite(rate_a) | (rate_a <= 0) | (rate_a >= 9000)


def _build_subnetwork(dims, use_activation_on_last=False):
    """Stack Linear layers with ReLU between them."""
    layers = []
    for i, (d_in, d_out) in enumerate(dims):
        layers.append(nn.Linear(d_in, d_out))
        if i < len(dims) - 1 or use_activation_on_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class OPF_DNN_MC(nn.Module):
    """M_C architecture: a shared input block feeding four output heads.

    Layer widths follow the paper's appendix, parameterized by the number of load buses
    l, buses n, non-slack generators g and generators g_total.
    """

    def __init__(self, n_loads, n_buses, n_gen_non_slack, n_gen):
        super().__init__()
        l, n, g, g_total = n_loads, n_buses, n_gen_non_slack, n_gen

        self.input_block = _build_subnetwork(
            [(2 * l, 4 * l), (4 * l, 4 * l)], use_activation_on_last=True)

        self.out_v = _build_subnetwork(
            [(4 * l, 8 * l), (8 * l, 4 * l), (4 * l, 2 * n), (2 * n, n)])
        self.out_theta = _build_subnetwork(
            [(4 * l, 8 * l), (8 * l, 4 * l), (4 * l, 2 * n), (2 * n, n)])
        self.out_pg = _build_subnetwork(
            [(4 * l, 8 * l), (8 * l, 4 * l), (4 * l, 2 * g), (2 * g, g)])
        self.out_qg = _build_subnetwork(
            [(4 * l, 8 * l), (8 * l, 4 * l), (4 * l, 2 * g_total), (2 * g_total, g_total)])

    def forward(self, x):
        """Return the four scaled predictions (v, theta, pg_non_slack, qg)."""
        h = self.input_block(x)
        return self.out_v(h), self.out_theta(h), self.out_pg(h), self.out_qg(h)


class OPFConstraints:
    """Caches the topology on the target device and scores the nine violation degrees.

    Everything is differentiable and in p.u., so the violations can be added straight
    into the loss. Predictions arrive scaled and are inverse transformed here.
    """

    def __init__(self, params, scalers, device):
        self.device = device
        self.params = params
        self.scalers = scalers

        self.n_buses = params['general']['n_buses']
        self.n_gen = params['general']['n_gen']
        self.n_gen_non_slack = params['general']['n_gen_non_slack']
        self.n_branches = params['general']['n_branches']
        self.n_loads = params['general']['n_loads']

        def _t(array, dtype=torch.float32):
            return torch.tensor(array, dtype=dtype, device=device)

        self.vm_min = _t(params['bus']['vm_min'])
        self.vm_max = _t(params['bus']['vm_max'])
        self.pg_min = _t(params['generator']['pg_min'].flatten())
        self.pg_max = _t(params['generator']['pg_max'].flatten())
        self.qg_min = _t(params['generator']['qg_min'].flatten())
        self.qg_max = _t(params['generator']['qg_max'].flatten())

        self.non_slack_gen_idx = params['general']['non_slack_gen_idx']

        # Slack buses are excluded from the KCL penalty: slack Pg is not predicted,
        # so the balance there cannot close no matter how good the prediction is
        slack_bus_indices = {
            params['general']['bus_id_to_idx'][int(params['general']['gen_bus_ids'][gi])]
            for gi, is_slack in enumerate(params['general']['slack_gen_mask']) if is_slack
        }
        self.non_slack_bus_mask = torch.ones(self.n_buses, dtype=torch.bool, device=device)
        for idx in slack_bus_indices:
            self.non_slack_bus_mask[idx] = False
        self.n_non_slack_buses = self.non_slack_bus_mask.sum().float()

        self.f_idx = _t(params['branch']['f_bus_idx'], dtype=torch.long)
        self.t_idx = _t(params['branch']['t_bus_idx'], dtype=torch.long)
        self.angmin = _t(params['branch']['angmin_rad']).unsqueeze(0)
        self.angmax = _t(params['branch']['angmax_rad']).unsqueeze(0)
        self.bus_gs = _t(params['bus']['gs']).unsqueeze(0)
        self.bus_bs = _t(params['bus']['bs']).unsqueeze(0)

        # Full pi-model coefficients, so the flows include line charging and the
        # transformer tap; the shared loader precomputes them from the branch CSV
        for key in ('Yff_g', 'Yff_b', 'Yft_g', 'Yft_b',
                    'Ytf_g', 'Ytf_b', 'Ytt_g', 'Ytt_b'):
            setattr(self, key, _t(params['branch'][key]).unsqueeze(0))

        # Unrated branches get an unreachable squared limit, so nu_4 ignores them
        rate_a = np.asarray(params['branch']['rate_a'], dtype=np.float64).copy()
        rate_a[unlimited_branch_mask(rate_a)] = 1e6
        self.s_max_sq = _t(rate_a ** 2)

        self.bus_gen_map = _t(params['topology']['bus_gen_map'])
        self.load_bus_indices = _t(
            [params['general']['bus_id_to_idx'][int(lid)]
             for lid in params['general']['load_bus_ids']], dtype=torch.long)

        self._cache_scaler_tensors()

    def _cache_scaler_tensors(self):
        """Hold the MinMax parameters as tensors so inverse transforms stay on device."""
        # 'vm' maps to the all-bus scaler this method builds, not the shared
        # generator-bus one, because the v head predicts every bus
        for name, key in (('vm', 'vm_all'), ('va', 'va'),
                          ('pg', 'pg'), ('qg', 'qg'), ('x', 'x')):
            scaler = self.scalers[key]
            setattr(self, f'_scaler_{name}_scale', torch.tensor(
                scaler.scale_, dtype=torch.float32, device=self.device))
            setattr(self, f'_scaler_{name}_min', torch.tensor(
                scaler.min_, dtype=torch.float32, device=self.device))

    def _inv_transform(self, x_scaled, name):
        """Undo a MinMax transform on a tensor."""
        scale = getattr(self, f'_scaler_{name}_scale')
        min_val = getattr(self, f'_scaler_{name}_min')
        return (x_scaled - min_val) / scale

    def compute_branch_flows(self, vm, va):
        """Branch flows at both ends, from the full pi-model.

        Returns (pf, qf, pt, qt). The two ends differ by the losses and by the tap,
        so the to-end cannot be obtained by negating the from-end.
        """
        vi = vm[:, self.f_idx]
        vj = vm[:, self.t_idx]
        theta_ij = va[:, self.f_idx] - va[:, self.t_idx]
        cos_t = torch.cos(theta_ij)
        sin_t = torch.sin(theta_ij)
        vivj = vi * vj

        pf = vi ** 2 * self.Yff_g + vivj * (self.Yft_g * cos_t + self.Yft_b * sin_t)
        qf = -vi ** 2 * self.Yff_b + vivj * (self.Yft_g * sin_t - self.Yft_b * cos_t)
        pt = vj ** 2 * self.Ytt_g + vivj * (self.Ytf_g * cos_t - self.Ytf_b * sin_t)
        qt = -vj ** 2 * self.Ytt_b + vivj * (-self.Ytf_g * sin_t - self.Ytf_b * cos_t)
        return pf, qf, pt, qt

    def compute_violations(self, vm_pred_scaled, va_pred_scaled, pg_pred_scaled,
                           qg_pred_scaled, x_scaled, vm_true_scaled, va_true_scaled):
        """Return the nine violation degrees, each already averaged over the batch."""
        batch = vm_pred_scaled.shape[0]

        vm_pred = self._inv_transform(vm_pred_scaled, 'vm')
        va_pred = self._inv_transform(va_pred_scaled, 'va')
        pg_pred_ns = self._inv_transform(pg_pred_scaled, 'pg')
        qg_pred = self._inv_transform(qg_pred_scaled, 'qg')
        x_raw = self._inv_transform(x_scaled, 'x')
        vm_true = self._inv_transform(vm_true_scaled, 'vm')
        va_true = self._inv_transform(va_true_scaled, 'va')

        pg_pred_full = torch.zeros(batch, self.n_gen, device=self.device)
        pg_pred_full[:, self.non_slack_gen_idx] = pg_pred_ns

        pd_raw = x_raw[:, :self.n_loads]
        qd_raw = x_raw[:, self.n_loads:]

        # Voltage magnitude bounds
        nu_2a = (torch.clamp(self.vm_min.unsqueeze(0) - vm_pred, min=0)
                 + torch.clamp(vm_pred - self.vm_max.unsqueeze(0), min=0)
                 ).mean(dim=1).mean()

        theta_diff = va_pred[:, self.f_idx] - va_pred[:, self.t_idx]
        nu_2b = (torch.clamp(self.angmin - theta_diff, min=0)
                 + torch.clamp(theta_diff - self.angmax, min=0)).mean(dim=1).mean()

        # Pg bounds, non-slack generators only
        pg_min_ns = self.pg_min[self.non_slack_gen_idx].unsqueeze(0)
        pg_max_ns = self.pg_max[self.non_slack_gen_idx].unsqueeze(0)
        nu_3a = (torch.clamp(pg_min_ns - pg_pred_ns, min=0)
                 + torch.clamp(pg_pred_ns - pg_max_ns, min=0)).mean(dim=1).mean()

        # Qg bounds, all generators
        nu_3b = (torch.clamp(self.qg_min.unsqueeze(0) - qg_pred, min=0)
                 + torch.clamp(qg_pred - self.qg_max.unsqueeze(0), min=0)
                 ).mean(dim=1).mean()

        pf_pred, qf_pred, pt_pred, qt_pred = self.compute_branch_flows(vm_pred, va_pred)

        # Thermal limits, checked at the more heavily loaded end of each branch
        s_sq = torch.maximum(pf_pred ** 2 + qf_pred ** 2, pt_pred ** 2 + qt_pred ** 2)
        nu_4 = torch.clamp(s_sq - self.s_max_sq.unsqueeze(0),
                           min=0).mean(dim=1).mean()

        # Ohm's law: the predicted flows against the flows the true state implies
        pf_true, qf_true, _, _ = self.compute_branch_flows(vm_true, va_true)
        nu_5a = torch.abs(pf_pred - pf_true).mean(dim=1).mean()
        nu_5b = torch.abs(qf_pred - qf_true).mean(dim=1).mean()

        pg_at_bus = torch.matmul(pg_pred_full, self.bus_gen_map.T)
        qg_at_bus = torch.matmul(qg_pred, self.bus_gen_map.T)

        pd_at_bus = torch.zeros(batch, self.n_buses, device=self.device)
        qd_at_bus = torch.zeros(batch, self.n_buses, device=self.device)
        pd_at_bus[:, self.load_bus_indices] = pd_raw
        qd_at_bus[:, self.load_bus_indices] = qd_raw

        pf_sum = torch.zeros(batch, self.n_buses, device=self.device)
        qf_sum = torch.zeros(batch, self.n_buses, device=self.device)
        f_expand = self.f_idx.unsqueeze(0).expand(batch, -1)
        t_expand = self.t_idx.unsqueeze(0).expand(batch, -1)
        pf_sum.scatter_add_(1, f_expand, pf_pred)
        qf_sum.scatter_add_(1, f_expand, qf_pred)
        pf_sum.scatter_add_(1, t_expand, pt_pred)
        qf_sum.scatter_add_(1, t_expand, qt_pred)

        vm_sq = vm_pred ** 2
        p_shunt = self.bus_gs * vm_sq
        q_shunt = -self.bus_bs * vm_sq

        mask = self.non_slack_bus_mask.unsqueeze(0)
        kcl_p = torch.abs(pf_sum - (pg_at_bus - pd_at_bus - p_shunt))
        kcl_q = torch.abs(qf_sum - (qg_at_bus - qd_at_bus - q_shunt))
        nu_6a = (kcl_p * mask).sum(dim=1).mean() / self.n_non_slack_buses
        nu_6b = (kcl_q * mask).sum(dim=1).mean() / self.n_non_slack_buses

        return {
            'nu_2a': nu_2a, 'nu_2b': nu_2b,
            'nu_3a': nu_3a, 'nu_3b': nu_3b,
            'nu_4': nu_4,
            'nu_5a': nu_5a, 'nu_5b': nu_5b,
            'nu_6a': nu_6a, 'nu_6b': nu_6b,
        }


def split_y(y_batch, n_buses, n_gen_non_slack, n_gen):
    """Split the stacked target tensor into (vm, va, pg_non_slack, qg)."""
    i = 0
    vm = y_batch[:, i:i + n_buses]
    i += n_buses
    va = y_batch[:, i:i + n_buses]
    i += n_buses
    pg = y_batch[:, i:i + n_gen_non_slack]
    i += n_gen_non_slack
    qg = y_batch[:, i:i + n_gen]
    return vm, va, pg, qg


def train_lagrangian_dual(model, train_loader, val_loader, scalers, params,
                          n_epochs=80, lr=1e-3, rho=1e-2, lambda_max=100.0,
                          early_stop_patience=None, early_stop_min_delta=1e-6,
                          device='cpu'):
    """Train with dual ascent on the constraint multipliers (Algorithm 1).

    Each multiplier is raised by rho times its violation after every weight update, so
    persistently violated constraints gain weight over training. lambda_max caps that
    growth, which matters because a constraint the model cannot satisfy would otherwise
    keep gaining weight until it dominates the loss.

    The best model by validation MSE is restored at the end. Early stopping only
    triggers if a patience is given; the paper instead runs a fixed epoch budget.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']

    opf_constraints = OPFConstraints(params, scalers, device)

    # Algorithm 1 starts every multiplier at zero, so the first epochs are pure
    # imitation learning and the constraint terms only grow once violations appear
    lambdas = {name: 0.0 for name in VIOLATION_NAMES}
    history = {'train_loss': [], 'val_loss': [], 'mse_loss': []}

    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    print(f"\n{'=' * 80}")
    print(f"Lagrangian Dual Training")
    print(f"{'=' * 80}")
    print(f"  lr={lr}, rho={rho}, epochs={n_epochs}, lambda_max={lambda_max}")
    print(f"  Early stopping: "
          f"{'patience=' + str(early_stop_patience) if early_stop_patience else 'disabled'}")
    print(f"\n{'Ep':>4} {'Lo':>9} {'Lc':>9} {'Val':>9} "
          f"{'l2a':>7} {'l3a':>7} {'l4':>7} {'l5a':>7} {'l6a':>7} "
          f"{'v2a':>8} {'v3a':>8} {'v6a':>8}")
    print(f"{'-' * 110}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = epoch_mse = epoch_lc = 0.0
        n_batches = 0
        epoch_violations = {name: 0.0 for name in VIOLATION_NAMES}

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            vm_pred, va_pred, pg_pred, qg_pred = model(X_batch)
            vm_true, va_true, pg_true, qg_true = split_y(
                Y_batch, n_buses, n_gen_non_slack, n_gen)

            lo = (criterion(vm_pred, vm_true) + criterion(va_pred, va_true)
                  + criterion(pg_pred, pg_true) + criterion(qg_pred, qg_true))

            violations = opf_constraints.compute_violations(
                vm_pred, va_pred, pg_pred, qg_pred, X_batch, vm_true, va_true)

            # The multipliers are plain floats, so no gradient flows through them
            lc = torch.zeros((), device=device)
            for name in VIOLATION_NAMES:
                lc = lc + lambdas[name] * violations[name]

            total_loss = lo + lc
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                for name in VIOLATION_NAMES:
                    nu_val = violations[name].item()
                    lambdas[name] = min(
                        lambda_max, max(0.0, lambdas[name] + rho * nu_val))
                    epoch_violations[name] += nu_val

            epoch_loss += total_loss.item()
            epoch_mse += lo.item()
            epoch_lc += lc.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches
        avg_mse = epoch_mse / n_batches
        avg_lc = epoch_lc / n_batches
        avg_viols = {k: v / n_batches for k, v in epoch_violations.items()}
        history['train_loss'].append(avg_train)
        history['mse_loss'].append(avg_mse)

        # Validation tracks pure MSE, so it stays comparable as the multipliers grow
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                vm_p, va_p, pg_p, qg_p = model(X_val)
                vm_t, va_t, pg_t, qg_t = split_y(
                    Y_val, n_buses, n_gen_non_slack, n_gen)
                loss = (criterion(vm_p, vm_t) + criterion(va_p, va_t)
                        + criterion(pg_p, pg_t) + criterion(qg_p, qg_t))
                val_loss += loss.item() * len(X_val)
                n_val += len(X_val)

        avg_val = val_loss / n_val if n_val > 0 else 0.0
        history['val_loss'].append(avg_val)

        if epoch % max(1, n_epochs // 40) == 0 or epoch == 1 or epoch == n_epochs:
            print(f"{epoch:>4} {avg_mse:>9.6f} {avg_lc:>9.4f} {avg_val:>9.6f} "
                  f"{lambdas['nu_2a']:>7.3f} {lambdas['nu_3a']:>7.3f} "
                  f"{lambdas['nu_4']:>7.3f} {lambdas['nu_5a']:>7.3f} "
                  f"{lambdas['nu_6a']:>7.3f} "
                  f"{avg_viols['nu_2a']:>8.5f} {avg_viols['nu_3a']:>8.5f} "
                  f"{avg_viols['nu_6a']:>8.5f}")

        if avg_val < best_val_loss - early_stop_min_delta:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stop_patience and patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch} "
                      f"(patience={early_stop_patience})")
                break

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        print(f"\nRestored best model from epoch {best_epoch} "
              f"(val_loss={best_val_loss:.6f})")

    print(f"\n  Final multipliers:")
    for name in VIOLATION_NAMES:
        print(f"    {name}: {lambdas[name]:.6f}")

    return history


def evaluate_on_test_set(model, X_test, raw_data, scalers, params, device,
                        verbose=True, label="Test"):
    """Evaluate the network directly and again after a power flow at its setpoints.

    raw_data must already be restricted to the samples in X_test.
    """
    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_gen']
    n_loads = params['general']['n_loads']
    bus_id_to_idx = params['general']['bus_id_to_idx']
    slack_gen_mask = params['general']['slack_gen_mask']
    ns_idx = params['general']['non_slack_gen_idx']
    gen_bus_indices = np.array(
        [bus_id_to_idx[int(gid)] for gid in params['general']['gen_bus_ids']])

    model.eval()
    with torch.no_grad():
        vm_s, va_s, pg_s, qg_s = model(X_test.to(device))

    y_pred_vm = scalers['vm_all'].inverse_transform(vm_s.cpu().numpy())
    y_pred_va = scalers['va'].inverse_transform(va_s.cpu().numpy())
    y_pred_pg_ns = scalers['pg'].inverse_transform(pg_s.cpu().numpy())
    y_pred_qg = scalers['qg'].inverse_transform(qg_s.cpu().numpy())

    y_pred_pg_full = reconstruct_full_pg(y_pred_pg_ns, params)
    y_pred_vm_gen = y_pred_vm[:, gen_bus_indices]

    x_raw = scalers['x'].inverse_transform(X_test.cpu().numpy())
    pd_pu = x_raw[:, :n_loads]
    qd_pu = x_raw[:, n_loads:]

    y_true_pg = raw_data['pg']
    y_true_vm = raw_data['vm']
    y_true_qg = raw_data['qg']
    y_true_va_rad = raw_data['va']

    n_samples = len(X_test)
    pf_results_list = []
    converge_flags = []

    if verbose:
        print(f"\n  [{label}] Running power flow for {n_samples} samples...")

    for i in range(n_samples):
        try:
            r1_pf = solve_pf_setpoints(
                pd_pu[i], qd_pu[i], y_pred_pg_ns[i], y_pred_vm_gen[i],
                params, GLOBAL_CASE_DATA)
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
        print(f"    Converged: {sum(converge_flags)}/{n_samples}")

    pf_metrics = evaluate_acopf_predictions(
        y_pred_pg=y_pred_pg_full,
        y_pred_vm=y_pred_vm,
        y_true_pg=y_true_pg,
        y_true_vm=y_true_vm,
        y_true_qg=y_true_qg,
        y_true_va_rad=y_true_va_rad,
        pf_results_list=pf_results_list,
        converge_flags=converge_flags,
        params=params,
        verbose=False,
    )

    # Metrics straight off the network, with no power flow involved
    pg_min = params['generator']['pg_min'].flatten()
    pg_max = params['generator']['pg_max'].flatten()
    qg_min = params['generator']['qg_min'].flatten()
    qg_max = params['generator']['qg_max'].flatten()
    vm_min = params['bus']['vm_min']
    vm_max = params['bus']['vm_max']

    pg_viol = (np.maximum(0, pg_min[ns_idx] - y_pred_pg_ns)
               + np.maximum(0, y_pred_pg_ns - pg_max[ns_idx]))
    qg_viol = np.maximum(0, qg_min - y_pred_qg) + np.maximum(0, y_pred_qg - qg_max)
    vm_viol = np.maximum(0, vm_min - y_pred_vm) + np.maximum(0, y_pred_vm - vm_max)

    # Slack Pg only exists after the power flow, so its violation is read from there
    slack_gen_idx = np.where(slack_gen_mask)[0]
    slack_viols = []
    if len(slack_gen_idx) > 0:
        for i in range(n_samples):
            if converge_flags[i]:
                gen = pf_results_list[i][0]['gen']
                viols = (np.maximum(0, gen[:, 9] - gen[:, 1])
                         + np.maximum(0, gen[:, 1] - gen[:, 8]))
                slack_viols.append(
                    viols[slack_gen_idx[0]] / params['general']['BASE_MVA'])

    # Branch flows implied by the predicted voltages, using the same full pi-model
    # coefficients the training loss uses
    br = params['branch']
    f_idx, t_idx = br['f_bus_idx'], br['t_bus_idx']

    rate_a = np.asarray(br['rate_a'], dtype=np.float64).copy()
    rate_a[unlimited_branch_mask(rate_a)] = np.inf

    vi = y_pred_vm[:, f_idx]
    vj = y_pred_vm[:, t_idx]
    theta_ij = y_pred_va[:, f_idx] - y_pred_va[:, t_idx]
    cos_t, sin_t = np.cos(theta_ij), np.sin(theta_ij)
    vivj = vi * vj

    pf = vi ** 2 * br['Yff_g'] + vivj * (br['Yft_g'] * cos_t + br['Yft_b'] * sin_t)
    qf = -vi ** 2 * br['Yff_b'] + vivj * (br['Yft_g'] * sin_t - br['Yft_b'] * cos_t)
    pt = vj ** 2 * br['Ytt_g'] + vivj * (br['Ytf_g'] * cos_t - br['Ytf_b'] * sin_t)
    qt = -vj ** 2 * br['Ytt_b'] + vivj * (-br['Ytf_g'] * sin_t - br['Ytf_b'] * cos_t)

    s_sq = np.maximum(pf ** 2 + qf ** 2, pt ** 2 + qt ** 2)
    branch_viol = np.maximum(0, s_sq - rate_a ** 2)

    direct_metrics = {
        'direct_mae_pg_percent': compute_mae_percentage(
            y_true_pg[:, ~slack_gen_mask], y_pred_pg_ns),
        'direct_mae_vm_percent': compute_mae_percentage(
            y_true_vm[:, gen_bus_indices], y_pred_vm[:, gen_bus_indices]),
        'direct_mae_qg_percent': compute_mae_percentage(y_true_qg, y_pred_qg),
        'direct_mae_va_deg': compute_mae_absolute(
            y_true_va_rad * (180.0 / np.pi), y_pred_va * (180.0 / np.pi)),
        'direct_pg_viol_pu': float(np.mean(np.max(pg_viol, axis=1))),
        'direct_pg_slack_viol_pu': float(np.mean(slack_viols)) if slack_viols else 0.0,
        'direct_qg_viol_pu': float(np.mean(np.max(qg_viol, axis=1))),
        'direct_vm_viol_pu': float(np.mean(np.max(vm_viol, axis=1))),
        'direct_branch_viol_pu': float(np.mean(np.max(branch_viol, axis=1))),
        # The network does not predict slack Pg, so cost needs the power flow
        'direct_cost_gap_percent': pf_metrics['cost_optimality_gap_percent'],
    }

    return {**pf_metrics, **direct_metrics}


def print_metrics(m, label="Test"):
    """Print the direct and post-power-flow metric blocks."""
    print(f"\n--- {label}: Network Output (no power flow) ---")
    print(f"MAE: Pg={m['direct_mae_pg_percent']:.4f}%  "
          f"Vm={m['direct_mae_vm_percent']:.4f}%  "
          f"Qg={m['direct_mae_qg_percent']:.4f}%  "
          f"Va={m['direct_mae_va_deg']:.4f} deg")
    print(f"Viol: Pg(non-slack)={m['direct_pg_viol_pu']:.6f}  "
          f"Pg(slack)={m['direct_pg_slack_viol_pu']:.6f}  "
          f"Qg={m['direct_qg_viol_pu']:.6f}  "
          f"Vm={m['direct_vm_viol_pu']:.6f}  "
          f"Branch={m['direct_branch_viol_pu']:.6f}")
    print(f"Cost Gap: {m['direct_cost_gap_percent']:.4f}%")

    print(f"\n--- {label}: After Power Flow ---")
    print(f"Convergence: {m['convergence_rate_percent']:.2f}% "
          f"({m['n_converged']}/{m['n_samples']})")
    print(f"MAE: Pg(non-slack)={m['mae_pg_non_slack_percent']:.4f}%  "
          f"Vm={m['mae_vm_percent']:.4f}%  "
          f"Qg={m['mae_qg_percent']:.4f}%  "
          f"Va={m['mae_va_deg']:.4f} deg")
    print(f"Viol: Pg(non-slack)={m['mean_pg_viol_non_slack_pu']:.6f}  "
          f"Pg(slack)={m['mean_pg_viol_slack_pu']:.6f}  "
          f"Qg={m['mean_max_qg_viol_pu']:.6f}  "
          f"Vm={m['mean_max_vm_viol_pu']:.6f}  "
          f"Branch={m['mean_max_branch_viol_pu']:.6f}")
    print(f"Cost Gap: {m['cost_optimality_gap_percent']:.4f}%")


def lagrangian_acopf_experiment(
        case_name,
        params_path,
        data_path,
        n_train_use=None,
        seed=42,
        n_epochs=80,
        early_stop_patience=None,
        early_stop_min_delta=1e-6,
        learning_rate=1e-3,
        batch_size=64,
        device='cuda',
        lagrangian_lr=0.01,
        lambda_max=100.0,
        **kwargs  # Absorbs settings that do not apply, such as hidden_sizes
):
    """Train the M_C^D model on a random split and evaluate on the test indices."""
    global GLOBAL_CASE_DATA

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 80}")
    print(f"Lagrangian Dual ACOPF (M_C^D)")
    print(f"{'=' * 80}")
    print(f"Case: {case_name}  |  Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load network parameters and PyPower case data
    # ------------------------------------------------------------------
    params = load_parameters_from_csv(case_name, params_path)
    GLOBAL_CASE_DATA = load_case_from_csv(case_name, params_path)

    # ------------------------------------------------------------------
    # 2. Load dataset. The shared loader's Y holds only non-slack Pg and
    #    generator-bus Vm, but this model predicts v and theta at every bus, so
    #    the targets are reassembled here from raw_data. Only the all-bus Vm
    #    scaler is new; the shared va, pg and qg scalers already cover the rest.
    # ------------------------------------------------------------------
    x_scaled, _, scalers, raw_data, cost_baseline = \
        load_and_scale_acopf_data(data_path, params, fit_scalers=True)

    n_loads = params['general']['n_loads']
    n_buses = params['general']['n_buses']
    n_gen = params['general']['n_gen']
    n_gen_non_slack = params['general']['n_gen_non_slack']

    scalers['vm_all'] = MinMaxScaler().fit(raw_data['vm'])
    y_scaled = np.hstack([
        scalers['vm_all'].transform(raw_data['vm']),
        scalers['va'].transform(raw_data['va']),
        scalers['pg'].transform(raw_data['pg_non_slack']),
        scalers['qg'].transform(raw_data['qg']),
    ]).astype('float32')

    print(f"\n[Dataset Info]")
    print(f"  Buses: {n_buses}, Generators: {n_gen} (Non-Slack: {n_gen_non_slack}), "
          f"Loads: {n_loads}")
    print(f"  Y layout: vm({n_buses}) + va({n_buses}) + pg({n_gen_non_slack}) "
          f"+ qg({n_gen}) = {y_scaled.shape[1]}")
    if cost_baseline:
        print(f"  Cost Baseline: {cost_baseline:.2f} $/h")

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = prepare_data_splits(
        x_scaled, y_scaled, n_train_use=n_train_use, seed=seed)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_scaled[train_idx], dtype=torch.float32),
                      torch.tensor(y_scaled[train_idx], dtype=torch.float32)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_scaled[val_idx], dtype=torch.float32),
                      torch.tensor(y_scaled[val_idx], dtype=torch.float32)),
        batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model = OPF_DNN_MC(n_loads, n_buses, n_gen_non_slack, n_gen).to(device)

    print(f"\n{'=' * 80}")
    print(f"Model Configuration")
    print(f"{'=' * 80}")
    print(f"Architecture: l={n_loads}, n={n_buses}, g={n_gen_non_slack}, "
          f"g_total={n_gen}")
    print(f"Input: 2l={2 * n_loads} -> v({n_buses}) + theta({n_buses}) "
          f"+ pg({n_gen_non_slack}) + qg({n_gen})")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'=' * 80}")

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    train_lagrangian_dual(
        model, train_loader, val_loader, scalers, params,
        n_epochs=n_epochs, lr=learning_rate, rho=lagrangian_lr,
        lambda_max=lambda_max,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        device=device)
    train_time = time.perf_counter() - t_start
    print(f"Training completed in {train_time:.2f} seconds")

    # ------------------------------------------------------------------
    # 6. Inference latency (forward pass only)
    # ------------------------------------------------------------------
    X_test = torch.tensor(x_scaled[test_idx], dtype=torch.float32)
    sample = X_test[:1].to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(sample)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            model(sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    latency_ms = float(np.mean(times)) * 1000

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    raw_test = {k: v[test_idx] for k, v in raw_data.items()}
    test_metrics = evaluate_on_test_set(
        model, X_test, raw_test, scalers, params, device, verbose=True, label="Test")

    print(f"\n{'=' * 80}")
    print(f"Final Results Summary")
    print(f"{'=' * 80}")
    print(f"\nCase: {case_name}")
    print_metrics(test_metrics, "Test")

    print(f"\n--- Performance ---")
    print(f"Inference Time: {latency_ms:.4f} ms/sample (forward pass only)")
    print(f"Training Time:  {train_time:.2f} s")
    print(f"{'=' * 80}")

    return test_metrics


if __name__ == "__main__":
    LAGRANGIAN_LR = 0.01   
    LAMBDA_MAX = 100.0

    print("\n" + "=" * 80)
    print("Loading Configuration")
    print("=" * 80)
    print(f"\n[Lagrangian Dual Configuration]")
    print(f"  Dual step size rho: {LAGRANGIAN_LR}")
    print(f"  Multiplier cap: {LAMBDA_MAX}")
    print("=" * 80)

    results = lagrangian_acopf_experiment(
        **acopf_config.get_all_paths(),
        **acopf_config.get_all_params(),
        lagrangian_lr=LAGRANGIAN_LR,
        lambda_max=LAMBDA_MAX,
    )

    print("\nExperiment completed successfully!")