"""Shared MATPOWER pi-model branch coefficients and direct flow metrics."""
import numpy as np


def branch_coefficients(branch):
    y = 1 / (np.asarray(branch['r_pu']) + 1j * np.asarray(branch['x_pu']))
    tap = np.where(np.asarray(branch['tap_ratio']) == 0, 1, branch['tap_ratio'])
    tap = tap * np.exp(1j * np.asarray(branch['shift_rad']))
    charging = 1j * np.asarray(branch['b_pu']) / 2
    return (y + charging) / abs(tap)**2, -y / np.conj(tap), -y / tap, y + charging


def direct_branch_relative_violation(vm, va, params):
    """Per-sample maximum of max(|S_from|, |S_to|)/rating - 1, clipped at zero."""
    br = params['branch']
    lookup = params['general']['bus_id_to_idx']
    f = [lookup[int(b)] for b in br['f_bus']]
    t = [lookup[int(b)] for b in br['t_bus']]
    v = np.asarray(vm) * np.exp(1j * np.asarray(va))
    vf, vt = v[:, f], v[:, t]
    yff, yft, ytf, ytt = branch_coefficients(br)
    sf = vf * np.conj(yff * vf + yft * vt)
    st = vt * np.conj(ytf * vf + ytt * vt)
    rate = np.asarray(br['rate_a'])
    rated = np.isfinite(rate) & (rate > 0)
    if not rated.any():
        return np.zeros(len(v))
    return np.maximum(0, np.maximum(abs(sf[:, rated]), abs(st[:, rated])) / rate[rated] - 1).max(axis=1)
