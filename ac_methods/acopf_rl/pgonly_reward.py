"""Physical constraint diagnostics and a bounded, label-free Summation reward."""
import numpy as np

TOLERANCES = dict(pg_pu=1e-5, qg_pu=1e-5, vm_pu=1e-5,
                  thermal_relative=1e-5, angle_rad=1e-5)


def constraint_components(result, base_mva):
    gen, bus, branch = (result[k] for k in ('gen', 'bus', 'branch'))
    if not all(np.isfinite(a).all() for a in (gen, bus, branch)):
        raise FloatingPointError('Nonfinite converged PF result')
    def violation(x, lo, hi):
        return np.maximum(lo-x, 0) + np.maximum(x-hi, 0)
    limited = np.isfinite(branch[:,5]) & (branch[:,5] > 0)
    rate = branch[limited,5]
    thermal = np.concatenate([
        np.maximum(np.hypot(branch[limited,13], branch[limited,14])/rate-1, 0),
        np.maximum(np.hypot(branch[limited,15], branch[limited,16])/rate-1, 0)])
    lookup = {int(b):i for i,b in enumerate(bus[:,0])}
    f = np.array([lookup[int(b)] for b in branch[:,0]], dtype=int)
    t = np.array([lookup[int(b)] for b in branch[:,1]], dtype=int)
    angle = np.deg2rad(bus[f,8]-bus[t,8])
    # Independent bounds, from CSV via the PyPower branch table; no +/-30 assumption.
    values = dict(pg_pu=violation(gen[:,1],gen[:,9],gen[:,8])/base_mva,
                  qg_pu=violation(gen[:,2],gen[:,4],gen[:,3])/base_mva,
                  vm_pu=violation(bus[:,7],bus[:,12],bus[:,11]),
                  thermal_relative=thermal,
                  angle_rad=violation(angle,np.deg2rad(branch[:,11]),np.deg2rad(branch[:,12])))
    return {k:dict(mean=float(v.mean()) if v.size else 0.,
                   maximum=float(v.max()) if v.size else 0.) for k,v in values.items()}


class BoundedSummation:
    """Reward in (-2, 0), with PF failure at -3; no probe-based fitting.

    Violation means retain separate units in diagnostics. Their dimensionless
    reward sum uses unit scales (1 pu power/voltage, relative loading, 1 radian).
    This is a trade-off objective, not a guarantee of constraint satisfaction.
    """
    failure_reward = -3.0

    def __init__(self, params):
        g = params['generator']
        magnitude = np.maximum(np.abs(g['pg_min']),np.abs(g['pg_max'])).reshape(-1)
        self.cost_scale = float(np.sum(np.abs(g['cost_c2']).reshape(-1)*magnitude**2
                             + np.abs(g['cost_c1']).reshape(-1)*magnitude
                             + np.abs(g['cost_c0']).reshape(-1)))
        if not np.isfinite(self.cost_scale) or self.cost_scale <= 0:
            raise ValueError('Expected positive physical cost scale')

    def __call__(self, cost, components):
        if not np.isfinite(cost):
            raise FloatingPointError('Nonfinite PF cost')
        penalty = sum(v['mean'] for v in components.values())
        # Both terms are bounded even when slack dispatch is outside its bounds.
        return float(-0.5*(1+np.tanh(cost/self.cost_scale))-penalty/(1+penalty))


def feasible(components):
    return all(components[k]['maximum'] <= tol for k,tol in TOLERANCES.items())
