"""Deterministic shared splits, with fixed held-out data for scaling experiments."""

import numpy as np


def make_splits(n_total, seed=42, mode="cross-system", pool_size=12000, train_size=None):
    order = np.random.default_rng(seed).permutation(n_total)
    if mode == "cross-system":
        if not 12 <= pool_size <= n_total:
            raise ValueError(f"pool_size={pool_size} must be between 12 and {n_total}")
        n_hold = pool_size // 12
        return tuple(np.split(order[:pool_size], [pool_size - 2 * n_hold, pool_size - n_hold]))
    if mode != "scaling":
        raise ValueError(mode)
    n_hold = n_total // 10
    available = n_total - 2 * n_hold
    if n_hold < 1 or train_size is None or not 1 <= train_size <= available:
        raise ValueError(f"train_size={train_size} exceeds the available pool ({available})")
    return order[2 * n_hold:2 * n_hold + train_size], order[:n_hold], order[n_hold:2 * n_hold]
