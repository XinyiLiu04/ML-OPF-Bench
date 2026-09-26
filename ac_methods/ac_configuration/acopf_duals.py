# -*- coding: utf-8 -*-
"""Loading of dual-variable CSVs, with the column reordering both dual-based methods need.

The exporter writes one column per generator, bus or branch, named with a trailing id and
ordered by that id. Everything on the Python side is indexed by CSV row position instead.
For cases with non-consecutive bus ids the two orders differ, so columns must be realigned
or every per-bus dual ends up attributed to the wrong bus without any error being raised.
"""

import os

import numpy as np
import pandas as pd


def _read_dual_csv(duals_dir, case_name, suffix):
    """Read one dual CSV and return (dataframe, column id list)."""
    path = os.path.join(duals_dir, f"{case_name}_{suffix}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dual variable file not found: {path}")

    df = pd.read_csv(path)
    try:
        col_ids = [int(col.rsplit('_', 1)[1]) for col in df.columns]
    except (IndexError, ValueError):
        raise ValueError(
            f"Cannot parse trailing ids from the columns of {path}; "
            f"expected names such as '{suffix}_12', got {list(df.columns)[:4]}")

    return df, col_ids


def load_dual_sorted(duals_dir, case_name, suffix):
    """Load a per-generator dual CSV with its columns sorted by id."""
    df, col_ids = _read_dual_csv(duals_dir, case_name, suffix)
    id_to_col = {cid: i for i, cid in enumerate(col_ids)}
    ordered = [df.columns[id_to_col[cid]] for cid in sorted(col_ids)]
    return df[ordered].values.astype(np.float32)


def load_dual_by_ids(duals_dir, case_name, suffix, target_ids):
    """Load a dual CSV and order its columns to match target_ids exactly.

    Use this for per-bus and per-branch duals, where target_ids is the id sequence in
    CSV row order, so the result lines up with bus_id_to_idx and the branch arrays.
    """
    df, col_ids = _read_dual_csv(duals_dir, case_name, suffix)
    id_to_col = {cid: i for i, cid in enumerate(col_ids)}

    missing = [int(i) for i in target_ids if int(i) not in id_to_col]
    if missing:
        raise ValueError(
            f"{case_name}_{suffix}.csv has no column for ids {missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}. The file has {len(col_ids)} columns "
            f"spanning ids {min(col_ids)}-{max(col_ids)}, but {len(target_ids)} were "
            f"requested; the dual export and the constraint CSVs disagree.")

    ordered = [df.columns[id_to_col[int(i)]] for i in target_ids]
    return df[ordered].values.astype(np.float32)


def load_thermal_duals(duals_dir, case_name, params):
    """Correct the legacy export's positional thermal labels; preserve JuMP signs."""
    import hashlib
    import json
    from pathlib import Path
    mapping_bytes = Path(__file__).with_name('thermal_dual_order.json').read_bytes()
    if hashlib.sha256(mapping_bytes).hexdigest() != 'bdca6b264305f6fff1357d4646528ed06057c37bdf3a7bb9d2575276e2532738':
        raise ValueError('Thermal mapping manifest hash mismatch')
    mapping = json.loads(mapping_bytes)[case_name]
    ids = params['general']['branch_ids']
    positions = {int(b): i for i, b in enumerate(ids)}
    br = params['branch']
    limited = sorted(int(b) for b, r in zip(ids, br['rate_a']) if np.isfinite(r) and r > 0)
    names = ('mu_sm_fr', 'mu_sm_to')
    source = []
    for name in names:
        path = Path(duals_dir) / f'{case_name}_{name}.csv'
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != mapping['sha256'][name]:
            raise ValueError(f'Unaudited thermal dual file: {path}')
        source.append(load_dual_by_ids(duals_dir, case_name, name, ids))
    if source[0].shape != source[1].shape:
        raise ValueError('Thermal dual shapes differ')
    arcs = mapping['constraint_arcs']
    if len(arcs) != 2 * len(limited):
        raise ValueError('Thermal mapping and rated branches disagree')
    result = [np.zeros_like(a) for a in source]
    seen = set()
    for j, (bid, f, t) in enumerate(arcs):
        col = positions[bid]
        ends = (int(br['f_bus'][col]), int(br['t_bus'][col]))
        if (f, t) == ends:
            end = 0
        elif (t, f) == ends:
            end = 1
        else:
            raise ValueError('Thermal mapping topology mismatch')
        if (bid, end) in seen:
            raise ValueError('Duplicate thermal mapping')
        seen.add((bid, end))
        result[end][:, col] = source[j % 2][:, positions[limited[j // 2]]]
    if seen != {(b, e) for b in limited for e in (0, 1)}:
        raise ValueError('Incomplete thermal mapping')
    return dict(zip(names, result))
