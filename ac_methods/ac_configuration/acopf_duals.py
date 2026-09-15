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