"""Unscaled benchmark data. Power and voltage magnitudes are in p.u."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import DatasetPaths, dataset_paths
from .splits import make_splits


@dataclass(frozen=True)
class Inputs:
    """Prediction inputs without reference labels; indices are original CSV row positions."""

    formulation: str
    indices: np.ndarray
    pd: np.ndarray
    qd: np.ndarray | None
    params: dict

    @property
    def x(self):
        return self.pd if self.qd is None else np.concatenate((self.pd, self.qd), axis=1)


@dataclass(frozen=True)
class Partition:
    inputs: Inputs
    targets: Mapping[str, np.ndarray]
    paths: DatasetPaths | None = None


@dataclass(frozen=True)
class DatasetSplit:
    train: Partition
    validation: Partition
    test: Partition


@dataclass(frozen=True)
class Dataset:
    """Unscaled references: powers/Vm in p.u., Va in radians.

    pg/qg: all generators; vm/va: all buses; vm_gen: generator buses.
    Network ordering and base MVA are in inputs.params['general'].
    """

    inputs: Inputs
    targets: Mapping[str, np.ndarray]
    paths: DatasetPaths

    def partition(self, indices):
        rows = np.asarray(indices)
        if rows.ndim != 1 or rows.dtype.kind not in 'iu':
            raise ValueError('indices must be a one-dimensional integer array')
        if np.any(rows < 0) or np.any(rows >= len(self.inputs.indices)):
            raise ValueError('indices are outside the dataset')
        if len(np.unique(rows)) != len(rows):
            raise ValueError('indices must not repeat samples')
        source = self.inputs
        return Partition(Inputs(source.formulation, source.indices[rows], source.pd[rows],
                                None if source.qd is None else source.qd[rows], source.params),
                         {key: value[rows] for key, value in self.targets.items()}, self.paths)

    def split(self, *, seed=42, mode='cross-system', pool_size=12000, train_size=None):
        """Use the same original-row split as the maintained baseline trainers."""
        indices = make_splits(len(self.inputs.indices), seed, mode, pool_size, train_size)
        return DatasetSplit(*(self.partition(rows) for rows in indices))


class _Unscaled:
    def transform(self, values):
        return values


def load_dataset(root, formulation, case='case118', scenario='base'):
    """Load local exports without scaling or filtering samples."""
    paths = dataset_paths(root, formulation, case, scenario)
    if formulation == 'ac':
        from ac_methods.ac_configuration.acopf_data_setup import (
            load_parameters_from_csv, load_and_scale_acopf_data,
        )
        params = load_parameters_from_csv(paths.case_name, str(paths.params_path))
        scalers = {name: _Unscaled() for name in ('x', 'pg', 'qg', 'vm', 'va')}
        _, _, _, raw, _ = load_and_scale_acopf_data(
            str(paths.data_path), params, fit_scalers=False, scalers=scalers)
        n_load = params['general']['n_loads']
        pd, qd = raw['x'][:, :n_load], raw['x'][:, n_load:]
        targets = {key: value for key, value in raw.items() if key != 'x'}
    else:
        from dc_methods.dc_configuration.dcopf_data_setup import load_parameters_from_csv, load_samples
        params = load_parameters_from_csv(paths.case_name, str(paths.params_path))
        pd, pg = load_samples(str(paths.data_path), params)
        qd = None
        targets = {'pg': pg, 'pg_non_slack': pg[:, params['general']['non_slack_gen_idx']]}
    arrays = {'pd': pd, **targets}
    if qd is not None:
        arrays['qd'] = qd
    for name, values in arrays.items():
        if values.ndim != 2 or len(values) != len(pd) or not np.isfinite(values).all():
            raise ValueError(f'{name} must be a finite matrix with one row per sample')
    if not len(pd):
        raise ValueError('dataset is empty')
    return Dataset(Inputs(formulation, np.arange(len(pd)), pd, qd, params), targets, paths)
