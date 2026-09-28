"""Training, prediction, and optional repair interfaces."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .datasets import Inputs, Partition


@dataclass(frozen=True)
class Prediction:
    """Setpoints in p.u. and input row order: non-slack Pg, generator-bus Vm.

    Invalid rows may contain NaN and remain in evaluation counts.
    valid indicates prediction success, not OPF feasibility.
    """

    pg: np.ndarray
    vm: np.ndarray | None = None
    valid: np.ndarray | None = None

    def validate(self, inputs: Inputs):
        n = len(inputs.indices)
        ns = inputs.params['general']['non_slack_gen_idx']
        if not isinstance(self.pg, np.ndarray) or self.pg.shape != (n, len(ns)):
            raise ValueError('pg must have shape (samples, non-slack generators)')
        valid = np.ones(n, dtype=bool) if self.valid is None else self.valid
        if not isinstance(valid, np.ndarray) or valid.shape != (n,) or valid.dtype != np.bool_:
            raise ValueError('valid must be a boolean array with one entry per sample')
        if not np.isfinite(self.pg[valid]).all():
            raise ValueError('valid pg rows must be finite')
        if inputs.formulation == 'ac':
            n_gen = inputs.params['general']['n_gen']
            if not isinstance(self.vm, np.ndarray) or self.vm.shape != (n, n_gen):
                raise ValueError('AC vm must have shape (samples, generators)')
            if not np.isfinite(self.vm[valid]).all():
                raise ValueError('valid vm rows must be finite')
        elif inputs.formulation == 'dc':
            if self.vm is not None:
                raise ValueError('DC predictions must not contain vm')
        else:
            raise ValueError(f'Unknown formulation: {inputs.formulation}')
        return self


@runtime_checkable
class Method(Protocol):
    """Fit on train, select on validation; predict preserves all input rows."""

    def fit(self, train: Partition, validation: Partition, *, seed: int) -> None: ...

    def predict(self, inputs: Inputs) -> Prediction: ...


@runtime_checkable
class Postprocessor(Protocol):
    """Optional method repair; retains row order and exposes failures via valid."""

    def postprocess(self, inputs: Inputs, prediction: Prediction) -> Prediction: ...


def predict_method(method: Method, inputs: Inputs, *, postprocess=False) -> Prediction:
    """Predict setpoints, optionally repair, and validate shapes and failure flags."""
    result = method.predict(inputs)
    if not isinstance(result, Prediction):
        raise TypeError('predict must return Prediction')
    result.validate(inputs)
    if postprocess:
        if not isinstance(method, Postprocessor):
            raise TypeError('method does not implement postprocess')
        repaired = method.postprocess(inputs, result)
        validate_repair(inputs, result, repaired)
        result = repaired
    return result


def validate_repair(inputs, raw, repaired):
    if not isinstance(repaired, Prediction):
        raise TypeError('postprocess must return Prediction')
    repaired.validate(inputs)
    if raw.valid is not None:
        valid = np.ones(len(inputs.indices), dtype=bool) if repaired.valid is None else repaired.valid
        if np.any(valid & ~raw.valid):
            raise ValueError('postprocess cannot silently clear prediction failures')
