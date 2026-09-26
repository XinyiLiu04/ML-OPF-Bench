"""Run-scoped split selection and trained model handoff."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .splits import make_splits

_RUN = ContextVar("ml_opf_bench_run", default=None)


@dataclass
class RunContext:
    experiment: object
    indices: tuple | None = None
    n_total: int | None = None
    epochs_completed: int | None = None


@dataclass
class TrainingState:
    model: object
    params: dict
    train_time_s: float
    artifacts: dict = field(default_factory=dict)


@contextmanager
def managed_run(experiment):
    context = RunContext(experiment)
    token = _RUN.set(context)
    try:
        yield context
    finally:
        _RUN.reset(token)


def is_managed():
    return _RUN.get() is not None


def record_epoch(epoch):
    context = _RUN.get()
    if context is not None:
        context.epochs_completed = epoch


def current_experiment():
    context = _RUN.get()
    if context is None:
        raise RuntimeError("No active experiment")
    return context.experiment


def training_indices(n_total, n_train_use=None, seed=42):
    context = _RUN.get()
    if context is None:
        return make_splits(n_total, seed, pool_size=n_total if n_train_use is None else n_train_use)
    spec = context.experiment
    if seed != spec.seed:
        raise ValueError("Method seed disagrees with experiment seed")
    if context.indices is None:
        context.indices = make_splits(n_total, seed, spec.mode, spec.pool_size, spec.train_size)
        context.n_total = n_total
    elif context.n_total != n_total:
        raise ValueError("Methods must split the original, unfiltered dataset")
    return tuple(index.copy() for index in context.indices)
