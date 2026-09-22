# ML-OPF-Bench
ML-OPF-Bench: Benchmarking Machine Learning for Optimal Power Flow

A benchmark of machine-learning methods for the AC optimal power flow problem, with a
shared data pipeline and a shared evaluation so that published numbers are comparable
across methods.

All method families are implemented against one interface. Each one predicts generator
setpoints, a power flow is solved at those setpoints, and the same metrics are reported:
prediction error against the reference ACOPF solution, constraint violation by category,
cost optimality gap, and power flow convergence rate.

Dataset: [`xinyi-liu/ML-OPF-Bench`](https://huggingface.co/datasets/xinyi-liu/ML-OPF-Bench)

- **Convergence rate is reported with its solver.** Methods that solve a real
  Newton-Raphson power flow report a real convergence rate. Methods whose output is an
  algebraic solution report `n/a`.
- **Inference time measures the method, not the pipeline.** For most methods it is the
  network forward pass. For Q-correction it includes the one or two power flows, because
  those *are* the method. Each script states what it timed.

## Installation

```bash
git clone https://github.com/<user>/ML-OPF-Bench.git
cd ML-OPF-Bench
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `pypower`.

## Getting the data

Download the dataset repository, then point the code at it:

```bash
huggingface-cli download xinyi-liu/ML-OPF-Bench --repo-type dataset \
  --local-dir ./ML-OPF-Bench-data

export ML_OPF_BENCH_DATA=$PWD/ML-OPF-Bench-data
```

Verify before running:

```bash
python ac_methods/ac_configuration/acopf_config.py
```

This prints the resolved configuration and checks the data and constraint directories,
reporting `[OK]` or `[MISSING]` for each.

## Running a method

Every method reads the same configuration and is launched the same way:

```bash
python ac_methods/acopf_dnn.py
```

All experiment settings live in one file, `ac_methods/ac_configuration/acopf_config.py`:

| Setting | Meaning |
| --- | --- |
| `CASE` | `'case30'`, `'case118'` or `'case300'` |
| `VARIANCE` | load sampling spread, e.g. `'v=0.12'` |
| `N_TRAIN_USE` | how many samples to draw from the case before splitting |
| `SEED` | controls the split |
| `N_EPOCHS_MAX`, `EARLY_STOP_PATIENCE`, `LEARNING_RATE`, `BATCH_SIZE`, `HIDDEN_SIZES` | training budget |
| `DEVICE` | `'auto'`, `'cuda'`, `'mps'` or `'cpu'` |

Method-specific hyperparameters are constants at the top of each method script's
`__main__`, not in the config, so the shared config stays the same for every method.

Changing `SEED` or `N_TRAIN_USE` changes the split, which means **every method must be
re-run** before its numbers are comparable again. Fix these two before starting a
production sweep.

## Adding a case

Add one line to `CASES` in the config. The folder names follow from the short name:

```python
CASES = {
    'case57': {'full_name': 'pglib_opf_case57_ieee', 'short_name': 'case57'},
}
```

## Repository layout

```
ac_methods/
├── ac_configuration/              shared across every method
│   ├── acopf_config.py            paths, case registry, training budget
│   ├── acopf_data_setup.py        constraint loading, scaling, the train/val/test split
│   ├── acopf_evaluation_metrics.py  the single scoring function
│   ├── acopf_pypower.py           PyPower case construction and the power flow call
│   └── acopf_duals.py             dual-variable CSV loading with column alignment
│
├── acopf_lr.py  acopf_dnn.py  acopf_qc.py  acopf_cp(fr).py  acopf_mu.py  acopf_activeset.py
├── acopf_rl.py  reward.py
├── acopf_pinn_main.py  acopf_pinnmodel.py  acopf_pinnlayer.py  acopf_densecorenetwork.py
├── gnn_main.py  gnn_model.py  gnn_utils.py  generate_subopt_state.py
└── algebraic_power_flow.py  deepopf_ngt_common.py
    paper_unsupervised_acopf.py  paper_semi_supervised_acopf.py
    unsupervised_learning_acopf.py  semi_supervised_acopf.py
```

Method scripts sit directly in `ac_methods/`, so Python puts that directory on the path
and `from ac_configuration import ...` resolves no matter which directory you launch
from.

## Metrics

`evaluate_acopf_predictions` returns these keys. Accuracy, cost and violation figures are
averaged over **converged samples only**; non-converged samples carry `NaN` and are
excluded rather than filled with a sentinel, which would otherwise dominate any mean
taken on a case with a low convergence rate.

**Accuracy** — `mae_pg_non_slack_percent`, `mae_pg_all_percent`, `mae_pg_slack_percent`,
`mae_vm_percent`, `mae_qg_percent`, `mae_va_deg`

**Violations, p.u.** — `mean_pg_viol_non_slack_pu`, `mean_pg_viol_slack_pu`,
`mean_max_qg_viol_pu`, `mean_max_vm_viol_pu`, `mean_max_branch_viol_pu`,
`mean_max_pg_viol_pu`

**Cost** — `cost_true_mean`, `cost_pred_mean`, `cost_optimality_gap_percent`

**Coverage** — `convergence_rate_percent`, `n_converged`, `n_samples`

Three conventions worth knowing when reading the numbers:

- **Slack Pg is never predicted.** It is whatever the power flow assigns, so it absorbs
  the imbalance left by the predicted setpoints. Its violation is reported separately
  from the non-slack generators because it reflects a different failure mode.
- **`mae_vm_percent` covers generator buses only**, so methods that predict voltage at
  every bus and methods that predict it only at generators stay comparable.
- **`mean_max_branch_viol_pu` is relative**: 1.0 means 100% over the thermal rating.

## License and citation
