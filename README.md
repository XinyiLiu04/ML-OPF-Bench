# ML-OPF-Bench
ML-OPF-Bench: Benchmarking Machine Learning for Optimal Power Flow

A benchmark of machine-learning methods for the AC and DC optimal power flow problems,
with a shared data pipeline and a shared evaluation per problem so that published numbers
are comparable across methods.

Within each problem, all method families are implemented against one interface. Each one
predicts generator setpoints and is scored with the same metrics: prediction error against
the reference OPF solution, constraint violation by category, and cost optimality gap.

- **ACOPF.** A power flow is solved at the predicted setpoints, and the power flow
  convergence rate is reported alongside the other metrics.
- **DCOPF.** Branch flows are an exact linear function of the dispatch through the PTDF
  matrix, so every prediction is evaluated algebraically. There is no power flow step and
  no convergence rate; the power balance error is reported instead.

Dataset: [`xinyi-liu/ML-OPF-Bench`](https://huggingface.co/datasets/xinyi-liu/ML-OPF-Bench)

- **Convergence rate is reported with its solver.** AC methods that solve a real
  Newton-Raphson power flow report a real convergence rate. AC methods whose output is an
  algebraic solution report `n/a`.
- **Inference time measures the method, not the pipeline.** For most methods it is the
  network forward pass. Where a solver step is part of the method it is included: the one
  or two power flows of AC Q-correction, the QP projection of DC DeepOPF, and the recovery
  LPs of DC active-set classification. Each script states what it timed.

## Installation

```bash
git clone https://github.com/<user>/ML-OPF-Bench.git
cd ML-OPF-Bench
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `pypower`.
The DC GNN additionally needs `torch_geometric`.

## Getting the data

Download the dataset repository, then point the code at it:

```bash
huggingface-cli download xinyi-liu/ML-OPF-Bench --repo-type dataset \
  --local-dir ./ML-OPF-Bench-data

export ML_OPF_BENCH_DATA=$PWD/ML-OPF-Bench-data
```

`ML_OPF_BENCH_DATA` must point at the folder containing both `ac_dataset/` and
`dc_dataset/`; the AC and DC code read from the same root.

Verify before running:

```bash
python ac_methods/ac_configuration/acopf_config.py
python dc_methods/dc_configuration/dcopf_config.py
```

Each prints the resolved configuration and checks the data and constraint paths,
reporting `[OK]` or `[MISSING]` for each.

## Running a method

Every method reads the configuration of its problem and is launched the same way:

```bash
python ac_methods/acopf_dnn.py
python dc_methods/dcopf_dnn.py
```

AC and DC each have one configuration file, `ac_methods/ac_configuration/acopf_config.py`
and `dc_methods/dc_configuration/dcopf_config.py`. The two are separate, since the
problems have different constraint schemas and metrics, but expose the same settings:

| Setting | Meaning |
| --- | --- |
| `CASE` | a key of `CASES`, e.g. `'case30'`, `'case118'` or `'case300'` |
| `VARIANCE` | load sampling spread, e.g. `'v=0.12'` |
| `N_TRAIN_USE` | how many samples to draw from the case before splitting |
| `SEED` | controls the split |
| `N_EPOCHS_MAX`, `EARLY_STOP_PATIENCE`, `LEARNING_RATE`, `BATCH_SIZE`, `HIDDEN_SIZES` | training budget |
| `DEVICE` | `'auto'`, `'cuda'`, `'mps'` or `'cpu'` |

Method-specific hyperparameters are constants at the top of each method script's
`__main__`, not in the config, so the shared config stays the same for every method. A
method that receives a shared setting it does not use prints that it is ignoring it.

Changing `SEED` or `N_TRAIN_USE` changes the split, which means **every method must be
re-run** before its numbers are comparable again. Fix these two before starting a
production sweep.

## Adding a case

Add one line to `CASES` in the config of the problem. The folder names follow from the
short name:

```python
CASES = {
    'case57': {'full_name': 'pglib_opf_case57_ieee', 'short_name': 'case57'},
}
```

## Repository layout

```
ac_methods/
├── ac_configuration/              shared across every AC method
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

dc_methods/
├── dc_configuration/              shared across every DC method
│   ├── dcopf_config.py            paths, case registry, training budget, device
│   ├── dcopf_data_setup.py        constraint, sample and dual loading, the split, slack reconstruction
│   ├── dcopf_evaluation_metrics.py  violations, cost and the single scoring function
│   └── dcopf_torch_utils.py       torch DC physics, MLP, early-stopping loop, latency timing
│
├── dcopf_lr.py        linear regression
├── dcopf_dnn.py       supervised MLP
├── dcopf_mu.py        Lagrangian dual training (Fioretto et al.)
├── dcopf_cp_qp.py     DeepOPF with QP projection (Pan et al.)
├── dcopf_as.py        active-set classification (Deka & Misra)
├── dcopf_ngt.py       DeepOPF-NGT, unsupervised (Huang et al.)
├── dcopf_engt.py      extended DeepOPF-NGT, semi-supervised
├── dcopf_gnn.py       spectral GNN (Owerko et al.)
└── dcopf_kkt_pinn.py  KKT-informed PINN with collocation points (Nellikkath & Chatzivasileiadis)
```

Method scripts sit directly in `ac_methods/` or `dc_methods/`, so Python puts that
directory on the path and `from ac_configuration import ...` or
`from dc_configuration import ...` resolves no matter which directory you launch from.
The DC side has no reinforcement-learning method.

## Model selection

- **Supervised methods** restore the checkpoint with the best validation loss and stop
  early after `EARLY_STOP_PATIENCE` epochs without improvement.
- **Unsupervised methods** (e.g. DC DeepOPF-NGT) have no label to overfit, so they train
  for the full budget and return the final epoch. A unit-weight validation score is
  printed for inspection only; the training objective cannot be used for selection
  because its adaptive weights change during training.

## Metrics

### ACOPF

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

### DCOPF

`evaluate_dispatch` returns these keys, averaged over **every test sample**; with no power
flow step there is nothing to fail to converge.

**Accuracy** — `mae_pg_non_slack`, `mae_pg_slack` (percent of the mean absolute true value)

**Violations** — `viol_pg_non_slack`, `viol_pg_slack` (p.u.), `viol_branch` (multiple of
the rating), `viol_balance` (p.u.)

**Cost** — `cost_gap_percent`

**Timing** — `train_time_s`, `inference_ms`, `inference_scope`

Conventions:

- **Slack Pg is reconstructed from the power balance**: the total load minus the
  predicted non-slack dispatch, split evenly among the generators at the slack bus. This
  makes `viol_balance` zero for methods that predict only non-slack units, and it is why
  the slack violation is reported separately.
- **Violations are mean of max**: the worst generator or branch of each sample, averaged
  over samples. `viol_branch` is relative, so 0.1 means 10% over the rating.
- **Only rated branches are constrained**: a branch counts when its `rate_a` is below
  `1e10`, the same threshold the data generator used, so evaluation and the reference
  solutions agree on which limits exist.
- **Some methods report extras** after the standard block, such as `qp_failure_rate`
  for DeepOPF, or `top1_accuracy`, `test_unseen_rate` and `recovery_failure_rate` for
  active-set classification. DeepOPF prints its results with and without the QP
  projection, active-set classification with Top-1 and Top-K selection.

## DC data format

```
ROOT/dc_dataset/dcopf_constraints/<short_name>/<full_name>_{gen_limits,gen_costs,branch_limits,
                                                branch_info,ptdf_matrix,bus_gen_map,bus_ids,base_mva}.csv
ROOT/dc_dataset/dcopf_datasets/<short_name>(<variance>)/<full_name>_dataset_with_duals.csv
```

| File | Content |
| --- | --- |
| `gen_limits` | `gen_id`, `pgmin`, `pgmax` |
| `gen_costs` | `gen_id`, `cost_c2`, `cost_c1`, `cost_c0`; quadratic in per-unit Pg |
| `branch_limits` | `branch_id`, `rate_a` |
| `branch_info` | `branch_id`, `f_bus`, `t_bus`, `r_pu`, `x_pu`, `rate_a` |
| `ptdf_matrix` | dense, branches × buses |
| `bus_gen_map` | dense, buses × generators |
| `bus_ids` | `bus_id`, in the column order of `ptdf_matrix` and `bus_gen_map` |
| `base_mva` | `parameter`, `value` |

All quantities are per unit. Bus ids need not be consecutive (`case300` has gaps), so
per-bus columns are always mapped through `bus_ids.csv`, never by position.

The sample file holds one row per solved sample: loads `pd<bus_id>` for load buses,
dispatch `pg<gen_id>`, the power balance multiplier `lambda`, and `mu_g_min_<gen_id>`,
`mu_g_max_<gen_id>`, `mu_line_max_<branch_id>`, `mu_line_min_<branch_id>` for rated
branches. The inequality multipliers are already sign-normalized, so each is nonnegative
and a constraint is active when its multiplier is positive; unlike the AC duals, none
needs negating. `lambda` follows the JuMP convention, the marginal cost of load.

Samples are generated by drawing each load independently from a Gaussian around its base
value, with standard deviation `variance × base load`, clipped at zero, and solving the
PTDF-based DCOPF with Ipopt. Samples whose solve fails are dropped, so a dataset holds
slightly fewer rows than the number of attempts.

To regenerate a DC case, from the repository root:

```bash
julia --project=data_generalization data_generalization/export_dc_constraints.jl CASE_FILE OUTPUT_DIR
julia --project=data_generalization data_generalization/generate_dc_dataset.jl CASE_FILE OUTPUT_DIR \
  [SAMPLES=50000] [VARIANCE=0.12] [SEED=42]
```

## License and citation