# Julia data generation

These scripts extract the generation paths associated with the existing ACOPF and
DCOPF CSV datasets. The notebooks remain as historical sources. Run the scripts
individually; including a script defines its module without starting generation.

## Environment

Run commands from the repository root. The committed environment records Julia
1.12.7 and locks package versions in `Manifest.toml`.

```bash
julia --project=data_generalization -e 'using Pkg; Pkg.instantiate()'
```

The environment includes PowerModels, Ipopt, JuMP, MathOptInterface, CSV,
DataFrames and ProgressMeter. No notebook kernel is required.

## Scripts and sources

Cell numbers below count code cells from the top, not notebook execution counters.

| Script | Notebook source | Purpose |
| --- | --- | --- |
| `generate_ac_dataset.jl` | `ACOPF dataset.ipynb`, cell 1 | Gaussian load sampling and ACOPF solutions |
| `generate_ac_duals.jl` | `ACOPF Constraints with duals.ipynb`, cell 2, V3 | Re-solve existing AC loads and extract duals |
| `export_ac_constraints.jl` | `ACOPF Constraints.ipynb`, cell 5 | AC parameters in p.u., including the bus-generator map |
| `generate_dc_dataset.jl` | `DCOPF Constraints with dual.ipynb`, cell 1 | Gaussian samples and a single system balance dual |
| `export_dc_constraints.jl` | `DCOPF Constraints.ipynb`, cell 1 | DC limits, topology, PTDF and ID maps |

The AC base generator is also duplicated in the AC constraints notebook. The DC
diagnostic variant and Gaussian mode of the sampling variant can produce the same
format. CSVs alone do not uniquely identify which equivalent cell was executed.
The selected scripts represent these generation paths, not a claim of exact
historical execution provenance.

The AC case300 dataset provides stronger evidence for V3: its 16,146 rows and
nonzero counts for all ten dual arrays match the saved V3 run, including 510,682
and 492,489 entries with absolute value above `1e-6` in `mu_sm_fr` and `mu_sm_to`.
This verifies correspondence with the saved run, not the correctness of branch
labels; see the thermal-dual limitation below.

## Running

Input and output paths are explicit command-line arguments. Relative paths are
resolved from the working directory. Each output directory must be new: scripts
reject an existing path to protect previous datasets. Output directory names are
user-supplied and do not determine sampling parameters.

Both dataset generators accept optional positional arguments in this order:
`SAMPLES VARIANCE SEED`, defaulting to `50000 0.12 42`. `SAMPLES` is the number of
attempts, not a guaranteed number of saved rows. `VARIANCE` retains the notebook
argument name: it is the relative Gaussian **standard deviation**, not the
statistical variance. Pd is clipped at zero; AC Qd is sampled independently and
is not clipped. The AC generator perturbs buses with positive base Pd, retaining
the original treatment of other loads.

Example using new output directories inside the ignored dataset folders:

```bash
julia --project=data_generalization data_generalization/export_ac_constraints.jl \
  test_systems/typ/pglib_opf_case30_ieee.m ac_dataset/rebuilt/acopf_constraints/case30

julia --project=data_generalization data_generalization/generate_ac_dataset.jl \
  test_systems/typ/pglib_opf_case30_ieee.m 'ac_dataset/rebuilt/acopf_datasets/case30(v=0.12)' 50000 0.12 42

julia --project=data_generalization data_generalization/generate_ac_duals.jl \
  test_systems/typ/pglib_opf_case30_ieee.m \
  'ac_dataset/rebuilt/acopf_datasets/case30(v=0.12)' \
  'ac_dataset/rebuilt/acopf_datasets/case30(v=0.12)_with_duals'

julia --project=data_generalization data_generalization/export_dc_constraints.jl \
  test_systems/typ/pglib_opf_case30_ieee.m dc_dataset/rebuilt/dcopf_constraints/case30

julia --project=data_generalization data_generalization/generate_dc_dataset.jl \
  test_systems/typ/pglib_opf_case30_ieee.m 'dc_dataset/rebuilt/dcopf_datasets/case30(v=0.12)' 50000 0.12 42
```

Use a small `SAMPLES` value for a smoke test. For API cases, pass a case from
`test_systems/api/`. Use `--help` with any script to see its argument order.
Adjust downstream training paths separately when using the `rebuilt` directories.

## CSV contract

Every filename starts with the case file's stem. Rows of sample files are aligned
within each generated dataset. IDs in columns follow the sorted source IDs.

| Script | Filename suffixes | Column convention |
| --- | --- | --- |
| AC samples | `_pd`, `_qd`, `_pg`, `_qg`, `_vm`, `_va` | `pd{id}`, `qd{id}`, `pg_{id}`, `qg_{id}`, `vm_{id}`, `va_{id}` |
| AC duals | `_mu_pg_min/max`, `_mu_qg_min/max`, `_mu_vm_min/max`, `_lambda_kcl_r/i`, `_mu_sm_fr/to` | `{suffix}_{id}` |
| DC samples | `_loads`, `_generations`, `_dataset_with_duals` | `pd{id}`, `pg{id}`, `lambda`, `mu_g_min/max_{id}`, `mu_line_max/min_{id}` |
| AC constraints | `_base_mva`, `_bus_data`, `_gen_data`, `_bus_gen_map`, `_branch_data`, `_slack_buses` | Historical headers preserved |
| DC constraints | `_gen_limits`, `_gen_costs`, `_branch_info`, `_branch_limits`, `_ptdf_matrix`, `_bus_gen_map`, `_bus_ids`, `_base_mva` | Historical headers preserved |

All suffixes end in `.csv`. The AC dual script writes only the ten dual files;
it reads Pd/Qd from the base dataset without copying or regenerating the other
base CSVs. It requires sorted, matching Pd/Qd columns and equal row counts.

Power quantities are p.u.; voltage magnitudes are p.u. and angles are radians.
Cost coefficients are the PowerModels coefficients evaluated with p.u. generation;
the objective remains a monetary cost per hour. AC dual signs are raw JuMP signs.
DC upper-bound multipliers are negated as in the original notebook. `lambda` is a
single system balance multiplier, not a vector of nodal LMPs.

## Preserved behavior and limitations

- AC samples retain only `LOCALLY_SOLVED` results; DC samples retain
  `LOCALLY_SOLVED` or `OPTIMAL` results. Other termination statuses are skipped.
  Unexpected exceptions now stop execution instead of being silently swallowed.
- AC dual extraction now stops on a failed solve or missing required dual data.
  It does not write a zero-filled failed row as V3 could. Dataset and dual files
  are written only after the full loop succeeds; constraint exporters write
  their files sequentially. A file-system error can still leave partial output.
- The V3 thermal-dual mapping is retained: successive quadratic inequalities
  are assigned to from/to columns in sorted branch-ID order. Constraint counts
  are checked, but matching counts do not validate branch order. Do not treat
  the thermal-dual column labels as independently verified branch assignments.
- AC constraint headers `va_deg` and `shift_deg` are inherited names. Values are
  copied from PowerModels after parsing, so they are radians despite these names.
  No unit conversion or column rename is introduced here.
- The DC PTDF construction retains the notebook's treatment of reactance and
  bus shunts, without adding transformer tap or phase-shift corrections.
  These scripts are not a revision of the historical DC formulation.
- The base generators retain the notebook's per-bus dictionary construction,
  which does not aggregate multiple load records at a bus. Use the original
  benchmark cases; new case families require checking this assumption.
- Identical seeds do not establish bitwise reproduction of old CSVs across
  Julia, solver, or package versions. Original per-run environments were not
  captured. Uniform sampling and the nodal-LMP variant remain in the notebooks.

## Validation

The extracted notebook functions and these scripts were run in the same locked
environment. For case30, ten Gaussian attempts with spread `0.12` and seed `42`
produced eight AC rows and ten DC rows. The eight AC rows were also used for dual
extraction. Static constraints were exported for case30, case118, case300 and
case118 API. All 75 resulting CSV tables matched the corresponding notebook
outputs exactly in column order and values.

This is a refactoring regression check on small samples, not a full regeneration
of the published datasets or independent validation of the inherited formulas.

## GitHub scope

The Julia scripts, this README and the Julia environment files are source files.
The repository `.gitignore` already excludes `ac_dataset/` and `dc_dataset/`;
generated CSVs belong with the dataset distribution, not the code commit.
The original notebooks are retained unchanged for reference.
