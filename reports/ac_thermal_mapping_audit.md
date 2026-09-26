# AC thermal dual mapping and direct-flow correction

Original label CSVs are unchanged. Mapping is reconstructed from the actual quadratic-constraint variable arcs by building PowerModels ACP models without solving. The loader checks both legacy CSV SHA256 hashes, the mapping manifest hash, rated-branch coverage, and endpoint topology. Unknown/changed exports fail explicitly.

## Independent label replay (first 200 samples per dataset)

The values below are mean sums over both thermal ends of absolute multiplier times squared apparent-power slack. They are diagnostics, not normalized feasibility metrics or full KKT certificates.

| Case | Scenario | Before | After |
|---|---|---:|---:|
| case30 | base | 113.99132 | 2.5933256e-05 |
| case118 | base | 4401.9992 | 0.0002467661 |
| case300 | base | 4747592.1 | 0.0090494368 |
| case118 | heavier_loads | 242047.45 | 0.030002453 |

The production loader is additionally checked across every row of all four thermal-dual datasets for exact column assignment and preservation of values. The 200-row physical replay does not establish full-dataset KKT stationarity; angle/reference duals remain unavailable and stationarity remains under investigation.

## AS impact

acopf_as.py load_active_sets uses only Pg/Qg/Vm bound duals; neither full nor pg_only reads thermal duals. No AS rerun is required for this mapping correction. The case300 unseen-active-set limitation remains unchanged.

## Direct metric

KKT training and direct evaluation now share ac_configuration/acopf_branch.py coefficients, including tap magnitude, phase shift and line charging. Direct branch violation is mean over samples of the maximum relative exceedance across branches and both ends: max(0, |S|/rate - 1). It is dimensionless (0.1 means 10% overload), matching the common PF evaluator. The legacy field direct_branch_viol_pu is retained for compatibility; it no longer contains squared-power residuals. Bus shunts belong to nodal balance, not line charging.

Old direct metrics require reevaluation; this reporting change alone does not require retraining. AC KKT supervision changes do require versioned retraining once the remaining KKT audit is cleared. No running immutable snapshot was modified.
