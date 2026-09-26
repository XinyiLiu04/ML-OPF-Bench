# AC KKT stationarity audit and experiment disposition

## Decision

Implementation gradient checks pass; the paired primal/dual label audit does not pass. Keep supervised-dual AC KKT formal runs quarantined. Do not infer label correctness from finite training or checkpoint replay.

## Implementation checks

- Independently reconstructed NumPy/PyPower Lagrangian finite differences agree with PyTorch voltage and generation gradients on case30/118/300, including both branch ends, taps, phase shift, charging, bus shunts, bounds, angle constraints and reference angle.
- Analytic one-bus quadratic-cost KKT and backward checks pass.
- Fixed a remaining reference condition: Im(Vref)=0 also admits a negative real reference voltage. The constraint now uses atan2(Im(Vref),Re(Vref))=0; regression rejects the negative-real case.
- Optional float64 parameter and layer-buffer construction permits an audit without pre-rounding coefficients to float32. Default training dtype is unchanged.
- 20 regression/protocol tests pass. A two-epoch, pool120 case30 CPU execution preflight completed; it is not a paper result.

## Existing-label replay

First200 rows per case, double-precision coefficients, corrected thermal mapping. Entries are maximum absolute gradient components over these rows. They carry the objective/Lagrangian units and are not normalized performance scores.

| Dataset | Generation | Voltage magnitude | Non-reference angle after allowing missing active-angle multipliers |
|---|---:|---:|---:|
| case30 / base | 9.094947e-13 | 105.26884 | 270.11361 |
| case118 / base | 3.6379788e-12 | 0.0040801742 | 0.011338371 |
| case300 / base | 1.8626451e-09 | 5.4478388 | 13.583338 |
| case118 / heavier_loads | 7.2759576e-12 | 558.53052 | 1765.8315 |

No branch-angle bound is within1e-5 radians of active in these audited rows, so missing angle-bound multipliers do not explain these residuals under complementary slackness. A missing reference multiplier cannot cancel non-reference angle components. Double precision removes most generation residuals but does not eliminate the large voltage outliers, especially case30/API.

The dual exporter in data_generalization/ACOPF Constraints with duals.ipynb independently solves OPF from saved loads and writes the resulting duals without saving its matching primal voltages/generation. Existing copied primal files match the canonical originals in checked case30/118/300 folders. This is a concrete provenance risk, not proof of the sole cause of the residual. The mapping is well-supported by complementarity replay; it does not establish joint stationarity. No OPF labels were generated or changed during this audit.

Worst angle residual rows (zero-based within the inspected prefix): case30=47, case118=130, case300=50, API=135. Preserve these as targeted provenance diagnostics. Do not suppress the residual, fit arbitrary multipliers on inactive bounds, or silently drop dual supervision to obtain a passing result.

## Rerun scope

| Family | Scope | Action |
|---|---|---|
| DC KKT | Three cross-system plus six case118 scaling configurations | Corrected versioned execution already authorized; preserve old attempts |
| AC NGT/E-NGT | Three cases for each method | Corrected-physics modified variants; first west queue active; case118 model reused for both shifts |
| AC KKT | Three cross-system plus six case118 scaling configurations | Hold until paired-label issue is resolved or an explicitly documented alternative protocol is selected |
| AC KKT direct metric | Saved predictions/checkpoints | Reevaluate full pi-model two-end relative overload; this alone requires no retraining |
| AC AS | Thermal-label mapping change | No retraining: implementation does not consume thermal duals |
| Other AC/DC methods | No demonstrated training impact from these targeted changes | Continue existing immutable-source runs and independent result audits |

Counts describe configurations, including ones not yet run; they are not counts of completed experiments that all need repeating. The original AC300AS unavailable outcome remains. No extra seeds, no DC RL, no snapshot edits.
