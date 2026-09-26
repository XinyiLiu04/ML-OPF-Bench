# Protocol notes for the new manuscripts

This is a writing input, not a results summary. Numerical conclusions await the complete versioned result collection and final audit.

## Evaluation protocol

All formal experiments use seed 42. Cross-system experiments use a 12,000-sample pool with 10,000 training, 1,000 validation, and 1,000 test samples. Training-size experiments use nested training subsets and fixed validation and test partitions. Preprocessing parameters are fitted using training data. No across-seed uncertainty is estimated.

Each case118 model is trained on the typical training distribution once. The frozen model and training preprocessing are evaluated on the typical, larger-variance, and heavier-load distributions. The heavier-load evaluation uses its own API network constraints. These evaluations are not three separately trained models.

AC accuracy, objective, and post-power-flow constraint metrics are conditional on successful power-flow convergence. Convergence coverage must accompany them. Low conditional error or a low objective value alone does not establish feasibility or success across all test samples. Training termination also does not establish optimization convergence.

## Implementation details that affect interpretation

The final AC active-set experiments use Pg-bound activity only and top-three recovery for all three systems. The earlier full Pg/Qg/Vm-label experiments are retained as superseded attempts. Their case300 unseen-class failure does not describe the final Pg-only experiment. This reduced activity representation must be disclosed in both manuscripts.

AC KKT uses a full branch pi model, including tap ratios, phase shifts, charging, and both branch ends. Existing thermal multipliers are remapped on reading using a checked mapping bound to the original CSV hashes; the original CSV files are unchanged. Stationarity includes generation and voltage derivatives and the explicit branch active/reactive-flow variable bounds used by the reference formulation. Eight additional flow-bound multiplier heads have physics supervision, without invented dual labels. Four paired diagnostic OPF samples reproduced the saved primal/dual values and identified omitted flow-bound terms. This is bounded diagnostic evidence, not an exact certificate for the whole dataset.

DC KKT evaluates complementarity and dual feasibility using physical dual variables. Multiplicative normalization preserves zero and sign; affine MinMax shifts are not used in these conditions.

The two AC NGT variants share corrected network physics, fixed demand components, data-derived angle-difference bounds, and reference-centered unbounded bus-angle outputs. The paper variant retains the repository paper loss and update implementation; the modified variant retains the modified loss and updates. Calling a variant paper does not independently establish exact reproduction of every detail in the published article. Formal variant comparisons use matching data partitions and training budgets. Short diagnostic comparisons are excluded from formal tables.

AC GNN uses Pg-only prediction with auxiliary-input generator voltages in all three systems. Distinguish this from the Pg-only activity labels of AS.

## Timing and compute

Experiments ran concurrently on three A10 instances; the concurrency cap changed during execution. Wall-clock training and inference measurements therefore reflect the recorded execution conditions. They must not be described as isolated timings.

IPOPT times are historical values from the original manuscript. The exact historical timing protocol and hardware comparability have not been independently verified. They provide context, not a controlled same-machine speedup comparison. The four KKT diagnostic solves do not supply replacement reference timings.

Any FLOP estimate must state its counting convention and coverage. Neural-network FLOPs alone exclude power-flow solves, optimization-based recovery, feature construction, and higher-order differentiation unless explicitly measured. Do not equate FLOPs with wall-clock runtime or claim compute scaling from the erroneous historical figures.

## Scaling and page allocation

Use data scaling and training dynamics as distinct analyses. Plot final held-out performance against training size; show recorded training progress against epochs or updates separately. Use comparable physical validation metrics when available, and identify method-specific training losses when they are the only histories retained. Do not fabricate dense curves from sparse log entries.

The main manuscript should include a compact selection of physically meaningful metrics and convergence coverage, without claiming a universal power law. Full tables, supplementary scaling metrics, recorded training histories, implementation audits, and compute-counting details belong in the arXiv appendix. Fits are descriptive single-seed summaries and need their fit range and limitations stated.

The submission version has no appendix and must remain within ten pages including references. The arXiv version contains the detailed appendix. Add one reference to a verified arXiv identifier when available; do not fabricate a link or identifier. Preserve the original introduction and method exposition where consistent with the implemented protocol.
