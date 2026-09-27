# RL reward and protocol audit

The running seed42 baseline remains unchanged. This audit does not establish that any proposed replacement will work better. Test results are descriptive evidence, not a tuning objective.

## Confirmed implementation findings

- Managed runs use physical action bounds and one-step episodes. Each reset samples a training load; each action sets non-slack Pg and generator-bus Vm, runs PF, receives reward, and terminates.
- The formal budget is 10,000,000 environment steps, derived from 10,000 training samples and 1,000 configuration epochs. PPO internally uses three update epochs. These are different quantities; the inherited budget is not evidence of an appropriate compute budget.
- Summation reward averages separately standardized negative cost and negative summed violations with weight 0.5. Its `valid` argument does not alter either term. Consequently feasibility is not prioritized by construction.
- Penalty includes Pg, Qg, Vm, and both-end apparent-power thermal overloads. It omits branch angle-difference constraints. It sums heterogeneous categories and system-size-dependent counts.
- Random-action normalization uses only converged probes. Case300 logged one converged probe out of 500; both empirical standard deviations are replaced by 1. This is a degenerate calibration, not a reliable scale estimate. The fixed nonconvergence reward -10 is not guaranteed to be below all converged rewards because standardized rewards are unbounded.
- Validation stopping monitors only mean reward, patience20, min_delta1e-6. A small improvement resets patience. The best-reward policy is restored, not necessarily the best-feasibility policy.
- PF exceptions are collapsed into the same nonconvergence reward, obscuring numerical failure versus implementation exceptions.
- Standalone main messages about ignored early stopping/final-policy retention are stale; managed behavior uses the validation callback. Standalone action-bound default differs from managed physical bounds.

## Existing evidence and limits

Case300 stopped at 258049 steps after validation reward stayed -10 for 20 checks. Its saved test PF convergence is zero. Case30 stopped at 761857 steps; saved PF convergence is 100%, but dispatch errors and violations are substantial. Neither result establishes OPF feasibility. Case118 remained running when inspected; validation reward improvements alone do not establish a valid dispatch. CPU utilization indicates active execution, not algorithmic progress toward feasibility.

Clipped Gaussian PPO actions on a [0,1] box may concentrate at boundaries; quantify this from saved policies and validation loads before attributing failure to clipping. Existing checkpoints omit reward calibration parameters and validation histories as structured artifacts; textual logs remain available.

## Proposed separately versioned revision

1. Keep physical bounds, original splits, seed42, and one-step definition. Preserve all original attempts.
2. Expose separate Pg/Qg/Vm/thermal/angle violation components and PF failure categories. Declare tolerances and scaling explicitly; do not silently mix units into a feasibility certificate.
3. Reject inadequate reward calibration rather than substitute unit scales. Prefer fixed, interpretable scales determined from training constraints and predeclared training-only diagnostics.
4. Use validation feasibility/violation together with cost for checkpoint selection, with a documented ordering. Record PF convergence separately from full feasibility.
5. Declare an explicit bounded step budget and validation schedule before a revised formal run; select them from training/validation diagnostics, not test outcomes. Do not claim equal epochs imply equal compute.
6. Record calibration, learning curves, actual steps, wall time, action boundary fractions, and validation component metrics. Treat action parameterization changes as an explicit variant.

No revised formal RL training has been launched by this audit. Reward choices and budget still require validation-only diagnostics before fixing the revised protocol.

## Bounded validation diagnostics

The first 32 saved validation samples (explicitly disjoint from test) were evaluated without training. On case30, policy, midpoint, and Pg-lower/Vm-midpoint controls all had 32 converged PF solves, but all displayed violations. Raw policy means were outside [0,1] for 0.7215909091 of entries. This is evidence of substantial clipping, not proof of its causal role. These validation observations must not be conflated with the previously inspected test samples.

On case300, all three controls had zero converged PF solves among 32 validation samples. A simple midpoint initialization therefore does not resolve the failure on this diagnostic subset. The next revision must first establish a training-only feasible exploration/initialization strategy and record failure categories; changing only the penalty coefficient is not yet supported. Frozen evaluator input uses the same saved x scaler and action_to_setpoints mapping as the diagnostic. No revised training has started.

## Validation PF initialization diagnosis

The first 32 saved case300 validation samples were replayed with original frozen source and verified data signatures. Label non-slack Pg and generator Vm yielded 32/32 converged PF solves for each of CSV, flat, and label voltage initializations, with both 10 and 50 Newton iterations. Maximum recovered bus-Vm discrepancy from labels was 1.216716940533047e-07 p.u. Policy and physical-box midpoint setpoints yielded 0/32 for every corresponding initialization/iteration setting. Q-limit enforcement was disabled, as in the formal evaluator; tolerance was 1e-8.

Thus these samples do not support blaming the default initial voltage or ten-iteration cap alone. The label control supports consistency of this bounded load/setpoint/PF path, not a full dataset correctness certificate. Newton failure does not establish mathematical infeasibility. Next protocol work should address coupled action feasibility and reward calibration rather than only increasing the iteration cap. Label warm starts here are diagnostic controls only and must not enter deployable inference or training. No revised training was launched.

Evidence: `runs/corrections-v1/rl-pf-initialization-case300-v1.json`; tool: `reports/diagnose_rl_pf_initialization.py`.

## Label-free aggregate-dispatch diagnostic

A bounded follow-up used the first 32 saved training loads and first 32 validation loads, disjoint from test. For every sample, total demand includes unchanged fixed bus loads. All-generator dispatch is pg_min + alpha*(pg_max-pg_min), with alpha chosen for total demand times (1+margin); only non-slack dispatch is sent to PF. Slack generation remains determined by PF. Margins 0 and 0.05 and CSV/midpoint generator voltages were compared. Neither OPF labels nor an optimization solver supply the actions.

At zero margin, 25/32 training and 24/32 validation cases converged with either voltage setting. At 0.05 margin, both subsets converged 32/32 with either voltage setting. The CSV-voltage legacy violation sums averaged 43.94575016838418 (train) and 44.27831056814388 (validation); these mixed-unit sums are diagnostic only, not feasibility metrics. Constraints are still violated. The assumed five-percent margin is not measured losses or an established optimum. This bounded observation supports investigating demand-aware action initialization, not a feasibility guarantee or performance claim.

Next gate before revised training: test bounded perturbations around the aggregate-dispatch center on training loads, report Pg/Qg/Vm/thermal/angle components separately, and predeclare the action map and bounded reward. Any demand-aware map is a new RL variant and must be used identically in training, checkpoint reload and evaluation. Preserve the original independently bounded PPO baseline. No revised training has started.

Evidence: `runs/corrections-v1/rl-balanced-case300-v1.json`; script: `reports/diagnose_rl_balanced_dispatch.py`.

## Approved DDPG Pg-only implementation

The user rejected demand-matched dispatch and generation margins. Those controls remain diagnostic history only and are not part of the implemented method. New variant `ddpg-pgonly` controls only physical-bound non-slack Pg. Generator voltages are fixed to `gen_data.csv:vg_pu`; the evaluator reads each scenario's network CSV. Slack Pg and Qg are determined by AC PF, without an auxiliary optimization solver or label-based action initialization.

Implementation: `ac_methods/acopf_rl/acopf_ddpg_pgonly.py`, selected via `--method RL --variant ddpg-pgonly`. Run IDs and checkpoints are separate from original PPO. Default explicit budget is 2,000,000 steps; DDPG uses single-step gamma=0, Gaussian noise std0.1, replay capacity1,000,000, and benchmark network widths/batches. These are project adaptations, not an exact reproduction of the reference paper's hyperparameters. The inherited Summation penalty, validation reward selection, and exception handling remain pending the separately discussed reward audit; the calibration now rejects fewer than32 converged probes. No formal revised experiment is launched by this implementation.

Tests check physical-bound action mapping, fixed Vm independence from actions, one-step environment behavior, separate run IDs, and a small DDPG update followed by serialized policy reload equality. These software tests do not establish AC-OPF feasibility or training convergence.

## Pg-only bounded reward v2 implementation and software smoke

The new variant now uses `BoundedSummation` with a cost scale computed from absolute polynomial cost coefficients and physical Pg bounds. Cost contributes -0.5*(1+tanh(cost/scale)); the sum of per-category mean violations contributes -p/(1+p), with explicit unit scales of one pu, relative thermal loading, and one radian. Thus converged rewards lie within [-2,0], above PF failure -3. No random-probe normalization or OPF labels determine this reward. This remains a soft trade-off and is not the reference paper's exact reward or a feasibility guarantee.

Separate maximum/mean violations cover all-generator Pg and Qg, all-bus Vm, both-end thermal limits, and independent actual branch angle bounds. Feasibility uses a declared 1e-5 tolerance in each category's respective units; this threshold is a project protocol choice, not a grid-code claim. Training PF exceptions now surface rather than being silently classified as divergence. Validation histories record PF success and feasibility separately; checkpoint selection still uses validation reward and must not be described as feasibility-first.

Local real-data case30 software smoke:128 environment steps with learning_starts16, network16x16,batch16; first16 validation samples only. PF16/16, feasibility0/16, reload action arrays identical. This validates execution and serialization, not training effectiveness. Record: runs/corrections-v1/ddpg-pgonly-validation-smoke-v2.json. Unit/protocol suite16passed. No revised formal matrix launched.

## Larger-system software verification

The same bounded128-step DDPG check was executed on case118 and case300, with seed42,16x16 network,batch16,learning_starts16. It uses the formal split and only first16 validation samples for post-training checks, asserts no test overlap, and saves independent diagnostic checkpoints. case118 action shape16x53:PF16/16,feasible0/16. case300 action shape16x68:PF0/16,feasible0/16. Both policy reload comparisons were exact. These results establish execution/serialization only;128steps cannot establish learning convergence. Pg-only has not by itself removed case300 PF failures. No demand matching, margins, or label warm starts were introduced.

Reproducible tool: reports/check_ddpg_pgonly.py --case case118 (or case300), requires the repository data root as cwd and refuses to overwrite existing outputs. Outputs: runs/corrections-v1/ddpg-pgonly-{case}-validation-smoke-v3.json and .pt. Formal training still needs a frozen protocol and must preserve all failures.
