# Paper algorithm within the benchmark

Reference: Huang, Chen and Low, IEEE TPWRS 2024, DOI 10.1109/TPWRS.2024.3373399.

`variant=paper` implements the published training equations independently of the modified implementation:

- Algorithms 1/2 use plain minibatch SGD, fixed epochs and final weights. Validation is diagnostic only.
- For epochs after the first, Eq.12 weights are calculated from the current minibatch before its gradient update. The ratio is detached and uses the signed objective, without absolute value, EMA, first-epoch normalization, or violation boosting. Algorithm 2 updates weights in both stages.
- E-NGT uses disjoint labeled/unlabeled partitions within the benchmark training split. Eq.13 includes the constraint terms, including load mismatch. Eq.14 covers all reconstructed buses, with reference-aligned angle labels.
- Eq.9 uses squared physical load mismatch. Thermal penalties sum the two directed branch-end violations, rather than selecting only the larger one.
- Hidden layers use ReLU and all output coordinates use sigmoid. Magnitudes map to the actual bus limits.

Explicit implementation conventions where the article does not specify sufficient detail:

- Sigmoid angle coordinates map to [-pi, pi], followed by subtraction of the reference angle. This is a periodic coordinate chart, not a branch-angle constraint. Branch angle limits remain the independent bounds from CSV; no +/-30 degree bus restriction is imposed. The article does not state the affine angle scaling, so exact author-code equivalence is not claimed.
- For exactly zero constraint loss, Eq.12 uses the configured finite coefficient cap; its residual and squared-loss gradient are zero. Nonzero losses use the literal signed-cost ratio. A negative reconstructed generation cost can consequently produce negative weights, as the literal equation implies outside the intended cost regime; no undocumented absolute-cost modification is applied. Nonfinite losses fail explicitly.
- The existing initial weights and coefficient cap are retained as exposed parameters. The paper says caps are tuned but does not provide all values needed to establish an identical configuration.

Benchmark adaptations retained intentionally:

- Shared hidden widths, batch sizes, learning-rate input, dataset, training-only scalers, seed, splits, fixed budgets, CLI, checkpoint serialization and common PF evaluation.
- Full pi-model, transformers, shunts and actual CSV limits replace the paper's simplified network presentation for these data.
- The benchmark's final PF pipeline is not the paper's direct algebraic evaluation and voltage postprocessing. Standalone script evaluation remains algebraic without that additional paper postprocessing. Neither route establishes reproduction of the article's reported performance or speedups.

The historical paper checkpoints were trained with earlier code. Changing this source does not retroactively change those results. Full retraining is a separate experiment. The benchmark's selected modified fixed-budget results remain unchanged.

Observed limitation: bounded diagnostics and the cancelled benchmark trials showed large initial SGD updates, sigmoid saturation and stalled training under these settings. Negative reconstructed costs can also yield negative Eq.12 constraint weights. This variant is provided for algorithm comparison, not as the implementation behind the benchmark's reported modified fixed-budget results.
