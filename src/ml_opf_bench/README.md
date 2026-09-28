# Core modules

| Module | Purpose |
| --- | --- |
| `__init__.py` | Public Python API |
| `datasets.py`, `splits.py` | Dataset loading and train/validation/test splits |
| `methods.py`, `registry.py` | Method interfaces, registration, and loading |
| `config.py`, `runtime.py` | Experiment settings and training state |
| `runner.py`, `cli.py` | Python and command-line execution |
| `evaluation.py`, `ac_evaluation.py`, `dc_evaluation.py` | Shared evaluation and AC/DC adapters |
| `io.py` | Results, configuration, and provenance |

Method-specific helpers live with their implementations:
[AC GNN](../../ac_methods/acopf_gnn/),
[AC RL](../../ac_methods/acopf_rl/), and
[DC RL](../../dc_methods/dcopf_rl/).
