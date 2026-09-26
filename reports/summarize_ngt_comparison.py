"""Summarize validation-only diagnostics without treating them as paper results."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("root",type=Path)
a=p.parse_args()
rows=[]
for path in sorted(a.root.glob("*/*/completed.json")):
    folder=path.parent
    manifest=json.loads((folder/"manifest.json").read_text())
    completed=json.loads(path.read_text())
    metric=next(iter(json.loads((folder/"validation.json").read_text()).values()))
    direct=json.loads((folder/"validation_direct.json").read_text())
    rows.append({"case":manifest["experiment"]["case"],"method":manifest["experiment"]["method"],
                 "variant":manifest["variant"],"epochs":completed["epochs"],"finite_gradients":completed["finite_gradients"],
                 "pg_mae_percent":metric["mae_pg_non_slack_percent"],"pf_convergence_percent":metric["convergence_rate_percent"],
                 "branch_violation_pu":metric["mean_max_branch_viol_pu"],"generator_violation_pu":metric["mean_max_pg_viol_pu"],
                 "direct_load_loss":direct["L_d"],"direct_generator_loss":direct["L_g"],"attempt":str(folder)})
print(json.dumps(rows,indent=2))
