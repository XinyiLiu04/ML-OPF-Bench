"""Build paper artifacts exclusively from a complete, version-consistent suite."""

import argparse
import json
import hashlib
import re
from pathlib import Path

import numpy as np

from dataclasses import replace
from ml_opf_bench.config import paper_experiments, Experiment


KEYS = {
    "dc": {"mae_pg": "mae_pg_non_slack", "viol_pg": "viol_pg_non_slack",
           "viol_slack": "viol_pg_slack", "viol_line": "viol_branch", "gap": "cost_gap_percent"},
    "ac": {"mae_pg": "mae_pg_non_slack_percent", "mae_vm": "mae_vm_percent",
           "viol_pg": "mean_pg_viol_non_slack_pu", "viol_slack": "mean_pg_viol_slack_pu",
           "viol_qg": "mean_max_qg_viol_pu", "viol_vm": "mean_max_vm_viol_pu",
           "viol_line": "mean_max_branch_viol_pu", "gap": "cost_optimality_gap_percent",
           "convergence": "convergence_rate_percent"},
}
LABELS = {"mae_pg": r"MAE $P_g$ (\%)", "mae_vm": r"MAE $V$ (\%)",
          "viol_pg": r"Viol $P_g$", "viol_slack": r"Viol $P_g^0$",
          "viol_qg": r"Viol $Q_g$", "viol_vm": r"Viol $V$", "viol_line": "Viol line",
          "gap": r"Gap (\%)", "convergence": r"PF (\%)",
          "inference_ms": "Infer (ms)", "train_time_s": "Train (s)"}


def read(path):
    return json.loads(path.read_text())


SOURCES = {
    "rl": ("paper-seed42-rl-5f2e191", "853109cd9c550fc6ae5a2cdfd92e9b00ecff2544450bbce85c5a574fa0c45c71"),
    "original": ("paper-seed42-2c1d52c", "39caf5accb8e14f0ae2ae179955629ce16e50491db439714d42930542ca5c0c0"),
    "dc_kkt": ("paper-seed42-correction-a824a10", "6956a7091fef4522a39060ff3f17d9f59104fa152fb199cad783d79e429a01f2"),
    "ac_kkt": ("paper-seed42-kkt-43b8584", "be0e5c8b1e6dee8c68a0bdfbd1c816b0076c2e1cfa9d7ae0292948d362440e68"),
    "ac_as": ("paper-seed42-as-pgonly-606ced8", "507e1986f14cb7a3efc051a0e811f438b1a89982783b687b88081d9edcd8de82"),
    "ac_ngt": ("paper-seed42-ngt-physical-v2", "97b710dcafb13be5361ba7c4c17d403a4d0dea3a5400e47a1488bb6a9294049b"),
    "paper": ("paper-seed42-ngt-paper-b2b362d", "59c274708d9291ad7e11e7d8f374ac2c34edf62c5e27be9ee649a0ed4fe216ea"),
}


def source_key(spec):
    if spec.method == "RL":
        return "rl"
    if spec.variant == "paper":
        return "paper"
    if spec.method == "KKT":
        return spec.formulation + "_kkt"
    if spec.formulation == "ac" and spec.method == "AS":
        return "ac_as"
    if spec.formulation == "ac" and spec.method in ("NGT", "E-NGT"):
        return "ac_ngt"
    return "original"


def collect(root, source_sha, seed):
    rows = []
    missing = []
    specs = list(paper_experiments(seed))
    if source_sha is not None:
        specs = [replace(spec, variant="modified") if spec.method == "RL" else spec for spec in specs]
    if source_sha is None:
        specs += [Experiment("dc", "RL", case, seed=seed) for case in ("case30", "case118", "case300")]
        specs += [replace(spec, variant="paper") for spec in specs
                  if spec.formulation == "ac" and spec.method in ("NGT", "E-NGT")
                  and spec.mode == "cross-system"]
    for spec in specs:
        if source_sha is None:
            folder, expected_sha = SOURCES[source_key(spec)]
            run_root = root / folder
        else:
            run_root, expected_sha = root, source_sha
        attempts = []
        for marker in (run_root / spec.run_id).glob("*/completed.json"):
            manifest = read(marker.parent / "manifest.json")
            experiment = {"variant": "modified"} | manifest["experiment"]
            expected = spec.as_dict() | {"workers": experiment["workers"]}
            if manifest["code"]["source_sha256"] == expected_sha and experiment == expected:
                attempts.append(marker.parent)
        if not attempts:
            failures = []
            for marker in (run_root / spec.run_id).glob("*/failed.json"):
                manifest = read(marker.parent / "manifest.json")
                experiment = {"variant": "modified"} | manifest["experiment"]
                if (source_sha is not None and spec.method == "AS" and manifest["code"]["source_sha256"] == expected_sha
                        and experiment == spec.as_dict() | {"workers": experiment["workers"]}
                        and read(marker)["error"].startswith("No validation sample shares an active set with training;")):
                    failures.append(marker)
            if failures:
                marker = sorted(failures)[-1]
                scenarios = ["base"] + (["larger_variance", "heavier_loads"] if spec.case == "case118" else [])
                for scenario in scenarios:
                    rows.append({"formulation": spec.formulation, "case": spec.case, "mode": spec.mode,
                                 "train_size": spec.train_size, "scenario": scenario, "method": spec.method,
                                 "attempt": str(marker.parent), "source_sha256": expected_sha,
                                 "status": "unavailable", "reason": read(marker)["error"],
                                 "epochs": None, "steps": None, "raw_metrics": {},
                                 "metrics": dict.fromkeys([*KEYS[spec.formulation], "inference_ms", "train_time_s", "viol_total"])})
                continue
            missing.append(spec.run_id)
            continue
        if len(attempts) != 1:
            raise ValueError(f"Ambiguous accepted attempts: {spec.run_id}")
        attempt = attempts[0]
        metadata = read(attempt / "checkpoint_metadata.json")
        scenarios = ["base"]
        if spec.case == "case118" and spec.mode == "cross-system":
            scenarios += ["larger_variance", "heavier_loads"]
        if read(attempt / "completed.json")["scenarios"] != scenarios:
            raise ValueError(f"Incomplete scenarios: {attempt}")
        for scenario in scenarios:
            for method, metrics in read(attempt / f"{scenario}.json").items():
                normalized = {key: metrics[value] for key, value in KEYS[spec.formulation].items()}
                normalized["feasibility"] = metrics.get("feasible_rate_percent")
                normalized.update({key: metrics[key] for key in ("inference_ms", "train_time_s")})
                normalized["viol_total"] = sum(normalized[k] for k in normalized if k.startswith("viol_")) \
                    if all(normalized[k] is not None for k in normalized if k.startswith("viol_")) else None
                rows.append({"formulation": spec.formulation, "case": spec.case, "mode": spec.mode,
                             "train_size": spec.train_size, "scenario": scenario, "method": method + (" (paper)" if spec.variant == "paper" else ""),
                             "attempt": str(attempt), "source_sha256": expected_sha,
                             "status": "completed", "variant": spec.variant, "run_id": spec.run_id,
                             "epochs": metadata["epochs_completed"], "steps": metadata["environment_steps"],
                             "metrics": normalized, "raw_metrics": metrics})
    if missing:
        raise ValueError("Formal suite is incomplete:\n" + "\n".join(missing))
    return rows


def number(value):
    if value is None or not np.isfinite(value):
        return "NR"
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 10000:
        return f"{value:.2e}"
    return f"{value:.3f}"


def collect_historical_references(source):
    raw = source.read_bytes()
    text = raw.decode()
    references = []
    for form, label in (("dc", "dc_stats"), ("ac", "ac_stats")):
        start = text.index("\\label{tab:" + label + "}")
        end = text.index("\\end{table", start)
        lines = [line for line in text[start:end].splitlines() if "Solver (IPOPT)" in line]
        if len(lines) != 1:
            raise ValueError(f"Expected one historical IPOPT row: {form}")
        values = re.findall(r"(?<![A-Za-z])\d+\.\d+", lines[0])
        if len(values) != 3:
            raise ValueError(f"Expected three historical timings: {form}")
        for case, value in zip(("case30", "case118", "case300"), values):
            timing = float(value)
            if not np.isfinite(timing) or timing <= 0:
                raise ValueError(f"Invalid historical timing: {form}/{case}")
            references.append({"formulation": form, "case": case, "inference_ms": timing,
                               "timing_kind": "historical_reported", "source": str(source),
                               "source_sha256": hashlib.sha256(raw).hexdigest(),
                               "source_table": label, "source_row": lines[0],
                               "same_hardware_verified": False,
                               "n_samples": None, "label_comparison": None})
    return references


def tables(rows, references, output):
    text = [r"\documentclass{article}", r"\usepackage[margin=12mm,landscape]{geometry}",
            r"\usepackage{booktabs,longtable,amsmath}", r"\begin{document}",
            r"\section*{Updated benchmark results: seed 42}",
            "NR denotes an unavailable metric. AC accuracy, cost and violations are conditional on "
            "power-flow convergence; the PF column gives the coverage. A small or negative cost gap "
            "does not establish feasibility. Line violations are relative to capacity; other violations "
            "are in per unit. Timings include preprocessing and method postprocessing. "
            "Epochs are executed training epochs; RL reports environment steps. "
            "No uncertainty across seeds is claimed."]
    limitations = sorted({r["case"] + " " + r["formulation"].upper() + " " + r["method"] + ": " + r["reason"]
                          for r in rows if r["status"] == "unavailable"})
    text += [r"\paragraph{Unavailable methods.}" + " ".join(limitations)] if limitations else []
    for form in ("dc", "ac"):
        columns = list(KEYS[form]) + ["inference_ms", "train_time_s"]
        for mode in ("cross-system", "scaling"):
            subset = [r for r in rows if r["formulation"] == form and r["mode"] == mode]
            if mode == "scaling":
                subset = [r for r in subset if r["scenario"] == "base" and r["method"] != "CP"]
            text += [rf"\section*{{{form.upper()} {mode}}}", r"\scriptsize",
                     r"\begin{longtable}{lll" + "r" * (len(columns) + 1) + "}", r"\toprule",
                     "Case/size & Scenario & Method & " + " & ".join(LABELS[k] for k in columns)
                     + r" & Epoch/steps \\", r"\midrule\endhead"]
            for row in sorted(subset, key=lambda r: (int(r["case"][4:]), r["train_size"] or 0,
                                                       r["scenario"], r["method"])):
                size = row["train_size"] if mode == "scaling" else row["case"][4:]
                scenario = row["scenario"].replace("_", " ")
                count = str(row["epochs"]) if row["epochs"] is not None else "--"
                if row["steps"] is not None:
                    count = f"{row['steps']} steps"
                text.append(f"{size} & {scenario} & {row['method']} & "
                            + " & ".join(number(row["metrics"][k]) for k in columns) + f" & {count}" + r" \\")
            text += [r"\bottomrule\end{longtable}", r"\normalsize"]
    text += [r"\section*{Historical IPOPT reference timing}",
             "These values are reproduced from the original manuscript, not remeasured in this run. "
             "The original manuscript states an Intel Xeon Platinum 8358 CPU and NVIDIA A10, "
             "but the exact timing protocol, sample count and execution conditions have not been independently verified. "
             "They are historical context, not a controlled same-hardware speedup comparison. "
             "These timings are not derived from the separate four-sample KKT diagnostic solves; original labels were preserved.",
             r"\begin{longtable}{llr}\toprule",
             r"Formulation & Case & Historical reported time (ms) \\",
             r"\midrule\endhead"]
    for row in references:
        text.append(f"{row['formulation'].upper()} & {row['case'][4:]} & "
                    + number(row["inference_ms"]) + r" \\")
    text += [r"\bottomrule\end{longtable}", r"\end{document}"]
    (output / "results_update.tex").write_text("\n".join(text) + "\n")


def plots(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fits = []
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for i, form in enumerate(("ac", "dc")):
        subset = [r for r in rows if r["formulation"] == form and r["mode"] == "scaling"
                  and r["scenario"] == "base" and r["method"] != "CP"]
        for j, key in enumerate(("mae_pg", "viol_pg", "viol_line", "train_time_s")):
            ax = axes[i, j]
            for method in sorted({r["method"] for r in subset}):
                values = sorted([r for r in subset if r["method"] == method], key=lambda r: r["train_size"])
                x = np.array([r["train_size"] for r in values], dtype=float)
                y = np.array([r["metrics"][key] if r["metrics"][key] is not None else np.nan for r in values])
                valid = np.isfinite(y) & (y > 0)
                line, = ax.plot(x[valid], y[valid], "o", label=method)
                fit = {"formulation": form, "method": method, "metric": key,
                       "positive_points": int(valid.sum()), "exponent": None, "r_squared": None}
                if valid.sum() >= 3:
                    slope, intercept = np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)
                    predicted = intercept + slope * np.log(x[valid])
                    ss_total = np.sum((np.log(y[valid]) - np.log(y[valid]).mean()) ** 2)
                    fit.update(exponent=float(slope), r_squared=float(1 - np.sum((np.log(y[valid]) - predicted) ** 2) / ss_total)
                               if ss_total > 0 else None)
                    ax.plot(x[valid], np.exp(predicted), color=line.get_color(), linewidth=1)
                fits.append(fit)
            ax.set(xscale="log", yscale="log", xlabel="Training samples", title=f"{form.upper()} {key}")
            ax.grid(alpha=0.2)
            if j == 0:
                ax.legend(fontsize=7)
    fig.savefig(output / "scaling_combined.pdf")
    plt.close(fig)
    for scenario in ("larger_variance", "heavier_loads"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for i, form in enumerate(("ac", "dc")):
            selected = [r for r in rows if r["formulation"] == form and r["mode"] == "cross-system"
                        and r["case"] == "case118" and r["scenario"] in ("base", scenario)]
            methods = sorted({r["method"] for r in selected})
            for j, key in enumerate(("mae_pg", "viol_total", "gap")):
                ax = axes[i, j]
                for scenario_name, offset in (("base", -0.18), (scenario, 0.18)):
                    values = [next(r["metrics"][key] for r in selected if r["method"] == method
                                   and r["scenario"] == scenario_name) for method in methods]
                    ax.bar(np.arange(len(methods)) + offset,
                           [abs(v) if v is not None else np.nan for v in values], width=0.36,
                           label=scenario_name.replace("_", " "))
                ax.set_xticks(np.arange(len(methods)), methods, rotation=60, ha="right")
                ax.set(title=f"{form.upper()} {'absolute cost gap' if key == 'gap' else key}", yscale="symlog")
                ax.grid(axis="y", alpha=0.2)
                if j == 0:
                    ax.legend(fontsize=8)
        fig.savefig(output / f"generalization_{scenario}.pdf")
        plt.close(fig)
    (output / "scaling_fits.json").write_text(json.dumps(fits, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--source-sha", help="Legacy single-source mode; omit for the versioned 126-run protocol")
    parser.add_argument("--historical-manuscript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = collect(args.runs.resolve(), args.source_sha, args.seed)
    references = collect_historical_references(args.historical_manuscript.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (args.output / "reference_timings.json").write_text(json.dumps(references, indent=2, allow_nan=False) + "\n")
    tables(rows, references, args.output)
    plots(rows, args.output)


if __name__ == "__main__":
    main()
