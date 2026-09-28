"""Shared evaluation results and physical setpoint evaluation."""

from dataclasses import dataclass

import numpy as np

from .io import json_value
from .methods import Prediction


_AC_METRICS = {
    'mae_pg_non_slack_percent': ('mae_pg_non_slack_percent', '%'),
    'mae_pg_slack_percent': ('mae_pg_slack_percent', '%'),
    'mae_pg_all_percent': ('mae_pg_all_percent', '%'),
    'mae_vm_percent': ('mae_vm_percent', '%'),
    'mae_qg_percent': ('mae_qg_percent', '%'),
    'mae_va_deg': ('mae_va_deg', 'degree'),
    'cost_optimality_gap_percent': ('cost_gap_percent', '%'),
    'cost_true_mean': ('cost_reference_mean', '$/h'),
    'cost_pred_mean': ('cost_prediction_mean', '$/h'),
    'mean_max_pg_viol_pu': ('mean_max_pg_viol_pu', 'p.u.'),
    'mean_pg_viol_non_slack_pu': ('mean_pg_viol_non_slack_pu', 'p.u.'),
    'mean_pg_viol_slack_pu': ('mean_pg_viol_slack_pu', 'p.u.'),
    'mean_max_qg_viol_pu': ('mean_max_qg_viol_pu', 'p.u.'),
    'mean_max_vm_viol_pu': ('mean_max_vm_viol_pu', 'p.u.'),
    'mean_max_branch_viol_pu': ('mean_max_branch_viol_ratio', 'rating ratio'),
}
_DC_METRICS = {
    'mae_pg_non_slack': ('mae_pg_non_slack_percent', '%'),
    'mae_pg_slack': ('mae_pg_slack_percent', '%'),
    'viol_pg_non_slack': ('mean_pg_viol_non_slack_pu', 'p.u.'),
    'viol_pg_slack': ('mean_pg_viol_slack_pu', 'p.u.'),
    'viol_branch': ('mean_max_branch_viol_ratio', 'rating ratio'),
    'viol_balance': ('mean_balance_residual_pu', 'p.u.'),
    'cost_gap_percent': ('cost_gap_percent', '%'),
}


def evaluation_indices(n_total, indices=None, limit=None):
    rows = np.arange(n_total) if indices is None else np.asarray(indices)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu':
        raise ValueError('indices must be a one-dimensional integer array')
    if np.any(rows < 0) or np.any(rows >= n_total) or len(np.unique(rows)) != len(rows):
        raise ValueError('indices must be unique and within the dataset')
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('eval_limit must be a positive integer')
        rows = rows[:limit]
    if not len(rows):
        raise ValueError('evaluation requires at least one sample')
    return rows


@dataclass(frozen=True)
class EvaluationResult:
    formulation: str
    records: dict
    samples: dict

    def to_dict(self):
        mapping = _AC_METRICS if self.formulation == 'ac' else _DC_METRICS
        outputs = {}
        n = len(self.samples['indices'])
        for name, record in self.records.items():
            valid = self.samples.get(f'{name}_valid', self.samples.get('input_valid', np.ones(n, dtype=bool)))
            pg = self.samples[f'{name}_pg_pred']
            finite = np.isfinite(pg).all(axis=1)
            converged = self.samples.get(f'{name}_converged')
            outputs[name] = {
                'metrics': {new: record[old] for old, (new, _) in mapping.items() if old in record},
                'units': {new: unit for old, (new, unit) in mapping.items() if old in record},
                'coverage': {
                    'n_samples': n,
                    'n_prediction_valid': int(valid.sum()),
                    'n_finite_dispatch': int(finite.sum()),
                    'n_converged': None if converged is None else int(converged.sum()),
                    'n_quality_samples': int(converged.sum()) if converged is not None else n,
                    'quality_population': 'PF-converged samples' if converged is not None else 'all samples',
                },
                'details': {key: value for key, value in record.items() if key not in mapping},
            }
            if 'conditional_metrics' in record:
                conditional = record['conditional_metrics']
                outputs[name]['conditional_metrics'] = {
                    'population': 'finite valid dispatches',
                    'n_samples': int((valid & finite).sum()),
                    'metrics': {new: conditional[old] for old, (new, _) in mapping.items() if old in conditional},
                }
                outputs[name]['details'].pop('conditional_metrics')
        return json_value({
            'schema_version': 1,
            'formulation': self.formulation,
            'definitions': {
                'mae_percent': '100 * mean(abs(pred-reference)) / (mean(abs(reference)) + 1e-8)',
                'cost_gap_percent': ('100 * mean((pred-reference)/(reference+1e-8))' if self.formulation == 'ac'
                                     else '100 * mean((pred-reference)/(abs(reference)+1e-8))'),
                'violations': 'mean of per-sample maxima; balance residual is a per-sample absolute sum residual',
                'reference_global_optimality': 'not certified by this evaluator',
                'pf_convergence_certifies_opf_feasibility': False,
            },
            'outputs': outputs,
        })


def evaluate_predictions(partition, prediction: Prediction, *, name='prediction'):
    """Evaluate p.u. setpoints with AC power flow or DC slack reconstruction."""
    inputs, targets = partition.inputs, partition.targets
    prediction.validate(inputs)
    if not len(inputs.indices):
        raise ValueError('evaluation requires at least one sample')
    params = inputs.params
    valid = np.ones(len(inputs.indices), dtype=bool) if prediction.valid is None else prediction.valid.copy()
    pg_set = prediction.pg.copy()
    pg_set[~valid] = np.nan
    arrays = {'indices': inputs.indices.copy(), 'input_valid': valid, 'pg_setpoints': pg_set, 'pd': inputs.pd}
    if inputs.formulation == 'dc':
        from dc_methods.dc_configuration.dcopf_data_setup import reconstruct_full_pg
        from dc_methods.dc_configuration.dcopf_evaluation_metrics import evaluate_dispatch
        pg = reconstruct_full_pg(pg_set, inputs.pd, params)
        metrics = evaluate_dispatch(pg, targets['pg'], inputs.pd, params)
        if not valid.all():
            metrics['conditional_metrics'] = (
                evaluate_dispatch(pg[valid], targets['pg'][valid], inputs.pd[valid], params)
                if valid.any() else {key: np.nan for key in metrics})
        arrays.update(pd=inputs.pd, pg_true=targets['pg'])
    else:
        from ac_methods.ac_configuration.acopf_pypower import load_case_from_csv, solve_pf_setpoints
        from ac_methods.ac_configuration.acopf_evaluation_metrics import evaluate_acopf_predictions
        if partition.paths is None:
            raise ValueError('AC evaluation requires DatasetPaths on the partition')
        arrays['qd'] = inputs.qd
        paths = partition.paths
        case = load_case_from_csv(paths.case_name, paths.params_path)
        general = params['general']
        pg = np.full((len(valid), general['n_gen']), np.nan)
        vm = np.full((len(valid), general['n_buses']), np.nan)
        qg = np.full_like(pg, np.nan)
        converged = np.zeros(len(valid), dtype=bool)
        solutions = []
        for pos in range(len(valid)):
            result = (solve_pf_setpoints(inputs.pd[pos], inputs.qd[pos], pg_set[pos],
                                         prediction.vm[pos], params, case)
                      if valid[pos] else ({'success': False},))
            solutions.append(result)
            converged[pos] = bool(result[0]['success'])
            if converged[pos]:
                pg[pos] = result[0]['gen'][:, 1] / general['BASE_MVA']
                qg[pos] = result[0]['gen'][:, 2] / general['BASE_MVA']
                vm[pos] = result[0]['bus'][:, 7]
        metrics = evaluate_acopf_predictions(pg, vm, targets['pg'], targets['vm'], targets['qg'],
                                             targets['va'], solutions, converged, params, verbose=False)
        arrays.update({f'{name}_converged': converged, f'{name}_vm_pred': vm,
                       f'{name}_qg_pred': qg, 'vm_setpoints': prediction.vm.copy(),
                       **{f'{key}_true': value for key, value in targets.items()}})
    arrays[f'{name}_pg_pred'] = pg
    return EvaluationResult(inputs.formulation, {name: metrics}, arrays)


def evaluate_baseline(spec, state, paths, indices=None):
    if spec.formulation == 'ac':
        from .ac_evaluation import evaluate_ac as evaluate
    else:
        from .dc_evaluation import evaluate_dc as evaluate
    records, arrays = evaluate(spec, state, paths, indices)
    return EvaluationResult(spec.formulation, records, arrays)
