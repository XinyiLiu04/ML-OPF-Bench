import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('audit_checkpoint',Path(__file__).parents[1]/'reports/audit_checkpoint.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_metrics_reject_difference_but_ignore_elapsed_time():
    old={'RL':{'cost':1.,'feasible_samples':10,'inference_ms':1.,'train_time_s':20.,'missing':None}}
    new={'RL':{'cost':1.,'feasible_samples':10,'inference_ms':5.,'train_time_s':30.,'missing':float('nan')}}
    assert all(r['matches'] for r in audit.compare_metrics(old,new,1e-5,1e-4))
    new['RL']['feasible_samples']=0
    assert not all(r['matches'] for r in audit.compare_metrics(old,new,1e-5,1e-4))
    new['RL']['extra']=1
    with pytest.raises(ValueError):audit.compare_metrics(old,new,1e-5,1e-4)
