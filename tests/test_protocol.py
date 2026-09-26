from pathlib import Path

import numpy as np
import pytest

from ml_opf_bench.config import Experiment, dataset_paths, paper_experiments
from ml_opf_bench.runtime import managed_run, training_indices
from ml_opf_bench.splits import make_splits


def test_constraints_follow_scenario():
    for form in ("ac", "dc"):
        base = dataset_paths(Path("/data"), form, "case118")
        wider = dataset_paths(Path("/data"), form, "case118", "larger_variance")
        heavy = dataset_paths(Path("/data"), form, "case118", "heavier_loads")
        assert base.params_path == wider.params_path
        assert heavy.params_path.name == "case118(api)"
        assert heavy.case_name.endswith("__api")
        assert wider.data_path != base.data_path


def test_resume_rejects_changed_shift_data(tmp_path, monkeypatch):
    import json
    from ml_opf_bench import suite
    spec = Experiment("ac", "DNN")
    scenarios = ["base", "larger_variance", "heavier_loads"]
    (tmp_path / "completed.json").write_text(json.dumps({"scenarios": scenarios}))
    for name in ("checkpoint.pt", "splits.npz"):
        (tmp_path / name).touch()
    for scenario in scenarios:
        (tmp_path / f"{scenario}_data_manifest.json").write_text('{"hash": "original"}')
        (tmp_path / f"{scenario}.json").write_text('{}')
        (tmp_path / f"{scenario}_samples.npz").touch()
    monkeypatch.setattr(suite, "data_signature", lambda paths: {"hash": "original"})
    assert suite.completed_scenarios_match(tmp_path, spec, "/data", {})
    monkeypatch.setattr(suite, "data_signature", lambda paths:
                        {"hash": "changed" if "api" in paths.case_name else "original"})
    assert not suite.completed_scenarios_match(tmp_path, spec, "/data", {})


def test_rl_accepts_unclipped_scaled_observations():
    from ml_opf_bench.registry import load_method
    module, _, _ = load_method("ac", "RL")
    params = {"general": {"n_loads": 1, "n_gen": 1, "n_gen_non_slack": 0, "BASE_MVA": 100},
              "generator": {"cost_c2": [1], "cost_c1": [1], "cost_c0": [0]}}
    observations = np.array([[-0.2, 1.2]], dtype=np.float32)
    env = module.AcopfEnv(observations, observations, [0], params, None, None, None)
    observation, _ = env.reset()
    assert env.observation_space.contains(observation)
    np.testing.assert_array_equal(observation, observations[0])
    with pytest.raises(ValueError, match="finite"):
        module.AcopfEnv(observations * np.nan, observations, [0], params, None, None, None)


def test_scaling_keeps_heldout_fixed_and_training_nested():
    small = make_splits(49532, mode="scaling", train_size=1000)
    large = make_splits(49532, mode="scaling", train_size=35000)
    np.testing.assert_array_equal(small[0], large[0][:1000])
    for i in (1, 2):
        np.testing.assert_array_equal(small[i], large[i])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert not np.intersect1d(large[i], large[j]).size


def test_cross_system_pool_is_not_training_count():
    train, val, test = make_splits(50000)
    assert (len(train), len(val), len(test)) == (10000, 1000, 1000)


def test_invalid_sizes_fail():
    with pytest.raises(ValueError):
        make_splits(12000, mode="scaling", train_size=35000)


def test_managed_split_rejects_prefiltering():
    spec = Experiment("ac", "GNN", pool_size=120)
    with managed_run(spec):
        training_indices(200)
        with pytest.raises(ValueError, match="unfiltered"):
            training_indices(199)


def test_paper_matrix_includes_kkt_scaling_without_pr2():
    specs = list(paper_experiments())
    for form in ("ac", "dc"):
        rows = [x for x in specs if x.formulation == form and x.method == "KKT" and x.mode == "scaling"]
        assert {x.train_size for x in rows} == {1000, 5000, 12000, 20000, 30000, 35000}
    assert all("PR2" not in x.method for x in specs)


def test_graph_batch_has_no_edges_between_samples():
    import torch
    from ml_opf_bench.registry import load_method
    load_method("ac", "GNN")
    from gnn_utils import collate_graph_batch
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    weights = torch.tensor([0.2, 0.2, 0.7, 0.7])
    _, batched, repeated, _ = collate_graph_batch(torch.zeros(3, 12), edges, weights, 3, "cpu")
    assert torch.equal(batched[0] // 3, batched[1] // 3)
    for i in range(3):
        mask = batched[0] // 3 == i
        assert torch.equal(batched[:, mask] - 3 * i, edges)
        assert torch.equal(repeated[mask], weights)


def test_high_capacity_ac_branches_remain_constrained():
    from ml_opf_bench.registry import load_method
    load_method("ac", "DNN")
    from ac_configuration.acopf_evaluation_metrics import calculate_single_sample_violations
    result = {"gen": np.zeros((1, 21)), "bus": np.zeros((1, 13)), "branch": np.zeros((2, 17))}
    result["branch"][0, 5] = 9900
    result["branch"][0, 13] = 10890
    result["branch"][1, 13] = 1e6  # Zero-rated lines are unconstrained.
    violation = calculate_single_sample_violations((result,), True, 100)[3]
    np.testing.assert_allclose(violation, 10890 / 9900 - 1)


def test_reference_loads_accept_dataset_headers_and_reject_empty_inputs(tmp_path):
    from ml_opf_bench.reference import ac_load_inputs
    pd_path, qd_path = tmp_path / "pd.csv", tmp_path / "qd.csv"
    pd_path.write_text("pd10,pd2\n0.3,0.1\n0.4,0.2\n")
    qd_path.write_text("qd2,qd10\n0.01,0.03\n0.02,0.04\n")
    ids, active, reactive = ac_load_inputs(pd_path, qd_path)
    assert ids == [2, 10]
    np.testing.assert_allclose(active, [[0.1, 0.3], [0.2, 0.4]])
    np.testing.assert_allclose(reactive, [[0.01, 0.03], [0.02, 0.04]])
    pd_path.write_text("irrelevant\n1\n")
    with pytest.raises(ValueError, match="No pd load columns"):
        ac_load_inputs(pd_path, qd_path)
