import numpy as np
import pytest
import torch
from ml_opf_bench.config import Experiment
from ml_opf_bench.registry import load_method


def test_pgonly_mapping_and_saved_policy(tmp_path):
    module, _, _ = load_method('ac', 'RL', 'ddpg-pgonly')
    from stable_baselines3 import DDPG
    from gymnasium import Env, spaces
    class TinyEnv(Env):
        observation_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        action_space = spaces.Box(0, 1, shape=(2,), dtype=np.float32)
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(3, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(3, dtype=np.float32), float(-np.sum(action**2)), True, False, {}
    bounds = dict(pg_min=np.array([1., 2.]), pg_max=np.array([3., 6.]),
                  vm_fixed=np.array([1.01, 1.03, 1.04]))
    pg, vm = module.action_to_setpoints(np.array([-1., 2.]), bounds)
    np.testing.assert_array_equal(pg, [1., 6.])
    np.testing.assert_array_equal(vm, bounds['vm_fixed'])
    vm[0] = 9
    assert bounds['vm_fixed'][0] == 1.01
    with pytest.raises(ValueError):
        module.action_to_setpoints(np.zeros(5), bounds)
    model = DDPG('MlpPolicy', TinyEnv(), learning_starts=0, batch_size=2,
                 buffer_size=16, policy_kwargs=dict(net_arch=[8]), seed=42)
    model.learn(4)
    obs = np.zeros((2, 3), dtype=np.float32)
    before = model.policy.predict(obs, deterministic=True)[0]
    torch.save(model.policy, tmp_path/'policy.pt')
    loaded = torch.load(tmp_path/'policy.pt', weights_only=False)
    np.testing.assert_array_equal(before, loaded.predict(obs, deterministic=True)[0])
    for action in before:
        np.testing.assert_array_equal(module.action_to_setpoints(action, bounds)[1], bounds['vm_fixed'])


def test_variant_separates_runs():
    spec = Experiment('ac', 'RL', variant='ddpg-pgonly')
    assert spec.run_id != Experiment('ac', 'RL').run_id
    with pytest.raises(ValueError):
        Experiment('ac', 'DNN', variant='ddpg-pgonly')


def test_environment_passes_only_pg_and_fixed_voltage(monkeypatch):
    module, _, _ = load_method('ac', 'RL', 'ddpg-pgonly')
    seen = []
    def pf(pd, qd, pg, vm, params, case):
        seen.append((pg.copy(), vm.copy()))
        return ({'success': False}, False)
    monkeypatch.setattr(module, 'solve_pf_setpoints', pf)
    params = {'general': dict(n_loads=1, n_gen=3, n_gen_non_slack=2, BASE_MVA=100),
              'generator': dict(cost_c2=0, cost_c1=1, cost_c0=0)}
    bounds = dict(pg_min=np.array([1., 2.]), pg_max=np.array([3., 6.]),
                  vm_fixed=np.array([1.01, 1.03, 1.04]))
    env = module.AcopfEnv(np.zeros((1,2)), np.zeros((1,2)), [0], params,
                          {}, bounds, lambda *args: 0)
    assert env.action_space.shape == (2,)
    env.reset(seed=42)
    for action in (np.zeros(2), np.ones(2)):
        _, _, terminated, truncated, _ = env.step(action)
        assert terminated and not truncated
    np.testing.assert_array_equal(seen[0][0], bounds['pg_min'])
    np.testing.assert_array_equal(seen[1][0], bounds['pg_max'])
    for _, vm in seen:
        np.testing.assert_array_equal(vm, bounds['vm_fixed'])
