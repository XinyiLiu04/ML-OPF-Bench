"""Validation selection for the feasibility-first DC RL variant."""
import copy
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class FeasibilityStopping(BaseCallback):
    def __init__(self, env, interval, patience, min_delta, diagnostics=None, archive=None):
        super().__init__()
        self.env = env
        self.diagnostics = diagnostics
        self.archive = archive
        self.interval, self.patience, self.min_delta = interval, patience, min_delta
        self.next_check = interval
        self.best_key = None
        self.best_state = None
        self.best_step = None
        self.stale = 0
        self.history = []

    def _better(self, key):
        if self.best_key is None:
            return True
        if key[0] != self.best_key[0]:
            return key[0] > self.best_key[0]
        for new, old in zip(key[1:], self.best_key[1:]):
            if abs(new-old) > self.min_delta:
                return new > old
        return False

    def _evaluate(self):
        actions, _ = self.model.predict(self.env.x_scaled, deterministic=True)
        infos, rewards = [], []
        for i, action in enumerate(actions):
            self.env._current_idx = i
            _, reward, _, _, info = self.env.step(action)
            infos.append(info)
            rewards.append(reward)
        count = sum(i['feasible'] for i in infos)
        violation = float(np.mean([i['normalized_violation'] for i in infos]))
        costs = [i['normalized_cost'] for i in infos if i['feasible']]
        cost = float(np.mean(costs)) if costs else None
        key = (count, -violation, -cost if cost is not None else 0.0)
        selected = self._better(key)
        if selected:
            self.best_key = key
            self.best_state = copy.deepcopy(self.model.policy.state_dict())
            self.best_step = self.num_timesteps
            self.stale = 0
        else:
            self.stale += 1
        record = dict(step=self.num_timesteps, feasible=count, samples=len(infos),
                      violation=violation, feasible_cost=cost,
                      reward=float(np.mean(rewards)), selected=selected)
        if self.diagnostics is not None:
            record["metrics"] = self.diagnostics(actions)
        self.history.append(record)
        print(record, flush=True)
        self.next_check = self.num_timesteps+self.interval

    def _on_training_start(self):
        self._evaluate()

    def _on_rollout_start(self):
        if self.num_timesteps >= self.next_check:
            self._evaluate()

    def _on_step(self):
        return self.stale < self.patience

    def _on_training_end(self):
        if not self.history or self.history[-1]['step'] != self.num_timesteps:
            self._evaluate()
        if self.archive is not None:
            self.archive(self.model, self)
        self.model.policy.load_state_dict(self.best_state)
