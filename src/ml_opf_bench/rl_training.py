"""Validation reward and early stopping for the one-step OPF environment."""

import copy

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ValidationRewardStopping(BaseCallback):
    def __init__(self, env, interval, patience, min_delta):
        super().__init__()
        self.validation_env = env
        self.interval = interval
        self.patience = patience
        self.min_delta = min_delta
        self.next_check = interval
        self.best_reward = -np.inf
        self.best_state = None
        self.stale = 0
        self.should_stop = False

    def _on_rollout_start(self):
        if self.num_timesteps < self.next_check:
            return
        env = self.validation_env
        actions, _ = self.model.predict(env.x_scaled, deterministic=True)
        rewards = []
        for i, action in enumerate(actions):
            env._current_idx = i
            rewards.append(env.step(action)[1])
        reward = float(np.mean(rewards))
        if not np.isfinite(reward):
            raise FloatingPointError("Nonfinite validation reward")
        if reward > self.best_reward + self.min_delta:
            self.best_reward = reward
            self.best_state = copy.deepcopy(self.model.policy.state_dict())
            self.stale = 0
        else:
            self.stale += 1
        self.next_check = self.num_timesteps + self.interval
        self.should_stop = self.stale >= self.patience
        print(f"Steps {self.num_timesteps}: validation reward={reward:.6g}, stale={self.stale}", flush=True)

    def _on_step(self):
        return not self.should_stop

    def _on_training_end(self):
        if self.best_state is not None:
            self.model.policy.load_state_dict(self.best_state)
