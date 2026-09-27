"""Validation-only learning curves for the fixed-voltage DDPG variant."""
import numpy as np
from ml_opf_bench.rl_training import ValidationRewardStopping as RewardStopping


class ValidationRewardStopping(RewardStopping):
    def __init__(self, *args):
        super().__init__(*args)
        self.history = []

    def _on_rollout_start(self):
        if self.num_timesteps < self.next_check:
            return
        # Parent performs the checkpoint selection. Capture the same evaluations.
        env = self.validation_env
        original = env.step
        records = []
        def record(action):
            result = original(action)
            records.append(result)
            return result
        env.step = record
        try:
            super()._on_rollout_start()
        finally:
            env.step = original
        infos = [r[4] for r in records]
        self.history.append(dict(steps=self.num_timesteps,
            reward=float(np.mean([r[1] for r in records])),
            pf_converged=int(sum(i['pf_converged'] for i in infos)),
            feasible=int(sum(i['feasible'] for i in infos)), total=len(infos)))
