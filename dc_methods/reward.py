import abc
import copy

import numpy as np


class RewardFunction(abc.ABC):
    def __init__(self,
                 penalty_weight: float = 0.5,
                 clip_range: tuple = None,
                 reward_scaling: str = None,
                 scaling_params: dict = None,
                 env=None):
        self.penalty_weight = penalty_weight
        self.clip_range = clip_range
        self.scaling_params = self.prepare_reward_scaling(
            reward_scaling, scaling_params, env)

    def prepare_reward_scaling(self, reward_scaling, scaling_params, env):
        if not isinstance(reward_scaling, str):
            return {'penalty_factor': 1, 'penalty_bias': 0,
                    'objective_factor': 1, 'objective_bias': 0}

        scaling_params = scaling_params or {}
        user_scaling_params = copy.copy(scaling_params)

        reward_scaler = select_reward_scaler(reward_scaling)
        try:
            scaling_params.update(reward_scaler(**scaling_params))
        except TypeError:
            raise ValueError(
                'scaling_params is missing required keys. '
                'Pass a pre-computed norm_params dict from estimate_scaling_params().')

        scaling_params.update(user_scaling_params)

        if np.isnan(scaling_params['penalty_bias']):
            scaling_params['penalty_bias'] = 0
        if np.isinf(scaling_params['penalty_factor']):
            scaling_params['penalty_factor'] = 1

        return scaling_params

    def __call__(self, objective: float, penalty: float, valid: bool) -> float:
        objective = self.adjust_objective(objective, valid)
        penalty   = self.adjust_penalty(penalty, valid)
        objective = self.scale_objective(objective)
        penalty   = self.scale_penalty(penalty)
        reward    = self.compute_total_reward(objective, penalty)
        if self.clip_range:
            reward = self.clip_reward(reward)
        return reward

    def clip_reward(self, reward):
        return np.clip(reward, self.clip_range[0], self.clip_range[1])

    def compute_total_reward(self, objective, penalty):
        if self.penalty_weight is None:
            return objective + penalty
        return objective * (1 - self.penalty_weight) + penalty * self.penalty_weight

    def scale_objective(self, objective):
        objective *= self.scaling_params['objective_factor']
        objective += self.scaling_params['objective_bias']
        return objective

    def scale_penalty(self, penalty):
        penalty *= self.scaling_params['penalty_factor']
        penalty += self.scaling_params['penalty_bias']
        return penalty

    def calculate_cost(self, penalty, valid):
        if valid:
            return 0.0
        return abs(penalty * self.scaling_params['penalty_factor'])

    @abc.abstractmethod
    def adjust_penalty(self, penalty: float, valid: bool) -> float:
        return penalty

    @abc.abstractmethod
    def adjust_objective(self, objective: float, valid: bool) -> float:
        return objective


# =====================================================================
# Scaler selection and scaling functions
# =====================================================================

def select_reward_scaler(reward_scaling: str):
    if reward_scaling == 'normalization':
        return calculate_normalization_params
    elif reward_scaling == 'minmax01':
        return calculate_minmax01_params
    elif reward_scaling == 'minmax11':
        return calculate_minmax11_params
    else:
        raise NotImplementedError(f'Unknown reward scaling: {reward_scaling}')


def calculate_normalization_params(std_objective, mean_objective,
                                   std_penalty, mean_penalty, **kwargs):
    """Scale to zero mean, unit std."""
    return {
        'objective_factor': 1 / std_objective,
        'objective_bias':   -mean_objective / std_objective,
        'penalty_factor':   1 / std_penalty,
        'penalty_bias':     -mean_penalty / std_penalty,
    }


def calculate_minmax01_params(min_objective, max_objective,
                              min_penalty, max_penalty, **kwargs):
    """Scale from [min, max] to [0, 1]."""
    diff_obj = max_objective - min_objective
    diff_pen = max_penalty   - min_penalty
    return {
        'objective_factor': 1 / diff_obj,
        'objective_bias':   -(min_objective / diff_obj),
        'penalty_factor':   1 / diff_pen,
        'penalty_bias':     -(min_penalty / diff_pen),
    }


def calculate_minmax11_params(min_objective, max_objective,
                              min_penalty, max_penalty, **kwargs):
    """Scale from [min, max] to [-1, 1]."""
    diff_obj = (max_objective - min_objective) / 2
    diff_pen = (max_penalty   - min_penalty)   / 2
    return {
        'objective_factor': 1 / diff_obj,
        'objective_bias':   -(min_objective / diff_obj + 1),
        'penalty_factor':   1 / diff_pen,
        'penalty_bias':     -(min_penalty / diff_pen + 1),
    }


# =====================================================================
# Concrete reward classes
# =====================================================================

class Summation(RewardFunction):
    """Weighted sum of objective and penalty."""
    def adjust_penalty(self, penalty, valid):
        return penalty

    def adjust_objective(self, objective, valid):
        return objective


class Replacement(RewardFunction):
    """Return objective only when valid, penalty only when invalid."""
    def __init__(self, valid_reward: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.valid_reward = valid_reward

    def adjust_penalty(self, penalty, valid):
        return penalty

    def adjust_objective(self, objective, valid):
        if valid:
            return objective + self.valid_reward
        return 0.0


class Parameterized(RewardFunction):
    """Interpolation between Summation and Replacement."""
    def __init__(self,
                 valid_reward: float = 0.0,
                 invalid_penalty: float = 0.5,
                 invalid_objective_share: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        assert valid_reward >= 0,           'valid_reward must be >= 0'
        assert invalid_penalty >= 0,        'invalid_penalty must be >= 0'
        assert 0 <= invalid_objective_share <= 1
        self.valid_reward             = valid_reward
        self.invalid_penalty          = invalid_penalty
        self.invalid_objective_share  = invalid_objective_share

    def adjust_penalty(self, penalty, valid):
        return penalty + self.valid_reward if valid else penalty - self.invalid_penalty

    def adjust_objective(self, objective, valid):
        if not valid:
            objective *= self.invalid_objective_share
        return objective

    def calculate_cost(self, penalty, valid):
        if valid:
            return 0.0
        return super().calculate_cost(penalty, valid) + self.invalid_penalty


class OnlyObjective(RewardFunction):
    """Ignore penalty entirely (for Safe RL)."""
    def __init__(self, **kwargs):
        super().__init__(penalty_weight=0.0, **kwargs)

    def adjust_penalty(self, penalty, valid):
        return 0.0

    def adjust_objective(self, objective, valid):
        return objective