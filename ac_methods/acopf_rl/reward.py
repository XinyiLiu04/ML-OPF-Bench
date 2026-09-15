# -*- coding: utf-8 -*-
"""Reward functions for the ACOPF RL environment.

Each reward combines a scaled objective with a scaled constraint penalty. The objective
and the penalty are on very different scales, so both are put through an affine scaling
fitted beforehand; without it the penalty term is invisible next to a cost in the
thousands. The classes differ in how they treat a sample that violates a constraint.
"""

import abc
import copy

import numpy as np


class RewardFunction(abc.ABC):
    """Scales the objective and the penalty, then combines them.

    penalty_weight interpolates between the two terms; None means an unweighted sum.
    clip_range, when given, bounds the final reward, which keeps a single catastrophic
    sample from dominating a policy-gradient update.
    """

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
        """Turn distribution statistics into the affine scaling coefficients."""
        if not isinstance(reward_scaling, str):
            return {'penalty_factor': 1, 'penalty_bias': 0,
                    'objective_factor': 1, 'objective_bias': 0}

        # The caller's dict is copied rather than updated in place, so passing a
        # statistics dict does not come back carrying scaling coefficients
        given = copy.copy(scaling_params) if scaling_params else {}
        resolved = copy.copy(given)

        reward_scaler = select_reward_scaler(reward_scaling)
        try:
            resolved.update(reward_scaler(**given))
        except TypeError:
            raise ValueError(
                'scaling_params is missing required keys. Pass a pre-computed '
                'statistics dict from estimate_scaling_params().')

        # Anything the caller set explicitly wins over the derived value
        resolved.update(given)

        if np.isnan(resolved['penalty_bias']):
            resolved['penalty_bias'] = 0
        if np.isinf(resolved['penalty_factor']):
            resolved['penalty_factor'] = 1

        return resolved

    def __call__(self, objective: float, penalty: float, valid: bool) -> float:
        objective = self.adjust_objective(objective, valid)
        penalty = self.adjust_penalty(penalty, valid)
        objective = self.scale_objective(objective)
        penalty = self.scale_penalty(penalty)
        reward = self.compute_total_reward(objective, penalty)
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
        return objective * self.scaling_params['objective_factor'] \
            + self.scaling_params['objective_bias']

    def scale_penalty(self, penalty):
        return penalty * self.scaling_params['penalty_factor'] \
            + self.scaling_params['penalty_bias']

    def calculate_cost(self, penalty, valid):
        """Constraint cost in the sense of constrained RL, kept separate from reward."""
        if valid:
            return 0.0
        return abs(penalty * self.scaling_params['penalty_factor'])

    @abc.abstractmethod
    def adjust_penalty(self, penalty: float, valid: bool) -> float:
        """Reshape the penalty before scaling."""

    @abc.abstractmethod
    def adjust_objective(self, objective: float, valid: bool) -> float:
        """Reshape the objective before scaling."""


def select_reward_scaler(reward_scaling: str):
    """Look up the function that turns statistics into scaling coefficients."""
    if reward_scaling == 'normalization':
        return calculate_normalization_params
    if reward_scaling == 'minmax01':
        return calculate_minmax01_params
    if reward_scaling == 'minmax11':
        return calculate_minmax11_params
    raise NotImplementedError(f'Unknown reward scaling: {reward_scaling}')


def calculate_normalization_params(std_objective, mean_objective,
                                   std_penalty, mean_penalty, **kwargs):
    """Map each term to zero mean and unit standard deviation."""
    return {
        'objective_factor': 1 / std_objective,
        'objective_bias': -mean_objective / std_objective,
        'penalty_factor': 1 / std_penalty,
        'penalty_bias': -mean_penalty / std_penalty,
    }


def calculate_minmax01_params(min_objective, max_objective,
                              min_penalty, max_penalty, **kwargs):
    """Map each term from its observed range onto [0, 1]."""
    diff_obj = max_objective - min_objective
    diff_pen = max_penalty - min_penalty
    return {
        'objective_factor': 1 / diff_obj,
        'objective_bias': -(min_objective / diff_obj),
        'penalty_factor': 1 / diff_pen,
        'penalty_bias': -(min_penalty / diff_pen),
    }


def calculate_minmax11_params(min_objective, max_objective,
                              min_penalty, max_penalty, **kwargs):
    """Map each term from its observed range onto [-1, 1]."""
    diff_obj = (max_objective - min_objective) / 2
    diff_pen = (max_penalty - min_penalty) / 2
    return {
        'objective_factor': 1 / diff_obj,
        'objective_bias': -(min_objective / diff_obj + 1),
        'penalty_factor': 1 / diff_pen,
        'penalty_bias': -(min_penalty / diff_pen + 1),
    }


class Summation(RewardFunction):
    """Weighted sum of the objective and the penalty, whether or not the sample is valid."""

    def adjust_penalty(self, penalty, valid):
        return penalty

    def adjust_objective(self, objective, valid):
        return objective


class Replacement(RewardFunction):
    """Reward the objective only when valid, and drop it entirely when not.

    An invalid sample earns nothing from its objective, so the agent cannot buy a low
    cost with a constraint violation. The flip side is a flat reward across all invalid
    samples, which gives no gradient toward being less invalid.
    """

    def __init__(self, valid_reward: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.valid_reward = valid_reward

    def adjust_penalty(self, penalty, valid):
        return penalty

    def adjust_objective(self, objective, valid):
        return objective + self.valid_reward if valid else 0.0


class Parameterized(RewardFunction):
    """Interpolates between Summation and Replacement.

    invalid_objective_share scales the objective on invalid samples: 1.0 recovers
    Summation and 0.0 recovers Replacement. valid_reward and invalid_penalty add a
    step at the feasibility boundary, so crossing it is worth something on its own.
    """

    def __init__(self,
                 valid_reward: float = 0.0,
                 invalid_penalty: float = 0.5,
                 invalid_objective_share: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        assert valid_reward >= 0, 'valid_reward must be >= 0'
        assert invalid_penalty >= 0, 'invalid_penalty must be >= 0'
        assert 0 <= invalid_objective_share <= 1, \
            'invalid_objective_share must be in [0, 1]'
        self.valid_reward = valid_reward
        self.invalid_penalty = invalid_penalty
        self.invalid_objective_share = invalid_objective_share

    def adjust_penalty(self, penalty, valid):
        return penalty + self.valid_reward if valid else penalty - self.invalid_penalty

    def adjust_objective(self, objective, valid):
        return objective if valid else objective * self.invalid_objective_share

    def calculate_cost(self, penalty, valid):
        if valid:
            return 0.0
        return super().calculate_cost(penalty, valid) + self.invalid_penalty


class OnlyObjective(RewardFunction):
    """Ignore the penalty in the reward; for constrained RL that handles it separately."""

    def __init__(self, **kwargs):
        super().__init__(penalty_weight=0.0, **kwargs)

    def adjust_penalty(self, penalty, valid):
        return 0.0

    def adjust_objective(self, objective, valid):
        return objective