"""Single-step DDPG baseline for the exported DC/PTDF model."""
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from ml_opf_bench.runtime import TrainingState
from dc_methods.dcopf_rl.dc_rl_validation import FeasibilityStopping
from dc_configuration.dcopf_data_setup import load_parameters_from_csv, load_samples, prepare_data_splits, reconstruct_full_pg
from dc_configuration.dcopf_evaluation_metrics import violations, compute_cost


def action_to_pg(action, params):
    c=params['constraints'];ns=params['general']['non_slack_gen_idx']
    action=np.asarray(action)
    if action.shape[-1]!=len(ns) or not np.isfinite(action).all():
        raise ValueError('Invalid non-slack Pg action')
    return c['pg_min'][ns]+np.clip(action,0,1)*(c['pg_max'][ns]-c['pg_min'][ns])


def dispatch_reward(v, cost, constraints, cost_scale):
    """Physical feasibility gate; normalized worst and average constraint excess."""
    c = constraints
    pg_scale = np.maximum(np.maximum(np.abs(c['pg_min']), np.abs(c['pg_max'])), 1.0)
    thermal = v['branch'] / c['rate_a'][c['constrained_branches']]
    raw = dict(pg_up=v['gen_up'], pg_lo=v['gen_lo'],
               thermal_relative=thermal, balance=v['balance'])
    maxima = {k: float(a.max()) if a.size else 0.0 for k, a in raw.items()}
    feasible = all(value <= 1e-5 for value in maxima.values())
    normalized = (v['gen_up']/pg_scale, v['gen_lo']/pg_scale, thermal,
                  v['balance']/pg_scale.sum())
    penalty = sum(float(a.max()+a.mean()) for a in normalized if a.size)
    # Strictly interior even when tanh saturates in floating point.
    cost_term = float(np.clip(0.5*(1+np.tanh(cost/cost_scale)),
                              np.finfo(float).eps, 1-np.finfo(float).eps))
    reward = -cost_term if feasible else -1.0-penalty
    if not np.isfinite(reward):
        raise FloatingPointError('Nonfinite DC reward')
    return reward, dict(feasible=feasible, violations=maxima, cost=cost,
                        normalized_violation=penalty, normalized_cost=cost_term)


class DcEnv(gym.Env):
    def __init__(self, X, loads, indices, params, seed=42):
        self.x_scaled=X[indices].astype(np.float32)
        self.loads=loads[indices];self.params=params;self._current_idx=0
        self.rng=np.random.default_rng(seed)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(X.shape[1],),dtype=np.float32)
        self.action_space=spaces.Box(0,1,shape=(len(params['general']['non_slack_gen_idx']),),dtype=np.float32)
        c=params['constraints'];m=np.maximum(np.abs(c['pg_min']),np.abs(c['pg_max']))
        self.cost_scale=float(np.sum(np.abs(c['cost_c2'])*m*m+np.abs(c['cost_c1'])*m+np.abs(c['cost_c0'])))
        if self.cost_scale<=0 or not np.isfinite(self.cost_scale):
            raise ValueError('Invalid physical cost scale')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:self.rng=np.random.default_rng(seed)
        self._current_idx=int(self.rng.integers(len(self.loads)))
        return self.x_scaled[self._current_idx],{}

    def step(self,action):
        load=self.loads[self._current_idx:self._current_idx+1]
        pg=reconstruct_full_pg(action_to_pg(action,self.params)[None,:],load,self.params)
        v=violations(pg,load,self.params)
        c=self.params['constraints']
        reward, info = dispatch_reward(v, float(compute_cost(pg,self.params)[0]),
                                       c, self.cost_scale)
        return self.x_scaled[self._current_idx],float(reward),True,False,info


def rl_experiment(case_name,params_path,data_path,n_train_use=12000,seed=42,
                  total_timesteps=2_000_000,hidden_sizes=None,batch_size=128,
                  learning_rate=3e-4,device='cpu',learning_starts=1024,
                  early_stop_patience=20,early_stop_min_delta=1e-6,
                  validation_diagnostics=None,training_archive=None,**unused):
    if total_timesteps<=0:raise ValueError('Positive explicit budget required')
    params=load_parameters_from_csv(case_name,params_path)
    loads,_=load_samples(data_path,params)
    train,val,_=prepare_data_splits(len(loads),n_train_use,seed)
    scaler=StandardScaler().fit(loads[train]);X=scaler.transform(loads)
    env=DcEnv(X,loads,train,params,seed)
    val_env=DcEnv(X,loads,val,params,seed)
    n=env.action_space.shape[0]
    model=DDPG('MlpPolicy',env,learning_rate=learning_rate,buffer_size=1_000_000,
        learning_starts=learning_starts,batch_size=batch_size,gamma=0.,
        train_freq=(1,'step'),gradient_steps=1,
        action_noise=NormalActionNoise(np.zeros(n),0.1*np.ones(n)),
        policy_kwargs=dict(net_arch=hidden_sizes or [256,128]),device=device,seed=seed)
    callback=FeasibilityStopping(val_env,10000,early_stop_patience,early_stop_min_delta,
                                validation_diagnostics,training_archive)
    start=time.perf_counter();model.learn(total_timesteps=total_timesteps,callback=callback)
    return TrainingState(model.policy,params,time.perf_counter()-start,
        dict(x_scaler=scaler,algorithm='DDPG',action_mode='pg_only',total_timesteps=model.num_timesteps,
             cost_scale=env.cost_scale,reward='feasibility_first_v2',
             selection='feasible_count_then_violation_then_feasible_cost',
             best_step=callback.best_step,validation_history=callback.history))
