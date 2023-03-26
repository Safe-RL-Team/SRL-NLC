import sys

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
import lasagne.nonlinearities as NL

from rllab.policies.deterministic_conv_policy_v4 import CategoricalConvPolicy

# Baseline
from sandbox.cpo.baselines.gaussian_conv_baseline import GaussianConvBaseline

# Environment
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
import gym_minigrid

# Test importing wrappers
from gym_minigrid.wrappers import *

# Policy optimization
from sandbox.cpo.algos.safe.pcpo_kl_bm import PCPO_KL_BM
from sandbox.cpo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.cpo.safety_constraints.bomberman import BMSafetyConstraint


ec2_mode = False

env = gym.make('MiniGrid-Bomberman-S-v0')

env = BombermanMissionWrapper(env)
env.max_steps = min(env.max_steps, 200)
env.reset()

def run_task(*_):
        trpo_stepsize = 1e-3 
        trpo_stepsize_valueFun = 1e-3
        trpo_subsample_factor = 0.8

        policy = CategoricalConvPolicy(
                    env_spec=env,
                    hidden_sizes=(30,15),
                    conv_filters=(5,6),
                    conv_filter_sizes=(3,3),
                    conv_strides=(1,1),
                    conv_pads=(0,0),
                    num_embedding_object = 11,
                    size_embedding=5,
                    hidden_sizes_prob = (5,5), 
                    hidden_nonlinearity=NL.tanh,
                    enable_hcMLP = True,
                 )
        
        # do not use hc since it does not have consider the hc information
        baseline = GaussianConvBaseline(
            env_spec=env,
            regressor_args={
                    'hidden_sizes': (30,15),
                    'conv_filters':(5,6),
                    'conv_filter_sizes':(3,3),
                    'conv_strides':(1,1),
                    'conv_pads':(0,0),
                    'hidden_nonlinearity': NL.rectify, #NL.rectify / NL.tanh
                    'hidden_sizes_mean': (20,10),
                    'num_embedding_object': 11,
                    'size_embedding': 10,
                    'learn_std':False,
                    'step_size':trpo_stepsize_valueFun,
                    'use_trust_region':True,
                    'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    }
        )
        
        # do not use hc since it does not have consider the hc information
        safety_baseline = GaussianConvBaseline(
            env_spec=env,
            regressor_args={
                    'hidden_sizes': (30,15),
                    'conv_filters':(5,6),
                    'conv_filter_sizes':(3,3),
                    'conv_strides':(1,1),
                    'conv_pads':(0,0),
                    'hidden_nonlinearity': NL.rectify, #NL.rectify / NL.tanh
                    'hidden_sizes_mean': (20,10),
                    'num_embedding_object': 11,
                    'size_embedding': 10,
                    'learn_std':False,
                    'use_trust_region':True,
                    'step_size':trpo_stepsize_valueFun,
                    'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    },
            target_key='safety_returns',
            )

        safety_constraint = BMSafetyConstraint(max_value=2, baseline=safety_baseline)

        algo = PCPO_KL_BM(
            env=env,
            policy=policy,
            baseline=baseline,
            safety_constraint=safety_constraint,
            safety_gae_lambda=0.9,
            safety_discount=1.0,
            batch_size=200*50,
            max_path_length=200,
            n_itr=2500,
            gae_lambda=0.95,
            discount=0.99,
            step_size=trpo_stepsize,
            optimizer_args={'subsample_factor':trpo_subsample_factor},
            plot=False,
            pause_for_plot=False,
        )

        algo.train()

for ii in [10]:
    run_experiment_lite(
        run_task,
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix='Result' + str(ii),
        seed=ii,
        mode = "ec2" if ec2_mode else "local",
        plot=False,
    )