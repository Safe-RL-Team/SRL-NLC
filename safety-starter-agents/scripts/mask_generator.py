#!/usr/bin/env python

import time
import numpy as np
import h5py
import random

import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger


def get_random_action(action_n):
    return random.randint(0, action_n - 1)

WORD_TO_IMAGE = {
    'water': (12, 2, 0),
    'lava': (9, 0, 0),
    'grass': (11, 1, 0)
}

def collect_data(fpath, env, max_ep_len=50, num_episodes=1000, render=True):
    profile = "test"
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    f = h5py.File(fpath + "/" + profile + '.h5', 'w')

    observations = []
    constrain_masks = []
    hc = []
    constrain_code = []
    missions = []
    wrapper = HazardWorldMissionWrapper(env)
    logger = EpochLogger()
    o, r, d, ep_ret, ep_cost = env.reset(), 0, False, 0, 0

    for e in range(num_episodes):
        print(env.avoid_obj, env.hc)
        avoid_obj = env.avoid_obj
        # constraint code
        cc = np.zeros((10), dtype=int)
        code = wrapper.encode_mission(o['mission'])
        cc[0:len(code)] = code
        for s in range(max_ep_len):
            a = get_random_action(env.action_space.n - 1)
            o, r, d, info = env.step(a)
            hc.append(env.hc)
            constrain_code.append(cc)
            missions.append(o['mission'])
            if render:
                env.render()
                time.sleep(1e-3)
            observations.append(o['image'])
            constrain_mask = np.all(o['image'] == WORD_TO_IMAGE[avoid_obj], axis=2).astype(int).flatten()
            constrain_masks.append(constrain_mask)
            # o_img = o['image'].flatten()
            ep_ret += r
            ep_cost += info.get('cost', 0)

            if d or (s == max_ep_len - 1):
                logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=s)
                print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(e, ep_ret, ep_cost, s))
                o, r, d, ep_ret, ep_cost = env.reset(), 0, False, 0, 0
                break


    if "observations" not in f.keys():
        observations_np = np.array(observations)
        episodes_masks_np = np.array(constrain_masks)
        episodes_hc_np = np.array(hc)
        episodes_constrain_code_np = np.array(constrain_code)

        f.create_dataset('obs',
                         data=observations_np,
                         chunks=True,
                         maxshape=(None,
                                   observations_np.shape[1],
                                   observations_np.shape[2],
                                   observations_np.shape[3]),
                         compression='gzip', compression_opts=9)
        f.create_dataset('constraint_mask',
                         data=episodes_masks_np,
                         chunks=True,
                         maxshape=(None,
                                   observations_np.shape[1] * observations_np.shape[2]),
                         compression='gzip', compression_opts=9)
        f.create_dataset('hc',
                         data=episodes_hc_np,
                         chunks=True,
                         maxshape=(None),
                         compression='gzip', compression_opts=9)
        f.create_dataset('constraint_code',
                         data=episodes_constrain_code_np,
                         chunks=True,
                         maxshape=(None),
                         compression='gzip', compression_opts=9)
        f.create_dataset('missions', (len(missions)), dtype=h5py.special_dtype(vlen=str), data=missions)

    f.close()
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

    with h5py.File(fpath + "/" + profile + '.h5', 'r') as f:
        for key in f.keys():
            print(f[key], key, f[key].name)

    # f['words'][0][0].decode("utf-8")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default="../data/mask")
    parser.add_argument(
        "--env", help="gym environment to load", default="MiniGrid-HazardWorld-B-v0"
    )
    parser.add_argument('--len', '-l', type=int, default=100)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env = gym.make(args.env)
    collect_data(args.fpath, env, args.len, args.episodes, not(args.norender))
