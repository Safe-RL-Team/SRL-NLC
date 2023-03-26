#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger

WORD_TO_IMAGE = {
    'water': (12, 2, 0),
    'lava': (9, 0, 0),
    'grass': (11, 1, 0)
}


def cheating_mask(obs, avoid_obj):
    constrain_mask = np.all(obs['image'] == WORD_TO_IMAGE[avoid_obj], axis=2).astype(int)
    return constrain_mask


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    logger = EpochLogger()
    env.set_max_step(max_ep_len)
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    avoid_obj = env.avoid_obj
    cost_lim = env.hc
    o_img = o['image']
    pre_violations = o["violations"]
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-4)

        c = o["violations"] - pre_violations
        pre_violations = o["violations"]
        constraint_mask = np.expand_dims(cheating_mask(o, avoid_obj), axis=2)
        cost_budget_mask = np.zeros(constraint_mask.shape)
        cost_budget_mask[constraint_mask == 1] = o["violations"] - cost_lim
        mask = np.append(constraint_mask, cost_budget_mask, axis=2)
        input_x = np.append(o_img, mask, axis=2)

        a = get_action(input_x)
        a = np.clip(a, 0, env.action_space.n - 1)
        a = int(a)
        o, r, d, info = env.step(a)
        o_img = o['image']

        ep_ret += r
        # ep_cost += info.get('cost', 0)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d' % (n, ep_ret, o["violations"], ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            avoid_obj = env.avoid_obj
            cost_lim = env.hc
            o_img = o['image']
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >= 0 else 'last',
                                        args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not (args.norender))
