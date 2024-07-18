import os
import sys
import random
import gymnasium as gym
import itertools
import argparse
import numpy as np
import torch
import trainer
from sac import SAC
from buffer import ReplayMemory
from wrapper import *

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

# TODO : Set MuJoCo task
parser.add_argument('--env-name', default="Walker2d-v4",
                    help='Mujoco Gym environment (default: HalfCheetah-v4)')

# TODO : normal or delayed
parser.add_argument('--type', default="Delayed",
                    help='Env-type (Normal | Delayed)')

# TODO : Set delayed steps
parser.add_argument('--obs-delayed-steps', default=0, type=int)
parser.add_argument('--act-delayed-steps', default=3, type=int)

parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')

parser.add_argument('--auto', type=float, default=True, metavar='G',
                    help='Automatically tuning temperature parameter α (default : False)')

parser.add_argument('--trial', default = 5,
                    help='number of trials (defalt : 1)')

parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')

parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='set seed (default: 123456)')

parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')

parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')

parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')

parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')

parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')

parser.add_argument('--num_steps', type=int, default=1e6, metavar='N',
                    help='maximum number of steps (default: 1e6)')

parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')

parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')

parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')

args = parser.parse_args()

if __name__ == '__main__':
    for trial_idx in range(0, args.trial):  # TODO : trials

        # TODO : Set seed
        args.seed = args.seed + trial_idx

        # TODO : Call Gym environment
        env_        = gym.make(args.env_name)
        state_dim  = env_.observation_space.shape[0]
        action_dim = env_.action_space.shape[0]
        action_bound = [env_.action_space.low[0], env_.action_space.high[0]]
        org_state_dim  = state_dim
        org_action_dim = action_dim

        if args.type == 'Delayed':
            env       = DelayedEnv(env_, seed=args.seed, obs_delayed_steps=args.obs_delayed_steps, act_delayed_steps=args.act_delayed_steps)
            eval_env  = DelayedEnv(env_, seed=args.seed, obs_delayed_steps=args.obs_delayed_steps, act_delayed_steps=args.act_delayed_steps)
            state_dim = state_dim + (state_dim * args.obs_delayed_steps) + (action_dim * args.act_delayed_steps)

        print(f"1. --------------- Call MuJoCo Env. '{args.env_name}' ---------------")
        print(f"2. --------------- Env. type '{args.type}' ---------------")
        if args.type == 'Delayed':
            print(f"      ------------ Delayed obs  '{args.obs_delayed_steps}' ------------   ")
            print(f"      ------------ Delayed acts '{args.act_delayed_steps}' ------------   ")

        # TODO : initialization (randomly or adding constant number)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # TODO : Call Agent and Replay buffer
        agent  = SAC(state_dim, action_dim, action_bound, org_state_dim, org_action_dim, args)
        memory = ReplayMemory(args.replay_size, args.seed)

        # TODO : Start training
        print(f"4. --------------- Start Training : {trial_idx} / {args.trial}---------------")
        trainer.train(env, eval_env, agent, memory, trial_idx, args)
