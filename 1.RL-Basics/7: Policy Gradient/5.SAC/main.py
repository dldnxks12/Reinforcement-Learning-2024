import os
import sys
import random
import gymnasium as gym
import itertools
import argparse
import numpy as np
import torch
from sac import SAC
from buffer import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

# TODO : environments
parser.add_argument('--env-name', default="HalfCheetah-v3",
                    help='Mujoco Gym environment (default: HalfCheetah-v4)')

# TODO : temperature
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')

# TODO : automatic tuning (c.f. SAC-v2)
parser.add_argument('--auto', type=float, default=True, metavar='G',
                    help='Automatically tuning temperature parameter α (default : False)')

# TODO : normal or delayed
parser.add_argument('--type', default="Normal",
                    help='Env-type (Normal | Delayed(num))')

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

parser.add_argument('--num_steps', type=int, default=1e5, metavar='N',
                    help='maximum number of steps (default: 1e6)')

# parser.add_argument('--num_steps', type=int, default=1e6, metavar='N',
#                     help='maximum number of steps (default: 1e6)')

parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')

parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')

parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')

args = parser.parse_args()

# TODO : 5번의 trial
for trial_idx in range(0, args.trial):

    # TODO : Call test environment
    env     = gym.make(args.env_name)

    print(f"1. --------------- Call MuJoCo Env. '{args.env_name}' ---------------")

    # TODO : seed initialization (randomly or adding constant number)
    # args.seed = args.seed + random.randint(1, 1000)
    args.seed = args.seed + trial_idx
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TODO : Call SAC Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # TODO : Call replay buffer
    memory = ReplayMemory(args.replay_size, args.seed)

    # TODO : Start training
    print(f"3. --------------- Start Training Loop ---------------")

    total_numsteps = 0
    updates        = 0

    for i_episode in itertools.count(1): # 1씩 증가시키는 무한 반복자

        episode_reward = 0
        episode_steps  = 0
        done = False

        state = env.reset(seed = args.seed)[0]

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()     # get random action
            else:
                action = agent.select_action(state)    # get action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step = default : 1
                for i in range(args.updates_per_step):
                    agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1

            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                done = True

            episode_steps  += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        # TODO : Evaluation loop
        if i_episode % 10 == 0 and args.eval is True:
            avg_reward = 0
            eval_episodes = 10
            for _  in range(eval_episodes):
                state = env.reset(seed = args.seed)[0]
                episode_reward = 0
                done = False

                while not done:
                    action = agent.select_action(state, evaluate=True) # get mean action

                    next_state, reward, terminated, truncated, _ = env.step(action)  # Step
                    if terminated or truncated:
                        done = True

                    episode_reward += reward
                    state = next_state

                avg_reward += episode_reward
            avg_reward /= eval_episodes

            print("----------------------------------------")
            print("Eval. Episodes: {}, Avg. Reward: {}".format(episode_steps, round(avg_reward, 2)))
            print("----------------------------------------")

        if total_numsteps > args.num_steps:
            agent.save_checkpoint(args.env_name, args.alpha, args.type, trial_idx+1, args.trial)
            break

    env.close()