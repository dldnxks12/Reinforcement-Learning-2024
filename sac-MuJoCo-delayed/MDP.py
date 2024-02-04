import os
import sys
import random
import gymnasium as gym
import itertools
import argparse
import numpy as np
import torch
from sac import SAC
from utils import make_delayed_env
from buffer import ReplayMemory
from temporary_buffer import TemporaryBuffer

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

parser.add_argument('--env-name', default="InvertedPendulum-v4")
parser.add_argument('--type', default="Normal", help='Normal | Delayed')
parser.add_argument('--delayed-steps', default=3)
parser.add_argument('--trial', default=5)
parser.add_argument('--eval', type=bool, default=True, help='eval a policy every 10 epi')
parser.add_argument('--alpha', type=float, default=0.2) # TODO : tunable parameter (temperature)
parser.add_argument('--auto', type=float, default=False)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--num_steps', type=int, default=1000001)
parser.add_argument('--replay_size', type=int, default=10000000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_size', type=int, default=256)

parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')

parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')

parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')

args = parser.parse_args()

for trial_idx in range(0, args.trial):
    args.seed = args.seed + random.randint(1, 1000)

    # Environment
    env = make_delayed_env(args, args.seed, act_delayed_steps=args.delayed_steps)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Memories
    memory                = ReplayMemory(args.replay_size, args.seed)
    temporary_memory      = TemporaryBuffer(args.delayed_steps)
    temporary_memory_eval = TemporaryBuffer(args.delayed_steps)

    # Training Loop
    total_numsteps = 0
    updates        = 0

    for i_episode in itertools.count(1): # 1씩 증가시키는 무한 반복자

        episode_reward = 0
        episode_steps  = 0

        done = False
        state = env.reset()[0]
        temporary_memory.clear()
        action_buf = []
        while not done:
            episode_steps  += 1
            total_numsteps += 1
            if episode_steps < args.delayed_steps:         # t < d
                action     = np.zeros_like(env.action_space.sample()) # 'no-ops'
                action_buf.append(action)
                _, _, _, _ = env.step(action)

            elif episode_steps == args.delayed_steps:      # t == d
                if args.start_steps > total_numsteps:
                    action = env.action_space.sample()     # get random action
                else:
                    action = np.zeros_like(env.action_space.sample()) # 'no-ops'

                next_state, _, _, _ = env.step(action)
                temporary_memory.states.append(next_state) # Put s(d+1)
                #temporary_memory.actions.append(action)   # Put a(d)
                action_buf.append(action)                  # put a(d)

            else: # t > d
                state = temporary_memory.states[-1]
                first_action_idx  = len(action_buf) - args.delayed_steps

                I = np.concatenate([state, action_buf[first_action_idx]])
                for i in range(first_action_idx+1, first_action_idx+args.delayed_steps):
                    I = np.concatenate([I, action_buf[i]])

                if episode_steps < args.start_steps:
                    action = env.action_space.sample()               # get random action
                else:
                    action = agent.select_action(I)    # get action from policy

                next_state, reward, done, _ = env.step(action)
                mask = 1 if episode_steps  == (env._max_episode_steps + args.delayed_steps) else float(not done)

                temporary_memory.states.append(next_state)  # Put s(t+1)
                action_buf.append(action)                   # Put a(t)

                I_next = np.concatenate([next_state, action_buf[first_action_idx+1]])
                for i in range(first_action_idx+2, first_action_idx+args.delayed_steps+1):
                    I_next = np.concatenate([I_next, action_buf[i]])

                memory.push(I, state, action, reward, I_next, next_state, mask)

            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step): # Number of updates per step in environment
                    critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1

        if i_episode % 5 == 0 and args.eval is True:
            avg_reward = 0.0
            episodes   = 5
            for _  in range(episodes):
                state = env.reset()[0]
                temporary_memory_eval.clear()
                eval_episode_steps = 0
                episode_reward = 0
                done = False
                eval_action_buf = []
                while not done:
                    eval_episode_steps += 1
                    if eval_episode_steps < args.delayed_steps:
                        action     = np.zeros_like(env.action_space.sample())
                        _, _, _, _ = env.step(action)
                        eval_action_buf.append(action)
                    elif eval_episode_steps == args.delayed_steps:
                        action = np.zeros_like(env.action_space.sample())
                        next_state, _, _, _ = env.step(action)
                        eval_action_buf.append(action)
                        temporary_memory_eval.states.append(next_state)
                    else:
                        state = temporary_memory_eval.states[-1]
                        first_action_idx = len(eval_action_buf) - args.delayed_steps
                        I = np.concatenate([state, eval_action_buf[first_action_idx]])
                        for i in range(first_action_idx + 1, first_action_idx + args.delayed_steps):
                            I = np.concatenate([I, eval_action_buf[i]])

                        action = agent.select_action(I, evaluate=True)
                        next_state, reward, done, _ = env.step(action)
                        eval_action_buf.append(action)
                        temporary_memory_eval.states.append(next_state)
                        episode_reward += reward

                avg_reward += episode_reward
            avg_reward /= episodes

            print("----------------------------------------")
            print("Eval-Epi: {}, Total-Steps: {}, Avg-Reward: {}".format(episode_steps, total_numsteps, round(avg_reward, 2)))
            print("----------------------------------------")

        if total_numsteps > args.num_steps:
            agent.save_checkpoint(args.env_name, args.alpha, args.type, trial_idx+1, args.trial)
            break

    env.close()