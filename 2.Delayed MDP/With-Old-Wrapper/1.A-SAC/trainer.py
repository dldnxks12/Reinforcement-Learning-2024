import os
import sys
import random
import gymnasium as gym
import itertools
import argparse
import numpy as np
import torch
import temporaryBuffer

def train(env, eval_env, agent, memory, trial_idx, args):
    total_numsteps = 0
    updates = 0

    total_delayed_steps   = args.obs_delayed_steps + args.act_delayed_steps
    temporary_buffer      = temporaryBuffer.TemporaryBuffer(total_delayed_steps)

    for i_episode in itertools.count(1):  # 1씩 증가시키는 무한 반복자
        episode_reward = 0
        episode_steps  = 0
        done = False

        _ = env.reset()
        temporary_buffer.clear()

        while not done:
            episode_steps  += 1
            total_numsteps += 1

            if episode_steps < total_delayed_steps: # if t < d
                action = np.zeros_like(env.action_space.sample()) # do no-ops
                _, _, _, _ = env.step(action)
                temporary_buffer.actions.append(action)

            elif episode_steps == total_delayed_steps: # if t = d
                if args.start_steps > total_numsteps:
                    action = env.action_space.sample()   # get random action
                else:
                    action = np.zeros_like(env.action_space.sample()) # do no-ops

                next_state, _, _, _ = env.step(action)

                temporary_buffer.actions.append(action)
                temporary_buffer.states.append(next_state)

            else: # if t > d
                last_state       = temporary_buffer.states[-1]
                first_act_idx = len(temporary_buffer.actions) - total_delayed_steps
                augmented_state = temporary_buffer.get_augmented_state(last_state, first_act_idx)
                if args.start_steps > total_numsteps:
                    action = env.action_space.sample()  # get random action
                else:
                    action = agent.select_action(augmented_state)  # get action from policy

                next_state, reward, done, _ = env.step(action)
                temporary_buffer.actions.append(action)
                temporary_buffer.states.append(next_state)

                episode_reward += reward

                true_done = 1 if episode_steps == env._max_episode_steps + args.obs_delayed_steps else float(not done)

                if episode_steps > 2 * total_delayed_steps: # if t > 2d
                    #  aug_s(t-d),  s(t-d),  a(t-d),  aug_s(t+1-d),  s(t+1-d)  <- Temporal Buffer
                    augmented_s, s, a, next_augmented_s, next_s = temporary_buffer.get_tuple()
                    #  Store (aug_s(t-d), s(t-d), a(t-d), r(t-d), aug_s(t+1-d), s(t+1-d)) in the replay memory.
                    memory.push(augmented_s, s, a, reward, next_augmented_s, next_s, true_done)

            if len(memory) > args.batch_size:
                # Number of updates per step = default : 1
                for i in range(args.updates_per_step):
                    agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps, round(episode_reward, 2)))

        # TODO : Evaluation loop
        if i_episode % 10 == 0 and args.eval is True:
            evaluate(eval_env, agent, episode_steps, args)

        if total_numsteps > args.num_steps:
            agent.save_checkpoint(args.env_name, args.alpha, args.type, trial_idx + 1, args.trial)
            break

    env.close()


def evaluate(eval_env, agent,episode_steps, args):
    avg_reward    = 0
    eval_episodes = 10
    total_delayed_steps   = args.obs_delayed_steps + args.act_delayed_steps
    eval_temporary_buffer = temporaryBuffer.TemporaryBuffer(total_delayed_steps)

    for _ in range(eval_episodes):
        eval_temporary_buffer.clear()
        eval_episode_reward = 0
        eval_step = 0
        _ = eval_env.reset()
        done = False

        while not done:
            if eval_step < total_delayed_steps:
                action = np.zeros_like(eval_env.action_space.sample())
                _, _, _, _ = eval_env.step(action)
                eval_temporary_buffer.actions.append(action)

            elif eval_step == total_delayed_steps:
                action = np.zeros_like(eval_env.action_space.sample())
                next_state, _, _, _ = eval_env.step(action)
                eval_temporary_buffer.states.append(next_state)
                eval_temporary_buffer.actions.append(action)
            else:
                last_state = eval_temporary_buffer.states[-1]
                first_act_idx = len(eval_temporary_buffer.actions) - total_delayed_steps
                augmented_state = eval_temporary_buffer.get_augmented_state(last_state, first_act_idx)
                action = agent.select_action(augmented_state, evaluate=True)  # get mean action
                next_state, reward, done, _ = eval_env.step(action)
                eval_temporary_buffer.actions.append(action)
                eval_temporary_buffer.states.append(next_state)
                eval_episode_reward += reward

            eval_step += 1
        avg_reward += eval_episode_reward
    avg_reward /= eval_episodes

    print("----------------------------------------")
    print("Eval. Episodes: {}, Avg. Reward: {}".format(episode_steps, round(avg_reward, 2)))
    print("----------------------------------------")
