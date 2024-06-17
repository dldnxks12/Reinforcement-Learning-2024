import math
import torch
from wrapper import DelayedEnv


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def make_delayed_env(args, random_seed, act_delayed_steps):
    import gymnasium as gym

    env_name    = args.env_name
    env         = gym.make(env_name)
    delayed_env = DelayedEnv(env, seed = random_seed, act_delayed_steps=act_delayed_steps)

    return delayed_env
