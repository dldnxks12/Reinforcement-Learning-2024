import torch
import utils
import numpy as np
import td3
import trainer
import gymnasium as gym
import sys

def main():
    device        = 'cuda' if torch.cuda.is_available() else 'cpu'

    env_name      = "HalfCheetah-v4"
    env           = gym.make('HalfCheetah-v4')
    env_eval      = gym.make('HalfCheetah-v4')
    d_sample      = 8
    index         = 1

    num_model = 3

    print(f"Device : {device} | Delayed Sample : {d_sample} | Environment : {env_name}" )

    action_dim   = env.action_space.shape[0]
    action_bound = [env.action_space.low[0], env.action_space.high[0]]
    state_dim = env.observation_space.shape[0] + (d_sample * action_dim)

    max_step     = 10000000
    agent = td3.TD3(state_dim, action_dim, action_bound, device, d_sample, max_step, num_model)

    train = trainer.Trainer(env, env_eval, agent, env_name, device, d_sample*action_dim, max_step, num_model, action_dim, index)
    train.run()

if __name__ == "__main__":
    main()
