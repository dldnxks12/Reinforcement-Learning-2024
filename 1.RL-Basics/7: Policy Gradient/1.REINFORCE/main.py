import sys
import torch
import network
import gymnasium as gym
import utils
import torch.nn.functional as F
from torch.optim import Adam

"""
Hyperparameter에 민감하고 Avg.reward variance가 큰 것 같음.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')
policy = network.PolicyNetwork(env.observation_space.shape[0], 2, 10).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr = 0.005)

episode = 0
episode_max = 10000

while episode < episode_max:
    states, actions, rewards, score = utils.gen_epsiode(env, device, policy)

    # TODO : Calculate returns from start state; s0
    G_0 = utils.G(rewards)

    loss_temp = 0
    for state, action in zip(states, actions):

        # TODO : return and calculate log π(at|st)
        soft_max_prob = policy(state)
        log_soft_max_prob = soft_max_prob[action].log()
        loss_temp += log_soft_max_prob

    loss = -(loss_temp) * G_0

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode : {episode} / {episode_max} | Avg. reward : {score}")
    episode += 1

env.close()



