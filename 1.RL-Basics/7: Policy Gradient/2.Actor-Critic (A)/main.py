import gymnasium as gym
import sys
import math
import numpy as np
from time import sleep
from network import *
from buffer import *
from utils import *
import torch
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

alpha = 0.01
gamma = 0.99
episode = 0
MAX_EPISODE = 10000

Net       = UnityNetwork().to(device)
optimizer = optim.Adam(Net.parameters(), lr = alpha)

memory= ReplayBuffer()

def train(memory, Net, optimizer):
    batch_size = 256
    states, actions, rewards, next_states, terminateds, truncateds = memory.sample(batch_size)

    loss  = 0

    for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):
        if terminated or truncated:
            td_target = reward
        else:
            td_target = reward + gamma * Net.v(next_state)

        delta = td_target - Net.v(state)
        pi = Net.pi(state)[action] + 1e-5

        loss += (delta**2) - (torch.log(pi)*delta.detach())

    loss = loss / batch_size
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

env = gym.make('CartPole-v1')

while episode < MAX_EPISODE:

    state = env.reset()
    state = torch.tensor(state[0]).float().to(device)
    score = 0

    # Make Experiences
    while True:
        policy = Net.pi(state)
        action = torch.multinomial(policy, 1).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(next_state).float().to(device)
        memory.put((state, action, reward, next_state, terminated, truncated))

        score += reward
        state = next_state

        if truncated or terminated:
            break

        if memory.size() > 500:
            train(memory, Net, optimizer)

    print(f"Episode : {episode} || Reward : {score} ")
    episode += 1

env.close()