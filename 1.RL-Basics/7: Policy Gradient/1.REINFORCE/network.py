# Q를 G로 근사 ... Q or V의 Network 학습 필요 XX

import gymnasium as gym
import sys
import math
import torch
import random
import numpy as np
from time import sleep
import torch.nn as nn
import torch.nn.functional as F

def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim = -1)
        return x