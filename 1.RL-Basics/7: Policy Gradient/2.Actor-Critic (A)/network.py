import torch
import torch.nn as nn
import torch.nn.functional as F

class UnityNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(4, 24)
        self.fc_pi = nn.Linear(24, 2)
        self.fc_v  = nn.Linear(24, 1)

    def pi(self, x):
        x      = F.relu(self.fc1(x))
        policy = F.softmax(self.fc_pi(x), dim = -1)
        return policy

    def v(self, x):
        v = F.relu(self.fc1(x))
        v = self.fc_v(v)
        return v