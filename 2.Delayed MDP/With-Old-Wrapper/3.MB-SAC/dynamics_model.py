import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device):
        super(DynamicsModel, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        state, action = self._format(state, action)
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        next_state = self.fc3(x)
        return next_state

    def _format(self, state, action):
        x, u = state, action

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u