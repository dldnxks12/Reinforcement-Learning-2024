import utils
import torch.nn as nn
import torch
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim = 256):
        super(Critic, self).__init__()

        self.device = device
        self.apply(utils.weight_init)

        self.fc_A1   = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc_A2   = nn.Linear(hidden_dim, hidden_dim)
        self.fc_Aout = nn.Linear(hidden_dim, 1)

        self.fc_B1   = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc_B2   = nn.Linear(hidden_dim, hidden_dim)
        self.fc_Bout = nn.Linear(hidden_dim, 1)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action):
        s, a = self._format(state, action)
        X = torch.cat([s, a], dim = 1)

        A = F.relu(self.fc_A1(X))
        A = F.relu(self.fc_A2(A))
        QA = self.fc_Aout(A)

        B = F.relu(self.fc_B1(X))
        B = F.relu(self.fc_B2(B))
        QB = self.fc_Bout(B)

        return QA, QB

    def Q_A(self, state, action):
        with torch.no_grad():
            s, a = self._format(state, action)
            if s.shape[0] == 128:
                X = torch.cat([s, a], dim=1)
            else:
                X = torch.cat([s.unsqueeze(0), a], dim = 1)
            A = F.relu(self.fc_A1(X))
            A = F.relu(self.fc_A2(A))
            QA = self.fc_Aout(A)
        return QA


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, device, hidden_dim = 256):
        super(Actor, self).__init__()
        self.apply(utils.weight_init)
        self.device = device
        self.action_bound = action_bound
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.action_rescale      = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)

        x = x.to(self.device)
        x  = F.relu(self.fc1(x))
        x  = F.relu(self.fc2(x))
        x  = self.fc3(x)
        x  = F.tanh(x)
        x  = x * self.action_rescale + self.action_rescale_bias

        return x
