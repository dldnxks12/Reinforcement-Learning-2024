import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define the dynamics model
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        next_state = self.fc3(x)
        return next_state


# Initialize the dynamics model
state_dim = 2
action_dim = 1
hidden_dim = 64
dynamics_model = DynamicsModel(state_dim, action_dim, hidden_dim)
optimizer = optim.Adam(dynamics_model.parameters(), lr=0.001)


# Simulated environment step function
def env_step(state, action):
    next_state = state + action + np.random.randn(state_dim) * 0.01
    reward = -np.sum(np.square(next_state))  # Reward is negative distance from origin
    return next_state, reward


# Training the dynamics model
def train_dynamics_model(dynamics_model, optimizer, data, epochs=100):
    dynamics_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for state, action, next_state in data:
            state = torch.tensor(state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            optimizer.zero_grad()
            predicted_next_state = dynamics_model(state, action)
            loss = nn.MSELoss()(predicted_next_state, next_state)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss / len(data)}')


# Generate training data
def generate_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        next_state, _ = env_step(state, action)
        data.append((state, action, next_state))
    return data


# Model Predictive Control (MPC)
def mpc_controller(dynamics_model, state, horizon=10):
    dynamics_model.eval()
    best_action = None
    best_value = -float('inf')
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    for _ in range(100):  # Sample random actions
        action_seq = [torch.tensor(np.random.randn(action_dim), dtype=torch.float32).unsqueeze(0) for _ in
                      range(horizon)]
        total_reward = 0.0
        current_state = state

        for action in action_seq:
            next_state = dynamics_model(current_state, action)
            reward = -torch.sum(next_state ** 2).item()  # Reward is negative distance from origin
            total_reward += reward
            current_state = next_state

        if total_reward > best_value:
            best_value = total_reward
            best_action = action_seq[0]

    return best_action.detach().numpy()


# Main training loop
data = generate_data()
train_dynamics_model(dynamics_model, optimizer, data)

# Test the MPC controller
state = np.random.randn(state_dim)
for _ in range(10):
    action = mpc_controller(dynamics_model, state)
    next_state, reward = env_step(state, action)
    print(f'State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}')
    state = next_state