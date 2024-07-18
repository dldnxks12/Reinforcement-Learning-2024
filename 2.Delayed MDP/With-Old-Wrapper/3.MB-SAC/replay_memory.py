import torch
import numpy as np

class ReplayMemory:
    def __init__(self, state_dim, action_dim, device, capacity=1e6):
        self.device = device
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0

        self.action_buffer = np.empty(shape=(self.capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.done_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.state_buffer = np.empty(shape=(self.capacity, state_dim), dtype=np.float32)
        self.next_state_buffer = np.empty(shape=(self.capacity, state_dim), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.size = min(self.size + 1, self.capacity)

        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.done_buffer[self.position] = done
        self.state_buffer[self.position] = state
        self.next_state_buffer[self.position] = next_state

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        actions = torch.FloatTensor(self.action_buffer[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.reward_buffer[idxs]).to(self.device)
        dones = torch.FloatTensor(self.done_buffer[idxs]).to(self.device)
        states = torch.FloatTensor(self.state_buffer[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_state_buffer[idxs]).to(self.device)

        return actions, rewards, dones, states, next_states