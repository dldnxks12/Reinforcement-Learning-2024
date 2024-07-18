from collections import deque
import random
class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen = 50000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, number):
        mini_batch = random.sample(self.buffer, number)
        states, actions, rewards, next_states, terminateds, truncateds = [], [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, terminated, truncated = transition

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)
            truncateds.append(truncated)

        return states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)