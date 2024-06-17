import numpy as np
from collections import deque


class TemporaryBuffer:
    def __init__(self, delayed_steps):
        self.d = delayed_steps
        self.states  = deque(maxlen=2)
        self.actions = deque(maxlen=delayed_steps + 1)

    def clear(self):
        self.states.clear()
        self.actions.clear()

    def get_augmented_state(self, last_observed_state, first_action_idx):
        aug_state = np.concatenate([last_observed_state, self.actions[first_action_idx]])
        for i in range(first_action_idx + 1, first_action_idx + self.d):
            aug_state = np.concatenate([aug_state, self.actions[i]])
        return aug_state

    def get_tuple(self):
        assert len(self.states) == 2 and len(self.actions) == self.d + 1

        I      = self.get_augmented_state(self.states[0], 0)
        I_next = self.get_augmented_state(self.states[1], 1)

        state      = self.states[0]
        next_state = self.states[1]

        self.states.popleft()
        self.actions.popleft()
        return I, state, I_next, next_state
