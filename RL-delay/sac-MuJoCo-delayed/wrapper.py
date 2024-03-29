from collections import deque
import gymnasium as gym
import numpy as np

class DelayedEnv(gym.Wrapper):
    def __init__(self, env, seed, act_delayed_steps):
        super(DelayedEnv, self).__init__(env)
        assert act_delayed_steps > 0
        self.env.action_space.seed(seed)

        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space

        self._max_episode_steps = self.env._max_episode_steps

        self.action_buffer = deque(maxlen=act_delayed_steps)

        self.act_delayed_steps = act_delayed_steps

    def reset(self):
        for _ in range(self.act_delayed_steps):
            self.action_buffer.append(np.zeros_like(self.env.action_space.sample()))
        init_state, _ = self.env.reset()
        return init_state

    def step(self, action):
        if self.act_delayed_steps > 0:
            delayed_action = self.action_buffer.popleft()
            self.action_buffer.append(action)
        else:
            delayed_action = action

        curr_obs, curr_reward, curr_terminated, curr_truncated, _ = self.env.step(delayed_action)
        curr_done = curr_terminated or curr_truncated

        return curr_obs, curr_reward, curr_done, \
            {'current_obs': curr_obs, 'current_reward': curr_reward, 'current_done': curr_done}


