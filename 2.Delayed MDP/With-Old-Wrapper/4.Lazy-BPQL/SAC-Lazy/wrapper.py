import random
from collections import deque
import gym
import numpy as np


class DelayedEnv(gym.Wrapper):
    def __init__(self, env, seed, obs_delayed_steps, max_obs_delayed_steps):
        super(DelayedEnv, self).__init__(env)
        assert obs_delayed_steps > 0
        assert max_obs_delayed_steps >= obs_delayed_steps
        self.env.action_space.seed(seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._max_episode_steps = self.env._max_episode_steps

        # TODO : augmented state 차원에 맞게 설정
        self.obs_buffer    = deque(maxlen=obs_delayed_steps)
        self.delay_buffer  = deque(maxlen=obs_delayed_steps)
        self.reward_buffer = deque(maxlen=obs_delayed_steps)
        self.done_buffer   = deque(maxlen=obs_delayed_steps)

        self.obs_delayed_steps      = obs_delayed_steps
        self.max_obs_delayed_steps  = max_obs_delayed_steps

    def reset(self):
        init_state, _ = self.env.reset()

        # TODO : initial state s1의 delayed time-steps 입니다.
        init_delay = random.randrange(1, self.max_obs_delayed_steps + 1)

        for _ in range(self.obs_delayed_steps):
            self.obs_buffer.append(init_state)
            self.delay_buffer.append(init_delay)
            self.reward_buffer.append(0)
            self.done_buffer.append(False)
        return init_state

    def step(self, action, no_ops = False):

        current_obs, current_reward, current_terminated, current_truncated, _ = self.env.step(action)
        current_done = current_terminated or current_truncated

        # TODO : state를 관측하지 못할 경우, no-ops 선택
        if no_ops:
            return current_obs, current_reward, 0, current_done, {'no-ops' : no_ops}

        # TODO : sample delayed time-steps of current_obs
        current_delayed_time_steps = random.randrange(1, self.max_obs_delayed_steps + 1)

        if self.obs_delayed_steps > 0:
            delayed_obs        = self.obs_buffer.popleft()
            delayed_reward     = self.reward_buffer.popleft()
            delayed_time_steps = self.delay_buffer.popleft()
            delayed_done       = self.done_buffer.popleft()

            self.obs_buffer.append(current_obs)
            self.reward_buffer.append(current_reward)
            self.delay_buffer.append(current_delayed_time_steps)
            self.done_buffer.append(current_done)
        else:
            delayed_obs = current_obs
            delayed_reward = current_reward
            delayed_time_steps = 0
            delayed_done = current_done

        return delayed_obs, delayed_reward, delayed_time_steps, delayed_done, {'current_obs': current_obs, 'current_reward': current_reward,
                                                           'current_done': current_done}



