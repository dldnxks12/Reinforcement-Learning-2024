import random
import gymnasium as gym
import sys
import numpy as np
def sort_list(lst):
    length = len(lst)
    for i in range(length):
        for j in range(i, length):
            if lst[i][1] >= lst[j][1]:
                lst[i], lst[j] = lst[j], lst[i]
    return lst

class DelayedEnv(gym.Wrapper):
    def __init__(self, env, seed,
                 init_obs_delayed_steps,
                 min_obs_delayed_steps=0,
                 max_obs_delayed_steps=15,
                 mode='parallel'):

        super(DelayedEnv, self).__init__(env)
        assert init_obs_delayed_steps > 0
        assert max_obs_delayed_steps >= init_obs_delayed_steps
        self.env.action_space.seed(seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._max_episode_steps = self.env._max_episode_steps

        self.obs_buffer    = []
        self.reward_buffer = []
        self.done_buffer   = []

        self.init_obs_delayed_steps = init_obs_delayed_steps
        self.min_obs_delayed_steps  = min_obs_delayed_steps
        self.max_obs_delayed_steps  = max_obs_delayed_steps

        self.state_idx  = 0
        self.local_step = 0
        self.init_state = None
        self.init_delay = 0

        self.mode = mode

    def reset(self, seed=None, option=None):
        init_state, _ = self.env.reset()
        self.init_state = init_state
        self.state_idx  = 1
        self.local_step = 1

        self.init_delay = self.init_obs_delayed_steps

        self.obs_buffer    = []
        self.reward_buffer = []
        self.done_buffer   = []

        # meta obs: (a, b, c) -> a = obs, b = collected time, c = how delayed
        self.obs_buffer.append([init_state, self.state_idx, self.init_delay])
        self.reward_buffer.append([0, self.state_idx, self.init_delay])
        self.done_buffer.append([False, self.state_idx, self.init_delay])
        return init_state

    def step(self, action, no_ops = False):

        current_obs, current_reward, current_terminated, current_truncated, _ = self.env.step(action)
        current_done = current_terminated or current_truncated
        data_dict = {'current_obs': current_obs, 'current_reward': current_reward, 'current_done': current_done}

        if not no_ops:
            self.local_step += 1
            self.state_idx  += 1

            # delay in [min_delayed_steps, max_delayed_steps]
            delay = random.randrange(self.min_obs_delayed_steps, self.max_obs_delayed_steps + 1)

            # meta obs: (a, b, c) -> a= obs, b=collected time, c= how delayed
            meta_obs    = [current_obs,    self.state_idx, delay]
            meta_reward = [current_reward, self.state_idx, delay]
            meta_done   = [current_done,   self.state_idx, delay]

            # Push the current obs, rwd, done with metadata.
            self.obs_buffer.append(meta_obs)
            self.reward_buffer.append(meta_reward)
            self.done_buffer.append(meta_done)

            # Check if obs. delay is out of bound
            assert len(self.obs_buffer) <= (self.max_obs_delayed_steps - self.min_obs_delayed_steps) + 1

            ret_meta_obs_list    = []
            ret_meta_reward_list = []
            ret_meta_done_list   = []

            state_lst = []
            for i, meta_obs in enumerate(self.obs_buffer):
                # print("Observable check part in 'normal condition' : ", meta_obs[1] + meta_obs[2], self.local_step)
                if meta_obs[1] + meta_obs[2] <= self.local_step:  # induced time + delay <= current time
                    ret_meta_obs_list.append(self.obs_buffer[i])
                    state_lst.append(self.obs_buffer[i][1])
                    ret_meta_reward_list.append(self.reward_buffer[i])
                    ret_meta_done_list.append(self.done_buffer[i])

            # Pop the delayed obs, rwd, done
            for meta_obs in ret_meta_obs_list:
                for i, obs in enumerate(self.obs_buffer):
                    if obs[1] == meta_obs[1]:
                        idx = i
                del self.obs_buffer[idx]

            for meta_reward in ret_meta_reward_list:
                for i, rwd in enumerate(self.reward_buffer):
                    if rwd[1] == meta_reward[1]:
                        idx = i
                del self.reward_buffer[idx]

            for meta_done in ret_meta_done_list:
                for i, dn in enumerate(self.done_buffer):
                    if dn[1] == meta_done[1]:
                        idx = i
                del self.done_buffer[idx]

            if self.mode == 'parallel':  # TCP communication
                return ret_meta_obs_list, ret_meta_reward_list, ret_meta_done_list, data_dict
            else:
                return ret_meta_obs_list, ret_meta_reward_list, ret_meta_done_list, data_dict

        else: # no-ops:
            self.local_step += 1

            ret_meta_obs_list = []
            ret_meta_reward_list = []
            ret_meta_done_list = []

            state_lst = []
            for i, meta_obs in enumerate(self.obs_buffer):
                # print("Observable check part in 'no-ops condition' : ", meta_obs[1] + meta_obs[2], self.local_step)
                if meta_obs[1] + meta_obs[2] <= self.local_step:  # induced time + delay <= current time
                    ret_meta_obs_list.append(self.obs_buffer[i])
                    state_lst.append(self.obs_buffer[i][1])
                    ret_meta_reward_list.append(self.reward_buffer[i])
                    ret_meta_done_list.append(self.done_buffer[i])

            # Pop the delayed obs, rwd, done
            for meta_obs in ret_meta_obs_list:
                for i, obs in enumerate(self.obs_buffer):
                    if obs[1] == meta_obs[1]:
                        idx = i
                del self.obs_buffer[idx]

            for meta_reward in ret_meta_reward_list:
                for i, rwd in enumerate(self.reward_buffer):
                    if rwd[1] == meta_reward[1]:
                        idx = i
                del self.reward_buffer[idx]

            for meta_done in ret_meta_done_list:
                for i, dn in enumerate(self.done_buffer):
                    if dn[1] == meta_done[1]:
                        idx = i
                del self.done_buffer[idx]

            if self.mode == 'parallel':  # TCP communication
                return ret_meta_obs_list, ret_meta_reward_list, ret_meta_done_list, data_dict
            else:
                return ret_meta_obs_list, ret_meta_reward_list, ret_meta_done_list, data_dict
