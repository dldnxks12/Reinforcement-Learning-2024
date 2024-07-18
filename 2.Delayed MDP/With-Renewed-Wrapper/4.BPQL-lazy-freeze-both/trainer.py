import sys
import numpy as np
from utils import log_to_txt
from collections import deque
class Trainer:
    def __init__(self, env, eval_env, agent, trial, args):
        self.args = args
        self.agent = agent
        self.trial = trial

        self.delayed_env      = env
        self.eval_delayed_env = eval_env

        self.start_step = args.start_step
        self.update_after = args.update_after
        self.max_step = args.max_step
        self.batch_size = args.batch_size
        self.update_every = args.update_every

        self.eval_flag = args.eval_flag
        self.eval_episode = args.eval_episode
        self.eval_freq = args.eval_freq

        self.episode = 0
        self.total_step  = 0
        self.local_step  = 0
        self.state_tag   = 0
        
        self.eval_local_step  = 0
        self.eval_state_tag   = 0
        self.eval_num = 0
        self.finish_flag = False
        
        self.init_obs_delayed_steps  = args.init_obs_delayed_steps
        self.min_obs_delayed_steps   = args.min_obs_delayed_steps
        self.max_obs_delayed_steps   = args.max_obs_delayed_steps

    def train(self):
        while not self.finish_flag:
            self.episode   += 1
            self.local_step = 0

            # Tag for ordering -> if an ordered state is processed, counts up +1
            self.state_tag  = 1

            # temporal buffer for observable states but cannot be processed immediately.
            obs_temp     = []
            reward_temp  = []
            done_temp    = []

            # Initialize the delayed environment & the temporal buffer
            self.delayed_env.reset()
            self.agent.temporary_buffer.clear()

            freeze_flag = False
            done = False

            # Episode starts here.
            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.local_step < self.init_obs_delayed_steps:  # if t < d
                    action     = np.zeros_like(self.delayed_env.action_space.sample())  # Select the 'no-op' action
                    obs_list, reward_list, done_list, _ = self.delayed_env.step(action)
                    self.agent.temporary_buffer.actions.append(action)

                    if len(obs_list) > 0: # Append observable states
                        for obs, rwd, dn in zip(obs_list, reward_list, done_list):
                            obs_temp.append(obs)
                            reward_temp.append(rwd)
                            done_temp.append(dn)

                elif self.local_step == self.init_obs_delayed_steps:  # if t == d
                    if self.total_step < self.start_step:
                        action = self.delayed_env.action_space.sample()
                    else:
                        action = np.zeros_like(self.delayed_env.action_space.sample())  # Select the 'no-op' action

                    obs_list, reward_list, done_list, _ = self.delayed_env.step(action)

                    if len(obs_list) > 0:  # Append observable states and sort
                        for obs, rwd, dn in zip(obs_list, reward_list, done_list):
                            obs_temp.append(obs)
                            reward_temp.append(rwd)
                            done_temp.append(dn)

                    obs_temp    = sort_list(obs_temp)
                    reward_temp = sort_list(reward_temp)
                    done_temp   = sort_list(done_temp)

                    # init_obs_delay_step 만큼 기다렸으니, s1이 무조건 관측 가능
                    # assert obs_temp[0][1] == 1

                    next_state = obs_temp[0][0]  # state
                    self.state_tag += 1   # state is processed -> count +1
                    # delete processed one
                    del obs_temp[0]
                    del reward_temp[0]
                    del done_temp[0]

                    # temporary buffer에는 처리될 순서에 맞게 넣어줘야함.
                    #                s(1)       <-     Env: a(d)
                    self.agent.temporary_buffer.actions.append(action)     # Put a(d) to the temporary buffer
                    self.agent.temporary_buffer.states.append(next_state)  # Put s(1) to the temporary buffer

                else:  # if t > d
                    if freeze_flag == True:
                        action = np.zeros_like(self.delayed_env.action_space.sample())

                    elif self.total_step < self.start_step:
                        action = self.delayed_env.action_space.sample()

                    else:
                        last_observed_state = self.agent.temporary_buffer.states[-1]
                        first_action_idx = len(self.agent.temporary_buffer.actions) - self.init_obs_delayed_steps

                        # Get the augmented state(t)
                        augmented_state = self.agent.temporary_buffer.get_augmented_state(last_observed_state, first_action_idx)
                        action = self.agent.get_action(augmented_state, evaluation=False)

                    # a(t) <- policy: augmented_state(t)
                    obs_list, reward_list, done_list, _ = self.delayed_env.step(action, no_ops = freeze_flag)

                    if len(obs_list) > 0:  # Append observable states and sort
                        for obs, rwd, dn in zip(obs_list, reward_list, done_list):
                            obs_temp.append(obs)
                            reward_temp.append(rwd)
                            done_temp.append(dn)

                    obs_temp = sort_list(obs_temp)
                    reward_temp = sort_list(reward_temp)
                    done_temp = sort_list(done_temp)

                    if len(obs_temp) > 0 and self.state_tag == obs_temp[0][1]:
                        # print(f"Current tag {self.state_tag} | Ready to be processed state idx {obs_temp[0][1]}")
                        freeze_flag = False # flag off
                        next_state = obs_temp[0][0]  # state
                        reward     = reward_temp[0][0]
                        done       = done_temp[0][0]
                        self.state_tag +=1

                        del obs_temp[0]
                        del reward_temp[0]
                        del done_temp[0]

                        true_done = 0.0 if self.local_step == self.delayed_env._max_episode_steps + self.init_obs_delayed_steps else float(done)

                        self.agent.temporary_buffer.actions.append(action)    # Put a(t) to the temporary buffer
                        self.agent.temporary_buffer.states.append(next_state) # Put s(t+1-d) to the temporary buffer
                        
                        if len(self.agent.temporary_buffer.states) >= (self.init_obs_delayed_steps + 2):
                            augmented_s, s, a, next_augmented_s, next_s = self.agent.temporary_buffer.get_tuple()

                            # Store [aug_s(t-d), s(t-d), a(t-d), r(t-d), aug_s(t+1-d), s(t+1-d)] in replay memory.
                            self.agent.replay_memory.push(augmented_s, s, a, reward, next_augmented_s, next_s, true_done)

                    # 관측가능한 state도 없고, 처리할 state도 없을 때 -> no-ops
                    else:
                        freeze_flag = True  # flag on
                        if len(obs_temp) == 0:
                            print("관측 가능한 state 없음.")
                        else:
                            print("처리할 수 있는 state 없음.")

                # Update parameters
                if self.agent.replay_memory.size >= self.batch_size \
                        and self.total_step >= self.update_after and \
                        self.total_step % self.update_every == 0:
                    total_actor_loss = 0
                    total_critic_loss = 0
                    total_log_alpha_loss = 0
                    for i in range(self.update_every):
                        # Train the policy and the beta Q-network (critic).
                        critic_loss, actor_loss, log_alpha_loss = self.agent.train()
                        total_critic_loss += critic_loss
                        total_actor_loss += actor_loss
                        total_log_alpha_loss += log_alpha_loss

                    # Print the loss.
                    if self.args.show_loss:
                        print("Loss  |  Actor loss {:.3f}  |  Critic loss {:.3f}  |  Log-alpha loss {:.3f}"
                              .format(total_actor_loss / self.update_every, total_critic_loss / self.update_every,
                                      total_log_alpha_loss / self.update_every))

                # Evaluate.
                if self.eval_flag and self.total_step % self.eval_freq == 0:
                    self.evaluate()

                # Raise finish flag.
                if self.total_step == self.max_step:
                    self.finish_flag = True

    def evaluate(self):
        self.eval_num += 1
        reward_list = []

        for epi in range(self.eval_episode):
            episode_reward = 0
            self.eval_local_step = 0

            # Tag for ordering -> if an ordered state is processed, counts up +1
            self.eval_state_tag = 1

            # temporal buffer for observable states but cannot be processed immediately.
            eval_obs_temp    = []
            eval_reward_temp = []
            eval_done_temp   = []

            self.eval_delayed_env.reset()
            self.agent.eval_temporary_buffer.clear()

            eval_freeze_flag = False
            done = False

            while not done:
                self.eval_local_step += 1

                if self.eval_local_step < self.init_obs_delayed_steps:
                    action = np.zeros_like(self.delayed_env.action_space.sample())
                    eval_obs_list, eval_reward_list, eval_done_list, _ = self.eval_delayed_env.step(action)
                    self.agent.eval_temporary_buffer.actions.append(action)

                    if len(eval_obs_list) > 0:  # Append observable states and sort
                        for obs, rwd, dn in zip(eval_obs_list, eval_reward_list, eval_done_list):
                            eval_obs_temp.append(obs)
                            eval_reward_temp.append(rwd)
                            eval_done_temp.append(dn)

                elif self.eval_local_step == self.init_obs_delayed_steps:
                    action = np.zeros_like(self.eval_delayed_env.action_space.sample())
                    eval_obs_list, eval_reward_list, eval_done_list, _ = self.eval_delayed_env.step(action)

                    if len(eval_obs_list) > 0:  # Append observable states and sort
                        for obs, rwd, dn in zip(eval_obs_list, eval_reward_list, eval_done_list):
                            eval_obs_temp.append(obs)
                            eval_reward_temp.append(rwd)
                            eval_done_temp.append(dn)

                    eval_obs_temp    = sort_list(eval_obs_temp)
                    eval_reward_temp = sort_list(eval_reward_temp)
                    eval_done_temp   = sort_list(eval_done_temp)

                    # 이 때는 init_obs_delay_step 만큼 기다렸으니, S1이 무조건 관측 가능
                    next_state = eval_obs_temp[0][0]  # state
                    self.eval_state_tag += 1  # state is processed -> count +1
                    # delete processed one
                    del eval_obs_temp[0]
                    del eval_reward_temp[0]
                    del eval_done_temp[0]

                    self.agent.eval_temporary_buffer.actions.append(action)
                    self.agent.eval_temporary_buffer.states.append(next_state)

                else:
                    if eval_freeze_flag == True:
                        action = np.zeros_like(self.eval_delayed_env.action_space.sample())
                    else:
                        last_observed_state = self.agent.eval_temporary_buffer.states[-1]
                        first_action_idx    = len(self.agent.eval_temporary_buffer.actions) - self.init_obs_delayed_steps
                        augmented_state     = self.agent.eval_temporary_buffer.get_augmented_state(last_observed_state, first_action_idx)
                        action = self.agent.get_action(augmented_state, evaluation=True)

                    eval_obs_list, eval_reward_list, eval_done_list, info = self.eval_delayed_env.step(action, no_ops = eval_freeze_flag)

                    if len(eval_obs_list) > 0:  # Append observable states and sort
                        for obs, rwd, dn in zip(eval_obs_list, eval_reward_list, eval_done_list):
                            eval_obs_temp.append(obs)
                            eval_reward_temp.append(rwd)
                            eval_done_temp.append(dn)

                    eval_obs_temp    = sort_list(eval_obs_temp)
                    eval_reward_temp = sort_list(eval_reward_temp)
                    eval_done_temp   = sort_list(eval_done_temp)

                    if len(eval_obs_temp) > 0 and self.eval_state_tag == eval_obs_temp[0][1]:
                        eval_freeze_flag = False
                        next_state = eval_obs_temp[0][0]  # state
                        reward = eval_reward_temp[0][0]
                        done = eval_done_temp[0][0]
                        self.eval_state_tag += 1

                        del eval_obs_temp[0]
                        del eval_reward_temp[0]
                        del eval_done_temp[0]

                        self.agent.eval_temporary_buffer.actions.append(action)
                        self.agent.eval_temporary_buffer.states.append(next_state)

                        episode_reward += reward

                    # TODO : 해당 state 아직 처리 불가 >> no-ops
                    else:
                        eval_freeze_flag = True
                        if len(eval_obs_temp) == 0:
                            print("관측 가능한 state 없음.")
                        else:
                            print("처리할 수 있는 state 없음.")

            reward_list.append(episode_reward)

        log_to_txt(self.args.env_name, self.args.random_seed,  self.init_obs_delayed_steps, self.min_obs_delayed_steps, self.max_obs_delayed_steps, self.total_step, sum(reward_list) / len(reward_list), self.trial)
        print("Eval  |  Total Steps {}  |  Episodes {}  |  Average Reward {:.2f}  |  Max reward {:.2f}  |  "
              "Min reward {:.2f}".format(self.total_step, self.episode, sum(reward_list) / len(reward_list),
                                          max(reward_list), min(reward_list)))


def sort_list(lst):
    length = len(lst)
    for i in range(length):
        for j in range(i, length):
            if lst[i][1] >= lst[j][1]:
                lst[i], lst[j] = lst[j], lst[i]
    return lst




