import numpy as np
from utils import log_to_txt

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
        self.total_step = 0
        self.local_step = 0
        self.eval_local_step = 0
        self.eval_num = 0
        self.finish_flag = False

        self.init_obs_delayed_steps = args.init_obs_delayed_steps
        self.min_obs_delayed_steps  = args.min_obs_delayed_steps
        self.max_obs_delayed_steps  = args.max_obs_delayed_steps

    def learning(self):
        # TODO : train agent w.r.t maximum delay : O_max
        while not self.finish_flag:
            self.episode   += 1
            self.local_step = 0

            # Tag for ordering -> if an ordered state is processed, counts up +1
            self.state_tag = 1

            # temporal buffer for observable states but cannot be processed immediately.
            obs_temp    = []
            reward_temp = []
            done_temp   = []

            # Initialize the delayed environment & the temporal buffer
            state = self.delayed_env.reset()

            freeze_flag = False
            done        = False

            # Episode starts here.
            while not done:
                self.local_step += 1
                self.total_step += 1

                if freeze_flag == True or self.local_step < self.init_obs_delayed_steps:
                    # TODO : Select no-ops
                    action = np.zeros_like(self.delayed_env.action_space.sample())

                elif self.total_step < self.start_step:
                    action = self.delayed_env.action_space.sample()

                else:
                    # TODO : Get action from delayed state
                    action = self.agent.get_action(state, evaluation=False)

                obs_list, reward_list, done_list, _ = self.delayed_env.step(action, no_ops = freeze_flag)

                if len(obs_list) > 0:  # Append observable states and sort
                    for obs, rwd, dn in zip(obs_list, reward_list, done_list):
                        obs_temp.append(obs)
                        reward_temp.append(rwd)
                        done_temp.append(dn)

                if len(obs_temp) > 0:
                    obs_temp    = sort_list(obs_temp)
                    reward_temp = sort_list(reward_temp)
                    done_temp   = sort_list(done_temp)

                if len(obs_temp) > 0 and self.state_tag == obs_temp[0][1]:
                    #print(f"Current tag {self.state_tag} | Ready to be processed state idx {obs_temp[0][1]}")
                    freeze_flag = False  # flag off
                    next_state  = obs_temp[0][0]  # state
                    reward      = reward_temp[0][0]
                    done        = done_temp[0][0]
                    self.state_tag += 1

                    del obs_temp[0]
                    del reward_temp[0]
                    del done_temp[0]

                    true_done = 0.0 if self.local_step == self.delayed_env._max_episode_steps + self.init_obs_delayed_steps else float(done)

                    self.agent.replay_memory.push(state, action, reward, next_state, true_done)

                    state = next_state

                # 관측가능한 state도 없고, 처리할 state도 없을 때 -> no-ops
                else:
                    freeze_flag = True  # flag on
                    # if len(obs_temp) == 0:
                    #     pass
                    #     # print("관측 가능한 state 없음.")
                    #     # print("-------------------")
                    # else:
                    #     pass
                    #     # print("처리할 수 있는 state 없음.")
                    #     # print("-------------------")
                    #     # print(f"C : {self.state_tag} | R : {obs_temp[0][1]}")
                    #     # print("-------------------")

                # Update parameters
                if self.agent.replay_memory.size >= self.batch_size \
                        and self.total_step >= self.update_after \
                        and self.total_step % self.update_every == 0:
                    for i in range(self.update_every):
                        # Train the actor and critic.
                        self.agent.train()

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

            state = self.eval_delayed_env.reset()
            self.agent.eval_temporary_buffer.clear()

            eval_freeze_flag = False
            done = False

            while not done:
                self.eval_local_step += 1

                if eval_freeze_flag == True or self.eval_local_step < self.init_obs_delayed_steps:
                    action = np.zeros_like(self.eval_delayed_env.action_space.sample())
                else:
                    action = self.agent.get_action(state, evaluation=True)

                eval_obs_list, eval_reward_list, eval_done_list, info = self.eval_delayed_env.step(action, no_ops = eval_freeze_flag)

                if len(eval_obs_list) > 0:  # Append observable states and sort
                    for obs, rwd, dn in zip(eval_obs_list, eval_reward_list, eval_done_list):
                        eval_obs_temp.append(obs)
                        eval_reward_temp.append(rwd)
                        eval_done_temp.append(dn)

                if len(eval_obs_temp) > 0:
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

                    episode_reward += reward
                    state = next_state

                # TODO : 해당 state 아직 처리 불가 >> no-ops
                else:
                    eval_freeze_flag = True
                    # if len(eval_obs_temp) == 0:
                    #     pass
                    #     # print("EVAL : 관측 가능한 state 없음.")
                    # else:
                    #     pass
                    #     # print("EVAL : 처리할 수 있는 state 없음.")
                    #     # print("-------------------")
                    #     # print(f"C : {self.eval_state_tag} | R : {eval_obs_temp[0][1]}")
                    #     # print("-------------------")

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



