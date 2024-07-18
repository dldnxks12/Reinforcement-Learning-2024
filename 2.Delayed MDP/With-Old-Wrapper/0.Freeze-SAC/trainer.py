import numpy as np
from utils import log_to_txt

class Trainer:
    def __init__(self, env, eval_env, agent, trial, args):
        self.args = args
        self.agent = agent
        self.trial = trial

        self.env = env                    # undelayed environment
        self.eval_delayed_env = eval_env  # delayed environment

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

        self.delayed_steps     = args.obs_delayed_steps
        self.max_delayed_steps = args.max_obs_delayed_steps

    def learning(self):
        # TODO : train agent w.r.t maximum delay : O_max
        while not self.finish_flag:
            self.episode   += 1
            self.local_step = 0

            state = self.env.reset()[0]
            done = False

            # Episode starts here.
            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.total_step < self.start_step:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.get_action(state, evaluation=False)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                if terminated or truncated:
                    done = True

                self.agent.replay_memory.push(state, action, reward, next_state, done)
                state = next_state

                # Update parameters
                if self.agent.replay_memory.size >= self.batch_size and self.total_step >= self.update_after and \
                        self.total_step % self.update_every == 0:
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
            self.eval_delayed_env.reset()
            self.agent.eval_temporary_buffer.clear()
            self.eval_local_step = 0

            state_tag = 0
            done = False
            while not done:
                self.eval_local_step += 1

                if self.eval_local_step < self.delayed_steps:
                    action = np.zeros_like(self.eval_delayed_env.action_space.sample())
                    _, _, _, _, _ = self.eval_delayed_env.step(action)
                    self.agent.eval_temporary_buffer.actions.append(action)

                elif self.eval_local_step == self.delayed_steps:
                    action = np.zeros_like(self.eval_delayed_env.action_space.sample())
                    next_observed_state, _, delay, _, _ = self.eval_delayed_env.step(action)
                    self.agent.eval_temporary_buffer.actions.append(action)
                    self.agent.eval_temporary_buffer.states.append(next_observed_state)
                    self.agent.eval_temporary_buffer.delays.append(delay)
                    state_tag += 1

                else:
                    # TODO : Check if the state is observable at current time-step
                    last_observed_delay = self.agent.eval_temporary_buffer.delays[-1]
                    if self.eval_local_step < last_observed_delay + state_tag:
                        # TODO : state 관측 불가 >> no-ops
                        action = np.zeros_like(self.eval_delayed_env.action_space.sample()) # no_ops
                        _, _, _, _, _ = self.eval_delayed_env.step(action, no_ops = True)

                    else:
                        # TODO : state 관측 가능
                        last_observed_state = self.agent.eval_temporary_buffer.states[-1]
                        action = self.agent.get_action(last_observed_state, evaluation=True)

                        next_observed_state, reward, delay, done, info = self.eval_delayed_env.step(action)

                        self.agent.eval_temporary_buffer.actions.append(action)
                        self.agent.eval_temporary_buffer.states.append(next_observed_state)
                        self.agent.eval_temporary_buffer.delays.append(delay)

                        state_tag  += 1
                        episode_reward += reward

            reward_list.append(episode_reward)

        log_to_txt(self.args.env_name, self.args.random_seed,  self.delayed_steps, self.max_delayed_steps, self.total_step, sum(reward_list) / len(reward_list), self.trial)
        print("Eval  |  Total Steps {}  |  Episodes {}  |  Average Reward {:.2f}  |  Max reward {:.2f}  |  "
              "Min reward {:.2f}".format(self.total_step, self.episode, sum(reward_list) / len(reward_list),
                                          max(reward_list), min(reward_list)))







