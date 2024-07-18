import numpy as np
from utils import log_to_txt


class Trainer:
    def __init__(self, env, eval_env, agent, args):
        self.args = args

        self.agent = agent
        self.env_name = args.env_name
        self.env = env
        self.eval_env = eval_env

        self.start_step = args.start_step
        self.update_after = args.update_after
        self.max_step = args.max_step
        self.batch_size = args.batch_size
        self.update_every = args.update_every

        self.target_noise_scale = args.target_noise_scale

        self.eval_flag = args.eval_flag
        self.eval_episode = args.eval_episode
        self.eval_freq = args.eval_freq

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0
        self.finish_flag = False

        self.target_noise_scale = args.target_noise_scale
        self.policy_update_delay = args.policy_update_delay

    def evaluate(self):
        import sys
        import torch
        # Evaluate process
        self.eval_num += 1
        reward_list = []

        Action_buffer = [torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0])]

        for epi in range(self.eval_episode):
            epi_reward = 0
            #state = self.eval_env.reset()
            state = self.eval_env.reset()[0]
            state = torch.FloatTensor(state)
            I = torch.concatenate([state.unsqueeze(0), torch.FloatTensor([Action_buffer[-3], Action_buffer[-2], Action_buffer[-1]]).unsqueeze(0)] , dim = 1)

            done = False

            while not done:
                action = self.agent.get_action(state, add_noise=False)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                epi_reward += reward
                state = next_state
            reward_list.append(epi_reward)

        if self.args.log:
            log_to_txt(self.args, self.total_step, sum(reward_list)/len(reward_list))
        print("Eval  |  total_step {}  |  episode {}  |  Average Reward {:.2f}  |  Max reward: {:.2f}  |  "
              "Min reward: {:.2f}".format(self.total_step, self.episode, sum(reward_list)/len(reward_list),
                                               max(reward_list), min(reward_list), np.std(reward_list)))

    def run(self):
        # Train-process start.
        while not self.finish_flag:
            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0

            state = self.env.reset()[0]

            done = False

            # Episode start.
            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.total_step >= self.start_step:
                    action = self.agent.get_action(state, add_noise=True)
                else:
                    action = self.env.action_space.sample()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.episode_reward += reward

                done_mask = 0.0 if self.local_step == self.env._max_episode_steps else float(done)
                self.agent.buffer.push(state, action, reward, next_state, done_mask)

                state = next_state

                # Update parameters
                if self.agent.buffer.size >= self.batch_size and self.total_step >= self.update_after and \
                        self.total_step % self.update_every == 0:
                    total_actor_loss = 0
                    total_critic_loss = 0
                    for i in range(self.update_every):
                        if (i+1) % self.policy_update_delay:
                            actor_loss, critic_loss = self.agent.train(option='both')  # train actor and critic both.
                        else:
                            actor_loss, critic_loss = self.agent.train(option='critic_only')  # train critic only
                        total_actor_loss += actor_loss
                        total_critic_loss += critic_loss

                    # Print loss.
                    if self.args.show_loss:
                        print("Loss  |  Actor loss {:.3f}  |  Critic loss {:.3f}"
                              .format(total_actor_loss/(self.update_every / self.policy_update_delay),
                                      total_critic_loss / self.update_every))

                # Evaluation.
                if self.eval_flag and self.total_step % self.eval_freq == 0:
                    self.evaluate()

                # Raise finish_flag.
                if self.total_step == self.max_step:
                    self.finish_flag = True










