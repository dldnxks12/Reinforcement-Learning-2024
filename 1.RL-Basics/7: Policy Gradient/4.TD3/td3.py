import torch
import torch.nn.functional as F
import numpy as np
from buffer_td3 import ReplayMemory
from network import Twin_Q_net, Policy
from utils import hard_update, soft_update


class TD3:
    def __init__(self, state_dim, action_dim, action_bound, device, args):
        self.args = args

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.device = device
        self.buffer = ReplayMemory(self.state_dim, self.action_dim, device, args.buffer_size)
        self.batch_size = args.batch_size

        self.gamma = args.gamma
        self.tau = args.tau
        self.act_noise_scale = args.act_noise_scale

        self.actor = Policy(self.state_dim, self.action_dim, self.action_bound, self.device, args.hidden_dims).to(self.device)
        self.target_actor = Policy(self.state_dim, self.action_dim, self.action_bound, self.device, args.hidden_dims).to(self.device)
        self.critic = Twin_Q_net(self.state_dim, self.action_dim, self.device, args.hidden_dims).to(self.device)
        self.target_critic = Twin_Q_net(self.state_dim, self.action_dim, self.device, args.hidden_dims).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        hard_update(self.actor, self.target_actor)
        hard_update(self.critic, self.target_critic)

    def get_action(self, state, add_noise=True):
        with torch.no_grad():
            noise = np.random.normal(loc=0, scale=abs(self.action_bound[1]) * self.act_noise_scale, size=self.action_dim)
            if add_noise:
                action = self.actor(state).cpu().numpy()[0] + noise
                action = np.clip(action, self.action_bound[0], self.action_bound[1])
            else:
                action = self.actor(state).cpu().numpy()[0]

        return action

    def train_actor(self, states):
        actor_loss = -self.critic.Q_A(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def train_critic(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            target_act_noise = (torch.randn_like(actions) * self.args.target_noise_scale).clamp(-self.args.target_noise_clip, self.args.target_noise_clip).to(self.device)
            next_target_action = (self.target_actor(next_states) + target_act_noise).clamp(self.action_bound[0], self.action_bound[1])
            next_q_values_A, next_q_values_B = self.target_critic(next_states, next_target_action)
            next_q_values = torch.min(next_q_values_A, next_q_values_B)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        q_values_A, q_values_B = self.critic(states, actions)
        critic_loss = ((q_values_A - target_q_values)**2).mean() + ((q_values_B - target_q_values)**2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def train(self, option='both'):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        if option == 'both':  # train actor and critic both.
            actor_loss = self.train_actor(states)
            critic_loss = self.train_critic(states, actions, rewards, next_states, dones)

            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.critic, self.target_critic, self.tau)

        elif option == 'critic_only':  # train critic only
            actor_loss = 0
            critic_loss = self.train_critic(states, actions, rewards, next_states, dones)

        else:
            raise Exception("Wrong train option")

        return actor_loss, critic_loss

