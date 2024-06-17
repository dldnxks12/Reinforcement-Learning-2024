# https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py

import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from network import GaussianPolicy, QNetwork

class SAC(object):
    def __init__(self, state_space, action_space, args):
        self.alpha  = args.alpha  # temperature
        self.gamma  = args.gamma  # discount factor for infinite horizon case
        self.tau    = args.tau    # update

        self.auto_tuning = args.auto
        self.target_update_interval   = args.target_update_interval
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"--------------- with {self.device} ---------------")

        self.critic        = QNetwork(state_space, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target = QNetwork(state_space, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.lr)

        hard_update(self.critic_target, self.critic)  # parameter 동기화

        # TODO : select analzable policy family >> e.g. Gaussian Policy
        self.policy        = GaussianPolicy(state_space, args.delayed_steps, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim  = Adam(self.policy.parameters(), lr=args.lr)

        if self.auto_tuning == True:
            # fixed entropy H : -dim(A)
            self.target_alpha = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha    = torch.zeros(1, requires_grad = True, device = self.device)
            self.alpha_optim  = Adam([self.log_alpha], lr = args.lr)


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state) # action (from distribution)
        else:
            _, _, action = self.policy.sample(state) # action (mean)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        I_batch, state_batch, action_batch, reward_batch, I_next_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        I_batch          = torch.FloatTensor(I_batch).to(self.device)
        state_batch      = torch.FloatTensor(state_batch).to(self.device)
        action_batch     = torch.FloatTensor(action_batch).to(self.device)
        reward_batch     = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        I_next_batch     = torch.FloatTensor(I_next_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch       = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(I_next_batch)
            qf1_next_target, qf2_next_target        = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value       = reward_batch + self.gamma * (min_qf_next_target) * mask_batch

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        acts, log_pi, _ = self.policy.sample(I_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, acts)
        min_qf_pi      = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.auto_tuning == True:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_alpha).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()


    # Save model parameters
    def save_checkpoint(self, env_name, temper, type, cur, tot, ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}_{}_[{}-{}]".format(env_name, type, temper, cur, tot)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict'          : self.policy.state_dict(),
                    'critic_state_dict'          : self.critic.state_dict(),
                    'critic_target_state_dict'   : self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(        checkpoint['policy_state_dict'])
            self.critic.load_state_dict(        checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict( checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(  checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(  checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()