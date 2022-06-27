'''
Built on on SAC implementation from 
https://github.com/pranz24/pytorch-soft-actor-critic
'''

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from dotmap import DotMap
import cv2

from .utils import soft_update, hard_update
from .model import GaussianPolicy, QNetwork, DeterministicPolicy, QNetworkCNN, \
    GaussianPolicyCNN, QNetworkConstraint, QNetworkConstraintCNN, DeterministicPolicyCNN
from .qrisk import QRiskWrapper


class SAC(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 args,
                 logdir):

        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        # Parameters for SAC
        self.gamma = args.agent_cfg.gamma
        self.tau = args.agent_cfg.tau
        self.alpha = args.agent_cfg.alpha
        self.env_name = args.env_name
        self.logdir = logdir
        self.policy_type = args.agent_cfg.policy
        self.target_update_interval = args.agent_cfg.target_update_interval
        self.automatic_entropy_tuning = args.agent_cfg.automatic_entropy_tuning
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.updates = 0
        self.cnn = args.cnn
        self.log_freq = args.log_freq

        # SAC setup
        if args.cnn:
            self.critic = QNetworkCNN(observation_space.shape, action_space.shape[0],
                                      args.agent_cfg.hidden_size,
                                      args.env_name).to(device=self.device)
            self.critic_target = QNetworkCNN(
                observation_space.shape, action_space.shape[0], args.agent_cfg.hidden_size,
                args.env_name).to(device=self.device)
        else:
            self.critic = QNetwork(observation_space.shape[0],
                                   action_space.shape[0],
                                   args.agent_cfg.hidden_size).to(device=self.device)
            self.critic_target = QNetwork(
                observation_space.shape[0], action_space.shape[0],
                args.agent_cfg.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.agent_cfg.lr)

        hard_update(self.critic_target, self.critic)
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A)
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1,
                                             requires_grad=True,
                                             device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.agent_cfg.lr)

            if args.cnn:
                self.policy = GaussianPolicyCNN(observation_space.shape,
                                                action_space.shape[0],
                                                args.agent_cfg.hidden_size,
                                                args.env_name,
                                                action_space).to(self.device)
            else:
                self.policy = GaussianPolicy(observation_space.shape[0],
                                             action_space.shape[0],
                                             args.agent_cfg.hidden_size,
                                             action_space).to(self.device)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            if args.cnn:
                self.policy = DeterministicPolicyCNN(observation_space.shape,
                                            action_space.shape[0],
                                            args.agent_cfg.hidden_size,
                                            args.env_name,
                                            action_space).to(self.device)
            else:
                self.policy = DeterministicPolicy(observation_space.shape[0],
                                              action_space.shape[0],
                                              args.agent_cfg.hidden_size,
                                              action_space).to(self.device)

        self.policy_optim = Adam(self.policy.parameters(), lr=args.agent_cfg.lr) 

        # Initialize safety critic
        self.safety_critic = QRiskWrapper(observation_space,
                                          action_space,
                                          args.agent_cfg.hidden_size,
                                          logdir,
                                          args,
                                          mode='recovery')

    def save(self):
        torch.save(self.policy.state_dict(), osp.join(self.logdir, "policy.ckpt"))
        torch.save(self.critic.state_dict(), osp.join(self.logdir, "critic.ckpt"))
        torch.save(self.safety_critic.safety_critic.state_dict(), osp.join(self.logdir, "safety_critic.ckpt"))

    def load(self, logdir):
        self.policy.load_state_dict(torch.load(osp.join(logdir, "policy.ckpt")))
        self.policy.eval()
        self.critic.load_state_dict(torch.load(osp.join(logdir, "critic.ckpt")))
        self.critic.eval()
        self.safety_critic.safety_critic.load_state_dict(torch.load(osp.join(logdir, "safety_critic.ckpt")))
        self.safety_critic.safety_critic.eval()

    def select_action(self, state, eval=False):
        '''
            Get action from current task policy
        '''
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        if eval is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def get_policy_uncertainty(self, state, eval=False):
        return 0 # Not implemented currently

    def update_parameters(self,
                          memory,
                          batch_size,
                          updates,
                          safety_critic=None):
        '''
        Train task policy and associated Q function with experience in replay buffer (memory)
        '''
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(
            self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            min_qf_next_target = torch.min(
                qf1_next_target,
                qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (
                min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi.float())
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        sqf1_pi, sqf2_pi = self.safety_critic(state_batch, pi.float())
        max_sqf_pi = torch.max(sqf1_pi, sqf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(
        ), alpha_loss.item(), alpha_tlogs.item()
