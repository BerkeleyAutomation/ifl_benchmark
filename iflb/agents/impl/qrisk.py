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
from .model import QNetworkConstraint, QNetworkConstraintCNN, DiscreteQNetworkConstraint, \
    DiscreteQNetworkConstraintCNN, StochasticPolicy, StochasticPolicyCNN


'''
Wrapper for training, querying, and visualizing Q_risk for Recovery RL
'''


class QRiskWrapper:
    def __init__(self, obs_space, ac_space, hidden_size, logdir,
                 args, mode='recovery'):
        self.env_name = args.env_name
        self.logdir = logdir
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.ac_space = ac_space
        self.images = args.cnn
        self.mode = mode

        if args.discrete:
            if not self.images:
                self.safety_critic = DiscreteQNetworkConstraint(
                    obs_space.shape[0], ac_space.n,
                    hidden_size).to(device=self.device)
                self.safety_critic_target = DiscreteQNetworkConstraint(
                    obs_space.shape[0], ac_space.n,
                    args.agent_cfg.hidden_size).to(device=self.device)
            else:
                self.safety_critic = DiscreteQNetworkConstraintCNN(
                    obs_space.shape, ac_space.n, hidden_size).to(self.device)
                self.safety_critic_target = DiscreteQNetworkConstraintCNN(
                    obs_space.shape, ac_space.n, hidden_size).to(self.device)
        else:
            if not self.images:
                self.safety_critic = QNetworkConstraint(
                    obs_space.shape[0], ac_space.shape[0],
                    hidden_size).to(device=self.device)
                self.safety_critic_target = QNetworkConstraint(
                    obs_space.shape[0], ac_space.shape[0],
                    args.agent_cfg.hidden_size).to(device=self.device)
            else:
                self.safety_critic = QNetworkConstraintCNN(
                    obs_space.shape, ac_space.shape[0], hidden_size).to(self.device)
                self.safety_critic_target = QNetworkConstraintCNN(
                    obs_space.shape, ac_space.shape[0], hidden_size).to(self.device)

        self.lr = args.agent_cfg.lr
        self.discrete = args.discrete
        self.safety_critic_optim = Adam(self.safety_critic.parameters(),
                                        lr=args.agent_cfg.lr)
        hard_update(self.safety_critic_target, self.safety_critic)

        self.tau = args.agent_cfg.tau_safe
        self.gamma_safe = args.agent_cfg.gamma_safe
        self.gamma = args.agent_cfg.gamma
        self.updates = 0
        self.target_update_interval = args.agent_cfg.target_update_interval
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        self.pos_fraction = args.agent_cfg.pos_fraction if args.agent_cfg.pos_fraction >= 0 else None
        self.ddpg_recovery = args.agent_cfg.get('ddpg_recovery', False)
        self.Q_sampling_recovery = args.agent_cfg.get('Q_sampling_recovery', False)
        if self.ddpg_recovery:
            if not self.images:
                self.policy = StochasticPolicy(obs_space.shape[0],
                                                ac_space.shape[0], hidden_size,
                                                ac_space).to(self.device)
            else:
                self.policy = StochasticPolicyCNN(obs_space.shape, ac_space.shape[0],
                                                    hidden_size, args.env_name,
                                                    ac_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.agent_cfg.lr)

    def update_parameters(self,
                          memory=None,
                          agent=None,
                          batch_size=None):
        '''
        Trains safety critic Q_risk and model-free recovery policy which performs
        gradient ascent on the safety critic

        Arguments:
            memory: Agent's replay buffer
            agent: forward agent
        '''
        policy = agent.policy
        if self.pos_fraction:
            batch_size = min(batch_size,
                             int((1 - self.pos_fraction) * len(memory)))
        else:
            batch_size = min(batch_size, len(memory))
        state_batch, action_batch, constraint_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size, pos_fraction=self.pos_fraction)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        if self.discrete:
            action_batch = torch.cat([torch.tensor([[a]]) for a in action_batch]).long().to(self.device)
        else:
            action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch).to(
            self.device).unsqueeze(1)

        with torch.no_grad():
            if isinstance(policy, list):
                next_state_action = agent.get_actions(next_state_batch, tensor=True)
            else:
                next_state_action, _, _ = policy.sample(
                    next_state_batch)
            if self.discrete:
                qf1_next_target, qf2_next_target = self.safety_critic_target(next_state_batch)
                next_state_action = torch.cat([torch.tensor([[a]]) for a in next_state_action]).long().to(self.device)
                qf1_next_target = qf1_next_target.gather(1, next_state_action)
                qf2_next_target = qf2_next_target.gather(1, next_state_action)
            else:
                qf1_next_target, qf2_next_target = self.safety_critic_target(
                    next_state_batch, next_state_action.float())

            if self.mode is not 'recovery':
                qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = constraint_batch + mask_batch * self.gamma * (
                    qf_next_target)
            else:
                qf_next_target = torch.max(qf1_next_target, qf2_next_target)
                next_q_value = constraint_batch + mask_batch * self.gamma_safe * (
                    qf_next_target)

        if self.discrete:
            qf1, qf2 = self.safety_critic(state_batch)
            qf1 = qf1.gather(1, action_batch)
            qf2 = qf2.gather(1, action_batch)
        else:
            qf1, qf2 = self.safety_critic(
                    state_batch, action_batch
                )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        self.safety_critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.safety_critic_optim.step()

        if self.ddpg_recovery:
            pi, log_pi, _ = self.policy.sample(state_batch)
            qf1_pi, qf2_pi = self.safety_critic(state_batch, pi.float())

            if self.mode is not 'recovery':
                sqf_pi = torch.min(qf1_pi, qf2_pi)
                policy_loss = ((0.2 * log_pi) - sqf_pi).mean()
            else:
                sqf_pi = torch.max(qf1_pi, qf2_pi)
                policy_loss = sqf_pi.mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

        if self.updates % self.target_update_interval == 0:
            soft_update(self.safety_critic_target, self.safety_critic,
                        self.tau)
        self.updates += 1

    def get_value(self, states, actions, encoded=False):
        '''
            Arguments:
                states, actions --> list of states and list of corresponding 
                actions to get Q_risk values for
            Returns: Q_risk(states, actions)
        '''
        with torch.no_grad():
            if self.discrete:
                q1, q2 = self.safety_critic(states)
                actions = torch.cat([torch.tensor([[a]]) for a in actions]).long().to(self.device)
                q1 = q1.gather(1, actions)
                q2 = q2.gather(1, actions)
            else:
                q1, q2 = self.safety_critic(states, actions)
            return torch.max(q1, q2)

    def select_action(self, state, eval=False):
        '''
            Gets action from model-free recovery policy

            Arguments:
                Current state
            Returns:
                action
        '''
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.ddpg_recovery:
            if eval is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]
        elif self.Q_sampling_recovery:
            if not self.images:
                state_batch = state.repeat(1000, 1)
            else:
                state_batch = state.repeat(1000, 1, 1, 1)
            sampled_actions = torch.FloatTensor(
                np.array([self.ac_space.sample()
                          for _ in range(1000)])).to(self.device)
            q_vals = self.get_value(state_batch, sampled_actions)
            min_q_value_idx = torch.argmin(q_vals)
            action = sampled_actions[min_q_value_idx]
            return action.detach().cpu().numpy()
        else:
            assert False

    def __call__(self, states, actions):
        return self.safety_critic(states, actions)
