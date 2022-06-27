import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from .replay_memory import ReplayMemory
from .model import DeterministicPolicy, \
    DeterministicPolicyCNN
from .qrisk import QRiskWrapper

class BC(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 args,
                 logdir):

        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        self.env_name = args.env_name
        self.logdir = logdir
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.cnn = args.cnn
        self.num_policies = args.agent_cfg.num_policies
        self.log_freq = args.log_freq

        self.observation_space = observation_space
        self.action_space = action_space
        self.args = args
        self.discrete = self.args.discrete
        self.init_policies()
    
        # Initialize safety critic
        self.safety_critic = QRiskWrapper(observation_space,
                                          action_space,
                                          args.agent_cfg.hidden_size,
                                          logdir,
                                          args,
                                          mode='recovery')

        # Initialize goal critic
        self.goal_critic = QRiskWrapper(observation_space,
                                          action_space,
                                          args.agent_cfg.hidden_size,
                                          logdir,
                                          args,
                                          mode='forward')

    def init_policies(self):
        args = self.args
        observation_space = self.observation_space
        action_space = self.action_space
        if args.discrete:
            # don't need ensemble for discrete policy
            if args.cnn:
                self.policy = DiscretePolicyCNN(observation_space.shape,
                                            action_space.n,
                                            args.agent_cfg.hidden_size,
                                            action_space).to(self.device)
            else:
                self.policy = DiscretePolicy(observation_space.shape[0],
                                                    action_space.n,
                                                    args.agent_cfg.hidden_size,
                                                    action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.agent_cfg.lr)
        else:
            if args.cnn:
                self.policy = [DeterministicPolicyCNN(observation_space.shape,
                                                action_space.shape[0],
                                                args.agent_cfg.hidden_size,
                                                action_space).to(self.device) for _ in range(self.num_policies)]
            else:
                self.policy = [DeterministicPolicy(observation_space.shape[0],
                                                    action_space.shape[0],
                                                    args.agent_cfg.hidden_size,
                                                    action_space).to(self.device) for _ in range(self.num_policies)]
            self.policy_optims = [Adam(self.policy[i].parameters(), lr=args.agent_cfg.lr) for i in range(self.num_policies)]

    def save(self):
        torch.save(self.policy, osp.join(self.logdir, "policy.ckpt"))
        torch.save(self.safety_critic.safety_critic.state_dict(), osp.join(self.logdir, "safety_critic.ckpt"))

    def load(self, logdir):
        self.policy = torch.load(osp.join(logdir, "policy.ckpt"))
        # for policy in self.policy:
        #     policy.eval()
        self.safety_critic.safety_critic.load_state_dict(osp.join(logdir, "safety_critic.ckpt"))
        # self.safety_critic.safety_critic.eval()

    def get_actions(self, states, tensor=False):
        '''
        Get action from current task policy (ensemble mean)
        '''
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        if self.discrete:
            with torch.no_grad():
                mean = self.policy.sample(states)[0]
        else:
            outputs = list()
            with torch.no_grad():
                for pi in self.policy:
                    outputs.append(pi.sample(states)[0])
            mean = torch.mean(torch.stack(outputs), axis=0)
        if tensor:
            return mean
        return mean.cpu().numpy()

    def get_policy_uncertainty(self, states):
        '''
        Get ensemble variance as estimate of epistemic uncertainty. Normalize by max variance.
        For discrete actions, calculate the entropy instead.
        '''
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        if self.discrete:
            with torch.no_grad():
                logits = self.policy(states)
                logits = nn.Softmax(dim=1)(logits).cpu().numpy()
                x = np.sum(-logits * np.log(logits), axis=1)
                return x
        else:
            outputs = list()
            with torch.no_grad():
                for pi in self.policy:
                    outputs.append(pi.sample(states)[0].cpu().numpy())
            max_variance = np.square(np.std(np.array([self.action_space.low, self.action_space.high]), axis=0)).mean()
            return np.square(np.std(np.array(outputs), axis=0)).mean(axis=1) / max_variance

    def train(self,
                 memory,
                 batch_size):
        '''
        train policy via behavior cloning on human data (1 gradient step)
        memory: ensemble of replay buffers
        '''
        # Sample a batch from memory
        for i, policy in enumerate(self.policy):
            state_batch, action_batch, _, _, _ = memory[i].sample(
                batch_size=batch_size)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)

            pi, _, _ = policy.sample(state_batch)
            policy_loss = F.mse_loss(pi, action_batch)

            self.policy_optims[i].zero_grad()
            policy_loss.backward()
            self.policy_optims[i].step()
        return policy_loss

    def train_discrete(self, memory, batch_size):
        state_batch, action_batch, _, _, _ = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        pi = self.policy(state_batch)
        policy_loss = nn.CrossEntropyLoss()(pi, action_batch)
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        return policy_loss

    def retrain(self,
                 memory,
                 batch_size):
        '''
        retrain policy from scratch bootstrapping from memory replay buf
        '''
        # Sample a batch from memory
        # auto compute grad steps from replay buf size & batch size
        gradient_steps = int((memory.size / batch_size) * 100)
        self.init_policies() # reinit policies
        for i, policy in enumerate(self.policy):
            tmp_buffer = ReplayMemory(memory.capacity, memory.seed)
            for _ in range(memory.size):
                elem = memory.buffer[np.random.randint(memory.size)]
                tmp_buffer.push(elem[0].copy(), elem[1].copy(), elem[2], elem[3].copy(), elem[4])
            for j in range(gradient_steps):
                state_batch, action_batch, _, _, _ = tmp_buffer.sample(
                    batch_size=batch_size)
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                pi, _, _ = policy.sample(state_batch)
                policy_loss = F.mse_loss(pi, action_batch)
                self.policy_optims[i].zero_grad()
                policy_loss.backward()
                self.policy_optims[i].step()
                if j % 500 == 0:
                    print("NN #{} Step {} Loss {}".format(i, j, policy_loss.item()))
        return policy_loss


