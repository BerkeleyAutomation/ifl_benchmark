'''
Networks based on SAC implementation from 
https://github.com/pranz24/pytorch-soft-actor-critic
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
'''
Global utilities
'''


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# Soft update of target critic network
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


# Hard update of target critic network
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


'''
Architectures for critic functions and policies for SAC model-free recovery
policies.
'''


# Q network architecture
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


# Q network architecture for image observations
class QNetworkCNN(nn.Module):
    def __init__(self, observation_shape, num_actions, hidden_dim, env_name):
        super(QNetworkCNN, self).__init__()
        h, w, c = observation_shape
        assert h == 64 and w == 64
        self.conv = nn.Sequential(
            nn.Conv2d(c, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            nn.Flatten(),
        )
        # Q1
        self.linear1 = nn.Sequential(
            nn.Linear(64 + num_actions, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2
        self.linear2 = nn.Sequential(
            nn.Linear(64 + num_actions, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        state = state.permute(0, 3, 1, 2)
        state = self.conv(state)
        xu = torch.cat([state, action], dim=-1)
        return torch.squeeze(self.linear1(xu), -1), torch.squeeze(self.linear2(xu), -1)

# Discrete Q_risk network architecture
class DiscreteQNetworkConstraint(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DiscreteQNetworkConstraint, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_inputs)
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):

        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = F.sigmoid(self.linear3(x1))

        x2 = F.relu(self.linear4(state))
        x2 = F.relu(self.linear5(x2))
        x2 = F.sigmoid(self.linear6(x2))

        return x1, x2

# Q_risk network architecture
class QNetworkConstraint(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetworkConstraint, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_inputs + num_actions)
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.sigmoid(self.linear3(x1))

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = F.sigmoid(self.linear6(x2))

        return x1, x2

# Discrete Q_risk network architecture for image observations
class DiscreteQNetworkConstraintCNN(nn.Module):
    def __init__(self, observation_shape, num_actions, hidden_dim):
        super(DiscreteQNetworkConstraintCNN, self).__init__()
        h, w, c = observation_shape
        assert h == 64 and w == 64
        self.conv = nn.Sequential(
            nn.Conv2d(c, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            nn.Flatten(),
        )
        # Q1
        self.linear1 = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Sigmoid()
        )
        # Q2
        self.linear2 = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Sigmoid()
        )

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        state = self.conv(state)
        return torch.squeeze(self.linear1(state), -1), torch.squeeze(self.linear2(state), -1)

# Q_risk network architecture for image observations
class QNetworkConstraintCNN(nn.Module):
    def __init__(self, observation_shape, num_actions, hidden_dim):
        super(QNetworkConstraintCNN, self).__init__()
        h, w, c = observation_shape
        assert h == 64 and w == 64
        self.conv = nn.Sequential(
            nn.Conv2d(c, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            nn.Flatten(),
        )
        # Q1
        self.linear1 = nn.Sequential(
            nn.Linear(64 + num_actions, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Q2
        self.linear2 = nn.Sequential(
            nn.Linear(64 + num_actions, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        state = state.permute(0, 3, 1, 2)
        state = self.conv(state)
        xu = torch.cat([state, action], dim=-1)
        return torch.squeeze(self.linear1(xu), -1), torch.squeeze(self.linear2(xu), -1)


# Gaussian policy for SAC
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample(
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        y_t = (action - self.action_bias)/self.action_scale
        x_t = torch.atanh(y_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


# Gaussian policy for SAC for image observations
class GaussianPolicyCNN(nn.Module):
    def __init__(self,
                 observation_shape,
                 num_actions,
                 hidden_dim,
                 env_name,
                 action_space=None):
        super(GaussianPolicyCNN, self).__init__()
        h, w, c = observation_shape
        assert h == 64 and w == 64 # CNN is currently structured for 64x64 img
        self.model = nn.Sequential(
            nn.Conv2d(c, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        #self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        state = self.model(state)
        mean = self.mean_linear(state)
        log_std = self.log_std_linear(state)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample(
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        y_t = (action - self.action_bias)/self.action_scale
        x_t = torch.atanh(y_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyCNN, self).to(device)

# Discrete policy
class DiscretePolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DiscretePolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        output = self.output(x)
        return output

    def sample(self, state):
        output = self.forward(state)
        _, output = torch.max(output, dim=1)
        return output, torch.tensor(0.), output

    def to(self, device):
        return super(DiscretePolicy, self).to(device)

# Deterministic policy for model free recovery
class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        #self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias 
        return mean

    def sample(self, state):
        mean = self.forward(state)
        #noise = self.noise.normal_(0., std=0.1)
        #noise = noise.clamp(-0.25, 0.25)
        action = mean #+ noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


# Stochastic policy for model free recovery
class StochasticPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(StochasticPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = torch.nn.Parameter(
            torch.as_tensor([np.log(0.1)] * num_actions))
        self.min_log_std = np.log(1e-6)

        self.apply(weights_init_)
        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        #print(self.log_std)
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        log_std = log_std.unsqueeze(0).repeat([len(mean), 1])
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample(self, state):
        dist = self.forward(state)
        action = dist.rsample()
        return action, dist.log_prob(action).sum(-1), dist.mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(StochasticPolicy, self).to(device)

# Discrete policy for image observations
class DiscretePolicyCNN(nn.Module):
    def __init__(self,
                 observation_shape,
                 num_actions,
                 hidden_dim,
                 action_space=None):
        super(DiscretePolicyCNN, self).__init__()
        # Process via a CNN and then collapse to linear
        h, w, c = observation_shape
        assert h == 64 and w == 64 # CNN is currently structured for 64x64 img
        self.model = nn.Sequential(
            nn.Conv2d(c, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        return self.model(state)

    def sample(self, state):
        logits = self.forward(state)
        _, preds = torch.max(logits, dim=1)
        return preds, torch.tensor(0.), preds

    def to(self, device):
        return super(DiscretePolicyCNN, self).to(device)

# Deterministic policy for model free recovery for image observations
class DeterministicPolicyCNN(nn.Module):
    def __init__(self,
                 observation_shape,
                 num_actions,
                 hidden_dim,
                 action_space=None):
        super(DeterministicPolicyCNN, self).__init__()
        # Process via a CNN and then collapse to linear
        h, w, c = observation_shape
        assert h == 64 and w == 64 # CNN is currently structured for 64x64 img
        self.model = nn.Sequential(
            nn.Conv2d(c, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Tanh()
        )
        self.noise = torch.Tensor(num_actions)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        return self.model(state) * self.action_scale + self.action_bias

    def sample(self, state):
        mean = self.forward(state)
        # noise = self.noise.normal_(0., std=0.1)
        # noise = noise.clamp(-0.25, 0.25)
        action = mean# + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicyCNN, self).to(device)


# Stochastic policy for model free recovery for image observations
class StochasticPolicyCNN(nn.Module):
    def __init__(self,
                 observation_shape,
                 num_actions,
                 hidden_dim,
                 env_name,
                 action_space=None):
        super(StochasticPolicyCNN, self).__init__()
        h, w, c = observation_shape
        assert h == 64 and w == 64 # CNN is currently structured for 64x64 img
        self.model = nn.Sequential(
            nn.Conv2d(c, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = torch.nn.Parameter(
            torch.as_tensor([0.0] * num_actions))
        self.min_log_std = np.log(1e-6)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        x = self.model(state)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        log_std = log_std.unsqueeze(0).repeat([len(mean), 1])
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample(self, state):
        dist = self.forward(state)
        action = dist.rsample()
        return action, dist.log_prob(action).sum(-1), dist.mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(StochasticPolicyCNN, self).to(device)
