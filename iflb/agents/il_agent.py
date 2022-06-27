"""
An implementation of a parallel Imitation Learning agent
"""
import numpy as np
import torch
from .base_agent import ParallelAgent
from .impl.bc import BC
from .impl.replay_memory import ReplayMemory
import pickle
import random

def torchify(x, device): return torch.tensor(x, dtype=torch.float32).to(device)

class SingleTaskParallelILAgent(ParallelAgent):
    def __init__(self, envs, exp_cfg, logdir):
        self.exp_cfg = exp_cfg
        self.cfg = exp_cfg.agent_cfg
        if self.cfg.updates_per_step == -1:
            self.cfg.updates_per_step = self.exp_cfg.num_humans
        self.envs = envs
        self.logdir = logdir
        self.device = torch.device("cuda" if self.exp_cfg.cuda else "cpu")

        # Experiment setup
        self.experiment_setup()

        # Shared memory across all env steps
        self.human_memory = ReplayMemory(self.cfg.replay_size, exp_cfg.seed)
        # Each ensemble member's memory samples with replacement from main memory when constructed
        self.ensemble_memories = [ReplayMemory(self.cfg.replay_size, exp_cfg.seed+i) for i in range(self.cfg.num_policies)]
        self.recovery_memory = ReplayMemory(self.cfg.replay_size, exp_cfg.seed)
        self.goal_memory = ReplayMemory(self.cfg.replay_size, exp_cfg.seed)


        self.total_numsteps = 0
        self.num_constraint_violations = 0
        self.num_goal_reached = 0
        self.num_unsafe_transitions = 0
        self.last_actions = None

    def experiment_setup(self):
        agent = self.agent_setup()
        self.forward_agent = agent 

    def agent_setup(self):
        if self.exp_cfg.vec_env:
            obs_space = self.envs.observation_space
            act_space = self.envs.action_space
        else:
            obs_space = self.envs[0].observation_space
            act_space = self.envs[0].action_space
        agent = BC(obs_space,
            act_space,
            self.exp_cfg,
            self.logdir)
        return agent

    def pretrain_critic_safe(self, constraint_demo_data):
        # Get data for safety critic training
        self.num_unsafe_transitions = 0
        for transition in constraint_demo_data:
            self.recovery_memory.push(*transition)
            self.num_constraint_violations += int(transition[2])
            self.num_unsafe_transitions += 1
            if self.num_unsafe_transitions == self.cfg.num_unsafe_transitions:
                break
        print("Number of Constraint Transitions: ",
              self.num_unsafe_transitions)
        print("Number of Constraint Violations: ",
              self.num_constraint_violations)
        batch_size = self.cfg.batch_size
        if self.cfg.pos_fraction > 0:
            batch_size = min(self.cfg.batch_size, int(self.num_constraint_violations / self.cfg.pos_fraction))
        for i in range(self.cfg.critic_safe_pretraining_steps):
            if i % 100 == 0:
                print("Safety Critic Update Step: ", i)
            self.forward_agent.safety_critic.update_parameters(
                memory=self.recovery_memory,
                agent=self.forward_agent,
                batch_size=min(batch_size,
                               len(constraint_demo_data)))
    
    def pretrain_critic_goal(self, data):
        # Get data for goal critic training
        self.num_unsafe_transitions = 0
        for transition in data:
            self.goal_memory.push(*transition)
            self.num_goal_reached += int(transition[2])
        batch_size = self.cfg.batch_size
        if self.cfg.pos_fraction > 0:
            batch_size = min(self.cfg.batch_size, int(self.num_goal_reached / self.cfg.pos_fraction))
        for i in range(self.cfg.critic_safe_pretraining_steps):
            if i % 100 == 0:
                print("Goal Critic Update Step: ", i)
            self.forward_agent.goal_critic.update_parameters(
                memory=self.goal_memory,
                agent=self.forward_agent,
                batch_size=min(batch_size, len(data)))
        
    def pretrain_with_task_data(self, task_demo_data):
        self.num_task_transitions = 0
        for transition in task_demo_data:
            self.human_memory.push(*transition)
            self.num_task_transitions += 1
            if self.num_task_transitions == self.cfg.num_task_transitions:
                break
        for i in range(self.cfg.num_policies):
            for _ in range(self.human_memory.size):
                elem = self.human_memory.buffer[np.random.randint(self.human_memory.size)]
                self.ensemble_memories[i].push(elem[0].copy(), elem[1].copy(), elem[2], elem[3].copy(), elem[4])
        print("Number of Task Transitions: ", self.num_task_transitions)

        # Pretrain BC policy
        print("Pretraining BC!")
        for i in range(self.cfg.policy_pretraining_steps):
            if self.exp_cfg.discrete:
                loss = self.forward_agent.train_discrete(
                    memory=self.human_memory,
                    batch_size=min(self.cfg.batch_size, self.human_memory.size)
                )
            else:
                loss = self.forward_agent.train(
                    memory=self.ensemble_memories,
                    batch_size=min(self.cfg.batch_size, self.human_memory.size)
                )
            if i % 100 == 0:
                print("Loss: ", loss.item())
    
    def add_transitions(self, transitions):
        def add_transition(memory, state, action, reward, next_state, mask):
            memory.push(state, action, reward, next_state, mask)
        for t in transitions:
            if t is not None:
                state, action, reward, next_state, done, info = t
                mask = float(not done)
                if self.cfg.safety_critic:
                    add_transition(self.recovery_memory, state, action, info['constraint'], next_state, mask)
                if self.cfg.goal_critic:
                    add_transition(self.goal_memory, state, action, info['success'], next_state, mask)
                if info['human']:
                    add_transition(self.human_memory, state, action, reward, next_state, mask)
                    for i in range(self.cfg.num_policies):
                        elem = self.human_memory.buffer[np.random.randint(self.human_memory.size)]
                        add_transition(self.ensemble_memories[i], elem[0].copy(), elem[1].copy(), elem[2], elem[3].copy(), elem[4])
                if info['constraint']:
                    self.num_constraint_violations += 1
                if info['success']:
                    self.num_goal_reached += 1

    def train(self, t): 
        if len(self.human_memory) > self.cfg.batch_size:
            # Number of updates per step in environment
            for i in range(self.cfg.updates_per_step):
                if len(self.human_memory) > self.cfg.batch_size:
                    if self.exp_cfg.discrete:
                        self.forward_agent.train_discrete(
                            memory=self.human_memory,
                            batch_size=self.cfg.batch_size
                        )
                    else:
                        self.forward_agent.train(
                            memory=self.ensemble_memories,
                            batch_size=self.cfg.batch_size
                        )
                if self.cfg.safety_critic and not self.cfg.disable_online_updates and len(
                        self.recovery_memory) > self.cfg.batch_size \
                        and self.num_constraint_violations / self.cfg.batch_size > self.cfg.pos_fraction:
                    self.forward_agent.safety_critic.update_parameters(
                        memory=self.recovery_memory,
                        agent=self.forward_agent,
                        batch_size=self.cfg.batch_size)
                
                if self.cfg.goal_critic and not self.cfg.disable_online_updates and len(
                        self.goal_memory) > self.cfg.batch_size \
                        and self.num_goal_reached / self.cfg.batch_size > self.cfg.pos_fraction:
                    self.forward_agent.goal_critic.update_parameters(
                        memory=self.goal_memory,
                        agent=self.forward_agent,
                        batch_size=self.cfg.batch_size)

    def get_actions(self, states, t):
        self.last_actions = self.forward_agent.get_actions(states)
        return self.last_actions

    def save(self):
        self.forward_agent.save()

    def load(self, logdir):
        self.forward_agent.load(logdir)

    def get_allocation_metrics(self, states, t):
        actions = self.last_actions
        if self.exp_cfg.vec_env:
            constraint_violation = self.envs.constraint_buf.cpu().numpy()
        else:
            constraint_violation = [env.constraint for env in self.envs]
        uncertainty = self.forward_agent.get_policy_uncertainty(states)

        metrics = {'constraint_violation': constraint_violation, 'uncertainty': uncertainty}

        if self.cfg.goal_critic:
            td_prob = self.forward_agent.goal_critic.get_value(
                torchify(states, self.device), 
                torchify(actions, self.device)).cpu().numpy()
            td_risk = 1 - td_prob
            metrics['td_risk'] = td_risk
        if self.cfg.safety_critic:
            risk = self.forward_agent.safety_critic.get_value(
                torchify(states, self.device), 
                torchify(actions, self.device)).cpu().numpy()
            metrics['risk'] = risk
        
        return metrics
