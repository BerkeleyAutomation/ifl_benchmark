"""
Main program that orchestrates the parallel experiment
"""
import datetime
import os
import os.path as osp
import pickle
import numpy as np
from isaacgym import gymutil, gymtorch, gymapi
import torch
from env.make_utils import register_env, make_env, make_ig_env
from plotting.process_logs import compute_stats

TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

def torchify(x): return torch.FloatTensor(x).to('cuda')

class ParallelExperiment:
    def __init__(self, exp_cfg, agent, supervisor, allocation):
        self.exp_cfg = exp_cfg
        # Logging setup
        self.logdir = os.path.join(
            self.exp_cfg.logdir, '{}_{}_{}'.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                self.exp_cfg.env_name,
                self.exp_cfg.logdir_suffix))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        print("LOGDIR: ", self.logdir)
        pickle.dump(self.exp_cfg,
                    open(os.path.join(self.logdir, "args.pkl"), "wb"))
        self.log_freq = self.exp_cfg.log_freq
        self.vec_env = self.exp_cfg.vec_env

        # Experiment setup
        self.experiment_setup(supervisor)
        self.agent = agent(self.envs, self.exp_cfg, self.logdir)
        self.allocation = allocation(self.exp_cfg)
        # Initially assign humans to the first num_humans envs
        assert self.exp_cfg.num_humans <= self.exp_cfg.num_envs
        # Every env starts in robot control
        self.assignments = np.zeros((self.exp_cfg.num_envs, self.exp_cfg.num_humans))

    def experiment_setup(self, supervisor):
        torch.manual_seed(self.exp_cfg.seed)
        np.random.seed(self.exp_cfg.seed)
        if self.vec_env:
            self.envs = make_ig_env(self.exp_cfg.isaacgym_cfg)
            self.state = self.envs.obs_buf.clone()
            assert self.state.shape == (self.exp_cfg.num_envs, self.envs.num_obs)
        else:
            register_env(self.exp_cfg.env_name)
            envs = [make_env(self.exp_cfg.env_name, idx=i) for i in range(self.exp_cfg.num_envs)]
            self.envs = envs 
            self.state = np.zeros((self.exp_cfg.num_envs, *self.envs[0].observation_space.shape), dtype=np.float32)
            for i, env in enumerate(envs):
                env.seed(self.exp_cfg.seed+i)
                env.action_space.seed(self.exp_cfg.seed+i)
        self.supervisor = supervisor(self.envs, self.exp_cfg)
        self.raw_data = [] # add all relevant data for each timestep

    def assign_humans(self, env_priorities):
        '''
        Fleet-DAgger assignment given a prioritization
        '''
        # Make sure there is at most 1 human each robot is assigned to
        assert 0 <= self.assignments.sum(1).max() <= 1
        
        # Humans that won't be reallocated this timestep. This includes humans that still have time before they can switch
        # and humans that are already assigned to high priority robots.
        humans_to_keep = set()

        # Find number of humans that could possibly be reassigned
        max_humans_to_assign = 0
        for i in range(self.exp_cfg.num_humans):
            # Reassign human if they are not yet assigned or
            if not np.sum(self.assignments[:, i]):
                max_humans_to_assign += 1
            # their time is up and they are not resetting a hard failure
            elif self.human_timers[i] >= self.exp_cfg.min_int_time and not self.blocked_envs[np.argmax(self.assignments[:,i])]:
                max_humans_to_assign += 1
            # otherwise keep them where they are
            else:
                humans_to_keep.add(i)

        assignment_count = 0 # next worst robot index
        human_idx = 0 # number of reassignable humans accounted for
        # Use env priorities to identify which humans should not be reassigned
        while human_idx < max_humans_to_assign and assignment_count < len(env_priorities):
            # check the next worst robot.
            env_assign_idx = env_priorities[assignment_count]
            # If someone is already assigned to this env...
            if self.assignments[env_assign_idx].sum():
                # find out who
                assigned_human = np.argmax(self.assignments[env_assign_idx])
                assignment_count += 1
                # This high priority robot already has a human assigned to it, so we will not reallocate assigned_human
                humans_to_keep.add(assigned_human)
                # If assigned_human is about to time out though, we will allocate a human to it, it'll just be the same
                # human since no need to swap in a new human in this high priority env that still needs help
                if self.human_timers[assigned_human] >= self.exp_cfg.min_int_time and not self.blocked_envs[np.argmax(self.assignments[:,i])]:
                    human_idx += 1
            else:
                # If no human assigned, we will be assigning a human, so move to next human.
                human_idx += 1

        # Actually assign humans
        assignment_count = 0
        human_idx = 0
        while human_idx < self.exp_cfg.num_humans:
            # If human is assigned but its allocation won't change, move onto next human.
            if human_idx in humans_to_keep:
                human_idx += 1
            # free humans if possible
            elif assignment_count >= min(self.exp_cfg.num_envs, len(env_priorities)):
                if self.assignments[:, human_idx].sum():
                    current_env = np.argmax(self.assignments[:, human_idx])
                    self.assignments[current_env][human_idx] = 0
                self.human_timers[human_idx] = 0
                human_idx += 1
            # Check if the next worst robot already has a human assigned.
            elif self.assignments[env_priorities[assignment_count]].sum():
                # If so, skip this robot.
                assignment_count += 1
            # If we have a human we can re-assign and a robot that doesn't already have a human, re-assign them
            else:
                if self.assignments[:, human_idx].sum():
                    current_env = np.argmax(self.assignments[:, human_idx])
                    self.assignments[current_env][human_idx] = 0
                new_env = env_priorities[assignment_count]
                self.assignments[new_env][human_idx] = 1
                self.human_timers[human_idx] = 0
                assignment_count += 1
                human_idx += 1

    def vec_step(self, action_list, allocation_metrics):
        """
        A vectorized version of step() for environments with a parallel step function
        """
        ret = []

        # Get env priorities
        env_priorities = self.allocation.allocate(allocation_metrics)
        # Assign humans based on env_priorities
        self.assign_humans(env_priorities)
        # Log all possibly relevant data
        # (human timers / episode steps values here are BEFORE stepping the env)
        step_data = {'state': np.copy(self.state.cpu().numpy()), 'action_list': np.copy(action_list), 'env_priorities': np.copy(env_priorities),
            'assignments': np.copy(self.assignments), 'human_timers': np.copy(self.human_timers), 'episode_steps': self.episode_steps,
            'constraint': [], 'idle': []}

        to_reset = set()
        if self.exp_cfg.supervisor_cfg.prefetch:
            self.supervisor.prefetch_actions(self.state)
        use_human_acs = [False] * self.exp_cfg.num_envs
        constraints = self.envs.constraint_buf.cpu().numpy()
        actions = np.array(action_list).copy()

        # Plan and execute actions in each env, with human interventions as necessary
        for env_idx in range(self.exp_cfg.num_envs):
            step_data['constraint'].append(constraints[env_idx])
            step_data['idle'].append(0)

            # Make sure there is at most 1 robot each human is assigned to
            assert self.exp_cfg.num_humans == 0 or 0 <= self.assignments.sum(0).max() <= 1

            # Check if a human is assigned to this environment
            use_human_ac = np.sum(self.assignments[env_idx]) == 1
            use_human_acs[env_idx] = use_human_ac
            if use_human_ac:
                # Get assigned human id
                human_idx = np.argmax(self.assignments[env_idx])
                self.human_timers[human_idx] += 1
                # Get human action to override robot policy
                # If environment is in a constraint violating state, get human to reset it
                if constraints[env_idx] and not self.blocked_envs[env_idx]:
                    # env is blocked
                    self.blocked_envs[env_idx] = 1
                    self.blocked_env_timers[env_idx] = self.exp_cfg.hard_reset_time
                    # Idles at this step
                    step_data['idle'][-1] = 1

                if constraints[env_idx]:
                    if self.blocked_env_timers[env_idx] == 0:
                        to_reset.add(env_idx)
                        self.reset_counts[env_idx] += 1
                        # env is no longer blocked because we reset it
                        self.blocked_envs[env_idx] = 0
                        self.episode_steps[env_idx] = 0
                    else:
                        self.blocked_env_timers[env_idx] -= 1
                        step_data['idle'][-1] = 1
                    ret.append(None)
                    continue # Once we reset, nothing more to log/do for this env
                # Otherwise get human action and override policy action
                else:
                    human_action = self.supervisor.get_action(self.state[env_idx], env_idx)
                    if human_action == 'reset': # reset supervisor
                        to_reset.add(env_idx)
                        ret.append(None)
                        continue
                    else:
                        actions[env_idx] = np.copy(human_action)
            elif constraints[env_idx]:
                # If the env is in constraint violating state, but no human available to help us...
                # env is blocked
                self.blocked_envs[env_idx] = 1
                self.blocked_env_timers[env_idx] = self.exp_cfg.hard_reset_time
                # Idles at this step
                step_data['idle'][-1] = 1
                ret.append(None)
                actions[env_idx] = np.zeros(actions.shape[1]) # execute a no-op
                continue # If idle, nothing more to log/do for this env
            ret.append(1) # if not None, this will be overwritten with the actual ret.

        # Actually execute actions in the environment
        if self.exp_cfg.noise:
            actions += np.random.normal(scale=self.exp_cfg.noise, size=actions.shape).astype(actions.dtype) # noise injection
        next_states, rewards, dones, _ = self.envs.step(torch.tensor(actions, device=self.envs.rl_device))
        next_states = next_states['obs']
        constraints = self.envs.constraint_buf.cpu().numpy()

        assert len(ret) == self.exp_cfg.num_envs
        for env_idx in range(self.exp_cfg.num_envs):
            if ret[env_idx] == 1:
                info = dict()
                info['env'] = env_idx
                info['human'] = use_human_acs[env_idx]
                info['constraint'] = constraints[env_idx]
                self.episode_steps[env_idx] += 1
                info['success'] = self.envs.success_buf[env_idx].cpu().numpy()
                ret[env_idx] = (np.copy(self.state[env_idx].cpu().numpy()), actions[env_idx],
                    rewards[env_idx].cpu().numpy(), next_states[env_idx].cpu().numpy(), dones[env_idx].cpu().numpy(), info)
            # if we hit a successful sink state, we are allowed a free reset or time out
            if dones[env_idx] and not constraints[env_idx]:
                to_reset.add(env_idx)
                self.reset_counts[env_idx] += 1
                self.episode_steps[env_idx] = 0
        # reset all the envs that need resetting and update state
        self.envs.reset_idx(torch.tensor(list(to_reset), device=self.envs.device, dtype=torch.long))
        self.state = self.envs.obs_buf.clone()

        # if real_act is nonzero and doesn't match the real_action in action_list, it's the human action
        step_data['real_act'] = [r[1] if r else None for r in ret]
        step_data['reward'] = [r[2] if r else None for r in ret]
        step_data['done'] = [r[4] if r else None for r in ret]
        step_data['info'] = [r[5] if r else None for r in ret]
        self.raw_data.append(step_data)
        return ret

    def step(self, action_list, allocation_metrics):
        """
        A function that synchronously steps all parallel environments at the
        same time.
        """
        ret = []

        # Get env priorities
        env_priorities = self.allocation.allocate(allocation_metrics)
        # Assign humans based on env_priorities
        self.assign_humans(env_priorities)
        # Log all possibly relevant data
        # (human timers / episode steps values here are BEFORE stepping the env)
        step_data = {'state': np.copy(self.state), 'action_list': np.copy(action_list), 'env_priorities': np.copy(env_priorities),
            'assignments': np.copy(self.assignments), 'human_timers': np.copy(self.human_timers), 'episode_steps': self.episode_steps,
            'constraint': [], 'idle': []}

        # Plan and execute actions in each env, with human interventions as necessary
        for env_idx in range(self.exp_cfg.num_envs):
            env = self.envs[env_idx]
            action = action_list[env_idx]
            step_data['constraint'].append(env.constraint)
            step_data['idle'].append(0)

            # Make sure there is at most 1 robot each human is assigned to
            assert self.exp_cfg.num_humans == 0 or 0 <= self.assignments.sum(0).max() <= 1

            # Check if a human is assigned to this environment
            use_human_ac = np.sum(self.assignments[env_idx]) == 1
            if use_human_ac:
                # Get assigned human id
                human_idx = np.argmax(self.assignments[env_idx])
                self.human_timers[human_idx] += 1
                # Get human action to override robot policy
                # If environment is in a constraint violating state, get human to reset it
                if env.constraint and not self.blocked_envs[env_idx]:
                    # env is blocked
                    self.blocked_envs[env_idx] = 1
                    self.blocked_env_timers[env_idx] = self.exp_cfg.hard_reset_time
                    # Idles at this step
                    step_data['idle'][-1] = 1

                if env.constraint:
                    if self.blocked_env_timers[env_idx] == 0:
                        self.state[env_idx] = env.reset(hard=True)
                        self.reset_counts[env_idx] += 1
                        # env is no longer blocked because we reset it
                        self.blocked_envs[env_idx] = 0
                        self.episode_steps[env_idx] = 0
                    else:
                        self.blocked_env_timers[env_idx] -= 1
                        step_data['idle'][-1] = 1
                    ret.append(None)
                    continue # Once we reset, nothing more to log/do for this env
                # Otherwise get human action and override policy action
                else:
                    action = self.supervisor.get_action(self.state[env_idx], env_idx)
            else:
                # If the env is in constraint violating state, but no human available to help us...
                if env.constraint:   
                    # env is blocked
                    self.blocked_envs[env_idx] = 1
                    self.blocked_env_timers[env_idx] = self.exp_cfg.hard_reset_time
                    # Idles at this step
                    step_data['idle'][-1] = 1
                    ret.append(None)
                    continue # If idle, nothing more to log/do for this env

            # Actually execute action in the environment
            if action == "reset":
                ret.append(None)
                done = True
            else:
                if self.exp_cfg.noise:
                    action += np.random.normal(scale=self.exp_cfg.noise, size=action.shape)
                next_state, reward, done, info = env.step(action)

                info['env'] = env_idx
                info['human'] = use_human_ac

                ret.append((np.copy(self.state[env_idx]), action, reward, next_state, done, info)) 
            self.episode_steps[env_idx] += 1

            # if we hit a successful sink state, we are allowed a free reset or time out
            if (done or self.episode_steps[env_idx] >= env.max_episode_steps) and not env.constraint:
                if self.episode_steps[env_idx] + 1 < env.max_episode_steps: 
                    print("REACHED GOAL!!!")
                next_state = env.reset()
                self.reset_counts[env_idx] += 1
                self.episode_steps[env_idx] = 0
            self.state[env_idx] = next_state

        # if real_act doesn't match the real_action in action_list, it's the human action
        step_data['real_act'] = [r[1] if r else None for r in ret]
        step_data['reward'] = [r[2] if r else None for r in ret]
        step_data['done'] = [r[4] if r else None for r in ret]
        step_data['info'] = [r[5] if r else None for r in ret]
        self.raw_data.append(step_data)
        return ret

    def reset_all(self):
        self.raw_data = []
        self.human_timers = [0 for _ in range(self.exp_cfg.num_humans)]
        self.blocked_envs = [0 for _ in range(self.exp_cfg.num_envs)]
        self.blocked_env_timers = [0 for _ in range(self.exp_cfg.num_envs)] # number of timesteps remaining for hard resets to complete
        self.t = 0
        self.reset_time_idx = -1
        if self.vec_env:
            self.envs.reset_idx(torch.arange(self.exp_cfg.num_envs, device=self.envs.device))
            self.envs.reset()
            self.state = self.envs.obs_buf.clone()
        else:
            for i, env in enumerate(self.envs):
                self.state[i] = self.envs[i].reset()
        self.episode_steps = [0] * self.exp_cfg.num_envs
        self.reset_counts = [0] * self.exp_cfg.num_envs

    def get_demos(self, suffix='task', suffix2=''):
        # Get offline task demo data, or constraint demo data if constraint=True
        if self.vec_env:
            p = pickle.load(open(
                'env/assets/isaacgym/demos/{}/{}{}.pkl'.format(suffix, self.exp_cfg.env_name, suffix2), 'rb'))
            return [(p['obs'][i], p['act'][i], p['rew'][i], p['obs2'][i], 1 - p['done'][i]) for i in range(p['obs'].shape[0])]
        else:
            constraint = (suffix == 'constraint')
            return self.envs[0].get_offline_data(
                self.exp_cfg.agent_cfg.num_unsafe_transitions if constraint else self.exp_cfg.agent_cfg.num_task_transitions,
                task_demos=not constraint)

    def run(self):
        self.reset_all()
        if 'pretrain_qrisk' in self.exp_cfg.agent_cfg and self.exp_cfg.agent_cfg.pretrain_qrisk:
            if 'safety_critic' in self.exp_cfg.agent_cfg and self.exp_cfg.agent_cfg.safety_critic:
                constraint_demo_data = self.get_demos(suffix='constraint')
                self.agent.pretrain_critic_safe(constraint_demo_data)
            if 'goal_critic' in self.exp_cfg.agent_cfg and self.exp_cfg.agent_cfg.goal_critic:
                goal_demo_data = self.get_demos(suffix='goal')
                self.agent.pretrain_critic_goal(goal_demo_data)
        if 'task_demos' in self.exp_cfg.agent_cfg and self.exp_cfg.agent_cfg.task_demos:
            task_demo_data = self.get_demos(suffix='task', suffix2='BC' if self.exp_cfg.agent_cfg.bc else '')
            self.agent.pretrain_with_task_data(task_demo_data)

        if self.exp_cfg.resume:
            self.agent.load(self.exp_cfg.resume)

        while self.t < self.exp_cfg.num_steps:
            actions = self.agent.get_actions(self.state, self.t)
            allocation_metrics = self.agent.get_allocation_metrics(self.state, self.t)
            # add assignment info to allocation metrics
            allocation_metrics['assignments'] = self.assignments.copy() 
            allocation_metrics['human_timers'] = self.human_timers.copy()
            allocation_metrics['blocked_envs'] = self.blocked_envs.copy()
            allocation_metrics['time'] = self.t
            if self.vec_env:
                ret = self.vec_step(actions, allocation_metrics)
            else:
                ret = self.step(actions, allocation_metrics)

            self.agent.add_transitions(ret)
            self.agent.train(self.t)
            if self.t % self.exp_cfg.log_freq == 0:
                self.dump_logs()
                self.agent.save()
                print("LOGDIR: ", self.logdir)
            self.t += 1

    def dump_logs(self):
        with open(osp.join(self.logdir, "raw_data.pkl"), "wb") as f:
            pickle.dump(self.raw_data, f)
        self.agent.save()
        max_steps = self.envs.max_episode_steps if self.vec_env else self.envs[0].max_episode_steps
        t, rew, succ, viol, switch, human, idle = compute_stats(self.logdir) # this will also save a run_stats.pkl
        print("Steps: %d AvgReward: %f Successes: %d Violations: %d Switches: %d Human Acts: %d Idle Time: %d, Num Blocked: %d"%(
            t, rew*max_steps/t/self.exp_cfg.num_envs, succ, viol, switch, human, idle, self.num_blocked))

    @property
    def num_blocked(self):
        return sum(self.blocked_envs)

