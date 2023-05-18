from .base_allocation import Allocation
import numpy as np

class RandomAllocation(Allocation):
    """
    A simple random allocation strategy.
    """
    def allocate(self, allocation_metrics):
        num_envs = self.exp_cfg.num_envs
        env_priorities = list(np.random.permutation(num_envs))
        if not self.cfg.free_humans:
            return env_priorities
        # below handles free_humans=True
        num_humans = self.exp_cfg.num_humans
        num_steps = self.exp_cfg.num_steps
        assignments = allocation_metrics['assignments']
        human_timers = allocation_metrics['human_timers']
        blocked_envs = allocation_metrics['blocked_envs']
        # number of busy humans in expectation
        mean_busy_humans = self.cfg.action_budget / num_steps 
        # gaussian noise
        busy_humans = round(np.random.normal(mean_busy_humans, scale=(self.cfg.std_dev * num_humans)))
        busy_humans = min(max(0, busy_humans), num_humans)
        # keep the robots currently being helped as high priority
        j = 0
        for i in range(num_humans):
            if (human_timers[i] > 0 and human_timers[i] < self.exp_cfg.min_int_time) \
                or (np.max(assignments[:,i]) > 0 and blocked_envs[np.argmax(assignments[:,i])]):
                env_idx = np.argmax(assignments[:,i])
                env_priorities.remove(env_idx)
                env_priorities.insert(0, env_idx)
                j += 1
        first_priorities = env_priorities[:j].copy()
        # prioritize constraints if the flag is set (after currently helped robots)
        index = j
        busy_humans = max(j, busy_humans)
        if self.cfg.constraints:
            for i in range(self.exp_cfg.num_envs):
                if i in first_priorities:
                    continue
                if index >= busy_humans:
                    break
                if allocation_metrics["constraint_violation"][i]:
                    env_priorities.remove(i)
                    env_priorities.insert(index, i)
                    index += 1
        return env_priorities[:busy_humans]