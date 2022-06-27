from .base_allocation import Allocation
import numpy as np

class CURAllocation(Allocation):
    """
    An allocation strategy that prioritizes robots by (C)onstraint Violations, (U)ncertainty, and (R)isk
    """
    def allocate(self, allocation_metrics):
        if 'C' in self.cfg.order:
            assert "constraint_violation" in allocation_metrics, "Agent {} does not provide the required metrics for {} allocation.".format(self.exp_cfg.agent, self.cfg.order)
        if 'R' in self.cfg.order:
            assert "risk" in allocation_metrics, "Agent {} does not provide the required metrics for {} allocation.".format(self.exp_cfg.agent, self.cfg.order)
        if 'U' in self.cfg.order:
            assert "uncertainty" in allocation_metrics, "Agent {} does not provide the required metrics for {} allocation.".format(self.exp_cfg.agent, self.cfg.order)
            
        
        num_to_free = 0 # number of robots that have zero priority
        free_idx = set()
        priorities = list()
        num_violating = 0

        for i in range(self.exp_cfg.num_envs):
            constraint_violation = allocation_metrics["constraint_violation"][i]
            try:
                risk = allocation_metrics["risk"][i][0]
            except:
                risk = allocation_metrics["risk"][i]
            uncertainty = allocation_metrics["uncertainty"][i]

            if constraint_violation:
                num_violating += 1
                if self.cfg.warmup_penalty > allocation_metrics["time"]:
                    cv = -1 # penalize instead of prioritize constraint violation during warmup
                elif num_violating > int(self.cfg.cv_thresh * self.exp_cfg.num_humans):
                    # treat CVs beyond threshold as non-factors in prioritization
                    cv = 0
                else:
                    cv = 1
            else:
                cv = 0

            if risk < self.cfg.risk_thresh:
                # treat risk below this threshold as 0 risk
                risk = 0

            if uncertainty < self.cfg.uncertainty_thresh:
                # treat uncertainty below this threshold as 0 uncertainty
                uncertainty = 0

            if (cv <= 0 or 'C' not in self.cfg.order) and (risk == 0 or 'R' not in self.cfg.order) and (uncertainty == 0 or 'U' not in self.cfg.order):
                # free human
                if i not in free_idx:
                    num_to_free += 1
                    free_idx.add(i)

            priority = list()
            for elem in self.cfg.order:
                if elem == 'C':
                    priority.append(cv)
                if elem == 'R':
                    priority.append(risk)
                if elem == 'U':
                    priority.append(uncertainty)
            priority.append(np.random.random())
            priority = tuple(priority)
            priorities.append(priority)

        env_priorities = sorted(range(len(priorities)), key = lambda x: priorities[x], reverse=True)
        if self.cfg.free_humans:
            return env_priorities[:self.exp_cfg.num_envs - num_to_free]
        else:
            return env_priorities
