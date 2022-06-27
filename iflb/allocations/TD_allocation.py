from .base_allocation import Allocation
import numpy as np

class TDAllocation(Allocation):
    """
    Implementation of Fleet-ThriftyDAgger: linear combination of novelty + risk
    """
    def __init__(self, exp_cfg):
        self.exp_cfg = exp_cfg
        self.cfg = exp_cfg.allocation_cfg
        self.risk_data = []
        self.novelty_data = []

    def allocate(self, allocation_metrics):
        alpha = self.cfg.alpha_weight
        priorities = []
        num_to_free = 0 # number of robots that have zero priority
        free_idx = set()
        count_zero = 0
        # running mean/std calculation for normalization
        self.risk_data.extend(allocation_metrics["td_risk"])
        self.novelty_data.extend(allocation_metrics["uncertainty"])
        risk_mean, risk_std = np.mean(np.array(self.risk_data).squeeze()), np.std(np.array(self.risk_data))
        novelty_mean, novelty_std = np.mean(np.array(self.novelty_data)), np.std(np.array(self.novelty_data))

        for i in range(self.exp_cfg.num_envs):
            risk = allocation_metrics["td_risk"][i]
            novelty = allocation_metrics["uncertainty"][i]
            constraint_violation = allocation_metrics["constraint_violation"][i]
            cv = 1 if constraint_violation else 0
            combined = 0

            risk_norm = (risk - risk_mean)/(risk_std + self.cfg.eps)
            novelty_norm = (novelty - novelty_mean)/(novelty_std + self.cfg.eps)
            combined = alpha * risk_norm  + (1 - alpha) * novelty_norm
            if combined <= self.cfg.combined_alpha_thresh:
                combined = 0
                count_zero += 1
            if combined == 0 and cv <= 0:
                # free human
                if i not in free_idx:
                    num_to_free += 1
                    free_idx.add(i)
            combined_rand = tuple((combined, cv, np.random.random()))
            priorities.append(combined_rand)

        env_priorities = sorted(range(len(priorities)), key = lambda x: priorities[x], reverse=True)
        if self.cfg.free_humans:
            return env_priorities[:self.exp_cfg.num_envs - num_to_free]
        else:
            return env_priorities





