"""
A wrapper for envs that define the supervisor functions themselves
"""
from .base_supervisor import ParallelSupervisor 

class AnalyticSupervisor(ParallelSupervisor):
    def __init__(self, envs, cfg):
        self.supervisor_fns = [env.human_action for env in envs]

    def get_action(self, state, env_idx=0):
        return self.supervisor_fns[env_idx](state)


