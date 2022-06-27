"""
A simple supervisor that free resets the environment
"""
from .base_supervisor import ParallelSupervisor

class ResetSupervisor(ParallelSupervisor):
    def __init__(self, envs, cfg):
        pass

    def get_action(self, state, env_idx=0):
        # actual reset calls are handled by parallel_experiment
        return "reset"


