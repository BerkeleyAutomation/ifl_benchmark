from .base_agent import ParallelAgent

class SingleTaskParallelRandomAgent(ParallelAgent):
    """
    A simple agent that samples randomly from each env's action space.
    """

    def __init__(self, envs, exp_cfg=None, logdir=None):
        self.envs = envs

    def add_transitions(self, transitions):
        return

    def pretrain(self):
        return

    def train(self, t):
        return

    def log_stats(self):
        return

    def save(self):
        return

    def get_actions(self, state_arr, t):
        action_list = []
        for env in self.envs:
            action_list.append(env.action_space.sample())
        return action_list
