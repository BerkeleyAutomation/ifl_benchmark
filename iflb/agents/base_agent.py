
class ParallelAgent:
    """
    An abstract class containing an API for all parallel agents to implement.
    """

    def __init__(self, envs, exp_cfg, logdir):
        raise NotImplementedError

    def add_transitions(self, transitions):
        raise NotImplementedError

    def train(self, t):
        raise NotImplementedError

    def get_actions(self, states, t):
        raise NotImplementedError

    def get_allocation_metrics(self, states, t): 
        # should be called after get_actions() in a given timestep
        raise NotImplementedError

    def save(self):
        # save model weights or other key info
        raise NotImplementedError
