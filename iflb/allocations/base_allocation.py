
class Allocation:
    """
    An abstract class containing an API for all allocation strategies to implement.
    """

    def __init__(self, exp_cfg):
        self.exp_cfg = exp_cfg
        self.cfg = exp_cfg.allocation_cfg

    def allocate(self, allocation_metrics):
        """
        return a list of integers (robot env indices) sorted by priority.
        can be shorter than num_envs if exp_cfg.free_humans is set.

        allocation_metrics: a dictionary from strings to lists of metrics ordered by robot env idx
        """
        raise NotImplementedError

