
class ParallelSupervisor:
    """
    An abstract class containing an API for all parallel supervisors to implement.
    """

    def __init__(self, envs, cfg):
        raise NotImplementedError

    def get_action(self, state):
        """
        return the supervisor action for an individual agent state
        """
        raise NotImplementedError


