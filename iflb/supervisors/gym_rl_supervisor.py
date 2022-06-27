"""
Supervisor for Isaac Gym environments that does a forward pass on a trained RL agent. 
Assumes the agent is trained with IsaacGymEnvs (https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
"""
import numpy as np
from .base_supervisor import ParallelSupervisor
from rl_games.torch_runner import Runner
from gym import spaces

class IsaacGymRLSupervisor(ParallelSupervisor):
    def __init__(self, envs, cfg):
        assert cfg.vec_env, "Must be Isaac Gym env to use this supervisor"
        icfg = cfg.isaacgym_cfg
        num_envs = icfg['task']['env']['numEnvs']
        obs_dim = icfg['task']['env']['numObservations']
        act_dim = icfg['task']['env']['numActions']
        icfg = icfg.copy()
        icfg['train']['params']['config']['env_info'] = {
                'action_space': spaces.Box(np.ones(act_dim) * -1., np.ones(act_dim) * 1.),
                'observation_space': spaces.Box(np.ones(obs_dim) * -np.Inf, np.ones(obs_dim) * np.Inf),
                'agents': num_envs,
                'batch': True
            }
        rlg_config_dict = icfg['train']
        # setup RL games runner
        runner = Runner()
        runner.load(rlg_config_dict)
        runner.reset()
        player = runner.create_player()
        player.restore(icfg['checkpoint'])
        player.has_batch_dimension = True
        self.player = player
        self.actions = None
        self.prefetch = cfg.supervisor_cfg.prefetch
        self.prefetched = False

    def prefetch_actions(self, states):
        """
        To optimize efficiency, this executes the NN forward pass for all states at once.
        Should be called once per timestep. 
        """
        self.actions = self.player.get_action(states, True)
        self.prefetched = True

    def get_action(self, state, env_idx=None):
        if self.prefetch:
            assert self.prefetched
            return self.actions[env_idx].cpu()
        else:
            self.player.has_batch_dimension = False
            return self.player.get_action(state, True).cpu()

