from arg_utils import get_parser
import os.path as osp
from iflb.parallel_experiment import ParallelExperiment
from iflb.agents import *
from iflb.supervisors import *
from iflb.allocations import *
from dotmap import DotMap
import yaml

def setup_isaac_gym(exp_cfg):
    # isaac gym config loading
    from omegaconf import DictConfig, OmegaConf
    from hydra import compose, initialize
    def omegaconf_to_dict(d):
        ret = {}
        for k, v in d.items():
            if isinstance(v, DictConfig):
                ret[k] = omegaconf_to_dict(v)
            else:
                ret[k] = v
        return ret
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)
    initialize(config_path="config/isaacgym_cfg")
    cfg = compose(config_name="config", overrides=["task={}".format(exp_cfg.env_name)])
    cfg_dict = omegaconf_to_dict(cfg)
    # overwrite numEnvs and render from CLI args
    cfg_dict['headless'] = not exp_cfg.render
    cfg_dict['task']['env']['numEnvs'] = exp_cfg.num_envs
    # overwrite training params
    cfg_dict['train']['params']['config']['num_actors'] = exp_cfg.num_envs
    dirname = osp.dirname(osp.abspath(__file__))
    checkpoint = osp.join(dirname, 'env/assets/isaacgym/supervisors/{}.pth'.format(exp_cfg.env_name))
    cfg_dict['checkpoint'] = checkpoint
    cfg_dict['train']['params']['load_path'] = checkpoint
    return cfg_dict

if __name__ == '__main__':
    # Get user arguments and construct config
    parser = get_parser()
    exp_cfg, _ = parser.parse_known_args()

    # Create experiment and run it
    # load agent
    agent = agent_map[exp_cfg.agent]
    dirname = osp.dirname(osp.abspath(__file__))
    filepath = osp.join(dirname, 'iflb/agents/cfg/{}'.format(agent_cfg_map.get(exp_cfg.agent, 'base_agent.yaml')))
    with open(filepath, "r") as fh:
        agent_cfg = yaml.safe_load(fh)
    for key in agent_cfg:
        if type(agent_cfg[key]) == bool:
            parser.add_argument('--{}'.format(key), action='store_true', default=agent_cfg[key])
            parser.add_argument('--no_{}'.format(key), action='store_false', dest='{}'.format(key))
        else:
            parser.add_argument('--{}'.format(key), type=type(agent_cfg[key]), default=agent_cfg[key])
    
    # load supervisor
    supervisor = supervisor_map[exp_cfg.supervisor]
    filepath = osp.join(dirname, 'iflb/supervisors/cfg/{}'.format(supervisor_cfg_map.get(exp_cfg.supervisor, 'base_supervisor.yaml')))
    with open(filepath, "r") as fh:
        supervisor_cfg = yaml.safe_load(fh)
    for key in supervisor_cfg:
        if type(supervisor_cfg[key]) == bool:
            parser.add_argument('--{}'.format(key), action='store_true', default=supervisor_cfg[key])
            parser.add_argument('--no_{}'.format(key), action='store_false', dest='{}'.format(key))
        else:
            parser.add_argument('--{}'.format(key), type=type(supervisor_cfg[key]), default=supervisor_cfg[key])
    
    # load allocation
    allocation = allocation_map[exp_cfg.allocation]
    filepath = osp.join(dirname, 'iflb/allocations/cfg/{}'.format(allocation_cfg_map.get(exp_cfg.allocation, 'base_allocation.yaml')))
    with open(filepath, "r") as fh:
        allocation_cfg = yaml.safe_load(fh)
    for key in allocation_cfg:
        if type(allocation_cfg[key]) == bool:
            parser.add_argument('--{}'.format(key), action='store_true', default=allocation_cfg[key])
            parser.add_argument('--no_{}'.format(key), action='store_false', dest='{}'.format(key))
        else:
            parser.add_argument('--{}'.format(key), type=type(allocation_cfg[key]), default=allocation_cfg[key])

    exp_cfg = vars(parser.parse_known_args()[0]) # get CLI args
    exp_cfg = DotMap(exp_cfg)
    exp_cfg.agent_cfg = DotMap()
    exp_cfg.supervisor_cfg = DotMap()
    exp_cfg.allocation_cfg = DotMap()

    # **NOTE**: Assumes that the keys don't overlap among cfg files!
    for key in agent_cfg:
        exp_cfg.agent_cfg[key] = exp_cfg[key]
        del exp_cfg[key]
    for key in supervisor_cfg:
        exp_cfg.supervisor_cfg[key] = exp_cfg[key]
        del exp_cfg[key]
    for key in allocation_cfg:
        exp_cfg.allocation_cfg[key] = exp_cfg[key]
        del exp_cfg[key]

    if exp_cfg.vec_env:
        # isaac gym conf loading
        ig_cfg = setup_isaac_gym(exp_cfg)
        exp_cfg.isaacgym_cfg = DotMap(ig_cfg)

    print("Experiment config:", exp_cfg)
    experiment = ParallelExperiment(exp_cfg, agent, supervisor, allocation)
    experiment.run()
