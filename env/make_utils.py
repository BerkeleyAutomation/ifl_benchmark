import gym
from gym.envs.registration import register
from env.isaacgym import isaacgym_task_map

# non-isaac-gym serial envs
ENV_ID = {
    'Navigation': 'Navigation-v0'
}

ENV_CLASS = {
    'Navigation': 'Navigation'
}

def register_env(env_name):
    assert env_name in ENV_ID, "unknown environment"
    env_id = ENV_ID[env_name]
    env_class = ENV_CLASS[env_name]
    register(id=env_id, entry_point='env.' + env_name.lower() + ":" + env_class)

def make_env(env_name, idx=0, kwargs=dict()):
    env_id = ENV_ID[env_name]
    return gym.make(env_id, idx=idx, **kwargs)

def make_ig_env(ig_cfg):
    # make an isaacgym env
    return isaacgym_task_map[ig_cfg['task_name']](
        cfg = ig_cfg['task'],
        sim_device = ig_cfg['sim_device'],
        graphics_device_id = ig_cfg['graphics_device_id'],
        headless = ig_cfg['headless']
    )