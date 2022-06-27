# Isaac Gym + Interactive Fleet Learning Instructions

This folder has all IsaacGymEnvs environments, but the ones currently fully supported are: Humanoid, Anymal, AllegroHand

To add a new environment from this folder (or a new custom Isaac Gym environment), do the following:

1. Check the asset root in `_create_envs()` in the `*.py` file for the desired environment and `*.yaml` config file (in `config/isaacgym_cfg/task`) to make sure assets are loaded correctly; they should be loading from `../assets/isaacgym`.
2. Add the following to the beginning of the `reset_idx()` function: `if len(env_ids) == 0: return`. Comment out the `self.reset_idx()` call in `post_physics_step()`. This is because the `parallel_experiment.py` code will do the resetting instead of the env stepping to properly handle hard resets.
3. Train an RL supervisor for the task by following the `README.md` in the [NVIDIA IsaacGymEnvs repo](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), then copy the `*.pth` model weights to `env/assets/isaacgym/supervisors`.
4. Update `self.constraint_buf` (indicators for constraint violation) and `self.success_buf` (indicators for task success) appropriately in the environment. This is the trickiest part as it is a design decision and unique to each environment depending on how the reward is computed; see the supported environments for example code.
5. Optionally add some demonstration data (5 demos or so) for an initial robot policy as a dictionary in a `*.pkl` file by running `python -m main @scripts/args_isaacgym_demos.txt --env_name [ENV]`, running `python scripts/extract_demos.py [LOGFILE]` on the generated log data, and copying it to `env/assets/isaacgym/demos/task`.
6. Optionally add offline constraint violation data similarly by running `python -m main @scripts/args_isaacgym_constraints.txt --env_name [ENV]`, running `python scripts/extract_constraints.py [LOGFILE]` on the generated log data, and copying it to `env/assets/isaacgym/demos/constraint`.
7. Finally, when running the main program, specify the environment with the `--env_name` CLI argument (name should match the one in the `isaacgym_task_map` in `__init__.py`) and make sure `--vecenv` is set. See `scripts/args_isaacgym.txt` for example CLI arguments. Double check the configuration files in `config/isaacgym_cfg` and adjust as desired.
