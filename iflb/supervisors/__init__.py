from .analytic_supervisor import AnalyticSupervisor
from .reset_supervisor import ResetSupervisor
from .gym_rl_supervisor import IsaacGymRLSupervisor

# Mappings from CLI option strings to supervisors
supervisor_map = {
    "Analytic": AnalyticSupervisor,
    "Reset": ResetSupervisor,
    "GymRL": IsaacGymRLSupervisor
}

supervisor_cfg_map = {
    "GymRL": "gym_rl_supervisor.yaml"
}