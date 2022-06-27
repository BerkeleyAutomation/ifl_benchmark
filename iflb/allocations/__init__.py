from .random_allocation import RandomAllocation
from .CUR_allocation import CURAllocation
from .TD_allocation import TDAllocation


# Mappings from CLI option strings to allocation strategies
allocation_map = {
    "random": RandomAllocation,
    "CUR": CURAllocation,
    "TD" : TDAllocation
}

allocation_cfg_map = {
    "CUR": "CUR_allocation.yaml",
    "TD" : "TD_allocation.yaml",
    "random": "random_allocation.yaml"
}