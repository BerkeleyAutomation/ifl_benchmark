from .il_agent import SingleTaskParallelILAgent
from .random_agent import SingleTaskParallelRandomAgent

# Mappings from CLI option strings to agents
agent_map = {
    "IL": SingleTaskParallelILAgent,
    "Random": SingleTaskParallelRandomAgent
}

agent_cfg_map = {
    "IL": "il_agent.yaml"
}