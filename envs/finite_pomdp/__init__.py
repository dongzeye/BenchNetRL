import os
from gymnasium import register

from .tiger import Tiger_FiniteHorizon
from .finite_pomdp import SparseRewardPOMDP
from .river_swim import RiverSwim


# register specific Tiger variants
for theta in [0.2, 0.3, 0.4]:
    register(
        id=f"Tiger-Theta{int(theta*10)}-v0",
        entry_point="envs.finite_pomdp.tiger:Tiger_FiniteHorizon",
        kwargs={
            "horizon": 10,
            "discount": 0.99,
            "theta": theta,
            "listen_cost": -1,
            "treasure_reward": 10,
            "tiger_penalty": -100,
        }
    )

# register specific RiverSwim variant
register(
    id="RiverSwim-Hard-v0",
    entry_point="envs.finite_pomdp.river_swim:RiverSwim",
    kwargs={
        "river_length": 6,
        "horizon": 40,
        "discount": 1.0,
        "params": {
            "transition_params": [0.6, 0.35, 0.05],
            "observation_params": [0.6, 0.2, 0.2],
            "reward_params": [0.005, 1.0],
            "init_state_probs": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }
)

# register specific SparseRewardPOMDP variants (Random POMDPs)
config_dir = os.path.join(os.path.dirname(__file__), "env_configs")
for i in range(5):
    register(
        id=f"SparseRewardPOMDP-Random{i}-v0",
        entry_point="envs.finite_pomdp.finite_pomdp:SparseRewardPOMDP",
        kwargs={
            "config_path": os.path.join(config_dir, f"random_sparse_reward_{i}.json")
        }
    )