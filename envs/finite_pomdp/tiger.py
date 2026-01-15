from itertools import count
from typing import Optional, Dict, List, Any, Tuple
from numpy.typing import ArrayLike, NDArray
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .finite_pomdp import FinitePOMDP

class Tiger_FiniteHorizon(FinitePOMDP):
    @staticmethod
    def build_transition_kernel_static(params: ArrayLike, n_states: int, n_actions: int) -> NDArray:
        # State labels: ['Tiger-Left', 'Tiger-Right', 'Win', 'Lose', 'End']
        # Action labels: ['Listen', 'Open-Left', 'Open-Right']
        transit_probs = np.array([
            [ # State 0 = Tiger-Left
                [1., 0., 0., 0., 0.], # Listen -> stay in Tiger-Left (0)
                [0., 0., 0., 1., 0.], # Open-Left -> Lose (3)
                [0., 0., 1., 0., 0.], # Open-Right -> Win (2)
            ], 
            [ # State 1 = Tiger-Right
                [0., 1., 0., 0., 0.], # Listen -> stay in Tiger-Right (1)
                [0., 0., 1., 0., 0.], # Open-Left -> Win (2)
                [0., 0., 0., 1., 0.], # Open-Right -> Lose (2)
            ], 
            [ # State 2 = Win
                [0., 0., 0., 0., 1.], # All actions -> End (4)
                [0., 0., 0., 0., 1.],  
                [0., 0., 0., 0., 1.], 
            ], 
            [ # State 3 = Lose
                [0., 0., 0., 0., 1.], # All actions -> End (4)
                [0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 1.],
            ], 
            [ # State 4 = End
                [0., 0., 0., 0., 1.], # All actions -> End (4)
                [0., 0., 0., 0., 1.],  
                [0., 0., 0., 0., 1.], 
            ], 
        ])
        return transit_probs

    @staticmethod
    def build_observation_kernel_static(params: ArrayLike, n_states: int, n_obs: int) -> NDArray:
        # params: [theta]
        theta = np.array(params).item()
        p = 0.5 + theta
        q = 0.5 - theta
        obs_probs = np.array([
            [p,  q,  0., 0.], # Tiger-Left
            [q,  p,  0., 0.], # Tiger-Right
            [0., 0., 1., 0.], # Win
            [0., 0., 0., 1.], # Lose
            [.5, .5, 0., 0.], # End
        ])
        return obs_probs

    @staticmethod
    def build_reward_matrix_static(params: ArrayLike, n_obs: int, n_actions: int) -> NDArray:
        # params: [listen_cost, treasure_reward, tiger_penalty]
        params = np.array(params)
        listen_cost, treasure_reward, tiger_penalty = params[0], params[1], params[2]
        rewards = np.zeros((n_obs, n_actions))
        rewards[:, 0] = listen_cost # Action 0: Listen 
        rewards[2, :] = treasure_reward # Observation 2: Win
        rewards[3, :] = tiger_penalty # Observation 3: Lose
        return rewards

    @staticmethod
    def build_init_state_probs_static(params: ArrayLike, n_states: int) -> NDArray:
        init_state_probs = np.zeros(n_states)
        init_state_probs[0:2] = 0.5
        return init_state_probs

    def __init__(
        self, 
        horizon: int = 10,
        discount: float = 1.0,
        theta: float = 0.35,
        listen_cost: float = -1.0, 
        treasure_reward: float = 10.0, 
        tiger_penalty: float = -100.0,
    ):
        params = {
            'transition_params': [], 
            'observation_params': [theta],
            'reward_params': [listen_cost, treasure_reward, tiger_penalty],
            'init_state_params': [],
        }
        super().__init__(
            state_labels=['Tiger-Left', 'Tiger-Right', 'Win', 'Lose', 'End'],
            action_labels=['Listen', 'Open-Left', 'Open-Right'],
            obs_labels=['Hear-Left', 'Hear-Right', 'Win', 'Lose'],
            horizon=horizon,
            params=params,
            discount=discount,
        )

    def _get_reward(self, obs: int, action: int) -> float:
        # Match TigerOld mixed discounting logic
        # action cost at t uses gamma^(t+1)
        # outcome reward for obs at t (from action t-1) uses gamma^t
        gamma = self.discount if self.discount is not None else 1.0
        curr_pwr = self.timestep # t+1
        prev_pwr = self.timestep - 1 # t
        
        listen_cost = self.params['reward_params'][0]
        treasure_reward = self.params['reward_params'][1]
        tiger_penalty = self.params['reward_params'][2]
        
        reward = 0.0
        if action == 0:
            reward += np.power(gamma, curr_pwr) * listen_cost
            
        if obs == 2:
            reward += np.power(gamma, prev_pwr) * treasure_reward
        elif obs == 3:
            reward += np.power(gamma, prev_pwr) * tiger_penalty
            
        return float(reward)

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        # Add extra termination condition from TigerOld
        if self._state == 4:
            terminated = True
        return obs, reward, terminated, truncated, info

    def generate_trajectory(self, actions: Optional[ArrayLike] = None) -> Dict[str, List[Any]]:
        obs, _ = self.reset()
        if actions is None:
           actions = np.zeros(self.horizon//2, dtype=int)
           _more_actions = self.np_random.integers(0, self.n_actions, size=self.horizon - actions.shape[0])
           actions = np.concatenate([actions, _more_actions])
           
        traj = {'obs': [obs], 'action': [], 'reward': []}
        for h in count():
            action = int(actions[h])
            obs, reward, terminated, _, _ = self.step(action)

            traj['action'].append(action)
            traj['obs'].append(obs)
            traj['reward'].append(reward)

            if terminated:
                break # end current episode
        return traj

    def generate_random_trajectories(self, n_episodes: int) -> List[Dict[str, List[Any]]]:
        return [self.generate_trajectory() for _ in range(n_episodes)]
