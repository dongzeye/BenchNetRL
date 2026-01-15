from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import create_finite_horizon_stationary_pomdp_file


class FinitePOMDP(gym.Env):
    metadata = {'render_modes': ['human']}

    @staticmethod
    def build_transition_kernel_static(params: ArrayLike, n_states: int, n_actions: int) -> NDArray:
        """
        Construct transition prob. matrixs for a batch of params.
        """
        raise NotImplementedError

        
    @staticmethod
    def build_observation_kernel_static(params: ArrayLike, n_states: int, n_obs: int) -> NDArray:
        """
        Construct observation prob. matrix Z[s, o] = Z(o | s) according to the params.
        """
        raise NotImplementedError
    
    @staticmethod
    def build_reward_matrix_static(params: ArrayLike, n_obs: int, n_actions: int) -> NDArray:
        """
        Construct reward matrix R[o, a] = R(o, a) according to the params.
        """
        raise NotImplementedError
    
    @staticmethod
    def build_init_state_probs_static(params: ArrayLike, n_states: int) -> NDArray:
        """
        Construct initial state prob. vector according to the params.
        """
        raise NotImplementedError
    
    def __init__(
        self, 
        state_labels: List[str], 
        action_labels: List[str], 
        obs_labels: List[str], 
        horizon: int, 
        params: Dict[str, Any],
        discount: Optional[float] = None,
    ):
        super().__init__()
        # State, action, and observation labels
        self.state_labels = state_labels
        self.action_labels = action_labels
        self.obs_labels = obs_labels

        # Number of states, actions, and observations
        self.n_states = len(self.state_labels)
        self.n_actions = len(self.action_labels)
        self.n_obs = len(self.obs_labels)

        # Spaces for state, action, and observation
        self.state_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_obs)

        self.horizon = horizon
        self.timestep = 0  # time-step counter
        
        # Discounting factor
        self.discount = discount

        # Internal state and observation trackers
        self._state = None  # current state
        self._obs = None    # current observation

        # POMDP parameters (initialize to empty; some env may be un-parameterized)
        self.params = {}
        # Transition prob. matrix T[s, a, s'] = T(s' | s, a)
        self.transit_probs = None
        # Observation prob. matrix Z[s, o] = Z(o | s)
        self.obs_probs = None
        # Reward matrix R[o, a] = R(o, a)
        self.rewards = None
        # Initial state prob. vector b0[s] = b0(s)
        self.init_state_probs = None

        # Set model params
        self.set_params(params)
    
    def build_transition_kernel(self, params: ArrayLike, *args: Any, **kwargs: Any) -> NDArray:
        return self.build_transition_kernel_static(params, self.n_states, self.n_actions)
    
    def build_observation_kernel(self, params: ArrayLike, *args: Any, **kwargs: Any) -> NDArray:
        return self.build_observation_kernel_static(params, self.n_states, self.n_obs)
    
    def build_reward_matrix(self, params: ArrayLike, *args: Any, **kwargs: Any) -> NDArray:
        return self.build_reward_matrix_static(params, self.n_obs, self.n_actions)
    
    def build_init_state_probs(self, params: ArrayLike, *args: Any, **kwargs: Any) -> NDArray:
        return self.build_init_state_probs_static(params, self.n_states)
    
    def set_params(self, params: Dict[str, Any], **kwargs: Any) -> None:
        params = params | kwargs
        params = {k: np.array(v) for k, v in params.items()}
        if 'transition_kernel' in params:
            self.transit_probs = params['transition_kernel']

        if 'observation_kernel' in params:
            self.obs_probs = params['observation_kernel']

        if 'reward_matrix' in params:
            self.rewards = params['reward_matrx']

        if 'init_state_probs' in params:
            self.init_state_probs = params['init_state_probs']

        if 'transition_params' in params:
            self.transit_probs = self.build_transition_kernel(params['transition_params'])

        if 'observation_params' in params:
            self.obs_probs = self.build_observation_kernel(params['observation_params'])

        if 'reward_params' in params:
            self.rewards = self.build_reward_matrix(params['reward_params'])

        if 'init_state_params' in params:
            self.init_state_probs = self.build_init_state_probs(params['init_state_params'])

        self.params = self.params | params
        # Normalize transition and observation probabilities to avoid floating point errors
        self.transit_probs = self.transit_probs.astype('float64') / np.sum(self.transit_probs, axis=-1, keepdims=True)
        self.obs_probs = self.obs_probs.astype('float64') / np.sum(self.obs_probs, axis=-1, keepdims=True)
        self.init_state_probs = self.init_state_probs.astype('float64') / np.sum(self.init_state_probs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Reset the environment to an initial state and observation.
        """
        super().reset(seed=seed, options=options)
        self._state = int(self.np_random.choice(self.n_states, p=self.init_state_probs))
        self._obs = self._get_obs()
        self.timestep = 0
        return self._obs, self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        return {'timestep': self.timestep}
    
    def _get_obs(self) -> int:
        obs_probs = self.obs_probs[self._state] # Z(* | s)
        return int(self.np_random.choice(self.n_obs, p=obs_probs))

    def _get_reward(self, obs: int, action: int) -> float:
        if self.discount is not None:
            return float(np.power(self.discount, self.timestep) * self.rewards[obs, action])
        else:
            return float(self.rewards[obs, action])
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        # Move to next time-step
        self.timestep += 1
        # Get new state
        next_state_probs = self.transit_probs[self._state, action] # T( * | s, a)
        self._state = int(self.np_random.choice(self.n_states, p=next_state_probs))
        # Get reward r(o, a)
        reward = self._get_reward(self._obs, action)
        # Get new obs
        self._obs = self._get_obs()

        # Episode ends once we enter time H
        terminated = (self.timestep >= self.horizon)
        
        return self._obs, reward, terminated, False, self._get_info()

    def render(self, mode='human', close=False):
        if close:
            return
        print(f"Observation: {self._obs}")

    def clone(self, remove_params: bool = True) -> 'FinitePOMDP':
        # Create a deep copy
        new_env = deepcopy(self)
        if remove_params:
            new_env.params = {}
            new_env.transit_probs = None
            new_env.obs_probs = None
            new_env.rewards = None
            new_env.init_state_probs = None
        return new_env
    
    def to_pomdp_file(self, discount: float, filepath: str, decimals: int = 6, header: Optional[str] = None) -> str:
        return create_finite_horizon_stationary_pomdp_file(
            horizon=self.horizon,
            states=self.state_labels,
            actions=self.action_labels,
            observations=self.obs_labels,
            init_state_probs=self.init_state_probs,
            transition_matrix=self.transit_probs,
            observation_matrix=self.obs_probs,
            reward_matrix=self.rewards,
            discount=discount,
            pomdp_path=filepath,
            decimals=decimals,
            header=header,
        )


class SparseRewardPOMDP(FinitePOMDP):
    @staticmethod
    def build_transition_kernel_static(params: ArrayLike, n_states: int, n_actions: int) -> NDArray:
        """
        Construct transition prob. matrixs for a batch of params.
        """
        raise NotImplementedError

        
    @staticmethod
    def build_observation_kernel_static(params: ArrayLike, n_states: int, n_obs: int) -> NDArray:
        """
        Construct observation prob. matrix Z[s, o] = Z(o | s) according to the params.
        """
        params = np.array(params)
        q1, q2, q3 = params[..., 0], params[..., 1], params[..., 2]
        batch_shape = params.shape[:-1]
        obs_probs = np.zeros(batch_shape + (n_states, n_obs))
        
        # At s = 0 or L-1, no observation noise
        obs_probs[..., 0, 0] = 1.0
        obs_probs[..., -1, -1] = 1.0
        
        # At s = 1, observe o = 1 w.p. q1 and o = 2 w.p. q2 + q3
        # Use tuple for specific indices
        obs_probs[..., 1, 1] = q1
        obs_probs[..., 1, 2] = q2 + q3
        
        # At s = L-2, observe o = L-2 w.p. q1 and o = L-3 w.p. q2 + q3
        obs_probs[..., -2, -2] = q1
        obs_probs[..., -2, -3] = q2 + q3
        
        # At s = 2, ..., L-2:
        #   o = s w.p. q1
        #   o = s + 1 w.p. q2
        #   o = s - 1 w.p. q3
        if n_states > 4:
            states = np.arange(2, n_states - 2)
            # Use advanced indexing for batch and states
            # Note: For multi-dimensional broadcasting, we need to be careful.
            # Assuming batch_shape is simple or empty.
            obs_probs[..., states, states] = q1[..., np.newaxis] if batch_shape else q1
            obs_probs[..., states, states + 1] = q2[..., np.newaxis] if batch_shape else q2
            obs_probs[..., states, states - 1] = q3[..., np.newaxis] if batch_shape else q3
            
        return obs_probs
    
    @staticmethod
    def build_reward_matrix_static(params: ArrayLike, n_obs: int, n_actions: int) -> NDArray:
        """
        Construct reward matrix R[o, a] = R(o, a) according to the params.
        """
        params = np.array(params)
        r1, r2 = params[..., 0], params[..., 1]
        batch_shape = params.shape[:-1]
        # R(o, a) = r1 if o = 0, r2 if o = L-1, and 0 otherwise.
        rewards = np.zeros(batch_shape + (n_obs, n_actions))
        rewards[..., 0, :] = r1
        rewards[..., -1, :] = r2
        return rewards
    
    @staticmethod
    def build_init_state_probs_static(params: ArrayLike, n_states: int) -> NDArray:
        """
        Construct initial state prob. vector according to the params.
        """
        raise NotImplementedError

    def __init__(
        self, 
        n_states: Optional[int] = None, 
        n_actions: Optional[int] = None, 
        n_obs: Optional[int] = None, 
        horizon: Optional[int] = None, 
        params: Optional[Dict[str, Any]] = None, 
        discount: Optional[float] = None, 
        config_path: Optional[str] = None
    ):
        if config_path is not None:
            import json
            import os
            # Use absolute path if config_path is not found relatively
            if not os.path.exists(config_path):
                # Try relative to the package root if needed, but here we assume path is relative to CWD or absolute
                pass
            with open(config_path, 'r') as f:
                config = json.load(f)
            n_states = config.get('n_states', n_states)
            n_actions = config.get('n_actions', n_actions)
            n_obs = config.get('n_obs', n_obs)
            horizon = config.get('horizon', horizon)
            params = config.get('params', params)
            discount = config.get('discount', discount)

        super().__init__(
            state_labels=[f's{i}' for i in range(n_states)],
            action_labels=[f'a{i}' for i in range(n_actions)],
            obs_labels=[f'o{i}' for i in range(n_obs)],
            horizon=horizon,
            params=params,
            discount=discount,
        )

