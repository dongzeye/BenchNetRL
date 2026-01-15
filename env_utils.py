import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from envs.poc_memory_env import PocMemoryEnv
from collections import deque

class VecObservationStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        # Ensure the low/high bounds are NumPy arrays and repeat them along the stacking axis.
        low = np.repeat(np.array(env.observation_space.low, dtype=np.float32), num_stack, axis=0)
        high = np.repeat(np.array(env.observation_space.high, dtype=np.float32), num_stack, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        # Reset the underlying env and ensure the observation is a NumPy array.
        obs, info = self.env.reset(**kwargs)
        obs = np.array(obs, dtype=np.float32)
        # Initialize the buffer with copies of the initial observation.
        self.frames = deque([obs.copy() for _ in range(self.num_stack)], maxlen=self.num_stack)
        return self._get_obs(), info

    def observation(self, observation):
        observation = np.array(observation, dtype=np.float32)
        # Append a copy to ensure consistency.
        self.frames.append(observation.copy())
        return self._get_obs()

    def _get_obs(self):
        # Concatenate along the first axis.
        return np.concatenate(list(self.frames), axis=0)

class OneHotObservationWrapper(gym.ObservationWrapper):
    """Converts discrete observations to one-hot encoded vectors."""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete), \
            "OneHotObservationWrapper only works with Discrete observation spaces"
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n,), dtype=np.float32
        )

    def observation(self, observation):
        one_hot = np.zeros(self.n, dtype=np.float32)
        one_hot[observation] = 1.0
        return one_hot


class TimeAwareObservationWrapper(gym.Wrapper):
    """Appends normalized remaining time (horizon - t) / horizon to observations."""
    def __init__(self, env, horizon=None):
        super().__init__(env)
        # NOTE: Only works if the env has a 'horizon' attribute or wrapper
        self.horizon = horizon or env.get_wrapper_attr("horizon")
        self.current_step = 0
        # Extend observation space by 1 dimension for time
        orig_space = env.observation_space
        orig_shape = orig_space.shape
        new_shape = (np.prod(orig_shape) + 1,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=new_shape, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        return self._add_time(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        if terminated or truncated:
            self.current_step = 0
        return self._add_time(obs), reward, terminated, truncated, info

    def _add_time(self, obs):
        obs = np.array(obs, dtype=np.float32).flatten()
        remaining_time = (self.horizon - self.current_step) / self.horizon
        return np.append(obs, remaining_time)


class MaskObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, mask_indices, mask_prob=1.0):
        super().__init__(env)
        self.mask_indices = mask_indices
        self.mask_prob = mask_prob

        # Keep original bounds for observation space (unchanged)
        self.observation_space = env.observation_space

    def observation(self, observation):
        observation = np.array(observation, dtype=np.float32)

        # Apply 50% masking probability for each index
        for i in self.mask_indices:
            if np.random.rand() < self.mask_prob:
                observation[i] = 0.0

        return observation

class RecordEpisodeStatistics(gym.Wrapper):
  def __init__(self, env, deque_size=100):
    super(RecordEpisodeStatistics, self).__init__(env)
    self.num_envs = getattr(env, "num_envs", 1)
    self.episode_returns = None
    self.episode_lengths = None
    # get if the env has lives
    self.has_lives = False
    env.reset()
    info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
    if info["lives"].sum() > 0:
      self.has_lives = True
      print("env has lives")

  def reset(self, **kwargs):
    observations = self.env.reset()
    self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
    self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    self.lives = np.zeros(self.num_envs, dtype=np.int32)
    self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
    self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    return observations

  def step(self, action):
    observations, rewards, term, trunc, infos = self.env.step(action)
    dones = term + trunc
    self.episode_returns += infos["reward"]
    self.episode_lengths += 1
    self.returned_episode_returns[:] = self.episode_returns
    self.returned_episode_lengths[:] = self.episode_lengths
    all_lives_exhausted = infos["lives"] == 0
    if self.has_lives:
      self.episode_returns *= 1 - all_lives_exhausted
      self.episode_lengths *= 1 - all_lives_exhausted
    else:
      self.episode_returns *= 1 - dones
      self.episode_lengths *= 1 - dones
    infos["r"] = self.returned_episode_returns
    infos["l"] = self.returned_episode_lengths
    return (
      observations,
      rewards,
      term,
      trunc,
      infos,
    )

def make_atari_env(gym_id, seed, idx, capture_video, run_name, frame_stack=1):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array", repeat_action_probability=0.0) if capture_video else gym.make(gym_id, repeat_action_probability=0.0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        #env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, frame_stack)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def make_classic_env(gym_id, seed, idx, capture_video, run_name, masked_indices=[], obs_stack=1, time_aware=False):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array") if capture_video else gym.make(gym_id)
        if isinstance(env.observation_space, gym.spaces.Discrete):
            env = OneHotObservationWrapper(env)
        if time_aware:
            env = TimeAwareObservationWrapper(env)
        if masked_indices:
            env = MaskObservationWrapper(env, masked_indices)
        if obs_stack > 1:
            env = VecObservationStackWrapper(env, num_stack=obs_stack)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def make_memory_gym_env(gym_id, seed, idx, capture_video, run_name):
    import memory_gym # noqa: F401 # register the envs
    def thunk():
        env = gym.make(
            gym_id,
            render_mode="rgb_array" if capture_video else None,
        )
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def make_minigrid_env(gym_id, seed, idx, capture_video, run_name, agent_view_size=3, tile_size=28, max_episode_steps=96, frame_stack=1):
    def thunk():
        env = gym.make(
            gym_id,
            agent_view_size=agent_view_size,
            tile_size=tile_size,
            render_mode="rgb_array" if capture_video else None,
        )
        env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=tile_size))
        if frame_stack > 1:
            env = gym.wrappers.FrameStack(env, frame_stack)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def make_poc_env(gym_id, seed, idx, capture_video, run_name, step_size=0.2, glob=False, freeze=False, max_episode_steps=96):
    def thunk():
        env = PocMemoryEnv(step_size=step_size, glob=glob, freeze=freeze, max_episode_steps=max_episode_steps)
        return env
    return thunk

def make_continuous_env(gym_id, seed, idx, capture_video, run_name, obs_stack=1):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array") if capture_video else gym.make(gym_id)
        if obs_stack > 1:
            env = VecObservationStackWrapper(env, num_stack=obs_stack)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk