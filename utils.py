import numpy as np
import pandas as pd
import pickle
import gymnasium as gym

def moving_average(rewards_over_seeds, window_size = 50):
    rewards_over_seeds = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]

    rewards_over_seeds = [[np.mean(rewards[max(i-window_size+1, 0):i+1]) for i in range(len(rewards))] for rewards in rewards_over_seeds]
    rewards_over_seeds = [[[reward] for reward in rewards] for rewards in rewards_over_seeds]
    return rewards_over_seeds

def save_python_obj(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load_python_obj(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

class MaxAndSkipObservation(
    gym.Wrapper,
    gym.utils.RecordConstructorArgs,
):
    """Skips the N-th frame (observation) and return the max values between the two last observations.

    No vector version of the wrapper exists.

    Note:
        This wrapper is based on the wrapper from [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#MaxAndSkipEnv)

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> obs0, *_ = env.reset(seed=123)
        >>> obs1, *_ = env.step(1)
        >>> obs2, *_ = env.step(1)
        >>> obs3, *_ = env.step(1)
        >>> obs4, *_ = env.step(1)
        >>> skip_and_max_obs = np.max(np.stack([obs3, obs4], axis=0), axis=0)
        >>> env = gym.make("CartPole-v1")
        >>> wrapped_env = MaxAndSkipObservation(env)
        >>> wrapped_obs0, *_ = wrapped_env.reset(seed=123)
        >>> wrapped_obs1, *_ = wrapped_env.step(1)
        >>> np.all(obs0 == wrapped_obs0)
        True
        >>> np.all(wrapped_obs1 == skip_and_max_obs)
        True

    Change logs:
     * v1.0.0 - Initially add
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        """This wrapper will return only every ``skip``-th frame (frameskipping) and return the max between the two last frames.

        Args:
            env (Env): The environment to apply the wrapper
            skip: The number of frames to skip
        """
        gym.utils.RecordConstructorArgs.__init__(self, skip=skip)
        gym.Wrapper.__init__(self, env)

        if not np.issubdtype(type(skip), np.integer):
            raise TypeError(
                f"The skip is expected to be an integer, actual type: {type(skip)}"
            )
        if skip < 2:
            raise ValueError(
                f"The skip value needs to be equal or greater than two, actual value: {skip}"
            )
        if env.observation_space.shape is None:
            raise ValueError("The observation space must have the shape attribute.")

        self._skip = skip
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )

    def step(
        self, action
    ):
        """Step the environment with the given action for ``skip`` steps.

        Repeat action, sum reward, and max over last observations.

        Args:
            action: The action to step through the environment with
        Returns:
            Max of the last two observations, reward, terminated, truncated, and info from the environment
        """
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = np.max(self._obs_buffer, axis=0)

        return max_frame, total_reward, terminated, truncated, info
