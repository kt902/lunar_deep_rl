from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np

class StopTrainingOnMaxEpisodes(BaseCallback):
    """
    Stop the training once a maximum number of episodes are played.

    For multiple environments presumes that, the desired behavior is that the agent trains on each env for ``max_episodes``
    and in total for ``max_episodes * n_envs`` episodes.

    :param max_episodes: Maximum number of episodes to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about when training ended by
        reaching ``max_episodes``
    """

    def __init__(self, max_episodes: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_episodes = max_episodes
        self._total_max_episodes = max_episodes
        self.n_episodes = 0
        self.reward_over_episodes = []

    def _init_callback(self) -> None:
        # At start set total max according to number of envirnments
        self._total_max_episodes = self.max_episodes * self.training_env.num_envs

    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        n_episodes = self.n_episodes + np.sum(self.locals["dones"]).item()

        if n_episodes > self.n_episodes:
            self.reward_over_episodes.append(self.training_env.envs[0].return_queue[-1])
        self.n_episodes = n_episodes

        continue_training = self.n_episodes < self._total_max_episodes

        if self.verbose >= 1 and not continue_training:
            mean_episodes_per_env = self.n_episodes / self.training_env.num_envs
            mean_ep_str = (
                f"with an average of {mean_episodes_per_env:.2f} episodes per env" if self.training_env.num_envs > 1 else ""
            )

            print(
                f"Stopping training with a total of {self.num_timesteps} steps because the "
                f"{self.locals.get('tb_log_name')} model reached max_episodes={self.max_episodes}, "
                f"by playing for {self.n_episodes} episodes "
                f"{mean_ep_str}"
            )
        return continue_training

callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=500, verbose=1)

env = gym.make('LunarLander-v2')
env = gym.wrappers.RecordEpisodeStatistics(env, 100) 

model = DQN("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=100000, log_interval=4, callback=callback_max_episodes)
model.save("dqn_cartpole")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# rewards_over_seeds_1 = load_python_obj('./rewards/rewards_over_seeds.pkl')
# rewards_over_seeds_2 = load_python_obj('/Users/kumbirai/rewards_over_seeds.pkl')
# rewards_over_seeds_3 = load_python_obj('./rewards/rewards_over_seeds_256.pkl')

rewards_over_seeds = [callback_max_episodes.reward_over_episodes]

plt.rcParams["figure.figsize"] = (10, 5)
rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="DQN 128 for LunarLander-v2"
)
plt.xlim(0, 600)
plt.ylim(-300, 300)
plt.show()
