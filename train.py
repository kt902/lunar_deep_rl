import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from utils import save_python_obj, load_python_obj
from agent import Agent


def train_agent_across_seeds(
    env,
    # agent_class = Agent,
    config = {},
    seeds = [2, 3, 5, 8], 
    n_episodes=1000, 
    max_t=1000, 
    eps_start=1.0, 
    eps_end=0.01, 
    eps_decay=0.995, 
    # num_neurons=128,
    solved_score=250.0
):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.n

    rewards_over_seeds = []
    # print(f"Agent class is default={agent_class==Agent}")

    # Create agent instance
    agent_args = {
        "state_size": obs_space_dims, 
        "action_size": action_space_dims,
        "config": config,
    }

    agent_config = config.get('agent', {})
    agent_args.update(agent_config)
    agent = Agent(**agent_args)

    # agent = agent_class(state_size=obs_space_dims, action_size=action_space_dims, num_neurons=num_neurons)

    # For each seed
    for seed in seeds: # Fibonacci seeds. A seed of 1 is somehow not great :'(. TODO: discuss in report
        reward_over_episodes = train_agent(
            solved_score=solved_score, 
            seed=seed, 
            agent=agent, 
            env=env, 
            n_episodes=n_episodes, 
            max_t=max_t, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay
        )
        rewards_over_seeds.append(reward_over_episodes)


    save_python_obj(rewards_over_seeds, f'./{file_name(agent.config)}.pkl')

    return rewards_over_seeds

def file_name(config):
    files_prefix = config.get('files_prefix', "")
    agent_config = config.get('agent', {})

    save_name = f"{files_prefix}dqn_{agent_config.get("num_neurons", 128)}"
    if agent_config.get("apply_hard_update", False):
        save_name = f"{save_name}_hard"
    else:
        save_name = f"{save_name}_soft"

    for (k, l) in [
        ("apply_double_dqn", "double"),
        ("apply_dueling_dqn", "dueling"),
        ("apply_rnd", "rnd"),
    ]:
        if agent_config.get(k, False):
            save_name = f"{save_name}_{l}"
    return save_name


def train_agent(seed=1, agent=None, env=None, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.996, solved_score=200.0):
    env = gym.wrappers.RecordEpisodeStatistics(env, 100) 

    reward_over_episodes = []

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    print('Number of neurons: ', agent.num_neurons)

    # Initialize epsilon
    eps = eps_start

    for episode in range(1, n_episodes+1):
        # gymnasium v26 requires users to set seed while resetting the environment
        state, _ = env.reset(seed=episode) #TODO: remove this seeding ?
        score = 0
        for t in range(max_t):
            action, _ = agent.act(state, eps)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        average_score = np.mean(env.return_queue)
        # agent.scheduler.step() 
        
        # TODO: let's make this not a list item
        reward_over_episodes.append(env.return_queue[-1])

        print('\rEpisode {}\tAverage Reward: {:.2f} Epsilon: {:.2f}'.format(episode, average_score, eps), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Reward: {:.2f} Epsilon: {:.2f}'.format(episode, average_score, eps))
        # if average_score >=solved_score:
        if all(value >=solved_score for value in env.return_queue):
            print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:.2f}'.format(episode, average_score))
            break

    # Always save the checkpoint
    torch.save(agent.net.state_dict(), f'./checkpoints/{file_name(agent.config)}.pth')

    return reward_over_episodes
