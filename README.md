# Pavlov in Space: Beyond Tabular Reinforcement Methods with LunarLander-v2

## Overview
This project implements a Deep Q-Network (DQN) with various extensions such as Double DQN and Dueling DQN to tackle the Lunar Lander environment from the Gymnasium framework. The repository contains Python scripts and a Jupyter notebook that detail the training process, experimentation, and evaluation of different DQN configurations.

## Project Structure
Below is a description of the key files in this repository:

### Python Files
- **`agent.py`**: Defines the DQN agent along with the basic functionality for action selection and learning.
- **`train.py`**: Defines functions used to train the DQN agent across different configurations and hyperparameters.
- **`utils.py`**: Contains utility functions for model saving/loading, and other helper functions used across the project.
- **`memory.py`**: Implements the replay buffer used in DQN for storing and retrieving experiences.
- **`nets.py`**: Contains the neural network architectures, including the basic, Dueling, and other network variations used by the agent.
- **`plots.py`**: Script for generating plots and visualizations from training and test results.

### Jupyter Notebook
- **`experiments.ipynb`**: Interactive notebook that demonstrates the setup, execution, and analysis of experiments, showcasing the impact of various DQN enhancements.

### Checkpoints and Model Files
- `checkpoint_*.pth`: Model checkpoints saved during training for different network sizes and configurations.
- `dqn_*.pkl`: Pickle files store training statistics for models trained under various configurations such as `hard`, `soft`, `dueling`, and combinations thereof.
