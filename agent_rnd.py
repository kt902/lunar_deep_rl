import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from memory import ReplayBuffer
from nets import QNet
from nets import DuelingQNet as QNet
import random

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.999           # discount factor
TAU = 1e-3              # for soft update of target parameters
# LR = 5e-4               # learning rate 
LR = 1e-3               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNDTarget(nn.Module):
    def __init__(self, state_size, output_size=128):
        super(RNDTarget, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)

class RNDPredictor(nn.Module):
    def __init__(self, state_size, output_size=128):
        super(RNDPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(
        self, 
        state_size, 
        action_size, 
        num_neurons=128, 
        update_target_every=1000,
        apply_double_dqn=False,
        apply_dueling_dqn=False,
        apply_rnd=False,
        apply_hard_update=False,
        config = {}):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            update_target_every (int): how often to perform a hard update of target network
        """

        print("")
        print("Agent initialized with:")
        for arg_name, arg_value in locals().items():
            if arg_name != 'self':
                print(f"{arg_name}: {arg_value}")
        print("")

        self.num_neurons = num_neurons
        self.state_size = state_size
        self.action_size = action_size
        self.update_target_every = update_target_every

        # Q-Networks
        self.net = QNet(state_size, action_size, num_neurons).to(device)
        self.target_net = QNet(state_size, action_size, num_neurons).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)

        # RND Networks
        self.rnd_target = RNDTarget(state_size).to(device)
        self.rnd_predictor = RNDPredictor(state_size).to(device)
        self.rnd_optimizer = optim.Adam(self.rnd_predictor.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # Initialize counter for target network updates
        self.target_update_counter = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.01):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        state = torch.tensor(np.array([state]), device=device)
        self.net.eval()
        with torch.no_grad():
            action_values = self.net(state)
        self.net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()), action_values
        else:
            return random.choice(np.arange(self.action_size)), action_values

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Compute intrinsic reward using the prediction error
        target_features = self.rnd_target(next_states).detach()
        predicted_features = self.rnd_predictor(next_states)
        intrinsic_reward = F.mse_loss(predicted_features, target_features, reduction='none').mean(1).unsqueeze(1)

        # Sum intrinsic and extrinsic rewards
        total_rewards = rewards + intrinsic_reward.detach()

        # Double DQN: Begin by selecting the best predicted action from the local model
        local_action_selection = self.net(next_states).detach().max(1)[1].unsqueeze(1)
        # Evaluate the selected action with the target model
        q_targets_next = self.target_net(next_states).detach().gather(1, local_action_selection)
        q_targets = total_rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.net(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update RND predictor network
        self.rnd_optimizer.zero_grad()
        intrinsic_loss = F.mse_loss(predicted_features, target_features)
        intrinsic_loss.backward()
        self.rnd_optimizer.step()

        # Hard or soft update the target network
        if self.update_target_every > 0:  # implies hard update every specified steps
            self.target_update_counter += 1
            if self.target_update_counter >= self.update_target_every:
                self.hard_update(self.net, self.target_net)
                self.target_update_counter = 0  # reset the update counter
        else:
            self.soft_update(self.net, self.target_net, TAU)

    def hard_update(self, local_model, target_model):
        """Hard update model parameters."""
        target_model.load_state_dict(local_model.state_dict())

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

