import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
from memory import ReplayBuffer
from nets import QNet, DuelingQNet, RNDNet
import random

# BUFFER_SIZE = int(1e5)  # replay buffer size
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.999            # discount factor
TAU = 1e-3              # for soft update of target parameters
# LR = 5e-4               # learning rate 
LR = 1e-3               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(
        self, 
        state_size, 
        action_size, 
        num_neurons=128, 
        update_target_every=1000, 
        apply_double_dqn = False,
        apply_dueling_dqn = False,
        apply_rnd = False,
        apply_hard_update = False,
        gamma = GAMMA,
        # update_every = UPDATE_EVERY, TODO: maybe revisit
        lr = LR,
        tau = TAU,
        batch_size = BATCH_SIZE,
        buffer_size = BUFFER_SIZE,
        config = {},
    ):
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

        self.config = config

        # self.update_every = update_every
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma

        self.apply_double_dqn = apply_double_dqn
        self.num_neurons = num_neurons
        self.state_size = state_size
        self.action_size = action_size
        self.update_target_every = update_target_every
        self.apply_rnd = apply_rnd
        self.apply_hard_update = apply_hard_update

        # Q-Network
        if apply_dueling_dqn:
            net_class = DuelingQNet
        else:
            net_class = QNet

        self.net = net_class(state_size, action_size, num_neurons).to(device)
        self.target_net = net_class(state_size, action_size, num_neurons).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.99)  # Adjust step_size and gamma as needed

        # self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)  # Starting with a higher learning rate
        # self.scheduler = StepLR(self.optimizer, step_size=300, gamma=0.95)  # Adjusting every 100 episodes

        self.target_update_counter = 0 # Initialize counter for target network updates

        # RND Networks
        if apply_rnd:
            self.rnd_target = RNDNet(state_size).to(device)
            self.rnd_predictor = RNDNet(state_size).to(device)
            self.rnd_optimizer = optim.Adam(self.rnd_predictor.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size)

        # # Initialize time step (for updating every self.update_every steps)
        # self.t_step = 0

        # Initialize time step
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_target_every
        # if self.t_step == 0:

        # If enougeh samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)
            # self.schduler.step()  # Update the learning rate based on the scheduler


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        state = torch.tensor(np.array([state]), device=device)
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.net.eval()
        with torch.no_grad():
            action_values = self.net(state)
        self.net.train()

        action_values = action_values.cpu().data.numpy()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values), action_values
        else:
            return random.choice(np.arange(self.action_size)), action_values

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences

        if self.apply_rnd:
             # Compute intrinsic reward using the prediction error
            target_features = self.rnd_target(next_states).detach()
            predicted_features = self.rnd_predictor(next_states)
            intrinsic_reward = F.mse_loss(predicted_features, target_features, reduction='none').mean(1).unsqueeze(1)
    
            # Sum intrinsic and extrinsic rewards
            rewards = rewards + intrinsic_reward.detach()

        if self.apply_double_dqn:
            # Get max predicted Q values (for next states) from target model
            # First, select the best action from the local model (action selection)
            local_action_selection = self.net(next_states).detach().max(1)[1].unsqueeze(1)
            
            # Now, evaluate the selected action with the target model (action evaluation)
            q_targets_next = self.target_net(next_states).detach().gather(1, local_action_selection)
        else:
            ## Compute and minimize the loss
            ### Extract next maximum estimated value from target network
            q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)

        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        ### Calculate expected value from local network
        q_expected = self.net(states).gather(1, actions)

        ### Loss calculation (we used Mean squared error)
        # loss = F.mse_loss(q_expected, q_targets)
        loss = F.huber_loss(q_expected, q_targets) # TODO: revisit
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 100) #TODO: revisit
        self.optimizer.step()

        if self.apply_rnd:
            # Update RND predictor network
            self.rnd_optimizer.zero_grad()
            intrinsic_loss = F.mse_loss(predicted_features, target_features)
            intrinsic_loss.backward()
            self.rnd_optimizer.step()

        if self.apply_hard_update:
            # Update the target network every self.update_target_every time steps
            # self.target_update_counter += 1
            # if self.target_update_counter >= self.update_target_every:
            if self.t_step == 0:
                self.hard_update(self.net, self.target_net)
                self.target_update_counter = 0  # reset the update counter
        else:
            # ------------------- update target network ------------------- #
            self.soft_update(self.net, self.target_net, self.tau)

    def hard_update(self, model, target_model):
        """Hard update model parameters.
        θ_target = θ_local
        """
        target_model.load_state_dict(model.state_dict())

    def soft_update(self, model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
