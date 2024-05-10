import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, num_neurons):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """

        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_size, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc3 = nn.Linear(num_neurons, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

# Attempt at batch normalization
# class QNet(nn.Module):
#     def __init__(self, state_size, action_size, num_neurons):
#         super(QNet, self).__init__()
#         self.fc1 = nn.Linear(state_size, num_neurons)
#         self.bn1 = nn.BatchNorm1d(num_neurons)  # Batch normalization layer after first fully connected layer
#         self.fc2 = nn.Linear(num_neurons, num_neurons)
#         self.bn2 = nn.BatchNorm1d(num_neurons)  # Batch normalization layer after second fully connected layer
#         self.fc3 = nn.Linear(num_neurons, action_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)  # Apply batch normalization before activation
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = self.bn2(x)  # Apply batch normalization before activation
#         x = F.relu(x)
#         x = self.fc3(x)
#         return x

class DuelingQNet(nn.Module):
    def __init__(self, state_size, action_size, num_neurons):
        super(DuelingQNet, self).__init__()
        self.fc1_value = nn.Linear(state_size, num_neurons)
        self.fc2_value = nn.Linear(num_neurons, 1)
        self.fc1_advantage = nn.Linear(state_size, num_neurons)
        self.fc2_advantage = nn.Linear(num_neurons, action_size)

    def forward(self, x):
        x_value = F.relu(self.fc1_value(x))
        x_value = self.fc2_value(x_value)
        
        x_advantage = F.relu(self.fc1_advantage(x))
        x_advantage = self.fc2_advantage(x_advantage)
        
        # Combine the value and advantage streams to get the Q-values
        x_final = x_value + (x_advantage - x_advantage.mean(dim=1, keepdim=True))
        return x_final

class RNDNet(nn.Module):
    def __init__(self, state_size, output_size=128):
        super(RNDNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)

