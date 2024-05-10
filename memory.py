import torch
import numpy as np
from collections import deque, namedtuple
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences ])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences ])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences ])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences ])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        # states = torch.stack([torch.from_numpy(e.state).float() for e in experiences]).to(device)
        # actions = torch.stack([torch.tensor([e.action], dtype=torch.long) for e in experiences]).to(device)
        # rewards = torch.stack([torch.tensor([e.reward], dtype=torch.float) for e in experiences]).to(device)
        # next_states = torch.stack([torch.from_numpy(e.next_state).float() for e in experiences]).to(device)
        # dones = torch.stack([torch.tensor([float(e.done)], dtype=torch.float) for e in experiences]).to(device)

        # print(states, actions, rewards, next_states, dones)
        # print(states.size(), actions.size(), rewards.size(), next_states.size(), dones.size())
        # raise Exception("Sorry, no numbers below zero")
        return (states, actions, rewards, next_states, dones)

    # def add(self, state, action, reward, next_state, done):
    #     # Note: Ensure that state, action, etc., are converted to torch tensors before being pushed to buffer
    #     state = torch.tensor([state], device=device, dtype=torch.float32)
    #     action = torch.tensor([[action]], device=device, dtype=torch.long)
    #     reward = torch.tensor([[reward]], device=device, dtype=torch.float32)  # Ensure 2D tensor
    #     next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
    #     done = torch.tensor([[done]], device=device, dtype=torch.float32)  # Ensure 2D tensor


    #     self.memory.append(self.experience(state, action, reward, next_state, done))

    # def sample(self):
    #     transitions = random.sample(self.memory, k=self.batch_size)
    #     batch = self.experience(*zip(*transitions))
    #     # batch = self.experience(*zip(*((s, a, r, ns, d) for s, a, r, ns, d in transitions if None not in (s, a, r, ns, d))))


    # #     # Stack together all items from the batch
    #     states = torch.cat(batch.state)
    #     actions = torch.cat(batch.action)
    #     rewards = torch.cat(batch.reward)
    #     next_states = torch.cat(batch.next_state)
    #     dones = torch.cat(batch.done)

    #     # print(states.size(), actions.size(), rewards.size(), next_states.size(), dones.size())
    #     # raise Exception("Sorry, no numbers below zero")
    #     return (states, actions, rewards, next_states, dones)
        # return batch

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# class ReplayBuffer:
#     """Fixed-size buffer to store experience tuples with priorities."""

#     def __init__(self, action_size, buffer_size, batch_size, alpha=0.6):
#         """Initialize a ReplayBuffer object.
#         Params
#         ======
#             action_size (int): dimension of each action
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#             alpha (float): prioritization exponent, higher means more prioritized
#         """
#         self.action_size = action_size
#         self.memory = deque(maxlen=buffer_size)
#         self.priorities = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#         self.alpha = alpha

#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to memory."""
#         max_priority = max(self.priorities) if self.priorities else 1.0  # If empty, use default priority of 1.0
#         self.memory.append(self.experience(state, action, reward, next_state, done))
#         self.priorities.append(max_priority)  # Initialize with the maximum priority

#     def sample(self):
#         """Sample a batch of experiences from memory based on priorities."""
#         if len(self.memory) == 0:
#             return []
        
#         # Convert priorities to probabilities
#         scaled_priorities = np.array(self.priorities) ** self.alpha
#         sample_probs = scaled_priorities / sum(scaled_priorities)
        
#         # Sample experiences based on probabilities
#         sample_indices = np.random.choice(range(len(self.memory)), size=self.batch_size, p=sample_probs)
#         experiences = [self.memory[idx] for idx in sample_indices]

#         states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

#         return (states, actions, rewards, next_states, dones), sample_indices, sample_probs[sample_indices]

#     def update_priorities(self, indices, errors, offset=0.01):
#         """Update priorities of sampled experiences based on the TD errors."""
#         for idx, error in zip(indices, errors):
#             self.priorities[idx] = (abs(error) + offset) ** self.alpha

#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)
