from collections import namedtuple, deque
import random

from click import MissingParameter
import torch
import numpy as np

class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, buffer_size, batch_size, seed, device=None):

        self.memory = deque(maxlen=buffer_size)
        self.extra_rewards = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones, extra_rewards=None):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
            if extra_rewards is not None:
                self.extra_rewards.extend(extra_rewards)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)
            if extra_rewards is not None:
                self.extra_rewards.append(extra_rewards)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences, _ = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones, _ = self.separate_out_data_types(
                experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences

    def sample_with_extra(self, num_experiences=None, separate_out_data_types=True, reset=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences, extra_rewards = self.pick_experiences(num_experiences, reset=reset)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones, extra_rewards = self.separate_out_data_types(experiences, extra_rewards)
            return states, actions, rewards, next_states, dones, extra_rewards
        else:
            return experiences, extra_rewards

    def separate_out_data_types(self, experiences, extra_rewards=None):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
        if extra_rewards is not None:
            extra_rewards = torch.cat(extra_rewards)
        return states, actions, rewards, next_states, dones, extra_rewards

    def pick_experiences(self, num_experiences=None, reset=True):
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size

        if self.extra_rewards:
            indices = random.sample(range(len(self.memory)), k=batch_size)
            samples = [self.memory[idx] for idx in indices]
            extra_rewards = [torch.clone(self.extra_rewards[idx]) for idx in indices]
            # for idx in indices:
            #     self.extra_rewards[idx][0, 0] = 0
            return samples, extra_rewards
        else:
            return random.sample(self.memory, k=batch_size), None

    def __len__(self):
        return len(self.memory)
