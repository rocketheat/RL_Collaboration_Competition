# individual network settings for each actor + critic pair
# see networkforall for details

from torch.optim import Adam
import numpy as np

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torch.autograd import Variable

from collections import namedtuple, deque
import random
import copy

from model import ActorNN, CriticNN

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class Critic:

    def __init__(self,
        device,
        state_size, action_size, random_seed,
        gamma, TAU, lr, weight_decay,
        checkpoint_folder = './'):

        self.DEVICE = device

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Hyperparameters
        self.GAMMA = gamma
        self.TAU = TAU
        self.LR = lr
        self.WEIGHT_DECAY = weight_decay

        self.CHECKPOINT_FOLDER = checkpoint_folder

        # Critic Network (w/ Target Network)
        self.local = CriticNN(state_size, action_size, random_seed).to(self.DEVICE)
        self.target = CriticNN(state_size, action_size, random_seed).to(self.DEVICE)
        self.optimizer = optim.Adam(self.local.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)

        self.checkpoint_full_name = self.CHECKPOINT_FOLDER + 'checkpoint_critic.pth'
        if os.path.isfile(self.checkpoint_full_name):
            self.local.load_state_dict(torch.load(self.checkpoint_full_name))
            self.target.load_state_dict(torch.load(self.checkpoint_full_name))

    def step(self, actor, memory):
        # Learn, if enough samples are available in memory
        experiences = memory.sample()
        if not experiences:
            return

        self.learn(actor, experiences)

    def learn(self, actor, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = actor.target(next_states)
        Q_targets_next = self.target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.local.parameters(), 1)
        self.optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = actor.local(states)
        actor_loss = - self.local(states, actions_pred).mean()
        # Minimize the loss
        actor.optimizer.zero_grad()
        actor_loss.backward()
        actor.optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.local, self.target)
        self.soft_update(actor.local, actor.target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        tau = self.TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def checkpoint(self):
        torch.save(self.local.state_dict(), self.checkpoint_full_name)


class Actor:

    def __init__(self,
        device,
        key,
        state_size, action_size, random_seed,
        memory, noise,
        lr, weight_decay,
        checkpoint_folder = './'):

        self.DEVICE = device

        self.KEY = key

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Hyperparameters
        self.LR = lr
        self.WEIGHT_DECAY = weight_decay

        self.CHECKPOINT_FOLDER = checkpoint_folder

        # Actor Network (w/ Target Network)
        self.local = ActorNN(state_size, action_size, random_seed).to(self.DEVICE)
        self.target = ActorNN(state_size, action_size, random_seed).to(self.DEVICE)
        self.optimizer = optim.Adam(self.local.parameters(), lr=self.LR)

        self.checkpoint_full_name = self.CHECKPOINT_FOLDER + 'checkpoint_actor_' + str(self.KEY) + '.pth'
        if os.path.isfile(self.checkpoint_full_name):
            self.local.load_state_dict(torch.load(self.checkpoint_full_name))
            self.target.load_state_dict(torch.load(self.checkpoint_full_name))

        # Replay memory
        self.memory = memory

        # Noise process
        self.noise = noise

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.DEVICE)

        self.local.eval()
        with torch.no_grad():
            action = self.local(state).cpu().data.numpy()
        self.local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

    def reset(self):
        self.noise.reset()

    def checkpoint(self):
        torch.save(self.local.state_dict(), self.checkpoint_full_name)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)

        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx

        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.DEVICE = device

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if len(self.memory) <= self.batch_size:
            return None

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
