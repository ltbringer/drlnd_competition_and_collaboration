import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from ddpg.actor import Actor
from ddpg.critic import Critic
from replay_buffer.memory import ReplayBuffer
from utils.torch_device_select import torch_device
from constants import constants


device = torch_device()


class Agent():
    """
    A DDPG Agent implementation
    """

    def __init__(
            self,
            state_size,
            action_size,
            random_seed,
            buffer_size=constants.BUFFER_SIZE,
            batch_size=constants.BATCH_SIZE,
            gamma=constants.GAMMA,
            tau=constants.TAU,
            actor_lr=constants.ACTOR_LR,
            critic_lr=constants.CRITIC_LR,
            weight_decay=constants.WEIGHT_DECAY,
            actor_layer_1_nodes=constants.FC1_UNITS,
            critic_layer_1_nodes=constants.FC1_UNITS,
            actor_layer_2_nodes=constants.FC2_UNITS,
            critic_layer_2_nodes=constants.FC2_UNITS
    ):
        """
        :param state_size:
        :param action_size:
        :param random_seed:
        :param buffer_size:
        :param batch_size:
        :param gamma:
        :param tau:
        :param actor_lr:
        :param critic_lr:
        :param weight_decay:
        :param actor_layer_1_nodes:
        :param critic_layer_1_nodes:
        :param actor_layer_2_nodes:
        :param critic_layer_2_nodes:
        """
        self.buffer_size    = buffer_size
        self.batch_size     = batch_size
        self.state_size     = state_size
        self.action_size    = action_size
        self.gamma          = gamma
        self.tau            = tau
        self.seed           = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local        = Actor(
            state_size,
            action_size,
            seed=random_seed,
            fc1_units=actor_layer_1_nodes,
            fc2_units=actor_layer_2_nodes
        ).to(device)

        self.actor_target       = Actor(
            state_size,
            action_size,
            seed=random_seed,
            fc1_units=actor_layer_1_nodes,
            fc2_units=actor_layer_2_nodes
        ).to(device)

        self.actor_optimizer    = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

        # Critic Network (w/ Target Network)
        self.critic_local       = Critic(
            state_size,
            action_size,
            seed=random_seed,
            fcs1_units=critic_layer_1_nodes,
            fc2_units=critic_layer_2_nodes
        ).to(device)

        self.critic_target      = Critic(
            state_size,
            action_size,
            seed=random_seed,
            fcs1_units=critic_layer_1_nodes,
            fc2_units=critic_layer_2_nodes
        ).to(device)

        self.critic_optimizer   = optim.Adam(self.critic_local.parameters(), lr=critic_lr, weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        self.memory.add(state, action, reward, next_state, done)

        # if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, add_noise=True, is_training=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        if is_training:
            self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def load_saved_actor_model(self, model_path):
        actor_saved_model = torch.load(model_path)
        self.actor_local.load_state_dict(actor_saved_model)
        return self

    def load_saved_critic_model(self, model_path):
        critic_saved_model = torch.load(model_path)
        self.critic_local.load_state_dict(critic_saved_model)
        return self


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
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
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state