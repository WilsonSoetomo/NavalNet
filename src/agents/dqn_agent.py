"""
DQN agent for Battleship - value-based, learns Q(s,a) for both placement and shooting.
Placement head: learns from episode return (Monte Carlo style). Shooting head: standard DQN.
"""

import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from battleship.constants import GRID_SIZE, HORIZONTAL, NUM_CELLS, SHIP_SIZES, VERTICAL

# Placement: row*10*2 + col*2 + orient -> 10*10*2 = 200 actions per ship
PLACEMENT_ACTIONS = GRID_SIZE * GRID_SIZE * 2
NUM_SHIPS = len(SHIP_SIZES)


class DQNNetwork(nn.Module):
    """Q-network: 10x10 observation -> Q-values for 100 actions (shooting)."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        # Input: (batch, 1, 10, 10) - add channel dim for 2D grid
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 64 * 10 * 10 = 6400
        self.fc = nn.Sequential(
            nn.Linear(64 * GRID_SIZE * GRID_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_CELLS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 10, 10)
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, 10, 10)
        x = x.float()
        features = self.conv(x)
        return self.fc(features)


class PlacementDQNNetwork(nn.Module):
    """Placement Q-network: 10x10 agent board + ship_index -> Q-values for 200 placement actions."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 64*10*10 + ship_onehot(5) = 6405
        self.fc = nn.Sequential(
            nn.Linear(64 * GRID_SIZE * GRID_SIZE + NUM_SHIPS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, PLACEMENT_ACTIONS),
        )

    def forward(self, x: torch.Tensor, ship_index: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        features = self.conv(x)
        # ship_index: (B,) long -> one-hot (B, 5)
        onehot = F.one_hot(ship_index.clamp(0, NUM_SHIPS - 1), num_classes=NUM_SHIPS).float()
        combined = torch.cat([features, onehot], dim=1)
        return self.fc(combined)


class ReplayBuffer:
    """Experience replay for shooting DQN."""

    def __init__(self, capacity: int):
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mask: np.ndarray,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        masks = torch.tensor(np.array([b[5] for b in batch]), dtype=torch.bool)
        return states, actions, rewards, next_states, dones, masks

    def __len__(self) -> int:
        return len(self.buffer)


class PlacementReplayBuffer:
    """Experience replay for placement DQN. Stores (state, ship_index, action, reward)."""

    def __init__(self, capacity: int):
        self.buffer: deque[
            tuple[np.ndarray, int, int, float]
        ] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        ship_index: int,
        action: int,
        reward: float,
    ) -> None:
        self.buffer.append((state, ship_index, action, reward))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        ship_indices = torch.tensor([b[1] for b in batch], dtype=torch.long)
        actions = torch.tensor([b[2] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        return states, ship_indices, actions, rewards

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent for Battleship. Placement head and shooting head both learn (value-based).
    """

    def __init__(
        self,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        placement_buffer_size: int = 10_000,
        placement_batch_size: int = 32,
        device: str | None = None,
        seed: int | None = None,
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.placement_batch_size = placement_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Shooting head
        self.policy_net = DQNNetwork().to(self.device)
        self.target_net = DQNNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self._steps = 0

        # Placement head
        self.placement_policy_net = PlacementDQNNetwork().to(self.device)
        self.placement_target_net = PlacementDQNNetwork().to(self.device)
        self.placement_target_net.load_state_dict(self.placement_policy_net.state_dict())
        self.placement_optimizer = torch.optim.Adam(
            self.placement_policy_net.parameters(), lr=lr
        )
        self.placement_replay_buffer = PlacementReplayBuffer(placement_buffer_size)
        self._placement_steps = 0

    def select_placement_action(
        self,
        placement_obs: np.ndarray,
        ship_index: int,
        valid_mask: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Select placement action (row*10*2 + col*2 + orient) using placement Q-network."""
        valid_actions = np.where(valid_mask)[0]
        if len(valid_actions) == 0:
            return 0

        if not deterministic and random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            x = torch.tensor(
                placement_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            si = torch.tensor([ship_index], dtype=torch.long, device=self.device)
            q = self.placement_policy_net(x, si).squeeze(0).cpu().numpy()
            q_masked = np.where(valid_mask, q, -np.inf)
            return int(np.argmax(q_masked))

    def select_shooting_action(
        self,
        state: np.ndarray,
        valid_mask: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Select shooting action (0-99). Uses epsilon-greedy with action masking.
        """
        valid_actions = np.where(valid_mask)[0]
        if len(valid_actions) == 0:
            return 0

        if not deterministic and random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device)
            q = self.policy_net(x).squeeze(0).cpu().numpy()
            q_masked = np.where(valid_mask, q, -np.inf)
            return int(np.argmax(q_masked))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mask: np.ndarray,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done, mask)

    def store_placement_transition(
        self,
        state: np.ndarray,
        ship_index: int,
        action: int,
        reward: float,
    ) -> None:
        """Store a placement (s, ship_idx, a) with episode return as reward (filled later or at end)."""
        self.placement_replay_buffer.push(state, ship_index, action, reward)

    def update(self) -> float | None:
        """Perform one DQN update. Returns loss or None if not enough samples."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_masks = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_masks = next_masks.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q = torch.where(next_masks, next_q, torch.tensor(-1e9, device=self.device))
            next_q_max = next_q.max(1)[0]
            targets = rewards + self.gamma * next_q_max * (1 - dones)

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

    def update_placement(self) -> float | None:
        """
        One placement DQN update. Uses Monte Carlo target: Q(s,a) ~ episode return.
        Returns loss or None if not enough samples.
        """
        if len(self.placement_replay_buffer) < self.placement_batch_size:
            return None

        states, ship_indices, actions, rewards = self.placement_replay_buffer.sample(
            self.placement_batch_size
        )
        states = states.to(self.device)
        ship_indices = ship_indices.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        q_values = self.placement_policy_net(states, ship_indices).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        loss = F.mse_loss(q_values, rewards)
        self.placement_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.placement_policy_net.parameters(), 1.0
        )
        self.placement_optimizer.step()

        self._placement_steps += 1
        if self._placement_steps % self.target_update_freq == 0:
            self.placement_target_net.load_state_dict(
                self.placement_policy_net.state_dict()
            )
        return loss.item()

    def decay_epsilon(self) -> None:
        """Decay exploration (call at end of episode if not done in update)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self._steps,
                "placement_policy_net": self.placement_policy_net.state_dict(),
                "placement_target_net": self.placement_target_net.state_dict(),
                "placement_optimizer": self.placement_optimizer.state_dict(),
                "placement_steps": self._placement_steps,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self._steps = ckpt.get("steps", 0)
        if "placement_policy_net" in ckpt:
            self.placement_policy_net.load_state_dict(ckpt["placement_policy_net"])
            self.placement_target_net.load_state_dict(ckpt["placement_target_net"])
            if "placement_optimizer" in ckpt:
                self.placement_optimizer.load_state_dict(ckpt["placement_optimizer"])
            self._placement_steps = ckpt.get("placement_steps", 0)
