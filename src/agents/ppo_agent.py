"""
PPO agent for Battleship - policy-based, on-policy.
Uses actor-critic architecture with separate heads for placement and shooting.
Placement head: outputs probability distribution over placement actions.
Shooting head: outputs probability distribution over shooting actions (0-99).
"""

import random
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


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for shooting phase.
    Actor: outputs action probabilities (policy)
    Critic: outputs state value V(s)
    """

    def __init__(self, hidden: int = 256):
        super().__init__()
        # Shared feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 64 * 10 * 10 = 6400
        self.shared_fc = nn.Sequential(
            nn.Linear(64 * GRID_SIZE * GRID_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor head: outputs logits for actions
        self.actor = nn.Linear(hidden, NUM_CELLS)

        # Critic head: outputs state value
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (action_logits, state_value)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 10, 10)
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, 10, 10)
        x = x.float()
        features = self.conv(x)
        shared = self.shared_fc(features)
        action_logits = self.actor(shared)
        state_value = self.critic(shared)
        return action_logits, state_value


class PlacementActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for placement phase.
    Actor: outputs probability distribution over placement actions.
    Critic: outputs state value V(s).
    """

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
        self.shared_fc = nn.Sequential(
            nn.Linear(64 * GRID_SIZE * GRID_SIZE + NUM_SHIPS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor head: outputs logits for placement actions
        self.actor = nn.Linear(hidden, PLACEMENT_ACTIONS)

        # Critic head: outputs state value
        self.critic = nn.Linear(hidden, 1)

    def forward(
        self, x: torch.Tensor, ship_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (action_logits, state_value)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        features = self.conv(x)
        # ship_index: (B,) long -> one-hot (B, 5)
        onehot = F.one_hot(ship_index.clamp(0, NUM_SHIPS - 1), num_classes=NUM_SHIPS).float()
        combined = torch.cat([features, onehot], dim=1)
        shared = self.shared_fc(combined)
        action_logits = self.actor(shared)
        state_value = self.critic(shared)
        return action_logits, state_value


class PPOAgent:
    """
    PPO agent for Battleship. Policy-based, on-policy learning.
    Uses actor-critic architecture with separate networks for placement and shooting.
    """

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        device: str | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
            update_epochs: Number of PPO update epochs per batch
            device: Device to use ('cuda' or 'cpu')
            seed: Random seed
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Shooting phase networks
        self.shooting_actor_critic = ActorCriticNetwork().to(self.device)
        self.shooting_optimizer = torch.optim.Adam(
            self.shooting_actor_critic.parameters(), lr=lr
        )

        # Placement phase networks
        self.placement_actor_critic = PlacementActorCriticNetwork().to(self.device)
        self.placement_optimizer = torch.optim.Adam(
            self.placement_actor_critic.parameters(), lr=lr
        )

        # Trajectory buffers (on-policy: collect, update, clear)
        self.shooting_trajectory: list[dict] = []
        self.placement_trajectory: list[dict] = []

    def select_shooting_action(
        self,
        state: np.ndarray,
        valid_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float]:
        """
        Select shooting action using policy. Returns (action, log_prob).
        """
        valid_actions = np.where(valid_mask)[0]
        if len(valid_actions) == 0:
            return 0, 0.0

        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device)
            action_logits, _ = self.shooting_actor_critic(x)
            action_logits = action_logits.squeeze(0).cpu().numpy()

            # Mask invalid actions
            masked_logits = np.where(valid_mask, action_logits, -np.inf)
            probs = F.softmax(torch.tensor(masked_logits), dim=0).numpy()

            if deterministic:
                action = int(np.argmax(masked_logits))
            else:
                action = int(np.random.choice(len(probs), p=probs))

            # Compute log probability
            log_prob = np.log(probs[action] + 1e-8)

        return action, log_prob

    def select_placement_action(
        self,
        placement_obs: np.ndarray,
        ship_index: int,
        valid_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float]:
        """
        Select placement action using policy. Returns (action, log_prob).
        """
        valid_actions = np.where(valid_mask)[0]
        if len(valid_actions) == 0:
            return 0, 0.0

        with torch.no_grad():
            x = torch.tensor(placement_obs, dtype=torch.float32, device=self.device).unsqueeze(
                0
            )
            si = torch.tensor([ship_index], dtype=torch.long, device=self.device)
            action_logits, _ = self.placement_actor_critic(x, si)
            action_logits = action_logits.squeeze(0).cpu().numpy()

            # Mask invalid actions
            masked_logits = np.where(valid_mask, action_logits, -np.inf)
            probs = F.softmax(torch.tensor(masked_logits), dim=0).numpy()

            if deterministic:
                action = int(np.argmax(masked_logits))
            else:
                action = int(np.random.choice(len(probs), p=probs))

            # Compute log probability
            log_prob = np.log(probs[action] + 1e-8)

        return action, log_prob

    def store_shooting_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store a shooting phase transition."""
        self.shooting_trajectory.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "log_prob": log_prob,
                "value": value,
            }
        )

    def store_placement_transition(
        self,
        state: np.ndarray,
        ship_index: int,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
    ) -> None:
        """Store a placement phase transition."""
        self.placement_trajectory.append(
            {
                "state": state,
                "ship_index": ship_index,
                "action": action,
                "reward": reward,
                "log_prob": log_prob,
                "value": value,
            }
        )

    def compute_gae(
        self, rewards: list[float], values: list[float], dones: list[bool], next_value: float = 0.0
    ) -> tuple[list[float], list[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        Returns (advantages, returns).
        """
        advantages = []
        returns = []
        gae = 0.0
        next_val = next_value

        for t in reversed(range(len(rewards))):
            if dones[t]:
                # Terminal state: no next value
                delta = rewards[t] - values[t]
                gae = delta
                next_val = 0.0
            else:
                # Non-terminal: use next value
                delta = rewards[t] + self.gamma * next_val - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                next_val = values[t]

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update_shooting(self) -> dict[str, float]:
        """
        Update shooting policy using PPO on collected trajectory.
        Returns dict with loss statistics.
        """
        if len(self.shooting_trajectory) == 0:
            return {}

        # Extract trajectory data
        states = torch.tensor(
            np.array([t["state"] for t in self.shooting_trajectory]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            [t["action"] for t in self.shooting_trajectory], dtype=torch.long, device=self.device
        )
        rewards = [t["reward"] for t in self.shooting_trajectory]
        dones = [t["done"] for t in self.shooting_trajectory]
        old_log_probs = torch.tensor(
            [t["log_prob"] for t in self.shooting_trajectory],
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.tensor(
            [t["value"] for t in self.shooting_trajectory],
            dtype=torch.float32,
            device=self.device,
        )

        # Compute next value for GAE (only if last step is not terminal)
        next_value = 0.0
        if not dones[-1]:
            with torch.no_grad():
                last_state = torch.tensor(
                    self.shooting_trajectory[-1]["next_state"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                _, next_value = self.shooting_actor_critic(last_state)
                next_value = next_value.item()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values.tolist(), dones, next_value)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.update_epochs):
            # Get current policy predictions
            action_logits, values = self.shooting_actor_critic(states)
            action_probs = F.softmax(action_logits, dim=1)
            action_dist = torch.distributions.Categorical(action_probs)

            # Compute new log probs
            new_log_probs = action_dist.log_prob(actions)

            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)

            # Entropy bonus
            entropy = action_dist.entropy().mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Update
            self.shooting_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.shooting_actor_critic.parameters(), self.max_grad_norm
            )
            self.shooting_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        # Clear trajectory (on-policy: use once)
        self.shooting_trajectory.clear()

        return {
            "policy_loss": total_policy_loss / self.update_epochs,
            "value_loss": total_value_loss / self.update_epochs,
            "entropy": total_entropy / self.update_epochs,
        }

    def update_placement(self, episode_return: float) -> dict[str, float]:
        """
        Update placement policy using PPO on collected trajectory.
        Uses Monte Carlo return (episode_return) for all placement steps.
        Returns dict with loss statistics.
        """
        if len(self.placement_trajectory) == 0:
            return {}

        # Extract trajectory data
        states = torch.tensor(
            np.array([t["state"] for t in self.placement_trajectory]),
            dtype=torch.float32,
            device=self.device,
        )
        ship_indices = torch.tensor(
            [t["ship_index"] for t in self.placement_trajectory],
            dtype=torch.long,
            device=self.device,
        )
        actions = torch.tensor(
            [t["action"] for t in self.placement_trajectory], dtype=torch.long, device=self.device
        )
        old_log_probs = torch.tensor(
            [t["log_prob"] for t in self.placement_trajectory],
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.tensor(
            [t["value"] for t in self.placement_trajectory],
            dtype=torch.float32,
            device=self.device,
        )

        # Use episode return as target (Monte Carlo)
        returns = torch.full_like(old_values, episode_return, device=self.device)
        advantages = returns - old_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.update_epochs):
            # Get current policy predictions
            action_logits, values = self.placement_actor_critic(states, ship_indices)
            action_probs = F.softmax(action_logits, dim=1)
            action_dist = torch.distributions.Categorical(action_probs)

            # Compute new log probs
            new_log_probs = action_dist.log_prob(actions)

            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)

            # Entropy bonus
            entropy = action_dist.entropy().mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Update
            self.placement_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.placement_actor_critic.parameters(), self.max_grad_norm
            )
            self.placement_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        # Clear trajectory (on-policy: use once)
        self.placement_trajectory.clear()

        return {
            "policy_loss": total_policy_loss / self.update_epochs,
            "value_loss": total_value_loss / self.update_epochs,
            "entropy": total_entropy / self.update_epochs,
        }

    def save(self, path: str | Path) -> None:
        """Save agent state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "shooting_actor_critic": self.shooting_actor_critic.state_dict(),
                "shooting_optimizer": self.shooting_optimizer.state_dict(),
                "placement_actor_critic": self.placement_actor_critic.state_dict(),
                "placement_optimizer": self.placement_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load agent state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.shooting_actor_critic.load_state_dict(ckpt["shooting_actor_critic"])
        if "shooting_optimizer" in ckpt:
            self.shooting_optimizer.load_state_dict(ckpt["shooting_optimizer"])
        self.placement_actor_critic.load_state_dict(ckpt["placement_actor_critic"])
        if "placement_optimizer" in ckpt:
            self.placement_optimizer.load_state_dict(ckpt["placement_optimizer"])
