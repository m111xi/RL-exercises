from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_5.policy_gradient import set_seed


class ActorNetwork(nn.Module):
    def __init__(
        self, observation_dim: int, action_dim: int, hidden_size: int = 256
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    def __init__(
        self, observation_dim: int, action_dim: int, hidden_size: int = 256
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(observation_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class DDPGAgent(AbstractAgent):
    """Basic DDPG implementation for continuous control."""

    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        noise_std: float = 0.2,
        noise_decay: float = 0.995,
        min_noise: float = 0.01,
        hidden_size: int = 256,
        seed: int = 0,
    ) -> None:
        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.min_noise = min_noise

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        self.actor = ActorNetwork(obs_dim, act_dim, hidden_size)
        self.actor_target = ActorNetwork(obs_dim, act_dim, hidden_size)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(obs_dim, act_dim, hidden_size)
        self.critic_target = CriticNetwork(obs_dim, act_dim, hidden_size)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.total_updates = 0

    def _scale_action(self, action: torch.Tensor) -> np.ndarray:
        action = action.cpu().numpy()
        scaled = self.action_low + (action + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )
        return np.clip(scaled, self.action_low, self.action_high)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32)

    def predict_action(
        self, state: np.ndarray, info: Dict[str, Any] = {}, evaluate: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        state_t = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t)
        action = action.squeeze(0)
        if not evaluate:
            noise = np.random.normal(scale=self.noise_std, size=action.shape)
            action = torch.clamp(
                action + torch.tensor(noise, dtype=torch.float32), -1.0, 1.0
            )
        return self._scale_action(action), {}

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done, info)

    def update_agent(
        self,
        training_batch: Optional[
            List[Tuple[Any, Any, float, Any, bool, Dict[str, Any]]]
        ] = None,
    ) -> float:
        if training_batch is None:
            if len(self.replay_buffer) < self.batch_size:
                return 0.0
            training_batch = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones, _ = zip(*training_batch)
        states_t = self._to_tensor(np.stack(states))
        actions_t = self._to_tensor(np.stack(actions))
        rewards_t = self._to_tensor(np.array(rewards, dtype=np.float32))
        next_states_t = self._to_tensor(np.stack(next_states))
        dones_t = self._to_tensor(np.array(dones, dtype=np.float32))

        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            td_target = rewards_t + self.gamma * (1.0 - dones_t) * target_q

        q_values = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(q_values, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.noise_std = max(self.noise_std * self.noise_decay, self.min_noise)
        self.total_updates += 1
        return float(critic_loss.item())

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
