"""
Discrete Soft Actor-Critic (SAC) implementation for Gymnasium discrete action environments.

This version uses a soft actor-critic update rule adapted for discrete actions.
"""

from __future__ import annotations

from typing import Any, Tuple

import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_6.networks import Policy
from torch.distributions import Categorical


class ReplayBuffer:
    def __init__(self, capacity: int = 1000000) -> None:
        self.capacity = capacity
        self.storage: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        transition = (state, action, reward, next_state, done)
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = np.random.choice(len(self.storage), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.storage[i] for i in indices]
        )
        return (
            torch.stack([torch.from_numpy(s).float() for s in states]),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack([torch.from_numpy(s).float() for s in next_states]),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.storage)


class QNetwork(nn.Module):
    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = int(np.prod(state_space.shape))
        self.hidden_size = hidden_size
        self.n_actions = action_space.n

        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SACAgent(AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: float | None = None,
        hidden_size: int = 128,
        seed: int = 0,
        replay_size: int = 100000,
        batch_size: int = 64,
        start_steps: int = 1000,
        update_every: int = 1,
        updates_per_step: int = 1,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_every = update_every
        self.updates_per_step = updates_per_step
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)

        self.policy = Policy(env.observation_space, env.action_space, hidden_size)
        self.q1 = QNetwork(env.observation_space, env.action_space, hidden_size)
        self.q2 = QNetwork(env.observation_space, env.action_space, hidden_size)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)

        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_critic
        )

        self.replay_buffer = ReplayBuffer(replay_size)

        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        action_count = env.action_space.n
        self.target_entropy = (
            target_entropy if target_entropy is not None else np.log(action_count)
        )

    @property
    def current_alpha(self) -> float:
        return float(self.log_alpha.exp().item()) if self.auto_alpha else self.alpha

    def predict_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[int, torch.Tensor]:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state_tensor).squeeze(0)
        if evaluate:
            action = int(torch.argmax(probs).item())
            return action, torch.tensor(0.0)
        dist = Categorical(probs)
        action = int(dist.sample().item())
        return action, dist.log_prob(torch.tensor(action))

    def compute_soft_value(
        self, q_values: torch.Tensor, log_probs: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        return (torch.exp(log_probs) * (q_values - alpha * log_probs)).sum(dim=1)

    def update(self) -> tuple[float, float, float] | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        with torch.no_grad():
            next_probs = self.policy(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            target_q1 = self.target_q1(next_states)
            target_q2 = self.target_q2(next_states)
            target_q = torch.min(target_q1, target_q2)
            next_value = self.compute_soft_value(
                target_q, next_log_probs, self.current_alpha
            )
            q_targets = rewards + self.gamma * (1.0 - dones) * next_value

        q1_values = self.q1(states)
        q2_values = self.q2(states)
        q1_pred = q1_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_pred = q2_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        critic_loss = F.mse_loss(q1_pred, q_targets) + F.mse_loss(q2_pred, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        probs = self.policy(states)
        log_probs = torch.log(probs + 1e-8)
        min_q = torch.min(self.q1(states), self.q2(states))
        policy_loss = (
            (probs * (self.current_alpha * log_probs - min_q)).sum(dim=1).mean()
        )

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.auto_alpha:
            entropy = -(probs * log_probs).sum(dim=1)
            alpha_loss = -(
                self.log_alpha * (entropy.detach() - self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_value = self.current_alpha
        else:
            alpha_loss = torch.tensor(0.0)
            alpha_value = self.alpha

        for param, target_param in zip(
            self.q1.parameters(), self.target_q1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.q2.parameters(), self.target_q2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return (
            float(policy_loss.item()),
            float(critic_loss.item()),
            float(alpha_loss.item()),
            float(alpha_value),
        )

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10000,
        eval_episodes: int = 5,
    ) -> list[dict[str, Any]]:
        eval_env = gym.make(self.env.spec.id)
        set_seed(eval_env, self.seed + 1000)
        state, _ = self.env.reset(seed=self.seed)
        step_count = 0
        records: list[dict[str, Any]] = []
        episode_steps = 0

        while step_count < total_steps:
            if step_count < self.start_steps:
                action = self.env.action_space.sample()
            else:
                action, _ = self.predict_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(state, action, float(reward), next_state, done)
            state = next_state
            step_count += 1
            episode_steps += 1

            if done:
                state, _ = self.env.reset()
                episode_steps = 0

            if step_count >= self.start_steps and step_count % self.update_every == 0:
                for _ in range(self.updates_per_step):
                    self.update()

            if step_count % eval_interval == 0:
                mean_return, std_return = self.evaluate(
                    eval_env, num_episodes=eval_episodes
                )
                records.append(
                    {
                        "eval_step": step_count,
                        "eval_return": mean_return,
                        "eval_std": std_return,
                        "alpha": self.current_alpha,
                    }
                )
                print(
                    f"[Eval] Step {step_count} AvgReturn {mean_return:.2f} ± {std_return:.2f} alpha={self.current_alpha:.4f}"
                )

        return records

    def evaluate(
        self, eval_env: gym.Env, num_episodes: int = 10
    ) -> Tuple[float, float]:
        self.policy.eval()
        returns = []
        with torch.no_grad():
            for _ in range(num_episodes):
                state, _ = eval_env.reset()
                done = False
                total_r = 0.0
                while not done:
                    action, _ = self.predict_action(state, evaluate=True)
                    state, r, term, trunc, _ = eval_env.step(action)
                    done = term or trunc
                    total_r += r
                returns.append(total_r)
        self.policy.train()
        return float(np.mean(returns)), float(np.std(returns))

    def save(self, path: str) -> None:
        data = {
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        torch.save(data, path)

    def load(self, path: str) -> None:
        data = torch.load(path)
        self.policy.load_state_dict(data["policy"])
        self.q1.load_state_dict(data["q1"])
        self.q2.load_state_dict(data["q2"])
        self.target_q1.load_state_dict(data["target_q1"])
        self.target_q2.load_state_dict(data["target_q2"])
        self.log_alpha = data["log_alpha"].requires_grad_(True)


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    agent = SACAgent(env)
    agent.train(total_steps=1000, eval_interval=250, eval_episodes=2)
