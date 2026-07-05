"""
Simple RND-DQN agent (idea taken from week_7).

Basically a DQN with epsilon-greedy + target network. On top of that we add
an RND bonus: one fixed target net and one trainable predictor net. The
prediction error (predictor vs target) is the intrinsic reward -> new/unseen
states give a big error -> agent explores more.

Kept this short on purpose so you can actually read the whole thing.
"""

from __future__ import annotations

import random
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def set_seed(env: gym.Env, seed: int) -> None:
    # seed everything so runs are reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)


def mlp(in_dim: int, out_dim: int, hidden: int = 128) -> nn.Sequential:
    # small helper: standard 2-hidden-layer MLP
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.data: list = []
        self.pos = 0

    def add(self, s, a, r, s2, done) -> None:
        item = (s, a, r, s2, done)
        # overwrite oldest once buffer is full (ring buffer)
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.int64),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(np.array(s2), dtype=torch.float32),
            torch.as_tensor(done, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.data)


class RNDDQNAgent:
    """DQN + RND exploration bonus."""

    def __init__(
        self,
        env: gym.Env,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay: int = 2000,
        target_update_freq: int = 500,
        buffer_capacity: int = 10000,
        rnd_reward_weight: float = 0.1,
        rnd_lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = epsilon_start
        self.eps_final = epsilon_final
        self.eps_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.rnd_weight = rnd_reward_weight

        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = env.action_space.n
        self.obs_dim = obs_dim

        # main Q-net + target Q-net (target is a delayed copy)
        self.q = mlp(obs_dim, n_actions)
        self.q_target = mlp(obs_dim, n_actions)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        # RND: fixed random target + trainable predictor
        self.rnd_target = mlp(obs_dim, 64)
        self.rnd_predictor = mlp(obs_dim, 64)
        for p in self.rnd_target.parameters():
            p.requires_grad = False  # target stays frozen, never trained
        self.rnd_opt = torch.optim.Adam(self.rnd_predictor.parameters(), lr=rnd_lr)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.total_steps = 0

    def epsilon(self) -> float:
        # linear decay from eps_start to eps_final
        frac = min(1.0, self.total_steps / self.eps_decay)
        return self.eps_start + frac * (self.eps_final - self.eps_start)

    def predict_action(self, state, evaluate: bool = False) -> int:
        # epsilon-greedy (no exploration when evaluating)
        if not evaluate and random.random() < self.epsilon():
            return self.env.action_space.sample()
        with torch.no_grad():
            s = torch.as_tensor(np.array(state), dtype=torch.float32).flatten()
            return int(self.q(s).argmax().item())

    def intrinsic_reward(self, next_state) -> float:
        # how "surprising" is this state = RND prediction error
        with torch.no_grad():
            s = torch.as_tensor(np.array(next_state), dtype=torch.float32).flatten()
            err = (self.rnd_predictor(s) - self.rnd_target(s)).pow(2).mean()
        return float(err.item())

    def update(self) -> None:
        # wait until we have enough samples
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, s2, done = self.buffer.sample(self.batch_size)

        # train the RND predictor to match the frozen target
        pred = self.rnd_predictor(s2)
        with torch.no_grad():
            target = self.rnd_target(s2)
        rnd_loss = (pred - target).pow(2).mean()
        self.rnd_opt.zero_grad()
        rnd_loss.backward()
        self.rnd_opt.step()

        # add the intrinsic bonus to the extrinsic reward
        with torch.no_grad():
            bonus = (self.rnd_predictor(s2) - self.rnd_target(s2)).pow(2).mean(dim=1)
        r_total = r + self.rnd_weight * bonus

        # normal DQN update
        q_vals = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s2).max(dim=1).values
            td_target = r_total + self.gamma * q_next * (1 - done)
        loss = nn.functional.mse_loss(q_vals, td_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # every now and then copy weights into the target net
        if self.total_steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())


def train(
    env: gym.Env,
    agent: RNDDQNAgent,
    total_steps: int,
    seed: int = 0,
    eval_env: Optional[gym.Env] = None,
    eval_every: int = 2000,
    n_eval_episodes: int = 5,
) -> list:
    """Train the agent, return a list of (step, eval_return)."""
    set_seed(env, seed)
    state, _ = env.reset(seed=seed)
    history = []

    for step in range(total_steps):
        agent.total_steps = step
        action = agent.predict_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.add(
            np.array(state).flatten(), action, reward,
            np.array(next_state).flatten(), float(done),
        )
        agent.update()
        state = next_state
        if done:
            state, _ = env.reset()

        # optional: log eval performance during training
        if eval_env is not None and step % eval_every == 0:
            ret = evaluate(eval_env, agent, n_eval_episodes, seed)
            history.append((step, ret))

    return history


def evaluate(env: gym.Env, agent: RNDDQNAgent, episodes: int, seed: int = 0) -> float:
    # run a few greedy episodes and average the return
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        total = 0.0
        while not done:
            action = agent.predict_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))
