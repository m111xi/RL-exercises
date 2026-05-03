from __future__ import annotations

from typing import Any, DefaultDict, Literal

from collections import defaultdict

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_3.epsilon_greedy_policy import EpsilonGreedyPolicy

State = Any


class TDLambdaAgent(AbstractAgent):
    """Tabular TD(lambda) agent with eligibility traces."""

    def __init__(
        self,
        env: gym.Env,
        policy: EpsilonGreedyPolicy,
        alpha: float = 0.5,
        gamma: float = 1.0,
        lambda_: float = 0.9,
        algorithm: Literal["sarsa", "qlearning"] = "sarsa",
    ) -> None:
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert 0 < alpha <= 1, "Learning rate has to be in (0, 1]"
        assert 0 <= lambda_ <= 1, "Lambda must be in [0, 1]"
        assert algorithm in ["sarsa", "qlearning"], (
            "algorithm must be 'sarsa' or 'qlearning'"
        )

        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_
        self.algorithm = algorithm
        self.n_actions = env.action_space.n

        self.Q: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )
        self.e: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )
        self.policy = policy

    def predict_action(
        self, state: np.ndarray, info: dict = {}, evaluate: bool = False
    ) -> Any:  # type: ignore
        return self.policy(self.Q, state, evaluate=evaluate), info

    def save(self, path: str) -> Any:  # type: ignore
        np.save(path, dict(self.Q))

    def load(self, path) -> Any:  # type: ignore
        loaded_q = np.load(path, allow_pickle=True).item()
        self.Q = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float),
            loaded_q,
        )

    def reset_traces(self) -> None:
        self.e = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

    def update_agent(self, batch) -> float:  # type: ignore
        state, action, reward, next_state, done, _ = batch[0]

        if self.algorithm == "sarsa":
            next_action, _ = self.predict_action(next_state, evaluate=False)
            target = (
                reward + self.gamma * self.Q[next_state][next_action]
                if not done
                else reward
            )
        else:  # qlearning
            target = (
                reward + self.gamma * np.max(self.Q[next_state]) if not done else reward
            )

        delta = target - self.Q[state][action]

        if self.algorithm == "sarsa":
            # SARSA(lambda): on-policy traces
            self.e[state][action] += 1.0
            for s, trace in self.e.items():
                self.Q[s] += self.alpha * delta * trace
                self.e[s] *= self.gamma * self.lambda_
        else:  # qlearning - Watkins' Q(lambda)
            # Reset traces if action is not greedy (off-policy)
            if action != np.argmax(self.Q[state]):
                self.reset_traces()
            else:
                self.e[state][action] += 1.0
                for s, trace in self.e.items():
                    self.Q[s] += self.alpha * delta * trace
                    self.e[s] *= self.gamma * self.lambda_

        if done:
            self.reset_traces()

        return self.Q[state][action]
