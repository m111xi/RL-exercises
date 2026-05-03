from __future__ import annotations

from typing import Any

import warnings

import numpy as np
from rich import print as printr
from rl_exercises.agent import AbstractAgent
from rl_exercises.environments import MarsRover


class PolicyIteration(AbstractAgent):
    """
    Policy Iteration Agent.

    This agent performs standard tabular policy iteration on an environment
    with known transition dynamics and rewards. The policy is evaluated and
    improved until convergence.

    Parameters
    ----------
    env : MarsRover
        Environment instance. This class is designed specifically for the MarsRover env.
    gamma : float, optional
        Discount factor for future rewards, by default 0.9.
    seed : int, optional
        Random seed for policy initialization, by default 333.
    filename : str, optional
        Path to save/load the policy, by default "policy.npy".
    """

    def __init__(
        self,
        env: MarsRover,
        gamma: float = 0.9,
        seed: int = 333,
        filename: str = "policy.npy",
        **kwargs: dict,
    ) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped  # type: ignore[assignment]
        self.env = env
        self.seed = seed
        self.filename = filename

        super().__init__(**kwargs)

        self.n_obs = self.env.observation_space.n  # type: ignore[attr-defined]
        self.n_actions = self.env.action_space.n  # type: ignore[attr-defined]

        # Get the MDP components (states, actions, transitions, rewards)
        self.S = self.env.states
        self.A = self.env.actions
        self.T = self.env.get_transition_matrix()
        self.R = self.env.rewards
        self.gamma = gamma
        self.R_sa = self.env.get_reward_per_action()

        # Initialize policy and Q-values
        rng = np.random.default_rng(seed=self.seed)
        self.pi: np.ndarray = rng.integers(0, self.n_actions, self.n_obs)
        self.Q = np.zeros_like(self.R_sa)

        self.policy_fitted: bool = False
        self.steps: int = 0

    def predict_action(  # type: ignore[override]
        self, observation: int, info: dict | None = None, evaluate: bool = False
    ) -> tuple[int, dict]:
        """
        Predict an action using the current policy.

        Parameters
        ----------
        observation : int
            The current observation/state.
        info : dict or None, optional
            Additional info passed during prediction (unused).
        evaluate : bool, optional
            Evaluation mode toggle (unused here), by default False.

        Returns
        -------
        tuple[int, dict]
            The selected action and an empty info dictionary.
        """
        return self.pi[observation], {}

    def update_agent(self, *args: tuple, **kwargs: dict) -> None:
        """Run policy iteration to compute the optimal policy and state-action values."""
        if not self.policy_fitted:
            # TODO: Call policy iteration with initialized values
            printr("Initial policy: ", self.pi)
            Q, pi, steps = policy_iteration(
                self.Q, self.pi, (self.S, self.A, self.T, self.R_sa, self.gamma)
            )
            self.Q = Q
            self.pi = pi
            self.steps = steps
            printr("Q: ", self.Q)
            printr("Final policy: ", self.pi)
            printr("Policy iteration steps:", self.steps)
            self.policy_fitted = True

    def save(self, *args: tuple[Any], **kwargs: dict) -> None:
        """
        Save the learned policy to a `.npy` file.

        Raises
        ------
        Warning
            If the policy has not yet been fitted.
        """
        if self.policy_fitted:
            np.save(self.filename, np.array(self.pi))
        else:
            warnings.warn("Tried to save policy but policy is not fitted yet.")

    def load(self, *args: tuple[Any], **kwargs: dict) -> np.ndarray:
        """
        Load the policy from file.

        Returns
        -------
        np.ndarray
            The loaded policy array.
        """
        self.pi = np.load(self.filename)
        self.policy_fitted = True
        return self.pi


def policy_evaluation(
    pi: np.ndarray,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Perform policy evaluation for a fixed policy.

    Parameters
    ----------
    pi : np.ndarray
        The current policy (array of actions).
    T : np.ndarray
        Transition probabilities T[s, a, s'].
    R_sa : np.ndarray
        Reward matrix R[s, a].
    gamma : float
        Discount factor.
    epsilon : float, optional
        Convergence threshold, by default 1e-8.

    Returns
    -------
    np.ndarray
        The evaluated value function V[s] for all states.
    """
    nS = R_sa.shape[0]
    V = np.zeros(nS)
    while True:
        delta = 0
        V_new = np.copy(V)

        for s in range(nS):
            a = pi[s]  # Aktion laut Policy

            value = 0
            for s_next in range(nS):
                value += T[s, a, s_next] * (R_sa[s, a] + gamma * V[s_next])

            V_new[s] = value
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < epsilon:
            break

    return V


def policy_improvement(
    V: np.ndarray,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Improve the current policy based on the value function.

    Parameters
    ----------
    V : np.ndarray
        Current value function.
    T : np.ndarray
        Transition probabilities T[s, a, s'].
    R_sa : np.ndarray
        Reward matrix R[s, a].
    gamma : float
        Discount factor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Q-function and the improved policy.
    """
    nS, nA = R_sa.shape
    Q = np.zeros((nS, nA))
    pi_new = None
    # TODO: implement Policy Improvement for all states
    for i in range(nS):
        for j in range(nA):
            Q_value = 0
            for k in range(nS):
                Q_value += T[i, j, k] * (R_sa[i, j] + gamma * V[k])
            Q[i, j] = Q_value

    pi_new = np.zeros(nS, dtype=int)
    for i in range(nS):
        maxQValue = Q[i, 0]
        maxQValueIndex = 0
        for j in range(nA):
            if Q[i, j] > maxQValue:
                maxQValue = Q[i, j]
                maxQValueIndex = j
        pi_new[i] = maxQValueIndex
    return Q, pi_new


def policy_iteration(
    Q: np.ndarray,
    pi: np.ndarray,
    MDP: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Full policy iteration loop until convergence.

    Parameters
    ----------
    Q : np.ndarray
        Initial Q-table (can be zeros).
    pi : np.ndarray
        Initial policy.
    MDP : tuple
        A tuple (S, A, T, R_sa, gamma) representing the MDP.
    epsilon : float, optional
        Convergence threshold for value updates, by default 1e-8.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        Final Q-table, final policy, and number of improvement steps.
    """
    S, A, T, R_sa, gamma = MDP

    # TODO: Combine evaluation and improvement in a loop.
    steps = 0
    while True:
        V = policy_evaluation(pi, T, R_sa, gamma, epsilon)
        Q, new_pi = policy_improvement(V, T, R_sa, gamma)
        if np.array_equal(new_pi, pi):
            break
        pi = new_pi
        steps += 1
    return Q, pi, steps


if __name__ == "__main__":
    algo = PolicyIteration(env=MarsRover())
    algo.update_agent()
