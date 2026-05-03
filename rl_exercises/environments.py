"""GridCore Env taken from https://github.com/automl/TabularTempoRL/"""

from __future__ import annotations

from typing import Any, SupportsFloat

from collections.abc import Sequence

import gymnasium as gym
import numpy as np


class MarsRover(gym.Env):
    """
    Simple Environment for a Mars Rover that can move in a 1D Space.

    The rover starts at position 2 and moves left or right based on discrete actions.
    The environment is stochastic: with a probability defined by a transition matrix,
    the action may be flipped. Each cell has an associated reward.

    Actions
    -------
    Discrete(2):
    - 0: go left
    - 1: go right

    Observations
    ------------
    Discrete(n): The current position of the rover (int).

    Reward
    ------
    Depends on the resulting cell after action is taken.

    Start/Reset State
    -----------------
    Always starts at position 2.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        transition_probabilities: np.ndarray = np.ones((5, 2)),
        rewards: list[float] = [1, 0, 0, 0, 10],
        horizon: int = 10,
        seed: int | None = None,
    ):
        """
        Initialize the Mars Rover environment.

        Parameters
        ----------
        transition_probabilities : np.ndarray, optional
            A (num_states, 2) array specifying the probability of actions being followed.
        rewards : list of float, optional
            Rewards assigned to each position, by default [1, 0, 0, 0, 10].
        horizon : int, optional
            Maximum number of steps per episode, by default 10.
        seed : int or None, optional
            Random seed for reproducibility, by default None.
        """
        self.rng = np.random.default_rng(seed)

        self.rewards = list(rewards)
        self.P = np.array(transition_probabilities)
        self.horizon = int(horizon)
        self.current_steps = 0
        self.position = 2  # start at middle

        # spaces
        n = self.P.shape[0]
        self.observation_space = gym.spaces.Discrete(n)
        self.action_space = gym.spaces.Discrete(2)

        # helpers
        self.states = np.arange(n)
        self.actions = np.arange(2)

        # transition matrix
        self.transition_matrix = self.T = self.get_transition_matrix()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Seed for environment reset (unused).
        options : dict, optional
            Additional reset options (unused).

        Returns
        -------
        state : int
            Initial state (always 2).
        info : dict
            An empty info dictionary.
        """
        self.current_steps = 0
        self.position = 2
        return self.position, {}

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            Action to take (0: left, 1: right).

        Returns
        -------
        next_state : int
            The resulting position of the rover.
        reward : float
            The reward at the new position.
        terminated : bool
            Whether the episode ended due to task success (always False here).
        truncated : bool
            Whether the episode ended due to reaching the time limit.
        info : dict
            An empty dictionary.
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        # stochastic flip with prob 1 - P[pos, action]
        p = float(self.P[self.position, action])
        follow = self.rng.random() < p
        a_used = action if follow else 1 - action

        self.position = self.get_next_state(self.position, a_used)

        reward = float(self.rewards[self.position])
        terminated = False
        truncated = self.current_steps >= self.horizon

        return self.position, reward, terminated, truncated, {}

    def get_reward_per_action(self) -> np.ndarray:
        """
        Return the expected reward function R[s, a] for each (state, action) pair.

        R[s, a] is the expected reward resulting from taking action a in state s,
        accounting for the transition probabilities.

        Returns
        -------
        R : np.ndarray
            A (num_states, num_actions) array of expected rewards.
        """
        nS, nA = self.observation_space.n, self.action_space.n
        R = np.zeros((nS, nA), dtype=float)
        T = self.get_transition_matrix()

        for s in range(nS):
            for a in range(nA):
                expected_reward = 0.0
                for next_s in range(nS):
                    expected_reward += T[s, a, next_s] * self.rewards[next_s]
                R[s, a] = float(expected_reward)
        return R

    def get_next_state(self, state: int, action: int) -> int:
        """
        Get the next state given a state and an action (assuming deterministic execution).

        Parameters
        ----------
        state : int
            The current state.
        action : int
            The action to take.

        Returns
        -------
        next_state : int
            The resulting state.
        """
        if len(self.states) == state + 1 and action == 1 or state == 0 and action == 0:
            return state
        if action == 1:
            return state + 1
        else:
            return state - 1

    def get_transition_matrix(
        self,
        S: np.ndarray | None = None,
        A: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Construct a transition matrix T[s, a, s'].

        Parameters
        ----------
        S : np.ndarray, optional
            Array of states. Uses internal states if None.
        A : np.ndarray, optional
            Array of actions. Uses internal actions if None.
        P : np.ndarray, optional
            Action success probabilities. Uses internal P if None.

        Returns
        -------
        T : np.ndarray
            A (num_states, num_actions, num_states) tensor where
            T[s, a, s'] = probability of transitioning to s' from s via a.
        """
        if S is None or A is None or P is None:
            S, A, P = self.states, self.actions, self.P

        nS, nA = len(S), len(A)
        T = np.zeros((nS, nA, nS), dtype=float)
        for s in range(nS):
            for a in range(nA):
                p_follow = float(P[s, a])
                s_follow = self.get_next_state(s, a)
                s_flip = self.get_next_state(s, 1 - a)
                T[s, a, s_follow] = p_follow
                T[s, a, s_flip] = 1.0 - p_follow

        return T

    def render(self, mode: str = "human"):
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : str
            Render mode (only "human" is supported).
        """
        print(f"[MarsRover] pos={self.position}, steps={self.current_steps}")


class ContextualMarsRover(MarsRover):
    """MarsRover with episode-level **ground friction** and **horizon** contexts.

    Friction is mapped to the same ``transition_probabilities`` mechanism as the
    base class: each entry ``P[s, a]`` is the probability that the commanded
    action is executed (otherwise it is flipped). **Higher friction** means
    **higher** follow probability for every state–action pair.

    **Horizon** is the episode step limit (same as ``MarsRover.horizon``); larger
    values give more time per episode.

    At ``reset``, pass ``options`` to pick contexts, e.g.
    ``options={"context_index": 0, "horizon_index": 1}``,
    ``options={"friction": 0.75, "horizon": 12}``, or
    ``options={"joint_context_index": k}`` to cycle through all
    ``(friction, horizon)`` pairs in row-major order (friction major).

    Friction values are clipped to ``[0, 1]``.

    Parameters
    ----------
    friction_levels : sequence of float, optional
        Discrete friction values (default: low / mid / high).
    horizon_levels : sequence of int, optional
        Discrete episode horizons. If omitted and ``horizon`` is the default ``10``,
        uses ``(10, 6, 16)`` (index ``0`` matches :class:`MarsRover`). If omitted and
        ``horizon`` is not ``10``, uses ``(horizon,)`` only.
    rewards, horizon, seed
        ``rewards`` and ``seed`` are forwarded unchanged.
    """

    def __init__(
        self,
        friction_levels: Sequence[float] | np.ndarray | None = None,
        horizon_levels: Sequence[int] | np.ndarray | None = None,
        rewards: list[float] | None = None,
        horizon: int = 10,
        seed: int | None = None,
    ) -> None:
        if friction_levels is None:
            friction_levels = (0.25, 0.55, 0.95)
        self.friction_levels = np.asarray(list(friction_levels), dtype=float)
        if self.friction_levels.size == 0:
            raise ValueError("friction_levels must contain at least one value")

        if horizon_levels is None:
            if int(horizon) == 10:
                self.horizon_levels = np.asarray((10, 6, 16), dtype=int)
            else:
                self.horizon_levels = np.asarray([int(horizon)], dtype=int)
        else:
            self.horizon_levels = np.asarray(list(horizon_levels), dtype=int)
        if self.horizon_levels.size == 0:
            raise ValueError("horizon_levels must contain at least one value")
        if np.any(self.horizon_levels < 1):
            raise ValueError("horizon_levels must be positive integers")

        default_rewards = [1, 0, 0, 0, 10] if rewards is None else list(rewards)
        n = len(default_rewards)

        self.context_index = 0
        self.horizon_context_index = 0
        self.friction = float(np.clip(self.friction_levels[0], 0.0, 1.0))
        h0 = int(self.horizon_levels[0])
        p0 = np.full((n, 2), self.friction, dtype=float)

        super().__init__(
            transition_probabilities=p0,
            rewards=default_rewards,
            horizon=h0,
            seed=seed,
        )

    def _apply_friction_to_transition_matrix(self) -> None:
        n = self.observation_space.n
        p = float(np.clip(self.friction, 0.0, 1.0))
        self.P = np.full((n, 2), p, dtype=float)
        self.transition_matrix = self.T = self.get_transition_matrix()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        opts = options or {}
        if "joint_context_index" in opts:
            n_f, n_h = len(self.friction_levels), len(self.horizon_levels)
            k = int(opts["joint_context_index"]) % (n_f * n_h)
            self.context_index = k // n_h
            self.horizon_context_index = k % n_h
            self.friction = float(
                np.clip(self.friction_levels[self.context_index], 0.0, 1.0)
            )
            self.horizon = int(self.horizon_levels[self.horizon_context_index])
        else:
            if "context_index" in opts:
                n_ctx = len(self.friction_levels)
                self.context_index = int(opts["context_index"]) % n_ctx
                self.friction = float(
                    np.clip(self.friction_levels[self.context_index], 0.0, 1.0)
                )
            elif "friction" in opts:
                self.friction = float(np.clip(float(opts["friction"]), 0.0, 1.0))
                self.context_index = int(
                    np.argmin(np.abs(self.friction_levels - self.friction))
                )

            if "horizon_index" in opts:
                n_h = len(self.horizon_levels)
                self.horizon_context_index = int(opts["horizon_index"]) % n_h
                self.horizon = int(self.horizon_levels[self.horizon_context_index])
            elif "horizon" in opts:
                self.horizon = max(1, int(opts["horizon"]))
                self.horizon_context_index = int(
                    np.argmin(np.abs(self.horizon_levels - self.horizon))
                )

        self._apply_friction_to_transition_matrix()
        obs, info = super().reset(seed=seed, options=options)
        out_info = dict(info)
        out_info["friction"] = self.friction
        out_info["context_index"] = self.context_index
        out_info["horizon"] = self.horizon
        out_info["horizon_context_index"] = self.horizon_context_index
        return obs, out_info

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        out_info = dict(info)
        out_info["friction"] = self.friction
        out_info["context_index"] = self.context_index
        out_info["horizon"] = self.horizon
        out_info["horizon_context_index"] = self.horizon_context_index
        return obs, reward, terminated, truncated, out_info


_CONTEXT_OPTION_KEYS = frozenset(
    {
        "joint_context_index",
        "context_index",
        "friction",
        "horizon_index",
        "horizon",
    }
)


class RoundRobinJointContextWrapper(gym.Wrapper):
    """Cycles friction×horizon contexts on every ``reset`` during training.

    Each call to ``reset`` without explicit context fields in ``options`` sets
    ``joint_context_index``. By default this cycles ``0, 1, …, n_friction * n_horizon - 1``
    (wrapping). Pass ``joint_context_indices`` to restrict training to a Section 6.4-style
    **train** set (e.g. a product of index ranges or a sparse mode-C list).

    If any of
    ``joint_context_index``, ``context_index``, ``friction``, ``horizon_index``,
    or ``horizon`` is already present in ``options``, it is passed through unchanged
    (for evaluation or manual control).

    Wrap a :class:`ContextualMarsRover` (typically inside :class:`gymnasium.wrappers.Monitor`).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: gym.Env,
        joint_context_indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        super().__init__(env)
        self._joint_rr = 0
        base = self.env.unwrapped
        n_f = len(base.friction_levels)  # type: ignore[attr-defined]
        n_h = len(base.horizon_levels)  # type: ignore[attr-defined]
        n_joint = max(1, n_f * n_h)
        if joint_context_indices is None:
            self._schedule: np.ndarray | None = None
        else:
            self._schedule = (
                np.asarray(list(joint_context_indices), dtype=int) % n_joint
            )
            if self._schedule.size == 0:
                raise ValueError(
                    "joint_context_indices must be non-empty when provided"
                )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        opts = dict(options or {})
        if not _CONTEXT_OPTION_KEYS.intersection(opts):
            base = self.env.unwrapped
            n_f = len(base.friction_levels)  # type: ignore[attr-defined]
            n_h = len(base.horizon_levels)  # type: ignore[attr-defined]
            n_joint = max(1, n_f * n_h)
            if self._schedule is None:
                opts["joint_context_index"] = self._joint_rr % n_joint
            else:
                opts["joint_context_index"] = int(
                    self._schedule[self._joint_rr % len(self._schedule)]
                )
            self._joint_rr += 1
        return self.env.reset(seed=seed, options=opts)


class MarsRoverPartialObsWrapper(gym.Wrapper):
    """
    Partially-observable wrapper for the MarsRover environment.

    This wrapper injects observation noise to simulate partial observability.
    With a specified probability, the true state (position) is replaced by a randomly
    selected incorrect position in the state space.

    Parameters
    ----------
    env : MarsRover
        The fully observable MarsRover environment to wrap.
    noise : float, default=0.1
        Probability in [0, 1] of returning a random incorrect position.
    seed : int or None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: MarsRover, noise: float = 0.1, seed: int | None = None):
        """
        Initialize the partial observability wrapper.

        Parameters
        ----------
        env : MarsRover
            The environment to wrap.
        noise : float, optional
            Probability of observing an incorrect state, by default 0.1.
        seed : int or None, optional
            Random seed for reproducibility, by default None.
        """
        super().__init__(env)
        assert 0.0 <= noise <= 1.0, "noise must be in [0,1]"
        self.noise = noise
        self.rng = np.random.default_rng(seed)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the base environment and return a noisy observation.

        Parameters
        ----------
        seed : int or None, optional
            Seed for the reset, by default None.
        options : dict or None, optional
            Additional reset options, by default None.

        Returns
        -------
        obs : int
            The (possibly noisy) initial observation.
        info : dict
            Additional info returned by the environment.
        """
        true_obs, info = self.env.reset(seed=seed, options=options)
        return self._noisy_obs(true_obs), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment and return a noisy observation.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        obs : int
            The (possibly noisy) resulting observation.
        reward : float
            The reward received.
        terminated : bool
            Whether the episode terminated.
        truncated : bool
            Whether the episode was truncated due to time limit.
        info : dict
            Additional information from the base environment.
        """
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy_obs(true_obs), reward, terminated, truncated, info

    def _noisy_obs(self, true_obs: int) -> int:
        """
        Return a possibly noisy version of the true observation.

        With probability `noise`, replaces the true observation with
        a randomly selected incorrect state.

        Parameters
        ----------
        true_obs : int
            The true observation/state index.

        Returns
        -------
        obs : int
            A noisy (or true) observation.
        """
        if self.rng.random() < self.noise:
            n = self.observation_space.n
            others = [s for s in range(n) if s != true_obs]
            return int(self.rng.choice(others))
        else:
            return int(true_obs)

    def render(self, mode: str = "human"):
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : str, optional
            Render mode, by default "human".

        Returns
        -------
        Any
            Rendered output from the base environment.
        """
        return self.env.render(mode=mode)


class RandomWalk(gym.Env):
    """
    A simple random walk environment with 5 states and 2 actions.

    The agent starts in the middle state (state 2) and can move left (action 0) or right (action 1).
    The episode ends when the agent reaches either end of the state space (state 0 or state 4).
    Rewards are given based on the resulting state after taking an action.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

        self.observation_space = gym.spaces.Discrete(5)
        self.action_space = gym.spaces.Discrete(2)

        self.position = 2  # start in the middle

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        self.position = 2
        return self.position, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if action == 0:
            self.position = max(0, self.position - 1)
        elif action == 1:
            self.position = min(4, self.position + 1)
        else:
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        reward = 0.0
        if self.position == 4:
            reward = 1.0

        terminated = self.position == 0 or self.position == 4
        truncated = False

        return self.position, reward, terminated, truncated, {}

    def render(self, mode: str = "human"):
        print(f"[RandomWalk] pos={self.position}")
