"""Tabular policy iteration on contextual MarsRover: with vs without context (Section 6.4 style).

**With context:** For each context (joint friction×horizon index) the dynamics are a separate
known MDP. The optimal policy can depend on the context, π(s, c). Here we compute the
**oracle** policy for each context by running :class:`~rl_exercises.week_2.policy_iteration.PolicyIteration`
on a plain :class:`~rl_exercises.environments.MarsRover` with that context's ``P`` and horizon.

**Without context (hidden):** The agent only chooses actions from physical state ``s``.
We approximate the paper's "no context in the observation" by planning on a **single**
MDP whose transitions are the **uniform mixture** over all **train** contexts. One policy
π(s) is then evaluated on each true test/validation context (mismatch whenever the real
world differs from the mixture).

Run::

    python -m rl_exercises.week_2.contextual_tabular

to print a small table (mean episode return) on the default protocol mode **B** with
validation contexts enabled.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from rl_exercises.environments import MarsRover
from rl_exercises.week_2.context_sets import (
    ContextProtocol,
    default_three_by_three_example,
    split_joint_index_explicit,
)
from rl_exercises.week_2.policy_iteration import PolicyIteration


class MixtureMarsRover(MarsRover):
    """MarsRover-shaped env with a fixed transition tensor (train-context mixture)."""

    def __init__(
        self,
        transition_tensor: np.ndarray,
        rewards: list[float],
        horizon: int,
        seed: int | None = None,
    ) -> None:
        self._T_fixed = np.asarray(transition_tensor, dtype=float)
        n = int(self._T_fixed.shape[0])
        super().__init__(
            transition_probabilities=np.ones((n, 2)),
            rewards=list(rewards),
            horizon=int(horizon),
            seed=seed,
        )
        self.transition_matrix = self.T = self._T_fixed

    def get_transition_matrix(
        self,
        S: np.ndarray | None = None,
        A: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._T_fixed


def mars_rover_at_joint(proto: ContextProtocol, joint: int, seed: int = 0) -> MarsRover:
    """Single-context MarsRover for one ``joint_context_index``."""
    fi, hi = split_joint_index_explicit(int(joint), proto.n_friction, proto.n_horizon)
    f = float(np.clip(proto.friction_levels[fi], 0.0, 1.0))
    h = int(proto.horizon_levels[hi])
    n = 5
    return MarsRover(
        transition_probabilities=np.full((n, 2), f),
        horizon=h,
        seed=seed,
    )


def narrow_train_to_fixed_horizon(
    proto: ContextProtocol,
    horizon_idx: int = 0,
) -> ContextProtocol:
    """Keep only train joints with a given horizon index (same episode length for mixture MDP)."""
    keep: list[int] = []
    for j in proto.train_joint_indices:
        _, hi = split_joint_index_explicit(int(j), proto.n_friction, proto.n_horizon)
        if int(hi) == int(horizon_idx):
            keep.append(int(j))
    if len(keep) < 1:
        return proto
    return ContextProtocol(
        friction_levels=proto.friction_levels,
        horizon_levels=proto.horizon_levels,
        train_joint_indices=np.asarray(keep, dtype=int),
        test_joint_indices=proto.test_joint_indices,
        validation_joint_indices=proto.validation_joint_indices,
    )


def mixture_transition_over_train(proto: ContextProtocol) -> np.ndarray:
    """Uniform average of ``T[s,a,s']`` over train joint contexts."""
    train_js = proto.train_joint_indices
    if train_js.size == 0:
        raise ValueError("protocol has empty train_joint_indices")
    acc = None
    for j in train_js:
        env = mars_rover_at_joint(proto, int(j))
        t = env.get_transition_matrix()
        acc = t.copy() if acc is None else acc + t
    assert acc is not None
    return acc / float(train_js.size)


def train_blind_policy(
    proto: ContextProtocol, gamma: float, seed: int
) -> PolicyIteration:
    """Policy iteration on the mixture MDP (no context in state)."""
    t_mix = mixture_transition_over_train(proto)
    ref = mars_rover_at_joint(proto, int(proto.train_joint_indices[0]))
    env = MixtureMarsRover(
        t_mix,
        list(ref.rewards),
        horizon=int(ref.horizon),
        seed=seed,
    )
    return PolicyIteration(env, gamma=gamma, seed=seed)


def oracle_policy_at_joint(
    proto: ContextProtocol, joint: int, gamma: float, seed: int
) -> PolicyIteration:
    return PolicyIteration(
        mars_rover_at_joint(proto, joint, seed=seed), gamma=gamma, seed=seed
    )


def mean_episode_return(
    env: MarsRover,
    policy: PolicyIteration,
    *,
    episodes: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    totals: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        g = 0.0
        done = False
        while not done:
            a, _ = policy.predict_action(obs, None)
            obs, r, term, trunc, _ = env.step(int(a))
            g += float(r)
            done = bool(term or trunc)
        totals.append(g)
    return float(np.mean(totals))


def evaluate_on_joints(
    proto: ContextProtocol,
    policy: PolicyIteration,
    joint_indices: np.ndarray,
    *,
    gamma: float,
    episodes: int,
    base_seed: int,
) -> list[tuple[int, str, float]]:
    rows: list[tuple[int, str, float]] = []
    for i, j in enumerate(np.asarray(joint_indices, dtype=int).ravel()):
        env = mars_rover_at_joint(proto, int(j), seed=base_seed + i)
        m = mean_episode_return(env, policy, episodes=episodes, seed=base_seed + 17 * i)
        rows.append((int(j), proto.label_joint(int(j)), m))
    return rows


def run_demo(
    *,
    mode: str = "B",
    include_validation: bool = True,
    gamma: float = 0.9,
    episodes: int = 40,
    seed: int = 42,
) -> None:
    proto = default_three_by_three_example(
        mode,  # type: ignore[arg-type]
        include_validation=include_validation,
    )
    # Mixture MDP requires a common horizon; keep train contexts at horizon index 0 only.
    proto = narrow_train_to_fixed_horizon(proto, horizon_idx=0)
    _quiet = patch("rl_exercises.week_2.policy_iteration.printr", lambda *a, **k: None)
    with _quiet:
        blind = train_blind_policy(proto, gamma=gamma, seed=seed)
        blind.update_agent()

    print("Protocol mode:", mode, "| train joints:", proto.train_joint_indices.tolist())
    print("Blind (mixture-MDP) policy -- mean return on each context:\n")

    def block(title: str, joints: np.ndarray) -> None:
        print(f"--- {title} ---")
        for j, lab, ret in evaluate_on_joints(
            proto, blind, joints, gamma=gamma, episodes=episodes, base_seed=seed
        ):
            with _quiet:
                oracle = oracle_policy_at_joint(proto, j, gamma=gamma, seed=seed)
                oracle.update_agent()
            env = mars_rover_at_joint(proto, j, seed=seed)
            opt = mean_episode_return(env, oracle, episodes=episodes, seed=seed + j)
            print(
                f"  joint={j:2d}  {lab:16s}  blind={ret:8.3f}  oracle={opt:8.3f}  gap={opt - ret:8.3f}"
            )
        print()

    block("TEST", proto.test_joint_indices)
    if proto.validation_joint_indices.size:
        block("VALIDATION", proto.validation_joint_indices)


if __name__ == "__main__":
    run_demo()
