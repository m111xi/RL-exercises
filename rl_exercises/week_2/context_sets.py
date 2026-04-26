"""Train / test / optional validation context sets for contextual MarsRover (Section 6.4 style).

We define **marginal** train supports per context dimension (friction, horizon). A test
context is **interpolation** if at least one dimension lies in that dimension's train
support; **extrapolation** if both dimensions lie outside their train supports.

Training **modes** (discrete analogue of the paper's A/B/C):

- **A — convex / product train set:** independent uniform (discrete) ranges on friction
  and horizon; the train set is their Cartesian product (a ``|F|×|H|`` block in index space).
- **B — narrow product:** same structure as A but small index ranges per dimension
  (``small variation``).
- **C — sparse / non-convex:** only an explicit list of ``(friction_idx, horizon_idx)``
  pairs. The convex hull of these points is larger than the set itself; held-out
  combinations of seen marginally values probe **combinatorial interpolation**.

Joint indices match :class:`~rl_exercises.environments.ContextualMarsRover`:
``joint = friction_idx * n_horizon + horizon_idx``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Literal, Sequence

import numpy as np

Mode = Literal["A", "B", "C"]


def joint_index(friction_idx: int, horizon_idx: int, n_horizon: int) -> int:
    return int(friction_idx) * int(n_horizon) + int(horizon_idx)


def split_joint_index_explicit(joint: int, n_friction: int, n_horizon: int) -> tuple[int, int]:
    n_joint = max(1, n_friction * n_horizon)
    k = int(joint) % n_joint
    return k // n_horizon, k % n_horizon


def _unique_sorted_ints(x: Iterable[int]) -> np.ndarray:
    return np.asarray(sorted(set(int(i) for i in x)), dtype=int)


def pairs_to_joint_indices(
    pairs: Sequence[tuple[int, int]],
    n_horizon: int,
) -> np.ndarray:
    out = [joint_index(fi, hi, n_horizon) for fi, hi in pairs]
    return _unique_sorted_ints(out)


def product_train_indices(
    friction_indices: Sequence[int],
    horizon_indices: Sequence[int],
    n_horizon: int,
) -> np.ndarray:
    idxs = [
        joint_index(fi, hi, n_horizon)
        for fi, hi in product(
            sorted(set(int(i) for i in friction_indices)),
            sorted(set(int(i) for i in horizon_indices)),
        )
    ]
    return _unique_sorted_ints(idxs)


def marginal_friction_from_joints_explicit(
    joint_indices: np.ndarray,
    n_friction: int,
    n_horizon: int,
) -> np.ndarray:
    n_f, n_h = int(n_friction), int(n_horizon)
    fis = [split_joint_index_explicit(int(j), n_f, n_h)[0] for j in joint_indices]
    return _unique_sorted_ints(fis)


def marginal_horizon_from_joints_explicit(
    joint_indices: np.ndarray,
    n_friction: int,
    n_horizon: int,
) -> np.ndarray:
    n_f, n_h = int(n_friction), int(n_horizon)
    his = [split_joint_index_explicit(int(j), n_f, n_h)[1] for j in joint_indices]
    return _unique_sorted_ints(his)


def train_marginals(
    train_joint_indices: np.ndarray,
    n_friction: int,
    n_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (friction_idx support, horizon_idx support) implied by train joints."""
    return (
        marginal_friction_from_joints_explicit(train_joint_indices, n_friction, n_horizon),
        marginal_horizon_from_joints_explicit(train_joint_indices, n_friction, n_horizon),
    )


def classify_interpolation_vs_extrapolation(
    friction_idx: int,
    horizon_idx: int,
    train_friction_support: Sequence[int] | np.ndarray,
    train_horizon_support: Sequence[int] | np.ndarray,
) -> str:
    """Per Section 6.4 wording (marginal train supports per dimension)."""
    fs = {int(x) for x in train_friction_support}
    hs = {int(x) for x in train_horizon_support}
    fi, hi = int(friction_idx), int(horizon_idx)
    f_out = fi not in fs
    h_out = hi not in hs
    if f_out and h_out:
        return "extrapolation"
    return "interpolation"


@dataclass(frozen=True)
class ContextProtocol:
    """Concrete train / val / test joint-context indices for a fixed friction×horizon grid."""

    friction_levels: np.ndarray
    horizon_levels: np.ndarray
    train_joint_indices: np.ndarray
    test_joint_indices: np.ndarray
    validation_joint_indices: np.ndarray

    @property
    def n_friction(self) -> int:
        return int(self.friction_levels.size)

    @property
    def n_horizon(self) -> int:
        return int(self.horizon_levels.size)

    def train_marginals(self) -> tuple[np.ndarray, np.ndarray]:
        return train_marginals(self.train_joint_indices, self.n_friction, self.n_horizon)

    def label_pair(self, friction_idx: int, horizon_idx: int) -> str:
        tf, th = self.train_marginals()
        return classify_interpolation_vs_extrapolation(
            friction_idx, horizon_idx, tf, th
        )

    def label_joint(self, joint_index: int) -> str:
        fi, hi = split_joint_index_explicit(
            joint_index, self.n_friction, self.n_horizon
        )
        return self.label_pair(fi, hi)


def sample_joint_indices(
    rng: np.random.Generator,
    joint_indices: np.ndarray,
    n: int,
    *,
    replace: bool = True,
) -> np.ndarray:
    """Sample ``n`` joint indices from a finite set (e.g. ``n=1000`` like the paper)."""
    pool = np.asarray(joint_indices, dtype=int)
    if pool.size == 0:
        raise ValueError("joint_indices must be non-empty")
    if not replace and n > pool.size:
        raise ValueError("n larger than pool without replacement")
    return rng.choice(pool, size=int(n), replace=replace)


def build_mode_a_protocol(
    friction_levels: Sequence[float] | np.ndarray,
    horizon_levels: Sequence[int] | np.ndarray,
    *,
    train_friction_idx: Sequence[int],
    train_horizon_idx: Sequence[int],
    test_friction_idx: Sequence[int],
    test_horizon_idx: Sequence[int],
    validation_joint_indices: np.ndarray | None = None,
) -> ContextProtocol:
    """Mode A: train and test sets are full rectangles (products) in index space."""
    fl = np.asarray(list(friction_levels), dtype=float)
    hl = np.asarray(list(horizon_levels), dtype=int)
    n_h = int(hl.size)
    train_j = product_train_indices(train_friction_idx, train_horizon_idx, n_h)
    test_j = product_train_indices(test_friction_idx, test_horizon_idx, n_h)
    val_j = (
        np.asarray(validation_joint_indices, dtype=int).copy()
        if validation_joint_indices is not None
        else np.zeros((0,), dtype=int)
    )
    return ContextProtocol(
        friction_levels=fl,
        horizon_levels=hl,
        train_joint_indices=train_j,
        test_joint_indices=test_j,
        validation_joint_indices=val_j,
    )


def build_mode_b_protocol(
    friction_levels: Sequence[float] | np.ndarray,
    horizon_levels: Sequence[int] | np.ndarray,
    *,
    train_friction_idx: Sequence[int],
    train_horizon_idx: Sequence[int],
    test_friction_idx: Sequence[int],
    test_horizon_idx: Sequence[int],
    validation_joint_indices: np.ndarray | None = None,
) -> ContextProtocol:
    """Mode B: same as A structurally; use small index ranges when you instantiate."""
    return build_mode_a_protocol(
        friction_levels,
        horizon_levels,
        train_friction_idx=train_friction_idx,
        train_horizon_idx=train_horizon_idx,
        test_friction_idx=test_friction_idx,
        test_horizon_idx=test_horizon_idx,
        validation_joint_indices=validation_joint_indices,
    )


def build_mode_c_protocol(
    friction_levels: Sequence[float] | np.ndarray,
    horizon_levels: Sequence[int] | np.ndarray,
    *,
    train_pairs: Sequence[tuple[int, int]],
    test_pairs: Sequence[tuple[int, int]],
    validation_pairs: Sequence[tuple[int, int]] | None = None,
) -> ContextProtocol:
    """Mode C: arbitrary (sparse) train pairs; test pairs chosen for interp / extrap analysis."""
    fl = np.asarray(list(friction_levels), dtype=float)
    hl = np.asarray(list(horizon_levels), dtype=int)
    n_h = int(hl.size)
    train_j = pairs_to_joint_indices(train_pairs, n_h)
    test_j = pairs_to_joint_indices(test_pairs, n_h)
    val_j = (
        pairs_to_joint_indices(validation_pairs, n_h)
        if validation_pairs is not None
        else np.zeros((0,), dtype=int)
    )
    return ContextProtocol(
        friction_levels=fl,
        horizon_levels=hl,
        train_joint_indices=train_j,
        test_joint_indices=test_j,
        validation_joint_indices=val_j,
    )


def default_three_by_three_example(
    mode: Mode,
    *,
    include_validation: bool = False,
) -> ContextProtocol:
    """Opinionated 3×3 default matching :class:`ContextualMarsRover` default grids.

    Uses friction ``(0.25, 0.55, 0.95)`` and horizons ``(10, 6, 16)``. If you change
    ``friction_levels`` / ``horizon_levels`` on the env, build a protocol with the
    explicit constructors instead of this helper.

    - **A:** train = test = full 3×3 product (all test contexts are interpolation).
    - **B:** train = 2×2 lower-left product; test = full 3×3 (includes extrapolation e.g.
      ``(2, 2)`` and combinatorial-style cases).
    - **C:** train = four corners; test = ``(1, 1)`` (extrapolation) and ``(0, 1)``
      (combinatorial interpolation). Optional validation context ``(1, 0)``.
    """
    fl = np.asarray([0.25, 0.55, 0.95], dtype=float)
    hl = np.asarray([10, 6, 16], dtype=int)
    n_h = int(hl.size)
    empty_val = np.zeros((0,), dtype=int)

    if mode == "A":
        idx = [0, 1, 2]
        train_j = product_train_indices(idx, idx, n_h)
        test_j = train_j.copy()
        val_j = empty_val
    elif mode == "B":
        train_j = product_train_indices([0, 1], [0, 1], n_h)
        test_j = product_train_indices([0, 1, 2], [0, 1, 2], n_h)
        val_j = (
            np.asarray([joint_index(2, 0, n_h), joint_index(0, 2, n_h)], dtype=int)
            if include_validation
            else empty_val
        )
    else:
        train_j = pairs_to_joint_indices([(0, 0), (0, 2), (2, 0), (2, 2)], n_h)
        test_j = pairs_to_joint_indices([(1, 1), (0, 1)], n_h)
        val_j = (
            pairs_to_joint_indices([(1, 0)], n_h) if include_validation else empty_val
        )

    return ContextProtocol(
        friction_levels=fl,
        horizon_levels=hl,
        train_joint_indices=train_j,
        test_joint_indices=test_j,
        validation_joint_indices=val_j,
    )


def reset_options_for_joint_indices(
    proto: ContextProtocol,
    joint_indices: np.ndarray | Sequence[int],
) -> list[dict[str, Any]]:
    """Build ``options`` dicts for :meth:`gymnasium.Env.reset` (explicit contexts).

    Uses ``context_index`` and ``horizon_index`` as accepted by
    :class:`~rl_exercises.environments.ContextualMarsRover`.
    """
    n_f, n_h = proto.n_friction, proto.n_horizon
    out: list[dict[str, Any]] = []
    for j in np.asarray(joint_indices, dtype=int).ravel():
        fi, hi = split_joint_index_explicit(int(j), n_f, n_h)
        out.append({"context_index": fi, "horizon_index": hi})
    return out


def summarize_protocol(proto: ContextProtocol) -> dict[str, object]:
    """Human-readable summary for logging."""
    tf, th = proto.train_marginals()
    rows = []
    for j in proto.test_joint_indices:
        fi, hi = split_joint_index_explicit(j, proto.n_friction, proto.n_horizon)
        rows.append(
            {
                "joint": int(j),
                "friction_idx": fi,
                "horizon_idx": hi,
                "label": proto.label_joint(int(j)),
            }
        )
    return {
        "n_friction": proto.n_friction,
        "n_horizon": proto.n_horizon,
        "train_marginal_friction_idx": tf.tolist(),
        "train_marginal_horizon_idx": th.tolist(),
        "n_train_joints": int(proto.train_joint_indices.size),
        "n_test_joints": int(proto.test_joint_indices.size),
        "n_val_joints": int(proto.validation_joint_indices.size),
        "test_contexts": rows,
    }
