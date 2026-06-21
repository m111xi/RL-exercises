from __future__ import annotations
import numpy as np
Curves = np.ndarray 

def iqm(values: np.ndarray, proportiontocut: float = 0.25) -> float:
    #Interquartile mean = 25% trimmed mean
    v = np.sort(np.asarray(values, dtype=float).ravel())
    n = v.size
    if n == 0:
        return float("nan")
    cut = int(proportiontocut * n)
    trimmed = v[cut : n - cut] if n - 2 * cut > 0 else v
    return float(trimmed.mean())


def _column_apply(curves: Curves, fn) -> np.ndarray:
    """Apply a scalar statistic ``fn`` to every eval point (column)."""
    curves = np.atleast_2d(curves)
    return np.array([fn(curves[:, t]) for t in range(curves.shape[1])])


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    #Percentile bootstrap CI for a scalar statistic over the seeds
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = np.array([statistic(values[i]) for i in idx])
    lo = float(np.percentile(boot, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot, 100 * (1 + ci) / 2))
    return lo, hi

def summarize_curves(
    curves: Curves,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    #Aggregate curves over the seed axis
    curves = np.atleast_2d(curves)
    n_seeds = curves.shape[0]
    rng = np.random.default_rng(seed)

    mean = _column_apply(curves, np.mean)
    median = _column_apply(curves, np.median)
    iqm_curve = _column_apply(curves, iqm)
    std = _column_apply(curves, lambda x: np.std(x, ddof=1) if x.size > 1 else 0.0)
    se = std / np.sqrt(max(n_seeds, 1))

    mean_lo, mean_hi, iqm_lo, iqm_hi = [], [], [], []
    for t in range(curves.shape[1]):
        col = curves[:, t]
        lo, hi = bootstrap_ci(col, np.mean, n_boot, rng=rng)
        mean_lo.append(lo)
        mean_hi.append(hi)
        lo, hi = bootstrap_ci(col, iqm, n_boot, rng=rng)
        iqm_lo.append(lo)
        iqm_hi.append(hi)

    return {
        "mean": mean,
        "median": median,
        "iqm": iqm_curve,
        "std": std,
        "se": se,
        "mean_ci_low": np.array(mean_lo),
        "mean_ci_high": np.array(mean_hi),
        "iqm_ci_low": np.array(iqm_lo),
        "iqm_ci_high": np.array(iqm_hi),
        "n_seeds": n_seeds,
    }
