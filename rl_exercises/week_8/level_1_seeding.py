from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rl_exercises.agent.buffer import SimpleBuffer
from rl_exercises.week_8.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl_exercises.week_8.sarsa_qlearning import TDAgent
from rl_exercises.week_8.stats_utils import iqm, summarize_curves

try:  #Prefer the central repo harness
    from rl_exercises.train_agent import make_env as _repo_make_env
except Exception:  # pragma: no cover - fallback for minimal local installs.
    _repo_make_env = None

ENV_NAME = "MarsRover"
ENV_KWARGS: dict[str, Any] = {
    "transition_probabilities": np.full((5, 2), 0.8),
    "rewards": [1, 0, 0, 0, 10],
    "horizon": 10,
}
ALGORITHM = "qlearning"
N_SEEDS = 20
NUM_EPISODES = 10_000
EVAL_INTERVAL = 500
EVAL_EPISODES = 50
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.2

SEED_COUNTS = [3, 10, 20]
DISPARATE_SETS = [[0, 1, 2], [7, 8, 9], [14, 15, 16]]

SAVE_PLOTS = True
OUTDIR = Path("./rl_exercises/week_8/week_8_plots")
OBSERVATIONS_FILE = Path(__file__).with_name("observations_l1.txt")

def _env_kwargs(seed: int) -> dict[str, Any]:
    """Return a fresh env kwargs dict for one run."""
    kwargs = dict(ENV_KWARGS)
    if ENV_NAME in {"MarsRover", "ContextualMarsRover", "RandomWalk"}:
        kwargs["seed"] = int(seed)
    return kwargs

def make_experiment_env(seed: int, *, for_evaluation: bool = False):
    #Create an environment
    kwargs = _env_kwargs(seed)
    if _repo_make_env is not None:
        try:
            return _repo_make_env(ENV_NAME, kwargs, for_evaluation=for_evaluation)
        except TypeError:
            # Older train_agent.py versions do not have the for_evaluation flag.
            return _repo_make_env(ENV_NAME, kwargs)

    import gymnasium as gym

    if ENV_NAME == "MarsRover":
        from rl_exercises.environments import MarsRover

        return MarsRover(**kwargs)
    if ENV_NAME == "ContextualMarsRover":
        from rl_exercises.environments import ContextualMarsRover

        return ContextualMarsRover(**kwargs)
    if ENV_NAME == "RandomWalk":
        from rl_exercises.environments import RandomWalk

        return RandomWalk(**kwargs)
    return gym.make(ENV_NAME, **kwargs)

def evaluate_agent(agent: TDAgent, seed: int, episodes: int = EVAL_EPISODES) -> float:
    #Evaluate one agent greedily and return the mean episode return
    env = make_experiment_env(seed + 1_000_000, for_evaluation=True)
    rewards: list[float] = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = agent.predict_action(obs, info, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
        rewards.append(total_reward)
    env.close()
    return float(np.mean(rewards))


def run_one_seed(seed: int) -> tuple[np.ndarray, np.ndarray]:
    #Train one TD agent and return (eval_points, eval_returns)
    env = make_experiment_env(seed, for_evaluation=False)
    policy = EpsilonGreedyPolicy(env, epsilon=EPSILON, seed=seed)
    agent = TDAgent(
        env=env,
        policy=policy,
        alpha=ALPHA,
        gamma=GAMMA,
        algorithm=ALGORITHM,
    )
    buffer = SimpleBuffer()

    eval_points = [0]
    eval_returns = [evaluate_agent(agent, seed)]

    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset(seed=seed + episode)
        done = False
        while not done:
            action, _ = agent.predict_action(obs, info, evaluate=False)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)
            buffer.add(obs, action, float(reward), next_obs, done, next_info)
            agent.update_agent(buffer.sample())
            obs, info = next_obs, next_info

        if episode % EVAL_INTERVAL == 0:
            eval_points.append(episode)
            eval_returns.append(evaluate_agent(agent, seed))

    env.close()
    return np.asarray(eval_points, dtype=int), np.asarray(eval_returns, dtype=float)


def run_many_seeds(seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    #Run the configured experiment for multiple seeds
    all_curves: list[np.ndarray] = []
    eval_points_ref: np.ndarray | None = None
    for i, seed in enumerate(seeds, start=1):
        print(f"Seed {seed} ({i}/{len(seeds)})")
        eval_points, curve = run_one_seed(seed)
        if eval_points_ref is None:
            eval_points_ref = eval_points
        elif not np.array_equal(eval_points_ref, eval_points):
            raise RuntimeError("All seeds must produce the same evaluation points")
        all_curves.append(curve)
    if eval_points_ref is None:
        raise ValueError("No seeds were provided")
    return eval_points_ref, np.vstack(all_curves)


def _final_row(summary: dict[str, np.ndarray]) -> str:
    #Format final-checkpoint statistics for observations_l1.txt
    return (
        f"mean={summary['mean'][-1]:.3f}, "
        f"median={summary['median'][-1]:.3f}, "
        f"IQM={summary['iqm'][-1]:.3f}, "
        f"std={summary['std'][-1]:.3f}, "
        f"SE={summary['se'][-1]:.3f}, "
        f"IQM 95% CI=[{summary['iqm_ci_low'][-1]:.3f}, "
        f"{summary['iqm_ci_high'][-1]:.3f}]"
    )


def save_plots(eval_points: np.ndarray, curves: np.ndarray) -> None:
    #Save the README-requested plots when ``SAVE_PLOTS`` is True
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = summarize_curves(curves)

    #Figure 1: aggregate metrics over time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eval_points, summary["mean"], label="Mean")
    ax.fill_between(
        eval_points,
        summary["mean_ci_low"],
        summary["mean_ci_high"],
        alpha=0.15,
    )
    ax.plot(eval_points, summary["median"], label="Median", linestyle="--")
    ax.plot(eval_points, summary["iqm"], label="IQM")
    ax.fill_between(
        eval_points,
        summary["iqm_ci_low"],
        summary["iqm_ci_high"],
        alpha=0.15,
    )
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Greedy eval return")
    ax.set_title(f"{ALGORITHM} on {ENV_NAME} (N={N_SEEDS}); shaded = 95% CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "learning_curves.png", dpi=130)
    plt.close(fig)

    #Figure 2: effect of seed count on IQM and CI width
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in SEED_COUNTS:
        sub = summarize_curves(curves[:n])
        ax.plot(eval_points, sub["iqm"], label=f"IQM (N={n})")
        ax.fill_between(eval_points, sub["iqm_ci_low"], sub["iqm_ci_high"], alpha=0.18)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("IQM greedy eval return")
    ax.set_title("Effect of number of seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "seed_count_effect.png", dpi=130)
    plt.close(fig)

    #Figure 3: several disjoint low-seed sets
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed_set in DISPARATE_SETS:
        sub = curves[seed_set]
        sub_iqm = np.array([iqm(sub[:, t]) for t in range(sub.shape[1])])
        ax.plot(eval_points, sub_iqm, label=f"seeds {seed_set}")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("IQM greedy eval return")
    ax.set_title("Disjoint 3-seed sets can tell different stories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "disparate_seeds.png", dpi=130)
    plt.close(fig)

    #Figure 4: uncertainty measures over time
    ci_half_width = (summary["iqm_ci_high"] - summary["iqm_ci_low"]) / 2
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eval_points, summary["std"], label="Std across seeds")
    ax.plot(eval_points, summary["se"], label="Standard error")
    ax.plot(eval_points, ci_half_width, label="IQM 95% CI half-width")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Uncertainty / spread")
    ax.set_title("Std, SE and 95% CI over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "uncertainty_over_time.png", dpi=130)
    plt.close(fig)

def main() -> None:
    seeds = list(range(N_SEEDS))

    print(f"Running {ALGORITHM} on {ENV_NAME} for {N_SEEDS} seeds ...")
    eval_points, curves = run_many_seeds(seeds)

    if SAVE_PLOTS:
        save_plots(eval_points, curves)

    print("\n=== Final-checkpoint summary (greedy eval return) ===")
    print(
        f"{'N seeds':>8} {'mean':>7} {'median':>7} {'IQM':>7} "
        f"{'std':>7} {'SE':>7} {'95% CI (IQM)':>22}"
    )
    for n in SEED_COUNTS:
        sub = summarize_curves(curves[:n])
        print(
            f"{n:>8} {sub['mean'][-1]:>7.3f} {sub['median'][-1]:>7.3f} "
            f"{sub['iqm'][-1]:>7.3f} {sub['std'][-1]:>7.3f} {sub['se'][-1]:>7.3f} "
            f"  [{sub['iqm_ci_low'][-1]:.3f}, {sub['iqm_ci_high'][-1]:.3f}]"
        )

    print("\nDisparate 3-seed sets (final IQM):")
    for seed_set in DISPARATE_SETS:
        sub = curves[seed_set]
        print(f"  seeds {seed_set}: IQM = {iqm(sub[:, -1]):.3f}")

    if SAVE_PLOTS:
        print(f"\nFigures written to: {OUTDIR}/")
    print(f"Observations written to: {OBSERVATIONS_FILE}")

if __name__ == "__main__":
    main()
