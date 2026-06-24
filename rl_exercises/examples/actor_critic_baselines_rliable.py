"""
Train ActorCriticAgent on multiple baseline strategies and evaluate with RLiable.

This script runs the four baseline modes from `rl_exercises/week_6/actor_critic.py` and saves
per-seed evaluation results. It then aggregates the results across seeds using the IQM
and plots confidence intervals with RLiable.

Usage:
    python rl_exercises/examples/actor_critic_baselines_rliable.py

Optional arguments are available for environment, number of seeds, and training duration.
"""

from __future__ import annotations

from typing import Any

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rl_exercises.week_6.actor_critic import ActorCriticAgent, set_seed
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve


def train_and_log(
    env_name: str,
    baseline_type: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    output_dir: Path,
) -> Path:
    env = gym.make(env_name)
    set_seed(env, seed)
    agent = ActorCriticAgent(
        env,
        lr_actor=5e-3,
        lr_critic=1e-2,
        gamma=0.99,
        gae_lambda=0.95,
        seed=seed,
        hidden_size=128,
        baseline_type=baseline_type,
        baseline_decay=0.9,
    )
    eval_env = gym.make(env_name)
    set_seed(eval_env, seed + 1000)

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    step_count = 0

    state, _ = env.reset(seed=seed)
    done = False
    trajectory: list[tuple[np.ndarray, int, float, np.ndarray, bool, Any]] = []

    while step_count < total_steps:
        action, logp = agent.predict_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((state, action, float(reward), next_state, done, logp))
        state = next_state
        step_count += 1

        if done or step_count >= total_steps:
            agent.update_agent(trajectory)
            trajectory = []
            state, _ = env.reset()
            done = False

        if step_count % eval_interval == 0:
            mean_return, std_return = agent.evaluate(
                eval_env, num_episodes=eval_episodes
            )
            records.append(
                {
                    "baseline": baseline_type,
                    "seed": seed,
                    "eval_step": step_count,
                    "eval_return": mean_return,
                    "eval_std": std_return,
                }
            )
            print(
                f"[Eval] env={env_name} baseline={baseline_type} seed={seed} step={step_count} "
                f"mean_return={mean_return:.2f} std={std_return:.2f}"
            )

    df = pd.DataFrame(records)
    output_file = output_dir / f"{baseline_type}_seed_{seed}.csv"
    df.to_csv(output_file, index=False)
    return output_file


def aggregate_results(
    result_dir: Path, baselines: list[str], seeds: list[int], env_name: str
) -> None:
    eval_scores: dict[str, np.ndarray] = {}
    eval_steps: np.ndarray | None = None

    for baseline in baselines:
        baseline_dfs = []
        for seed in seeds:
            path = result_dir / baseline / f"{baseline}_seed_{seed}.csv"
            df = pd.read_csv(path)
            if eval_steps is None:
                eval_steps = df["eval_step"].to_numpy()
            else:
                assert np.array_equal(eval_steps, df["eval_step"].to_numpy()), (
                    f"Mismatch in eval steps for baseline {baseline} seed {seed}"
                )
            baseline_dfs.append(df["eval_return"].to_numpy())
        eval_scores[baseline] = np.vstack(baseline_dfs)

    if eval_steps is None:
        raise ValueError("No evaluation results found.")

    def iqm(scores: np.ndarray) -> np.ndarray:
        return np.array(
            [metrics.aggregate_iqm(scores[:, i]) for i in range(scores.shape[-1])]
        )

    iqm_scores, iqm_cis = get_interval_estimates(eval_scores, iqm, reps=1000)

    fig = plt.figure(figsize=(10, 6))
    plot_sample_efficiency_curve(
        eval_steps,
        iqm_scores,
        iqm_cis,
        algorithms=baselines,
        xlabel="Training steps",
        ylabel="IQM evaluation return",
    )
    plt.legend(title="Baseline", loc="best")
    plt.title(f"Actor-Critic baselines on {env_name}")
    plt.tight_layout()
    figure_path = result_dir / "actor_critic_baselines_iqm.png"
    plt.savefig(figure_path)
    plt.close(fig)
    print(f"Saved RLiable plot to {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ActorCritic baselines and evaluate with RLiable. "
    )
    parser.add_argument("--env", default="LunarLander-v3", help="Gym environment")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds to run",
    )
    parser.add_argument(
        "--steps", type=int, default=200000, help="Total training steps"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=10000, help="Evaluation interval"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=5, help="Evaluation episodes"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "actor_critic_baselines_results",
        help="Directory to save CSV results and plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = ["none", "avg", "value", "gae"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for baseline in baselines:
        baseline_dir = args.output_dir / baseline
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            print(f"Training env={args.env} baseline={baseline} seed={seed}")
            train_and_log(
                env_name=args.env,
                baseline_type=baseline,
                seed=seed,
                total_steps=args.steps,
                eval_interval=args.eval_interval,
                eval_episodes=args.eval_episodes,
                output_dir=baseline_dir,
            )

    aggregate_results(args.output_dir, baselines, args.seeds, args.env)


if __name__ == "__main__":
    main()
