"""
Train PPO on LunarLander-v3 and compare against Actor-Critic.

This script runs three experiments:
- Actor-Critic agent (value-baseline) on LunarLander-v3
- PPO with vanilla settings (default `configs/agent/ppo.yaml`)
- PPO with two enhancements: value-function clipping and linear learning-rate annealing

It saves per-seed evaluation CSVs and plots an RLiable IQM curve.

Usage from repository root:
    python -m rl_exercises.examples.ppo_actor_critic_comparison

If you want to run it directly, the script prepends the repository root to sys.path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

# Ensure repository root is importable when running the script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_exercises.week_6.actor_critic import ActorCriticAgent, set_seed
from rl_exercises.week_6.ppo import PPOAgent


def load_ppo_config() -> Any:
    config_path = ROOT / "configs" / "agent" / "ppo.yaml"
    return OmegaConf.load(config_path)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    set_seed(env, seed)
    return env


def train_actor_critic(
    env_name: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    output_dir: Path,
) -> Path:
    env = make_env(env_name, seed)
    eval_env = make_env(env_name, seed + 1000)
    agent = ActorCriticAgent(
        env,
        lr_actor=5e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        seed=seed,
        hidden_size=128,
        baseline_type="value",
        baseline_decay=0.9,
    )

    records: list[dict[str, Any]] = []
    state, _ = env.reset(seed=seed)
    step_count = 0
    trajectory: list[tuple[np.ndarray, int, float, np.ndarray, bool, Any]] = []

    while step_count < total_steps:
        action, logp = agent.predict_action(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        trajectory.append((state, action, float(reward), next_state, done, logp))
        state = next_state
        step_count += 1

        if done or step_count >= total_steps:
            agent.update_agent(trajectory)
            trajectory = []
            state, _ = env.reset()

        if step_count % eval_interval == 0:
            mean_r, std_r = agent.evaluate(eval_env, num_episodes=eval_episodes)
            records.append(
                {
                    "algorithm": "ActorCritic",
                    "seed": seed,
                    "eval_step": step_count,
                    "eval_return": mean_r,
                    "eval_std": std_r,
                }
            )
            print(
                f"[Eval] ActorCritic seed={seed} step={step_count} return={mean_r:.2f} std={std_r:.2f}"
            )

    output_file = output_dir / f"actor_critic_seed_{seed}.csv"
    pd.DataFrame(records).to_csv(output_file, index=False)
    return output_file


def train_ppo(
    env_name: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    output_dir: Path,
    use_value_clip: bool = False,
    use_lr_annealing: bool = False,
    label: str = "PPO",
) -> Path:
    cfg = load_ppo_config()
    env = make_env(env_name, seed)
    eval_env = make_env(env_name, seed + 1000)
    agent = PPOAgent(
        env,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,
        gamma=cfg.agent.gamma,
        gae_lambda=cfg.agent.gae_lambda,
        clip_eps=cfg.agent.clip_eps,
        epochs=cfg.agent.epochs,
        batch_size=cfg.agent.batch_size,
        ent_coef=cfg.agent.ent_coef,
        vf_coef=cfg.agent.vf_coef,
        seed=seed,
        hidden_size=cfg.agent.hidden_size,
        use_value_clip=use_value_clip,
        use_lr_annealing=use_lr_annealing,
    )

    records: list[dict[str, Any]] = []
    state, _ = env.reset(seed=seed)
    step_count = 0
    trajectory: list[tuple[np.ndarray, int, torch.Tensor, torch.Tensor, float, float, np.ndarray]] = []

    while step_count < total_steps:
        action, logp, ent, val = agent.predict(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        trajectory.append((state, action, logp, ent, float(reward), float(done), next_state))
        state = next_state
        step_count += 1

        if done or step_count >= total_steps:
            agent.update(trajectory)
            trajectory = []
            state, _ = env.reset()

        if step_count % eval_interval == 0:
            mean_r, std_r = agent.evaluate(eval_env, num_episodes=eval_episodes)
            records.append(
                {
                    "algorithm": label,
                    "seed": seed,
                    "eval_step": step_count,
                    "eval_return": mean_r,
                    "eval_std": std_r,
                }
            )
            print(
                f"[Eval] {label} seed={seed} step={step_count} return={mean_r:.2f} std={std_r:.2f}"
            )

    output_file = output_dir / f"{label.lower().replace(' ', '_')}_seed_{seed}.csv"
    pd.DataFrame(records).to_csv(output_file, index=False)
    return output_file


def aggregate_results(result_dir: Path, algorithms: list[str], seeds: list[int]) -> None:
    eval_scores: dict[str, np.ndarray] = {}
    eval_steps: np.ndarray | None = None

    def filename_prefix(algo: str) -> str:
        if algo == "ActorCritic":
            return "actor_critic"
        return algo.lower().replace(" ", "_")

    for algo in algorithms:
        algo_dfs = []
        for seed in seeds:
            path = result_dir / f"{filename_prefix(algo)}_seed_{seed}.csv"
            df = pd.read_csv(path)
            if eval_steps is None:
                eval_steps = df["eval_step"].to_numpy()
            else:
                assert np.array_equal(eval_steps, df["eval_step"].to_numpy()), (
                    f"Mismatch in eval steps for algorithm {algo} seed {seed}"
                )
            algo_dfs.append(df["eval_return"].to_numpy())
        eval_scores[algo] = np.vstack(algo_dfs)

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
        algorithms=algorithms,
        xlabel="Training steps",
        ylabel="IQM evaluation return",
    )
    plt.legend(title="Algorithm", loc="best")
    plt.title("LunarLander-v3: PPO vs Actor-Critic")
    plt.tight_layout()
    figure_path = result_dir / "ppo_actor_critic_comparison_iqm.png"
    plt.savefig(figure_path)
    plt.close(fig)
    print(f"Saved RLiable plot to {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO vs Actor-Critic on LunarLander-v3 and plot RLiable comparison."
    )
    parser.add_argument("--env", default="LunarLander-v3", help="Gym environment")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds to run",
    )
    parser.add_argument("--steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--eval-interval", type=int, default=5000, help="Evaluation interval")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "examples" / "ppo_comparison_results",
        help="Directory to save CSV results and plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ["ActorCritic", "PPO", "PPO Enhanced"]
    for seed in args.seeds:
        print(f"Training ActorCritic seed={seed}")
        train_actor_critic(
            env_name=args.env,
            seed=seed,
            total_steps=args.steps,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            output_dir=args.output_dir,
        )

        print(f"Training PPO vanilla seed={seed}")
        train_ppo(
            env_name=args.env,
            seed=seed,
            total_steps=args.steps,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            output_dir=args.output_dir,
            use_value_clip=False,
            use_lr_annealing=False,
            label="PPO",
        )

        print(f"Training PPO enhanced seed={seed}")
        train_ppo(
            env_name=args.env,
            seed=seed,
            total_steps=args.steps,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            output_dir=args.output_dir,
            use_value_clip=True,
            use_lr_annealing=True,
            label="PPO Enhanced",
        )

    aggregate_results(args.output_dir, algorithms, args.seeds)


if __name__ == "__main__":
    main()
