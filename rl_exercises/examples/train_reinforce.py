"""train_reinforce.py

Run the REINFORCE experiments for the Week 5 level-2 tasks with a fixed
sequence of runs. This script is intentionally hardcoded and does not use
CLI arguments.

Tasks performed:
- baseline CartPole-v1 training
- trajectory-length sweep on CartPole-v1
- LunarLander-v2 training run
- optional DQN/Q-learning compare if result files are available

Results are saved under `outputs/reinforce_runs` and a summary note is
appended to `rl_exercises/week_5/observations_l2.txt`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import sys
from pathlib import Path

# Ensure repository root is on sys.path so imports work when running the script directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import pandas as pd
from rl_exercises.week_5.policy_gradient import REINFORCEAgent, set_seed


def train_episode(agent: REINFORCEAgent, env: gym.Env) -> float:
    state, _ = env.reset()
    batch = []
    episode_reward = 0.0

    while True:
        action, info = agent.predict_action(state, evaluate=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        batch.append((state, action, reward, next_state, terminated or truncated, info))
        episode_reward += float(reward)
        if terminated or truncated:
            break
        state = next_state

    agent.update_agent(batch)
    return episode_reward


def evaluate_episode(agent: REINFORCEAgent, env: gym.Env) -> float:
    state, _ = env.reset()
    episode_reward = 0.0

    while True:
        action, _ = agent.predict_action(state, evaluate=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += float(reward)
        if terminated or truncated:
            break
        state = next_state

    return episode_reward


def run_single_seed(
    env_name: str,
    seed: int,
    episodes: int,
    eval_interval: int,
    eval_episodes: int,
    lr: float,
    gamma: float,
    hidden_size: int,
    max_episode_steps: Optional[int],
    output_dir: Path,
) -> Dict[str, List[float]]:
    env = gym.make(env_name)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    set_seed(env, seed)

    agent = REINFORCEAgent(env, lr=lr, gamma=gamma, seed=seed, hidden_size=hidden_size)

    train_rewards: List[float] = []
    eval_rewards: List[float] = []
    eval_episodes_list: List[int] = []

    for episode in range(episodes):
        tr = train_episode(agent, env)
        train_rewards.append(tr)
        if (episode + 1) % eval_interval == 0:
            evs = [evaluate_episode(agent, env) for _ in range(eval_episodes)]
            eval_rewards.append(float(np.mean(evs)))
            eval_episodes_list.append(episode + 1)

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"episode": range(1, len(train_rewards) + 1), "reward": train_rewards}
    ).to_csv(seed_dir / "train_rewards.csv", index=False)
    pd.DataFrame({"episode": eval_episodes_list, "reward": eval_rewards}).to_csv(
        seed_dir / "eval_rewards.csv", index=False
    )

    env.close()
    return {"train": train_rewards, "eval": eval_rewards}


def compare_with_dqn(target_reward: float = 350.0) -> Optional[Dict[str, float]]:
    base = ROOT / "results"
    reward_columns = [
        "reward",
        "train_rewards",
        "train_reward",
        "return",
        "episode_reward",
    ]
    for sub in ("qlearning", "dqn"):
        path = base / sub
        if not path.exists():
            continue
        for seed_dir in path.rglob("seed_*"):
            tr_file = seed_dir / "train_rewards.csv"
            if not tr_file.exists():
                continue
            try:
                df = pd.read_csv(tr_file)
            except Exception:
                continue
            col = None
            for candidate in reward_columns:
                if candidate in df.columns:
                    col = candidate
                    break
            if col is None:
                continue
            episodes = df.index.to_numpy()
            if "episode" in df.columns:
                episodes = df["episode"].to_numpy()
            elif "steps" in df.columns:
                episodes = df["steps"].to_numpy()
            meets = df[df[col] >= target_reward]
            if not meets.empty:
                first_row = meets.iloc[0]
                episode_value = (
                    int(first_row["episode"])
                    if "episode" in df.columns
                    else int(first_row.name)
                )
                return {sub: float(episode_value)}
    return None


def main() -> None:
    base_output = ROOT / "outputs" / "reinforce_runs"
    base_output.mkdir(parents=True, exist_ok=True)

    # Task 1: baseline CartPole-v1
    baseline_dir = base_output / "cartpole_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    print("=== Task 1: CartPole-v1 baseline training ===")
    baseline_summary: Dict[str, List[float]] = {}
    for seed in (0, 1, 2):
        print(f"Running baseline seed={seed}")
        result = run_single_seed(
            env_name="CartPole-v1",
            seed=seed,
            episodes=300,
            eval_interval=50,
            eval_episodes=5,
            lr=1e-2,
            gamma=0.99,
            hidden_size=128,
            max_episode_steps=None,
            output_dir=baseline_dir,
        )
        baseline_summary[f"seed_{seed}"] = result

    # Task 2: trajectory-length sweep on CartPole-v1
    sweep_dir = base_output / "cartpole_trajectory_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_lengths = [50, 100, 200, 500]
    sweep_summary: Dict[str, Dict[str, List[float]]] = {}
    print("\n=== Task 2: Trajectory-length sweep ===")
    for max_steps in sweep_lengths:
        tag_dir = sweep_dir / f"max_steps_{max_steps}"
        tag_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running sweep max_episode_steps={max_steps}")
        sweep_summary[f"max_steps_{max_steps}"] = {}
        for seed in (0, 1):
            print(f"  seed={seed}")
            result = run_single_seed(
                env_name="CartPole-v1",
                seed=seed,
                episodes=200,
                eval_interval=50,
                eval_episodes=5,
                lr=1e-2,
                gamma=0.99,
                hidden_size=128,
                max_episode_steps=max_steps,
                output_dir=tag_dir,
            )
            sweep_summary[f"max_steps_{max_steps}"][f"seed_{seed}"] = result

    # Task 3: LunarLander-v2 experiments
    lunar_dir = base_output / "lunar_lander"
    lunar_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== Task 3: LunarLander-v2 training ===")
    lunar_summary: Dict[str, Dict[str, List[float]]] = {}
    for seed in (0, 1):
        print(f"Running LunarLander-v3 seed={seed}")
        result = run_single_seed(
            env_name="LunarLander-v3",
            seed=seed,
            episodes=300,
            eval_interval=50,
            eval_episodes=5,
            lr=1e-3,
            gamma=0.99,
            hidden_size=256,
            max_episode_steps=None,
            output_dir=lunar_dir,
        )
        lunar_summary[f"seed_{seed}"] = result

    # Optional comparison with existing DQN/Q-learning results
    print("\n=== Task 4: DQN/Q-learning comparison (if available) ===")
    dqn_comparison = compare_with_dqn()
    print("DQN/Q-learning comparison result:", dqn_comparison)

    # Log experiment summary into observations file
    obs_file = ROOT / "rl_exercises" / "week_5" / "observations_l2.txt"
    with obs_file.open("a", encoding="utf-8") as fh:
        fh.write("\n---\n")
        fh.write("Automatic REINFORCE experiment run\n")
        fh.write(
            "Baseline: CartPole-v1, seeds=0,1,2, episodes=300, lr=1e-2, gamma=0.99\n"
        )
        fh.write("Trajectory sweep: CartPole-v1, max_steps=50,100,200,500, seeds=0,1\n")
        fh.write("LunarLander-v2: seeds=0,1, episodes=300, lr=1e-3, hidden_size=256\n")
        fh.write(f"DQN/Q-learning comparison: {dqn_comparison}\n")
        fh.write(f"Results saved under: {base_output.resolve()}\n")

    print("\nAll tasks completed. Results saved in:")
    print(" ", base_output.resolve())


if __name__ == "__main__":
    main()
