"""
Compare REINFORCE (Gaussian policy) vs DDPG on Pendulum-v1.

This script trains each agent separately for the same number of episodes
and saves training/evaluation rewards to CSV for direct comparison.
"""

from __future__ import annotations

from typing import List

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rl_exercises.week_5.ddpg import DDPGAgent
from rl_exercises.week_5.policy_gradient import REINFORCEGaussianAgent, set_seed


def train_reinforce_episode(agent: REINFORCEGaussianAgent, env: gym.Env) -> float:
    state, _ = env.reset()
    batch = []
    ep_reward = 0.0
    done = False
    while not done:
        action, info = agent.predict_action(state, evaluate=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        batch.append((state, action, float(reward), next_state, done, info))
        ep_reward += float(reward)
        state = next_state

    agent.update_agent(batch)
    return ep_reward


def evaluate_reinforce(
    agent: REINFORCEGaussianAgent, env: gym.Env, n_eval: int = 3
) -> float:
    vals = []
    for _ in range(n_eval):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = agent.predict_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            state = next_state
        vals.append(ep_reward)
    return float(np.mean(vals))


def train_ddpg_episode(agent: DDPGAgent, env: gym.Env) -> float:
    state, _ = env.reset()
    ep_reward = 0.0
    done = False
    while not done:
        action, _ = agent.predict_action(state, evaluate=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(state, action, float(reward), next_state, done, {})
        agent.update_agent()
        ep_reward += float(reward)
        state = next_state
    return ep_reward


def evaluate_ddpg(agent: DDPGAgent, env: gym.Env, n_eval: int = 3) -> float:
    vals = []
    for _ in range(n_eval):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = agent.predict_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            state = next_state
        vals.append(ep_reward)
    return float(np.mean(vals))


def main() -> None:
    env_name = "Pendulum-v1"
    episodes = 200
    eval_interval = 10
    eval_n = 3
    seed = 42

    output_dir = Path("outputs") / "comparison_reinforce_ddpg"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Comparing REINFORCE (Gaussian) vs DDPG on {env_name} for {episodes} episodes\n"
    )

    # REINFORCE (Gaussian)
    env_r = gym.make(env_name)
    set_seed(env_r, seed)
    reinforce_agent = REINFORCEGaussianAgent(
        env_r, lr=1e-3, gamma=0.99, seed=seed, hidden_size=128
    )

    reinforce_train: List[float] = []
    reinforce_eval: List[float] = []
    reinforce_eval_episodes: List[int] = []

    print("Training REINFORCE (Gaussian)...")
    for ep in range(1, episodes + 1):
        tr = train_reinforce_episode(reinforce_agent, env_r)
        reinforce_train.append(tr)
        if ep % eval_interval == 0:
            ev = evaluate_reinforce(reinforce_agent, env_r, n_eval=eval_n)
            reinforce_eval.append(ev)
            reinforce_eval_episodes.append(ep)
    env_r.close()

    # DDPG
    env_d = gym.make(env_name)
    set_seed(env_d, seed)
    ddpg_agent = DDPGAgent(
        env_d,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        hidden_size=128,
        seed=seed,
    )

    ddpg_train: List[float] = []
    ddpg_eval: List[float] = []
    ddpg_eval_episodes: List[int] = []

    print("Training DDPG...")
    for ep in range(1, episodes + 1):
        tr = train_ddpg_episode(ddpg_agent, env_d)
        ddpg_train.append(tr)
        if ep % eval_interval == 0:
            ev = evaluate_ddpg(ddpg_agent, env_d, n_eval=eval_n)
            ddpg_eval.append(ev)
            ddpg_eval_episodes.append(ep)
    env_d.close()

    # Save results
    comparison_df = pd.DataFrame(
        {
            "episode": reinforce_eval_episodes,
            "reinforce_eval": reinforce_eval,
            "ddpg_eval": ddpg_eval,
        }
    )
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)

    reinforce_df = pd.DataFrame(
        {
            "episode": list(range(1, len(reinforce_train) + 1)),
            "train_reward": reinforce_train,
        }
    )
    reinforce_df.to_csv(output_dir / "reinforce_train.csv", index=False)

    ddpg_df = pd.DataFrame(
        {
            "episode": list(range(1, len(ddpg_train) + 1)),
            "train_reward": ddpg_train,
        }
    )
    ddpg_df.to_csv(output_dir / "ddpg_train.csv", index=False)

    # Save plots
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        reinforce_df["episode"],
        reinforce_df["train_reward"],
        label="REINFORCE Gaussian",
        color="tab:blue",
    )
    ax.plot(
        ddpg_df["episode"], ddpg_df["train_reward"], label="DDPG", color="tab:orange"
    )
    ax.set_title("Training Rewards: REINFORCE Gaussian vs DDPG")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        reinforce_eval_episodes,
        reinforce_eval,
        label="REINFORCE Gaussian",
        marker="o",
        color="tab:blue",
    )
    ax.plot(ddpg_eval_episodes, ddpg_eval, label="DDPG", marker="o", color="tab:orange")
    ax.set_title("Evaluation Rewards: REINFORCE Gaussian vs DDPG")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Mean Reward over {eval_n} evals")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_curve.png")
    plt.close(fig)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Episodes: {episodes}")
    print()
    print("Final 3 Evaluation Rewards:")
    print(f"  REINFORCE: {reinforce_eval[-3:]}")
    print(f"  DDPG:      {ddpg_eval[-3:]}")
    print()
    print("Mean eval (last 3 evals):")
    print(f"  REINFORCE: {np.mean(reinforce_eval[-3:]):.2f}")
    print(f"  DDPG:      {np.mean(ddpg_eval[-3:]):.2f}")
    print()
    print(f"Results saved to: {output_dir}")
    print("  - comparison.csv (side-by-side eval comparison)")
    print("  - reinforce_train.csv, ddpg_train.csv (training curves)")
    print("  - training_curve.png")
    print("  - evaluation_curve.png")


if __name__ == "__main__":
    main()
