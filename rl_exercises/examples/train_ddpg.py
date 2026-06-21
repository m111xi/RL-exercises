"""Train and compare DDPG against the Week 5 REINFORCE implementation.

This script trains a simple DDPG agent on a continuous environment and
saves training/evaluation rewards. It also appends a short note to the
Level 3 observations file.
"""

from __future__ import annotations

from typing import Dict, List

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from rl_exercises.week_5.ddpg import DDPGAgent
from rl_exercises.week_5.policy_gradient import set_seed


def train_episode(agent: DDPGAgent, env: gym.Env) -> float:
    state, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = agent.predict_action(state, evaluate=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(state, action, float(reward), next_state, done, {})
        agent.update_agent()
        total_reward += float(reward)
        state = next_state
    return total_reward


def evaluate_episode(agent: DDPGAgent, env: gym.Env) -> float:
    state, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = agent.predict_action(state, evaluate=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        state = next_state
    return total_reward


def run_ddpg(env_name: str, episodes: int, output_dir: Path) -> Dict[str, List[float]]:
    env = gym.make(env_name)
    set_seed(env, 0)
    agent = DDPGAgent(
        env,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100000,
        batch_size=64,
        noise_std=0.2,
        noise_decay=0.995,
        min_noise=0.01,
        hidden_size=256,
        seed=0,
    )

    train_rewards: List[float] = []
    eval_rewards: List[float] = []
    eval_episodes_list: List[int] = []

    for episode in range(1, episodes + 1):
        ep_reward = train_episode(agent, env)
        train_rewards.append(ep_reward)

        if episode % 10 == 0:
            evals = [evaluate_episode(agent, env) for _ in range(3)]
            eval_rewards.append(float(np.mean(evals)))
            eval_episodes_list.append(episode)
            print(
                f"Episode {episode:3d} | Train {ep_reward:7.2f} | Eval {np.mean(evals):7.2f}"
            )

    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"episode": list(range(1, len(train_rewards) + 1)), "reward": train_rewards}
    ).to_csv(output_dir / "train_rewards.csv", index=False)
    pd.DataFrame({"episode": eval_episodes_list, "reward": eval_rewards}).to_csv(
        output_dir / "eval_rewards.csv", index=False
    )

    return {"train": train_rewards, "eval": eval_rewards}


def main() -> None:
    base_output = Path("outputs") / "ddpg_runs"
    base_output.mkdir(parents=True, exist_ok=True)

    env_name = "Pendulum-v1"
    episodes = 200
    print(f"Training DDPG on {env_name} for {episodes} episodes")
    summary = run_ddpg(env_name, episodes, base_output / "pendulum")

    obs_file = Path("rl_exercises") / "week_5" / "observations_l3.txt"
    with obs_file.open("a", encoding="utf-8") as fh:
        fh.write("\n---\n")
        fh.write(
            f"DDPG run: {env_name}, episodes={episodes}, lr_actor=1e-3, lr_critic=1e-3, hidden_size=256\n"
        )
        fh.write(
            f"Final evaluation rewards: {summary['eval'][-3:] if summary['eval'] else []}\n"
        )
        fh.write(f"Results saved under: {base_output.resolve()}\n")

    print("Done. Results saved to", base_output.resolve())


if __name__ == "__main__":
    main()
