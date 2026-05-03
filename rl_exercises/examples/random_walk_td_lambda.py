"""TD(lambda) evaluation on the RandomWalk environment.

This script trains a tabular TD(lambda) value estimator on the RandomWalk
environment from `rl_exercises.environments`.

The setup follows the classical random-walk control experiment:
- 100 independent training sets
- each training set consists of 10 episodes
- random policy generated episodes
- lambda values: 1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0

The script computes the RMS error between the true state values and the
learned state-value estimates for the nonterminal states.
"""

from __future__ import annotations

from typing import Iterable

import os

import matplotlib.pyplot as plt
import numpy as np
from rl_exercises.environments import RandomWalk

TRUE_STATE_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
TERMINAL_STATES = {0, 4}
LAMBDAS = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
N_RUNS = 100
EPISODES_PER_RUN = 10
ALPHA = 0.1
GAMMA = 1.0
RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results", "random_walk_td_lambda")
)


def generate_episode(
    env: RandomWalk, rng: np.random.Generator
) -> list[tuple[int, float, int]]:
    state, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
    transitions: list[tuple[int, float, int]] = []
    done = False
    while not done:
        action = int(rng.integers(env.action_space.n))
        next_state, reward, terminated, truncated, _ = env.step(action)
        transitions.append((state, float(reward), next_state))
        state = next_state
        done = terminated or truncated
    return transitions


def td_lambda_update(
    V: np.ndarray,
    transitions: Iterable[tuple[int, float, int]],
    lambda_: float,
    alpha: float,
    gamma: float,
) -> None:
    n_states = len(V)
    traces = np.zeros(n_states, dtype=float)
    for state, reward, next_state in transitions:
        if state in TERMINAL_STATES:
            continue

        delta = reward + gamma * V[next_state] - V[state]
        traces[state] += 1.0
        V += alpha * delta * traces
        traces *= gamma * lambda_

        V[0] = 0.0
        V[4] = 1.0


def run_experiment(
    lambda_values: list[float], n_runs: int, episodes: int, alpha: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    all_mean_rmse = np.zeros((len(lambda_values),), dtype=float)
    average_values = np.zeros((len(lambda_values), len(TRUE_STATE_VALUES)), dtype=float)

    for i, lambda_ in enumerate(lambda_values):
        run_values = np.zeros((n_runs, len(TRUE_STATE_VALUES)), dtype=float)
        rmse_values = np.zeros((n_runs,), dtype=float)

        for run in range(n_runs):
            rng = np.random.default_rng(run)
            env = RandomWalk(seed=run)
            V = np.zeros_like(TRUE_STATE_VALUES)
            V[4] = 1.0

            for _ in range(episodes):
                transitions = generate_episode(env, rng)
                td_lambda_update(V, transitions, lambda_, alpha, gamma)

            run_values[run] = V
            rmse_values[run] = np.sqrt(np.mean((V[1:4] - TRUE_STATE_VALUES[1:4]) ** 2))

        average_values[i] = np.mean(run_values, axis=0)
        all_mean_rmse[i] = np.mean(rmse_values)

    return all_mean_rmse, average_values


def plot_results(
    lambda_values: list[float], rmse: np.ndarray, average_values: np.ndarray
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(lambda_values, rmse, marker="o", linewidth=2)
    plt.title("RMS Error after 10 Episodes")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("RMSE on nonterminal states")
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    states = np.arange(len(TRUE_STATE_VALUES))
    plt.plot(states, TRUE_STATE_VALUES, "k--", marker="o", label="True values")
    for i, lambda_ in enumerate(lambda_values):
        if lambda_ in {0.0, 0.3, 0.7, 1.0}:
            plt.plot(
                states,
                average_values[i],
                marker="x",
                label=f"learned $\lambda$={lambda_}",
            )
    plt.title("True vs Learned State Values")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "td_lambda_random_walk.png")
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved plot to: {out_path}")

    summary_path = os.path.join(RESULTS_DIR, "td_lambda_summary.npy")
    np.save(
        summary_path, {"lambdas": lambda_values, "rmse": rmse, "values": average_values}
    )
    print(f"Saved raw results to: {summary_path}")


def main() -> None:
    print("Running TD(lambda) on RandomWalk...")
    print(f"Lambda values: {LAMBDAS}")
    rmse, average_values = run_experiment(
        LAMBDAS, N_RUNS, EPISODES_PER_RUN, ALPHA, GAMMA
    )
    for lambda_, error in zip(LAMBDAS, rmse):
        print(f"lambda={lambda_:0.1f}: RMSE={error:.4f}")
    plot_results(LAMBDAS, rmse, average_values)


if __name__ == "__main__":
    main()
