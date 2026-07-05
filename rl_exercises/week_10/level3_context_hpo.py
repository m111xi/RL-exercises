"""
Level 3: Context & hyperparameters

Reference: the paper "Hyperparameters in Contextual RL are Highly Situational"
(Eimer et al., 2022) shows that the best hyperparameters depend a lot on the
context of the environment - a config that's great in one context can be bad
in another.

We reproduce the core idea on a small scale:
- We build contextual CartPole variants by changing gravity (low / normal /
  high). That's our "context".
- For each context we do a small HPO (random search) and see which config wins.
- Then we test: how good is the "winner config from context A" in the other
  contexts (cross-context transfer)?
"""

import random
import gymnasium as gym
import numpy as np
from rnd_dqn import RNDDQNAgent, train, evaluate

TRAIN_STEPS = 7000
N_TRIALS = 5
SEED = 0

#contexts: different gravity values in CartPole
CONTEXTS = {
    "low_gravity":  4.9,
    "normal":       9.8,
    "high_gravity": 15.0,
}

def make_env(gravity: float):
    env = gym.make("CartPole-v1")
    #gravity lives on the unwrapped env
    env.unwrapped.gravity = gravity
    return env

def sample_config(rng: random.Random) -> dict:
    return {
        "lr": rng.choice([1e-2, 5e-3, 1e-3, 5e-4]),
        "gamma": rng.choice([0.95, 0.98, 0.99]),
        "epsilon_decay": rng.choice([1000, 2000, 4000]),
        "rnd_reward_weight": rng.choice([0.0, 0.1, 0.5]),
    }

def run(config: dict, gravity: float, seed: int = SEED) -> float:
    env = make_env(gravity)
    eval_env = make_env(gravity)
    agent = RNDDQNAgent(env, seed=seed, **config)
    train(env, agent, TRAIN_STEPS, seed=seed, eval_env=None)
    return evaluate(eval_env, agent, episodes=5, seed=seed)

def optimize_for_context(gravity: float, rng: random.Random):
    #small random search inside one context
    best_cfg, best_score = None, -np.inf
    for _ in range(N_TRIALS):
        cfg = sample_config(rng)
        sc = run(cfg, gravity)
        if sc > best_score:
            best_score, best_cfg = sc, cfg
    return best_cfg, best_score

def main() -> None:
    rng = random.Random(2024)
    lines = []
    best_per_context = {}
    #HPO per context
    for name, g in CONTEXTS.items():
        print(f"HPO for context {name} (g={g}) ...")
        cfg, sc = optimize_for_context(g, rng)
        best_per_context[name] = cfg
        lines.append(f"Context {name}: best config score={sc:.1f}")
        lines.append(f"    {cfg}")
    lines.append("")

    #cross-context transfer matrix: config from row, tested in column
    print("Building transfer matrix ...")
    names = list(CONTEXTS.keys())
    matrix = {}
    for src in names:
        matrix[src] = {}
        for tgt in names:
            matrix[src][tgt] = run(best_per_context[src], CONTEXTS[tgt])

    #write the table
    header = "Config\\Test    " + "  ".join(f"{n:>12s}" for n in names)
    lines.append("Transfer matrix (row=tuned in, column=tested in):")
    lines.append(header)
    for src in names:
        row = f"{src:14s} " + "  ".join(f"{matrix[src][tgt]:12.1f}" for tgt in names)
        lines.append(row)
    lines.append("")

    #check: is the config tuned in a context also the best one there?
    situational = False
    for tgt in names:
        best_src = max(names, key=lambda s: matrix[s][tgt])
        native = matrix[tgt][tgt]
        best_val = matrix[best_src][tgt]
        lines.append(
            f"In context {tgt}: native config={native:.1f}, "
            f"best foreign ({best_src})={best_val:.1f}"
        )
        if best_src != tgt and best_val > native + 5:
            situational = True

    lines.append("")
    #do the best configs even differ?
    unique_cfgs = {str(sorted(c.items())) for c in best_per_context.values()}

    with open("level3_results.txt", "w") as f:
        f.write("Level 3: Context & hyperparameters\n")
        f.write("\n".join(lines) + "\n")
    print("\nDone level3_results.txt")

if __name__ == "__main__":
    main()