"""
Level 1: How well does HPO generalize in RL?

Setup (generalization across seeds):
- We tune the RND-DQN hyperparameters on CartPole using ONE training seed
  (random search).
- Then we take the best config we found AND a default config and test both
  on several NEW seeds.
- Question: does the config tuned on one seed also do better on unseen seeds,
  or did we just overfit to that single seed?
"""

import random
import gymnasium as gym
import numpy as np
from rnd_dqn import RNDDQNAgent, train, evaluate

ENV_NAME = "CartPole-v1"
TRAIN_STEPS = 4000        # keep it small so runs are fast
N_TRIALS = 5              # random search budget
OPT_SEED = 0             # seed we tune on
TEST_SEEDS = [10, 11, 12, 13]  # unseen seeds

def make_env():
    return gym.make(ENV_NAME)

def sample_config(rng: random.Random) -> dict:
    # just pick random values from a small grid
    return {
        "lr": rng.choice([1e-2, 5e-3, 1e-3, 5e-4]),
        "gamma": rng.choice([0.95, 0.98, 0.99]),
        "epsilon_decay": rng.choice([1000, 2000, 4000]),
        "rnd_reward_weight": rng.choice([0.0, 0.1, 0.5]),
        "target_update_freq": rng.choice([250, 500, 1000]),
    }

DEFAULT_CONFIG = {
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_decay": 2000,
    "rnd_reward_weight": 0.1,
    "target_update_freq": 500,
}

def run_config(config: dict, seed: int) -> float:
    # train one agent with this config on this seed, return eval return
    env = make_env()
    eval_env = make_env()
    agent = RNDDQNAgent(env, seed=seed, **config)
    train(env, agent, TRAIN_STEPS, seed=seed, eval_env=None)
    return evaluate(eval_env, agent, episodes=5, seed=seed)


def main() -> None:
    rng = random.Random(123)
    lines = []

    #random search on a single seed
    print("Running random search ...")
    best_config, best_score = None, -np.inf
    for t in range(N_TRIALS):
        cfg = sample_config(rng)
        score = run_config(cfg, seed=OPT_SEED)
        print(f"  Trial {t}: score={score:.1f}  {cfg}")
        lines.append(f"Trial {t}: score={score:.1f}  {cfg}")
        if score > best_score:
            best_score, best_config = score, cfg

    lines.append("")
    lines.append(f"Best config (on seed {OPT_SEED}): {best_config}")
    lines.append(f"Score on tuning seed: {best_score:.1f}")
    lines.append("")

    #generalization: test on new seeds
    print("Testing on unseen seeds ...")
    best_test, default_test = [], []
    for s in TEST_SEEDS:
        b = run_config(best_config, seed=s)
        d = run_config(DEFAULT_CONFIG, seed=s)
        best_test.append(b)
        default_test.append(d)
        print(f"  Seed {s}: tuned={b:.1f}  default={d:.1f}")
        lines.append(f"Seed {s}: tuned={b:.1f}  default={d:.1f}")

    lines.append("")
    lines.append(
        f"Tuned    over test seeds: mean={np.mean(best_test):.1f} "
        f"+/- {np.std(best_test):.1f}"
    )
    lines.append(
        f"Default  over test seeds: mean={np.mean(default_test):.1f} "
        f"+/- {np.std(default_test):.1f}"
    )

    with open("level1_results.txt", "w") as f:
        f.write("Level 1: HPO generalization (across seeds)\n")
        f.write("\n".join(lines) + "\n")
    print("\nDone level1_results.txt")

if __name__ == "__main__":
    main()