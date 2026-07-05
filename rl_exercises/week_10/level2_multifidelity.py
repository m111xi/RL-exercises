"""
Level 2: Multi-fidelity in RL

The idea of multi-fidelity HPO: first train lots of configs only for a short
time (low fidelity = few steps), throw away the bad ones, and only keep
training the good ones. This saves budget - BUT only if early success also
means late success.

A setup where early scores can be MISLEADING:
- The RND exploration bonus (rnd_reward_weight) helps in the long run for
  exploring, but early on it often slows learning down because the agent
  spends time in "novel" states instead of good ones.
- A config WITHOUT the exploration bonus often looks better early but can
  plateau later.
- So successive halving might kill good (explorative) configs too early.
"""

import random
import gymnasium as gym
import numpy as np
from rnd_dqn import RNDDQNAgent, train, evaluate

ENV_NAME = "CartPole-v1"
#budgets (steps) for the fidelity stages
LOW_FIDELITY = 1500
MID_FIDELITY = 4000
HIGH_FIDELITY = 9000

def make_env():
    return gym.make(ENV_NAME)

#A few hand picked configs, some with and some without the exploration bonus
CANDIDATES = [
    {"name": "no_rnd_lr1e-2",   "lr": 1e-2, "rnd_reward_weight": 0.0},
    {"name": "no_rnd_lr1e-3",   "lr": 1e-3, "rnd_reward_weight": 0.0},
    {"name": "rnd_lr1e-3",      "lr": 1e-3, "rnd_reward_weight": 0.5},
    {"name": "rnd_lr5e-4",      "lr": 5e-4, "rnd_reward_weight": 0.5},
    {"name": "rnd_strong",      "lr": 1e-3, "rnd_reward_weight": 1.0},
    {"name": "no_rnd_lr5e-4",   "lr": 5e-4, "rnd_reward_weight": 0.0},
]

#drop the "name" key before passing to the agent
def run(config: dict, steps: int, seed: int = 0) -> float:
    cfg = {k: v for k, v in config.items() if k != "name"}
    env = make_env()
    eval_env = make_env()
    agent = RNDDQNAgent(env, seed=seed, **cfg)
    train(env, agent, steps, seed=seed, eval_env=None)
    return evaluate(eval_env, agent, episodes=5, seed=seed)

def successive_halving(seed: int = 0):
    #First all configs short, then top half medium, then best one long
    history = {"low": {}, "mid": {}, "high": {}}

    #stage 1: everyone at low fidelity
    for c in CANDIDATES:
        history["low"][c["name"]] = run(c, LOW_FIDELITY, seed)
    #keep the better half
    ranked = sorted(CANDIDATES, key=lambda c: history["low"][c["name"]], reverse=True)
    survivors_mid = ranked[: len(ranked) // 2]

    #stage 2: medium fidelity
    for c in survivors_mid:
        history["mid"][c["name"]] = run(c, MID_FIDELITY, seed)
    ranked_mid = sorted(survivors_mid, key=lambda c: history["mid"][c["name"]], reverse=True)
    survivors_high = ranked_mid[: max(1, len(ranked_mid) // 2)]

    #stage 3: high fidelity
    for c in survivors_high:
        history["high"][c["name"]] = run(c, HIGH_FIDELITY, seed)
    return history

def full_reference(seed: int = 0) -> dict:
    #Train ALL configs at full budget -> the true ranking (expensive, just for comparison)
    return {c["name"]: run(c, HIGH_FIDELITY, seed) for c in CANDIDATES}

def main() -> None:
    seed = 0
    print("Running successive halving ...")
    hist = successive_halving(seed)
    print("Reference: training all configs at full budget ...")
    ref = full_reference(seed)

    lines = []
    lines.append("Low-fidelity ranking (after few steps):")
    for name, sc in sorted(hist["low"].items(), key=lambda x: -x[1]):
        lines.append(f"  {name:18s} {sc:6.1f}")
    lines.append("")
    lines.append("Full ranking (all at high fidelity, reference):")
    for name, sc in sorted(ref.items(), key=lambda x: -x[1]):
        lines.append(f"  {name:18s} {sc:6.1f}")
    lines.append("")
    sh_winner = max(hist["high"], key=hist["high"].get)
    true_winner = max(ref, key=ref.get)
    lines.append(f"Winner according to successive halving: {sh_winner}")
    lines.append(f"True winner (full budget):             {true_winner}")
    lines.append("")

    # did the true winner get killed early?
    low_rank = sorted(hist["low"], key=hist["low"].get, reverse=True)
    pos = low_rank.index(true_winner) + 1
    lines.append(
        f"The true winner was ranked #{pos} out of {len(CANDIDATES)} in the "
        f"low-fidelity ranking."
    )
    with open("level2_results.txt", "w") as f:
        f.write("Level 2: Multi-fidelity in RL\n")
        f.write("\n".join(lines) + "\n")
    print("\nDone level2_results.txt")

if __name__ == "__main__":
    main()