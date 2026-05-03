# Ignore "imported but unused"
# flake8: noqa: F401
from typing import Any, List, SupportsFloat

import os
import warnings
from functools import partial

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import rl_exercises
from gymnasium.core import Env
from gymnasium.wrappers import TimeLimit
from hydra.utils import get_class
from minigrid.wrappers import FlatObsWrapper
from omegaconf import DictConfig, OmegaConf
from rich import print as printr
from rl_exercises.agent import AbstractAgent, RandomAgent
from rl_exercises.agent.buffer import SimpleBuffer
from rl_exercises.environments import (
    ContextualMarsRover,
    MarsRover,
    RoundRobinJointContextWrapper,
)
from rl_exercises.week_2.context_sets import (
    default_three_by_three_example,
    summarize_protocol,
)
from rl_exercises.week_2.policy_iteration import PolicyIteration
from rl_exercises.week_2.value_iteration import ValueIteration
from rl_exercises.week_3 import TDAgent
from rl_exercises.week_3.epsilon_greedy_policy import EpsilonGreedyPolicy

# from rl_exercises.week_4 import EpsilonGreedyPolicy as TabularEpsilonGreedyPolicy
# from rl_exercises.week_4 import SARSAAgent
# from rl_exercises.week_5 import EpsilonGreedyPolicy, TabularQAgent, VFAQAgent
# from rl_exercises.week_6 import DQN, ReplayBuffer
# from rl_exercises.week_7 import REINFORCE
# from rl_exercises.week_8 import EpsilonDecayPolicy, EZGreedyPolicy
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm


@hydra.main("configs", "base", version_base="1.1")  # type: ignore[misc]
def train(cfg: DictConfig) -> float:
    """Train the agent.

    Parameters
    ----------
    cfg : DictConfig
        Agent/experiment configuration

    Returns
    -------
    float
        Mean return of n eval episodes

    Raises
    ------
    NotImplementedError
        _description_
    """
    env = make_env(cfg.env_name, cfg.env_kwargs)
    printr(cfg)
    if cfg.agent == "sb3":
        return train_sb3(env, cfg)
    elif cfg.agent == "random":
        agent = RandomAgent(env)
    else:
        agent = eval(cfg.agent_class)(  # type: ignore
            env,
            policy=EpsilonGreedyPolicy(env, epsilon=cfg.epsilon, seed=cfg.seed),
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            algorithm=cfg.algorithm,
        )

    buffer_cls = eval(cfg.buffer_cls)
    buffer = buffer_cls(**cfg.buffer_kwargs)
    state, info = env.reset(seed=cfg.seed)
    train_reward_buffer = {"steps": [], "train_rewards": []}
    eval_reward_buffer = {"eval_steps": [], "eval_rewards": []}

    for step in range(int(cfg.training_steps)):
        action, info = agent.predict_action(state, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        buffer.add(state, action, reward, next_state, (truncated or terminated), info)
        train_reward_buffer["steps"].append(step)
        train_reward_buffer["train_rewards"].append(reward)

        if len(buffer) > cfg.batch_size or (
            cfg.update_after_episode_end and (terminated or truncated)
        ):
            batch = buffer.sample(cfg.batch_size)
            agent.update_agent(batch)

        state = next_state

        if terminated or truncated:
            state, info = env.reset(seed=cfg.seed)

        if step % cfg.eval_every_n_steps == 0:
            eval_performance = evaluate(
                make_env(cfg.env_name, cfg.env_kwargs, for_evaluation=True),
                agent,
                cfg.n_eval_episodes,
                cfg.seed,
            )
            print(f"Eval reward after {step} steps was {eval_performance}.")
            eval_reward_buffer["eval_steps"].append(step)
            eval_reward_buffer["eval_rewards"].append(eval_performance)

    agent.save(str(os.path.abspath("model.csv")))
    pd.DataFrame(train_reward_buffer).to_csv(
        os.path.abspath("train_rewards.csv"), index=False
    )
    pd.DataFrame(eval_reward_buffer).to_csv(
        os.path.abspath("eval_rewards.csv"), index=False
    )
    final_eval = evaluate(
        make_env(cfg.env_name, cfg.env_kwargs, for_evaluation=True),
        agent,
        cfg.n_eval_episodes,
    )
    print(f"Final eval reward was: {final_eval}")
    return final_eval


def train_sb3(env: gym.Env, cfg: DictConfig) -> float:
    """Train stablebaselines agent on env.

    Parameters
    ----------
    env : gym.Env
        Environment
    cfg : DictConfig
        Agent/experiment configuration

    Returns
    -------
    float
        Mean rewards
    """
    # Create agent
    model = eval(cfg.agent_class)(
        "MlpPolicy",
        env,
        verbose=cfg.verbose,
        tensorboard_log=cfg.log_dir,
        seed=cfg.seed,
        **cfg.agent_kwargs,
    )

    # Train agent
    model.learn(total_timesteps=cfg.total_timesteps)

    # Save agent
    model.save(cfg.model_fn)

    # Evaluate
    env = Monitor(gym.make(cfg.env_id), filename="eval")
    means = evaluate(env, model, episodes=cfg.n_eval_episodes, seed=cfg.seed)
    performance = np.mean(means)
    return performance


def evaluate(
    env: gym.Env, agent: AbstractAgent, episodes: int = 100, seed: int = 0
) -> float:
    """Evaluate a given Policy on an Environment.

    Parameters
    ----------
    env: gym.Env
        Environment to evaluate on
    policy: Callable[[np.ndarray], int]
        Policy to evaluate
    episodes: int
        Evaluation episodes

    Returns
    -------
    float
        Mean evaluation rewards
    """
    episode_rewards: List[float] = []
    pbar = tqdm(total=episodes)
    for _ in range(episodes):
        obs, info = env.reset(seed=seed)
        episode_rewards.append(0)
        done = False
        episode_steps = 0
        while not done:
            action, _ = agent.predict_action(obs, info, evaluate=True)  # type: ignore[arg-type]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_rewards[-1] += reward
            episode_steps += 1
            if terminated or truncated:
                done = True
                pbar.set_postfix(
                    {
                        "episode reward": episode_rewards[-1],
                        "episode step": episode_steps,
                    }
                )
        pbar.update(1)
    env.close()
    return np.mean(episode_rewards)


def _env_kwargs_as_dict(env_kwargs: Any) -> dict[str, Any]:
    if env_kwargs is None:
        return {}
    if OmegaConf.is_config(env_kwargs):
        return OmegaConf.to_container(env_kwargs, resolve=True)  # type: ignore[assignment]
    return dict(env_kwargs)


def _resolve_train_joint_context_schedule(
    env_kwargs: dict[str, Any],
) -> tuple[list[int] | None, dict[str, Any] | None]:
    """Section 6.4-style train joint indices for :class:`RoundRobinJointContextWrapper`.

    ``env_kwargs`` may contain:

    - ``context_protocol``: ``{"mode": "A"|"B"|"C", "include_validation": bool}`` using
      the default 3×3 grid from :func:`~rl_exercises.week_2.context_sets.default_three_by_three_example`.
    - ``train_joint_context_indices``: explicit list of ``joint_context_index`` values.

    If both are set, ``context_protocol`` wins. Keys are popped by the caller.
    """
    protocol_cfg = env_kwargs.get("context_protocol")
    explicit = env_kwargs.get("train_joint_context_indices")
    if protocol_cfg is not None:
        if OmegaConf.is_config(protocol_cfg):
            protocol_cfg = OmegaConf.to_container(protocol_cfg, resolve=True)  # type: ignore[assignment]
        if not isinstance(protocol_cfg, dict):
            raise TypeError(
                "env_kwargs['context_protocol'] must be a mapping with a 'mode' key."
            )
        mode = protocol_cfg["mode"]
        include_val = bool(protocol_cfg.get("include_validation", False))
        proto = default_three_by_three_example(
            mode,  # type: ignore[arg-type]
            include_validation=include_val,
        )
        meta = {"context_protocol": dict(protocol_cfg), **summarize_protocol(proto)}
        return proto.train_joint_indices.tolist(), meta
    if explicit is not None:
        if OmegaConf.is_config(explicit):
            explicit = OmegaConf.to_container(explicit, resolve=True)  # type: ignore[assignment]
        return [int(x) for x in explicit], None  # type: ignore[union-attr]
    return None, None


def make_env(
    env_name: str,
    env_kwargs: Any = None,
    *,
    for_evaluation: bool = False,
) -> gym.Env:
    """Make environment based on name and kwargs.

    Parameters
    ----------
    env_name : str
        Environment name
    env_kwargs : dict or DictConfig, optional
        Optional env config, by default {}. For ``ContextualMarsRover`` you may set
        ``context_protocol`` (``{"mode": "A"|"B"|"C", "include_validation": bool}``) and/or
        ``train_joint_context_indices`` to restrict round-robin training to a train set;
        see :mod:`rl_exercises.week_2.context_sets`.
    for_evaluation : bool, optional
        If True, ``round_robin_context`` is ignored so evaluation uses a fixed
        default context unless you pass explicit ``options`` on ``reset``.

    Returns
    -------
    gym.Env
        Instantiated env
    """
    kwargs = _env_kwargs_as_dict(env_kwargs)
    if env_name == "MarsRover":
        env = MarsRover(**kwargs)
        # env = TimeLimit(env, max_episode_steps=env.horizon)
    elif env_name == "ContextualMarsRover":
        rr = bool(kwargs.pop("round_robin_context", False)) and not for_evaluation
        train_schedule, protocol_meta = _resolve_train_joint_context_schedule(kwargs)
        kwargs.pop("context_protocol", None)
        kwargs.pop("train_joint_context_indices", None)
        env = ContextualMarsRover(**kwargs)
        if rr:
            env = RoundRobinJointContextWrapper(
                env,
                joint_context_indices=train_schedule,
            )
        if protocol_meta is not None and not for_evaluation:
            printr("[ContextualMarsRover] train/test protocol summary:", protocol_meta)
    elif "MiniGrid" in env_name:
        env = gym.make(env_name, **kwargs)
        # env = RGBImgObsWrapper(env)
        env = FlatObsWrapper(env)
    else:
        env = gym.make(env_name, **kwargs)
    env = Monitor(env, filename="train")
    return env


if __name__ == "__main__":
    train()
