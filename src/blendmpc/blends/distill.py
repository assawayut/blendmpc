"""MPC-as-expert dataset collection for policy distillation.

Run the MPC in closed loop and record ``(obs, u_mpc)`` pairs, ready for
behavior cloning (DAgger-style aggregation works by passing a ``policy`` that
drives the environment while the MPC labels the visited states).
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

from ..core import MPCPolicy


def collect_expert_dataset(
    env: gym.Env,
    mpc: MPCPolicy,
    obs_to_state: Callable[[np.ndarray], np.ndarray],
    episodes: int = 10,
    policy: Callable[[np.ndarray], np.ndarray] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(observations, expert_actions, episode_returns)``.

    If ``policy`` is None the MPC's own action drives the environment
    (pure expert rollouts); otherwise ``policy(obs)`` drives and the MPC only
    labels (DAgger-style).
    """
    all_obs, all_us, returns = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=None if seed is None else seed + ep)
        mpc.reset()
        done, ep_ret = False, 0.0
        while not done:
            u_mpc = mpc.action(obs_to_state(obs))
            all_obs.append(np.asarray(obs, dtype=float))
            all_us.append(np.asarray(u_mpc, dtype=float))
            u = u_mpc if policy is None else np.asarray(policy(obs), dtype=float)
            obs, reward, terminated, truncated, _ = env.step(u)
            ep_ret += float(reward)
            done = terminated or truncated
        returns.append(ep_ret)
    return np.array(all_obs), np.array(all_us), np.array(returns)
