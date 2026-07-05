"""Residual RL over an MPC base controller.

The classic pattern (Johannink et al. 2019; recent loco-manipulation works):
the environment action becomes a *correction* added to the MPC's action, so
the RL agent only has to learn what the model misses. With a zero policy the
wrapper reproduces pure MPC, which gives RL a strong, safe starting point.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

from ..core import MPCPolicy


class ResidualMPCEnv(gym.Wrapper):
    """Gymnasium wrapper: ``u = clip(u_mpc(x) + scale * a_rl, low, high)``.

    Parameters
    ----------
    env:
        The underlying environment.
    mpc:
        Base controller. Its warm start is reset on every episode.
    obs_to_state:
        Maps environment observations to the MPC state vector.
    residual_scale:
        Fraction of the action range the residual may span (action_space is
        rescaled to ``[-1, 1]`` for the agent).
    """

    def __init__(
        self,
        env: gym.Env,
        mpc: MPCPolicy,
        obs_to_state: Callable[[np.ndarray], np.ndarray],
        residual_scale: float = 0.3,
    ) -> None:
        super().__init__(env)
        self.mpc = mpc
        self._obs_to_state = obs_to_state
        self._low = np.asarray(env.action_space.low, dtype=float)
        self._high = np.asarray(env.action_space.high, dtype=float)
        self._scale = residual_scale * 0.5 * (self._high - self._low)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32
        )
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.mpc.reset()
        self._last_obs = obs
        return obs, info

    def step(self, action):
        u_mpc = self.mpc.action(self._obs_to_state(self._last_obs))
        u = np.clip(u_mpc + self._scale * np.asarray(action), self._low, self._high)
        obs, reward, terminated, truncated, info = self.env.step(u)
        self._last_obs = obs
        info["u_mpc"] = u_mpc
        info["u_applied"] = u
        return obs, reward, terminated, truncated, info
