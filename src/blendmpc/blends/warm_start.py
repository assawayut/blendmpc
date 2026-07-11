"""Policy-generated warm starts for MPC.

An RL policy (or any state-feedback controller) rolls the model forward to
produce ``(xs, us)`` used to initialize the trajectory optimizer. This is the
standard trick for cutting MPC iterations and escaping poor local minima with
a learned global policy (e.g. RL-warm-started MPPI/DDP in recent humanoid
work).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core import MPCPolicy, MPCSolution


class PolicyWarmStartMPC(MPCPolicy):
    """Wrap an :class:`MPCPolicy` so cold solves are seeded by a policy rollout.

    Parameters
    ----------
    mpc:
        The wrapped MPC.
    policy:
        ``x -> u`` feedback used to generate the seed trajectory.
    dynamics:
        ``(x, u) -> x_next`` model used for the seed rollout.
    horizon:
        Number of running nodes in the wrapped MPC's problem.
    always:
        If True, re-seed from the policy on *every* solve instead of only on
        cold starts. Use with care: constantly replacing the shifted plan
        discards the solver's refinements and empirically degrades closed-loop
        performance; prefer the default cold-start-only seeding.
    compare_with_default:
        If True, a cold solve is attempted from *both* the policy rollout and
        the wrapped MPC's own default init, keeping the lower-cost solution.
        A learned seed changes which local basin the solver lands in — it can
        rescue starts the default init misses *and* lose ones it handled;
        best-of-two at the (once-per-episode) cold solve keeps the union of
        their successes without the per-step plan-switching pathology.
    """

    def __init__(
        self,
        mpc: MPCPolicy,
        policy: Callable[[np.ndarray], np.ndarray],
        dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
        horizon: int,
        always: bool = False,
        compare_with_default: bool = False,
    ) -> None:
        super().__init__()
        self.mpc = mpc
        self._policy = policy
        self._dynamics = dynamics
        self._horizon = horizon
        self._always = always
        self._compare = compare_with_default

    def rollout(self, x0: np.ndarray):
        xs, us = [np.asarray(x0, dtype=float)], []
        for _ in range(self._horizon):
            u = np.asarray(self._policy(xs[-1]), dtype=float)
            us.append(u)
            xs.append(np.asarray(self._dynamics(xs[-1], u), dtype=float))
        return xs, us

    def solve(
        self,
        x0: np.ndarray,
        us_init: list[np.ndarray] | None = None,
        xs_init: list[np.ndarray] | None = None,
    ) -> MPCSolution:
        cold = us_init is None
        if cold or self._always:
            xs_init, us_init = self.rollout(x0)
        seeded = self.mpc.solve(x0, us_init=us_init, xs_init=xs_init)
        if cold and self._compare:
            default = self.mpc.solve(x0)  # wrapped MPC's own cold init
            if default.cost < seeded.cost:
                return default
        return seeded

    def reset(self) -> None:
        super().reset()
        self.mpc.reset()
