"""Learned terminal cost for Crocoddyl shooting problems.

Injects a learned value function ``V(x)`` (any callable, e.g. a critic from an
RL run) as the terminal node of a ``ShootingProblem``, shortening the horizon
an MPC needs for good closed-loop behavior (Blending MPC & Value Function
Approximation, Bhardwaj et al. 2021).

Derivatives are obtained by finite differences via ``ActionModelNumDiff``, so
any black-box V works; pass analytic gradients through a custom model for
speed once the prototype is validated.
"""

from __future__ import annotations

from typing import Callable

import crocoddyl
import numpy as np


class _ValueTerminalModel(crocoddyl.ActionModelAbstract):
    """Terminal action model whose cost is ``scale * V(x)``."""

    def __init__(self, state, value_fn: Callable[[np.ndarray], float], scale: float):
        crocoddyl.ActionModelAbstract.__init__(self, state, 0, 1)
        self._v = value_fn
        self._scale = scale

    def calc(self, data, x, u=None):
        data.xnext = np.asarray(x, dtype=float).copy()
        data.cost = self._scale * float(self._v(np.asarray(x, dtype=float)))

    def calcDiff(self, data, x, u=None):
        raise NotImplementedError  # wrapped in ActionModelNumDiff


def make_learned_terminal(
    state, value_fn: Callable[[np.ndarray], float], scale: float = 1.0
) -> crocoddyl.ActionModelNumDiff:
    """Build a terminal model for a learned cost-to-go ``V(x)``.

    Note: pass a *cost* convention value function (lower is better). For an RL
    critic trained on rewards, wrap it as ``lambda x: -critic(x)``.
    """
    return crocoddyl.ActionModelNumDiff(_ValueTerminalModel(state, value_fn, scale))


def with_learned_terminal(
    x0: np.ndarray,
    running_models,
    value_fn: Callable[[np.ndarray], float],
    scale: float = 1.0,
) -> crocoddyl.ShootingProblem:
    """Assemble a ShootingProblem whose terminal cost is a learned V(x)."""
    terminal = make_learned_terminal(running_models[0].state, value_fn, scale)
    return crocoddyl.ShootingProblem(
        np.asarray(x0, dtype=float), list(running_models), terminal
    )
