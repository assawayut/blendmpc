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


class _AnalyticValueTerminalModel(crocoddyl.ActionModelAbstract):
    """Terminal model with user-supplied gradient and Hessian of V."""

    def __init__(self, state, value_fn, grad_fn, hess_fn, scale):
        crocoddyl.ActionModelAbstract.__init__(self, state, 0, 1)
        self._v, self._g, self._h = value_fn, grad_fn, hess_fn
        self._scale = scale

    def calc(self, data, x, u=None):
        data.xnext = np.asarray(x, dtype=float).copy()
        data.cost = self._scale * float(self._v(np.asarray(x, dtype=float)))

    def calcDiff(self, data, x, u=None):
        x = np.asarray(x, dtype=float)
        data.Fx = np.eye(self.state.ndx)
        data.Lx = self._scale * np.asarray(self._g(x), dtype=float)
        data.Lxx = self._scale * np.asarray(self._h(x), dtype=float)


def make_learned_terminal(
    state,
    value_fn: Callable[[np.ndarray], float],
    scale: float = 1.0,
    grad_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    hess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
):
    """Build a terminal model for a learned cost-to-go ``V(x)``.

    Pass a *cost* convention value function (lower is better). For an RL
    critic trained on rewards, wrap it as ``lambda x: -critic(x)``.

    When ``grad_fn``/``hess_fn`` are given (e.g. from torch autograd), an
    analytic terminal model is used. Otherwise derivatives come from finite
    differences — beware that float32 function noise wrecks finite-difference
    Hessians at NumDiff's default step; prefer analytic derivatives (or a
    float64 value function) for neural-network Vs.
    """
    if grad_fn is not None and hess_fn is not None:
        return _AnalyticValueTerminalModel(state, value_fn, grad_fn, hess_fn, scale)
    return crocoddyl.ActionModelNumDiff(_ValueTerminalModel(state, value_fn, scale))


def with_learned_terminal(
    x0: np.ndarray,
    running_models,
    value_fn: Callable[[np.ndarray], float],
    scale: float = 1.0,
    grad_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    hess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> crocoddyl.ShootingProblem:
    """Assemble a ShootingProblem whose terminal cost is a learned V(x)."""
    terminal = make_learned_terminal(
        running_models[0].state, value_fn, scale, grad_fn=grad_fn, hess_fn=hess_fn
    )
    return crocoddyl.ShootingProblem(
        np.asarray(x0, dtype=float), list(running_models), terminal
    )
