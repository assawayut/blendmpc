"""Crocoddyl model of Gymnasium's ``Pendulum-v1`` swing-up task.

Dynamics and control limits exactly mirror the Gymnasium environment
(semi-implicit Euler, ``g=10, m=1, l=1, dt=0.05``, ``|u| <= 2``; there is a
unit test pinning the match at 1e-10).

The *optimization cost* is a smooth surrogate of Gymnasium's reward: the
angle term is ``2*(1 - cos(theta))`` instead of ``angle_normalize(theta)**2``.
Both are ``~theta^2`` near upright, but Gym's normalized-angle cost has a
gradient discontinuity at the hanging position — exactly where swing-up
episodes start — which makes gradient-based OC solvers flip between plans.
Closed-loop performance is still *measured* with Gym's own reward.

The physical parameters ``m``, ``l``, ``g`` are configurable so the model can
be *deliberately* mismatched against the plant — the setting where blends like
residual RL earn their keep.
"""

from __future__ import annotations

import crocoddyl
import numpy as np

G, M, L, DT = 10.0, 1.0, 1.0, 0.05
U_MAX = 2.0


def angle_normalize(theta: float) -> float:
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def obs_to_state(obs: np.ndarray) -> np.ndarray:
    """Map Pendulum-v1 observation (cos, sin, thetadot) to state (theta, thetadot)."""
    return np.array([np.arctan2(obs[1], obs[0]), obs[2]])


class ActionModelPendulum(crocoddyl.ActionModelAbstract):
    """Discrete pendulum action model with analytic derivatives."""

    def __init__(
        self,
        terminal: bool = False,
        m: float = M,
        l: float = L,  # noqa: E741
        g: float = G,
    ):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(2), 1, 1)
        self._terminal = terminal
        self._grav = 1.5 * g / l  # gravity term:  3g/(2l)
        self._gain = 3.0 / (m * l**2)  # control term: 3/(ml^2)
        if not terminal:
            self.u_lb = np.array([-U_MAX])
            self.u_ub = np.array([U_MAX])

    def calc(self, data, x, u=None):
        th, thdot = x
        tau = 0.0 if (u is None or self._terminal) else float(u[0])
        # Semi-implicit Euler, as in gymnasium's PendulumEnv.step
        newthdot = thdot + (self._grav * np.sin(th) + self._gain * tau) * DT
        newth = th + newthdot * DT
        data.xnext = np.array([newth, newthdot])
        # Smooth surrogate of gym's angle cost (see module docstring)
        data.cost = 2.0 * (1.0 - np.cos(th)) + 0.1 * thdot**2 + 0.001 * tau**2

    def calcDiff(self, data, x, u=None):
        th, thdot = x
        tau = 0.0 if (u is None or self._terminal) else float(u[0])
        c = self._grav * np.cos(th) * DT
        b = self._gain * DT
        # newthdot = thdot + c'(th) + b*tau ; newth = th + DT*newthdot
        data.Fx = np.array([[1.0 + DT * c, DT], [c, 1.0]])
        data.Fu = np.array([[DT * b], [b]])
        data.Lx = np.array([2.0 * np.sin(th), 0.2 * thdot])
        data.Lxx = np.array([[2.0 * np.cos(th), 0.0], [0.0, 0.2]])
        data.Lu = np.array([0.002 * tau])
        data.Luu = np.array([[0.002]])
        data.Lxu = np.zeros((2, 1))


def make_pendulum_problem(
    x0: np.ndarray,
    horizon: int = 50,
    m: float = M,
    l: float = L,  # noqa: E741
    g: float = G,
) -> crocoddyl.ShootingProblem:
    running = ActionModelPendulum(m=m, l=l, g=g)
    terminal = ActionModelPendulum(terminal=True, m=m, l=l, g=g)
    return crocoddyl.ShootingProblem(
        np.asarray(x0, dtype=float), [running] * horizon, terminal
    )
