"""Crocoddyl model of Gymnasium's ``Pendulum-v1`` swing-up task.

Dynamics, cost, and control limits exactly mirror the Gymnasium environment
(semi-implicit Euler, ``g=10, m=1, l=1, dt=0.05``, ``|u| <= 2``,
``cost = angle^2 + 0.1*thetadot^2 + 0.001*u^2``), so the MPC's internal model
matches the plant and closed-loop return is directly comparable to RL scores.
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

    def __init__(self, terminal: bool = False):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(2), 1, 1)
        self._terminal = terminal
        if not terminal:
            self.u_lb = np.array([-U_MAX])
            self.u_ub = np.array([U_MAX])

    def calc(self, data, x, u=None):
        th, thdot = x
        tau = 0.0 if (u is None or self._terminal) else float(u[0])
        # Semi-implicit Euler, as in gymnasium's PendulumEnv.step
        newthdot = thdot + (1.5 * G / L * np.sin(th) + 3.0 / (M * L**2) * tau) * DT
        newth = th + newthdot * DT
        data.xnext = np.array([newth, newthdot])
        ang = angle_normalize(th)
        data.cost = ang**2 + 0.1 * thdot**2 + 0.001 * tau**2

    def calcDiff(self, data, x, u=None):
        th, thdot = x
        tau = 0.0 if (u is None or self._terminal) else float(u[0])
        c = 1.5 * G / L * np.cos(th) * DT
        b = 3.0 / (M * L**2) * DT
        # newthdot = thdot + c'(th) + b*tau ; newth = th + DT*newthdot
        data.Fx = np.array([[1.0 + DT * c, DT], [c, 1.0]])
        data.Fu = np.array([[DT * b], [b]])
        ang = angle_normalize(th)
        data.Lx = np.array([2.0 * ang, 0.2 * thdot])
        data.Lxx = np.diag([2.0, 0.2])
        data.Lu = np.array([0.002 * tau])
        data.Luu = np.array([[0.002]])
        data.Lxu = np.zeros((2, 1))


def make_pendulum_problem(
    x0: np.ndarray, horizon: int = 50
) -> crocoddyl.ShootingProblem:
    running = ActionModelPendulum()
    terminal = ActionModelPendulum(terminal=True)
    return crocoddyl.ShootingProblem(
        np.asarray(x0, dtype=float), [running] * horizon, terminal
    )
