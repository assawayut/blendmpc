"""Crocoddyl backend for :class:`blendmpc.core.MPCPolicy`."""

from __future__ import annotations

from typing import Callable

import crocoddyl
import numpy as np

from ..core import MPCPolicy, MPCSolution


class CrocoddylMPC(MPCPolicy):
    """Receding-horizon MPC over a Crocoddyl ``ShootingProblem``.

    Parameters
    ----------
    problem_factory:
        Callable ``x0 -> crocoddyl.ShootingProblem``. Called once; on
        subsequent steps only ``problem.x0`` is updated, so the factory should
        not capture per-step state.
    solver_cls:
        Crocoddyl solver class (default ``SolverBoxFDDP`` so control limits
        declared on the action models are honored).
    max_iter / max_iter_first:
        Solver iterations for warm-started steps and for cold solves.
    u_init:
        Constant control used to initialize cold solves (default zeros).
        A small nonzero value breaks the symmetry of stationary states —
        e.g. a hanging pendulum, where a zero rollout is a stationary point
        of the OCP and DDP declares convergence without moving.
    multistart_iter:
        If > 0, every warm-started solve *also* attempts a cold solve with
        this iteration budget, and the lower-cost candidate wins. This guards
        receding-horizon MPC against a failure mode where warm starts trap
        the solver in a poor local basin found on the first step: once a
        "lazy" solution is shifted forward, a handful of iterations per step
        can never escape it. Set to 0 to disable.
    """

    def __init__(
        self,
        problem_factory: Callable[[np.ndarray], crocoddyl.ShootingProblem],
        solver_cls=crocoddyl.SolverBoxFDDP,
        max_iter: int = 5,
        max_iter_first: int = 100,
        u_init: np.ndarray | None = None,
        multistart_iter: int = 0,
    ) -> None:
        super().__init__()
        self._factory = problem_factory
        self._solver_cls = solver_cls
        self._max_iter = max_iter
        self._max_iter_first = max_iter_first
        self._u_init = u_init
        self._multistart_iter = multistart_iter
        self._problem = None
        self._solver = None

    def _cold_init(self, x0: np.ndarray):
        T = self._problem.T
        if self._u_init is None:
            us = [np.zeros(m.nu) for m in self._problem.runningModels]
        else:
            u0 = np.asarray(self._u_init, dtype=float)
            us = [u0.copy() for _ in range(T)]
        return [x0] * (T + 1), us

    def _attempt(self, xs_init, us_init, maxiter: int) -> MPCSolution:
        solved = self._solver.solve(list(xs_init), list(us_init), maxiter, False)
        return MPCSolution(
            xs=[np.asarray(x) for x in self._solver.xs],
            us=[np.asarray(u) for u in self._solver.us],
            cost=self._solver.cost,
            solved=solved,
            info={"iter": self._solver.iter},
        )

    def solve(
        self,
        x0: np.ndarray,
        us_init: list[np.ndarray] | None = None,
        xs_init: list[np.ndarray] | None = None,
    ) -> MPCSolution:
        x0 = np.asarray(x0, dtype=float)
        if self._problem is None:
            self._problem = self._factory(x0)
            self._solver = self._solver_cls(self._problem)
        self._problem.x0 = x0

        candidates = []
        if us_init is not None:
            candidates.append(self._attempt(xs_init, us_init, self._max_iter))
            if self._multistart_iter > 0:
                cold_xs, cold_us = self._cold_init(x0)
                candidates.append(
                    self._attempt(cold_xs, cold_us, self._multistart_iter)
                )
        else:
            cold_xs, cold_us = self._cold_init(x0)
            candidates.append(self._attempt(cold_xs, cold_us, self._max_iter_first))
        return min(candidates, key=lambda c: c.cost)

    def reset(self) -> None:
        super().reset()
        self._problem = None
        self._solver = None
