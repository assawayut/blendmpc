"""Crocoddyl backend for :class:`blendmpc.core.MPCPolicy`."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

import crocoddyl

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
        Solver iterations for warm-started steps and for the cold first solve.
    """

    def __init__(
        self,
        problem_factory: Callable[[np.ndarray], "crocoddyl.ShootingProblem"],
        solver_cls=crocoddyl.SolverBoxFDDP,
        max_iter: int = 5,
        max_iter_first: int = 100,
    ) -> None:
        super().__init__()
        self._factory = problem_factory
        self._solver_cls = solver_cls
        self._max_iter = max_iter
        self._max_iter_first = max_iter_first
        self._problem = None
        self._solver = None

    def solve(
        self,
        x0: np.ndarray,
        us_init: Optional[List[np.ndarray]] = None,
        xs_init: Optional[List[np.ndarray]] = None,
    ) -> MPCSolution:
        x0 = np.asarray(x0, dtype=float)
        if self._problem is None:
            self._problem = self._factory(x0)
            self._solver = self._solver_cls(self._problem)
        self._problem.x0 = x0

        T = self._problem.T
        cold = us_init is None
        if cold:
            us_init = [np.zeros(m.nu) for m in self._problem.runningModels]
            xs_init = [x0] * (T + 1)
        maxiter = self._max_iter_first if cold else self._max_iter
        solved = self._solver.solve(list(xs_init), list(us_init), maxiter, False)
        return MPCSolution(
            xs=[np.asarray(x) for x in self._solver.xs],
            us=[np.asarray(u) for u in self._solver.us],
            cost=self._solver.cost,
            solved=solved,
            info={"iter": self._solver.iter},
        )

    def reset(self) -> None:
        super().reset()
        self._problem = None
        self._solver = None
