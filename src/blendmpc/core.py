"""Solver-agnostic MPC policy interface.

Every blend in :mod:`blendmpc.blends` composes against :class:`MPCPolicy`, so
any trajectory-optimization backend (Crocoddyl, acados, a hand-rolled iLQR)
can participate by implementing ``solve()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class MPCSolution:
    """Result of one open-loop trajectory optimization."""

    xs: List[np.ndarray]
    us: List[np.ndarray]
    cost: float
    solved: bool
    info: dict = field(default_factory=dict)


class MPCPolicy(ABC):
    """A receding-horizon controller built on a trajectory optimizer.

    Subclasses implement :meth:`solve`; the base class turns it into a
    stateful step-by-step policy with warm-start shifting, so it can be used
    anywhere a Gymnasium-style policy is expected via :meth:`action`.
    """

    def __init__(self) -> None:
        self._last: Optional[MPCSolution] = None

    @abstractmethod
    def solve(
        self,
        x0: np.ndarray,
        us_init: Optional[List[np.ndarray]] = None,
        xs_init: Optional[List[np.ndarray]] = None,
    ) -> MPCSolution:
        """Solve the OCP from ``x0``, optionally warm-started."""

    def action(self, x0: np.ndarray) -> np.ndarray:
        """Receding-horizon step: solve from ``x0`` and return the first control.

        The previous solution (shifted by one node, last node repeated) warm
        starts the solver.
        """
        us_init, xs_init = self._shifted_warm_start(x0)
        self._last = self.solve(x0, us_init=us_init, xs_init=xs_init)
        return np.asarray(self._last.us[0]).copy()

    def reset(self) -> None:
        """Drop the internal warm start (call at episode boundaries)."""
        self._last = None

    @property
    def last_solution(self) -> Optional[MPCSolution]:
        return self._last

    def _shifted_warm_start(self, x0: np.ndarray):
        if self._last is None:
            return None, None
        us = list(self._last.us[1:]) + [self._last.us[-1]]
        xs = [np.asarray(x0)] + list(self._last.xs[2:]) + [self._last.xs[-1]]
        return us, xs
