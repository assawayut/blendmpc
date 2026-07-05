"""acados backend for :class:`blendmpc.core.MPCPolicy`.

Requires the acados C library plus the ``acados_template`` Python package
(https://docs.acados.org/installation). ``ACADOS_SOURCE_DIR`` and
``LD_LIBRARY_PATH`` must point at your acados build.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from ..core import MPCPolicy, MPCSolution


class AcadosMPC(MPCPolicy):
    """Receding-horizon MPC over an ``AcadosOcp``.

    Parameters
    ----------
    ocp_factory:
        Callable ``x0 -> AcadosOcp``. Called once at the first solve (and
        after :meth:`reset`); generated C code is compiled into ``build_dir``.
        The OCP must pin the initial state through stage-0 bounds
        (``constraints.x0``) — this class updates them every step.
    build_dir:
        Where generated code and the solver library go. Defaults to a
        temporary directory; pass a stable path to reuse compiled code across
        runs.
    """

    def __init__(
        self,
        ocp_factory: Callable[[np.ndarray], AcadosOcp],
        build_dir: str | None = None,
        u_init: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self._factory = ocp_factory
        self._build_dir = build_dir
        self._u_init = u_init
        self._solver: AcadosOcpSolver | None = None
        self._N = 0

    def _make_solver(self, x0: np.ndarray) -> None:
        ocp = self._factory(x0)
        self._N = ocp.solver_options.N_horizon
        build_dir = self._build_dir or tempfile.mkdtemp(prefix="blendmpc_acados_")
        ocp.code_export_directory = os.path.join(build_dir, "c_generated_code")
        self._solver = AcadosOcpSolver(
            ocp, json_file=os.path.join(build_dir, "acados_ocp.json"), verbose=False
        )

    def solve(
        self,
        x0: np.ndarray,
        us_init: list[np.ndarray] | None = None,
        xs_init: list[np.ndarray] | None = None,
    ) -> MPCSolution:
        x0 = np.asarray(x0, dtype=float)
        if self._solver is None:
            self._make_solver(x0)
        s, N = self._solver, self._N

        s.set(0, "lbx", x0)
        s.set(0, "ubx", x0)
        if us_init is None:
            # Cold init: hold x0 with a constant control. As with the
            # Crocoddyl backend, a small nonzero u_init breaks the symmetry
            # of stationary states so SQP does not settle in the lazy basin.
            u0 = (
                np.zeros(s.acados_ocp.dims.nu)
                if self._u_init is None
                else np.asarray(self._u_init, dtype=float)
            )
            us_init = [u0] * N
            xs_init = [x0] * (N + 1)
        for k in range(N):
            s.set(k, "u", np.asarray(us_init[k], dtype=float))
        for k in range(N + 1):
            s.set(k, "x", np.asarray(xs_init[k], dtype=float))

        status = s.solve()
        return MPCSolution(
            xs=[s.get(k, "x") for k in range(N + 1)],
            us=[s.get(k, "u") for k in range(N)],
            cost=s.get_cost(),
            solved=status == 0,
            info={"status": status},
        )

    def reset(self) -> None:
        """Clear the warm start; the compiled solver is kept.

        Unlike the Crocoddyl backend (where rebuilding the problem is cheap),
        recreating an acados solver means C code generation and compilation,
        and the OCP structure does not depend on the initial state — stage-0
        bounds are re-pinned on every solve.
        """
        super().reset()
