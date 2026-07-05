"""acados/CasADi twin of :mod:`blendmpc.envs.pendulum`.

Same discrete semi-implicit Euler dynamics, smooth surrogate cost, and torque
limits as the Crocoddyl model, expressed as an ``AcadosOcp`` — so the two
backends solve the *same* OCP and blends behave identically over either.
"""

from __future__ import annotations

import numpy as np

from .pendulum import DT, U_MAX, G, L, M


def make_pendulum_ocp(
    x0: np.ndarray,
    horizon: int = 30,
    m: float = M,
    l: float = L,  # noqa: E741
    g: float = G,
):
    """Build the pendulum swing-up ``AcadosOcp`` (imports acados lazily)."""
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp

    th = ca.SX.sym("th")
    thdot = ca.SX.sym("thdot")
    u = ca.SX.sym("u")

    grav = 1.5 * g / l
    gain = 3.0 / (m * l**2)
    newthdot = thdot + (grav * ca.sin(th) + gain * u) * DT
    newth = th + newthdot * DT

    model = AcadosModel()
    model.name = "blendmpc_pendulum"
    model.x = ca.vertcat(th, thdot)
    model.u = u
    model.disc_dyn_expr = ca.vertcat(newth, newthdot)

    angle_cost = 2.0 * (1.0 - ca.cos(th)) + 0.1 * thdot**2
    model.cost_expr_ext_cost = angle_cost + 0.001 * u**2
    model.cost_expr_ext_cost_e = angle_cost

    ocp = AcadosOcp()
    ocp.model = model
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([-U_MAX])
    ocp.constraints.ubu = np.array([U_MAX])
    ocp.constraints.x0 = np.asarray(x0, dtype=float)

    opts = ocp.solver_options
    opts.N_horizon = horizon
    opts.tf = horizon * DT
    opts.integrator_type = "DISCRETE"
    opts.nlp_solver_type = "SQP"
    opts.nlp_solver_max_iter = 50
    opts.hessian_approx = "EXACT"
    opts.regularize_method = "CONVEXIFY"
    opts.globalization = "MERIT_BACKTRACKING"
    opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    return ocp
