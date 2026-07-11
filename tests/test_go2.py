"""Go2 whole-body task tests (skipped without mujoco/robot_descriptions).

The closed-loop test doubles as the convention check: a wrong quaternion
order or velocity frame in the MuJoCo->Pinocchio mapping makes the robot
fall within a few control steps.
"""

import numpy as np
import pytest

pytest.importorskip("crocoddyl")
pytest.importorskip("mujoco")
pytest.importorskip("robot_descriptions")

from blendmpc.envs.go2 import (
    STAND_HEIGHT,
    Go2BalanceEnv,
    make_go2_balance_problem,
    obs_to_state,
    quasi_static_torque,
    stand_state,
)
from blendmpc.solvers.crocoddyl import CrocoddylMPC


def test_balance_ocp_recovers_from_perturbed_start():
    import crocoddyl

    x = stand_state()
    x[2] -= 0.04
    x[7:19] += 0.1
    problem = make_go2_balance_problem(x, horizon=25)
    solver = crocoddyl.SolverBoxFDDP(problem)
    us0 = problem.quasiStatic([x] * 25)
    assert solver.solve([x] * 26, us0, 200, False)
    assert abs(solver.xs[-1][2] - STAND_HEIGHT) < 0.02
    tau_max = max(np.abs(u).max() for u in solver.us)
    assert tau_max <= 45.43 + 1e-9


def test_mpc_balances_in_mujoco():
    env = Go2BalanceEnv()
    mpc = CrocoddylMPC(
        lambda x0: make_go2_balance_problem(x0, horizon=25),
        max_iter=3,
        max_iter_first=100,
        u_init=quasi_static_torque(stand_state()),
    )
    obs, _ = env.reset(seed=0)
    mpc.reset()
    done, terminated = False, False
    while not done:
        obs, _, terminated, truncated, _ = env.step(mpc.action(obs_to_state(obs)))
        done = terminated or truncated
    assert not terminated  # survived the full episode
    assert abs(obs[2] - STAND_HEIGHT) < 0.02


def test_trot_in_place_steps_and_survives():
    """The cyclic trot MPC must actually lift its feet (not weight-shift)
    and keep the robot up. Airborne fraction of the FL foot is the guard:
    the swing phase is 37.5% of the cycle; dragging feet gives ~0%."""
    import mujoco

    from blendmpc.envs.go2 import make_go2_trot_cycle
    from blendmpc.solvers.crocoddyl import CrocoddylCyclicMPC

    env = Go2BalanceEnv(max_steps=300)
    fl = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "FL")
    mpc = CrocoddylCyclicMPC(
        lambda x0: make_go2_trot_cycle(),
        u_init=quasi_static_torque(stand_state()),
    )
    obs, _ = env.reset(seed=0)
    mpc.reset()
    done, terminated, airborne, steps = False, False, 0, 0
    while not done:
        obs, _, terminated, truncated, _ = env.step(mpc.action(obs_to_state(obs)))
        done = terminated or truncated
        steps += 1
        airborne += not any(
            fl in (env.data.contact[i].geom1, env.data.contact[i].geom2)
            for i in range(env.data.ncon)
        )
    assert not terminated
    assert airborne / steps > 0.10


def test_forward_trot_tracks_commanded_velocity():
    """Forward trot at 0.3 m/s: no fall, speed within 5 cm/s, stays on line."""
    from blendmpc.envs.go2 import make_go2_trot_cycle
    from blendmpc.solvers.crocoddyl import CrocoddylCyclicMPC

    env = Go2BalanceEnv(command_vx=0.3, max_steps=400)
    mpc = CrocoddylCyclicMPC(
        lambda x0: make_go2_trot_cycle(vx=0.3),
        u_init=quasi_static_torque(stand_state()),
    )
    obs, _ = env.reset(seed=0)
    mpc.reset()
    done, terminated, steps = False, False, 0
    while not done:
        obs, _, terminated, truncated, _ = env.step(mpc.action(obs_to_state(obs)))
        done = terminated or truncated
        steps += 1
    assert not terminated
    mean_vx = float(obs[0]) / (steps * 0.02)
    assert abs(mean_vx - 0.3) < 0.05
    assert abs(float(obs[1])) < 0.1
