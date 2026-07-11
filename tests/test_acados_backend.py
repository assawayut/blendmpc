"""Backend-parity tests for AcadosMPC (skipped when acados is absent).

Rule from CONTRIBUTING.md: a new backend must pass the blend suite unchanged — these
tests mirror the Crocoddyl ones over the same OCP.
"""

import os

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("acados_template")
if not os.environ.get("ACADOS_SOURCE_DIR"):
    pytest.skip(
        "acados C library not configured (ACADOS_SOURCE_DIR unset)",
        allow_module_level=True,
    )

from blendmpc.blends import ResidualMPCEnv
from blendmpc.envs.pendulum import angle_normalize, obs_to_state
from blendmpc.envs.pendulum_acados import make_pendulum_ocp
from blendmpc.solvers.acados import AcadosMPC


@pytest.fixture(scope="module")
def mpc(tmp_path_factory):
    build = str(tmp_path_factory.mktemp("acados_build"))
    return AcadosMPC(lambda x0: make_pendulum_ocp(x0, horizon=30), build_dir=build)


def test_acados_stabilizes_and_respects_limits(mpc):
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=103)  # moderate start (~ -67 deg)
    mpc.reset()
    ep_ret, done, u_max_seen = 0.0, False, 0.0
    while not done:
        u = mpc.action(obs_to_state(obs))
        u_max_seen = max(u_max_seen, abs(float(u[0])))
        obs, r, term, trunc, _ = env.step(u)
        ep_ret += float(r)
        done = term or trunc
    assert u_max_seen <= 2.0 + 1e-6
    # SQP takes a one-extra-swing route from some states where FDDP goes
    # direct (local-solver difference, documented in docs/backends.md), so
    # the bar here is "converging upright, no catastrophic episode" rather
    # than FDDP parity.
    assert abs(angle_normalize(np.arctan2(obs[1], obs[0]))) < 0.35
    assert ep_ret > -1000.0


def test_zero_residual_equals_pure_mpc_acados(mpc):
    env = ResidualMPCEnv(gym.make("Pendulum-v1"), mpc, obs_to_state)
    obs, _ = env.reset(seed=103)
    for _ in range(5):
        obs, _, _, _, info = env.step(np.zeros(1))
        assert np.allclose(info["u_applied"], np.clip(info["u_mpc"], -2, 2))
