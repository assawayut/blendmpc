import gymnasium as gym
import numpy as np
import pytest

crocoddyl = pytest.importorskip("crocoddyl")

from blendmpc.blends import ResidualMPCEnv, collect_expert_dataset
from blendmpc.envs.pendulum import (
    DT,
    G,
    L,
    M,
    make_pendulum_problem,
    obs_to_state,
)
from blendmpc.solvers.crocoddyl import CrocoddylMPC


def make_mpc(horizon=40):
    return CrocoddylMPC(lambda x0: make_pendulum_problem(x0, horizon=horizon))


def test_pendulum_model_matches_gym_dynamics():
    env = gym.make("Pendulum-v1").unwrapped
    env.reset(seed=1)
    env.state = np.array([1.0, 0.5])
    obs, _, _, _, _ = env.step(np.array([1.2]))
    model = make_pendulum_problem(np.zeros(2), 1).runningModels[0]
    data = model.createData()
    model.calc(data, np.array([1.0, 0.5]), np.array([1.2]))
    assert np.allclose(obs_to_state(obs), data.xnext, atol=1e-10)


def test_mpc_swings_up():
    mpc = make_mpc()
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)
    mpc.reset()
    ep_ret, done = 0.0, False
    while not done:
        u = mpc.action(obs_to_state(obs))
        obs, r, term, trunc, _ = env.step(u)
        ep_ret += float(r)
        done = term or trunc
    # random ~ -1200; anything better than -600 means a successful swing-up
    assert ep_ret > -600.0
    assert abs(np.arctan2(obs[1], obs[0])) < 0.2  # upright at the end


def test_zero_residual_equals_pure_mpc():
    env = ResidualMPCEnv(gym.make("Pendulum-v1"), make_mpc(), obs_to_state)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, _, _, _, info = env.step(np.zeros(1))
        assert np.allclose(info["u_applied"], np.clip(info["u_mpc"], -2, 2))


def test_expert_dataset_shapes():
    env = gym.make("Pendulum-v1")
    obs, us, rets = collect_expert_dataset(
        env, make_mpc(horizon=20), obs_to_state, episodes=1, seed=0
    )
    assert obs.shape == (200, 3) and us.shape == (200, 1) and rets.shape == (1,)
