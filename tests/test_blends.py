import gymnasium as gym
import numpy as np
import pytest

crocoddyl = pytest.importorskip("crocoddyl")

from blendmpc.blends import ResidualMPCEnv, collect_expert_dataset
from blendmpc.envs.pendulum import (
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


def test_symmetry_breaking_cold_init_escapes_hanging_start():
    """Regression: a near-hanging start (seed 101, theta0 ~ 160 deg) is a
    stationary point of the OCP; with a zero cold init the solver 'converges'
    to doing nothing and warm-start shifting locks that in for the whole
    episode. A small nonzero cold-init control must escape it."""
    from blendmpc.envs.pendulum import angle_normalize

    env = gym.make("Pendulum-v1")
    mpc = CrocoddylMPC(
        lambda x0: make_pendulum_problem(x0, horizon=30),
        max_iter=5,
        max_iter_first=300,
        u_init=np.array([0.2]),
    )
    obs, _ = env.reset(seed=101)
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(mpc.action(obs_to_state(obs)))
        done = term or trunc
    assert abs(angle_normalize(np.arctan2(obs[1], obs[0]))) < 0.2


def test_analytic_terminal_matches_numdiff():
    """grad_fn/hess_fn path should agree with the NumDiff path on a smooth V."""
    from blendmpc.blends.terminal_cost import with_learned_terminal
    from blendmpc.envs.pendulum import ActionModelPendulum

    def V(x):
        return float(x @ x)

    def gradV(x):
        return 2.0 * x

    def hessV(x):
        return 2.0 * np.eye(2)

    x0 = np.array([1.2, -0.4])
    costs = {}
    for tag, kw in (
        ("numdiff", {}),
        ("analytic", {"grad_fn": gradV, "hess_fn": hessV}),
    ):
        mpc = CrocoddylMPC(
            lambda x0, kw=kw: with_learned_terminal(
                x0, [ActionModelPendulum()] * 8, V, **kw
            ),
            max_iter_first=100,
        )
        costs[tag] = mpc.solve(x0).cost
    assert np.isfinite(costs["analytic"])
    assert abs(costs["analytic"] - costs["numdiff"]) < 1e-3 * max(
        1.0, abs(costs["numdiff"])
    )


def test_warm_start_compare_with_default_never_worse():
    """Best-of-two cold start must not be worse than either init alone."""
    from blendmpc.blends import PolicyWarmStartMPC
    from blendmpc.envs.pendulum import DT, G, L, M

    def bad_policy(x):
        return np.array([2.0])  # deliberately poor constant seed

    def dyn(x, u):
        th, thdot = x
        newthdot = thdot + (1.5 * G / L * np.sin(th) + 3.0 / (M * L**2) * u[0]) * DT
        return np.array([th + newthdot * DT, newthdot])

    x0 = np.array([2.5, 0.0])
    factory = lambda x0: make_pendulum_problem(x0, horizon=20)  # noqa: E731
    plain = CrocoddylMPC(factory).solve(x0).cost
    seeded = (
        PolicyWarmStartMPC(CrocoddylMPC(factory), bad_policy, dyn, 20).solve(x0).cost
    )
    best2 = (
        PolicyWarmStartMPC(
            CrocoddylMPC(factory), bad_policy, dyn, 20, compare_with_default=True
        )
        .solve(x0)
        .cost
    )
    assert best2 <= min(plain, seeded) + 1e-6
