"""Learned-terminal-cost benchmark: can cost-to-go fix horizon myopia?

The warm-start benchmark exposed that pendulum MPC's "hanging trap" is not a
solver failure but *objective myopia*: within a 1.5 s horizon, staying down is
genuinely cheaper than investing in a swing-up. A terminal value function
V(x) ~ cost-to-go makes hanging expensive at the horizon boundary, so the
swing-up plan wins even with short horizons.

V is fitted by Monte-Carlo regression: from states sampled uniformly over
the state space, roll the expert MPC for K steps and sum the realized Gym
costs. Truncating at a fixed K makes the target stationary (episode-boundary
cost-to-go is time-varying and unlearnable by a stationary V), and uniform
sampling covers the off-corridor states the solver probes during backward
passes. Same units as the OCP stage costs by construction, so scale=1.
"""

from __future__ import annotations

import csv
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from blendmpc.blends.terminal_cost import with_learned_terminal
from blendmpc.envs.pendulum import (
    ActionModelPendulum,
    angle_normalize,
    make_pendulum_problem,
    obs_to_state,
)
from blendmpc.solvers.crocoddyl import CrocoddylMPC

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "distill_pendulum", "results", "expert_dataset.npz")
EVAL_SEEDS = list(range(100, 115))
EP_LEN = 200


def collect_ctg_dataset(n_states=1500, k=60, seed=0):
    """K-step expert cost-to-go from uniformly sampled states."""
    rng = np.random.default_rng(seed)
    states = np.stack(
        [rng.uniform(-np.pi, np.pi, n_states), rng.uniform(-8, 8, n_states)], 1
    )
    env = gym.make("Pendulum-v1").unwrapped
    xs, ys = [], []
    mpc = make_mpc(30)
    for x0 in states:
        env.reset()
        env.state = x0.copy()
        mpc.reset()
        obs = np.array([np.cos(x0[0]), np.sin(x0[0]), x0[1]], dtype=np.float32)
        ctg = 0.0
        for _ in range(k):
            x = obs_to_state(obs)
            u = mpc.action(x)
            # smooth OCP stage cost, so V extends the OCP objective exactly
            ctg += 2.0 * (1.0 - np.cos(x[0])) + 0.1 * x[1] ** 2 + 0.001 * u[0] ** 2
            obs, r, *_ = env.step(u)
        xs.append(x0)
        ys.append(ctg)
    return np.array(xs), np.array(ys)


def value_iteration(n_th=301, n_om=301, n_u=17, sweeps=3000, gamma=0.999):
    """Oracle V* on a dense grid — feasible only because the state is 2-D."""
    from blendmpc.envs.pendulum import DT, G, L, M

    th = np.linspace(-np.pi, np.pi, n_th, endpoint=False)
    om = np.linspace(-8, 8, n_om)
    TH, OM = np.meshgrid(th, om, indexing="ij")
    dth, dom = th[1] - th[0], om[1] - om[0]
    us = np.linspace(-2, 2, n_u)
    gathers = []
    for u in us:
        newom = np.clip(
            OM + (1.5 * G / L * np.sin(TH) + 3 / (M * L**2) * u) * DT, -8, 8
        )
        newth = TH + newom * DT
        fi = (newth + np.pi) / dth
        i0 = np.floor(fi).astype(int) % n_th
        i1 = (i0 + 1) % n_th
        wi = fi - np.floor(fi)
        fj = np.clip((newom + 8) / dom, 0, n_om - 1)
        j0 = np.floor(fj).astype(int)
        j1 = np.minimum(j0 + 1, n_om - 1)
        wj = fj - j0
        cost = 2.0 * (1.0 - np.cos(TH)) + 0.1 * OM**2 + 0.001 * u**2
        gathers.append((i0, i1, wi, j0, j1, wj, cost))
    V = np.zeros((n_th, n_om))
    for it in range(sweeps):  # noqa: B007 (used after the loop)
        Q = np.full((n_u, n_th, n_om), np.inf)
        for a, (i0, i1, wi, j0, j1, wj, cost) in enumerate(gathers):
            Vn = (
                (1 - wi) * (1 - wj) * V[i0, j0]
                + wi * (1 - wj) * V[i1, j0]
                + (1 - wi) * wj * V[i0, j1]
                + wi * wj * V[i1, j1]
            )
            Q[a] = cost + gamma * Vn
        Vnew = Q.min(0)
        delta = np.abs(Vnew - V).max()
        V = Vnew
        if delta < 1e-4:
            break
    print(f"  VI converged in {it + 1} sweeps (delta={delta:.2e})")
    pts = np.stack([TH.ravel(), OM.ravel()], 1)
    return pts, V.ravel()


def fit_value_function():
    cache = os.path.join(HERE, "results", "ctg_dataset.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        xs, ys = d["xs"], d["ys"]
    else:
        print("labeling cost-to-go dataset with expert rollouts...", flush=True)
        xs, ys = collect_ctg_dataset()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, xs=xs, ys=ys)
    print(f"  targets: min={ys.min():.0f} mean={ys.mean():.0f} max={ys.max():.0f}")
    # normalize inputs (tanh nets saturate on |thdot|<=8) and targets
    return _fit_net(xs, ys, "MC")


def fit_oracle_value_function():
    cache = os.path.join(HERE, "results", "vi_dataset.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        xs, ys = d["xs"], d["ys"]
    else:
        print("running value iteration...", flush=True)
        xs, ys = value_iteration()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, xs=xs, ys=ys)
    idx = np.random.default_rng(0).choice(len(xs), 25000, replace=False)
    return _fit_net(xs[idx], ys[idx], "VI")


def _fit_net(xs, ys, tag):
    print(
        f"  [{tag}] targets: min={ys.min():.0f} mean={ys.mean():.0f} max={ys.max():.0f}"
    )
    X = torch.tensor(
        np.stack([xs[:, 0] / np.pi, xs[:, 1] / 8.0], 1), dtype=torch.float32
    )
    Y = torch.tensor(ys[:, None] / 100.0, dtype=torch.float32)

    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(300):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i : i + 256]
            loss = nn.functional.mse_loss(net(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    print(f"  [{tag}] V fit: final loss {loss.item():.4f} on {len(X)} samples")

    net.double()

    def _vt(x):  # torch expression of V, differentiable in x=(th, thdot)
        xin = torch.stack([x[0] / np.pi, x[1] / 8.0])
        return 100.0 * net(xin)[0]

    def _wrap(x):
        return torch.tensor(
            [angle_normalize(float(x[0])), float(x[1])],
            dtype=torch.float64,
            requires_grad=True,
        )

    def value(x):
        with torch.no_grad():
            return float(_vt(_wrap(x)))

    def grad(x):
        xt = _wrap(x)
        return torch.autograd.grad(_vt(xt), xt)[0].numpy()

    def hess(x):
        xt = _wrap(x)
        return torch.autograd.functional.hessian(_vt, xt).numpy()

    print(f"  V(hanging)={value([np.pi, 0]):.0f}  V(upright)={value([0, 0]):.0f}")
    return value, grad, hess


def make_mpc(horizon, value_fns=None):
    if value_fns is None:
        factory = lambda x0: make_pendulum_problem(x0, horizon=horizon)  # noqa: E731
    else:
        v, g, h = value_fns

        def factory(x0):
            return with_learned_terminal(
                x0, [ActionModelPendulum()] * horizon, v, grad_fn=g, hess_fn=h
            )

    return CrocoddylMPC(factory, max_iter=5, max_iter_first=300, u_init=np.array([0.2]))


def episode(mpc, seed):
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
    mpc.reset()
    done, ep_ret = False, 0.0
    while not done:
        obs, r, term, trunc, _ = env.step(mpc.action(obs_to_state(obs)))
        ep_ret += float(r)
        done = term or trunc
    return ep_ret, abs(angle_normalize(np.arctan2(obs[1], obs[0]))) < 0.2


def main():
    value_fns = fit_value_function()
    oracle_fns = fit_oracle_value_function()
    arms = [
        ("H=30 plain", 30, None),
        ("H=8 plain", 8, None),
        ("H=8 + V(MC-1500)", 8, value_fns),
        ("H=8 + V*(VI)", 8, oracle_fns),
        ("H=30 + V*(VI)", 30, oracle_fns),
    ]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "terminal.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "mean", "worst", "upright"])
        for name, H, vf in arms:
            rets, ups = [], 0
            for s in EVAL_SEEDS:
                r, up = episode(make_mpc(H, vf), s)
                rets.append(r)
                ups += up
            w.writerow([name, f"{np.mean(rets):.1f}", f"{min(rets):.1f}", f"{ups}/15"])
            print(
                f"{name:18s} mean={np.mean(rets):7.1f} worst={min(rets):8.1f} "
                f"upright={ups:2d}/15",
                flush=True,
            )


if __name__ == "__main__":
    main()
