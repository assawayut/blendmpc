"""Warm-start benchmark: does a learned policy seed fix MPC's local minima?

Plain receding-horizon MPC on the pendulum gets trapped in the hanging local
minimum from near-hanging starts on the negative side (seed 113) even with a
symmetry-breaking cold init. ``PolicyWarmStartMPC`` seeds the cold solve with
a rollout of the behavior-cloned student from the distillation benchmark —
testing whether a cheap learned policy can point the optimizer at the right
basin. Also reports solver iterations, the classic warm-start win.
"""

from __future__ import annotations

import csv
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from blendmpc.blends import PolicyWarmStartMPC
from blendmpc.envs.pendulum import (
    DT,
    G,
    L,
    M,
    angle_normalize,
    make_pendulum_problem,
    obs_to_state,
)
from blendmpc.solvers.crocoddyl import CrocoddylMPC

HERE = os.path.dirname(os.path.abspath(__file__))
BC_PATH = os.path.join(HERE, "..", "distill_pendulum", "results", "bc_policy.pt")
EVAL_SEEDS = list(range(100, 115))
HORIZON = 30


def load_bc_policy():
    net = nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
    )
    net.load_state_dict(torch.load(BC_PATH))
    net.eval()

    @torch.no_grad()
    def policy(x):  # state (theta, thetadot) -> control
        o = torch.tensor([np.cos(x[0]), np.sin(x[0]), x[1]], dtype=torch.float32)
        return np.clip(net(o).numpy(), -2.0, 2.0)

    return policy


def dynamics(x, u):
    th, thdot = x
    newthdot = thdot + (1.5 * G / L * np.sin(th) + 3.0 / (M * L**2) * u[0]) * DT
    return np.array([th + newthdot * DT, newthdot])


def make_plain():
    return CrocoddylMPC(
        lambda x0: make_pendulum_problem(x0, horizon=HORIZON),
        max_iter=5,
        max_iter_first=300,
        u_init=np.array([0.2]),
    )


def episode(mpc, seed):
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
    mpc.reset()
    done, ep_ret, iters = False, 0.0, 0
    while not done:
        u = mpc.action(obs_to_state(obs))
        sol = mpc.last_solution
        iters += sol.info.get("iter", 0)
        obs, r, term, trunc, _ = env.step(u)
        ep_ret += float(r)
        done = term or trunc
    upright = abs(angle_normalize(np.arctan2(obs[1], obs[0]))) < 0.2
    return ep_ret, upright, iters


def main():
    policy = load_bc_policy()
    arms = {
        "plain_mpc": lambda: make_plain(),
        "policy_seed_cold": lambda: PolicyWarmStartMPC(
            make_plain(), policy, dynamics, HORIZON, always=False
        ),
        "policy_seed_always": lambda: PolicyWarmStartMPC(
            make_plain(), policy, dynamics, HORIZON, always=True
        ),
        "policy_seed_best2": lambda: PolicyWarmStartMPC(
            make_plain(), policy, dynamics, HORIZON, compare_with_default=True
        ),
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "warmstart.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "mean", "worst", "upright", "iters_per_step"])
        for name, factory in arms.items():
            rets, ups, iters = [], 0, 0
            for s in EVAL_SEEDS:
                r, up, it = episode(factory(), s)
                rets.append(r)
                ups += up
                iters += it
            ips = iters / (len(EVAL_SEEDS) * 200)
            w.writerow(
                [
                    name,
                    f"{np.mean(rets):.1f}",
                    f"{min(rets):.1f}",
                    f"{ups}/15",
                    f"{ips:.2f}",
                ]
            )
            print(
                f"{name:20s} mean={np.mean(rets):7.1f} worst={min(rets):8.1f} "
                f"upright={ups:2d}/15 iters/step={ips:5.2f}",
                flush=True,
            )
    # the previously-trapped seed, called out explicitly
    for name, factory in arms.items():
        r, up, _ = episode(factory(), 113)
        print(f"seed 113 {name:20s} R={r:8.1f} upright={up}")


if __name__ == "__main__":
    main()
