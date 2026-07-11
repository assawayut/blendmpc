"""Distillation benchmark: behavior-clone the MPC expert, compare quality/latency.

Collects (obs, u_mpc) pairs with ``collect_expert_dataset``, fits a small MLP
by MSE, and evaluates student vs expert on the shared 15 eval seeds. Also
saves the student net and the expert dataset — the warm-start and
terminal-cost benchmarks reuse them.
"""

from __future__ import annotations

import csv
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from blendmpc.blends import collect_expert_dataset
from blendmpc.envs.pendulum import make_pendulum_problem, obs_to_state
from blendmpc.solvers.crocoddyl import CrocoddylMPC

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
EVAL_SEEDS = list(range(100, 115))
COLLECT_SEEDS = 50  # episodes 0..49 -> 10k pairs


def make_mpc():
    return CrocoddylMPC(
        lambda x0: make_pendulum_problem(x0, horizon=30),
        max_iter=5,
        max_iter_first=300,
        u_init=np.array([0.2]),
    )


def make_student():
    return nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
    )


def evaluate(policy_fn, reset_fn=None):
    rets = []
    for s in EVAL_SEEDS:
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset(seed=s)
        if reset_fn is not None:
            reset_fn()
        done, ep_ret = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy_fn(obs))
            ep_ret += float(r)
            done = term or trunc
        rets.append(ep_ret)
    return rets


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)

    print("collecting expert dataset...", flush=True)
    env = gym.make("Pendulum-v1")
    mpc = make_mpc()
    obs, us, ep_rets = collect_expert_dataset(
        env, mpc, obs_to_state, episodes=COLLECT_SEEDS, seed=0
    )
    np.savez(os.path.join(RESULTS, "expert_dataset.npz"), obs=obs, us=us)
    print(f"  {len(obs)} pairs, expert training-episode mean {ep_rets.mean():.0f}")

    def fit(X, Y):
        net = make_student()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(200):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 256):
                idx = perm[i : i + 256]
                loss = nn.functional.mse_loss(net(X[idx]), Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        print(f"  final BC loss {loss.item():.4f} on {len(X)} pairs")
        return net

    def policy_of(net):
        @torch.no_grad()
        def fn(o):
            u = net(torch.tensor(o, dtype=torch.float32)).numpy()
            return np.clip(u, -2.0, 2.0)

        return fn

    X = torch.tensor(obs, dtype=torch.float32)
    Y = torch.tensor(us, dtype=torch.float32)
    student = fit(X, Y)

    # One DAgger round: the student drives, the MPC labels visited states.
    print("DAgger round: student drives, expert labels...", flush=True)
    obs2, us2, _ = collect_expert_dataset(
        env,
        make_mpc(),
        obs_to_state,
        episodes=COLLECT_SEEDS,
        policy=policy_of(student),
        seed=1000,
    )
    X2 = torch.cat([X, torch.tensor(obs2, dtype=torch.float32)])
    Y2 = torch.cat([Y, torch.tensor(us2, dtype=torch.float32)])
    student_bc, student = student, fit(X2, Y2)
    # Plain BC evaluates better here: the MPC expert is near-bang-bang, and on
    # student-visited states its labels are multimodal (+-2 both valid), so
    # MSE-regression DAgger averages them into useless mid-torques — the
    # irreducible round-2 loss above is the signature. Keep the BC student as
    # the artifact downstream benchmarks (warm start) reuse.
    torch.save(student_bc.state_dict(), os.path.join(RESULTS, "bc_policy.pt"))
    torch.save(student.state_dict(), os.path.join(RESULTS, "dagger_policy.pt"))
    student_fn = policy_of(student)

    expert_mpc = make_mpc()

    def expert_fn(o):
        return expert_mpc.action(obs_to_state(o))

    # latency (after warm-up)
    student_fn(np.zeros(3, dtype=np.float32))
    t0 = time.perf_counter()
    for _ in range(1000):
        student_fn(np.array([1.0, 0.0, 0.5], dtype=np.float32))
    t_student = (time.perf_counter() - t0) / 1000
    expert_mpc.reset()
    expert_fn(np.array([1.0, 0.0, 0.5], dtype=np.float32))
    t0 = time.perf_counter()
    for _ in range(200):
        expert_fn(np.array([1.0, 0.0, 0.5], dtype=np.float32))
    t_expert = (time.perf_counter() - t0) / 200

    rows = []
    for name, fn, reset in (
        ("expert_mpc", expert_fn, expert_mpc.reset),
        ("bc_student", policy_of(student_bc), None),
        ("dagger_student", student_fn, None),
    ):
        rets = evaluate(fn, reset_fn=reset)
        rows.append((name, np.mean(rets), np.std(rets), min(rets)))
        print(
            f"{name:12s} mean={np.mean(rets):7.1f} std={np.std(rets):6.1f} "
            f"worst={min(rets):7.1f}",
            flush=True,
        )
    print(
        f"latency: expert {t_expert * 1e3:.2f} ms/action, "
        f"student {t_student * 1e6:.0f} us/action "
        f"({t_expert / t_student:.0f}x)"
    )

    with open(os.path.join(RESULTS, "distill.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "mean", "std", "worst", "latency_s"])
        w.writerow([*rows[0], f"{t_expert:.6f}"])
        w.writerow([*rows[1], f"{t_student:.6f}"])
        w.writerow([*rows[2], f"{t_student:.6f}"])


if __name__ == "__main__":
    main()
