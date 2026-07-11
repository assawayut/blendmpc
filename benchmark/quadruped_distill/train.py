"""Distill the Go2 trot MPC into a microsecond phase-conditioned policy.

The gait MPC is time-varying, so a memoryless clone cannot represent it: the
student gets the gait phase as an input feature (sin/cos of the cycle
position) next to the 37-dimensional state. Data comes from expert rollouts
with the env's reset randomization; training is plain behavior cloning
(the MSE-DAgger trap from the pendulum applies here too).

Outputs results/distill.csv and results/bc_trot.pt.
"""

from __future__ import annotations

import csv
import os
import time

import mujoco
import numpy as np
import torch
import torch.nn as nn

from blendmpc.envs.go2 import (
    Go2BalanceEnv,
    make_go2_trot_cycle,
    obs_to_state,
    quasi_static_torque,
    stand_state,
)
from blendmpc.solvers.crocoddyl import CrocoddylCyclicMPC

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
EVAL_SEEDS = list(range(100, 105))
N_CYCLE = 32
COLLECT_EPISODES = 60


def make_expert():
    return CrocoddylCyclicMPC(
        lambda x0: make_go2_trot_cycle(),
        u_init=quasi_static_torque(stand_state()),
    )


def phase_feats(t):
    a = 2 * np.pi * (t % N_CYCLE) / N_CYCLE
    return np.array([np.sin(a), np.cos(a)])


def collect(env):
    X, Y = [], []
    expert = make_expert()
    for ep in range(COLLECT_EPISODES):
        obs, _ = env.reset(seed=ep)
        expert.reset()
        done, t = False, 0
        while not done:
            u = expert.action(obs_to_state(obs))
            X.append(np.concatenate([obs, phase_feats(t)]))
            Y.append(u)
            obs, _, term, trunc, _ = env.step(u)
            done = term or trunc
            t += 1
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def evaluate(env, policy, fl_geom):
    rets, ups, air, steps = [], 0, 0, 0
    for s in EVAL_SEEDS:
        obs, _ = env.reset(seed=s)
        policy.reset()
        done, term, R = False, False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            R += float(r)
            done = term or trunc
            steps += 1
            air += not any(
                fl_geom in (env.data.contact[i].geom1, env.data.contact[i].geom2)
                for i in range(env.data.ncon)
            )
        rets.append(R)
        ups += not term
    return np.mean(rets), ups, air / steps


class StudentPolicy:
    def __init__(self, net):
        self.net = net
        self.t = 0

    def reset(self):
        self.t = 0

    @torch.no_grad()
    def __call__(self, obs):
        x = torch.tensor(
            np.concatenate([obs, phase_feats(self.t)]), dtype=torch.float32
        )
        self.t += 1
        return self.net(x).numpy()


class ExpertPolicy:
    def __init__(self):
        self.mpc = make_expert()

    def reset(self):
        self.mpc.reset()

    def __call__(self, obs):
        return self.mpc.action(obs_to_state(obs))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    env = Go2BalanceEnv()
    fl = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "FL")

    print("collecting expert rollouts...", flush=True)
    X, Y = collect(env)
    print(f"  {len(X)} pairs")
    mu, sd = X.mean(0), X.std(0) + 1e-6

    net = nn.Sequential(
        nn.Linear(39, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 12),
    )
    # input normalization folded into the first layer
    with torch.no_grad():
        W = net[0].weight / torch.tensor(sd)
        net[0].bias.copy_(net[0].bias - W @ torch.tensor(mu))
        net[0].weight.copy_(W)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _epoch in range(60):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i : i + 512]
            loss = nn.functional.mse_loss(net(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    print(f"  BC loss {loss.item():.4f}")
    torch.save(net.state_dict(), os.path.join(RESULTS, "bc_trot.pt"))

    student = StudentPolicy(net)
    expert = ExpertPolicy()

    # latency
    student(env.reset(seed=0)[0])
    t0 = time.perf_counter()
    for _ in range(1000):
        student(np.zeros(37))
    t_student = (time.perf_counter() - t0) / 1000
    expert.reset()
    obs, _ = env.reset(seed=0)
    expert(obs)
    t0 = time.perf_counter()
    for _ in range(200):
        expert(obs)
    t_expert = (time.perf_counter() - t0) / 200

    rows = []
    for name, pol in (("expert_mpc", expert), ("bc_student", student)):
        mean_ret, ups, air = evaluate(env, pol, fl)
        rows.append([name, f"{mean_ret:.2f}", f"{ups}/5", f"{air:.2f}"])
        print(
            f"{name:12s} R={mean_ret:7.2f} survival={ups}/5 airborne={air:.0%}",
            flush=True,
        )
    print(
        f"latency: expert {t_expert * 1e3:.2f} ms, student "
        f"{t_student * 1e6:.0f} us ({t_expert / t_student:.0f}x)"
    )
    with open(os.path.join(RESULTS, "distill.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "return", "survival", "airborne", "latency_s"])
        w.writerow(rows[0] + [f"{t_expert:.6f}"])
        w.writerow(rows[1] + [f"{t_student:.6f}"])


if __name__ == "__main__":
    main()
