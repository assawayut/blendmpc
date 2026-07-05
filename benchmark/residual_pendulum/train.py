"""M1 benchmark: residual SAC vs from-scratch SAC vs MPC under model mismatch.

Plant: Pendulum-v1 with mass increased to ``--mass`` (default 1.4).
The MPC's internal model keeps the nominal mass 1.0, so pure MPC degrades;
residual SAC starts from the MPC baseline and learns to compensate the
mismatch; from-scratch SAC starts from nothing. The residual has full
authority (scale 1.0): a bounded residual inherits the flawed base's ceiling
and plateaus below from-scratch SAC (measured: -385 vs -236 at scale 0.5).
``oracle`` runs MPC with the true plant mass (not plotted: with a local DDP
solver the harder true model is optimized *worse* than the optimistic nominal
one, so it is not an upper bound; see PLAN.md).

Each arm is evaluated on the same fixed seeds so curves are comparable.
Results append to results/<mode>.csv as: mode,seed,step,ret0,ret1,...
"""

from __future__ import annotations

import argparse
import csv
import os

import gymnasium as gym
import numpy as np

from blendmpc.blends import ResidualMPCEnv
from blendmpc.envs.pendulum import make_pendulum_problem, obs_to_state
from blendmpc.solvers.crocoddyl import CrocoddylMPC

EVAL_SEEDS = list(range(100, 115))
HORIZON = 30
RESIDUAL_SCALE = 1.0


def make_plant(mass: float) -> gym.Env:
    env = gym.make("Pendulum-v1")
    env.unwrapped.m = mass
    return env


def make_residual_env(mass: float, model_mass: float) -> ResidualMPCEnv:
    mpc = CrocoddylMPC(
        lambda x0: make_pendulum_problem(x0, horizon=HORIZON, m=model_mass),
        max_iter=5,
        max_iter_first=300,
        u_init=np.array([0.2]),
    )
    return ResidualMPCEnv(
        make_plant(mass), mpc, obs_to_state, residual_scale=RESIDUAL_SCALE
    )


def evaluate(env: gym.Env, policy_fn) -> list[float]:
    rets = []
    for s in EVAL_SEEDS:
        obs, _ = env.reset(seed=s)
        done, ep_ret = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy_fn(obs))
            ep_ret += float(r)
            done = term or trunc
        rets.append(ep_ret)
    return rets


def append_row(path: str, mode: str, seed: int, step: int, rets: list[float]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([mode, seed, step] + [f"{r:.2f}" for r in rets])


def run_sac(mode: str, args) -> None:
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(2)
    if mode == "residual":
        train_env = make_residual_env(args.mass, model_mass=1.0)
        eval_env = make_residual_env(args.mass, model_mass=1.0)
    else:  # scratch
        train_env = make_plant(args.mass)
        eval_env = make_plant(args.mass)

    model = SAC("MlpPolicy", train_env, seed=args.seed, verbose=0)

    def eval_now(step: int) -> None:
        rets = evaluate(eval_env, lambda o: model.predict(o, deterministic=True)[0])
        append_row(args.out, mode, args.seed, step, rets)
        print(
            f"[{mode} seed={args.seed}] step={step:6d} mean={np.mean(rets):8.1f}",
            flush=True,
        )

    class Cb(BaseCallback):
        def _on_training_start(self):
            eval_now(0)

        def _on_step(self):
            if self.num_timesteps % args.eval_freq == 0:
                eval_now(self.num_timesteps)
            return True

    model.learn(total_timesteps=args.steps, callback=Cb())


def run_mpc(mode: str, args) -> None:
    model_mass = args.mass if mode == "oracle" else 1.0
    env = make_residual_env(args.mass, model_mass=model_mass)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    rets = evaluate(env, lambda o: zero)
    append_row(args.out, mode, 0, 0, rets)
    print(f"[{mode}] mean={np.mean(rets):8.1f}  episodes={rets}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode", required=True, choices=["residual", "scratch", "mpc", "oracle"]
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--eval-freq", type=int, default=1500)
    p.add_argument("--mass", type=float, default=1.4)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.out = os.path.join(here, "results", f"{args.mode}.csv")

    if args.mode in ("residual", "scratch"):
        run_sac(args.mode, args)
    else:
        run_mpc(args.mode, args)


if __name__ == "__main__":
    main()
