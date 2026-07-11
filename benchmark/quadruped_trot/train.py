"""Quadruped trot benchmark: residual SAC over gait MPC, overloaded.

Plant: MuJoCo Go2 trotting in place with the trunk mass scaled by
``--payload`` (default 3.0 — roughly the robot's own total mass as an
unmodeled load; an overload stress test, not a rated condition). The cyclic
MPC steps a trot gait with a nominal-mass model. Arms:

- ``mpc``      nominal-model trot MPC on the overloaded plant
- ``oracle``   trot MPC whose model knows the true trunk mass
- ``residual`` SAC over the nominal trot MPC (scale 0.1, training rewards
               x100 — the recipe from the balance benchmark)
- ``scratch``  omitted: it cannot even stand in this budget (see the
               balance benchmark)

Evaluation: mean return over 5 fixed episodes (seeds 100-104), 500 steps at
50 Hz. Results append to results/<mode>.csv.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from blendmpc.blends import ResidualMPCEnv
from blendmpc.envs.go2 import (
    Go2BalanceEnv,
    make_go2_trot_cycle,
    obs_to_state,
    quasi_static_torque,
    stand_state,
)
from blendmpc.solvers.crocoddyl import CrocoddylCyclicMPC

EVAL_SEEDS = list(range(100, 105))
RESIDUAL_SCALE = 0.1
# Balance rewards are ~0.005/step; unscaled they drown in SAC's entropy
# bonus and the policy is effectively paid to stay noisy. Training sees
# scaled rewards; evaluation always uses the raw metric.
REWARD_SCALE = 100.0


def make_mpc(model_mass_scale: float = 1.0) -> CrocoddylCyclicMPC:
    return CrocoddylCyclicMPC(
        lambda x0: make_go2_trot_cycle(trunk_mass_scale=model_mass_scale),
        max_iter=3,
        u_init=quasi_static_torque(stand_state()),
    )


def make_residual_env(payload: float) -> ResidualMPCEnv:
    return ResidualMPCEnv(
        Go2BalanceEnv(trunk_mass_scale=payload),
        make_mpc(),
        obs_to_state,
        residual_scale=RESIDUAL_SCALE,
    )


def evaluate(env, policy_fn) -> list[float]:
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


def append_row(path, mode, seed, step, rets):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([mode, seed, step] + [f"{r:.2f}" for r in rets])


def run_sac(mode: str, args) -> None:
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(2)
    from gymnasium.wrappers import TransformReward

    train_env = TransformReward(
        make_residual_env(args.payload), lambda r: REWARD_SCALE * r
    )
    eval_env = make_residual_env(args.payload)

    model = SAC("MlpPolicy", train_env, seed=args.seed, verbose=0)

    def eval_now(step):
        rets = evaluate(eval_env, lambda o: model.predict(o, deterministic=True)[0])
        append_row(args.out, mode, args.seed, step, rets)
        print(
            f"[{mode} seed={args.seed}] step={step:6d} mean={np.mean(rets):8.2f}",
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
    scale = args.payload if mode == "oracle" else 1.0
    env = Go2BalanceEnv(trunk_mass_scale=args.payload)
    mpc = make_mpc(model_mass_scale=scale)

    def policy(obs):
        return mpc.action(obs_to_state(obs))

    rets = []
    for s in EVAL_SEEDS:
        obs, _ = env.reset(seed=s)
        mpc.reset()
        done, ep_ret = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            ep_ret += float(r)
            done = term or trunc
        rets.append(ep_ret)
    append_row(args.out, mode, 0, 0, rets)
    print(
        f"[{mode}] mean={np.mean(rets):8.2f}  episodes={np.round(rets, 2)}", flush=True
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["mpc", "oracle", "residual"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--eval-freq", type=int, default=6000)
    p.add_argument("--payload", type=float, default=3.0)
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
