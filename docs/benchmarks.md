# Benchmarks

## Residual SAC under model mismatch (Pendulum-v1)

![Residual SAC vs SAC from scratch vs MPC under model mismatch](assets/residual_pendulum_light.png#only-light)
![Residual SAC vs SAC from scratch vs MPC under model mismatch](assets/residual_pendulum_dark.png#only-dark)

The plant's mass is **1.4**; the MPC's Crocoddyl model keeps the nominal
**1.0** — a 40% model error. All arms are evaluated on the same 15 fixed
episodes; SAC arms use 3 seeds (band = min–max).

| Arm | 0 steps | 3k steps | 15k steps |
|---|---|---|---|
| MPC alone (nominal model) | −629 | −629 | −629 |
| **Residual SAC over MPC** | **−738** | **−406** | **−273** |
| SAC from scratch | −1452 | −832 | −236 |

Two claims the figure supports: the residual agent **never passes through the
catastrophic exploration phase** (from-scratch SAC sits near −1450 for its
first ~2,500 steps — on hardware, that phase is crashes), and it **beats the
mismatched MPC from ~2k steps on**.

Reproduce with `benchmark/residual_pendulum/` (about 5 CPU-minutes per arm);
full configuration and honest caveats — including why there is *no oracle
line* and why residual authority matters — in
[that directory's README](https://github.com/assawayut/blendmpc/tree/main/benchmark/residual_pendulum).

## Expert distillation

BC student (2×64 MLP) cloned from the MPC expert; 15 shared eval seeds:

| Arm | mean | worst | latency / action |
|---|---|---|---|
| MPC expert | −320 | −1498 | 0.6 ms |
| **BC student** (10k pairs) | −423 | −1519 | **24 µs (25×)** |
| BC + 1 DAgger round | −558 | −1576 | 24 µs |

Naive MSE-DAgger *hurts* here: the bang-bang expert gives multimodal labels on
student-visited states, and regression averages them into useless mid-torques.
Details: `benchmark/distill_pendulum/`.

## Policy warm start

The BC student seeds the MPC's cold solves (`PolicyWarmStartMPC`):

| Arm | mean | worst | upright | iters/step |
|---|---|---|---|---|
| plain MPC | −320 | −1498 | 14/15 | 2.04 |
| policy seed (cold) | −373 | −1492 | 13/15 | 1.95 |
| policy seed (every solve) | −772 | −1816 | 9/15 | 2.64 |
| policy seed + best-of-two | −320 | −1498 | 14/15 | 2.00 |

Seeding is basin *selection*: it rescues the start that traps plain MPC and
loses a different one. Best-of-two reveals the deeper truth — from a
near-hanging start the hanging plan genuinely has lower open-loop cost over
1.5 s, so no warm start can fix what is really **objective myopia**. Details:
`benchmark/warmstart_pendulum/`.

## Learned terminal cost

Cost-to-go V(x) as the terminal node — the fix for that myopia:

| Arm | mean | worst | upright |
|---|---|---|---|
| H=30 plain | −320 | −1498 | 14/15 |
| H=8 plain | −1536 | −1614 | 0/15 |
| **H=8 + V(MC-1500)** | −414 | −970 | **14/15** |
| **H=8 + V\*(value iteration)** | −434 | **−790** | **15/15** |
| H=30 + V\*(VI) | −649 | −1390 | 13/15 |

An 8-step MPC with a learned terminal out-robusts the 30-step MPC — and V
helps *only* as a replacement for the missing tail (at H=30 it adds bias, not
information). Implementation traps (cost-unit consistency, float32
finite-difference Hessians, stationary targets, input normalization) are
documented in `benchmark/terminal_pendulum/`.

## Quadruped balance under payload mismatch (Go2)

The residual pattern at robot scale: whole-body Crocoddyl MPC (four feet in
rigid contact, 37-dimensional state, 12 torques, ~3 ms per solve at 50 Hz) on
the MuJoCo menagerie Go2 whose trunk mass is doubled — about the rated
payload, unmodeled.

![Go2 payload benchmark](assets/quadruped_balance_light.png#only-light)
![Go2 payload benchmark](assets/quadruped_balance_dark.png#only-dark)

| Arm | return |
|---|---|
| MPC, true-mass model (oracle) | −2.49 |
| **Residual SAC over nominal MPC** | **−3.50** (best seed −2.69) |
| MPC, nominal model | −5.67 |
| SAC from scratch | −12.79 (starts at −92) |

Two findings that differ from the pendulum, documented in
`benchmark/quadruped_balance/`: residual authority is a *stability hazard* on
a platform with unrecoverable states (0.3 authority collapsed evaluation to
−149 before recovering; 0.1 works), and per-step rewards that are small
against SAC's entropy bonus quietly pay the policy to stay noisy — training
rewards are scaled ×100, evaluation is not. Also: Crocoddyl's
`ShootingProblem.quasiStatic` returned uninitialized memory for this contact
problem; `blendmpc.envs.go2.quasi_static_torque` computes static torques
analytically with Pinocchio instead.
