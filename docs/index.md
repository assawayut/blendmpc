# blendmpc

**Composable blends of model predictive control and reinforcement learning for robotics.**

Recent robotics research keeps combining MPC and RL in the same handful of
ways — yet every paper reimplements them from scratch. `blendmpc` packages
these patterns as small, composable modules over
[Gymnasium](https://gymnasium.farama.org/) and standard trajectory-optimization
backends ([Crocoddyl](https://github.com/loco-3d/crocoddyl),
[acados](https://github.com/acados/acados)).

![Residual SAC vs SAC from scratch vs MPC under model mismatch](assets/residual_pendulum_light.png#only-light)
![Residual SAC vs SAC from scratch vs MPC under model mismatch](assets/residual_pendulum_dark.png#only-dark)

*The [model-mismatch benchmark](benchmarks.md): the plant is 40% heavier than
the MPC's model. Residual SAC starts where MPC ends and never passes through
the catastrophic exploration phase that from-scratch RL pays for.*

## The four blends

| Blend | Pattern | Read more |
|---|---|---|
| `ResidualMPCEnv` | RL learns a correction on top of an MPC base controller | [Residual RL](blends/residual.md) |
| `PolicyWarmStartMPC` | a learned policy seeds the trajectory optimizer | [Policy warm start](blends/warm-start.md) |
| `make_learned_terminal` | a learned value function becomes the MPC terminal cost | [Learned terminal cost](blends/terminal-cost.md) |
| `collect_expert_dataset` | MPC labels states for policy distillation / DAgger | [Expert distillation](blends/distillation.md) |

## How is this different from `mpcrl` / `mpc4rl`?

Those libraries follow the Gros & Zanon programme: **MPC *is* the function
approximator**, and RL tunes its parameters. `blendmpc` is complementary: MPC
and RL stay **separate modules that compose** — an RL policy from any library
(SB3, CleanRL, ...) plugs in beside a physics-based MPC, in either direction.

## Design in one paragraph

Everything composes against one small interface,
[`blendmpc.MPCPolicy`](backends.md): backends implement
`solve(x0, us_init, xs_init) -> MPCSolution`, and the base class turns that
into a receding-horizon policy with warm-start shifting. Blends only see the
interface and Gymnasium spaces — never a solver type. A new backend must
require zero changes to blends, and vice versa.
