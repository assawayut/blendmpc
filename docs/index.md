# blendmpc

blendmpc implements four common ways of combining trajectory-optimization MPC
with reinforcement learning: residual RL on top of an MPC controller, learned
value functions as MPC terminal costs, warm-starting the solver with a
learned policy, and collecting MPC rollouts to train imitation policies.
These patterns keep showing up in robotics papers as one-off implementations;
here they are ordinary library components that work with
[Gymnasium](https://gymnasium.farama.org/) environments and the
[Crocoddyl](https://github.com/loco-3d/crocoddyl) and
[acados](https://github.com/acados/acados) solvers.

![Go2 walking under the cyclic gait MPC](assets/go2_walk.gif)

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
