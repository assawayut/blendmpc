# blendmpc

**Composable blends of model predictive control and reinforcement learning for robotics.**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/residual_pendulum_dark.png">
  <img alt="Learning curves on Pendulum-v1 with 40% mass mismatch. Residual SAC over MPC starts at return -738, near the MPC baseline of -629, improves past it from 2k steps and reaches -273 — never entering the catastrophic range. SAC from scratch spends its first 2,500 steps near -1450 before converging to -236." src="media/residual_pendulum_light.png">
</picture>

*Model mismatch benchmark ([reproduce it](benchmark/residual_pendulum/)): the plant is 40% heavier than the MPC's model. The residual agent starts where MPC ends and never passes through the catastrophic exploration phase that from-scratch RL pays for.*

Recent robotics research keeps combining MPC and RL in the same handful of ways — yet every paper reimplements them from scratch. `blendmpc` packages these patterns as small, composable modules over [Gymnasium](https://gymnasium.farama.org/) and standard trajectory-optimization backends ([Crocoddyl](https://github.com/loco-3d/crocoddyl) and [acados](https://github.com/acados/acados)):

| Blend | Pattern | Literature |
|---|---|---|
| `ResidualMPCEnv` | RL learns a correction on top of an MPC base controller: `u = u_mpc + a_rl` | residual RL (Johannink et al. '19; loco-manipulation works '24–'26) |
| `PolicyWarmStartMPC` | a learned policy seeds the trajectory optimizer | RL-warm-started MPC/MPPI for humanoids |
| `make_learned_terminal` | a learned value function becomes the MPC terminal cost, shortening the horizon | MPC + value blending (Bhardwaj et al. '21), RLMPC |
| `collect_expert_dataset` | MPC labels states for policy distillation / DAgger | MPC-guided policy learning |

## How is this different from `mpcrl` / `mpc4rl`?

Those excellent libraries follow the Gros & Zanon programme: **MPC *is* the function approximator**, and RL tunes its parameters (Q-learning/DPG on a CasADi/acados OCP). `blendmpc` is complementary: MPC and RL stay **separate modules that compose** — an RL policy from any library (SB3, CleanRL, ...) plugs in beside a physics-based MPC, in either direction.

## Quickstart

```python
import gymnasium as gym
import numpy as np

from blendmpc.blends import ResidualMPCEnv
from blendmpc.envs.pendulum import make_pendulum_problem, obs_to_state
from blendmpc.solvers.crocoddyl import CrocoddylMPC

mpc = CrocoddylMPC(lambda x0: make_pendulum_problem(x0, horizon=40))
env = ResidualMPCEnv(gym.make("Pendulum-v1"), mpc, obs_to_state)

obs, _ = env.reset(seed=0)
done = False
while not done:
    action = np.zeros(env.action_space.shape)  # zero residual = pure MPC
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

With a zero residual this is already a working swing-up controller (the bundled pendulum model matches Gymnasium's dynamics exactly, and control limits are handled by BoxFDDP). Train any RL agent on `env` and it starts from MPC-level performance instead of from scratch.

```bash
pip install -e ".[crocoddyl,test]"
python examples/01_residual_pendulum.py
pytest
```

## Design

Everything composes against one small interface (`blendmpc.MPCPolicy`):

```
solve(x0, us_init, xs_init) -> MPCSolution     # one open-loop OCP
action(x0) -> u                                # receding horizon + warm-start shifting
```

Backends implement `solve()`; blends only see the interface. Adding an acados or MuJoCo-MPC backend touches nothing else.

## Roadmap

- [x] v0.1 — core interface, Crocoddyl backend, 4 blends, pendulum demo + tests
- [x] Residual SAC benchmark (Stable-Baselines3) with learning curves vs pure MPC / pure RL under model mismatch
- [x] acados backend
- [x] Documentation site (mkdocs-material, one page per blend with the underlying papers)
- [ ] MuJoCo quadruped task (torque-limited locomotion)
- [ ] Benchmark table: pure MPC vs pure RL vs each blend, wall-clock and sample efficiency
- [ ] PyPI release

Contributions and issue reports are very welcome — especially additional backends and blend patterns from papers you'd like to see reusable.

## License

MIT
