# blendmpc

[![CI](https://github.com/assawayut/blendmpc/actions/workflows/ci.yml/badge.svg)](https://github.com/assawayut/blendmpc/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://github.com/assawayut/blendmpc)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-assawayut.github.io%2Fblendmpc-blue)](https://assawayut.github.io/blendmpc/)

**Ready-made building blocks for combining MPC with reinforcement learning.**

You have a model-based controller (MPC). You have RL. Making them work
*together* — an RL policy correcting an MPC, a learned value function
extending its horizon, a neural network warm-starting the solver — is one of
the most active ideas in robot control right now. But every paper rebuilds
the same glue code from scratch.

blendmpc gives you that glue as four small, tested, benchmarked modules. They
plug into the tools you already use: [Gymnasium](https://gymnasium.farama.org/)
environments, RL libraries like [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3),
and the trajectory optimizers [Crocoddyl](https://github.com/loco-3d/crocoddyl)
and [acados](https://github.com/acados/acados).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/residual_pendulum_dark.png">
  <img alt="Learning curves on Pendulum-v1 with 40% mass mismatch. Residual SAC over MPC starts at return -738, near the MPC baseline of -629, improves past it from 2k steps and reaches -273 — never entering the catastrophic range. SAC from scratch spends its first 2,500 steps near -1450 before converging to -236." src="media/residual_pendulum_light.png">
</picture>

*What that buys you, in one picture: the plant is 40% heavier than the MPC's
model. RL trained **on top of** the MPC (blue) starts at controller-level
performance and fixes the model error. RL from scratch (green) spends its
first 2,500 steps crashing.
([reproduce this](benchmark/residual_pendulum/))*

## Install

```bash
git clone https://github.com/assawayut/blendmpc && cd blendmpc
pip install -e ".[crocoddyl,test]"     # PyPI release coming soon
```

The Crocoddyl backend installs straight from PyPI wheels. The acados backend
is optional and needs the [acados C library](https://docs.acados.org/installation);
everything acados-related skips cleanly when it's absent.

## 60 seconds to a working blend

```python
import gymnasium as gym
import numpy as np

from blendmpc.blends import ResidualMPCEnv
from blendmpc.envs.pendulum import make_pendulum_problem, obs_to_state
from blendmpc.solvers.crocoddyl import CrocoddylMPC

# 1. An MPC: a Crocoddyl problem + a receding-horizon driver around it
mpc = CrocoddylMPC(lambda x0: make_pendulum_problem(x0, horizon=30))

# 2. Wrap any Gymnasium env: actions are now *corrections* to the MPC
env = ResidualMPCEnv(gym.make("Pendulum-v1"), mpc, obs_to_state)

# 3. Zero correction == pure MPC. This already swings the pendulum up:
obs, _ = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(np.zeros(1))
    done = terminated or truncated
```

From here, train any RL agent on `env` (it's a normal Gymnasium environment)
and it starts from a working controller instead of from random flailing:

```python
from stable_baselines3 import SAC
SAC("MlpPolicy", env).learn(total_timesteps=15_000)
```

## Which blend do I want?

| Your situation | What you want | Use |
|---|---|---|
| A working MPC, but the model is wrong (mass, friction, cables...) | RL that fixes the model error without unsafe exploration | [`ResidualMPCEnv`](src/blendmpc/blends/residual.py) |
| MPC needs a long horizon to behave well, which is too slow | a learned value function standing in for the missing tail | [`make_learned_terminal`](src/blendmpc/blends/terminal_cost.py) |
| The optimizer is slow to converge or lands in bad local solutions | a policy that points the solver at the right answer | [`PolicyWarmStartMPC`](src/blendmpc/blends/warm_start.py) |
| A good MPC that's too heavy to deploy (rate, compute, no model on-board) | a fast neural imitation of it | [`collect_expert_dataset`](src/blendmpc/blends/distill.py) |

Each has a [docs page](https://assawayut.github.io/blendmpc/) with the math and the papers it implements.

## Results — including the failures

Every blend ships with a benchmark you can rerun in minutes on a laptop. The
negative results are documented too, because they're the expensive part to
discover yourself:

| Blend | What we measured |
|---|---|
| [Residual RL](benchmark/residual_pendulum/) | Starts at MPC level, no catastrophic exploration phase, beats the mismatched MPC from ~2k steps. *Caveat: a bounded correction inherits the base controller's ceiling — give it full authority.* |
| [Learned terminal cost](benchmark/terminal_pendulum/) | **An 8-step MPC with a learned value function is more robust than a 30-step MPC** (15/15 vs 14/15 swing-ups). *Caveat: V must be in the OCP's own cost units, and finite differences on a float32 network are unusable — use the analytic-gradient API.* |
| [Policy warm start](benchmark/warmstart_pendulum/) | Rescues starting states that trap the plain MPC. *Caveat: seeding selects the solver's local solution — it can also lose states, and re-seeding every step is harmful.* |
| [Distillation](benchmark/distill_pendulum/) | A 2×64 MLP recovers the MPC at 25× lower latency. *Caveat: naive MSE-DAgger makes it worse — bang-bang experts give contradictory labels.* |

The deepest lesson from these: what looks like an MPC solver getting "stuck"
is often the *objective* preferring the trap (a short horizon genuinely favors
doing nothing). Warm starts can't fix that; a learned terminal cost can. The
benchmarks walk through that story end to end.

## How it's put together

One small interface. Backends implement it; blends build on it; neither knows
the other's internals.

```
MPCPolicy.solve(x0, us_init, xs_init) -> MPCSolution   # one trajectory optimization
MPCPolicy.action(x0) -> u                              # receding horizon, warm-started
```

- **Backends**: `CrocoddylMPC` (BoxFDDP, control limits in the solver),
  `AcadosMPC` (SQP, compiled C). A new backend must pass the existing blend
  test suite unchanged.
- **Blends** see only `MPCPolicy` and Gymnasium spaces.
- **Env/model pairs** (`blendmpc.envs`) keep the OCP model honest: the bundled
  pendulum model matches Gymnasium's dynamics to 1e-10, with a test pinning it.

## Relation to `mpcrl` and `mpc4rl`

Those libraries make the MPC itself the learnable object (RL tunes the OCP's
parameters — the Gros & Zanon line of work). blendmpc is the complementary
take: the MPC and the RL agent stay **separate parts that compose**, so you
can swap either one independently. If you want parameter-learning MPC, use
[mpcrl](https://github.com/FilippoAiraldi/mpc-reinforcement-learning) — it's good.

## Roadmap

- [x] Core interface, Crocoddyl + acados backends, the four blends
- [x] Reproducible benchmark per blend (see above)
- [x] Docs site with math + papers per blend
- [ ] MuJoCo quadruped task (torque-limited locomotion)
- [ ] PyPI release

Contributions are very welcome — especially new backends, and blend patterns
from papers you wish were reusable. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

If blendmpc is useful in your research, please cite it via
[CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button).

MIT license.
