# Getting started

## Install

```bash
pip install blendmpc            # not yet on PyPI — for now:
pip install -e ".[crocoddyl,test]"
```

The Crocoddyl backend installs from PyPI wheels. The acados backend needs the
[acados C library built from source](https://docs.acados.org/installation)
with `ACADOS_SOURCE_DIR` and `LD_LIBRARY_PATH` set; blendmpc detects it at
import time and all acados features (and tests) are skipped when absent.

## Your first blend: residual RL in ~20 lines

```python
import gymnasium as gym
import numpy as np

from blendmpc.blends import ResidualMPCEnv
from blendmpc.envs.pendulum import make_pendulum_problem, obs_to_state
from blendmpc.solvers.crocoddyl import CrocoddylMPC

# 1. An MPC policy: a Crocoddyl shooting problem + receding-horizon driver
mpc = CrocoddylMPC(lambda x0: make_pendulum_problem(x0, horizon=30),
                   max_iter=5, max_iter_first=300, u_init=np.array([0.2]))

# 2. Wrap any Gymnasium env: actions become corrections on top of the MPC
env = ResidualMPCEnv(gym.make("Pendulum-v1"), mpc, obs_to_state)

# 3. Zero residual == pure MPC. Train any RL agent on `env` to improve on it.
obs, _ = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(np.zeros(1))
    done = terminated or truncated
```

Train SB3 SAC on it exactly as on any other env:

```python
from stable_baselines3 import SAC
SAC("MlpPolicy", env).learn(total_timesteps=15_000)
```

## The core interface

Every backend implements one method; everything else is inherited:

```python
class MPCPolicy(ABC):
    def solve(self, x0, us_init=None, xs_init=None) -> MPCSolution: ...
    def action(self, x0) -> np.ndarray   # receding horizon + warm-start shift
    def reset(self) -> None              # call at episode boundaries
```

`MPCSolution` carries `xs`, `us`, `cost`, `solved`, and solver-specific
`info`. Blends never see anything below this interface.

## Practical notes (learned the hard way)

!!! warning "Warm starts can trap receding-horizon MPC"
    A solver warm-started from its own shifted plan inherits the basin of the
    *first* solve forever. If your task has a stationary "do nothing" state
    (a hanging pendulum, a robot at rest), pass a small symmetry-breaking
    `u_init` so the cold solve escapes it. See the regression test
    `test_symmetry_breaking_cold_init_escapes_hanging_start`.

!!! warning "Kinked costs break gradient-based solvers"
    Costs built on normalized angles (`angle_normalize(θ)²`) have gradient
    discontinuities that make DDP/SQP flip between plans. Use smooth
    surrogates (`2(1-cos θ)`) inside the OCP; evaluate closed-loop performance
    with whatever metric you like.
