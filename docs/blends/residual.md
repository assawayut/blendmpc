# Residual RL over MPC

**`blendmpc.blends.ResidualMPCEnv`** — the environment action becomes a
*correction* added to an MPC base controller:

$$u_t = \operatorname{clip}\big(u^{\text{MPC}}_t(x_t) + \sigma \, a^{\text{RL}}_t,\; u_{\min},\, u_{\max}\big)$$

With a zero policy the wrapper reproduces pure MPC, so a training run starts
from the base controller's performance instead of from random flailing — the
core promise of residual RL for hardware.

```python
env = ResidualMPCEnv(gym.make("Pendulum-v1"), mpc, obs_to_state,
                     residual_scale=1.0)
```

- `residual_scale` sets the residual's authority as a fraction of the action
  range. **A bounded residual inherits the base controller's ceiling**: in the
  [benchmark](../benchmarks.md), scale 0.5 plateaus at −385 while scale 1.0
  reaches −273 — with the same safe start. Start at 1.0 and reduce only if
  early training must stay close to the base controller.
- `info["u_mpc"]` and `info["u_applied"]` expose both action components per
  step for logging and analysis.
- The wrapper resets the MPC's warm start at every episode boundary.

## When to use it

- You have *any* decent base controller and RL exploration from scratch is
  unsafe or sample-expensive.
- Your model is wrong in ways that are hard to write down (mismatched mass,
  unmodeled friction, cable drag) but consistent — exactly what a residual
  policy can learn.

## Papers

- Johannink et al., *Residual Reinforcement Learning for Robot Control*, ICRA 2019.
- Silver et al., *Residual Policy Learning*, 2018.
- RAMBO: *RL-Augmented Model-Based Whole-Body Control for Loco-Manipulation*, 2025 (arXiv:2504.06662).
