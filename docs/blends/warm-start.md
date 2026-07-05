# Policy warm start

**`blendmpc.blends.PolicyWarmStartMPC`** — a learned policy rolls the model
forward to produce the trajectory that initializes the optimizer:

$$ (x^{0}_{0:N},\, u^{0}_{0:N-1}) \;=\; \text{rollout}(x_0, \pi_\theta, f) $$

Trajectory optimizers are local: what they converge to is decided by where
they start. A global-ish learned policy pointing the solver at the right basin
cuts iterations and avoids poor local minima — the pattern behind
RL-warm-started MPPI/DDP in recent humanoid work.

```python
mpc = PolicyWarmStartMPC(
    CrocoddylMPC(problem_factory),
    policy=lambda x: actor(x),          # any state-feedback callable
    dynamics=lambda x, u: f(x, u),      # model used for the seed rollout
    horizon=30,
    always=False,                        # True: re-seed every solve
)
```

With `always=False` the policy seeds only *cold* solves (first step of an
episode); warm steps use the usual shift-and-repeat. With `always=True` the
policy competes with the shifted plan every step — useful when the policy is
stronger than local refinement.

## When to use it

- Long horizons or contact-rich problems where the solver's cold solve finds
  "lazy" local minima (see [getting started](../getting-started.md) on
  stationary-state traps).
- Cutting solve time in high-rate MPC: a good seed can reduce iterations
  per step severalfold.

## Papers

- Mansard et al., *Using a Memory of Motion to Efficiently Warm-Start a Nonlinear Predictive Controller*, ICRA 2018.
- *RGB: RL Guided Whole-Body MPPI for Humanoid Control*, 2026 (arXiv:2606.25123).
