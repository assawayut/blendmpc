# Learned terminal cost

**`blendmpc.blends.terminal_cost`** — a learned value function $V_\theta$
becomes the terminal node of the OCP:

$$\min_{u_{0:N-1}} \; \sum_{k=0}^{N-1} \ell(x_k, u_k) \; + \; \alpha\, V_\theta(x_N)$$

A perfect terminal value function makes a *one-step* MPC optimal; in practice
a decent learned critic lets you shorten the horizon substantially while
keeping closed-loop quality — trading offline learning for online compute.

```python
from blendmpc.blends.terminal_cost import with_learned_terminal

problem = with_learned_terminal(
    x0, running_models,
    value_fn=lambda x: -critic(x),   # critics estimate reward: negate for cost
    scale=1.0,
)
mpc = CrocoddylMPC(lambda x0: with_learned_terminal(x0, models, value_fn))
```

Derivatives of the black-box $V_\theta$ come from finite differences
(`ActionModelNumDiff`), so any callable works — a PyTorch critic, a fitted
polynomial, a tabulated value function. For production speed, wrap analytic
gradients in a custom terminal model instead.

!!! note "Cost convention"
    Crocoddyl minimizes cost; RL critics estimate *reward*. Pass
    `lambda x: -critic(x)`.

## Papers

- Bhardwaj, Boots & Mukadam, *Blending MPC & Value Function Approximation for Efficient Reinforcement Learning*, ICLR 2021 (arXiv:2012.05909).
- Lowrey et al., *Plan Online, Learn Offline* (POLO), ICLR 2019.
- *Reinforcement Learning-Based Model Predictive Control (RLMPC)*, IEEE 2023.
