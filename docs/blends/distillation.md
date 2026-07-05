# Expert distillation

**`blendmpc.blends.collect_expert_dataset`** — run the MPC in closed loop and
record $(o_t, u^{\text{MPC}}_t)$ pairs, ready for behavior cloning:

```python
obs, expert_us, returns = collect_expert_dataset(
    env, mpc, obs_to_state, episodes=50, seed=0
)
# then: fit any policy class to (obs, expert_us)
```

Two collection modes:

- **`policy=None`** (default): the MPC's own action drives the environment —
  pure expert rollouts, the standard imitation setup.
- **`policy=my_policy`**: the *student* drives and the MPC only labels the
  states actually visited — DAgger-style aggregation, which fixes the
  distribution-shift failure of plain behavior cloning.

Why distill at all? The MPC needs a model and per-step optimization; the
distilled policy is a single forward pass, deployable at kHz rates or on
hardware too weak to optimize online — and it can then serve as the
[warm-start policy](warm-start.md) or the base for
[residual RL](residual.md), closing the loop between the blends.

## Papers

- Ross, Gordon & Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger), AISTATS 2011.
- Levine & Koltun, *Guided Policy Search*, ICML 2013.
- *Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation*, 2026 (arXiv:2606.05687).
