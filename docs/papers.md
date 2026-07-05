# Papers

## Patterns implemented

| Blend | Primary sources |
|---|---|
| [Residual RL](blends/residual.md) | Johannink et al. ICRA 2019; Silver et al. 2018 |
| [Policy warm start](blends/warm-start.md) | Mansard et al. ICRA 2018; RL-guided MPPI (arXiv:2606.25123) |
| [Learned terminal cost](blends/terminal-cost.md) | Bhardwaj et al. ICLR 2021; POLO, ICLR 2019 |
| [Expert distillation](blends/distillation.md) | DAgger, AISTATS 2011; Guided Policy Search, ICML 2013 |

## Surveys and context

- *Synthesis of Model Predictive Control and Reinforcement Learning: Survey
  and Classification*, 2025 (arXiv:2502.02133) — classifies the MPC×RL
  landscape; blendmpc implements its "MPC and RL as separate interacting
  modules" family.
- For the complementary "MPC *as* function approximator" family (Gros &
  Zanon), see [`mpcrl`](https://github.com/FilippoAiraldi/mpc-reinforcement-learning)
  and [`mpc4rl`](https://arxiv.org/abs/2501.15897).

## Using blendmpc in a paper?

Please cite via the repository's `CITATION.cff`, and open a PR to add your
work here — a growing list of papers using the library is the best signal for
new users.
