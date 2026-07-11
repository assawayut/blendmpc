# Learned terminal cost benchmark (Pendulum-v1)

Give a *short-horizon* MPC a learned cost-to-go V(x) as its terminal cost
(`make_learned_terminal` with torch-autograd derivatives), and test whether it
fixes the horizon myopia exposed by the
[warm-start benchmark](../warmstart_pendulum/). Two V sources:

- **V(MC-1500)** — practical: 1500 states sampled uniformly, each labeled by a
  60-step expert-MPC rollout summing the OCP's stage costs.
- **V\*(VI)** — oracle: value iteration on a dense 301×301 grid (possible only
  because the state is 2-D); quantifies what perfect value estimation buys.

Shared 15 eval seeds; exact model:

| Arm | mean | worst | upright |
|---|---|---|---|
| H=30 plain | −320 | −1498 | 14/15 |
| H=8 plain | −1536 | −1614 | 0/15 |
| **H=8 + V(MC-1500)** | **−414** | **−970** | **14/15** |
| **H=8 + V\*(VI)** | **−434** | **−790** | **15/15** |
| H=30 + V\*(VI) | −649 | −1390 | 13/15 |

```bash
python ../distill_pendulum/train.py   # optional: warms shared caches
python train.py
```

## What this shows

- **An 8-step MPC with a learned terminal beats a 30-step MPC on robustness**:
  15/15 swing-ups with worst episode −790, versus 14/15 and a −1498
  catastrophic episode for the long horizon. V(hanging)≈151 makes "stay down"
  expensive at the horizon boundary, so the swing-up plan wins the cost
  comparison that [defeats warm-starting](../warmstart_pendulum/).
- Even the cheap 1500-rollout V achieves 14/15 — the mechanism does not
  require oracle-quality values.
- At H=30, adding V\* is mildly harmful (13/15, mean −649): the long horizon
  already covers the swing-up, so the terminal adds bias without adding
  information. **V replaces the missing tail of a short horizon; it is not a
  general-purpose bonus.**

## Hard-won implementation notes

1. **The terminal must extend the OCP's own cost.** Fitting V to Gym's
   (kinked, differently-scaled) reward instead of the OCP's smooth stage cost
   overweights the terminal ~2.5× and collapses success to 4–6/15.
2. **Finite differences on a float32 network are unusable** — NumDiff's small
   step amplifies float32 evaluation noise into garbage Hessians (0–1/15
   upright). Pass analytic `grad_fn`/`hess_fn` (torch autograd) to
   `make_learned_terminal`.
3. **Cost-to-go targets must be stationary**: truncate MC rollouts at a fixed
   K; to-end-of-episode sums are time-varying and unlearnable by a stationary
   V (fit RMSE ~163 vs ~45).
4. **Normalize network inputs and targets** — a tanh MLP fed |thdot| ≤ 8 raw
   collapses to a constant.
