# Policy warm-start benchmark (Pendulum-v1)

Seed the Crocoddyl MPC's cold solves with a rollout of the BC student from the
[distillation benchmark](../distill_pendulum/), via `PolicyWarmStartMPC`.
Shared 15 eval seeds; exact model.

| Arm | mean | worst | upright | iters/step |
|---|---|---|---|---|
| plain MPC | −320 | −1498 | 14/15 | 2.04 |
| policy seed (cold solves) | −373 | −1492 | 13/15 | 1.95 |
| policy seed (every solve) | −772 | −1816 | 9/15 | 2.64 |
| policy seed + best-of-two | −320 | −1498 | 14/15 | 2.00 |

```bash
python ../distill_pendulum/train.py   # produces the BC policy first
python train.py
```

## What this shows (read carefully — it's subtle)

- **A learned seed changes which basin the solver lands in — in both
  directions.** It rescues seed 113 (−583, upright), the start that traps
  plain MPC (−1498), *and* loses a different seed plain MPC handled. Seeding
  is basin selection, not a free win.
- **Re-seeding every solve is harmful** (9/15): constantly replacing the
  shifted plan discards the solver's refinements. Same lesson as per-step
  multistart — plan coherence across receding-horizon steps matters.
- **Best-of-two cold starts** (`compare_with_default=True`) exactly reproduces
  plain MPC here — and that's the interesting finding: from a near-hanging
  start, the *hanging plan genuinely has lower open-loop cost* over a 1.5 s
  horizon. The "trap" is not a solver failure but **objective myopia**: the
  swing-up investment doesn't amortize within the horizon. No warm start can
  fix an objective that prefers the trap — which is exactly the problem the
  [learned terminal cost](../terminal_pendulum/) solves.
