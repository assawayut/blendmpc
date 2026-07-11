# Quadruped trot under overload (Go2)

The locomotion milestone: a trot-in-place gait executed by
`CrocoddylCyclicMPC`, which advances a periodic contact schedule through the
receding horizon with `ShootingProblem.circularAppend` (one gait cycle =
0.64 s: two 0.08 s double-support phases and two 0.24 s diagonal-pair swings
with a 6 cm sine apex; `blendmpc.envs.go2.make_go2_trot_cycle`). The MuJoCo
plant carries **three times its trunk mass** — about the robot's own total
mass as unmodeled load. This is an overload stress test, not a rated
condition: the nominal-model gait still never falls, but tracking degrades
about fivefold.

Arms: nominal-model trot MPC, an informed-model (oracle) trot MPC, and
residual SAC over the nominal MPC using the recipe from the
[balance benchmark](../quadruped_balance/) (authority 0.1, training rewards
×100). SAC from scratch is omitted — it cannot even stand in this budget.

## Reproduce

```bash
python train.py --mode mpc
python train.py --mode oracle
for s in 0 1 2; do python train.py --mode residual --seed $s; done
python plot.py
```

## Results

Mean return over the 5 fixed eval episodes (residual: mean across 3 seeds at
60k steps):

| Arm | return |
|---|---|
| **Residual SAC over nominal trot MPC** | **−5.81** (band ±0.1) |
| Trot MPC, true-mass model (oracle) | −11.79 |
| Trot MPC, nominal model | −24.12 |

The residual starts at the nominal gait's level and **passes the true-model
controller at about 10k steps**, converging to twice the oracle's score. A
better mass model cannot get there: the remaining error is contact timing and
soft-contact effects that the rigid-contact OCP cannot represent — but a
learned correction can absorb. For scale, −5.8 under a 3× overload matches
the nominal gait's score under a 2× load.

## Forward walking

The same machinery walks: `make_go2_trot_cycle(vx=0.3)` returns an
`update_fn` that advances the swing-foothold schedule (stride = vx × cycle)
and the base reference through the receding horizon, and
`CrocoddylCyclicMPC` stamps each node as it enters the window. Measured
closed loop (no payload): 0.2 m/s and 0.3 m/s track to within a few mm/s
with ~2 mm lateral drift over 10 s; at 0.4 m/s the gait falls — the stride
outgrows the 0.24 s swing at these parameters. Under payload while walking
at 0.3 m/s: ×2 degrades tracking cost by ~5% (and the informed model is no
better than the nominal one there); ×2.5 is marginal (2/3 episodes survive).
There is a regression test for the 0.3 m/s walk.

## Horizon ablation (and why there is no terminal-cost benchmark here)

On the pendulum, an 8-step MPC needed a learned terminal cost to avoid
horizon myopia. Not here — shorter windows just work, because the
long-horizon structure lives in the externally imposed contact schedule
rather than in the OCP window:

| horizon | return | FL airborne | solve |
|---|---|---|---|
| 32 (full cycle) | −6.2 | 22% | 3.1 ms |
| 16 | −5.4 | 22% | 1.6 ms |
| 8 | −4.5 | 24% | 0.9 ms |
| 4 | −2.4 | 17% | 0.5 ms |

The same holds for the balance task (H=2 outperforms H=25 at a tenth of the
solve time). A learned terminal cost pays off when the task's payoff lies
beyond the horizon and the solver must *discover* it — swing-up yes,
schedule-driven gaits no. If you need rate, shrink the horizon first.

## Notes

- **The gait actually steps.** The regression test asserts the front-left
  foot is airborne for >10% of control steps (swing occupies 37.5% of the
  cycle); at the default swing weight the measured fraction is ~23%. With the
  original softer swing cost the same "trot" silently degenerated into
  weight-shifting with feet never leaving the ground — check contact states,
  not plans.
- **Model fidelity is not monotonically useful.** At 2× payload the informed
  model trots *worse* than the nominal one (−8.5 vs −5.0: it plans more
  conservatively and pays for it); at 3× the ranking flips and the informed
  model is clearly better (−11.8 vs −24.1). Same solver, same gait — the
  benefit of a better model depends on how hard the task pushes the plant.
- Payload steadies the trot slightly at moderate loads (added trunk inertia
  damps step-induced wobble): nominal at 2× scores −5.0 versus −6.3 with no
  payload at all.
- Rigid-contact model vs MuJoCo's soft contacts means touchdown timing never
  matches exactly; 50 Hz replanning absorbs it at these gait parameters. No
  impulse models are used at contact switches.
