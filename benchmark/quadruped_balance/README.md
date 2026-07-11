# Quadruped balance under payload mismatch (Go2)

Whole-body MPC on a real quadruped model: the Crocoddyl OCP holds all four
feet in rigid contact and tracks the standing pose under torque limits
(`blendmpc.envs.go2`, 37-dimensional state, 12 torques, 50 Hz, ~3 ms per
solve). The plant is the MuJoCo menagerie Go2 whose **trunk mass is doubled**
— roughly the robot's rated payload, carried unmodeled. Arms:

- `mpc` — nominal-model MPC on the loaded plant
- `oracle` — MPC whose model knows the true trunk mass (upper bound; unlike
  the pendulum benchmark there is no swing-up nonconvexity here, so the
  informed model *does* behave as an upper bound)
- `residual` — SAC over the nominal MPC, `residual_scale=0.3` (payload
  compensation needs a few N·m per joint; full authority just makes
  exploration noisier)
- `scratch` — SAC alone on raw torques, same budget, for scale

## Reproduce

```bash
python train.py --mode mpc
python train.py --mode oracle
for s in 0 1 2; do
  python train.py --mode residual --seed $s
  python train.py --mode scratch  --seed $s
done
python plot.py
```

Requires the `quadruped` extra (`pip install mujoco robot_descriptions`);
robot descriptions download and cache on first use. Each SAC seed takes
~20 minutes on CPU.

## Results

Mean return over the 5 fixed eval episodes (SAC arms: mean across 3 seeds
at 60k steps):

| Arm | return |
|---|---|
| MPC, true-mass model (oracle) | −2.49 |
| **Residual SAC over nominal MPC** | **−3.50** (best seed −2.69) |
| MPC, nominal model | −5.67 |
| SAC from scratch | −12.79 (starts at −92) |

The residual agent starts at the controller's level (−5.89 at step 0 — no
catastrophic phase), recovers about two thirds of the model-error gap on
average, and its best seed essentially reaches the informed-model controller.
From-scratch SAC never gets close to even the nominal MPC in this budget.

## Notes

- MPC replanning is remarkably robust to payload on its own: up to ~1.6× trunk
  mass the nominal controller barely degrades. The benchmark uses 2.0× because
  below that there is little for a residual to recover.
- **Residual authority is a stability hazard here — the opposite of the
  pendulum lesson.** With `residual_scale=0.3`, SAC's exploration noise
  (±7–14 N·m at 50 Hz) knocks the robot over so often that training data is
  mostly falls; evaluation collapses from −10 to −149 before partially
  recovering, and the "improved" policy is 6× worse than the MPC it started
  from. On the pendulum, a bounded residual capped performance and full
  authority was right. Which regime you are in depends on whether the plant
  has unrecoverable states.
- **Tiny per-step rewards drown in SAC's entropy bonus.** This task pays
  ~0.005 per step; against the entropy term for a 12-D action space, reward
  differences are numerically invisible and the policy is effectively paid to
  stay noisy — eval degrades even at low authority. Scaling training rewards
  by 100 (evaluation unchanged) fixes it. If a residual agent makes a good
  controller worse, check reward scale before anything else.
- `Crocoddyl ShootingProblem.quasiStatic()` returned uninitialized memory for
  this contact problem on some runs (values like `-3.8e30`, or silently zeroed
  hip torques on others). `blendmpc.envs.go2.quasi_static_torque` computes the
  static torques analytically with Pinocchio instead. If your MPC behaves
  nondeterministically across runs, check your cold-start torques first.
- The observation is the Pinocchio-convention state, so the MuJoCo→Pinocchio
  conversion (quaternion order, base-velocity frames) lives in one place; the
  closed-loop test in `tests/test_go2.py` is the regression guard for it.
