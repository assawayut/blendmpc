# Residual SAC under model mismatch (Pendulum-v1)

The M1 benchmark behind the README figure. **Task**: Gymnasium `Pendulum-v1`
swing-up, but the plant's mass is **1.4** while the MPC's internal Crocoddyl
model keeps the nominal **1.0** — a 40% model error. Three arms, all evaluated
on the same 15 fixed seeds (episodes 100–114):

- **MPC alone** — `CrocoddylMPC` (BoxFDDP, horizon 30, 5 warm-started
  iterations/step, symmetry-breaking cold init `u=0.2`), nominal model.
- **Residual SAC** — SB3 SAC trained on `ResidualMPCEnv` with full authority
  (`residual_scale=1.0`): `u = clip(u_mpc + a_rl)`. 3 seeds.
- **SAC from scratch** — same SAC hyperparameters (SB3 defaults) directly on
  the perturbed plant. 3 seeds.

## Reproduce

```bash
python train.py --mode mpc
for s in 0 1 2; do
  python train.py --mode residual --seed $s
  python train.py --mode scratch  --seed $s
done
python plot.py   # writes media/residual_pendulum_{light,dark}.png
```

Each SAC run takes ~5 minutes on CPU (the MPC solves ~1k problems/second on
this model). Results land in `results/*.csv` as
`mode,seed,step,ret_ep0,...,ret_ep14`.

## Results

Mean return over the 15 fixed eval episodes (SAC arms: mean across 3 seeds):

| Arm | 0 steps | 3k steps | 15k steps |
|---|---|---|---|
| MPC alone (nominal model) | −629 | −629 | −629 |
| **Residual SAC over MPC** | **−738** | **−406** | **−273** |
| SAC from scratch | −1452 | −832 | −236 |

The two claims the figure supports: the residual agent **never passes through
the catastrophic exploration phase** (from-scratch SAC sits near −1450 for its
first ~2,500 steps — on hardware, that phase is crashes), and it **beats the
mismatched MPC from ~2k steps on**. From-scratch SAC ends slightly higher
(−236 vs −273): with dense reward and cheap simulation, Pendulum ultimately
favors tabula-rasa RL; the residual's value is where samples are expensive or
failures are unacceptable.

## Honest notes (read before citing)

- **Pendulum is RL-easy.** From-scratch SAC solves it in a few thousand steps,
  so the residual advantage here is the *start*, not the asymptote: it begins
  ~2x better than random and never passes through the catastrophic exploration
  phase. On harder tasks the sample-efficiency gap widens; this task was chosen
  because it reproduces in minutes on CPU.
- **Residual authority matters.** With `residual_scale=0.5` the residual
  plateaus at −385: a bounded correction inherits the flawed base's ceiling.
  Full authority (1.0) removes the cap while keeping the safe start.
- **No oracle line.** MPC given the *true* mass performs *worse* than the
  nominal model here (−909 vs −423 in our sweep): the heavier model presents a
  harder nonconvex swing-up problem, and the local DDP solver falls into the
  "give up" basin more often. Model fidelity is not closed-loop performance —
  worth knowing before you trust an oracle baseline anywhere.
- **Local minima and warm starts.** A receding-horizon MPC that warm-starts
  from its own shifted plan can lock itself into the lazy local minimum found
  on step one (hanging is a stationary point of the OCP). The
  symmetry-breaking cold init (`u_init=0.2`) plus the smooth surrogate angle
  cost (`2(1-cos)` instead of Gym's kinked normalized angle) are what make the
  MPC arm reliable; both are documented in `blendmpc.envs.pendulum`.
