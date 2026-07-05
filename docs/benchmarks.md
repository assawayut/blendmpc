# Benchmarks

## Residual SAC under model mismatch (Pendulum-v1)

![Residual SAC vs SAC from scratch vs MPC under model mismatch](assets/residual_pendulum_light.png#only-light)
![Residual SAC vs SAC from scratch vs MPC under model mismatch](assets/residual_pendulum_dark.png#only-dark)

The plant's mass is **1.4**; the MPC's Crocoddyl model keeps the nominal
**1.0** — a 40% model error. All arms are evaluated on the same 15 fixed
episodes; SAC arms use 3 seeds (band = min–max).

| Arm | 0 steps | 3k steps | 15k steps |
|---|---|---|---|
| MPC alone (nominal model) | −629 | −629 | −629 |
| **Residual SAC over MPC** | **−738** | **−406** | **−273** |
| SAC from scratch | −1452 | −832 | −236 |

Two claims the figure supports: the residual agent **never passes through the
catastrophic exploration phase** (from-scratch SAC sits near −1450 for its
first ~2,500 steps — on hardware, that phase is crashes), and it **beats the
mismatched MPC from ~2k steps on**.

Reproduce with `benchmark/residual_pendulum/` (about 5 CPU-minutes per arm);
full configuration and honest caveats — including why there is *no oracle
line* and why residual authority matters — in
[that directory's README](https://github.com/CHANGEME/blendmpc/tree/main/benchmark/residual_pendulum).
