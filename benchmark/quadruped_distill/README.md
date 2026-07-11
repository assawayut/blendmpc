# Distilling the trot MPC (Go2)

Behavior-clone the trot-in-place gait MPC into a small MLP. The expert is
time-varying (it tracks a gait schedule), so a memoryless clone cannot
represent it — the student takes the gait phase (sin/cos of the cycle
position) as an input next to the 37-dimensional state. 60 expert episodes
(30k pairs), plain MSE behavior cloning.

```bash
python train.py
```

## Results

| Arm | return | survival | FL airborne | latency / action |
|---|---|---|---|---|
| Trot MPC (expert) | −6.23 | 5/5 | 22% | 4.0 ms |
| **BC student** | −6.75 | 5/5 | **22%** | **28 µs (142×)** |

The student reproduces the *gait*, not just the pose: identical airborne
fraction, all episodes survive, return within 8% of the expert. At 28 µs per
action the policy runs at multi-kHz rates on embedded hardware with no
solver, model, or gait scheduler on board — the phase counter is the only
runtime state.

## Notes

- Phase conditioning is what makes this work; without it the regression
  target is multivalued (same state, different gait phase → different
  torque) and the student collapses toward weight-shifting.
- The pendulum's DAgger warning applies unchanged: aggregate labels only if
  your student class can represent multimodal targets.
