# Distillation benchmark (Pendulum-v1)

Behavior-clone the Crocoddyl MPC expert (H=30, exact model) into a 2×64 MLP
via `collect_expert_dataset`, then compare on the shared 15 eval seeds.

| Arm | mean | worst | latency / action |
|---|---|---|---|
| MPC expert | −320 | −1498 | 0.6 ms |
| **BC student** (10k pairs) | **−423** | −1519 | **24 µs (25×)** |
| BC + 1 DAgger round (20k pairs) | −558 | −1576 | 24 µs |

```bash
python train.py
```

## Honest notes

- The student recovers most of the expert's behavior at **25× lower latency**
  and with no model or solver at deployment. On this small problem the MPC is
  already sub-millisecond; the latency ratio grows with problem size.
- **Naive DAgger hurt.** The MPC expert is near-bang-bang: on the messier
  states the student visits, +2 and −2 are often *both* valid labels, and
  MSE regression averages contradictory labels into useless mid-torques (the
  round-2 fit loss jumps from 0.0005 to 0.72 — the signature of label
  conflict, not underfitting). Multimodal experts need multimodal students
  (e.g. discretized outputs or diffusion policies) before DAgger pays off.
- Artifacts saved for the other benchmarks: `results/bc_policy.pt` (the plain
  BC student — the better one) and `results/expert_dataset.npz`.
