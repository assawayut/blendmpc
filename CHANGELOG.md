# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `MPCPolicy` / `MPCSolution` core interface with receding-horizon
  warm-start shifting.
- Crocoddyl backend (`CrocoddylMPC`, BoxFDDP by default).
- Blends: `ResidualMPCEnv`, `PolicyWarmStartMPC`, learned terminal cost
  (`make_learned_terminal`), `collect_expert_dataset`.
- Exact Crocoddyl model of Gymnasium `Pendulum-v1` with analytic derivatives.
- Residual pendulum example and test suite.
- CI (GitHub Actions), pre-commit (ruff), contribution guidelines.
- Pendulum model accepts physical parameters (`m`, `l`, `g`) for
  model-mismatch experiments.
- `CrocoddylMPC`: symmetry-breaking cold init (`u_init`) and optional
  cold-restart multistart (`multistart_iter`) to escape lazy local minima in
  receding-horizon operation.
- `benchmark/residual_pendulum/`: residual SAC vs from-scratch SAC vs MPC
  under 40% mass mismatch, with the README learning-curve figure.
- acados backend (`AcadosMPC`) with the same warm-start and
  symmetry-breaking cold-init semantics as the Crocoddyl backend;
  `reset()` keeps the compiled solver. acados/CasADi twin of the pendulum
  OCP (`blendmpc.envs.pendulum_acados`) and backend-parity tests (skipped
  when acados is absent).
- Documentation site (mkdocs-material): getting started, one page per
  blend with math and paper citations, backends guide, benchmark results,
  papers index. Built strictly in CI and deployed to GitHub Pages on push
  to main.
- Benchmarks for the remaining blends (each with honest negative results
  documented in its README): distillation (`benchmark/distill_pendulum/`),
  policy warm start (`benchmark/warmstart_pendulum/`), learned terminal
  cost (`benchmark/terminal_pendulum/`).
- `PolicyWarmStartMPC`: `compare_with_default` option — best-of-two cold
  starts (policy seed vs the wrapped MPC's own init), keeping the
  lower-cost solution.
- `make_learned_terminal` / `with_learned_terminal`: optional analytic
  `grad_fn`/`hess_fn` (e.g. torch autograd) — finite differences on a
  float32 network produce unusable Hessians.
- Unitree Go2 whole-body balance task (`blendmpc.envs.go2`, `quadruped`
  extra): four-feet-in-contact Crocoddyl OCP, MuJoCo menagerie plant with
  torque actuation, one shared Pinocchio-convention state representation,
  and an analytic quasi-static torque routine (Crocoddyl's
  `ShootingProblem.quasiStatic` returned uninitialized memory for this
  contact problem).
- `benchmark/quadruped_balance/`: residual SAC over whole-body MPC under a
  doubled unmodeled trunk mass, with oracle and from-scratch arms.
- `CrocoddylCyclicMPC`: receding-horizon MPC over a periodic sequence of
  node models (gait schedules), advancing the cycle with
  `ShootingProblem.circularAppend`.
- Trot-in-place gait for Go2 (`make_go2_trot_cycle`) with a regression test
  asserting the feet actually leave the ground, and
  `benchmark/quadruped_trot/`: residual SAC over the gait MPC under a 3×
  unmodeled overload, where the learned residual ends 2× better than the
  true-model controller.
- Forward-velocity locomotion: `make_go2_trot_cycle(vx=...)` returns a
  per-node reference updater (foothold schedule with stride vx × cycle,
  advancing base reference) that `CrocoddylCyclicMPC` applies as nodes
  enter the horizon; `Go2BalanceEnv(command_vx=...)` switches the reward
  to velocity tracking. 0.2–0.3 m/s track to within a few mm/s closed
  loop; regression test included.

### Changed
- Pendulum model cost uses a smooth surrogate angle term (`2(1-cos)`)
  instead of Gym's kinked normalized angle, which breaks gradient-based OC
  at the hanging position (closed-loop scores are still Gym's reward).
- `ResidualMPCEnv` action space dtype is float32 (Gymnasium/SB3 convention).
