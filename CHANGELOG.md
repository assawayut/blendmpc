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

### Changed
- Pendulum model cost uses a smooth surrogate angle term (`2(1-cos)`)
  instead of Gym's kinked normalized angle, which breaks gradient-based OC
  at the hanging position (closed-loop scores are still Gym's reward).
- `ResidualMPCEnv` action space dtype is float32 (Gymnasium/SB3 convention).
