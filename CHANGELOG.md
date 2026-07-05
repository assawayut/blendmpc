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
