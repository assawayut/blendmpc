# Contributing to blendmpc

Thanks for considering a contribution! The project is young and small — this
is a great time to shape it.

## What's most welcome

- **New blend patterns** from the MPC×RL literature (`src/blendmpc/blends/`).
  A blend must: compose against `MPCPolicy` only (no solver-specific code),
  cite the paper(s) it implements in its module docstring, and ship with a
  unit test, a closed-loop behavioral test, and an example script.
- **New solver backends** (`src/blendmpc/solvers/`). A backend implements
  `MPCPolicy.solve()` and must pass the existing blend test suite unchanged.
- **New env/model pairs** (`src/blendmpc/envs/`). The OCP model should match
  the environment's dynamics exactly where possible, with a test proving it
  (see `test_pendulum_model_matches_gym_dynamics`).
- Bug reports with a minimal script — behavioral bugs in control code are
  subtle, numbers beat descriptions.

## Development setup

```bash
git clone https://github.com/CHANGEME/blendmpc && cd blendmpc
pip install -e ".[crocoddyl,test]"
pre-commit install
pytest
```

## Ground rules

- Style is enforced by ruff via pre-commit; CI must be green.
- Keep the core thin: solver features belong in solvers, RL algorithms belong
  in RL libraries. When in doubt, open an issue before writing code.
- Every public function gets a docstring; blends document their math and cite
  their sources.
