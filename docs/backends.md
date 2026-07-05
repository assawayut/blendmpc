# Backends

A backend adapts one trajectory-optimization library to
`blendmpc.core.MPCPolicy`. The contract:

- implement `solve(x0, us_init, xs_init) -> MPCSolution` — one open-loop OCP
  from `x0`, warm-started when trajectories are given;
- the existing blend test suite must pass over the new backend **unchanged**.

## Crocoddyl — `blendmpc.solvers.crocoddyl.CrocoddylMPC`

```python
CrocoddylMPC(problem_factory,           # x0 -> crocoddyl.ShootingProblem
             solver_cls=SolverBoxFDDP,  # honors control limits
             max_iter=5,                # warm-started iterations per step
             max_iter_first=100,        # cold-solve budget
             u_init=None,               # symmetry-breaking cold init
             multistart_iter=0)         # optional per-step cold restart
```

Installs from PyPI (`pip install crocoddyl`). FDDP's feasibility-driven
structure is strong on swing-up-like problems; the box solvers enforce control
bounds inside the backward pass.

## acados — `blendmpc.solvers.acados.AcadosMPC`

```python
AcadosMPC(ocp_factory,        # x0 -> AcadosOcp (CasADi model, EXTERNAL cost)
          build_dir=None,     # where generated C code is compiled
          u_init=None)        # symmetry-breaking cold init
```

Needs the [acados C library](https://docs.acados.org/installation) built from
source (`ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH`). The first solve generates and
compiles problem-specific C code into `build_dir`; `reset()` keeps the
compiled solver and only clears the warm start. See
`blendmpc.envs.pendulum_acados` for a complete `AcadosOcp` factory using
discrete dynamics and an external cost.

!!! note "Same OCP, different local solutions"
    On the pendulum swing-up, FDDP (Crocoddyl) and SQP (acados) solve the
    *same* OCP — verified by cross-seeding their solutions — but from some
    initial states SQP settles on a one-extra-swing route where FDDP goes
    direct. Local solvers differ in which basin they find, not just in speed.
    If a backend "underperforms", seed it with the other backend's solution
    before concluding the adapter is wrong.

## Writing a new backend

1. Subclass `MPCPolicy`, implement `solve()`. Map your solver's warm-start
   API to `us_init`/`xs_init` and return an `MPCSolution` (put solver
   diagnostics in `info`).
2. Decide what `reset()` costs: rebuild cheap per-episode state, keep
   expensive compiled artifacts.
3. Copy `tests/test_acados_backend.py` as a template: interface parity, torque
   limits respected, stabilization, zero-residual-equals-MPC through
   `ResidualMPCEnv`.
