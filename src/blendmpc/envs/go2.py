"""Unitree Go2 balance task: Crocoddyl whole-body OCP + MuJoCo plant.

The OCP keeps all four feet in rigid contact (``ContactModel3D``) and tracks
the standing base pose under torque limits; the plant is the MuJoCo menagerie
Go2 with direct torque actuators. Model and plant share joint ordering; the
observation is the Pinocchio-convention state ``x = [q, v]`` (base quaternion
xyzw, base velocity in the body frame), so ``obs_to_state`` is the identity.

Requires the ``quadruped`` extra: ``pip install mujoco robot_descriptions``.
Descriptions are downloaded and cached by ``robot_descriptions`` on first use.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
STAND_JOINTS = np.array([0.0, 0.9, -1.8] * 4)
STAND_HEIGHT = 0.27
CONTROL_DT = 0.02

_pin_robot = None


def load_go2_pinocchio():
    """Load (and cache) the Go2 Pinocchio model with a free-flyer base."""
    global _pin_robot
    if _pin_robot is None:
        import pinocchio as pin
        from robot_descriptions.loaders.pinocchio import load_robot_description

        _pin_robot = load_robot_description(
            "go2_description", root_joint=pin.JointModelFreeFlyer()
        )
    return _pin_robot


def stand_state():
    import pinocchio as pin

    model = load_go2_pinocchio().model
    q = pin.neutral(model)
    q[2] = STAND_HEIGHT
    q[7:] = STAND_JOINTS
    return np.concatenate([q, np.zeros(model.nv)])


class _Go2OcpContext:
    """Shared pieces for building Go2 whole-body nodes."""

    def __init__(self, trunk_mass_scale: float = 1.0):
        import crocoddyl
        import pinocchio as pin

        model = load_go2_pinocchio().model
        if trunk_mass_scale != 1.0:
            model = model.copy()
            inert = model.inertias[1]  # base link, attached to the free-flyer
            model.inertias[1] = pin.Inertia(
                inert.mass * trunk_mass_scale,
                inert.lever,
                inert.inertia * trunk_mass_scale,
            )
        self.model = model
        self.x_stand = stand_state()
        data = model.createData()
        pin.forwardKinematics(model, data, self.x_stand[: model.nq])
        pin.updateFramePlacements(model, data)
        self.foot_id = {f: model.getFrameId(f) for f in FEET}
        self.foot_ref = {f: data.oMf[self.foot_id[f]].translation.copy() for f in FEET}
        self.base_id = model.getFrameId("base")
        self.state = crocoddyl.StateMultibody(model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.nu = self.actuation.nu
        self.tau_max = model.effortLimit[6:]

    def node(
        self,
        stance_feet=FEET,
        swing_refs: dict | None = None,
        terminal: bool = False,
        dt: float = CONTROL_DT,
        swing_weight: float = 2e2,
    ):
        import crocoddyl
        import pinocchio as pin

        contacts = crocoddyl.ContactModelMultiple(self.state, self.nu)
        for f in stance_feet:
            contacts.addContact(
                f,
                crocoddyl.ContactModel3D(
                    self.state,
                    self.foot_id[f],
                    self.foot_ref[f],
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    self.nu,
                    np.array([0.0, 50.0]),
                ),
            )
        costs = crocoddyl.CostModelSum(self.state, self.nu)
        Mref = pin.SE3(np.eye(3), np.array([0.0, 0.0, STAND_HEIGHT]))
        costs.addCost(
            "base",
            crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(
                    np.array([10.0, 10.0, 50.0, 30.0, 30.0, 10.0])
                ),
                crocoddyl.ResidualModelFramePlacement(
                    self.state, self.base_id, Mref, self.nu
                ),
            ),
            10.0,
        )
        wx = np.concatenate(
            [np.zeros(6), 0.5 * np.ones(12), np.full(6, 2.0), 0.05 * np.ones(12)]
        )
        costs.addCost(
            "xreg",
            crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(wx),
                crocoddyl.ResidualModelState(self.state, self.x_stand, self.nu),
            ),
            1.0,
        )
        for f, ref in (swing_refs or {}).items():
            costs.addCost(
                f"swing_{f}",
                crocoddyl.CostModelResidual(
                    self.state,
                    crocoddyl.ResidualModelFrameTranslation(
                        self.state, self.foot_id[f], ref, self.nu
                    ),
                ),
                swing_weight,
            )
        if not terminal:
            costs.addCost(
                "ureg",
                crocoddyl.CostModelResidual(
                    self.state, crocoddyl.ResidualModelControl(self.state, self.nu)
                ),
                1e-3,
            )
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contacts, costs, 0.0, True
        )
        am = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0 if terminal else dt)
        if not terminal:
            am.u_lb = -self.tau_max
            am.u_ub = self.tau_max
        return am


def make_go2_balance_problem(
    x0: np.ndarray,
    horizon: int = 25,
    dt: float = CONTROL_DT,
    trunk_mass_scale: float = 1.0,
):
    """Whole-body balance ShootingProblem: four feet in contact, track stand.

    ``trunk_mass_scale`` scales the base-link inertia in the *model* — pass
    the plant's value to build an informed ("oracle") controller.
    """
    import crocoddyl

    ctx = _Go2OcpContext(trunk_mass_scale)
    return crocoddyl.ShootingProblem(
        np.asarray(x0, dtype=float),
        [ctx.node(dt=dt)] * horizon,
        ctx.node(terminal=True),
    )


TROT_PAIR_A = ("FL_foot", "RR_foot")
TROT_PAIR_B = ("FR_foot", "RL_foot")


def make_go2_trot_cycle(
    trunk_mass_scale: float = 1.0,
    double_support: int = 4,
    swing: int = 12,
    apex: float = 0.06,
    swing_weight: float = 1e3,
    vx: float = 0.0,
):
    """One trot gait cycle as a list of node models, plus a terminal.

    Cycle layout (dt = ``CONTROL_DT``): ``double_support`` all-feet nodes,
    ``swing`` nodes with the FL+RR pair tracking a sine-apex swing reference,
    then the same again for FR+RL. Intended for
    :class:`blendmpc.solvers.crocoddyl.CrocoddylCyclicMPC`.

    With ``vx == 0`` (trot in place) references are static and the return
    value is ``(cycle, terminal)``. With ``vx > 0`` the return value is
    ``(cycle, terminal, update_fn)``: swing targets follow a foothold
    schedule with stride ``vx * T_cycle`` (each foot placed so it sits at
    its nominal offset at mid-stance), the base reference advances at
    ``vx``, and ``update_fn(model, node_index, terminal=False)`` stamps a
    node with the references for its absolute time. Stance contact
    references are not updated — the contact models use velocity-level
    Baumgarte stabilization (position gain 0), so foothold positions enter
    only through the swing costs.
    """
    import pinocchio as pin

    ctx = _Go2OcpContext(trunk_mass_scale)
    n_cycle = 2 * (double_support + swing)
    t_cycle = n_cycle * CONTROL_DT

    def swing_ref(f, s):
        ref = ctx.foot_ref[f].copy()
        ref[2] += apex * np.sin(np.pi * s)
        return ref

    cycle = []
    slot_info = []  # per slot: None or (pair, s, nodes_to_touchdown)
    for pair in (TROT_PAIR_A, TROT_PAIR_B):
        stance = [f for f in FEET if f not in pair]
        for _ in range(double_support):
            cycle.append(ctx.node())
            slot_info.append(None)
        for k in range(swing):
            s = (k + 0.5) / swing
            cycle.append(
                ctx.node(
                    stance_feet=stance,
                    swing_refs={f: swing_ref(f, s) for f in pair},
                    swing_weight=swing_weight,
                )
            )
            slot_info.append((pair, s, swing - k))
    terminal = ctx.node(terminal=True)
    if vx == 0.0:
        return cycle, terminal

    # constant part: regulate toward the commanded body velocity
    x_ref = ctx.x_stand.copy()
    x_ref[ctx.model.nq] = vx
    for am in [*cycle, terminal]:
        am.differential.costs.costs["xreg"].cost.residual.reference = x_ref

    stance_nodes = n_cycle - swing  # each foot's stance duration, in nodes
    stride = vx * t_cycle

    def update_fn(am, node_index: int, terminal: bool = False):
        t = node_index * CONTROL_DT
        costs = am.differential.costs.costs
        costs["base"].cost.residual.reference = pin.SE3(
            np.eye(3), np.array([vx * t, 0.0, STAND_HEIGHT])
        )
        if terminal:
            return
        info = slot_info[node_index % n_cycle]
        if info is None:
            return
        pair, s, to_td = info
        # foothold: nominal offset under the body at mid-stance of the
        # upcoming stance phase; the previous foothold is one stride behind
        t_mid_stance = t + (to_td + stance_nodes / 2.0) * CONTROL_DT
        for f in pair:
            x_land = ctx.foot_ref[f][0] + vx * t_mid_stance
            ref = swing_ref(f, s)
            ref[0] = x_land - (1.0 - s) * stride
            costs[f"swing_{f}"].cost.residual.reference = ref

    return cycle, terminal, update_fn


def quasi_static_torque(x0: np.ndarray) -> np.ndarray:
    """Gravity-compensating joint torques at ``x0`` (cold-start init for MPC).

    Computed directly with Pinocchio (static equilibrium with least-norm
    contact forces at the four feet) rather than Crocoddyl's
    ``ShootingProblem.quasiStatic``, which we observed returning
    uninitialized memory for contact problems on some runs.
    """
    import pinocchio as pin

    model = load_go2_pinocchio().model
    data = model.createData()
    q = np.asarray(x0, dtype=float)[: model.nq]
    g = pin.computeGeneralizedGravity(model, data, q)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    Jc = np.vstack(
        [
            pin.getFrameJacobian(
                model, data, model.getFrameId(f), pin.LOCAL_WORLD_ALIGNED
            )[:3]
            for f in FEET
        ]
    )  # (12, nv)
    # Static equilibrium: [0; tau] + Jc^T f = g. The 6 unactuated base rows
    # determine f (least-norm); the joint rows then give tau.
    f = np.linalg.lstsq(Jc[:, :6].T, g[:6], rcond=None)[0]
    return g[6:] - Jc[:, 6:].T @ f


class Go2BalanceEnv(gym.Env):
    """MuJoCo Go2 balance environment (Gymnasium API).

    Action: 12 joint torques (N·m), clipped to actuator limits.
    Observation: Pinocchio-convention state ``[q(19), v(18)]``.
    Reward: negative weighted base-pose error plus a small effort penalty.
    Episode ends when the trunk drops or tilts past recovery, or at 500 steps.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trunk_mass_scale: float = 1.0,
        max_steps: int = 500,
        command_vx: float = 0.0,
    ):
        import mujoco
        from robot_descriptions import go2_mj_description

        self.command_vx = command_vx

        scene = os.path.join(os.path.dirname(go2_mj_description.MJCF_PATH), "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(
            scene if os.path.exists(scene) else go2_mj_description.MJCF_PATH
        )
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco
        trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        assert trunk >= 0
        self.model.body_mass[trunk] *= trunk_mass_scale
        self.model.body_inertia[trunk] *= trunk_mass_scale
        self._frame_skip = max(1, round(CONTROL_DT / self.model.opt.timestep))
        self._max_steps = max_steps
        self._key = self.model.key("home").id

        tau_max = load_go2_pinocchio().model.effortLimit[6:]
        self.action_space = gym.spaces.Box(
            low=-tau_max.astype(np.float32), high=tau_max.astype(np.float32)
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float64
        )
        self.np_random = np.random.default_rng()

    # -- state conversion (MuJoCo -> Pinocchio conventions) ------------------
    def _obs(self) -> np.ndarray:
        import pinocchio as pin

        qpos, qvel = self.data.qpos, self.data.qvel
        q = np.empty(19)
        q[0:3] = qpos[0:3]
        q[3:6] = qpos[4:7]  # x, y, z of the quaternion
        q[6] = qpos[3]  # w last in Pinocchio
        q[7:] = qpos[7:]
        R = pin.Quaternion(q[3:7]).toRotationMatrix()
        v = np.empty(18)
        v[0:3] = R.T @ qvel[0:3]  # world linear -> body frame
        v[3:6] = qvel[3:6]  # angular velocity already in body frame
        v[6:] = qvel[6:]
        return np.concatenate([q, v])

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        mujoco, m, d = self._mujoco, self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self._key)
        rng = self.np_random
        d.qpos[2] += rng.uniform(-0.03, 0.02)
        rpy = rng.uniform(-0.12, 0.12, 3)
        d.qpos[3:7] = _rpy_to_wxyz(rpy)
        d.qpos[7:] += rng.uniform(-0.12, 0.12, 12)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        mujoco, m, d = self._mujoco, self.model, self.data
        d.ctrl[:] = np.clip(action, self.action_space.low, self.action_space.high)
        for _ in range(self._frame_skip):
            mujoco.mj_step(m, d)
        self._t += 1
        obs = self._obs()
        z = obs[2]
        import pinocchio as pin

        rpy = pin.rpy.matrixToRpy(pin.Quaternion(obs[3:7]).toRotationMatrix())
        if self.command_vx == 0.0:
            xy_err = 5.0 * (obs[0] ** 2 + obs[1] ** 2)
        else:
            # locomotion: track commanded forward speed, hold the lateral line
            xy_err = 4.0 * (obs[19] - self.command_vx) ** 2 + 5.0 * obs[1] ** 2
        pose_err = (
            50.0 * (z - STAND_HEIGHT) ** 2
            + xy_err
            + 3.0 * (rpy[0] ** 2 + rpy[1] ** 2)
            + 1.0 * rpy[2] ** 2
        )
        effort = 1e-5 * float(np.square(d.ctrl).sum())
        reward = -(pose_err + effort)
        fallen = bool(z < 0.15 or abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8)
        if fallen:
            reward -= 10.0
        terminated = fallen
        truncated = self._t >= self._max_steps
        return obs, reward, terminated, truncated, {}


def _rpy_to_wxyz(rpy):
    import pinocchio as pin

    q = pin.Quaternion(pin.rpy.rpyToMatrix(*rpy)).coeffs()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def obs_to_state(obs: np.ndarray) -> np.ndarray:
    """The observation already is the Pinocchio state."""
    return np.asarray(obs, dtype=float)
