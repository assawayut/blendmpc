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
    x_stand = stand_state()

    data = model.createData()
    q_stand = x_stand[: model.nq]
    pin.forwardKinematics(model, data, q_stand)
    pin.updateFramePlacements(model, data)
    foot_ids = [model.getFrameId(f) for f in FEET]
    foot_refs = [data.oMf[i].translation.copy() for i in foot_ids]
    base_id = model.getFrameId("base")

    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu

    def node(terminal: bool = False):
        contacts = crocoddyl.ContactModelMultiple(state, nu)
        for fid, ref in zip(foot_ids, foot_refs):
            contacts.addContact(
                model.frames[fid].name,
                crocoddyl.ContactModel3D(
                    state,
                    fid,
                    ref,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    nu,
                    np.array([0.0, 50.0]),
                ),
            )
        costs = crocoddyl.CostModelSum(state, nu)
        Mref = pin.SE3(np.eye(3), np.array([0.0, 0.0, STAND_HEIGHT]))
        costs.addCost(
            "base",
            crocoddyl.CostModelResidual(
                state,
                crocoddyl.ActivationModelWeightedQuad(
                    np.array([10.0, 10.0, 50.0, 30.0, 30.0, 10.0])
                ),
                crocoddyl.ResidualModelFramePlacement(state, base_id, Mref, nu),
            ),
            10.0,
        )
        wx = np.concatenate(
            [np.zeros(6), 0.5 * np.ones(12), np.full(6, 2.0), 0.05 * np.ones(12)]
        )
        costs.addCost(
            "xreg",
            crocoddyl.CostModelResidual(
                state,
                crocoddyl.ActivationModelWeightedQuad(wx),
                crocoddyl.ResidualModelState(state, x_stand, nu),
            ),
            1.0,
        )
        if not terminal:
            costs.addCost(
                "ureg",
                crocoddyl.CostModelResidual(
                    state, crocoddyl.ResidualModelControl(state, nu)
                ),
                1e-3,
            )
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            state, actuation, contacts, costs, 0.0, True
        )
        am = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0 if terminal else dt)
        if not terminal:
            am.u_lb = -model.effortLimit[6:]
            am.u_ub = model.effortLimit[6:]
        return am

    return crocoddyl.ShootingProblem(
        np.asarray(x0, dtype=float), [node()] * horizon, node(terminal=True)
    )


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

    def __init__(self, trunk_mass_scale: float = 1.0, max_steps: int = 500):
        import mujoco
        from robot_descriptions import go2_mj_description

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
        pose_err = (
            50.0 * (z - STAND_HEIGHT) ** 2
            + 5.0 * (obs[0] ** 2 + obs[1] ** 2)
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
