"""Render the Go2 forward walk to a GIF (headless: MUJOCO_GL=osmesa or egl).

Writes docs/assets/go2_walk.gif. Frames are rendered every other control
step (25 fps at the 50 Hz control rate).
"""

from __future__ import annotations

import os

import mujoco
from PIL import Image

from blendmpc.envs.go2 import (
    Go2BalanceEnv,
    make_go2_trot_cycle,
    obs_to_state,
    quasi_static_torque,
    stand_state,
)
from blendmpc.solvers.crocoddyl import CrocoddylCyclicMPC

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "assets", "go2_walk.gif")
VX, SECONDS, W, H = 0.3, 8.0, 400, 300


def main() -> None:
    env = Go2BalanceEnv(command_vx=VX)
    mpc = CrocoddylCyclicMPC(
        lambda x0: make_go2_trot_cycle(vx=VX),
        u_init=quasi_static_torque(stand_state()),
    )
    obs, _ = env.reset(seed=0)

    renderer = mujoco.Renderer(env.model, H, W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 120.0, -12.0, 1.3

    frames = []
    steps = int(SECONDS / 0.02)
    for t in range(steps):
        obs, _, term, trunc, _ = env.step(mpc.action(obs_to_state(obs)))
        if term:
            raise RuntimeError(f"fell at step {t}")
        if t % 3 == 0:
            cam.lookat[:] = [float(obs[0]), float(obs[1]), 0.22]
            renderer.update_scene(env.data, camera=cam)
            img = Image.fromarray(renderer.render())
            frames.append(img.quantize(colors=128, dither=Image.Dither.NONE))

    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=60,
        loop=0,
        optimize=True,
    )
    print(
        f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB, "
        f"{len(frames)} frames, walked {float(obs[0]):.2f} m)"
    )


if __name__ == "__main__":
    main()
