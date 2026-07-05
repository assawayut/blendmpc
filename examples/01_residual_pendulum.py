"""Pure-MPC baseline through the residual wrapper (zero residual = pure MPC).

This is the starting point for residual RL: any agent trained on
``ResidualMPCEnv`` begins from this controller's performance instead of from
scratch. Typical Pendulum-v1 scores: random ~ -1200, well-tuned SAC ~ -150,
this MPC ~ -120 to -400 depending on the initial state.
"""

import gymnasium as gym
import numpy as np

from blendmpc.blends import ResidualMPCEnv
from blendmpc.envs.pendulum import make_pendulum_problem, obs_to_state
from blendmpc.solvers.crocoddyl import CrocoddylMPC


def main(episodes: int = 3, seed: int = 0) -> float:
    mpc = CrocoddylMPC(lambda x0: make_pendulum_problem(x0, horizon=40))
    env = ResidualMPCEnv(gym.make("Pendulum-v1"), mpc, obs_to_state)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_ret = False, 0.0
        while not done:
            action = np.zeros(env.action_space.shape)  # zero residual -> pure MPC
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        theta = np.degrees(np.arctan2(obs[1], obs[0]))
        print(f"episode {ep}: return={ep_ret:8.1f}  final angle={theta:6.1f} deg")
        returns.append(ep_ret)
    mean_ret = float(np.mean(returns))
    print(f"mean return over {episodes} episodes: {mean_ret:.1f}")
    return mean_ret


if __name__ == "__main__":
    main()
