import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import sys

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}

class InvertedDoublePendulumEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2.0) ** 2

        v1, v2 = self.data.qvel[1:3]
        vel_penalty = 1e-2 * v1**2 + 5e-3 * v2**2
        alive_bonus = 5

        r =  alive_bonus + 3*(ob[3] + ob[4]) - (3*vel_penalty) - dist_penalty  # Origin for 60 epi

        # if y > 0.9 and (ob[3]+ob[4]) > 1.9: # Origin for 60 epi
        #      r += 1
        #      if vel_penalty < 0.01:
        #         r += 5 * (1 - 10*vel_penalty)
        #         print("# ------------- ", vel_penalty)

        if y > 0.9 and (ob[3]+ob[4]) > 1.9: # Origin for 60 epi
             r += 1
             if vel_penalty < 0.01:
                r += 5 * (1 - 10*vel_penalty)
                print("# ------------- ", vel_penalty)
                if ob[3] > 0.99 and vel_penalty < 0.005:
                    r = r * 1.5
                    print("reward bumping !!!!!!!!!!!!!!!! ")


        # Terminal truncated condition
        terminated = False #bool(y <= 1) # True/False
        truncated  = False
        #truncated = True if vel_penalty > 15 else False

        if self.render_mode == "human":
            self.render()
        return ob, r, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart x pos
                np.sin(self.data.qpos[1:]),  # link angles
                np.cos(self.data.qpos[1:]),
                np.clip(self.data.qvel, -10, 10),
                np.clip(self.data.qfrc_constraint, -10, 10),
            ]
        ).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + np.array([0,np.pi, 0]) +
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        return self._get_obs()