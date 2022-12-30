from typing import Tuple

import brax
from brax import jumpy as jp
from brax import math
from brax.envs import env


class Mesh(env.Env):

    def __init__(self, legacy_spring=False, **kwargs):
        config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)

    def reset(self, rng: jp.ndarray) -> env.State:
        qp = self.sys.default_qp()
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            # 'hits': zero,
            # 'weightedHits': zero,
            # 'movingToTarget': zero,
            # 'torsoIsUp': zero,
            # 'torsoHeight': zero
        }
        info = {
            # 'rng': rng
        }
        # qp.defaults[0].qps[1].pos.x = 0
        return env.State(qp, obs, reward, done, metrics, info)

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)

        reward = 0.

        return state.replace(qp=qp, obs=obs, reward=reward)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        return ([
        ])


_SYSTEM_CONFIG = """
  dt: 0.05 substeps: 10 friction: 1.0
  gravity { z: -9.8 }
  dynamics_mode: "pbd"
  bodies {
    name: "Mesh" mass: 1
    colliders { mesh { name: "Cylinder" scale: 0.1 } }
    inertia { x: 1 y: 1 z: 1 }
  }
  bodies { name: "Ground" frozen: { all: true } colliders { plane {} } }
  defaults {
    # Initial position is high up in the air.
    qps { name: "Mesh" pos: {x: 0 y: 0 z: 0} }
  }
  mesh_geometries {
    name: "Cylinder"
    path: "stonehenge.stl"
  }
  """
