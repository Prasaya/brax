# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains a humanoid to run in the +x direction."""
from typing import Tuple, List
import functools
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import brax
# from brax.envs import multiagent_env
from brax.envs import env
from brax.physics import bodies
from brax.physics.base import take

from google.protobuf import text_format


class DoubleHumanoid(env.Env):
    """Trains a humanoid to run in the +x direction."""

    def __init__(self, **kwargs):
        # TODO: define a function to copy the system config automatically based on num_agents
        self.num_agents = 2
        config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
        super().__init__(config, **kwargs)

        # body info
        self.body_parts = ["torso", "lwaist", "pelvis", 
                           "right_thigh", "right_shin", 
                           "left_thigh", "left_shin", 
                           "right_upper_arm", "right_lower_arm",
                           "left_upper_arm", "left_lower_arm"
                           ]
        self.world_parts = ["floor"]
        # actuator info
        self.agent_dof = 17
        assert self.agent_dof * self.num_agents == self.sys.num_joint_dof
        self.torque_1d_act_idx = jnp.array([2, 6, 10, 13, 16])
        self.torque_2d_act_idx = jnp.array([[0, 1], [11, 12], [14, 15]])
        self.torque_3d_act_idx = jnp.array([[3, 4, 5], [7, 8, 9]])
        # joint info
        self.agent_joints = 10
        self.num_joints_1d = 5
        self.num_joints_2d = 3
        self.num_joints_3d = 2

        # info to differentiate humanoids
        all_bodies = bodies.Body.from_config(config)  # body object only used to get object mass and inertia
        all_bodies = take(all_bodies, all_bodies.idx[:-len(self.world_parts)])  # skip the world bodies
        self.num_body_parts = len(self.body_parts)

        for i in range(self.num_agents):
            # get system body idx from self.sys
            body_idxs = {f"{body_part}{i}": self.sys.body_idx[f"{body_part}{i}"] for body_part in self.body_parts}
            setattr(self, f"agent{i}_idxs", body_idxs)

            # get mass, inertia from Body object
            body = take(all_bodies, all_bodies.idx[i * self.num_body_parts: i * self.num_body_parts + self.num_body_parts ])
            assert len(body.idx) == self.num_body_parts
            setattr(self, f"mass{i}", body.mass.reshape(-1, 1))
            setattr(self, f"inertia{i}", body.inertia)
        self.mass = jnp.array([getattr(self, f"mass{i}") for i in range(self.num_agents)])
        self.inertia = jnp.array([getattr(self, f"inertia{i}") for i in range(self.num_agents)])

        self.floor_idx = self.sys.body_idx["floor"]
        # how far apart to initialize humanoids
        self.field_distance = 20

    def update_parts_xyz(self, carry, part_idx):
        qp_pos, xyz_offset = carry
        qp_pos = jax.ops.index_update(qp_pos, jax.ops.index[part_idx], 
                                      xyz_offset+qp_pos[jax.ops.index[part_idx]]
                                      )
        return (qp_pos, xyz_offset), ()

    def set_agent_xyz(self, carry, part_idxs): 
        qp_pos, rng = carry
        rng, xyz_offset = self._random_target(rng)
        (qp_pos, xyz_offset), _ = jax.lax.scan(
          self.update_parts_xyz, (qp_pos, xyz_offset), part_idxs
          )
        return (qp_pos, rng), ()
    
    def reset(self, rng: jnp.ndarray) -> env.State:
        """Resets the environment to an initial state."""
        qp = self.sys.default_qp()
        # move the humanoids to different positions
        pos = qp.pos
        agents_parts_idxs = jnp.array([list(getattr(self, f"agent{i}_idxs").values()) for i in range(self.num_agents)])
        (pos, rng), _ = jax.lax.scan(
          self.set_agent_xyz, (pos, rng), agents_parts_idxs
          )
        qp = dataclasses.replace(qp, pos=pos)

        info = self.sys.info(qp)
        qp, info = self.sys.step(qp,
                                 jax.random.uniform(rng, (self.action_size,)) * .5) # action size is for all agents

        all_obs = self._get_obs(qp, info, jnp.zeros((self.num_agents, self.agent_dof)))
        reward = jnp.zeros((self.num_agents,))
        done = 0
        steps = jnp.zeros(1)
        metrics = {
            'reward_linvel': jnp.zeros((self.num_agents,)),
            'reward_quadctrl': jnp.zeros((self.num_agents,)),
            'reward_alive': jnp.zeros((self.num_agents,)),
            'reward_impact': jnp.zeros((self.num_agents,))
        }
        return env.State(rng, qp, info, all_obs, reward, done, steps, metrics)

    def step(self, state: env.State, action: jnp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        rng = state.rng
        # note the minus sign.  reverse torque improves performance over a range of
        # hparams.  as to why: ¯\_(ツ)_/¯
        qp, info = self.sys.step(state.qp, -action.flatten())
        all_obs = self._get_obs(qp, info, action) # should this be - action?

        reward, lin_vel_cost, quad_ctrl_cost, alive_bonus, quad_impact_cost = self._compute_reward(state, action, qp)
        metrics = {
            'reward_linvel': lin_vel_cost,
            'reward_quadctrl': -quad_ctrl_cost,
            'reward_alive': alive_bonus,
            'reward_impact': -quad_impact_cost
        }

        steps = state.steps + self.action_repeat
        done = self._compute_done(qp, steps)
    
        return env.State(rng, qp, info, all_obs, reward, done, steps, metrics)

    def _get_obs(self, qp: brax.QP, info: brax.Info, action: jnp.ndarray):
        all_obs = []
        # TODO: figure out how to jit self._get_agent_obs
        # (qp, info, action), all_obs = jax.lax.scan(
        #   self._get_agent_obs, (qp, info, action), jnp.arange(self.num_agents))
        for agent_idx in range(self.num_agents):
           (qp, info, action), obs = self._get_agent_obs((qp, info, action), agent_idx)
           all_obs.append(obs)
        return all_obs

    def _compute_reward(self, state: env.State, action: jnp.ndarray, qp: brax.QP):
        # self.mass has shape (num_agents, num_bodies_per_agent, 1)
        # self.inertia has shape (num_agents, num_bodies_per_agent, 3)

        # TODO: how to ensure ordering of reshaping is correct??
        pos_before = jnp.reshape(state.qp.pos[:-1], (self.num_agents, self.num_body_parts, 3))  # ignore floor at last index
        pos_after = jnp.reshape(qp.pos[:-1], (self.num_agents, self.num_body_parts, 3))  # ignore floor at last index

        com_before = jnp.sum(pos_before * self.mass, axis=1) / jnp.sum(self.mass, axis=1)
        com_after = jnp.sum(pos_after * self.mass, axis=1) / jnp.sum(self.mass, axis=1)

        lin_vel_cost = 1.25 * (com_after[:, 0] - com_before[:, 0]) / self.sys.config.dt

        reshaped_actions = jnp.reshape(action, (self.num_agents, self.agent_dof))
        quad_ctrl_cost = .01 * jnp.sum(jnp.square(reshaped_actions), axis=1)
        # can ignore contact cost, see: https://github.com/openai/gym/issues/1541
        quad_impact_cost = jnp.zeros(self.num_agents)
        alive_bonus = 5.0 * jnp.ones(self.num_agents)

        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        return reward, lin_vel_cost, quad_ctrl_cost, alive_bonus, quad_impact_cost
    
    def _compute_done(self, qp: brax.QP, steps: int, done_thres=0.75):
        """Return done if the proportion of agents that are done surpasses
        done_thres
        """
        torsos_idxs = jnp.arange(self.num_agents) * self.num_body_parts
        torsos_zdim = take(qp.pos[:, 2], torsos_idxs)

        done_cond0 = jnp.where(steps >= self.episode_length, x=1.0, y=0.0)
        done_cond1 = jnp.where(torsos_zdim < 0.6, x=1.0, y=0.0)
        done_cond2 = jnp.where(torsos_zdim > 2.1, x=1.0, y=0.0)

        done_vec = done_cond0 + done_cond1 + done_cond2
        done_vec = jnp.where(done_vec > 0.0, x=1.0, y=0.0)

        done_ratio = jnp.sum(done_vec) / self.num_agents
        done = jnp.where(done_ratio > done_thres, x=1.0, y=0.0)
        return done

    def _get_agent_obs(self, carry, agent_idx) -> jnp.ndarray:
        """Observe humanoid body position, velocities, and angles."""
        qp, info, action = carry
        qpos, qvel = self._get_agent_qpos_qvel(agent_idx, qp)
        qfrc_actuator = self._get_agent_qfrc(agent_idx, action[agent_idx])
        cfrc_ext = self._get_agent_cfrc_ext(agent_idx, info)
        cinert, cvel = self._get_agent_com_obs(agent_idx, qp)
        obs = jnp.expand_dims(jnp.concatenate(qpos + qvel + cinert + cvel + qfrc_actuator + \
                               cfrc_ext), axis=0)
        return (qp, info, action), obs

    def _get_agent_qpos_qvel(self, agent_idx: int, qp: brax.QP) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Some pre-processing to pull joint angles and velocities
        """
        # TODO: move outside this function
        joint_1d_angle, joint_1d_vel = self.sys.joint_revolute.angle_vel(qp)
        joint_2d_angle, joint_2d_vel = self.sys.joint_universal.angle_vel(qp)
        joint_3d_angle, joint_3d_vel = self.sys.joint_spherical.angle_vel(qp)

        idx_offset = agent_idx * self.num_joints_1d
        joint_1d_angle = take(joint_1d_angle, jnp.arange(idx_offset, idx_offset + self.num_joints_1d))
        joint_1d_vel = take(joint_1d_vel, jnp.arange(idx_offset, idx_offset + self.num_joints_1d))
        idx_offset = agent_idx * self.num_joints_2d
        joint_2d_angle = take(joint_2d_angle, jnp.arange(idx_offset, idx_offset + self.num_joints_2d))
        joint_2d_vel = take(joint_2d_vel, jnp.arange(idx_offset, idx_offset + self.num_joints_2d))
        idx_offset = agent_idx * self.num_joints_3d
        joint_3d_angle = take(joint_3d_angle, jnp.arange(idx_offset, idx_offset + self.num_joints_3d))
        joint_3d_vel = take(joint_3d_vel, jnp.arange(idx_offset, idx_offset + self.num_joints_3d))

        # qpos:
        # Z of the torso of agent idx (1,)
        # orientation of the torso as quaternion (4,)
        # joint angles, all dofs (8,)
        agent_torso_idx = agent_idx * self.num_body_parts
        qpos = [
            qp.pos[agent_torso_idx, 2:], qp.rot[agent_torso_idx],
            *joint_1d_angle, *joint_2d_angle, *joint_3d_angle
        ]

        # qvel:
        # velocity of the torso (3,)
        # angular velocity of the torso (3,)
        # joint angle velocities, all dofs (8,)
        qvel = [
            qp.vel[agent_torso_idx], qp.ang[agent_torso_idx],
            *joint_1d_vel, *joint_2d_vel, *joint_3d_vel
        ]
        return qpos, qvel

    def _get_agent_qfrc(self, agent_idx: int, agent_action: jnp.ndarray) -> List[jnp.ndarray]:
        # actuator forces
        idx_offset = agent_idx * self.num_joints_1d
        torque_1d = take(agent_action, self.torque_1d_act_idx)
        torque_1d *= take(self.sys.torque_1d.strength, 
          jnp.arange(idx_offset, idx_offset + self.num_joints_1d))

        idx_offset = agent_idx * self.num_joints_2d
        torque_2d = take(agent_action, self.torque_2d_act_idx)
        torque_2d = torque_2d.reshape(torque_2d.shape[:-2] + (-1,))
        torque_2d *= jnp.repeat(take(self.sys.torque_2d.strength, 
          jnp.arange(idx_offset, idx_offset + self.num_joints_2d)),
          2)

        idx_offset = agent_idx * self.num_joints_3d
        torque_3d = take(agent_action, self.torque_3d_act_idx)
        torque_3d = torque_3d.reshape(torque_3d.shape[:-2] + (-1,))
        torque_3d *= jnp.repeat(take(self.sys.torque_3d.strength, 
          jnp.arange(idx_offset, idx_offset + self.num_joints_3d)),
          3)
        qfrc_actuator = [torque_1d, torque_2d, torque_3d]
        return qfrc_actuator

    def _get_agent_cfrc_ext(self, agent_idx: int, info: brax.Info) -> List[jnp.ndarray]:
        agent_torso_idx = agent_idx * self.num_body_parts
        # external contact forces:
        # delta velocity (3,), delta ang (3,) * num bodies in the system
        cfrc_ext = [info.contact.vel[agent_torso_idx:agent_torso_idx + self.num_body_parts],
                    info.contact.ang[agent_torso_idx:agent_torso_idx + self.num_body_parts]
                    ]
        # flatten bottom dimension
        cfrc_ext = [x.reshape(x.shape[:-2] + (-1,)) for x in cfrc_ext]
        return cfrc_ext

    def _get_agent_com_obs(self, agent_idx: int, qp: brax.QP) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """Get center of mass observations for one agent"""
        agent_torso_idx = agent_idx * self.num_body_parts
        agent_mass = getattr(self, f"mass{agent_idx}")
        agent_inertia = getattr(self, f"inertia{agent_idx}")
        
        body_pos = qp.pos[agent_torso_idx:agent_torso_idx + self.num_body_parts]  # ignore floor at last index
        body_vel = qp.vel[agent_torso_idx:agent_torso_idx + self.num_body_parts]  # ignore floor at last index

        com_vec = jnp.sum(body_pos * agent_mass, axis=0) / jnp.sum(agent_mass)
        com_vel = body_vel * agent_mass / jnp.sum(agent_mass)

        def v_outer(a):
          return jnp.outer(a, a)

        def v_cross(a, b):
          return jnp.cross(a, b)

        v_outer = jax.vmap(v_outer, in_axes=[0])
        v_cross = jax.vmap(v_cross, in_axes=[0, 0])

        disp_vec = body_pos - com_vec
        # there are 11 bodies for each humanoid
        com_inert = agent_inertia + agent_mass.reshape(
            (11, 1, 1)) * ((jnp.linalg.norm(disp_vec, axis=1)**2.).reshape(
                (11, 1, 1)) * jnp.stack([jnp.eye(3)] * 11) - v_outer(disp_vec))

        cinert = [com_inert.reshape(-1)]

        square_disp = (1e-7 + (jnp.linalg.norm(disp_vec, axis=1)**2.)).reshape(
            (11, 1))
        com_angular_vel = (v_cross(disp_vec, body_vel) / square_disp)
        cvel = [com_vel.reshape(-1), com_angular_vel.reshape(-1)]

        return cinert, cvel

    def _random_target(self, rng: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns a target location in a random circle on xz plane."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = self.field_distance * jax.random.uniform(rng1)
        ang = jnp.pi * 2. * jax.random.uniform(rng2)
        target_x = dist * jnp.cos(ang)
        target_y = dist * jnp.sin(ang)
        target_z = 0
        target = jnp.array([target_x, target_y, target_z]).transpose()
        return rng, target

_HUMANOID0_CONFIG ="""
bodies {
  name: "torso0"
  colliders {
    position {
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.07
      length: 0.28
    }
  }
  colliders {
    position {
      z: 0.19
    }
    capsule {
      radius: 0.09
      length: 0.18
    }
  }
  colliders {
    position {
      x: -0.01
      z: -0.12
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.06
      length: 0.24
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 8.907463
}
bodies {
  name: "lwaist0"
  colliders {
    position {
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.06
      length: 0.24
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.2619467
}
bodies {
  name: "pelvis0"
  colliders {
    position {
      x: -0.02
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.09
      length: 0.32
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 6.6161942
}
bodies {
  name: "right_thigh0"
  colliders {
    position {
      y: 0.005
      z: -0.17
    }
    rotation {
      x: -178.31532
    }
    capsule {
      radius: 0.06
      length: 0.46014702
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.751751
}
bodies {
  name: "right_shin0"
  colliders {
    position {
      z: -0.15
    }
    rotation {
      x: -180.0
    }
    capsule {
      radius: 0.049
      length: 0.398
      end: -1
    }
  }
  colliders {
    position {
      z: -0.35
    }
    capsule {
      radius: 0.075
      length: 0.15
      end: 1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5228419
}
bodies {
  name: "left_thigh0"
  colliders {
    position {
      y: -0.005
      z: -0.17
    }
    rotation {
      x: 178.31532
    }
    capsule {
      radius: 0.06
      length: 0.46014702
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.751751
}
bodies {
  name: "left_shin0"
  colliders {
    position {
      z: -0.15
    }
    rotation {
      x: -180.0
    }
    capsule {
      radius: 0.049
      length: 0.398
      end: -1
    }
  }
  colliders {
    position {
      z: -0.35
    }
    capsule {
      radius: 0.075
      length: 0.15
      end: 1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5228419
}
bodies {
  name: "right_upper_arm0"
  colliders {
    position {
      x: 0.08
      y: -0.08
      z: -0.08
    }
    rotation {
      x: 135.0
      y: 35.26439
      z: -75.0
    }
    capsule {
      radius: 0.04
      length: 0.35712814
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.6610805
}
bodies {
  name: "right_lower_arm0"
  colliders {
    position {
      x: 0.09
      y: 0.09
      z: 0.09
    }
    rotation {
      x: -45.0
      y: 35.26439
      z: 15.0
    }
    capsule {
      radius: 0.031
      length: 0.33912814
    }
  }
  colliders {
    position {
      x: 0.18
      y: 0.18
      z: 0.18
    }
    capsule {
      radius: 0.04
      length: 0.08
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.2295402
}
bodies {
  name: "left_upper_arm0"
  colliders {
    position {
      x: 0.08
      y: 0.08
      z: -0.08
    }
    rotation {
      x: -135.0
      y: 35.26439
      z: 75.0
    }
    capsule {
      radius: 0.04
      length: 0.35712814
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.6610805
}
bodies {
  name: "left_lower_arm0"
  colliders {
    position {
      x: 0.09
      y: -0.09
      z: 0.09
    }
    rotation {
      x: 45.0
      y: 35.26439
      z: -15.0
    }
    capsule {
      radius: 0.031
      length: 0.33912814
    }
  }
  colliders {
    position {
      x: 0.18
      y: -0.18
      z: 0.18
    }
    capsule {
      radius: 0.04
      length: 0.08
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.2295402
}
joints {
  name: "abdomen_z0"
  stiffness: 15000.0
  parent: "torso0"
  child: "lwaist0"
  parent_offset {
    x: -0.01
    z: -0.195
  }
  child_offset {
    z: 0.065
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  angle_limit {
    min: -75.0
    max: 30.0
  }
}
joints {
  name: "abdomen_x0"
  stiffness: 15000.0
  parent: "lwaist0"
  child: "pelvis0"
  parent_offset {
    z: -0.065
  }
  child_offset {
    z: 0.1
  }
  rotation {
    x: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -35.0
    max: 35.0
  }
}
joints {
  name: "right_hip_x0"
  stiffness: 8000.0
  parent: "pelvis0"
  child: "right_thigh0"
  parent_offset {
    y: -0.1
    z: -0.04
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  limit_strength: 2000.0
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -30.0
    max: 70.0
  }
}
joints {
  name: "right_knee0"
  stiffness: 15000.0
  parent: "right_thigh0"
  child: "right_shin0"
  parent_offset {
    y: 0.01
    z: -0.383
  }
  child_offset {
    z: 0.02
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -160.0
    max: -2.0
  }
}
joints {
  name: "left_hip_x0"
  stiffness: 8000.0
  parent: "pelvis0"
  child: "left_thigh0"
  parent_offset {
    y: 0.1
    z: -0.04
  }
  child_offset {
  }
  angular_damping: 20.0
  limit_strength: 2000.0
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -30.0
    max: 70.0
  }
}
joints {
  name: "left_knee0"
  stiffness: 15000.0
  parent: "left_thigh0"
  child: "left_shin0"
  parent_offset {
    y: -0.01
    z: -0.383
  }
  child_offset {
    z: 0.02
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -160.0
    max: -2.0
  }
}
joints {
  name: "right_shoulder0"
  stiffness: 15000.0
  parent: "torso0"
  child: "right_upper_arm0"
  parent_offset {
    y: -0.17
    z: 0.06
  }
  child_offset {
  }
  rotation {
    x: 135.0
    y: 35.26439
  }
  angular_damping: 20.0
  angle_limit {
    min: -85.0
    max: 60.0
  }
  angle_limit {
    min: -85.0
    max: 60.0
  }
}
joints {
  name: "right_elbow0"
  stiffness: 15000.0
  parent: "right_upper_arm0"
  child: "right_lower_arm0"
  parent_offset {
    x: 0.18
    y: -0.18
    z: -0.18
  }
  child_offset {
  }
  rotation {
    x: 135.0
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -90.0
    max: 50.0
  }
}
joints {
  name: "left_shoulder0"
  stiffness: 15000.0
  parent: "torso0"
  child: "left_upper_arm0"
  parent_offset {
    y: 0.17
    z: 0.06
  }
  child_offset {
  }
  rotation {
    x: 45.0
    y: -35.26439
  }
  angular_damping: 20.0
  angle_limit {
    min: -60.0
    max: 85.0
  }
  angle_limit {
    min: -60.0
    max: 85.0
  }
}
joints {
  name: "left_elbow0"
  stiffness: 15000.0
  parent: "left_upper_arm0"
  child: "left_lower_arm0"
  parent_offset {
    x: 0.18
    y: 0.18
    z: -0.18
  }
  child_offset {
  }
  rotation {
    x: 45.0
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -90.0
    max: 50.0
  }
}
actuators {
  name: "abdomen_z0"
  joint: "abdomen_z0"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "abdomen_x0"
  joint: "abdomen_x0"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "right_hip_x0"
  joint: "right_hip_x0"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "right_knee0"
  joint: "right_knee0"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "left_hip_x0"
  joint: "left_hip_x0"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "left_knee0"
  joint: "left_knee0"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "right_shoulder0"
  joint: "right_shoulder0"
  strength: 75.0
  torque {
  }
}
actuators {
  name: "right_elbow0"
  joint: "right_elbow0"
  strength: 75.0
  torque {
  }
}
actuators {
  name: "left_shoulder0"
  joint: "left_shoulder0"
  strength: 75.0
  torque {
  }
}
actuators {
  name: "left_elbow0"
  joint: "left_elbow0"
  strength: 75.0
  torque {
  }
}
collide_include {
  first: "floor"
  second: "left_shin0"
}
collide_include {
  first: "floor"
  second: "right_shin0"
}
""" 

_HUMANOID1_CONFIG = """
bodies {
  name: "torso1"
  colliders {
    position {
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.07
      length: 0.28
    }
  }
  colliders {
    position {
      z: 0.19
    }
    capsule {
      radius: 0.09
      length: 0.18
    }
  }
  colliders {
    position {
      x: -0.01
      z: -0.12
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.06
      length: 0.24
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 8.907463
}
bodies {
  name: "lwaist1"
  colliders {
    position {
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.06
      length: 0.24
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.2619467
}
bodies {
  name: "pelvis1"
  colliders {
    position {
      x: -0.02
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.09
      length: 0.32
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 6.6161942
}
bodies {
  name: "right_thigh1"
  colliders {
    position {
      y: 0.005
      z: -0.17
    }
    rotation {
      x: -178.31532
    }
    capsule {
      radius: 0.06
      length: 0.46014702
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.751751
}
bodies {
  name: "right_shin1"
  colliders {
    position {
      z: -0.15
    }
    rotation {
      x: -180.0
    }
    capsule {
      radius: 0.049
      length: 0.398
      end: -1
    }
  }
  colliders {
    position {
      z: -0.35
    }
    capsule {
      radius: 0.075
      length: 0.15
      end: 1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5228419
}
bodies {
  name: "left_thigh1"
  colliders {
    position {
      y: -0.005
      z: -0.17
    }
    rotation {
      x: 178.31532
    }
    capsule {
      radius: 0.06
      length: 0.46014702
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.751751
}
bodies {
  name: "left_shin1"
  colliders {
    position {
      z: -0.15
    }
    rotation {
      x: -180.0
    }
    capsule {
      radius: 0.049
      length: 0.398
      end: -1
    }
  }
  colliders {
    position {
      z: -0.35
    }
    capsule {
      radius: 0.075
      length: 0.15
      end: 1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5228419
}
bodies {
  name: "right_upper_arm1"
  colliders {
    position {
      x: 0.08
      y: -0.08
      z: -0.08
    }
    rotation {
      x: 135.0
      y: 35.26439
      z: -75.0
    }
    capsule {
      radius: 0.04
      length: 0.35712814
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.6610805
}
bodies {
  name: "right_lower_arm1"
  colliders {
    position {
      x: 0.09
      y: 0.09
      z: 0.09
    }
    rotation {
      x: -45.0
      y: 35.26439
      z: 15.0
    }
    capsule {
      radius: 0.031
      length: 0.33912814
    }
  }
  colliders {
    position {
      x: 0.18
      y: 0.18
      z: 0.18
    }
    capsule {
      radius: 0.04
      length: 0.08
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.2295402
}
bodies {
  name: "left_upper_arm1"
  colliders {
    position {
      x: 0.08
      y: 0.08
      z: -0.08
    }
    rotation {
      x: -135.0
      y: 35.26439
      z: 75.0
    }
    capsule {
      radius: 0.04
      length: 0.35712814
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.6610805
}
bodies {
  name: "left_lower_arm1"
  colliders {
    position {
      x: 0.09
      y: -0.09
      z: 0.09
    }
    rotation {
      x: 45.0
      y: 35.26439
      z: -15.0
    }
    capsule {
      radius: 0.031
      length: 0.33912814
    }
  }
  colliders {
    position {
      x: 0.18
      y: -0.18
      z: 0.18
    }
    capsule {
      radius: 0.04
      length: 0.08
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.2295402
}
joints {
  name: "abdomen_z1"
  stiffness: 15000.0
  parent: "torso1"
  child: "lwaist1"
  parent_offset {
    x: -0.01
    z: -0.195
  }
  child_offset {
    z: 0.065
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  angle_limit {
    min: -75.0
    max: 30.0
  }
}
joints {
  name: "abdomen_x1"
  stiffness: 15000.0
  parent: "lwaist1"
  child: "pelvis1"
  parent_offset {
    z: -0.065
  }
  child_offset {
    z: 0.1
  }
  rotation {
    x: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -35.0
    max: 35.0
  }
}
joints {
  name: "right_hip_x1"
  stiffness: 8000.0
  parent: "pelvis1"
  child: "right_thigh1"
  parent_offset {
    y: -0.1
    z: -0.04
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  limit_strength: 2000.0
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -30.0
    max: 70.0
  }
}
joints {
  name: "right_knee1"
  stiffness: 15000.0
  parent: "right_thigh1"
  child: "right_shin1"
  parent_offset {
    y: 0.01
    z: -0.383
  }
  child_offset {
    z: 0.02
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -160.0
    max: -2.0
  }
}
joints {
  name: "left_hip_x1"
  stiffness: 8000.0
  parent: "pelvis1"
  child: "left_thigh1"
  parent_offset {
    y: 0.1
    z: -0.04
  }
  child_offset {
  }
  angular_damping: 20.0
  limit_strength: 2000.0
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -30.0
    max: 70.0
  }
}
joints {
  name: "left_knee1"
  stiffness: 15000.0
  parent: "left_thigh1"
  child: "left_shin1"
  parent_offset {
    y: -0.01
    z: -0.383
  }
  child_offset {
    z: 0.02
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -160.0
    max: -2.0
  }
}
joints {
  name: "right_shoulder1"
  stiffness: 15000.0
  parent: "torso1"
  child: "right_upper_arm1"
  parent_offset {
    y: -0.17
    z: 0.06
  }
  child_offset {
  }
  rotation {
    x: 135.0
    y: 35.26439
  }
  angular_damping: 20.0
  angle_limit {
    min: -85.0
    max: 60.0
  }
  angle_limit {
    min: -85.0
    max: 60.0
  }
}
joints {
  name: "right_elbow1"
  stiffness: 15000.0
  parent: "right_upper_arm1"
  child: "right_lower_arm1"
  parent_offset {
    x: 0.18
    y: -0.18
    z: -0.18
  }
  child_offset {
  }
  rotation {
    x: 135.0
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -90.0
    max: 50.0
  }
}
joints {
  name: "left_shoulder1"
  stiffness: 15000.0
  parent: "torso1"
  child: "left_upper_arm1"
  parent_offset {
    y: 0.17
    z: 0.06
  }
  child_offset {
  }
  rotation {
    x: 45.0
    y: -35.26439
  }
  angular_damping: 20.0
  angle_limit {
    min: -60.0
    max: 85.0
  }
  angle_limit {
    min: -60.0
    max: 85.0
  }
}
joints {
  name: "left_elbow1"
  stiffness: 15000.0
  parent: "left_upper_arm1"
  child: "left_lower_arm1"
  parent_offset {
    x: 0.18
    y: 0.18
    z: -0.18
  }
  child_offset {
  }
  rotation {
    x: 45.0
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -90.0
    max: 50.0
  }
}
actuators {
  name: "abdomen_z1"
  joint: "abdomen_z1"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "abdomen_x1"
  joint: "abdomen_x1"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "right_hip_x1"
  joint: "right_hip_x1"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "right_knee1"
  joint: "right_knee1"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "left_hip_x1"
  joint: "left_hip_x1"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "left_knee1"
  joint: "left_knee1"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "right_shoulder1"
  joint: "right_shoulder1"
  strength: 75.0
  torque {
  }
}
actuators {
  name: "right_elbow1"
  joint: "right_elbow1"
  strength: 75.0
  torque {
  }
}
actuators {
  name: "left_shoulder1"
  joint: "left_shoulder1"
  strength: 75.0
  torque {
  }
}
actuators {
  name: "left_elbow1"
  joint: "left_elbow1"
  strength: 75.0
  torque {
  }
}
collide_include {
  first: "floor"
  second: "left_shin1"
}
collide_include {
  first: "floor"
  second: "right_shin1"
}
"""

_ENV_CONFIG = """
bodies {
  name: "floor"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen { all: true }
}
friction: 1.0
gravity {
  z: -9.81
}
angular_damping: -0.05
baumgarte_erp: 0.1
dt: 0.015
substeps: 8
"""

_SYSTEM_CONFIG = _HUMANOID0_CONFIG + _HUMANOID1_CONFIG + _ENV_CONFIG