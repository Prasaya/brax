import brax
from brax import jumpy as jp
from brax.envs import env
from jax.experimental.host_callback import call
from google.protobuf import text_format
import jax


class Humanoid(env.Env):
    print("Using new humanoid 2")

    # CHANGED: health_z_range from (0.8, 2.1) to (1.2, 2.1)
    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.1, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        legacy_spring=False,
        **kwargs
    ):
        config = _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)
        # print(self.sys.__dict__)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.torso_idx = self.sys.body.index["torso"]
        self.target_idx = self.sys.body.index["Target"]
        print("Target index : ", self.sys.body.index["Target"])
        # print("Torso index : ", self.torso_idx)
        # print( "QP position of torso index : ", env.State.qp.pos[self.torso_idx] )
        # jax.experimental.host_callback.id_print(f"QP position of torso index : {env.State.qp.pos[self.torso_idx]}")
        print("System body consists  : ", self.sys.body)

    def reset(self, rng: jp.ndarray) -> env.State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jp.random_split(rng, 3)

        qpos = self.sys.default_angle() + self._noise(rng1)
        qvel = self._noise(rng2)

        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        obs = self._get_obs(qp, self.sys.info(qp), jp.zeros(self.action_size))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "target_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }
        pos = jp.index_update(
            qp.pos, self.target_idx, jax.numpy.array([10.0, 10.0, 1.5])
        )
        qp = qp.replace(pos=pos)
        return env.State(qp, obs, reward, done, metrics)

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)

        # forward reward for moving forward
        com_before = self._center_of_mass(state.qp)
        com_after = self._center_of_mass(qp)
        velocity = (com_after - com_before) / self.sys.config.dt
        in_between = jp.where(com_after[0] < 41.0, 1.0, 0.0)
        in_between = jp.where(com_after[0] > 31.0, in_between, -1.0)
        in_between = jp.where(com_after[0] < 21.0, 1.0, in_between)
        in_between = jp.where(com_after[0] > 11.0, in_between, -1.0)
        forward_reward = (
            self._forward_reward_weight * velocity[0]
            + self._forward_reward_weight * velocity[1] * in_between * 0.35
        )
        # forward_reward = 0.

        # small reward for torso moving towards target
        torso_delta = com_after - com_before
        target_rel = qp.pos[self.target_idx] - com_after
        target_dist = jp.norm(target_rel)
        target_dir = target_rel / (1e-6 + target_dist)
        # target_reward = self._forward_reward_weight * jp.dot(velocity, target_dir)
        target_reward = 0.0

        # jax.experimental.host_callback.id_print(qp.pos[self.target_idx])

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
        is_healthy = jp.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(qp, info, action)
        reward = forward_reward + healthy_reward - ctrl_cost + target_reward
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            forward_reward=forward_reward,
            target_reward=target_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jp.norm(com_after),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
        )

        return state.replace(qp=qp, obs=obs, reward=reward, done=done)

    def _get_obs(self, qp: brax.QP, info: brax.Info, action: jp.ndarray) -> jp.ndarray:
        """Observe humanoid body position, velocities, and angles."""
        angle_vels = [j.angle_vel(qp) for j in self.sys.joints]

        # qpos: position and orientation of the torso and the joint angles.
        joint_angles = [angle for angle, _ in angle_vels]
        if self._exclude_current_positions_from_observation:
            qpos = [qp.pos[0, 2:], qp.rot[0]] + joint_angles
        else:
            qpos = [qp.pos[0], qp.rot[0]] + joint_angles

        # qvel: velocity of the torso and the joint angle velocities.
        joint_velocities = [vel for _, vel in angle_vels]
        qvel = [qp.vel[0], qp.ang[0]] + joint_velocities

        # center of mass obs:
        com = self._center_of_mass(qp)
        mass_sum = jp.sum(self.sys.body.mass[:-1])

        def com_vals(body, qp):
            d = qp.pos - com
            com_inr = body.mass * jp.eye(3) * jp.norm(d) ** 2
            com_inr += jp.diag(body.inertia) - jp.outer(d, d)
            com_vel = body.mass * qp.vel / mass_sum
            com_ang = jp.cross(d, qp.vel) / (1e-7 + jp.norm(d) ** 2)

            return com_inr, com_vel, com_ang

        com_inr, com_vel, com_ang = jp.vmap(com_vals)(self.sys.body, qp)
        cinert = [com_inr[:12].ravel()]
        cvel = [com_vel[:12].ravel(), com_ang[:12].ravel()]

        # actuator forces
        qfrc_actuator = []
        for act in self.sys.actuators:
            torque = jp.take(action, act.act_index)
            torque = torque.reshape(torque.shape[:-2] + (-1,))
            torque *= jp.repeat(act.strength, act.act_index.shape[-1])
            qfrc_actuator.append(torque)

        return jp.concatenate(qpos + qvel + cinert + cvel + qfrc_actuator)

    def _center_of_mass(self, qp):
        mass, pos = self.sys.body.mass[:12], qp.pos[:12]
        return jp.sum(jp.vmap(jp.multiply)(mass, pos), axis=0) / jp.sum(mass)

    def _noise(self, rng):
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        return jp.random_uniform(rng, (self.sys.num_joint_dof,), low, hi)


_SYSTEM_CONFIG = """
  bodies {
  name: "worldbody"
  colliders {
    position {
    }
    sphere {
      radius: 0.01
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.00418879
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
  }
}
bodies {
  name: "root"
  colliders {
    position {
      y: -0.05
    }
    rotation {
      x: -0.0
      y: -90.0
    }
    capsule {
      radius: 0.1
      length: 0.34
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 8.58702
}
bodies {
  name: "lhipjoint"
  colliders {
    position {
      x: 0.0509685
      y: -0.0459037
      z: 0.024723
    }
    rotation {
      x: -118.30627
      y: -42.71965
      z: -66.43646
    }
    capsule {
      radius: 0.008
      length: 0.05974726
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.010940569
}
bodies {
  name: "lfemur"
  colliders {
    position {
      y: -0.115473
    }
    rotation {
      x: -90.0
      z: -20.000038
    }
    capsule {
      radius: 0.085
      length: 0.336
    }
  }
  colliders {
    position {
      y: -0.202473
    }
    rotation {
      x: -90.0
      z: -20.000038
    }
    capsule {
      radius: 0.07
      length: 0.504452
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 13.387368
}
bodies {
  name: "ltibia"
  colliders {
    position {
      y: -0.202846
    }
    rotation {
      x: -90.0
      z: -20.000038
    }
    capsule {
      radius: 0.04
      length: 0.4451228
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.10339
}
bodies {
  name: "lfoot"
  colliders {
    position {
      x: -0.026999997
      y: -0.05
      z: -0.0113878
    }
    rotation {
      x: -79.69992
      y: 3.7234762
      z: -0.3357849
    }
    capsule {
      radius: 0.025
      length: 0.21
    }
  }
  colliders {
    position {
      x: 0.027000003
      y: -0.05
      z: -0.0113878
    }
    rotation {
      x: -79.69994
      y: -3.72354
      z: -39.664368
    }
    capsule {
      radius: 0.025
      length: 0.21
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.7592182
}
bodies {
  name: "ltoes"
  colliders {
    position {
      y: -0.01
      z: -0.01
    }
    sphere {
      radius: 0.025
    }
  }
  colliders {
    position {
      x: 0.03
      y: -0.01
      z: -0.01
    }
    sphere {
      radius: 0.025
    }
  }
  colliders {
    position {
      x: -0.03
      y: -0.01
      z: -0.01
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.19634955
}
bodies {
  name: "rhipjoint"
  colliders {
    position {
      x: -0.0509685
      y: -0.0459037
      z: 0.024723
    }
    rotation {
      x: -118.306305
      y: 44.350132
      z: 68.624
    }
    capsule {
      radius: 0.008
      length: 0.05974726
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.010940569
}
bodies {
  name: "rfemur"
  colliders {
    position {
      y: -0.115473
    }
    rotation {
      x: -90.0
      z: 20.000038
    }
    capsule {
      radius: 0.085
      length: 0.336
    }
  }
  colliders {
    position {
      y: -0.202473
    }
    rotation {
      x: -90.0
      z: 20.000038
    }
    capsule {
      radius: 0.07
      length: 0.504452
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 13.387368
}
bodies {
  name: "rtibia"
  colliders {
    position {
      y: -0.202846
    }
    rotation {
      x: -90.0
      z: 20.000038
    }
    capsule {
      radius: 0.04
      length: 0.4451228
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.10339
}
bodies {
  name: "rfoot"
  colliders {
    position {
      x: -0.026999997
      y: -0.05
      z: -0.0113878
    }
    rotation {
      x: -78.624016
      y: 4.106348
      z: 39.59082
    }
    capsule {
      radius: 0.025
      length: 0.21
    }
  }
  colliders {
    position {
      x: 0.027000003
      y: -0.05
      z: -0.0113878
    }
    rotation {
      x: -78.624
      y: -4.106305
      z: 0.40916047
    }
    capsule {
      radius: 0.025
      length: 0.21
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.7592182
}
bodies {
  name: "rtoes"
  colliders {
    position {
      y: -0.01
      z: -0.01
    }
    sphere {
      radius: 0.025
    }
  }
  colliders {
    position {
      x: 0.03
      y: -0.01
      z: -0.01
    }
    sphere {
      radius: 0.025
    }
  }
  colliders {
    position {
      x: -0.03
      y: -0.01
      z: -0.01
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.19634955
}
bodies {
  name: "lowerback"
  colliders {
    position {
      x: 0.00282931
      y: 0.0566065
      z: 0.01
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.085
      length: 0.26080033
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.633427
}
bodies {
  name: "upperback"
  colliders {
    position {
      x: 0.000256264
      y: 0.0567802
      z: 0.02
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.09
      length: 0.27085233
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 5.365538
}
bodies {
  name: "thorax"
  colliders {
    position {
      y: 0.0569725
      z: 0.02
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.095
      length: 0.3040412
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 6.82476
}
bodies {
  name: "lowerneck"
  colliders {
    position {
      x: -0.00165071
      y: 0.0452401
      z: 0.00534359
    }
    rotation {
      x: 90.0
      z: -0.0
    }
    capsule {
      radius: 0.075
      length: 0.1955845
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.5726905
}
bodies {
  name: "upperneck"
  colliders {
    position {
      x: 0.000500875
      y: 0.0449956
      z: -0.00224644
    }
    rotation {
      x: 90.0
      z: -0.0
    }
    capsule {
      radius: 0.05
      length: 0.1450544
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.87745523
}
bodies {
  name: "head"
  colliders {
    position {
      x: 0.000341465
      y: 0.048184
      z: 0.025
    }
    rotation {
      x: 90.0
      z: -0.0
    }
    capsule {
      radius: 0.095
      length: 0.238208
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.9581995
}
bodies {
  name: "lclavicle"
  colliders {
    position {
      x: 0.0918817
      y: 0.0382636
      z: 0.00535704
    }
    rotation {
      x: 97.918434
      y: -67.19682
      z: 74.69879
    }
    capsule {
      radius: 0.075
      length: 0.27
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.8877208
}
bodies {
  name: "lhumerus"
  colliders {
    position {
      y: -0.138421
    }
    rotation {
      x: -90.0
      z: 59.99998
    }
    capsule {
      radius: 0.042
      length: 0.3331578
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.6911142
}
bodies {
  name: "lradius"
  colliders {
    position {
      y: -0.0907679
    }
    rotation {
      x: -90.0
      z: 59.99998
    }
    capsule {
      radius: 0.03
      length: 0.22338222
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.5750497
}
bodies {
  name: "lwrist"
  colliders {
    position {
      y: -0.03
    }
    rotation {
      x: -90.0
      y: -8.102844e-15
      z: -180.0
    }
    capsule {
      radius: 0.02
      length: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.10890855
}
bodies {
  name: "lhand"
  colliders {
    position {
      y: -0.016752
    }
    rotation {
      x: -90.0
      y: -0.0
      z: -180.0
    }
    capsule {
      radius: 0.02
      length: 0.13
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.14660765
}
bodies {
  name: "lfingers"
  colliders {
    position {
      x: -0.024
      y: -0.025
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.0065
      length: 0.093
    }
  }
  colliders {
    position {
      x: -0.008
      y: -0.03
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.0065
      length: 0.093
    }
  }
  colliders {
    position {
      x: 0.008
      y: -0.03
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.006
      length: 0.092
    }
  }
  colliders {
    position {
      x: 0.024
      y: -0.025
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.0055
      length: 0.091
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.041789986
}
bodies {
  name: "lthumb"
  colliders {
    position {
      y: -0.03
    }
    rotation {
      x: -90.0
      y: -0.0
      z: -180.0
    }
    capsule {
      radius: 0.008
      length: 0.076
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.014208376
}
bodies {
  name: "rclavicle"
  colliders {
    position {
      x: -0.0918817
      y: 0.0382636
      z: 0.00535704
    }
    rotation {
      x: 97.918434
      y: 67.19682
      z: -74.69879
    }
    capsule {
      radius: 0.075
      length: 0.27
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.8877208
}
bodies {
  name: "rhumerus"
  colliders {
    position {
      y: -0.138421
    }
    rotation {
      x: -90.0
      z: -59.99998
    }
    capsule {
      radius: 0.042
      length: 0.3331578
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.6911142
}
bodies {
  name: "rradius"
  colliders {
    position {
      y: -0.0907679
    }
    rotation {
      x: -89.99996
      y: -0.0012363585
      z: -59.998886
    }
    capsule {
      radius: 0.03
      length: 0.22338222
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.5750497
}
bodies {
  name: "rwrist"
  colliders {
    position {
      y: -0.03
    }
    rotation {
      x: -90.0
      z: -180.0
    }
    capsule {
      radius: 0.02
      length: 0.1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.10890855
}
bodies {
  name: "rhand"
  colliders {
    position {
      y: -0.016752
    }
    rotation {
      x: -90.0
      z: -180.0
    }
    capsule {
      radius: 0.02
      length: 0.13
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.14660765
}
bodies {
  name: "rfingers"
  colliders {
    position {
      x: 0.024
      y: -0.025
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.0065
      length: 0.093
    }
  }
  colliders {
    position {
      x: 0.008
      y: -0.03
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.0065
      length: 0.093
    }
  }
  colliders {
    position {
      x: -0.008
      y: -0.03
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.006
      length: 0.092
    }
  }
  colliders {
    position {
      x: -0.024
      y: -0.025
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.0055
      length: 0.091
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.041789986
}
bodies {
  name: "rthumb"
  colliders {
    position {
      y: -0.03
    }
    rotation {
      x: -90.0
      y: -1.0947753e-10
      z: 180.0
    }
    capsule {
      radius: 0.008
      length: 0.076
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.014208376
}
joints {
  name: "$worldbody.root"
  stiffness: 5000.0
  parent: "worldbody"
  child: "root"
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "$root.lhipjoint"
  stiffness: 5000.0
  parent: "root"
  child: "lhipjoint"
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lfemurrz"
  stiffness: 1.0
  parent: "lhipjoint"
  child: "lfemur"
  parent_offset {
    x: 0.101937
    y: -0.0918074
    z: 0.0494461
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -1.0472
    max: 1.22173
  }
  reference_rotation {
    x: -0.0
    z: 19.999975
  }
}
joints {
  name: "lfemurry"
  stiffness: 1.0
  parent: "lhipjoint"
  child: "lfemur"
  parent_offset {
    x: 0.101937
    y: -0.0918074
    z: 0.0494461
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -1.22173
    max: 1.22173
  }
  reference_rotation {
    x: -0.0
    z: 19.999975
  }
}
joints {
  name: "lfemurrx"
  stiffness: 1.0
  parent: "lhipjoint"
  child: "lfemur"
  parent_offset {
    x: 0.101937
    y: -0.0918074
    z: 0.0494461
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -2.79253
    max: 0.349066
  }
  reference_rotation {
    x: -0.0
    z: 19.999975
  }
}
joints {
  name: "ltibiarx"
  stiffness: 1.0
  parent: "lfemur"
  child: "ltibia"
  parent_offset {
    y: -0.404945
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: 0.01
    max: 2.96706
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lfootrz"
  stiffness: 1.0
  parent: "ltibia"
  child: "lfoot"
  parent_offset {
    y: -0.415693
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -1.22173
    max: 0.349066
  }
  reference_rotation {
    x: -90.0
  }
}
joints {
  name: "lfootrx"
  stiffness: 1.0
  parent: "ltibia"
  child: "lfoot"
  parent_offset {
    y: -0.415693
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -0.785398
    max: 0.8
  }
  reference_rotation {
    x: -90.0
  }
}
joints {
  name: "ltoesrx"
  stiffness: 1.0
  parent: "lfoot"
  child: "ltoes"
  parent_offset {
    y: -0.156372
    z: -0.0227756
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -1.5708
    max: 0.349066
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "$root.rhipjoint"
  stiffness: 5000.0
  parent: "root"
  child: "rhipjoint"
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "rfemurrz"
  stiffness: 1.0
  parent: "rhipjoint"
  child: "rfemur"
  parent_offset {
    x: -0.101937
    y: -0.0918074
    z: 0.0494461
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -1.22173
    max: 1.0472
  }
  reference_rotation {
    z: -19.999975
  }
}
joints {
  name: "rfemurry"
  stiffness: 1.0
  parent: "rhipjoint"
  child: "rfemur"
  parent_offset {
    x: -0.101937
    y: -0.0918074
    z: 0.0494461
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -1.22173
    max: 1.22173
  }
  reference_rotation {
    z: -19.999975
  }
}
joints {
  name: "rfemurrx"
  stiffness: 1.0
  parent: "rhipjoint"
  child: "rfemur"
  parent_offset {
    x: -0.101937
    y: -0.0918074
    z: 0.0494461
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -2.79253
    max: 0.349066
  }
  reference_rotation {
    z: -19.999975
  }
}
joints {
  name: "rtibiarx"
  stiffness: 1.0
  parent: "rfemur"
  child: "rtibia"
  parent_offset {
    y: -0.404945
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: 0.01
    max: 2.96706
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "rfootrz"
  stiffness: 1.0
  parent: "rtibia"
  child: "rfoot"
  parent_offset {
    y: -0.415693
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.349066
    max: 1.22173
  }
  reference_rotation {
    x: -90.0
  }
}
joints {
  name: "rfootrx"
  stiffness: 1.0
  parent: "rtibia"
  child: "rfoot"
  parent_offset {
    y: -0.415693
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -0.785398
    max: 0.8
  }
  reference_rotation {
    x: -90.0
  }
}
joints {
  name: "rtoesrx"
  stiffness: 1.0
  parent: "rfoot"
  child: "rtoes"
  parent_offset {
    y: -0.156372
    z: -0.0227756
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -1.5708
    max: 0.349066
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lowerbackrz"
  stiffness: 1.0
  parent: "root"
  child: "lowerback"
  parent_offset {
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lowerbackry"
  stiffness: 1.0
  parent: "root"
  child: "lowerback"
  parent_offset {
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lowerbackrx"
  stiffness: 1.0
  parent: "root"
  child: "lowerback"
  parent_offset {
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -0.349066
    max: 0.785398
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "upperbackrz"
  stiffness: 1.0
  parent: "lowerback"
  child: "upperback"
  parent_offset {
    x: 0.000565862
    y: 0.113213
    z: -0.00805298
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "upperbackry"
  stiffness: 1.0
  parent: "lowerback"
  child: "upperback"
  parent_offset {
    x: 0.000565862
    y: 0.113213
    z: -0.00805298
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "upperbackrx"
  stiffness: 1.0
  parent: "lowerback"
  child: "upperback"
  parent_offset {
    x: 0.000565862
    y: 0.113213
    z: -0.00805298
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -0.349066
    max: 0.785398
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "thoraxrz"
  stiffness: 1.0
  parent: "upperback"
  child: "thorax"
  parent_offset {
    x: 0.000512528
    y: 0.11356
    z: 0.000936821
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "thoraxry"
  stiffness: 1.0
  parent: "upperback"
  child: "thorax"
  parent_offset {
    x: 0.000512528
    y: 0.11356
    z: 0.000936821
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "thoraxrx"
  stiffness: 1.0
  parent: "upperback"
  child: "thorax"
  parent_offset {
    x: 0.000512528
    y: 0.11356
    z: 0.000936821
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -0.349066
    max: 0.785398
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lowerneckrz"
  stiffness: 1.0
  parent: "thorax"
  child: "lowerneck"
  parent_offset {
    y: 0.113945
    z: 0.00468037
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lowerneckry"
  stiffness: 1.0
  parent: "thorax"
  child: "lowerneck"
  parent_offset {
    y: 0.113945
    z: 0.00468037
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lowerneckrx"
  stiffness: 1.0
  parent: "thorax"
  child: "lowerneck"
  parent_offset {
    y: 0.113945
    z: 0.00468037
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -0.349066
    max: 0.785398
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "upperneckrz"
  stiffness: 1.0
  parent: "lowerneck"
  child: "upperneck"
  parent_offset {
    x: -0.00330143
    y: 0.0904801
    z: 0.0106872
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "upperneckry"
  stiffness: 1.0
  parent: "lowerneck"
  child: "upperneck"
  parent_offset {
    x: -0.00330143
    y: 0.0904801
    z: 0.0106872
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "upperneckrx"
  stiffness: 1.0
  parent: "lowerneck"
  child: "upperneck"
  parent_offset {
    x: -0.00330143
    y: 0.0904801
    z: 0.0106872
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -0.349066
    max: 0.785398
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "headrz"
  stiffness: 1.0
  parent: "upperneck"
  child: "head"
  parent_offset {
    x: 0.00100175
    y: 0.13
    z: -0.00449288
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "headry"
  stiffness: 1.0
  parent: "upperneck"
  child: "head"
  parent_offset {
    x: 0.00100175
    y: 0.13
    z: -0.00449288
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -0.523599
    max: 0.523599
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "headrx"
  stiffness: 1.0
  parent: "upperneck"
  child: "head"
  parent_offset {
    x: 0.00100175
    y: 0.13
    z: -0.00449288
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -0.349066
    max: 0.785398
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lclaviclerz"
  stiffness: 1.0
  parent: "thorax"
  child: "lclavicle"
  parent_offset {
    y: 0.113945
    z: 0.00468037
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    max: 0.349066
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lclaviclery"
  stiffness: 1.0
  parent: "thorax"
  child: "lclavicle"
  parent_offset {
    y: 0.113945
    z: 0.00468037
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -0.349066
    max: 0.174533
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lhumerusrz"
  stiffness: 1.0
  parent: "lclavicle"
  child: "lhumerus"
  parent_offset {
    x: 0.18
    y: 0.09
    z: 0.0107141
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -1.1
    max: 1.5708
  }
  reference_rotation {
    x: 149.99997
    y: 3.1805547e-15
    z: 90.0
  }
}
joints {
  name: "lhumerusry"
  stiffness: 1.0
  parent: "lclavicle"
  child: "lhumerus"
  parent_offset {
    x: 0.18
    y: 0.09
    z: 0.0107141
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -1.5708
    max: 1.5708
  }
  reference_rotation {
    x: 149.99997
    y: 3.1805547e-15
    z: 90.0
  }
}
joints {
  name: "lhumerusrx"
  stiffness: 1.0
  parent: "lclavicle"
  child: "lhumerus"
  parent_offset {
    x: 0.18
    y: 0.09
    z: 0.0107141
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -1.0472
    max: 1.5708
  }
  reference_rotation {
    x: 149.99997
    y: 3.1805547e-15
    z: 90.0
  }
}
joints {
  name: "lradiusrx"
  stiffness: 1.0
  parent: "lhumerus"
  child: "lradius"
  parent_offset {
    y: -0.276843
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -0.174533
    max: 2.96706
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lwristry"
  stiffness: 1.0
  parent: "lradius"
  child: "lwrist"
  parent_offset {
    y: -0.181536
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    max: 3.14159
  }
  reference_rotation {
    x: -180.0
    y: -60.000023
    z: -180.0
  }
}
joints {
  name: "lhandrz"
  stiffness: 1.0
  parent: "lwrist"
  child: "lhand"
  parent_offset {
    y: -0.0907676
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.785398
    max: 0.785398
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lhandrx"
  stiffness: 1.0
  parent: "lwrist"
  child: "lhand"
  parent_offset {
    y: -0.0907676
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -1.5708
    max: 1.5708
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lfingersrx"
  stiffness: 1.0
  parent: "lhand"
  child: "lfingers"
  parent_offset {
    y: -0.06
  }
  child_offset {
    y: 0.015
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    max: 1.5708
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "lthumbrz"
  stiffness: 1.0
  parent: "lhand"
  child: "lthumb"
  parent_offset {
    x: -0.025
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.785398
    max: 0.785398
  }
  reference_rotation {
    z: -44.999935
  }
}
joints {
  name: "lthumbrx"
  stiffness: 1.0
  parent: "lhand"
  child: "lthumb"
  parent_offset {
    x: -0.025
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    max: 1.57
  }
  reference_rotation {
    z: -44.999935
  }
}
joints {
  name: "rclaviclerz"
  stiffness: 1.0
  parent: "thorax"
  child: "rclavicle"
  parent_offset {
    y: 0.113945
    z: 0.00468037
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.349066
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "rclaviclery"
  stiffness: 1.0
  parent: "thorax"
  child: "rclavicle"
  parent_offset {
    y: 0.113945
    z: 0.00468037
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -0.174533
    max: 0.349066
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "rhumerusrz"
  stiffness: 1.0
  parent: "rclavicle"
  child: "rhumerus"
  parent_offset {
    x: -0.18
    y: 0.09
    z: 0.0107141
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -1.1
    max: 1.5708
  }
  reference_rotation {
    x: 149.99997
    y: -3.1805547e-15
    z: -90.0
  }
}
joints {
  name: "rhumerusry"
  stiffness: 1.0
  parent: "rclavicle"
  child: "rhumerus"
  parent_offset {
    x: -0.18
    y: 0.09
    z: 0.0107141
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -1.5708
    max: 1.5708
  }
  reference_rotation {
    x: 149.99997
    y: -3.1805547e-15
    z: -90.0
  }
}
joints {
  name: "rhumerusrx"
  stiffness: 1.0
  parent: "rclavicle"
  child: "rhumerus"
  parent_offset {
    x: -0.18
    y: 0.09
    z: 0.0107141
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -1.0472
    max: 1.5708
  }
  reference_rotation {
    x: 149.99997
    y: -3.1805547e-15
    z: -90.0
  }
}
joints {
  name: "rradiusrx"
  stiffness: 1.0
  parent: "rhumerus"
  child: "rradius"
  parent_offset {
    y: -0.276843
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -0.174533
    max: 2.96706
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "rwristry"
  stiffness: 1.0
  parent: "rradius"
  child: "rwrist"
  parent_offset {
    y: -0.181536
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: 90.0
  }
  angle_limit {
    min: -3.14159
  }
  reference_rotation {
    x: -180.0
    y: 60.000023
    z: -180.0
  }
}
joints {
  name: "rhandrz"
  stiffness: 1.0
  parent: "rwrist"
  child: "rhand"
  parent_offset {
    y: -0.0907676
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.785398
    max: 0.785398
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "rhandrx"
  stiffness: 1.0
  parent: "rwrist"
  child: "rhand"
  parent_offset {
    y: -0.0907676
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    min: -1.5708
    max: 1.5708
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "rfingersrx"
  stiffness: 1.0
  parent: "rhand"
  child: "rfingers"
  parent_offset {
    y: -0.06
  }
  child_offset {
    y: 0.015
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    max: 1.5708
  }
  reference_rotation {
    x: -0.0
    z: -0.0
  }
}
joints {
  name: "rthumbrz"
  stiffness: 1.0
  parent: "rhand"
  child: "rthumb"
  parent_offset {
    x: 0.025
  }
  child_offset {
  }
  rotation {
    x: -0.0
    y: -90.0
  }
  angle_limit {
    min: -0.785398
    max: 0.785398
  }
  reference_rotation {
    x: -0.0
    z: 44.999935
  }
}
joints {
  name: "rthumbrx"
  stiffness: 1.0
  parent: "rhand"
  child: "rthumb"
  parent_offset {
    x: 0.025
  }
  child_offset {
  }
  rotation {
    x: -0.0
    z: -0.0
  }
  angle_limit {
    max: 1.57
  }
  reference_rotation {
    x: -0.0
    z: 44.999935
  }
}
actuators {
  name: "headrx"
  joint: "headrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "headry"
  joint: "headry"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "headrz"
  joint: "headrz"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lclaviclery"
  joint: "lclaviclery"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lclaviclerz"
  joint: "lclaviclerz"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lfemurrx"
  joint: "lfemurrx"
  strength: 120.0
  angle {
  }
}
actuators {
  name: "lfemurry"
  joint: "lfemurry"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "lfemurrz"
  joint: "lfemurrz"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "lfingersrx"
  joint: "lfingersrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lfootrx"
  joint: "lfootrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lfootrz"
  joint: "lfootrz"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lhandrx"
  joint: "lhandrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lhandrz"
  joint: "lhandrz"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lhumerusrx"
  joint: "lhumerusrx"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "lhumerusry"
  joint: "lhumerusry"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "lhumerusrz"
  joint: "lhumerusrz"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "lowerbackrx"
  joint: "lowerbackrx"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "lowerbackry"
  joint: "lowerbackry"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "lowerbackrz"
  joint: "lowerbackrz"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "lowerneckrx"
  joint: "lowerneckrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lowerneckry"
  joint: "lowerneckry"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lowerneckrz"
  joint: "lowerneckrz"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lradiusrx"
  joint: "lradiusrx"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "lthumbrx"
  joint: "lthumbrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lthumbrz"
  joint: "lthumbrz"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "ltibiarx"
  joint: "ltibiarx"
  strength: 80.0
  angle {
  }
}
actuators {
  name: "ltoesrx"
  joint: "ltoesrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "lwristry"
  joint: "lwristry"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "rclaviclery"
  joint: "rclaviclery"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "rclaviclerz"
  joint: "rclaviclerz"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "rfemurrx"
  joint: "rfemurrx"
  strength: 120.0
  angle {
  }
}
actuators {
  name: "rfemurry"
  joint: "rfemurry"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "rfemurrz"
  joint: "rfemurrz"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "rfingersrx"
  joint: "rfingersrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "rfootrx"
  joint: "rfootrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "rfootrz"
  joint: "rfootrz"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "rhandrx"
  joint: "rhandrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "rhandrz"
  joint: "rhandrz"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "rhumerusrx"
  joint: "rhumerusrx"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "rhumerusry"
  joint: "rhumerusry"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "rhumerusrz"
  joint: "rhumerusrz"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "rradiusrx"
  joint: "rradiusrx"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "rthumbrx"
  joint: "rthumbrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "rthumbrz"
  joint: "rthumbrz"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "rtibiarx"
  joint: "rtibiarx"
  strength: 80.0
  angle {
  }
}
actuators {
  name: "rtoesrx"
  joint: "rtoesrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "rwristry"
  joint: "rwristry"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "thoraxrx"
  joint: "thoraxrx"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "thoraxry"
  joint: "thoraxry"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "thoraxrz"
  joint: "thoraxrz"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "upperbackrx"
  joint: "upperbackrx"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "upperbackry"
  joint: "upperbackry"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "upperbackrz"
  joint: "upperbackrz"
  strength: 40.0
  angle {
  }
}
actuators {
  name: "upperneckrx"
  joint: "upperneckrx"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "upperneckry"
  joint: "upperneckry"
  strength: 20.0
  angle {
  }
}
actuators {
  name: "upperneckrz"
  joint: "upperneckrz"
  strength: 20.0
  angle {
  }
}
friction: 1.0
gravity {
  z: -9.81
}
velocity_damping: 1.0
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
  first: "worldbody"
  second: "root"
}
collide_include {
  first: "worldbody"
  second: "lhipjoint"
}
collide_include {
  first: "worldbody"
  second: "lfemur"
}
collide_include {
  first: "worldbody"
  second: "ltibia"
}
collide_include {
  first: "worldbody"
  second: "lfoot"
}
collide_include {
  first: "worldbody"
  second: "ltoes"
}
collide_include {
  first: "worldbody"
  second: "rhipjoint"
}
collide_include {
  first: "worldbody"
  second: "rfemur"
}
collide_include {
  first: "worldbody"
  second: "rtibia"
}
collide_include {
  first: "worldbody"
  second: "rfoot"
}
collide_include {
  first: "worldbody"
  second: "rtoes"
}
collide_include {
  first: "worldbody"
  second: "lowerback"
}
collide_include {
  first: "worldbody"
  second: "upperback"
}
collide_include {
  first: "worldbody"
  second: "thorax"
}
collide_include {
  first: "worldbody"
  second: "lowerneck"
}
collide_include {
  first: "worldbody"
  second: "upperneck"
}
collide_include {
  first: "worldbody"
  second: "head"
}
collide_include {
  first: "worldbody"
  second: "lclavicle"
}
collide_include {
  first: "worldbody"
  second: "lhumerus"
}
collide_include {
  first: "worldbody"
  second: "lradius"
}
collide_include {
  first: "worldbody"
  second: "lwrist"
}
collide_include {
  first: "worldbody"
  second: "lhand"
}
collide_include {
  first: "worldbody"
  second: "lfingers"
}
collide_include {
  first: "worldbody"
  second: "lthumb"
}
collide_include {
  first: "worldbody"
  second: "rclavicle"
}
collide_include {
  first: "worldbody"
  second: "rhumerus"
}
collide_include {
  first: "worldbody"
  second: "rradius"
}
collide_include {
  first: "worldbody"
  second: "rwrist"
}
collide_include {
  first: "worldbody"
  second: "rhand"
}
collide_include {
  first: "worldbody"
  second: "rfingers"
}
collide_include {
  first: "worldbody"
  second: "rthumb"
}
collide_include {
  first: "root"
  second: "lfemur"
}
collide_include {
  first: "root"
  second: "ltibia"
}
collide_include {
  first: "root"
  second: "lfoot"
}
collide_include {
  first: "root"
  second: "ltoes"
}
collide_include {
  first: "root"
  second: "rfemur"
}
collide_include {
  first: "root"
  second: "rtibia"
}
collide_include {
  first: "root"
  second: "rfoot"
}
collide_include {
  first: "root"
  second: "rtoes"
}
collide_include {
  first: "root"
  second: "upperback"
}
collide_include {
  first: "root"
  second: "thorax"
}
collide_include {
  first: "root"
  second: "lowerneck"
}
collide_include {
  first: "root"
  second: "upperneck"
}
collide_include {
  first: "root"
  second: "head"
}
collide_include {
  first: "root"
  second: "lclavicle"
}
collide_include {
  first: "root"
  second: "lhumerus"
}
collide_include {
  first: "root"
  second: "lradius"
}
collide_include {
  first: "root"
  second: "lwrist"
}
collide_include {
  first: "root"
  second: "lhand"
}
collide_include {
  first: "root"
  second: "lfingers"
}
collide_include {
  first: "root"
  second: "lthumb"
}
collide_include {
  first: "root"
  second: "rclavicle"
}
collide_include {
  first: "root"
  second: "rhumerus"
}
collide_include {
  first: "root"
  second: "rradius"
}
collide_include {
  first: "root"
  second: "rwrist"
}
collide_include {
  first: "root"
  second: "rhand"
}
collide_include {
  first: "root"
  second: "rfingers"
}
collide_include {
  first: "root"
  second: "rthumb"
}
collide_include {
  first: "lhipjoint"
  second: "ltibia"
}
collide_include {
  first: "lhipjoint"
  second: "lfoot"
}
collide_include {
  first: "lhipjoint"
  second: "ltoes"
}
collide_include {
  first: "lhipjoint"
  second: "rfemur"
}
collide_include {
  first: "lhipjoint"
  second: "rtibia"
}
collide_include {
  first: "lhipjoint"
  second: "rfoot"
}
collide_include {
  first: "lhipjoint"
  second: "rtoes"
}
collide_include {
  first: "lhipjoint"
  second: "upperback"
}
collide_include {
  first: "lhipjoint"
  second: "thorax"
}
collide_include {
  first: "lhipjoint"
  second: "lowerneck"
}
collide_include {
  first: "lhipjoint"
  second: "upperneck"
}
collide_include {
  first: "lhipjoint"
  second: "head"
}
collide_include {
  first: "lhipjoint"
  second: "lclavicle"
}
collide_include {
  first: "lhipjoint"
  second: "lhumerus"
}
collide_include {
  first: "lhipjoint"
  second: "lradius"
}
collide_include {
  first: "lhipjoint"
  second: "lwrist"
}
collide_include {
  first: "lhipjoint"
  second: "lhand"
}
collide_include {
  first: "lhipjoint"
  second: "lfingers"
}
collide_include {
  first: "lhipjoint"
  second: "lthumb"
}
collide_include {
  first: "lhipjoint"
  second: "rclavicle"
}
collide_include {
  first: "lhipjoint"
  second: "rhumerus"
}
collide_include {
  first: "lhipjoint"
  second: "rradius"
}
collide_include {
  first: "lhipjoint"
  second: "rwrist"
}
collide_include {
  first: "lhipjoint"
  second: "rhand"
}
collide_include {
  first: "lhipjoint"
  second: "rfingers"
}
collide_include {
  first: "lhipjoint"
  second: "rthumb"
}
collide_include {
  first: "lfemur"
  second: "lfoot"
}
collide_include {
  first: "lfemur"
  second: "ltoes"
}
collide_include {
  first: "lfemur"
  second: "rhipjoint"
}
collide_include {
  first: "lfemur"
  second: "rfemur"
}
collide_include {
  first: "lfemur"
  second: "rtibia"
}
collide_include {
  first: "lfemur"
  second: "rfoot"
}
collide_include {
  first: "lfemur"
  second: "rtoes"
}
collide_include {
  first: "lfemur"
  second: "lowerback"
}
collide_include {
  first: "lfemur"
  second: "upperback"
}
collide_include {
  first: "lfemur"
  second: "thorax"
}
collide_include {
  first: "lfemur"
  second: "lowerneck"
}
collide_include {
  first: "lfemur"
  second: "upperneck"
}
collide_include {
  first: "lfemur"
  second: "head"
}
collide_include {
  first: "lfemur"
  second: "lclavicle"
}
collide_include {
  first: "lfemur"
  second: "lhumerus"
}
collide_include {
  first: "lfemur"
  second: "lradius"
}
collide_include {
  first: "lfemur"
  second: "lwrist"
}
collide_include {
  first: "lfemur"
  second: "lhand"
}
collide_include {
  first: "lfemur"
  second: "lfingers"
}
collide_include {
  first: "lfemur"
  second: "lthumb"
}
collide_include {
  first: "lfemur"
  second: "rclavicle"
}
collide_include {
  first: "lfemur"
  second: "rhumerus"
}
collide_include {
  first: "lfemur"
  second: "rradius"
}
collide_include {
  first: "lfemur"
  second: "rwrist"
}
collide_include {
  first: "lfemur"
  second: "rhand"
}
collide_include {
  first: "lfemur"
  second: "rfingers"
}
collide_include {
  first: "lfemur"
  second: "rthumb"
}
collide_include {
  first: "ltibia"
  second: "ltoes"
}
collide_include {
  first: "ltibia"
  second: "rhipjoint"
}
collide_include {
  first: "ltibia"
  second: "rfemur"
}
collide_include {
  first: "ltibia"
  second: "rtibia"
}
collide_include {
  first: "ltibia"
  second: "rfoot"
}
collide_include {
  first: "ltibia"
  second: "rtoes"
}
collide_include {
  first: "ltibia"
  second: "lowerback"
}
collide_include {
  first: "ltibia"
  second: "upperback"
}
collide_include {
  first: "ltibia"
  second: "thorax"
}
collide_include {
  first: "ltibia"
  second: "lowerneck"
}
collide_include {
  first: "ltibia"
  second: "upperneck"
}
collide_include {
  first: "ltibia"
  second: "head"
}
collide_include {
  first: "ltibia"
  second: "lclavicle"
}
collide_include {
  first: "ltibia"
  second: "lhumerus"
}
collide_include {
  first: "ltibia"
  second: "lradius"
}
collide_include {
  first: "ltibia"
  second: "lwrist"
}
collide_include {
  first: "ltibia"
  second: "lhand"
}
collide_include {
  first: "ltibia"
  second: "lfingers"
}
collide_include {
  first: "ltibia"
  second: "lthumb"
}
collide_include {
  first: "ltibia"
  second: "rclavicle"
}
collide_include {
  first: "ltibia"
  second: "rhumerus"
}
collide_include {
  first: "ltibia"
  second: "rradius"
}
collide_include {
  first: "ltibia"
  second: "rwrist"
}
collide_include {
  first: "ltibia"
  second: "rhand"
}
collide_include {
  first: "ltibia"
  second: "rfingers"
}
collide_include {
  first: "ltibia"
  second: "rthumb"
}
collide_include {
  first: "lfoot"
  second: "rhipjoint"
}
collide_include {
  first: "lfoot"
  second: "rfemur"
}
collide_include {
  first: "lfoot"
  second: "rtibia"
}
collide_include {
  first: "lfoot"
  second: "rfoot"
}
collide_include {
  first: "lfoot"
  second: "rtoes"
}
collide_include {
  first: "lfoot"
  second: "lowerback"
}
collide_include {
  first: "lfoot"
  second: "upperback"
}
collide_include {
  first: "lfoot"
  second: "thorax"
}
collide_include {
  first: "lfoot"
  second: "lowerneck"
}
collide_include {
  first: "lfoot"
  second: "upperneck"
}
collide_include {
  first: "lfoot"
  second: "head"
}
collide_include {
  first: "lfoot"
  second: "lclavicle"
}
collide_include {
  first: "lfoot"
  second: "lhumerus"
}
collide_include {
  first: "lfoot"
  second: "lradius"
}
collide_include {
  first: "lfoot"
  second: "lwrist"
}
collide_include {
  first: "lfoot"
  second: "lhand"
}
collide_include {
  first: "lfoot"
  second: "lfingers"
}
collide_include {
  first: "lfoot"
  second: "lthumb"
}
collide_include {
  first: "lfoot"
  second: "rclavicle"
}
collide_include {
  first: "lfoot"
  second: "rhumerus"
}
collide_include {
  first: "lfoot"
  second: "rradius"
}
collide_include {
  first: "lfoot"
  second: "rwrist"
}
collide_include {
  first: "lfoot"
  second: "rhand"
}
collide_include {
  first: "lfoot"
  second: "rfingers"
}
collide_include {
  first: "lfoot"
  second: "rthumb"
}
collide_include {
  first: "ltoes"
  second: "rhipjoint"
}
collide_include {
  first: "ltoes"
  second: "rfemur"
}
collide_include {
  first: "ltoes"
  second: "rtibia"
}
collide_include {
  first: "ltoes"
  second: "rfoot"
}
collide_include {
  first: "ltoes"
  second: "rtoes"
}
collide_include {
  first: "ltoes"
  second: "lowerback"
}
collide_include {
  first: "ltoes"
  second: "upperback"
}
collide_include {
  first: "ltoes"
  second: "thorax"
}
collide_include {
  first: "ltoes"
  second: "lowerneck"
}
collide_include {
  first: "ltoes"
  second: "upperneck"
}
collide_include {
  first: "ltoes"
  second: "head"
}
collide_include {
  first: "ltoes"
  second: "lclavicle"
}
collide_include {
  first: "ltoes"
  second: "lhumerus"
}
collide_include {
  first: "ltoes"
  second: "lradius"
}
collide_include {
  first: "ltoes"
  second: "lwrist"
}
collide_include {
  first: "ltoes"
  second: "lhand"
}
collide_include {
  first: "ltoes"
  second: "lfingers"
}
collide_include {
  first: "ltoes"
  second: "lthumb"
}
collide_include {
  first: "ltoes"
  second: "rclavicle"
}
collide_include {
  first: "ltoes"
  second: "rhumerus"
}
collide_include {
  first: "ltoes"
  second: "rradius"
}
collide_include {
  first: "ltoes"
  second: "rwrist"
}
collide_include {
  first: "ltoes"
  second: "rhand"
}
collide_include {
  first: "ltoes"
  second: "rfingers"
}
collide_include {
  first: "ltoes"
  second: "rthumb"
}
collide_include {
  first: "rhipjoint"
  second: "rtibia"
}
collide_include {
  first: "rhipjoint"
  second: "rfoot"
}
collide_include {
  first: "rhipjoint"
  second: "rtoes"
}
collide_include {
  first: "rhipjoint"
  second: "upperback"
}
collide_include {
  first: "rhipjoint"
  second: "thorax"
}
collide_include {
  first: "rhipjoint"
  second: "lowerneck"
}
collide_include {
  first: "rhipjoint"
  second: "upperneck"
}
collide_include {
  first: "rhipjoint"
  second: "head"
}
collide_include {
  first: "rhipjoint"
  second: "lclavicle"
}
collide_include {
  first: "rhipjoint"
  second: "lhumerus"
}
collide_include {
  first: "rhipjoint"
  second: "lradius"
}
collide_include {
  first: "rhipjoint"
  second: "lwrist"
}
collide_include {
  first: "rhipjoint"
  second: "lhand"
}
collide_include {
  first: "rhipjoint"
  second: "lfingers"
}
collide_include {
  first: "rhipjoint"
  second: "lthumb"
}
collide_include {
  first: "rhipjoint"
  second: "rclavicle"
}
collide_include {
  first: "rhipjoint"
  second: "rhumerus"
}
collide_include {
  first: "rhipjoint"
  second: "rradius"
}
collide_include {
  first: "rhipjoint"
  second: "rwrist"
}
collide_include {
  first: "rhipjoint"
  second: "rhand"
}
collide_include {
  first: "rhipjoint"
  second: "rfingers"
}
collide_include {
  first: "rhipjoint"
  second: "rthumb"
}
collide_include {
  first: "rfemur"
  second: "rfoot"
}
collide_include {
  first: "rfemur"
  second: "rtoes"
}
collide_include {
  first: "rfemur"
  second: "lowerback"
}
collide_include {
  first: "rfemur"
  second: "upperback"
}
collide_include {
  first: "rfemur"
  second: "thorax"
}
collide_include {
  first: "rfemur"
  second: "lowerneck"
}
collide_include {
  first: "rfemur"
  second: "upperneck"
}
collide_include {
  first: "rfemur"
  second: "head"
}
collide_include {
  first: "rfemur"
  second: "lclavicle"
}
collide_include {
  first: "rfemur"
  second: "lhumerus"
}
collide_include {
  first: "rfemur"
  second: "lradius"
}
collide_include {
  first: "rfemur"
  second: "lwrist"
}
collide_include {
  first: "rfemur"
  second: "lhand"
}
collide_include {
  first: "rfemur"
  second: "lfingers"
}
collide_include {
  first: "rfemur"
  second: "lthumb"
}
collide_include {
  first: "rfemur"
  second: "rclavicle"
}
collide_include {
  first: "rfemur"
  second: "rhumerus"
}
collide_include {
  first: "rfemur"
  second: "rradius"
}
collide_include {
  first: "rfemur"
  second: "rwrist"
}
collide_include {
  first: "rfemur"
  second: "rhand"
}
collide_include {
  first: "rfemur"
  second: "rfingers"
}
collide_include {
  first: "rfemur"
  second: "rthumb"
}
collide_include {
  first: "rtibia"
  second: "rtoes"
}
collide_include {
  first: "rtibia"
  second: "lowerback"
}
collide_include {
  first: "rtibia"
  second: "upperback"
}
collide_include {
  first: "rtibia"
  second: "thorax"
}
collide_include {
  first: "rtibia"
  second: "lowerneck"
}
collide_include {
  first: "rtibia"
  second: "upperneck"
}
collide_include {
  first: "rtibia"
  second: "head"
}
collide_include {
  first: "rtibia"
  second: "lclavicle"
}
collide_include {
  first: "rtibia"
  second: "lhumerus"
}
collide_include {
  first: "rtibia"
  second: "lradius"
}
collide_include {
  first: "rtibia"
  second: "lwrist"
}
collide_include {
  first: "rtibia"
  second: "lhand"
}
collide_include {
  first: "rtibia"
  second: "lfingers"
}
collide_include {
  first: "rtibia"
  second: "lthumb"
}
collide_include {
  first: "rtibia"
  second: "rclavicle"
}
collide_include {
  first: "rtibia"
  second: "rhumerus"
}
collide_include {
  first: "rtibia"
  second: "rradius"
}
collide_include {
  first: "rtibia"
  second: "rwrist"
}
collide_include {
  first: "rtibia"
  second: "rhand"
}
collide_include {
  first: "rtibia"
  second: "rfingers"
}
collide_include {
  first: "rtibia"
  second: "rthumb"
}
collide_include {
  first: "rfoot"
  second: "lowerback"
}
collide_include {
  first: "rfoot"
  second: "upperback"
}
collide_include {
  first: "rfoot"
  second: "thorax"
}
collide_include {
  first: "rfoot"
  second: "lowerneck"
}
collide_include {
  first: "rfoot"
  second: "upperneck"
}
collide_include {
  first: "rfoot"
  second: "head"
}
collide_include {
  first: "rfoot"
  second: "lclavicle"
}
collide_include {
  first: "rfoot"
  second: "lhumerus"
}
collide_include {
  first: "rfoot"
  second: "lradius"
}
collide_include {
  first: "rfoot"
  second: "lwrist"
}
collide_include {
  first: "rfoot"
  second: "lhand"
}
collide_include {
  first: "rfoot"
  second: "lfingers"
}
collide_include {
  first: "rfoot"
  second: "lthumb"
}
collide_include {
  first: "rfoot"
  second: "rclavicle"
}
collide_include {
  first: "rfoot"
  second: "rhumerus"
}
collide_include {
  first: "rfoot"
  second: "rradius"
}
collide_include {
  first: "rfoot"
  second: "rwrist"
}
collide_include {
  first: "rfoot"
  second: "rhand"
}
collide_include {
  first: "rfoot"
  second: "rfingers"
}
collide_include {
  first: "rfoot"
  second: "rthumb"
}
collide_include {
  first: "rtoes"
  second: "lowerback"
}
collide_include {
  first: "rtoes"
  second: "upperback"
}
collide_include {
  first: "rtoes"
  second: "thorax"
}
collide_include {
  first: "rtoes"
  second: "lowerneck"
}
collide_include {
  first: "rtoes"
  second: "upperneck"
}
collide_include {
  first: "rtoes"
  second: "head"
}
collide_include {
  first: "rtoes"
  second: "lclavicle"
}
collide_include {
  first: "rtoes"
  second: "lhumerus"
}
collide_include {
  first: "rtoes"
  second: "lradius"
}
collide_include {
  first: "rtoes"
  second: "lwrist"
}
collide_include {
  first: "rtoes"
  second: "lhand"
}
collide_include {
  first: "rtoes"
  second: "lfingers"
}
collide_include {
  first: "rtoes"
  second: "lthumb"
}
collide_include {
  first: "rtoes"
  second: "rclavicle"
}
collide_include {
  first: "rtoes"
  second: "rhumerus"
}
collide_include {
  first: "rtoes"
  second: "rradius"
}
collide_include {
  first: "rtoes"
  second: "rwrist"
}
collide_include {
  first: "rtoes"
  second: "rhand"
}
collide_include {
  first: "rtoes"
  second: "rfingers"
}
collide_include {
  first: "rtoes"
  second: "rthumb"
}
collide_include {
  first: "lowerback"
  second: "thorax"
}
collide_include {
  first: "lowerback"
  second: "lowerneck"
}
collide_include {
  first: "lowerback"
  second: "upperneck"
}
collide_include {
  first: "lowerback"
  second: "head"
}
collide_include {
  first: "lowerback"
  second: "lclavicle"
}
collide_include {
  first: "lowerback"
  second: "lhumerus"
}
collide_include {
  first: "lowerback"
  second: "lradius"
}
collide_include {
  first: "lowerback"
  second: "lwrist"
}
collide_include {
  first: "lowerback"
  second: "lhand"
}
collide_include {
  first: "lowerback"
  second: "lfingers"
}
collide_include {
  first: "lowerback"
  second: "lthumb"
}
collide_include {
  first: "lowerback"
  second: "rclavicle"
}
collide_include {
  first: "lowerback"
  second: "rhumerus"
}
collide_include {
  first: "lowerback"
  second: "rradius"
}
collide_include {
  first: "lowerback"
  second: "rwrist"
}
collide_include {
  first: "lowerback"
  second: "rhand"
}
collide_include {
  first: "lowerback"
  second: "rfingers"
}
collide_include {
  first: "lowerback"
  second: "rthumb"
}
collide_include {
  first: "upperback"
  second: "lowerneck"
}
collide_include {
  first: "upperback"
  second: "upperneck"
}
collide_include {
  first: "upperback"
  second: "head"
}
collide_include {
  first: "upperback"
  second: "lclavicle"
}
collide_include {
  first: "upperback"
  second: "lhumerus"
}
collide_include {
  first: "upperback"
  second: "lradius"
}
collide_include {
  first: "upperback"
  second: "lwrist"
}
collide_include {
  first: "upperback"
  second: "lhand"
}
collide_include {
  first: "upperback"
  second: "lfingers"
}
collide_include {
  first: "upperback"
  second: "lthumb"
}
collide_include {
  first: "upperback"
  second: "rclavicle"
}
collide_include {
  first: "upperback"
  second: "rhumerus"
}
collide_include {
  first: "upperback"
  second: "rradius"
}
collide_include {
  first: "upperback"
  second: "rwrist"
}
collide_include {
  first: "upperback"
  second: "rhand"
}
collide_include {
  first: "upperback"
  second: "rfingers"
}
collide_include {
  first: "upperback"
  second: "rthumb"
}
collide_include {
  first: "thorax"
  second: "upperneck"
}
collide_include {
  first: "thorax"
  second: "head"
}
collide_include {
  first: "thorax"
  second: "lhumerus"
}
collide_include {
  first: "thorax"
  second: "lradius"
}
collide_include {
  first: "thorax"
  second: "lwrist"
}
collide_include {
  first: "thorax"
  second: "lhand"
}
collide_include {
  first: "thorax"
  second: "lfingers"
}
collide_include {
  first: "thorax"
  second: "lthumb"
}
collide_include {
  first: "thorax"
  second: "rhumerus"
}
collide_include {
  first: "thorax"
  second: "rradius"
}
collide_include {
  first: "thorax"
  second: "rwrist"
}
collide_include {
  first: "thorax"
  second: "rhand"
}
collide_include {
  first: "thorax"
  second: "rfingers"
}
collide_include {
  first: "thorax"
  second: "rthumb"
}
collide_include {
  first: "lowerneck"
  second: "head"
}
collide_include {
  first: "lowerneck"
  second: "lhumerus"
}
collide_include {
  first: "lowerneck"
  second: "lradius"
}
collide_include {
  first: "lowerneck"
  second: "lwrist"
}
collide_include {
  first: "lowerneck"
  second: "lhand"
}
collide_include {
  first: "lowerneck"
  second: "lfingers"
}
collide_include {
  first: "lowerneck"
  second: "lthumb"
}
collide_include {
  first: "lowerneck"
  second: "rhumerus"
}
collide_include {
  first: "lowerneck"
  second: "rradius"
}
collide_include {
  first: "lowerneck"
  second: "rwrist"
}
collide_include {
  first: "lowerneck"
  second: "rhand"
}
collide_include {
  first: "lowerneck"
  second: "rfingers"
}
collide_include {
  first: "lowerneck"
  second: "rthumb"
}
collide_include {
  first: "upperneck"
  second: "lclavicle"
}
collide_include {
  first: "upperneck"
  second: "lhumerus"
}
collide_include {
  first: "upperneck"
  second: "lradius"
}
collide_include {
  first: "upperneck"
  second: "lwrist"
}
collide_include {
  first: "upperneck"
  second: "lhand"
}
collide_include {
  first: "upperneck"
  second: "lfingers"
}
collide_include {
  first: "upperneck"
  second: "lthumb"
}
collide_include {
  first: "upperneck"
  second: "rclavicle"
}
collide_include {
  first: "upperneck"
  second: "rhumerus"
}
collide_include {
  first: "upperneck"
  second: "rradius"
}
collide_include {
  first: "upperneck"
  second: "rwrist"
}
collide_include {
  first: "upperneck"
  second: "rhand"
}
collide_include {
  first: "upperneck"
  second: "rfingers"
}
collide_include {
  first: "upperneck"
  second: "rthumb"
}
collide_include {
  first: "head"
  second: "lclavicle"
}
collide_include {
  first: "head"
  second: "lhumerus"
}
collide_include {
  first: "head"
  second: "lradius"
}
collide_include {
  first: "head"
  second: "lwrist"
}
collide_include {
  first: "head"
  second: "lhand"
}
collide_include {
  first: "head"
  second: "lfingers"
}
collide_include {
  first: "head"
  second: "lthumb"
}
collide_include {
  first: "head"
  second: "rclavicle"
}
collide_include {
  first: "head"
  second: "rhumerus"
}
collide_include {
  first: "head"
  second: "rradius"
}
collide_include {
  first: "head"
  second: "rwrist"
}
collide_include {
  first: "head"
  second: "rhand"
}
collide_include {
  first: "head"
  second: "rfingers"
}
collide_include {
  first: "head"
  second: "rthumb"
}
collide_include {
  first: "lclavicle"
  second: "lradius"
}
collide_include {
  first: "lclavicle"
  second: "lwrist"
}
collide_include {
  first: "lclavicle"
  second: "lhand"
}
collide_include {
  first: "lclavicle"
  second: "lfingers"
}
collide_include {
  first: "lclavicle"
  second: "lthumb"
}
collide_include {
  first: "lclavicle"
  second: "rhumerus"
}
collide_include {
  first: "lclavicle"
  second: "rradius"
}
collide_include {
  first: "lclavicle"
  second: "rwrist"
}
collide_include {
  first: "lclavicle"
  second: "rhand"
}
collide_include {
  first: "lclavicle"
  second: "rfingers"
}
collide_include {
  first: "lclavicle"
  second: "rthumb"
}
collide_include {
  first: "lhumerus"
  second: "lwrist"
}
collide_include {
  first: "lhumerus"
  second: "lhand"
}
collide_include {
  first: "lhumerus"
  second: "lfingers"
}
collide_include {
  first: "lhumerus"
  second: "lthumb"
}
collide_include {
  first: "lhumerus"
  second: "rclavicle"
}
collide_include {
  first: "lhumerus"
  second: "rhumerus"
}
collide_include {
  first: "lhumerus"
  second: "rradius"
}
collide_include {
  first: "lhumerus"
  second: "rwrist"
}
collide_include {
  first: "lhumerus"
  second: "rhand"
}
collide_include {
  first: "lhumerus"
  second: "rfingers"
}
collide_include {
  first: "lhumerus"
  second: "rthumb"
}
collide_include {
  first: "lradius"
  second: "lhand"
}
collide_include {
  first: "lradius"
  second: "lfingers"
}
collide_include {
  first: "lradius"
  second: "lthumb"
}
collide_include {
  first: "lradius"
  second: "rclavicle"
}
collide_include {
  first: "lradius"
  second: "rhumerus"
}
collide_include {
  first: "lradius"
  second: "rradius"
}
collide_include {
  first: "lradius"
  second: "rwrist"
}
collide_include {
  first: "lradius"
  second: "rhand"
}
collide_include {
  first: "lradius"
  second: "rfingers"
}
collide_include {
  first: "lradius"
  second: "rthumb"
}
collide_include {
  first: "lwrist"
  second: "lfingers"
}
collide_include {
  first: "lwrist"
  second: "lthumb"
}
collide_include {
  first: "lwrist"
  second: "rclavicle"
}
collide_include {
  first: "lwrist"
  second: "rhumerus"
}
collide_include {
  first: "lwrist"
  second: "rradius"
}
collide_include {
  first: "lwrist"
  second: "rwrist"
}
collide_include {
  first: "lwrist"
  second: "rhand"
}
collide_include {
  first: "lwrist"
  second: "rfingers"
}
collide_include {
  first: "lwrist"
  second: "rthumb"
}
collide_include {
  first: "lhand"
  second: "rclavicle"
}
collide_include {
  first: "lhand"
  second: "rhumerus"
}
collide_include {
  first: "lhand"
  second: "rradius"
}
collide_include {
  first: "lhand"
  second: "rwrist"
}
collide_include {
  first: "lhand"
  second: "rhand"
}
collide_include {
  first: "lhand"
  second: "rfingers"
}
collide_include {
  first: "lhand"
  second: "rthumb"
}
collide_include {
  first: "lfingers"
  second: "rclavicle"
}
collide_include {
  first: "lfingers"
  second: "rhumerus"
}
collide_include {
  first: "lfingers"
  second: "rradius"
}
collide_include {
  first: "lfingers"
  second: "rwrist"
}
collide_include {
  first: "lfingers"
  second: "rhand"
}
collide_include {
  first: "lfingers"
  second: "rfingers"
}
collide_include {
  first: "lfingers"
  second: "rthumb"
}
collide_include {
  first: "lthumb"
  second: "rclavicle"
}
collide_include {
  first: "lthumb"
  second: "rhumerus"
}
collide_include {
  first: "lthumb"
  second: "rradius"
}
collide_include {
  first: "lthumb"
  second: "rwrist"
}
collide_include {
  first: "lthumb"
  second: "rhand"
}
collide_include {
  first: "lthumb"
  second: "rfingers"
}
collide_include {
  first: "lthumb"
  second: "rthumb"
}
collide_include {
  first: "rclavicle"
  second: "rradius"
}
collide_include {
  first: "rclavicle"
  second: "rwrist"
}
collide_include {
  first: "rclavicle"
  second: "rhand"
}
collide_include {
  first: "rclavicle"
  second: "rfingers"
}
collide_include {
  first: "rclavicle"
  second: "rthumb"
}
collide_include {
  first: "rhumerus"
  second: "rwrist"
}
collide_include {
  first: "rhumerus"
  second: "rhand"
}
collide_include {
  first: "rhumerus"
  second: "rfingers"
}
collide_include {
  first: "rhumerus"
  second: "rthumb"
}
collide_include {
  first: "rradius"
  second: "rhand"
}
collide_include {
  first: "rradius"
  second: "rfingers"
}
collide_include {
  first: "rradius"
  second: "rthumb"
}
collide_include {
  first: "rwrist"
  second: "rfingers"
}
collide_include {
  first: "rwrist"
  second: "rthumb"
}
friction: 1.0
gravity {
  z: -9.81
}
angular_damping: -0.05
dt: 0.015
substeps: 8
dynamics_mode: "pbd"


  """
