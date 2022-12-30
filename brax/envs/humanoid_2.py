import brax
from brax import jumpy as jp
from brax.envs import env
from jax.experimental.host_callback import call
from google.protobuf import text_format
import jax

class Humanoid(env.Env):

  print("Using new humanoid 2")

  # CHANGED: health_z_range from (0.8, 2.1) to (1.2, 2.1)
  def __init__(self,
               forward_reward_weight=1.25,
               ctrl_cost_weight=0.1,
               healthy_reward=5.0,
               terminate_when_unhealthy=True,
               healthy_z_range=(1.1, 2.0),
               reset_noise_scale=1e-2,
               exclude_current_positions_from_observation=True,
               legacy_spring=False,
               **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
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

    self.torso_idx = self.sys.body.index['torso']
    self.target_idx = self.sys.body.index['Target']
    print("Target index : ", self.sys.body.index['Target'])
    # print("Torso index : ", self.torso_idx)
    # print( "QP position of torso index : ", env.State.qp.pos[self.torso_idx] )
    # jax.experimental.host_callback.id_print(f"QP position of torso index : {env.State.qp.pos[self.torso_idx]}")
    print( "System body consists  : ", self.sys.body )


  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)

    qpos = self.sys.default_angle() + self._noise(rng1)
    qvel = self._noise(rng2)

    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    obs = self._get_obs(qp, self.sys.info(qp), jp.zeros(self.action_size))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'target_reward': zero,
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    pos = jp.index_update(qp.pos, self.target_idx, jax.numpy.array([10., 10., 1.5]))
    qp = qp.replace(pos=pos)
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    
    # forward reward for moving forward
    com_before = self._center_of_mass(state.qp)
    com_after = self._center_of_mass(qp)
    velocity = (com_after - com_before) / self.sys.config.dt
    forward_reward = self._forward_reward_weight * velocity[0]
    # forward_reward = 0.

    # small reward for torso moving towards target
    torso_delta = com_after - com_before
    target_rel = qp.pos[self.target_idx] - com_after
    target_dist = jp.norm(target_rel)
    target_dir = target_rel / (1e-6 + target_dist)
    # target_reward = self._forward_reward_weight * jp.dot(velocity, target_dir)
    target_reward = 0.

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

  def _get_obs(self, qp: brax.QP, info: brax.Info,
               action: jp.ndarray) -> jp.ndarray:
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
    name: "torso"
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
    name: "lwaist"
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
    name: "pelvis"
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
    name: "right_thigh"
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
    name: "right_shin"
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
    name: "left_thigh"
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
    name: "left_shin"
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
    name: "right_upper_arm"
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
    name: "right_lower_arm"
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
    name: "left_upper_arm"
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
    name: "left_lower_arm"
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
  
  joints {
    name: "abdomen_yz"
    parent: "torso"
    child: "lwaist"
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
    angle_limit {
      min: -45.0
      max: 45.0
    }
    angle_limit {
      min: -65.0
      max: 30.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "abdomen_x"
    parent: "lwaist"
    child: "pelvis"
    parent_offset {
      z: -0.065
    }
    child_offset {
      z: 0.1
    }
    rotation {
      x: 90.0
    }
    angle_limit {
      min: -35.0
      max: 35.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_hip_xyz"
    parent: "pelvis"
    child: "right_thigh"
    parent_offset {
      y: -0.1
      z: -0.04
    }
    child_offset {
    }
    rotation {
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angle_limit {
      min: -30.0
      max: 70.0
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_knee"
    parent: "right_thigh"
    child: "right_shin"
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
    angle_limit {
      min: -160.0
      max: -2.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_hip_xyz"
    parent: "pelvis"
    child: "left_thigh"
    parent_offset {
      y: 0.1
      z: -0.04
    }
    child_offset {
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angle_limit {
      min: -30.0
      max: 70.0
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_knee"
    parent: "left_thigh"
    child: "left_shin"
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
    angle_limit {
      min: -160.0
      max: -2.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_shoulder12"
    parent: "torso"
    child: "right_upper_arm"
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
    angle_limit {
      min: -5.0
      max: 5.0
    }
    angle_limit {
      min: -90.0
      max: +40.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_elbow"
    parent: "right_upper_arm"
    child: "right_lower_arm"
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
    angle_limit {
      min: -90.0
      max: 50.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_shoulder12"
    parent: "torso"
    child: "left_upper_arm"
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
    angle_limit {
      min: -5.0
      max: 5.0
    }
    angle_limit {
      min: -40.0
      max: 90.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_elbow"
    parent: "left_upper_arm"
    child: "left_lower_arm"
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
    angle_limit {
      min: -90.0
      max: 50.0
    }
    angular_damping: 30.0
  }
  actuators {
    name: "abdomen_yz"
    joint: "abdomen_yz"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "abdomen_x"
    joint: "abdomen_x"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_hip_xyz"
    joint: "right_hip_xyz"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_knee"
    joint: "right_knee"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "left_hip_xyz"
    joint: "left_hip_xyz"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "left_knee"
    joint: "left_knee"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_shoulder12"
    joint: "right_shoulder12"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "right_elbow"
    joint: "right_elbow"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "left_shoulder12"
    joint: "left_shoulder12"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "left_elbow"
    joint: "left_elbow"
    strength: 100.0
    torque {
    }
  }
  collide_include {
    first: "floor"
    second: "left_shin"
  }
  collide_include {
    first: "floor"
    second: "right_shin"
  }
  collide_include {
    first: "Wall1"
    second: "right_shin"
  }
  collide_include {
    first: "Wall1"
    second: "left_shin"
  }
  collide_include {
    first: "Wall2"
    second: "right_shin"
  }
  collide_include {
    first: "Wall2"
    second: "left_shin"
  }
  collide_include {
    first: "Wall11"
    second: "right_shin"
  }
  collide_include {
    first: "Wall11"
    second: "left_shin"
  }
   collide_include {
    first: "Wall12"
    second: "right_shin"
  }
  collide_include {
    first: "Wall12"
    second: "left_shin"
  }



  bodies {
    name: "Target"
    colliders {
      position {
        x: +10
      }
      sphere {
        radius: 0.1
      }
    }
    frozen {
      all: true
    }
  }
  bodies {
    name: "Wall1"
    colliders {
      position {
        x: 5.0
        y: 2.5
        z: 0
      }
      box {
        halfsize { x: 5.0 y: 0.5 z: 0.5 }
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

  bodies {
    name: "Wall2"
    colliders {
      position {
        x: 10.0
        y: 2.5
        z: 0
      }
      box {
        halfsize { x: 1.0 y: 5.0 z: 0.5 }
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


  bodies {
    name: "Wall11"
    colliders {
      position {
        x: 15.0
        y: -7.5
        z: 0
      }
      box {
        halfsize { x: 5.0 y: 0.5 z: 0.5 }
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

  bodies {
    name: "Wall12"
    colliders {
      position {
        x: 20.0
        y: -7.5
        z: 0
      }
      box {
        halfsize { x: 1.0 y: 6.0 z: 0.5 }
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




  bodies {
    name: "Wall3"
    colliders {
      position {
        x: 10.0
        y: 6.5
        z: 0
      }
      box {
        halfsize { x: 0.5 y: 4 z: 0.5 }
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

  bodies {
    name: "Wall4"
    colliders {
      position {
        x: 15.0
        y: 5.5
        z: 0
      }
      box {
        halfsize { x: 0.5 y: 4 z: 0.5 }
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

  bodies {
    name: "Wall5"
    colliders {
      position {
        x: 17.5
        y: 13.0
        z: 0
      }
      box {
        halfsize { x: 7.5 y: 0.5 z: 0.5 }
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

  bodies {
    name: "Wall6"
    colliders {
      position {
        x: 20.0
        y: 8.0
        z: 0
      }
      box {
        halfsize { x: 5.0 y: 0.5 z: 0.5 }
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


  defaults {
    angles {
      name: "left_knee"
      angle { x: -25. y: 0 z: 0 }
    }
    angles {
      name: "right_knee"
      angle { x: -25. y: 0 z: 0 }
    }
    angles {
      name: "right_shoulder12"
      angle {x: 35. y: 0 z: 0 }
    }
    angles {
      name: "left_shoulder12"
      angle { x: -35. y: 0 z: 0 }
    }
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



_SYSTEM_CONFIG_SPRING = """
  
"""
