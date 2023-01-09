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
    self.target_radius = 0.5
    self.target_x_distance = 15
    self.target_y_distance = 3
    # print("Target index : ", self.sys.body.index['Target'])
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
        'x_position_torso': zero,
        'x_position_target': zero,
        'y_position': zero,
        'y_position_torso': zero,
        'y_position_target': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    info = {'rng': rng}
    pos = jp.index_update(qp.pos, self.target_idx, jax.numpy.array([5., 0., 1.0]))
    qp = qp.replace(pos=pos)
    return env.State(qp, obs, reward, done, metrics, info)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    
    # forward reward for moving forward
    com_before = self._center_of_mass(state.qp)
    com_after = self._center_of_mass(qp)
    velocity = (com_after - com_before) / self.sys.config.dt
    in_between = jp.where(com_after[0] >21., 0.0, -1.0)
    in_between = jp.where(com_after[0] < 14., 1.0, in_between)
    in_between = jp.where(com_after[0] > 7., in_between, -1.0)
    forward_reward = self._forward_reward_weight * velocity[0] + self._forward_reward_weight * velocity[1] * in_between * 0.35
    # forward_reward = self._forward_reward_weight * velocity[0]
    # forward_reward = 0.

    # small reward for torso moving towards target
    torso_delta = qp.pos[self.torso_idx] - state.qp.pos[self.torso_idx]
    target_rel = qp.pos[self.target_idx] - qp.pos[self.torso_idx]
    target_dist = jp.norm(target_rel)
    target_dir = target_rel / (1e-6 + target_dist)
    # target_reward = 1.5 * jp.dot(velocity, target_dir)
    target_reward = 0.

    # big reward for reaching target and facing it
    target_hit = target_dist < self.target_radius
    target_hit = jp.where(target_hit, jp.float32(1), jp.float32(0))
    # target_reward += target_hit * 100

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
        x_position_torso=qp.pos[self.torso_idx][0],
        x_position_target=qp.pos[self.target_idx][0],
        y_position=com_after[1],
        y_position_torso=qp.pos[self.torso_idx][1],
        y_position_target=qp.pos[self.target_idx][1],
        distance_from_origin=jp.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )

    # Teleport hit targets
    rng, rng1, rng2 = jp.random_split(state.info['rng'], 3)
    x_dist = self.target_radius + jp.random_uniform(rng1, low=5.0, high=15.0)
    y_dist = self.target_radius + self.target_y_distance * jp.random_uniform(rng1)
    ang = jp.pi * 2. * jp.random_uniform(rng2)
    target_x = qp.pos[self.target_idx][0] + x_dist * abs(jp.cos(ang))
    target_y = qp.pos[self.target_idx][1] + y_dist * jp.sin(ang)
    target_z = 1.0
    target = jp.array([target_x, target_y, target_z]).transpose()
    
    target = jp.where(target_hit, target, qp.pos[self.target_idx])
    pos = jp.index_update(qp.pos, self.target_idx, target)
    qp = qp.replace(pos=pos)
    state.info.update(rng=rng)

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
      min: 30.0
      max: 40.0
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
      min: -40.0
      max: -30.0
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
    first: "floor"
    second: "torso"
  }
  collide_include {
    first: "Mesh1"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh1"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh2"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh2"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh3"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh3"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh4"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh4"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh11"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh11"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh12"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh12"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh13"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh13"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh14"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh14"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh21"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh21"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh22"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh22"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh23"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh23"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh24"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh24"
    second: "right_shin"
  }
  collide_include {
    first: "Mesh0"
    second: "left_shin"
  }
  collide_include {
    first: "Mesh0"
    second: "right_shin"
  }
  collide_include {
    first: "Wall1"
    second: "left_shin"
  }
  collide_include {
    first: "Wall1"
    second: "right_shin"
  }
  collide_include {
    first: "Wall2"
    second: "left_shin"
  }
  collide_include {
    first: "Wall2"
    second: "right_shin"
  }

  bodies {
    name: "Target"
    colliders {
      position {
        x: 0
      }
      sphere {
        radius: 0.001
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
        x: 10.0
        y: 3.8
        z: 0
      }
      box {
        halfsize { x: 20.0 y: 0.5 z: 0.5 }
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
        y: -6.3
        z: 0
      }
      box {
        halfsize { x: 20.0 y: 0.5 z: 0.5 }
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
    name: "Mesh1" mass: 1
    colliders { 
      position { x: 7.0 y: -1.0 z: 0 }
      mesh { name: "Box1" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box1"
    path: "box.ply"
  }
  bodies {
    name: "Mesh2" mass: 1
    colliders { 
      position { x: 7.0 y: 0.2 z: 0 }
      mesh { name: "Box2" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box2"
    path: "box.ply"
  }
  bodies {
    name: "Mesh3" mass: 1
    colliders { 
      position { x: 7.0 y: 1.4 z: 0 }
      mesh { name: "Box3" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box3"
    path: "box.ply"
  }
  bodies {
    name: "Mesh4" mass: 1
    colliders { 
      position { x: 7.0 y: 2.6 z: 0 }
      mesh { name: "Box4" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box4"
    path: "box.ply"
  }


  bodies {
    name: "Mesh11" mass: 1
    colliders { 
      position { x: 14.0 y: -1.5 z: 0 }
      mesh { name: "Box11" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box11"
    path: "box.ply"
  }
  bodies {
    name: "Mesh12" mass: 1
    colliders { 
      position { x: 14.0 y: -2.7 z: 0 }
      mesh { name: "Box12" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box12"
    path: "box.ply"
  }
  bodies {
    name: "Mesh13" mass: 1
    colliders { 
      position { x: 14.0 y: -3.9 z: 0 }
      mesh { name: "Box13" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box13"
    path: "box.ply"
  }
  bodies {
    name: "Mesh14" mass: 1
    colliders { 
      position { x: 14.0 y: -5.1 z: 0 }
      mesh { name: "Box14" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box14"
    path: "box.ply"
  }

  bodies {
    name: "Mesh21" mass: 1
    colliders { 
      position { x: 21.0 y: -1.0 z: 0 }
      mesh { name: "Box21" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box21"
    path: "box.ply"
  }
  bodies {
    name: "Mesh22" mass: 1
    colliders { 
      position { x: 21.0 y: 0.2 z: 0 }
      mesh { name: "Box22" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box22"
    path: "box.ply"
  }
  bodies {
    name: "Mesh23" mass: 1
    colliders { 
      position { x: 21.0 y: 1.4 z: 0 }
      mesh { name: "Box23" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box23"
    path: "box.ply"
  }
  bodies {
    name: "Mesh24" mass: 1
    colliders { 
      position { x: 21.0 y: 2.6 z: 0 }
      mesh { name: "Box24" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box24"
    path: "box.ply"
  }  
  bodies {
    name: "Mesh0" mass: 1
    colliders { 
      position { x: 26.0 y: -2.0 z: 0 }
      mesh { name: "Box0" scale: 1 } }
      frozen { all: true }
  }
  mesh_geometries {
    name: "Box0"
    path: "box.ply"
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
