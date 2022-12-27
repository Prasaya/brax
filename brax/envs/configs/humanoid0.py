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