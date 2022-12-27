from datetime import datetime
import functools
import time
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys

import brax

from brax import envs
from brax import jumpy as jp
from brax.io import html
from brax.io import model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
import logging

dir_var = str(datetime.now())
folder_name = "results/result" + dir_var
os.mkdir(folder_name)


def ant_mountain(count: int, cutoff: int) -> brax.System:
  # config = envs.create('humanoidNevus').sys.config
  env_name = "humanoidNevus"  
  env = envs.get_environment(env_name=env_name)
  config = env.sys.config

  repeat = count - 1

  # clone the ant
  for lst in (config.bodies, config.joints, config.actuators):
    for obj in list(lst):
      if obj.name == 'floor':
        continue
      for i in range(repeat):
        new_obj = lst.add()
        new_obj.CopyFrom(obj)
        for attr in ('name', 'joint', 'parent', 'child'):
          if hasattr(new_obj, attr):
            setattr(new_obj, attr, f'{getattr(new_obj, attr)}_{i}')

  # sprinkle ants in a delicate spiral
  default = config.defaults.add()
  old_default = config.defaults
  angle = default.angles.add(name="left_knee_0")
  angle.x = -25.0
  angle.y = 0
  angle.z = 0
  angle2 = default.angles.add(name="right_knee_0")
  angle2.x = -25.0
  angle2.y = 0
  angle2.z = 0
  # angle3 = default.angles.add(name="left_knee")
  # angle3.x = -25.0
  # angle3.y = 0
  # angle3.z = 0
  # angle4 = default.angles.add(name="right_knee")
  # angle4.x = -25.0
  # angle4.y = 0
  # angle4.z = 0
  # print(default)
  # print('Before:', old_default)
  # default = config.defaults.add()
  # print('Middle:', default)
  # for i in range(repeat):  
  #   # angle = default.angles.add(name="left_knee_{i}")
  #   # angle.x = -25.0
  #   # angle = default.angles.add(name=f"right_knee{i}")
  #   # angle.x = -25.0
  #   qp = default.qps.add(name=f'torso')
  #   # qp.pos.x = jnp.sin(i * jnp.pi / 2)
  #   # qp.pos.y = jnp.cos(i * jnp.pi / 2)
  #   qp.pos.x = 23
  #   qp.pos.y = 1.0
  #   qp.pos.z = 200.0
  # print('After:', default)

  # turn on all collisions:
  del config.collide_include[:]

  config.collider_cutoff = cutoff
  # print(config)
  return brax.System(config)


num_ants = 2 #@param {type:"slider", min:1, max:25, step:1}
cutoff = 0 #@param {type:"slider", min:0, max:5000, step:1}
sys = ant_mountain(num_ants, cutoff)
qps = [sys.default_qp()]
act = jnp.array([0] * 8 * (num_ants))

for _ in range(5):
  print('iteration: ', _)
  qp, _ = sys.step(qps[-1], act)
  qps.append(qp)

# HTML(html.render(sys, qps))

html.save_html(os.path.join(folder_name, "initial_render.html"), sys, qps, True)

