from datetime import datetime
import functools
import time
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
  config = envs.create('ant').sys.config
  repeat = count - 1

  # clone the ant
  for lst in (config.bodies, config.joints, config.actuators):
    for obj in list(lst):
      if obj.name == 'Ground':
        continue
      for i in range(repeat):
        new_obj = lst.add()
        new_obj.CopyFrom(obj)
        for attr in ('name', 'joint', 'parent', 'child'):
          if hasattr(new_obj, attr):
            setattr(new_obj, attr, f'{getattr(new_obj, attr)}_{i}')

  # sprinkle ants in a delicate spiral
  # print('config.defaults: ', config.defaults)
  default = config.defaults.add()
  # print('Before: ', default)
  for i in range(repeat):  
    qp = default.qps.add(name=f'$ Torso_{i}')
    print('Before:', (qp.pos))
    qp.pos.x = jnp.sin(i * jnp.pi / 2)
    qp.pos.x = 23
    qp.pos.y = jnp.cos(i * jnp.pi / 2)
    qp.pos.z = (i + 1) * 2
    print(qp)
  # print('After', default)
  # turn on all collisions:
  del config.collide_include[:]

  config.collider_cutoff = cutoff
  # print('Config:', config)
  return brax.System(config)


num_ants = 2 #@param {type:"slider", min:1, max:25, step:1}
cutoff = 0 #@param {type:"slider", min:0, max:5000, step:1}
sys = ant_mountain(num_ants, cutoff)
qps = [sys.default_qp()]
act = jnp.array([0] * 8 * (num_ants))

for _ in range(2):
  qp, _ = sys.step(qps[-1], act)
  qps.append(qp)

# HTML(html.render(sys, qps))

html.save_html(os.path.join(folder_name, "initial_render.html"), sys, qps, True)

# def testSpeed(num_ants: int, cutoff: int) -> float:
#   batch_size = 1024
#   sys = ant_mountain(num_ants, cutoff)
#   num_devices = len(jax.devices())
#   num_per_device = batch_size // num_devices
#   act = jnp.array([[0] * 8 * (num_ants)] * num_per_device)
#   qp = sys.default_qp()
#   qp = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *[qp] * num_per_device)
#   qp = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *[qp] * num_devices)

#   @functools.partial(jax.pmap, axis_name='i')
#   def run_sys(qp):
#     qp, _ = jax.vmap(sys.step)(qp, act)
#     return qp

#   sps = []
#   for seed in range(6):
#     jax.device_put(qp)
#     t = time.time()
#     run_sys(qp).pos.block_until_ready()
#     sps.append(batch_size / (time.time() - t))

#   mean_sps = jnp.mean(jnp.array(sps[1:]))
#   return mean_sps

# allpairs_sps = []
# cutoff_sps = []

# for i in range(1, 11):
#   allpairs_sps.append(testSpeed(i, 0))
#   cutoff_sps.append(testSpeed(i, i * 9))

# plt.plot(range(1, len(allpairs_sps) + 1), allpairs_sps, label = "All Pairs")
# plt.plot(range(1, len(cutoff_sps) + 1), cutoff_sps, label = "Near Neighbors")
# plt.legend()
# plt.xlabel('# Ants')
# plt.ylabel('Steps / second')
# plt.yscale('log')
# plt.title(f'Near Neighbors Cull vs. All Pairs.\nDevice: {jax.devices()[0].device_kind}')
# plt.show()
# plt.savefig(os.path.join(folder_name, "graph.png"))

