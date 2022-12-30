from datetime import datetime
import functools
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
# logging.basicConfig(level=logging.INFO)
# loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# for logger in loggers:
#    logger.setLevel(logging.INFO)


dir_var = str(datetime.now())
folder_name = "results/result" + dir_var
os.mkdir(folder_name)

"""First let's pick an environment to train an agent:"""

env_name = "humanoidNevus"
env = envs.get_environment(env_name=env_name)
state = env.reset(rng=jp.random_prngkey(seed=0))

html.save_html(os.path.join(folder_name, "initial_render.html"),
               env.sys, [state.qp], True)

# sys.exit(0)


"""# Training

Brax provides out of the box the following training algorithms:

* [Proximal policy optimization](https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py)

Trainers take as input an environment function and some hyperparameters, and return an inference function to operate the environment.
"""

# Hyperparameters for humanoid.
train_fn = functools.partial(ppo.train,
                             num_timesteps=50_000_000,
                             episode_length=1000,
                             action_repeat=1,
                             num_envs=2048,
                             learning_rate=3e-4,
                             entropy_cost=1e-3,
                             discounting=0.99,
                             unroll_length=20,
                             batch_size=512,
                             num_minibatches=4,
                             normalize_observations=True,
                             reward_scaling=5.,
                             num_evals=20,
                             num_updates_per_batch=8,
                             )

max_y = 130000
min_y = {'reacher': -100, 'pusher': -150}.get(env_name, 0)

xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    # clear_output(wait=True)
    plt.xlim([0, train_fn.keywords['num_timesteps']])
    plt.ylim([min_y, max_y])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(xdata, ydata)
    plt.show()
    plt.savefig(os.path.join(folder_name, "graph.png"))
    print("Environment steps : ", num_steps,
          "      Reward : ", metrics['eval/episode_reward'])


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

"""The trainers return an inference function, parameters, and the final set of metrics gathered during evaluation.

# Saving and Loading Policies

Brax can save and load trained policies:
"""

model.save_params('/tmp/params', params)
params = model.load_params('/tmp/params')
inference_fn = make_inference_fn(params)

"""# Visualizing a Policy's Behavior

We can use the policy to generate a rollout for visualization:
"""

# @title Visualizing a trajectory of the learned inference function

# create an env with auto-reset
env = envs.create(env_name=env_name)
jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
rng = jax.random.PRNGKey(seed=0)
state = jit_env_reset(rng=rng)
for _ in range(1000):
    rollout.append(state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)

# with open("render_output.html", "rw") as f:
html.save_html(os.path.join(folder_name, "render.html"),
               env.sys, [s.qp for s in rollout], True)
# HTML(html.render(env.sys, [s.qp for s in rollout]))
