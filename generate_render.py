import os
import sys

import jax
from brax import envs
from brax.io import html
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks


def generate_render(model_path: str, env_name: str, render_output: str):
    env = envs.create(env_name=env_name)
    normalize = running_statistics.normalize
    network_factory = ppo_networks.make_ppo_networks
    ppo_network = network_factory(
        env.observation_size, env.action_size, preprocess_observations_fn=normalize
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    params = model.load_params(model_path)
    inference_fn = make_inference_fn(params)
    # create an env with auto-reset
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

    html.save_html(render_output, env.sys, [s.qp for s in rollout], True)

model_path = input('Enter model path: ')
if not os.path.exists(model_path):
    print(f"Could not find model at {model_path}")
    sys.exit(1)
env_name = input('Enter environment name: ')
if not env_name:
    env_name = "humanoidNevus"
output_path = input('Enter output path: ')

generate_render(
    model_path,
    env_name,
    output_path,
)
# generate_render(
#     "/home/nevus/rl/brax/aparams/2023-02-11 15:59:46.230149/10117120",
#     "humanoidNevus",
#     "/home/nevus/rl/brax/results/generate/render.html"
# )
