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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Any, Callable, Dict, Optional

from absl import logging
import flax
import jax
import jax.numpy as jnp
from brax import envs
from brax.training import distribution
from brax.training import env
from brax.training import networks
from brax.training import normalization


def compute_gae(truncation: jnp.ndarray,
                rewards: jnp.ndarray,
                values: jnp.ndarray,
                bootstrap_value: jnp.ndarray,
                lambda_: float = 1.0,
                discount: float = 0.99):
  r"""Calculates the Generalized Advantage Estimation (GAE).

  Args:
    truncation: A float32 tensor of shape [T, B] with truncation signal.
    rewards: A float32 tensor of shape [T, B] containing rewards generated by
      following the behaviour policy.
    values: A float32 tensor of shape [T, B] with the value function estimates
      wrt. the target policy.
    bootstrap_value: A float32 of shape [B] with the value function estimate at
      time T.
    lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
      lambda_=1.
    discount: TD discount.

  Returns:
    A float32 tensor of shape [T, B]. Can be used as target to
      train a baseline (V(x_t) - vs_t)^2.
    A float32 tensor of shape [T, B] of advantages.
  """

  truncation_mask = 1 - truncation
  # Append bootstrapped value to get [v1, ..., v_t+1]
  values_t_plus_1 = jnp.concatenate(
      [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
  deltas = rewards + discount * values_t_plus_1 - values
  deltas *= truncation_mask

  acc = jnp.zeros_like(bootstrap_value)
  vs_minus_v_xs = []

  def compute_vs_minus_v_xs(carry, target_t):
    lambda_, acc = carry
    truncation_mask, delta = target_t
    acc = delta + discount * truncation_mask * lambda_ * acc
    return (lambda_, acc), (acc)

  (_, _), (vs_minus_v_xs) = jax.lax.scan(compute_vs_minus_v_xs, (lambda_, acc),
                                         (truncation_mask, deltas),
                                         length=int(truncation_mask.shape[0]),
                                         reverse=True)
  # Add V(x_s) to get v_s.
  vs = jnp.add(vs_minus_v_xs, values)

  vs_t_plus_1 = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)],
                                axis=0)
  advantages = (rewards + discount * vs_t_plus_1 - values) * truncation_mask
  return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


@flax.struct.dataclass
class StepData:
  """Contains data for one environment step."""
  obs: jnp.ndarray
  rewards: jnp.ndarray
  dones: jnp.ndarray
  actions: jnp.ndarray
  logits: jnp.ndarray


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  optimizer: flax.optim.Optimizer
  key: jnp.ndarray
  normalizer_params: Any


def compute_ppo_loss(
        models: Dict[str, Any],
        data: StepData,
        rng: jnp.ndarray,
        parametric_action_distribution: distribution.ParametricDistribution,
        policy_apply: Any,
        value_apply: Any,
        entropy_cost: float = 1e-4,
        discounting: float = 0.9,
        reward_scaling: float = 1.0,
        lambda_: float = 0.95,
        ppo_epsilon: float = 0.3):
  """Computes PPO loss."""
  policy_params, value_params = models['policy'], models['value']
  policy_logits = policy_apply(policy_params, data.obs[:-1])
  baseline = value_apply(value_params, data.obs)
  baseline = jnp.squeeze(baseline, axis=-1)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = baseline[-1]
  baseline = baseline[:-1]

  # At this point, we have unroll length + 1 steps. The last step is only used
  # as bootstrap value, so it's removed.

  # already removed at data generation time
  # actions = actions[:-1]
  # logits = logits[:-1]

  rewards = data.rewards[1:] * reward_scaling
  truncation = data.dones[1:]

  target_action_log_probs = parametric_action_distribution.log_prob(
      policy_logits, data.actions)
  behaviour_action_log_probs = parametric_action_distribution.log_prob(
      data.logits, data.actions)

  vs, advantages = compute_gae(
      truncation=truncation,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=lambda_,
      discount=discounting)
  rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
  surrogate_loss1 = rho_s * advantages
  surrogate_loss2 = jnp.clip(rho_s, 1 - ppo_epsilon,
                             1 + ppo_epsilon) * advantages

  policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

  # Value function loss
  v_error = vs - baseline
  v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

  # Entropy reward
  entropy = jnp.mean(
      parametric_action_distribution.entropy(policy_logits, rng))
  entropy_loss = entropy_cost * -entropy

  return policy_loss + v_loss + entropy_loss, {
      'total_loss': policy_loss + v_loss + entropy_loss,
      'policy_loss': policy_loss,
      'v_loss': v_loss,
      'entropy_loss': entropy_loss
  }


def train(
    environment_fn: Callable[..., envs.Env],
    num_timesteps,
    episode_length: int,
    num_agents: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate=1e-4,
    entropy_cost=1e-4,
    discounting=0.9,
    seed=0,
    unroll_length=10, # what is the purpose of unroll_length?
    batch_size=32,
    num_minibatches=16,
    num_update_epochs=2,
    log_frequency=10,
    normalize_observations=False,
    reward_scaling=1.,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
  """PPO training."""
  assert num_agents * batch_size * num_minibatches % num_envs == 0
  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(), process_count, process_id, local_device_count,
      local_devices_to_use)

  key = jax.random.PRNGKey(seed)
  key, key_models, key_env, key_eval = jax.random.split(key, 4)
  # Make sure every process gets a different random key, otherwise they will be
  # doing identical work.
  key_env = jax.random.split(key_env, process_count)[process_id]
  key = jax.random.split(key, process_count)[process_id]
  # key_models should be the same, so that models are initialized the same way
  # for different processes

  core_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_envs // local_devices_to_use // process_count,
      episode_length=episode_length)
  key_envs = jax.random.split(key_env, local_devices_to_use)
  tmp_env_states = []
  for key in key_envs: # device splitting, not batch_size
    first_state, step_fn = env.wrap(core_env, key) # a set of first_states from current device batch
    tmp_env_states.append(first_state)

  first_state = jax.tree_multimap(lambda *args: jnp.stack(args),
                                  *tmp_env_states)

  core_eval_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_eval_envs,
      episode_length=episode_length)
  eval_first_state, eval_step_fn = env.wrap(core_eval_env, key_eval)

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=core_env.agent_action_size)

  policy_model, value_model = networks.make_models(
      parametric_action_distribution.param_size,
      core_env.agent_observation_size)
  key_policy, key_value = jax.random.split(key_models)

  optimizer_def = flax.optim.Adam(learning_rate=learning_rate)
  optimizer = optimizer_def.create({'policy': policy_model.init(key_policy),
                                    'value': value_model.init(key_value)})
  optimizer = normalization.bcast_local_devices(
      optimizer, local_devices_to_use)

  normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
      normalization.create_observation_normalizer(
          core_env.agent_observation_size, normalize_observations,
          num_leading_batch_dims=2, pmap_to_devices=local_devices_to_use))

  key_debug = jax.random.PRNGKey(seed + 666)

  loss_fn = functools.partial(
      compute_ppo_loss,
      parametric_action_distribution=parametric_action_distribution,
      policy_apply=policy_model.apply,
      value_apply=value_model.apply,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling)

  grad_loss = jax.grad(loss_fn, has_aux=True)

  def do_one_step_eval(carry, unused_target_t):
    # TODO: modify eval_step_fn to apply to n agent states
    state, policy_params, normalizer_params, key = carry
    gather_obs, gather_rew, gather_done, device_batchsize = reshape_state(state.core)

    key, key_sample = jax.random.split(key)
    # TODO: Make this nicer ([0] comes from pmapping).
    obs = obs_normalizer_apply_fn(
        jax.tree_map(lambda x: x[0], normalizer_params), gather_obs)
    logits = policy_model.apply(policy_params, obs)
    actions = parametric_action_distribution.sample(logits, key_sample)
    ungathered_actions = ungather_actions(actions, device_batchsize)
    nstate = eval_step_fn(state, ungathered_actions)
    return (nstate, policy_params, normalizer_params, key), ()

  @jax.jit
  def run_eval(state, key, policy_params, normalizer_params):
    policy_params = jax.tree_map(lambda x: x[0], policy_params)
    (state, _, _, key), _ = jax.lax.scan(
        do_one_step_eval, (state, policy_params, normalizer_params, key), (),
        length=episode_length // action_repeat)
    return state, key

  def gather_state(state_core):
    # collapse agents into the batch
    # assumption: shape is (device_batchsize, obs)
    device_batchsize = state_core.obs.shape[0]
    gather_obs = jnp.reshape(state_core.obs, newshape=(device_batchsize*self.num_agents, -1))
    gather_rew = jnp.reshape(state_core.reward, newshape=(device_batchsize*self.num_agents, -1))
    gather_done = jnp.reshape(state_core.done, newshape=(device_batchsize*self.num_agents, -1))
    return gather_obs, gather_rew, gather_done, device_batchsize

  def ungather_actions(actions, device_batchsize):
    ungathered_actions = jnp.reshape(actions, newshape=(device_batchsize, self.num_agents, -1))
    return ungathered_actions

  def do_one_step(carry, unused_target_t):
    # MODIFY DATA BY RESHAPE INTO BATCH x NUM_AGENTS
    state, normalizer_params, policy_params, key = carry
    gather_obs, gather_rew, gather_done, device_batchsize = reshape_state(state.core)

    key, key_sample = jax.random.split(key)
    normalized_obs = obs_normalizer_apply_fn(normalizer_params, gather_obs)
    logits = policy_model.apply(policy_params, normalized_obs)
    actions = parametric_action_distribution.sample_no_postprocessing(
        logits, key_sample)
    postprocessed_actions = parametric_action_distribution.postprocess(actions)
    ungathered_actions = ungather_actions(actions, device_batchsize)
    nstate = step_fn(state, ungathered_actions)

    data = StepData(obs=gather_obs,
                    rewards=gather_rew,
                    dones=gather_done,
                    actions=actions,
                    logits=logits)
    return (nstate, normalizer_params, policy_params, key), data

  def generate_unroll(carry, unused_target_t):
    state, normalizer_params, policy_params, key = carry
    (state, _, _, key), data = jax.lax.scan(
        do_one_step, (state, normalizer_params, policy_params, key), (),
        length=unroll_length)
    gather_obs, gather_rew, gather_done, device_batchsize = reshape_state(state.core)

    data = data.replace(
        obs=jnp.concatenate(
            [data.obs, jnp.expand_dims(gather_obs, axis=0)]),
        rewards=jnp.concatenate(
            [data.rewards, jnp.expand_dims(gather_rew, axis=0)]),
        dones=jnp.concatenate(
            [data.dones, jnp.expand_dims(gather_done, axis=0)]))
    return (state, normalizer_params, policy_params, key), data

  def update_model(carry, data):
    optimizer, key = carry
    key, key_loss = jax.random.split(key)
    loss_grad, metrics = grad_loss(optimizer.target, data, key_loss)
    loss_grad = jax.lax.pmean(loss_grad, axis_name='i')
    optimizer = optimizer.apply_gradient(loss_grad)
    return (optimizer, key), metrics

  def minimize_epoch(carry, unused_t):
    optimizer, data, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)
    permutation = jax.random.permutation(key_perm, data.obs.shape[1])

    def convert_data(data, permutation):
      data = jnp.take(data, permutation, axis=1, mode='clip')
      data = jnp.reshape(
          data, [data.shape[0], num_minibatches, -1] + list(data.shape[2:]))
      data = jnp.swapaxes(data, 0, 1)
      return data
    ndata = jax.tree_map(lambda x: convert_data(x, permutation), data)
    (optimizer, _), metrics = jax.lax.scan(
        update_model, (optimizer, key_grad), ndata, length=num_minibatches)
    return (optimizer, data, key), metrics

  def run_epoch(carry, unused_t): # TODO: figure out how batch size fits in here
    training_state, state = carry
    key_minimize, key_generate_unroll, new_key = jax.random.split(
        training_state.key, 3)
    (state, _, _, _), data = jax.lax.scan(
        generate_unroll, (state, training_state.normalizer_params,
                          training_state.optimizer.target['policy'],
                          key_generate_unroll), (),
        length=batch_size * num_minibatches // num_envs)
    # make unroll first
    data = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    data = jax.tree_map(
        lambda x: jnp.reshape(x, [x.shape[0], -1] + list(x.shape[3:])), data)

    # Update normalization params and normalize observations.
    normalizer_params = obs_normalizer_update_fn(
        training_state.normalizer_params, data.obs[:-1])
    data = data.replace(
        obs=obs_normalizer_apply_fn(normalizer_params, data.obs))

    (optimizer, _, _), metrics = jax.lax.scan(
        minimize_epoch, (training_state.optimizer, data, key_minimize),
        (), length=num_update_epochs)

    new_training_state = TrainingState(optimizer=optimizer,
                                       normalizer_params=normalizer_params,
                                       key=new_key)
    return (new_training_state, state), metrics

  num_epochs = num_timesteps // (
      batch_size * unroll_length * num_minibatches * action_repeat)

  def _minimize_loop(training_state, state):
    (training_state, state), losses = jax.lax.scan(
        run_epoch, (training_state, state), (),
        length=num_epochs // log_frequency)
    losses = jax.tree_map(jnp.mean, losses)
    return (training_state, state), losses

  minimize_loop = jax.pmap(_minimize_loop, axis_name='i')

  training_state = TrainingState(
      optimizer=optimizer,
      key=jnp.stack(jax.random.split(key, local_devices_to_use)),
      normalizer_params=normalizer_params)
  training_walltime = 0
  eval_walltime = 0
  sps = 0
  eval_sps = 0
  losses = {}
  state = first_state
  metrics = {}

  for it in range(log_frequency + 1):
    logging.info('starting iteration %s %s', it, time.time() - xt)
    t = time.time()

    if process_id == 0:
      eval_state, key_debug = (
          run_eval(eval_first_state, key_debug,
                   training_state.optimizer.target['policy'],
                   training_state.normalizer_params))
      eval_state.total_episodes.block_until_ready()
      eval_walltime += time.time() - t
      eval_sps = (episode_length * eval_first_state.core.reward.shape[0] /
                  (time.time() - t))
      metrics = dict(
          dict({f'eval/episode_{name}': value / eval_state.total_episodes
                for name, value in eval_state.total_metrics.items()}),
          **dict({
              'eval/total_episodes': eval_state.total_episodes,
              'speed/sps': sps,
              'speed/eval_sps': eval_sps,
              'speed/training_walltime': training_walltime,
              'speed/eval_walltime': eval_walltime,
              'speed/timestamp': training_walltime,
              'losses/total_loss': jnp.mean(losses.get('total_loss', 0)),
              'losses/policy_loss': jnp.mean(losses.get('policy_loss', 0)),
              'losses/value_loss': jnp.mean(losses.get('v_loss', 0)),
              'losses/entropy_loss': jnp.mean(losses.get('entropy_loss',
                                                         0))}))
      logging.info(metrics)
      if progress_fn:
        progress_fn(int(training_state.normalizer_params[0][0]) * action_repeat,
                    metrics)

    if it == log_frequency:
      break

    t = time.time()
    previous_step = training_state.normalizer_params[0][0]
    # optimization
    (training_state, state), losses = minimize_loop(training_state, state)
    jax.tree_map(lambda x: x.block_until_ready(), losses)
    sps = ((training_state.normalizer_params[0][0] - previous_step) /
           (time.time() - t)) * action_repeat
    training_walltime += time.time() - t

  # To undo the pmap.
  normalizer_params = jax.tree_map(lambda x: x[0],
                                   training_state.normalizer_params)
  policy_params = jax.tree_map(lambda x: x[0],
                               training_state.optimizer.target['policy'])

  logging.info('total steps: %s', normalizer_params[0] * action_repeat)

  _, inference = make_params_and_inference_fn(core_env.agent_observation_size,
                                              core_env.agent_action_size,
                                              normalize_observations)
  params = normalizer_params, policy_params

  if process_count > 1: 
    # Make sure all processes stay up until the end of main.
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    assert x[0] == jax.device_count()

  return (inference, params, metrics)

def make_params_and_inference_fn(agent_observation_size, agent_action_size,
                                 normalize_observations):
  """Creates params and inference function for the PPO agent."""
  obs_normalizer_params, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
      normalize_observations)
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=agent_action_size)
  policy_model, _ = networks.make_models(
      parametric_action_distribution.param_size,
      agent_observation_size)

  def inference_fn(params, obs, key):
    normalizer_params, policy_params = params
    obs = obs_normalizer_apply_fn(normalizer_params, obs)
    action = parametric_action_distribution.sample(
        policy_model.apply(policy_params, obs), key)
    return action

  params = (obs_normalizer_params, policy_model.init(jax.random.PRNGKey(0)))
  return params, inference_fn
