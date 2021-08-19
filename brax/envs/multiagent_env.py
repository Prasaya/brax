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

"""A brax environment for training and inference."""

import abc
from typing import Dict

from flax import struct
import jax
import jax.numpy as jnp

import brax
from brax.envs import Env


class MultiagentEnv(Env):
  """Wrapper to add multiagent functionality"""
  def __init__(self, num_agents: int = 1, **kwargs):
    self.num_agents = n_agents
    super(MultiagentEnv, self).__init__(**kwargs)

  @property
  def agent_observation_size(self) -> int:
    """The size of the observation vector returned in step and reset."""
    # TODO: update this!
    rng = jax.random.PRNGKey(0)
    if self.batch_size:
      rng = jax.random.split(rng, self.batch_size)
    reset_state = self.reset(rng)
    return reset_state.obs.shape[-1]

  @property
  def agent_action_size(self) -> int: 
    """The size of the action vector for each agent."""
    # TODO: update this!
    return self.sys.num_joint_dof / self.num_agents ## 
