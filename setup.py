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

"""setup.py for Brax.

Install for development:

  pip intall -e .
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name="brax",
    version="0.0.3",
    description=("A differentiable physics engine written in JAX."),
    author="Brax Authors",
    author_email="no-reply@google.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/google/brax",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    scripts=["bin/learn"],
    install_requires=[
        # "absl-py",
        "clu<=0.0.3",
        "dataclasses",
        "flax<=0.3.4",
        "gym<=0.18.3",
        # "grpcio",
        "jax<=0.2.16",
        "jaxlib<=0.1.68",
        "numpy<=1.19.2",
        "optax<=0.0.8",
        # TODO: restore this once tfp-nightly and tensorflow are compatible
        # breakage caused by https://github.com/tensorflow/probability/commit/fdbdece116a98e101420ce38e8a45aa1e7e5656f
        "tfp-nightly[jax]<=0.13.0.dev20210422",
        "tensorflow<=2.4.2",
        "absl-py~=0.10",
        "flatbuffers~=1.12.0",
        "gast==0.3.3",
        # "siz==1.15.0",
        "protobuf<3.20,>=3.9.2",
        "zipp>=3.1.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="JAX reinforcement learning rigidbody physics",
)
