# Copyright 2026 Rigetti & Co, LLC.
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
"""Magic (non-stabilizerness) measures of quantum states."""

from functools import singledispatch
from math import isqrt

import jax
import jax.numpy as jnp
from jax import Array

from ._characteristic_function import characteristic_function
from ._quantum_objects import DensityMatrix, StateVector


@singledispatch
def stabilizer_norm(state) -> Array:
    r"""The stabilizer norm :math:`\mathcal{D}(\rho)` of a state.

    .. math:: \mathcal{D}(\rho) = \frac{1}{D}\sum_{x,z} |\mathrm{Tr}[W_{x,z}\,\rho]|

    The sum runs over the :math:`D^2` Weyl operators :math:`W_{x,z}` of
    :func:`characteristic_function`, which for qubits are the Pauli strings. [CATMAGIC]_ Lemma 1,
    Eq. (19) introduces it as the st-norm :math:`\|\rho\|_{st} = \sum_j |a_j|` of the Pauli
    expansion :math:`\rho = \sum_j a_j P_j`, and [ROMFTQC]_ Eq. (22) writes it as above. Only
    :math:`|\mathrm{Tr}[W\rho]|` enters, so the phase convention of the Weyl operators does not
    matter.

    Both papers define it for qubits. For qudits of any dimension it is extended here by replacing
    the Pauli strings with the :math:`D^2` Weyl operators. The properties below carry over unchanged: a
    pure stabilizer state is a uniform combination of the :math:`D` Weyl operators in its
    stabilizer group, each with :math:`|\mathrm{Tr}[W\rho]| = 1`, and Weyl operators factor across
    subsystems.

    How to read it:

    * It equals 1 on every pure stabilizer state and is at most 1 on every mixture of stabilizer
      states, because it is convex ([CATMAGIC]_ Lemma 1; [ROMFTQC]_ D1). A value above 1 therefore
      certifies that the state is not a stabilizer state ([ROMFTQC]_ D2). The converse fails beyond
      one qubit: many magic states score at most 1. For a single qubit it is exact, marking out the
      stabilizer octahedron in the Bloch ball.
    * It is a lower bound on the robustness of magic :math:`\mathcal{R}(\rho)`, the least
      :math:`\ell_1` norm of a decomposition of :math:`\rho` into stabilizer states, which governs
      the cost of classical simulation: :math:`\mathcal{D}(\rho) \le \mathcal{R}(\rho)`, and for
      :math:`n` qubits :math:`(\mathcal{D}(\rho) - 2^{-n}) / (1 - 2^{-n}) \le \mathcal{R}(\rho)`
      ([ROMFTQC]_ D3, D3*, Eq. (23)).
    * It is multiplicative,
      :math:`\mathcal{D}(\rho_1 \otimes \rho_2) = \mathcal{D}(\rho_1)\,\mathcal{D}(\rho_2)`
      ([CATMAGIC]_ Eq. (21); [ROMFTQC]_ D4, Eq. (35)).
    * The maximally mixed state gives :math:`1/D`; the single-qubit T state
      :math:`(|0\rangle + e^{i\pi/4}|1\rangle)/\sqrt{2}` gives :math:`(1 + \sqrt{2})/2`.

    Computed from :func:`characteristic_function` in :math:`O(D^2 \log D)` time. Dispatches on the
    type of *state*; ensembles broadcast; the function is jitted and differentiable.

    :param state: A ``StateVector`` or ``DensityMatrix``, possibly an ensemble.
    :return: Real array of shape ``(*ensemble,)``.

    .. [CATMAGIC] Catalysis and activation of magic states in fault-tolerant architectures.
             Campbell.
             Phys. Rev. A 83, 032317 (2011).
             https://doi.org/10.1103/PhysRevA.83.032317
             https://arxiv.org/abs/1010.0104
    .. [ROMFTQC] Application of a resource theory for magic states to fault-tolerant quantum
             computing.
             Howard, Campbell.
             Phys. Rev. Lett. 118, 090501 (2017).
             https://doi.org/10.1103/PhysRevLett.118.090501
             https://arxiv.org/abs/1609.07488
    """
    raise TypeError(
        f"stabilizer_norm() does not support type {type(state)!r}. Expected a StateVector or DensityMatrix."
    )


@stabilizer_norm.register(DensityMatrix)
@jax.jit
def _density_matrix_stabilizer_norm(state: DensityMatrix) -> Array:
    chi = characteristic_function(state)
    # |chi| directly: sqrt(|chi|**2) has a NaN gradient wherever chi is zero.
    return jnp.sum(jnp.abs(chi), axis=-1) / isqrt(chi.shape[-1])


@stabilizer_norm.register(StateVector)
@jax.jit
def _state_vector_stabilizer_norm(state: StateVector) -> Array:
    chi = characteristic_function(state)
    # |chi| directly: sqrt(|chi|**2) has a NaN gradient wherever chi is zero.
    return jnp.sum(jnp.abs(chi), axis=-1) / isqrt(chi.shape[-1])
