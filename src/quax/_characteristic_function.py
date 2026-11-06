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
"""Weyl characteristic functions of states and observables."""

from functools import singledispatch
from math import isqrt

import jax
import jax.numpy as jnp
from jax import Array

from ._fast_weyl_transforms import weyl_coefficients, xz_order
from ._operator_basis import weyl_phases
from ._pauli_vector import square_dims
from ._promotion import promote_state_vector_to_density_matrix
from ._quantum_objects import DensityMatrix, Observable, StateVector


def _characteristic(matrix: Array, dims: tuple[int, ...]) -> Array:
    r""":math:`\mathrm{Tr}[W_{x,z} A]` for every Weyl operator, in basis order."""
    traces = weyl_coefficients(matrix, dims).reshape(*matrix.shape[:-2], -1)
    return jnp.take(traces, xz_order(dims), axis=-1) * weyl_phases(dims)


@singledispatch
def characteristic_function(operator) -> Array:
    r"""The Weyl characteristic function :math:`\chi_\rho = \mathrm{Tr}[W_{x,z}\,\rho]`.

    For qudits of dimensions :math:`d_i`, the Weyl operators are

    .. math::

        W_{x,z} = \bigotimes_{i} \tau_{d_i}^{x_i z_i} X_i^{x_i} Z_i^{z_i},
        \qquad
        \tau_d = (-1)^d e^{i\pi/d},

    with :math:`X|j\rangle = |j + 1 \bmod d\rangle` and :math:`Z|j\rangle = e^{2\pi i j/d}|j\rangle`
    ([SWCLIFF]_ Eq. (2.1)). For qubits they are the Paulis :math:`I, X, Y, Z` and :math:`\chi`
    equals :func:`~quax.to_pauli_vector`. For odd :math:`d` they are Gross's Weyl operators
    ([HUDSON]_ Eq. (3)).
    See the :doc:`magic tutorial </examples/magic>` for background and examples.

    The :math:`D^2` values come in the order of :func:`~quax.weyl_basis_labels` and
    :func:`~quax.n_qudit_basis`, qudit 0 most significant. For :math:`d > 2`,
    :func:`~quax.n_qudit_basis` holds the plain :math:`X^x Z^z`; :func:`~quax.weyl_phases` gives the
    phases that turn them into :math:`W_{x,z}`. The state is recovered as
    :math:`\rho = \frac{1}{D} \sum_{x,z} \chi_\rho(x,z)\, W_{x,z}^\dagger`.

    Computed in :math:`O(D^2 \log D)` time without building the operator basis. Dispatches on the
    type of *operator*; ensembles broadcast; jitted and differentiable.

    :param operator: A ``DensityMatrix``, ``StateVector`` or ``Observable``, possibly an ensemble.
    :return: Complex array of shape ``(*ensemble, D**2)``.

    .. [HUDSON] Hudson's theorem for finite-dimensional quantum systems.
             Gross.
             J. Math. Phys. 47, 122107 (2006).
             https://doi.org/10.1063/1.2393152
             https://arxiv.org/abs/quant-ph/0602001
    .. [SWCLIFF] Schur-Weyl duality for the Clifford group with applications: property testing,
             a robust Hudson theorem, and de Finetti representations.
             Gross, Nezami, Walter.
             Commun. Math. Phys. 385, 1325-1393 (2021).
             https://doi.org/10.1007/s00220-021-04118-7
             https://arxiv.org/abs/1712.08628
    """
    raise TypeError(
        f"characteristic_function() does not support type {type(operator)!r}. "
        "Expected a DensityMatrix, StateVector or Observable."
    )


@characteristic_function.register(DensityMatrix)
@jax.jit
def _density_matrix_characteristic_function(operator: DensityMatrix) -> Array:
    return _characteristic(operator.matrix, tuple(operator.dims))


@characteristic_function.register(StateVector)
@jax.jit
def _state_vector_characteristic_function(operator: StateVector) -> Array:
    return characteristic_function(promote_state_vector_to_density_matrix(operator))


@characteristic_function.register(Observable)
@jax.jit
def _observable_characteristic_function(operator: Observable) -> Array:
    return _characteristic(operator.matrix, square_dims(operator))


@jax.jit
def characteristic_distribution(operator) -> Array:
    r"""The characteristic distribution :math:`\Xi_\rho(x,z) = |\chi_\rho(x,z)|^2 / D`.

    It sums to the purity :math:`\mathrm{Tr}[\rho^2]`, so for a pure state it is a probability
    distribution over the Weyl operators ([SWCLIFF]_ Eq. (3.2)). A pure stabilizer state puts
    :math:`1/D` on each of its :math:`D` stabilizers and zero elsewhere ([SWCLIFF]_ Eq. (3.3)).
    Its Rényi entropies give :func:`stabilizer_renyi_entropy`. :math:`\Xi` is Leone et al.'s
    symbol; Gross uses it for the characteristic function itself.
    See the :doc:`magic tutorial </examples/magic>` for background and examples.

    Returned in the order of :func:`characteristic_function`.

    :param operator: A ``DensityMatrix``, ``StateVector`` or ``Observable``, possibly an ensemble.
    :return: Real array of shape ``(*ensemble, D**2)``.
    """
    chi = characteristic_function(operator)
    return jnp.abs(chi) ** 2 / isqrt(chi.shape[-1])
