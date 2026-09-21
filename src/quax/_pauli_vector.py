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
r"""Pauli-vector representation of states and observables.

The Pauli-Liouville matrix of a channel is its matrix in the Hermitian operator basis
:math:`B_a` of :func:`~quax.n_qudit_herm_basis` (for qubits the Pauli strings), normalized so
that :math:`\mathrm{Tr}[B_a B_b] = d\,\delta_{ab}`.  The objects a channel acts on have a matching
representation: the real vector

.. math:: r_b = \mathrm{Tr}[B_b\,\rho], \qquad \rho = \frac{1}{d} \sum_b r_b B_b,

whose first entry is the trace and whose remaining entries are the (generalized) Bloch vector.
With the same vector :math:`c` for an observable, :math:`\mathrm{Tr}[O\rho] = c \cdot r / d` and
:math:`r(S\rho) = R\,r(\rho)` for the Pauli-Liouville matrix :math:`R` of a channel :math:`S`.
"""

from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp
from jax import Array

from ._operator_basis import n_qudit_herm_basis
from ._quantum_objects import DensityMatrix, Observable, StateVector


def _basis_matrices(dims: tuple[int, ...]) -> Array:
    """The Hermitian basis of ``dims`` as a ``(d**2, d, d)`` constant, safe to build under a trace."""
    # The basis is a cached NumPy constant; evaluating eagerly keeps tracers out of the cache.
    with jax.ensure_compile_time_eval():
        return jnp.asarray(n_qudit_herm_basis(tuple(dims)).matrix)


def to_pauli_vector(operator: DensityMatrix | StateVector | Observable) -> Array:
    r"""The coefficients of a state or observable in the Hermitian operator basis.

    .. math:: r_b = \mathrm{Tr}[B_b\,A],

    with :math:`B_b` the elements of :func:`~quax.n_qudit_herm_basis` for the object's dimensions,
    so that :math:`A = \frac{1}{d}\sum_b r_b B_b`, :math:`r_0 = \mathrm{Tr}[A]` and, for a qubit,
    :math:`r_{1:}` is the Bloch vector.  The convention matches :func:`~quax.superop_to_pauli_liouville`:
    ``to_pauli_vector(S @ rho) == to_pauli_liouville(S).matrix @ to_pauli_vector(rho)`` and
    ``estimate(rho, O) == to_pauli_vector(O) @ to_pauli_vector(rho) / d``.

    Ensembles broadcast; the function is jittable and differentiable.

    :param operator: A density matrix, state vector or observable (Hermitian), possibly an ensemble.
    :return: Real array of shape ``(*ensemble, d**2)``.
    """
    if isinstance(operator, StateVector):
        from ._promotion import promote_state_vector_to_density_matrix

        density = promote_state_vector_to_density_matrix(operator)
        dims, matrix = tuple(density.dims), density.matrix
    elif isinstance(operator, DensityMatrix):
        dims, matrix = tuple(operator.dims), operator.matrix
    else:
        dims_out, dims_in = operator.dims
        if tuple(dims_out) != tuple(dims_in):
            raise ValueError(f"A Pauli vector needs a square operator, got dims {operator.dims}.")
        dims, matrix = tuple(dims_out), operator.matrix
    return jnp.real(jnp.einsum("...ij,kji->...k", matrix, _basis_matrices(dims)))


def pauli_vector_to_matrix(vector: Array, dims: tuple[int, ...]) -> Array:
    r"""The matrix :math:`\frac{1}{d}\sum_b r_b B_b` with the given coefficients, ``(*ensemble, d, d)``.

    Inverse of :func:`to_pauli_vector`; used by ``DensityMatrix.from_pauli_vector`` and
    ``Observable.from_pauli_vector``.

    :param vector: Real coefficients of shape ``(*ensemble, d**2)``.
    :param dims: The qudit dimensions, e.g. ``(2, 2)``.
    """
    dims = tuple(dims)
    d = reduce(mul, dims, 1)
    basis = _basis_matrices(dims)
    coefficients = jnp.asarray(vector)
    if coefficients.shape[-1] != d * d:
        raise ValueError(f"Expected {d * d} coefficients for dims {dims}, got {coefficients.shape[-1]}.")
    return jnp.einsum("...k,kij->...ij", coefficients.astype(complex), basis) / d
