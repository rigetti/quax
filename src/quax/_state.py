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
import jax
import jax.numpy as jnp

from ._quantum_objects import DensityMatrix, StateVector


@jax.jit(static_argnames=("n_qubits",))
def zero_state_vector(n_qubits: int) -> StateVector:
    """
    Construct a vector corresponding to ``|0>``.

    :param n_qubits: The number of qubits.
    :return: The state vector ``|000...0>`` for `n_qubits`.
    """
    state_matrix = jnp.zeros((2**n_qubits), complex)
    state_matrix = state_matrix.at[0].set(complex(1.0, 0))
    return StateVector.from_matrix(state_matrix, (2,) * n_qubits, 0)


@jax.jit(static_argnames=("n_qubits",))
def zero_state_matrix(n_qubits: int) -> DensityMatrix:
    """
    Construct a matrix corresponding to ``|0><0|``.

    :param n_qubits: The number of qubits.
    :return: The state matrix ``|000...0><000...0|`` for `n_qubits`.
    """
    state_matrix = jnp.zeros((2**n_qubits, 2**n_qubits), complex)
    state_matrix = state_matrix.at[0, 0].set(complex(1.0, 0))
    return DensityMatrix.from_matrix(state_matrix, (2,) * n_qubits, 0)


@jax.jit(static_argnames=("n_qubits",))
def mixed_state_matrix(n_qubits: int) -> DensityMatrix:
    """
    Construct a matrix corresponding to the maximally mixed state.

    :param n_qubits: The number of qubits.
    :return: The state matrix  ``I / d`` where ``d = 2**n_qubits``.
    """
    d = 2**n_qubits
    state_matrix = jnp.eye(d, dtype=complex) / d
    return DensityMatrix.from_matrix(state_matrix, (2,) * n_qubits, 0)


@jax.jit
def tensor_state_vectors(state_a: StateVector, state_b: StateVector) -> StateVector:
    """
    Compute the tensor product of two state vectors.

    :param state_a: The first state vector.
    :param state_b: The second state vector.
    :return: The tensor product state vector.
    """
    new_data = jnp.kron(state_a.matrix, state_b.matrix)
    new_dims = state_a.dims + state_b.dims
    return StateVector.from_matrix(new_data, new_dims, 0)


@jax.jit
def tensor_density_matrices(state_a: DensityMatrix, state_b: DensityMatrix) -> DensityMatrix:
    """
    Compute the tensor product of two density matrices.

    :param state_a: The first density matrix.
    :param state_b: The second density matrix.
    :return: The tensor product density matrix.
    """
    new_data = jnp.kron(state_a.matrix, state_b.matrix)
    new_dims = state_a.dims + state_b.dims
    return DensityMatrix.from_matrix(new_data, new_dims, 0)
