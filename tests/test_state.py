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

import jax.numpy as jnp
import pytest
import qutip as qt

from quax import (
    mixed_state_matrix,
    tensor_density_matrices,
    tensor_state_vectors,
    zero_state_matrix,
    zero_state_vector,
)


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_zero_state_vector(num_qubits):
    """Check that the zero state vector is constructed correctly."""
    psi = zero_state_vector(num_qubits)
    assert psi.matrix.shape == (2**num_qubits,)
    assert psi.dims == (2,) * num_qubits
    assert jnp.allclose(psi.matrix[1:], jnp.zeros(2**num_qubits - 1))
    assert jnp.isclose(psi.matrix[0], 1.0)


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_zero_state_matrix(num_qubits):
    """Check that the zero state matrix is constructed correctly."""
    rho = zero_state_matrix(num_qubits)
    assert rho.matrix.shape == (2**num_qubits, 2**num_qubits)
    assert rho.dims == (2,) * num_qubits
    assert jnp.allclose(rho.matrix[0, 0], 1.0)
    assert jnp.allclose(rho.matrix[0, 1:], jnp.zeros(2**num_qubits - 1))
    assert jnp.allclose(rho.matrix[1:, 0], jnp.zeros(2**num_qubits - 1))
    assert jnp.allclose(rho.matrix[1:, 1:], jnp.zeros((2**num_qubits - 1, 2**num_qubits - 1)))


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_mixed_state_matrix(num_qubits):
    """Check that the mixed state matrix is constructed correctly."""
    rho = mixed_state_matrix(num_qubits)
    assert rho.matrix.shape == (2**num_qubits, 2**num_qubits)
    assert rho.dims == (2,) * num_qubits

    qt_ref = qt.maximally_mixed_dm(rho.d).full()
    assert jnp.allclose(rho.matrix, qt_ref)


@pytest.mark.parametrize("num_qubits_a,num_qubits_b", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_tensor_state_vectors(num_qubits_a, num_qubits_b):
    """Check that the tensor product of state vectors is computed correctly."""
    state_a = zero_state_vector(num_qubits_a)
    state_b = zero_state_vector(num_qubits_b)
    tensor_state = tensor_state_vectors(state_a, state_b)
    expected_dims = (2,) * (num_qubits_a + num_qubits_b)
    assert tensor_state.dims == expected_dims
    assert tensor_state.matrix.shape == (2 ** (num_qubits_a + num_qubits_b),)
    assert jnp.isclose(tensor_state.matrix[0], 1.0)
    assert jnp.allclose(tensor_state.matrix[1:], jnp.zeros(2 ** (num_qubits_a + num_qubits_b) - 1))


@pytest.mark.parametrize("num_qubits_a,num_qubits_b", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_tensor_density_matrices(num_qubits_a, num_qubits_b):
    """Check that the tensor product of density matrices is computed correctly."""
    state_a = zero_state_matrix(num_qubits_a)
    state_b = zero_state_matrix(num_qubits_b)
    tensor_state = tensor_density_matrices(state_a, state_b)
    expected_dims = (2,) * (num_qubits_a + num_qubits_b)
    assert tensor_state.dims == expected_dims
    assert tensor_state.matrix.shape == (2 ** (num_qubits_a + num_qubits_b), 2 ** (num_qubits_a + num_qubits_b))
    assert jnp.isclose(tensor_state.matrix[0, 0], 1.0)
    assert jnp.allclose(tensor_state.matrix[0, 1:], jnp.zeros(2 ** (num_qubits_a + num_qubits_b) - 1))
    assert jnp.allclose(tensor_state.matrix[1:, 0], jnp.zeros(2 ** (num_qubits_a + num_qubits_b) - 1))
    assert jnp.allclose(
        tensor_state.matrix[1:, 1:],
        jnp.zeros((2 ** (num_qubits_a + num_qubits_b) - 1, 2 ** (num_qubits_a + num_qubits_b) - 1)),
    )
