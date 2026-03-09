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
import numpy as np
import pytest
import qutip as qt

from quax import (
    Choi,
    DensityMatrix,
    StateVector,
    Unitary,
    choi_to_kraus,
    choi_to_pauli_liouville,
    choi_to_superop,
    fidelity,
    kraus_to_choi,
    pauli_liouville_to_choi,
    process_fidelity,
    random_choi,
    random_density_matrix,
    random_state_vector,
    random_unitary,
    superop_to_choi,
    tensor_choi,
    tensor_density_matrix,
    tensor_kraus,
    tensor_pauli_liouville,
    tensor_state_vector,
    tensor_superop,
    tensor_unitary,
    unitary_entanglement_fidelity,
)


def qt_tensor(a, b):
    """Helper to tensor QuTiP Qobjs with broadcasting support."""
    # Accept scalar Qobj or numpy array(dtype=object) of Qobj
    A = np.asarray(a, dtype=object)
    B = np.asarray(b, dtype=object)

    A, B = np.broadcast_arrays(A, B)
    out_shape = A.shape

    results = [qt.tensor(x, y).full() for x, y in zip(A.ravel(), B.ravel())]
    dense = np.stack(results, axis=0)
    dense = dense.reshape(out_shape + dense.shape[1:])  # out_shape + matrix_shape
    return jnp.asarray(dense)  # numeric ndarray (complex)


def qt_super_tensor(a, b):
    """Helper to super_tensor QuTiP Qobjs with broadcasting support (for Choi)."""
    # Accept scalar Qobj or numpy array(dtype=object) of Qobj
    A = np.asarray(a, dtype=object)
    B = np.asarray(b, dtype=object)

    A, B = np.broadcast_arrays(A, B)
    out_shape = A.shape

    results = [qt.to_choi(qt.super_tensor(qt.to_super(x), qt.to_super(y))).full() for x, y in zip(A.ravel(), B.ravel())]
    dense = np.stack(results, axis=0)
    dense = dense.reshape(out_shape + dense.shape[1:])  # out_shape + matrix_shape
    return jnp.asarray(dense)  # numeric ndarray (complex)


@pytest.mark.parametrize("seed", [58, 3854, 2047, 475])
@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("size_a, size_b", [((), ()), ((3,), ()), ((3,), (3,)), ((3, 4), ()), ((3, 4), (3, 4))])
def test_tensor_superoperators(seed, num_qubits, size_a, size_b):
    """Test tensor product for all representations."""
    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)
    d = 2**num_qubits
    kraus_rank = d

    # Generate two random channels
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    tensor_dims = ((2,) * num_qubits * 2, (2,) * num_qubits * 2)
    choi_a = random_choi(dims=dims, rank=kraus_rank, key=key1, size=size_a)
    choi_b = random_choi(dims=dims, rank=kraus_rank, key=key2, size=size_b)

    # Compute the reference result using QuTiP
    qobj_a = choi_a._to_qobj()
    qobj_b = choi_b._to_qobj()

    ensemble_size = jnp.broadcast_shapes(size_a, size_b)

    # Use qt_super_tensor helper for broadcasting
    qobj_tensored_ref = Choi.from_matrix(qt_super_tensor(qobj_a, qobj_b), tensor_dims)
    assert qobj_tensored_ref.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"

    # Tensor product chois
    choi_composed = tensor_choi(choi_a, choi_b)
    assert choi_composed.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    fid_choi = process_fidelity(choi_composed, qobj_tensored_ref)
    assert jnp.allclose(fid_choi, 1.0, atol=1e-6), "Composed Choi operators don't match"

    # Tensor Superops
    super_a = choi_to_superop(choi_a)
    super_b = choi_to_superop(choi_b)
    super_composed = tensor_superop(super_a, super_b)
    assert super_composed.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    choi_from_super = superop_to_choi(super_composed)
    fid_super = process_fidelity(choi_from_super, qobj_tensored_ref)
    assert jnp.allclose(fid_super, 1.0, atol=1e-6), "Composed SuperOp operators don't match"

    # Tensor PauliLiouville
    pauli_a = choi_to_pauli_liouville(choi_a)
    pauli_b = choi_to_pauli_liouville(choi_b)
    pauli_composed = tensor_pauli_liouville(pauli_a, pauli_b)
    assert pauli_composed.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    choi_from_pauli = pauli_liouville_to_choi(pauli_composed)
    fid_pauli = process_fidelity(choi_from_pauli, qobj_tensored_ref)
    assert jnp.allclose(fid_pauli, 1.0, atol=1e-6), "Composed PauliLiouville operators don't match"

    # Tensor KrausMaps
    kraus_a = choi_to_kraus(choi_a)
    kraus_b = choi_to_kraus(choi_b)
    kraus_composed = tensor_kraus(kraus_a, kraus_b)
    assert kraus_composed.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    choi_from_kraus = kraus_to_choi(kraus_composed)
    fid_kraus = process_fidelity(choi_from_kraus, qobj_tensored_ref)
    assert jnp.allclose(fid_kraus, 1.0, atol=1e-6), "Composed KrausMap operators don't match"


@pytest.mark.parametrize("seed", [58, 3854, 2047, 475])
@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("size_a, size_b", [((), ()), ((3,), ()), ((3,), (3,)), ((3, 4), ()), ((3, 4), (3, 4))])
def test_tensor_unitaries(seed, num_qubits, size_a, size_b):
    """Test tensor product for unitaries."""
    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)

    # Generate two random unitaries
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    tensor_dims = ((2,) * num_qubits * 2, (2,) * num_qubits * 2)
    unitary_a = random_unitary(dims=dims, key=key1, size=size_a)
    unitary_b = random_unitary(dims=dims, key=key2, size=size_b)

    # Compute the reference result using QuTiP
    qobj_a = unitary_a._to_qobj()
    qobj_b = unitary_b._to_qobj()

    ensemble_size = jnp.broadcast_shapes(size_a, size_b)

    # Use qt_tensor helper for broadcasting
    qobj_composed_ref = Unitary.from_matrix(qt_tensor(qobj_a, qobj_b), tensor_dims)
    assert qobj_composed_ref.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"

    tensored_unitaries = tensor_unitary(unitary_a, unitary_b)
    # Check match
    fid = unitary_entanglement_fidelity(qobj_composed_ref, tensored_unitaries)
    assert jnp.allclose(fid, 1.0, atol=1e-6), "Composed Unitary operators don't match"


@pytest.mark.parametrize("seed", [58, 3854, 2047, 475])
@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("size_a, size_b", [((), ()), ((3,), ()), ((3,), (3,)), ((3, 4), ()), ((3, 4), (3, 4))])
def test_tensor_state_vectors(seed, num_qubits, size_a, size_b):
    """Test tensor product for state vectors."""
    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)

    # Generate two random state vectors
    dims = (2,) * num_qubits
    psi_a = random_state_vector(dims=dims, key=key1, size=size_a)
    psi_b = random_state_vector(dims=dims, key=key2, size=size_b)

    # Compute the reference result using QuTiP
    qobj_a = psi_a._to_qobj()
    qobj_b = psi_b._to_qobj()

    tensor_dims = dims + dims

    ensemble_size = jnp.broadcast_shapes(size_a, size_b)

    # Use qt_tensor helper for broadcasting and flatten to get state vector
    tensored_data = qt_tensor(qobj_a, qobj_b)
    # Flatten the last dimension (d, d) -> (d*d,) for each element
    tensored_data = tensored_data.reshape(ensemble_size + (-1,))
    qobj_tensored_ref = StateVector.from_matrix(tensored_data, tensor_dims)
    assert qobj_tensored_ref.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"

    tensored_states = tensor_state_vector(psi_a, psi_b)
    # Check match
    assert tensored_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    fid = fidelity(qobj_tensored_ref, tensored_states)
    assert jnp.allclose(fid, 1.0, atol=1e-6), "Tensored StateVectors don't match"


@pytest.mark.parametrize("seed", [58, 3854, 2047, 475])
@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("size_a, size_b", [((), ()), ((3,), ()), ((3,), (3,)), ((3, 4), ()), ((3, 4), (3, 4))])
def test_tensor_density_matrices(seed, num_qubits, size_a, size_b):
    """Test tensor product for density matrices."""
    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)
    d = 2**num_qubits

    # Generate two random density matrices
    dims = (2,) * num_qubits
    rho_a = random_density_matrix(rank=d, dims=dims, key=key1, size=size_a)
    rho_b = random_density_matrix(rank=d, dims=dims, key=key2, size=size_b)

    # Compute the reference result using QuTiP
    qobj_a = rho_a._to_qobj()
    qobj_b = rho_b._to_qobj()

    tensor_dims = dims + dims

    ensemble_size = jnp.broadcast_shapes(size_a, size_b)

    # Use qt_tensor helper for broadcasting
    qobj_tensored_ref = DensityMatrix.from_matrix(qt_tensor(qobj_a, qobj_b), tensor_dims)
    assert qobj_tensored_ref.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"

    tensored_states = tensor_density_matrix(rho_a, rho_b)
    # Check match
    assert tensored_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    fid = fidelity(qobj_tensored_ref, tensored_states)
    assert jnp.allclose(fid, 1.0, atol=1e-6), "Tensored DensityMatrices don't match"
