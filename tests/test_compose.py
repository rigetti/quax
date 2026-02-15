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

"""Tests for JAX-based superoperator composition functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qutip as qt

from quax import (
    Choi,
    Unitary,
    choi_to_kraus,
    choi_to_pauli_liouville,
    choi_to_superop,
    compose_choi,
    compose_kraus,
    compose_pauli_liouville,
    compose_superop,
    compose_unitary,
    kraus_to_choi,
    pauli_liouville_to_choi,
    process_fidelity,
    random_choi_BCSZ,
    random_unitary,
    superop_to_choi,
    unitary_entanglement_fidelity,
)


@pytest.mark.parametrize("seed", [58, 3854, 2047, 475])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("size_a, size_b", [((), ()), ((3,), ()), ((3,), (3,)), ((3, 4), ()), ((3, 4), (3, 4))])
def test_compose_superoperators(seed, num_qubits, size_a, size_b):
    """Test composition for all representations."""
    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)
    d = 2**num_qubits
    kraus_rank = d

    # Generate two random channels
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    choi_a = random_choi_BCSZ(dims=dims, rank=kraus_rank, key=key1, size=size_a)
    choi_b = random_choi_BCSZ(dims=dims, rank=kraus_rank, key=key2, size=size_b)

    # Compute the reference result using QuTiP
    qobj_a = choi_a._to_qobj()
    qobj_b = choi_b._to_qobj()

    def qt_compose(a, b):
        # Accept scalar Qobj or numpy array(dtype=object) of Qobj
        A = np.asarray(a, dtype=object)
        B = np.asarray(b, dtype=object)

        A, B = np.broadcast_arrays(A, B)
        out_shape = A.shape

        mats = [
            qt.to_choi((qt.to_super(x) @ qt.to_super(y))).full() for x, y in zip(A.ravel(), B.ravel())
        ]  # each is (d,d) ndarray
        dense = np.stack(mats, axis=0)
        dense = dense.reshape(out_shape + dense.shape[1:])  # out_shape + (d,d)
        return jnp.asarray(dense)  # numeric ndarray (complex)

    ensemble_size = jnp.broadcast_shapes(size_a, size_b)
    composed_data = qt_compose(qobj_a, qobj_b)
    qobj_composed_ref = Choi.from_matrix(composed_data, dims)

    # Compose chois
    choi_composed = compose_choi(choi_a, choi_b)
    assert choi_composed.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    fid_choi = process_fidelity(choi_composed, qobj_composed_ref)
    assert jnp.allclose(fid_choi, 1.0, atol=1e-6), "Composed Choi operators don't match"

    # Compose Superops
    super_a = choi_to_superop(choi_a)
    super_b = choi_to_superop(choi_b)
    super_composed = compose_superop(super_a, super_b)
    assert super_composed.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    choi_from_super = superop_to_choi(super_composed)
    fid_super = process_fidelity(choi_from_super, qobj_composed_ref)
    assert jnp.allclose(fid_super, 1.0, atol=1e-6), "Composed SuperOp operators don't match"

    # Compose PauliLiouville
    pauli_a = choi_to_pauli_liouville(choi_a)
    pauli_b = choi_to_pauli_liouville(choi_b)
    pauli_composed = compose_pauli_liouville(pauli_a, pauli_b)
    assert pauli_composed.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    choi_from_pauli = pauli_liouville_to_choi(pauli_composed)
    fid_pauli = process_fidelity(choi_from_pauli, qobj_composed_ref)
    assert jnp.allclose(fid_pauli, 1.0, atol=1e-6), "Composed PauliLiouville operators don't match"

    # Compose KrausMaps
    kraus_a = choi_to_kraus(choi_a)
    kraus_b = choi_to_kraus(choi_b)
    kraus_composed = compose_kraus(kraus_a, kraus_b)
    assert kraus_composed.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    choi_from_kraus = kraus_to_choi(kraus_composed)
    fid_kraus = process_fidelity(choi_from_kraus, qobj_composed_ref)
    assert jnp.allclose(fid_kraus, 1.0, atol=1e-6), "Composed KrausMap operators don't match"


@pytest.mark.parametrize("seed", [58, 3854, 2047, 475])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("size_a, size_b", [((), ()), ((3,), ()), ((3,), (3,)), ((3, 4), ()), ((3, 4), (3, 4))])
def test_compose_unitaries(seed, num_qubits, size_a, size_b):
    """Test unitary and kraus composition."""
    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)

    # Generate two random channels
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    unitary_a = random_unitary(dims=dims, key=key1, size=size_a)
    unitary_b = random_unitary(dims=dims, key=key2, size=size_b)

    # Compute the reference result using QuTiP
    qobj_a = unitary_a._to_qobj()
    qobj_b = unitary_b._to_qobj()

    def qt_compose(a, b):
        # Accept scalar Qobj or numpy array(dtype=object) of Qobj
        A = np.asarray(a, dtype=object)
        B = np.asarray(b, dtype=object)

        A, B = np.broadcast_arrays(A, B)
        out_shape = A.shape

        mats = [(x @ y).full() for x, y in zip(A.ravel(), B.ravel())]  # each is (d,d) ndarray
        dense = np.stack(mats, axis=0)
        dense = dense.reshape(out_shape + dense.shape[1:])  # out_shape + (d,d)
        return jnp.asarray(dense)  # numeric ndarray (complex)

    composed_data = qt_compose(qobj_a, qobj_b)
    qobj_composed_ref = Unitary.from_matrix(composed_data, dims)

    # Compose unitaries

    composed_unitaries = compose_unitary(unitary_a, unitary_b)
    # Check match
    fid = unitary_entanglement_fidelity(qobj_composed_ref, composed_unitaries)
    assert jnp.allclose(fid, 1.0, atol=1e-6), "Composed Unitary operators don't match"
