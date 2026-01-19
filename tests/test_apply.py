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

"""Tests for JAX-based superoperator application."""

from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qutip as qt

from quax import (
    DensityMatrix,
    StateVector,
    Unitary,
    apply_choi_to_density_matrix,
    apply_kraus_to_density_matrix,
    apply_pauli_liouville_to_density_matrix,
    apply_superop_to_density_matrix,
    apply_unitary_to_state_vector,
    choi_to_kraus,
    choi_to_pauli_liouville,
    choi_to_superop,
    compute_choi_observables_from_states,
    compute_kraus_observables_from_states,
    compute_pauli_liouville_observables_from_states,
    compute_superop_observables_from_states,
    fidelity,
    partial_trace,
    random_choi_BCSZ,
    random_density_matrix,
    random_state_vector,
    random_unitary,
)

from quax.ensembles import PAULIS


@pytest.mark.parametrize("seed", [58, 3854, 2047, 475])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_compute_observables_random_channel(seed, num_qubits):
    """Test observable computation for random channels against qutip."""
    key = jax.random.key(seed)
    d = 2**num_qubits
    dims = (2,) * num_qubits
    kraus_rank = d

    num_states = 10
    num_observables = 8

    choi = random_choi_BCSZ(dims=(dims, dims), rank=kraus_rank, key=key)
    kraus_map = choi_to_kraus(choi)
    superop = choi_to_superop(choi)
    pl = choi_to_pauli_liouville(choi)

    # Generate random input states
    states = random_density_matrix(rank=kraus_rank, dims=dims, key=key, size=(num_states,))

    # Generate random Pauli observables of weight num_qubits
    subkeys = jax.random.split(key, num=num_observables)
    observables = Unitary(
        data=jnp.asarray(
            [
                reduce(
                    jnp.kron,
                    jax.random.choice(
                        k,
                        PAULIS.data,
                        shape=(num_qubits,),
                        replace=True,
                    ),
                )
                for k in subkeys
            ]
        ),
        dims=((2,) * num_qubits, (2,) * num_qubits),
    )

    # Compute with JAX implementations
    observables_choi = compute_choi_observables_from_states(choi, states, observables)
    observables_kraus = compute_kraus_observables_from_states(kraus_map, states, observables)
    observables_superop = compute_superop_observables_from_states(superop, states, observables)
    observables_pl = compute_pauli_liouville_observables_from_states(pl, states, observables)

    # Compute with qutip
    dims_super = [[[2] * num_qubits, [2] * num_qubits], [[2] * num_qubits, [2] * num_qubits]]
    dims_state = [[2] * num_qubits, [2] * num_qubits]
    dims_obs = [[2] * num_qubits, [2] * num_qubits]
    super_qutip = qt.Qobj(superop.data, superrep="super", dims=dims_super)
    states_qutip = [qt.Qobj(rho, dims=dims_state) for rho in states.data]
    observables_qutip = [qt.Qobj(obs, isunitary=True, dims=dims_obs) for obs in observables.data]

    expected_observables = []  # jnp.zeros((num_states, num_observables), dtype=jnp.float64)
    for i, rho in enumerate(states_qutip):
        # Apply the channel using superop
        rho_out = super_qutip(rho)
        expected_observables_row = []
        for j, obs in enumerate(observables_qutip):
            expected_observables_row.append(jnp.real(qt.expect(obs, rho_out)))
        expected_observables.append(expected_observables_row)

    expected_observables = jnp.array(expected_observables)

    # Compare results
    assert jnp.allclose(observables_choi, expected_observables, atol=1e-6)
    assert jnp.allclose(observables_kraus, expected_observables, atol=1e-6)
    assert jnp.allclose(observables_superop, expected_observables, atol=1e-6)
    assert jnp.allclose(observables_pl, expected_observables, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("num_qubits", [2, 3, 4])
def test_partial_trace(seed, num_qubits):
    """Test that our partial trace implementation matches qutip's."""
    key = jax.random.key(seed)
    dims = (2,) * num_qubits
    d = 2**num_qubits
    rank = d
    rho = random_density_matrix(rank=rank, dims=dims, key=key)

    # Convert the density matrix to a qutip object
    rho_qobj = rho._to_qobj()

    # Choose random subsystems to trace out (trace out half the qubits)
    subsystems = tuple(jax.random.choice(key, num_qubits, (num_qubits // 2,), replace=False).tolist())

    rho_traced = partial_trace(rho, subsystems)

    rho_qutip_traced = jnp.array(rho_qobj.ptrace(subsystems).full())

    assert jnp.allclose(rho_traced.data, rho_qutip_traced, atol=1e-6)

    ## Test for Choi matrices
    choi = random_choi_BCSZ(dims=(dims, dims), rank=rank, key=key)
    choi_qobj = choi._to_qobj()
    choi_traced = partial_trace(choi, subsystems)
    choi_qutip_traced = jnp.array(choi_qobj.ptrace(subsystems).full())
    assert jnp.allclose(choi_traced.data, choi_qutip_traced, atol=1e-6)


@pytest.mark.parametrize("seed", [58, 3854])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((5,), (5,)),
        ((3, 4), (3, 4)),
        ((3, 4), ()),
    ],
)
def test_apply_superoperator_to_density_matrix(seed, num_qubits, ensemble_size):
    """Test the application of superoperators to density matrices."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    d = 2**num_qubits
    dims = (2,) * num_qubits
    kraus_rank = d
    ensemble_size_0, ensemble_size_1 = ensemble_size

    choi = random_choi_BCSZ(dims=(dims, dims), rank=kraus_rank, key=key, size=ensemble_size_0)

    # Generate random input states
    state = random_density_matrix(rank=kraus_rank, dims=dims, key=subkey, size=ensemble_size_1)
    ensemble_size = jnp.broadcast_shapes(ensemble_size_0, ensemble_size_1)

    qobj_state = state._to_qobj()
    qobj_choi = choi._to_qobj()

    def qt_apply(a, b):
        # Accept scalar Qobj or numpy array(dtype=object) of Qobj
        A = np.asarray(a, dtype=object)
        B = np.asarray(b, dtype=object)

        A, B = np.broadcast_arrays(A, B)
        out_shape = A.shape

        mats = [
            qt.vector_to_operator(qt.to_super(choi) @ qt.operator_to_vector(rho)).full()
            for choi, rho in zip(A.ravel(), B.ravel())
        ]  # each is (d,d) ndarray
        dense = np.stack(mats, axis=0)
        dense = dense.reshape(out_shape + dense.shape[1:])  # out_shape + (d,d)
        return jnp.asarray(dense)  # numeric ndarray (complex)

    qobj_applied_ref = DensityMatrix(data=qt_apply(qobj_choi, qobj_state), dims=dims)

    # Apply Chois
    applied_states = apply_choi_to_density_matrix(choi, state)
    assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    assert jnp.allclose(fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), (
        "Applied density matrices fidelity too low"
    )
    assert jnp.allclose(applied_states.data, qobj_applied_ref.data, atol=1e-6), "Applied density matrices don't match"

    # Apply Kraus
    kraus_map = choi_to_kraus(choi)
    applied_states = apply_kraus_to_density_matrix(kraus_map, state)
    assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    assert jnp.allclose(fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), (
        "Applied density matrices fidelity too low"
    )
    assert jnp.allclose(applied_states.data, qobj_applied_ref.data, atol=1e-6), "Applied density matrices don't match"

    # Apply SuperOp
    superop = choi_to_superop(choi)
    applied_states = apply_superop_to_density_matrix(superop, state)
    assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    assert jnp.allclose(fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), (
        "Applied density matrices fidelity too low"
    )
    assert jnp.allclose(applied_states.data, qobj_applied_ref.data, atol=1e-6), "Applied density matrices don't match"

    # Apply PauliLiouville
    pl = choi_to_pauli_liouville(choi)
    applied_states = apply_pauli_liouville_to_density_matrix(pl, state)
    assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    assert jnp.allclose(fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), (
        "Applied density matrices fidelity too low"
    )
    assert jnp.allclose(applied_states.data, qobj_applied_ref.data, atol=1e-6), "Applied density matrices don't match"


@pytest.mark.parametrize("seed", [58, 3854])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((5,), (5,)),
        ((3, 4), (3, 4)),
        ((3, 4), ()),
    ],
)
def test_apply_operator_to_state_vector(seed, num_qubits, ensemble_size):
    """Test the application of superoperators to density matrices."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    dims = (2,) * num_qubits
    ensemble_size_0, ensemble_size_1 = ensemble_size

    unitary = random_unitary(dims=(dims, dims), key=key, size=ensemble_size_0)

    # Generate random input states
    state = random_state_vector(dims=dims, key=subkey, size=ensemble_size_1)
    ensemble_size = jnp.broadcast_shapes(ensemble_size_0, ensemble_size_1)

    qobj_state = state._to_qobj()
    qobj_unitary = unitary._to_qobj()

    def qt_apply(U, ket):
        # Accept scalar Qobj or numpy array(dtype=object) of Qobj
        A = np.asarray(U, dtype=object)
        B = np.asarray(ket, dtype=object)

        A, B = np.broadcast_arrays(A, B)
        out_shape = A.shape

        # Apply: Qobj @ Qobj -> Qobj (ket)
        outs = [(u @ v) for u, v in zip(A.ravel(), B.ravel())]

        # Convert each ket to dense and flatten from (d,1) -> (d,)
        vecs = [np.asarray(o.full()).reshape(-1) for o in outs]  # (d,) each

        dense = np.stack(vecs, axis=0)  # (N, d)
        dense = dense.reshape(out_shape + dense.shape[1:])  # out_shape + (d,)

        return jnp.asarray(dense)

    qobj_applied_ref = StateVector(data=qt_apply(qobj_unitary, qobj_state), dims=dims)

    # Apply Unitaries
    applied_states = apply_unitary_to_state_vector(unitary, state)
    assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    assert jnp.allclose(fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), (
        "Applied density matrices fidelity too low"
    )
    assert jnp.allclose(applied_states.data, qobj_applied_ref.data, atol=1e-6), "Applied density matrices don't match"

    # Apply Kraus
    # applied_states = apply_kraus_to_statevector(kraus, state)
    # assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    # assert jnp.allclose(fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), "Applied density matrices fidelity too low"
    # assert jnp.allclose(applied_states.data, qobj_applied_ref.data, atol=1e-6), "Applied density matrices don't match"
