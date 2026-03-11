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
import quax as qx


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

    choi = qx.random_choi(dims=(dims, dims), rank=kraus_rank, key=key)
    kraus_map = qx.choi_to_kraus(choi)
    superop = qx.choi_to_superop(choi)
    pl = qx.choi_to_pauli_liouville(choi)

    # Generate random input states
    states = qx.random_density_matrix(rank=kraus_rank, dims=dims, key=key, size=(num_states,))

    # Generate random Pauli observables of weight num_qubits
    subkeys = jax.random.split(key, num=num_observables)
    observables = qx.Unitary.from_matrix(
        jnp.asarray(
            [
                reduce(
                    jnp.kron,
                    jax.random.choice(
                        k,
                        qx.ensembles.PAULIS.matrix,
                        shape=(num_qubits,),
                        replace=True,
                    ),
                )
                for k in subkeys
            ]
        ),
        ((2,) * num_qubits, (2,) * num_qubits),
    )

    # Compute with JAX implementations
    observables_choi = qx.compute_choi_observables_from_states(choi, states, observables)
    observables_kraus = qx.compute_kraus_observables_from_states(kraus_map, states, observables)
    observables_superop = qx.compute_superop_observables_from_states(superop, states, observables)
    observables_pl = qx.compute_pauli_liouville_observables_from_states(pl, states, observables)

    # Compute with qutip
    dims_super = [[[2] * num_qubits, [2] * num_qubits], [[2] * num_qubits, [2] * num_qubits]]
    dims_state = [[2] * num_qubits, [2] * num_qubits]
    dims_obs = [[2] * num_qubits, [2] * num_qubits]
    super_qutip = qt.Qobj(superop.matrix, superrep="super", dims=dims_super)
    states_qutip = [qt.Qobj(rho, dims=dims_state) for rho in states.matrix]
    observables_qutip = [qt.Qobj(obs, isunitary=True, dims=dims_obs) for obs in observables.matrix]

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
    rho = qx.random_density_matrix(rank=rank, dims=dims, key=key)

    # Convert the density matrix to a qutip object
    rho_qobj = rho._to_qobj()

    # Choose random subsystems to trace out (trace out half the qubits)
    subsystems = tuple(jax.random.choice(key, num_qubits, (num_qubits // 2,), replace=False).tolist())

    rho_traced = qx.partial_trace(rho, subsystems)

    rho_qutip_traced = jnp.array(rho_qobj.ptrace(subsystems).full())

    assert jnp.allclose(rho_traced.matrix, rho_qutip_traced, atol=1e-6)

    ## Test for Choi matrices
    choi = qx.random_choi(dims=(dims, dims), rank=rank, key=key)
    choi_qobj = choi._to_qobj()
    choi_traced = qx.partial_trace(choi, subsystems)
    choi_qutip_traced = jnp.array(choi_qobj.ptrace(subsystems).full())
    assert jnp.allclose(choi_traced.matrix, choi_qutip_traced, atol=1e-6)


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

    choi = qx.random_choi(dims=(dims, dims), rank=kraus_rank, key=key, size=ensemble_size_0)

    # Generate random input states
    state = qx.random_density_matrix(rank=kraus_rank, dims=dims, key=subkey, size=ensemble_size_1)
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

    qobj_applied_ref = qx.DensityMatrix.from_matrix(qt_apply(qobj_choi, qobj_state), dims)

    # Apply Chois
    applied_states = qx.apply_choi_to_density_matrix(choi, state)
    assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    assert jnp.allclose(qx.fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), (
        "Applied density matrices fidelity too low"
    )
    assert jnp.allclose(applied_states.matrix, qobj_applied_ref.matrix, atol=1e-6), (
        "Applied density matrices don't match"
    )

    # Apply Kraus
    kraus_map = qx.choi_to_kraus(choi)
    applied_states = qx.apply_kraus_to_density_matrix(kraus_map, state)
    assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    assert jnp.allclose(qx.fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), (
        "Applied density matrices fidelity too low"
    )
    assert jnp.allclose(applied_states.matrix, qobj_applied_ref.matrix, atol=1e-6), (
        "Applied density matrices don't match"
    )

    # Apply SuperOp
    superop = qx.choi_to_superop(choi)
    applied_states = qx.apply_superop_to_density_matrix(superop, state)
    assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    assert jnp.allclose(qx.fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), (
        "Applied density matrices fidelity too low"
    )
    assert jnp.allclose(applied_states.matrix, qobj_applied_ref.matrix, atol=1e-6), (
        "Applied density matrices don't match"
    )

    # Apply PauliLiouville
    pl = qx.choi_to_pauli_liouville(choi)
    applied_states = qx.apply_pauli_liouville_to_density_matrix(pl, state)
    assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    assert jnp.allclose(qx.fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), (
        "Applied density matrices fidelity too low"
    )
    assert jnp.allclose(applied_states.matrix, qobj_applied_ref.matrix, atol=1e-6), (
        "Applied density matrices don't match"
    )


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

    unitary = qx.random_unitary(dims=(dims, dims), key=key, size=ensemble_size_0)

    # Generate random input states
    state = qx.random_state_vector(dims=dims, key=subkey, size=ensemble_size_1)
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

    qobj_applied_ref = qx.StateVector.from_matrix(qt_apply(qobj_unitary, qobj_state), dims)

    # Apply Unitaries
    applied_states = qx.apply_unitary_to_state_vector(unitary, state)
    assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    assert jnp.allclose(qx.fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), (
        "Applied density matrices fidelity too low"
    )
    assert jnp.allclose(applied_states.matrix, qobj_applied_ref.matrix, atol=1e-6), (
        "Applied density matrices don't match"
    )

    # Apply Kraus
    # applied_states = apply_kraus_to_statevector(kraus, state)
    # assert applied_states.ensemble_size == ensemble_size, "Broadcasted ensemble sizes do not match"
    # assert jnp.allclose(fidelity(applied_states, qobj_applied_ref), 1.0, atol=1e-6), "Applied density matrices fidelity too low"
    # assert jnp.allclose(applied_states.data, qobj_applied_ref.data, atol=1e-6), "Applied density matrices don't match"


def test_targeted_apply_superop():
    """Test that targeted_apply works correctly."""
    seed = 90573
    key = jax.random.key(seed)
    initial_state = qx.random_density_matrix(3, (2, 2, 2), key)

    # X on qubit 0
    reference_operator = qx.gates.X | qx.gates.I | qx.gates.I
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_superop(qx.unitary_to_superop(qx.gates.X), initial_state, (0,))
    assert rho_targeted == rho_reference

    # X on qubit 2
    reference_operator = qx.gates.I | qx.gates.I | qx.gates.X
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_superop(qx.unitary_to_superop(qx.gates.X), initial_state, (2,))
    assert rho_targeted == rho_reference

    # CNOT on qubits 1 0
    reference_operator = (qx.gates.SWAP @ qx.gates.CNOT @ qx.gates.SWAP) | qx.gates.I
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_superop(qx.unitary_to_superop(qx.gates.CNOT), initial_state, (1, 0))
    assert rho_targeted == rho_reference

    # CNOT on qubits 0 2
    reference_operator = (qx.gates.I | qx.gates.SWAP) @ (qx.gates.CNOT | qx.gates.I) @ (qx.gates.I | qx.gates.SWAP)
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_superop(qx.unitary_to_superop(qx.gates.CNOT), initial_state, (0, 2))
    assert rho_targeted == rho_reference

    # Depolarizing channel on qubit 1
    s = qx.depolarizing_channel_superoperator(0.05, 1)
    reference_operator = qx.gates.I | s | qx.gates.I
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_superop(s, initial_state, (1,))
    assert rho_targeted == rho_reference

    # Depolarizing channel on qubit 0 1 2
    s = qx.depolarizing_channel_superoperator(0.05, 3)
    reference_operator = s
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_superop(s, initial_state, (0, 1, 2))
    assert rho_targeted == rho_reference


def test_targeted_apply_kraus_map():
    """Test that targeted_apply works correctly."""
    seed = 90573
    key = jax.random.key(seed)
    initial_state = qx.random_density_matrix(3, (2, 2, 2), key)

    # X on qubit 0
    reference_operator = qx.gates.X | qx.gates.I | qx.gates.I
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_kraus_map(qx.unitary_to_kraus_map(qx.gates.X), initial_state, (0,))
    assert rho_targeted == rho_reference

    # X on qubit 2
    reference_operator = qx.gates.I | qx.gates.I | qx.gates.X
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_kraus_map(qx.unitary_to_kraus_map(qx.gates.X), initial_state, (2,))
    assert rho_targeted == rho_reference

    # CNOT on qubits 1 0
    reference_operator = (qx.gates.SWAP @ qx.gates.CNOT @ qx.gates.SWAP) | qx.gates.I
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_kraus_map(qx.unitary_to_kraus_map(qx.gates.CNOT), initial_state, (1, 0))
    assert rho_targeted == rho_reference

    # CNOT on qubits 0 2
    reference_operator = (qx.gates.I | qx.gates.SWAP) @ (qx.gates.CNOT | qx.gates.I) @ (qx.gates.I | qx.gates.SWAP)
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_kraus_map(qx.unitary_to_kraus_map(qx.gates.CNOT), initial_state, (0, 2))
    assert rho_targeted == rho_reference

    # Depolarizing channel on qubit 1
    s = qx.depolarizing_channel_superoperator(0.05, 1)
    reference_operator = qx.gates.I | s | qx.gates.I
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_kraus_map(qx.superop_to_kraus(s), initial_state, (1,))
    assert rho_targeted == rho_reference

    # Depolarizing channel on qubit 0 1 2
    s = qx.depolarizing_channel_superoperator(0.05, 3)
    reference_operator = s
    rho_reference = reference_operator @ initial_state
    rho_targeted = qx.targeted_apply_kraus_map(qx.superop_to_kraus(s), initial_state, (0, 1, 2))
    assert rho_targeted == rho_reference


@pytest.mark.parametrize("seed", [90573, 42])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((3,), ()),
        ((), (3,)),
        ((3,), (3,)),
        ((2, 3), (2, 3)),
    ],
)
def test_targeted_apply_superop_ensemble(seed, ensemble_size):
    """Test that targeted_apply_superop works correctly with ensembles."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    ens_op, ens_rho = ensemble_size

    initial_state = qx.random_density_matrix(3, (2, 2, 2), key, size=ens_rho)

    # Single-qubit depolarizing channel on qubit 1
    s_1q = qx.depolarizing_channel_superoperator(0.05, 1)

    # Build ensemble of superops if needed (stack copies with different noise params)
    if ens_op:
        keys = jax.random.split(subkey, num=reduce(lambda a, b: a * b, ens_op, 1))
        noise_params = jnp.linspace(0.01, 0.1, len(keys))
        superop_mats = jnp.stack([qx.depolarizing_channel_superoperator(float(p), 1).data for p in noise_params])
        superop_mats = superop_mats.reshape(ens_op + superop_mats.shape[1:])
        s_ensemble = qx.SuperOp(data=superop_mats, num_qubits=1)
    else:
        s_ensemble = s_1q

    # Apply targeted
    rho_targeted = qx.targeted_apply_superop(s_ensemble, initial_state, (1,))

    # Build full-system reference via tensor and apply element-wise
    broadcast_ens = jnp.broadcast_shapes(ens_op, ens_rho)
    assert rho_targeted.ensemble_size == broadcast_ens

    # Build reference: tensor I | S | I and apply
    s_eye = qx.unitary_to_superop(qx.gates.I)
    reference_operator = s_eye | s_ensemble | s_eye
    rho_reference = reference_operator @ initial_state
    assert jnp.allclose(rho_targeted.matrix, rho_reference.matrix, atol=1e-6)


@pytest.mark.parametrize("seed", [90573, 42])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((3,), ()),
        ((), (3,)),
        ((3,), (3,)),
        ((2, 3), (2, 3)),
    ],
)
def test_targeted_apply_kraus_map_ensemble(seed, ensemble_size):
    """Test that targeted_apply_kraus_map works correctly with ensembles."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    ens_op, ens_rho = ensemble_size

    initial_state = qx.random_density_matrix(3, (2, 2, 2), key, size=ens_rho)

    # Single-qubit depolarizing channel on qubit 1
    s_1q = qx.depolarizing_channel_superoperator(0.05, 1)
    k_1q = qx.superop_to_kraus(s_1q)

    # Build ensemble of kraus maps if needed
    if ens_op:
        keys = jax.random.split(subkey, num=reduce(lambda a, b: a * b, ens_op, 1))
        noise_params = jnp.linspace(0.01, 0.1, len(keys))
        kraus_list = [
            qx.superop_to_kraus(qx.depolarizing_channel_superoperator(float(p), 1)).data for p in noise_params
        ]
        kraus_mats = jnp.stack(kraus_list)
        kraus_mats = kraus_mats.reshape(ens_op + kraus_mats.shape[1:])
        k_ensemble = qx.KrausMap(data=kraus_mats, num_qubits=1)
        # Also build corresponding superops for the reference
        superop_list = [qx.depolarizing_channel_superoperator(float(p), 1).data for p in noise_params]
        superop_mats = jnp.stack(superop_list).reshape(ens_op + jnp.stack(superop_list).shape[1:])
        s_ensemble = qx.SuperOp(data=superop_mats, num_qubits=1)
    else:
        k_ensemble = k_1q
        s_ensemble = s_1q

    # Apply targeted
    rho_targeted = qx.targeted_apply_kraus_map(k_ensemble, initial_state, (1,))

    # Check ensemble size
    broadcast_ens = jnp.broadcast_shapes(ens_op, ens_rho)
    assert rho_targeted.ensemble_size == broadcast_ens

    # Build reference: tensor I | S | I and apply (using superop for reference)
    s_eye = qx.unitary_to_superop(qx.gates.I)
    reference_operator = s_eye | s_ensemble | s_eye
    rho_reference = reference_operator @ initial_state
    assert jnp.allclose(rho_targeted.matrix, rho_reference.matrix, atol=1e-6)


def test_targeted_apply_unitary():
    """Test that targeted_apply_unitary works correctly."""
    seed = 90573
    key = jax.random.key(seed)
    initial_state = qx.random_state_vector((2, 2, 2), key)

    # X on qubit 0
    reference_operator = qx.gates.X | qx.gates.I | qx.gates.I
    psi_reference = reference_operator @ initial_state
    psi_targeted = qx.targeted_apply_unitary(qx.gates.X, initial_state, (0,))
    assert psi_targeted == psi_reference

    # X on qubit 2
    reference_operator = qx.gates.I | qx.gates.I | qx.gates.X
    psi_reference = reference_operator @ initial_state
    psi_targeted = qx.targeted_apply_unitary(qx.gates.X, initial_state, (2,))
    assert psi_targeted == psi_reference

    # CNOT on qubits 1 0
    reference_operator = (qx.gates.SWAP @ qx.gates.CNOT @ qx.gates.SWAP) | qx.gates.I
    psi_reference = reference_operator @ initial_state
    psi_targeted = qx.targeted_apply_unitary(qx.gates.CNOT, initial_state, (1, 0))
    assert psi_targeted == psi_reference

    # CNOT on qubits 0 2
    reference_operator = (qx.gates.I | qx.gates.SWAP) @ (qx.gates.CNOT | qx.gates.I) @ (qx.gates.I | qx.gates.SWAP)
    psi_reference = reference_operator @ initial_state
    psi_targeted = qx.targeted_apply_unitary(qx.gates.CNOT, initial_state, (0, 2))
    assert psi_targeted == psi_reference


@pytest.mark.parametrize("seed", [90573, 42])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((3,), ()),
        ((), (3,)),
        ((3,), (3,)),
        ((2, 3), (2, 3)),
    ],
)
def test_targeted_apply_unitary_ensemble(seed, ensemble_size):
    """Test that targeted_apply_unitary works correctly with ensembles."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    ens_op, ens_psi = ensemble_size

    initial_state = qx.random_state_vector((2, 2, 2), key, size=ens_psi)

    # Build ensemble of unitaries if needed (stack copies with different noise params)
    if ens_op:
        u_ensemble = qx.gates.RX(jnp.ones(ens_op) * jnp.pi / 3)
    else:
        u_ensemble = qx.gates.RX(jnp.pi / 3)

    # Apply targeted
    psi_targeted = qx.targeted_apply_unitary(u_ensemble, initial_state, (1,))

    # Build full-system reference via tensor and apply element-wise
    broadcast_ens = jnp.broadcast_shapes(ens_op, ens_psi)
    assert psi_targeted.ensemble_size == broadcast_ens

    # Build reference: tensor I | U | I and apply
    u_eye = qx.gates.I
    reference_operator = u_eye | u_ensemble | u_eye
    psi_reference = reference_operator @ initial_state
    assert jnp.allclose(psi_targeted.matrix, psi_reference.matrix, atol=1e-6)


def test_targeted_apply_kraus_map_trajectory_unitary():
    """A single-operator Kraus map (unitary) should match targeted_apply_unitary exactly."""
    seed = 90573
    key = jax.random.key(seed)
    key, sample_key = jax.random.split(key)
    initial_state = qx.random_state_vector((2, 2, 2), key)

    # X on qubit 0
    kraus = qx.unitary_to_kraus_map(qx.gates.X)
    psi_trajectory = qx.targeted_apply_kraus_map_trajectory(kraus, initial_state, sample_key, (0,))
    psi_reference = qx.targeted_apply_unitary(qx.gates.X, initial_state, (0,))
    assert psi_trajectory == psi_reference

    # X on qubit 2
    kraus = qx.unitary_to_kraus_map(qx.gates.X)
    psi_trajectory = qx.targeted_apply_kraus_map_trajectory(kraus, initial_state, sample_key, (2,))
    psi_reference = qx.targeted_apply_unitary(qx.gates.X, initial_state, (2,))
    assert psi_trajectory == psi_reference

    # CNOT on qubits 1 0
    kraus = qx.unitary_to_kraus_map(qx.gates.CNOT)
    psi_trajectory = qx.targeted_apply_kraus_map_trajectory(kraus, initial_state, sample_key, (1, 0))
    psi_reference = qx.targeted_apply_unitary(qx.gates.CNOT, initial_state, (1, 0))
    assert psi_trajectory == psi_reference

    # CNOT on qubits 0 2
    kraus = qx.unitary_to_kraus_map(qx.gates.CNOT)
    psi_trajectory = qx.targeted_apply_kraus_map_trajectory(kraus, initial_state, sample_key, (0, 2))
    psi_reference = qx.targeted_apply_unitary(qx.gates.CNOT, initial_state, (0, 2))
    assert psi_trajectory == psi_reference


def test_targeted_apply_kraus_map_trajectory_normalization():
    """Output states should always be normalized."""
    key = jax.random.key(42)
    key, sample_key = jax.random.split(key)
    initial_state = qx.random_state_vector((2, 2, 2), key)

    # Depolarizing channel on qubit 1
    s = qx.depolarizing_channel_superoperator(0.1, 1)
    kraus = qx.superop_to_kraus(s)
    psi_out = qx.targeted_apply_kraus_map_trajectory(kraus, initial_state, sample_key, (1,))
    norm = jnp.sum(jnp.abs(psi_out.matrix) ** 2)
    assert jnp.allclose(norm, 1.0, atol=1e-6)


@pytest.mark.parametrize("seed", [42, 123, 7])
def test_targeted_apply_kraus_map_trajectory_statistical_convergence(seed):
    """Averaging |ψ⟩⟨ψ| over many trajectories should converge to ∑ K_i ρ K_i†."""
    key = jax.random.key(seed)
    key, state_key = jax.random.split(key)
    initial_state = qx.random_state_vector((2, 2), state_key)

    # Depolarizing channel on qubit 0
    p = 0.1
    s = qx.depolarizing_channel_superoperator(p, 1)
    kraus = qx.superop_to_kraus(s)

    # Reference: apply Kraus map to density matrix
    rho_initial = qx.promote_state_vector_to_density_matrix(initial_state)
    rho_reference = qx.targeted_apply_kraus_map(kraus, rho_initial, (0,))

    # Monte Carlo: use an ensemble of keys to get all trajectories at once
    n_trajectories = 5000
    sample_keys = jax.random.split(key, num=n_trajectories)

    psi_out = qx.targeted_apply_kraus_map_trajectory(kraus, initial_state, sample_keys, (0,))
    # psi_out.matrix has shape (n_trajectories, d) — compute |ψ⟩⟨ψ| and average
    rho_avg = jnp.mean(psi_out.matrix[:, :, None] * psi_out.matrix[:, None, :].conj(), axis=0)
    rho_avg = qx.DensityMatrix.from_matrix(rho_avg, (2, 2))

    assert qx.fidelity(rho_avg, rho_reference) > 0.99

    # Second example: 3-qubit state with a 1q depolarizing channel on qubit 2
    # followed by a 2q depolarizing channel on qubits (0, 1)
    key, state_key2 = jax.random.split(key)
    initial_state_3q = qx.random_state_vector((2, 2, 2), state_key2)

    s_1q = qx.depolarizing_channel_superoperator(0.05, 1)
    kraus_1q = qx.superop_to_kraus(s_1q)
    s_2q = qx.depolarizing_channel_superoperator(0.08, 2)
    kraus_2q = qx.superop_to_kraus(s_2q)

    # Reference: apply both channels to density matrix
    rho_initial_3q = qx.promote_state_vector_to_density_matrix(initial_state_3q)
    rho_ref_3q = qx.targeted_apply_kraus_map(kraus_1q, rho_initial_3q, (2,))
    rho_ref_3q = qx.targeted_apply_kraus_map(kraus_2q, rho_ref_3q, (0, 1))

    # Monte Carlo: apply both channels sequentially per trajectory
    key, key1, key2 = jax.random.split(key, 3)
    sample_keys_1 = jax.random.split(key1, num=n_trajectories)
    sample_keys_2 = jax.random.split(key2, num=n_trajectories)

    psi_mid = qx.targeted_apply_kraus_map_trajectory(kraus_1q, initial_state_3q, sample_keys_1, (2,))
    psi_out_3q = qx.targeted_apply_kraus_map_trajectory(kraus_2q, psi_mid, sample_keys_2, (0, 1))

    rho_avg_3q = jnp.mean(psi_out_3q.matrix[:, :, None] * psi_out_3q.matrix[:, None, :].conj(), axis=0)
    rho_avg_3q = qx.DensityMatrix.from_matrix(rho_avg_3q, (2, 2, 2))

    assert qx.fidelity(rho_avg_3q, rho_ref_3q) > 0.99


@pytest.mark.parametrize("seed", [90573, 42])
@pytest.mark.parametrize("ens_key", [(), (3,)])
@pytest.mark.parametrize("ens_psi", [(), (3,)])
@pytest.mark.parametrize("ens_op", [(), (3,)])
def test_targeted_apply_kraus_map_trajectory_ensemble(seed, ens_op, ens_psi, ens_key):
    """Test ensemble broadcasting for trajectory Kraus application, including key ensembles."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)

    initial_state = qx.random_state_vector((2, 2, 2), key, size=ens_psi)

    # Build ensemble of Kraus maps if needed
    if ens_op:
        noise_params = jnp.linspace(0.01, 0.1, reduce(lambda a, b: a * b, ens_op, 1))
        kraus_list = [
            qx.superop_to_kraus(qx.depolarizing_channel_superoperator(float(p), 1)).data for p in noise_params
        ]
        kraus_mats = jnp.stack(kraus_list).reshape(ens_op + jnp.stack(kraus_list).shape[1:])
        k_ensemble = qx.KrausMap(data=kraus_mats, num_qubits=1)
    else:
        k_ensemble = qx.superop_to_kraus(qx.depolarizing_channel_superoperator(0.05, 1))

    # Build ensemble of keys if needed
    if ens_key:
        n_keys = reduce(lambda a, b: a * b, ens_key, 1)
        sample_keys = jax.random.split(subkey, num=n_keys).reshape(ens_key)
    else:
        sample_keys = subkey

    psi_out = qx.targeted_apply_kraus_map_trajectory(k_ensemble, initial_state, sample_keys, (1,))

    # Check ensemble size matches broadcast
    broadcast_ens = jnp.broadcast_shapes(ens_op, ens_psi, ens_key)
    assert psi_out.ensemble_size == broadcast_ens

    # Check normalization for all ensemble elements
    norms = jnp.sum(jnp.abs(psi_out.matrix) ** 2, axis=-1)
    assert jnp.allclose(norms, jnp.ones_like(norms), atol=1e-6)
