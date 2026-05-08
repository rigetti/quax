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
from quax import n_qudit_herm_basis, DensityMatrix, StateVector

from .instrument_helpers import basis_dm_multi


def _identity_unitary(d: int) -> qx.Unitary:
    """Identity unitary for a d-dimensional qudit."""
    return qx.Unitary.from_matrix(jnp.eye(d, dtype=complex), ((d,), (d,)))


def _identity_superop(d: int) -> qx.SuperOp:
    """Identity superoperator for a d-dimensional qudit."""
    return qx.unitary_to_superop(_identity_unitary(d))


class TestEstimate:
    """Tests for observable estimation."""

    @staticmethod
    def _random_observable(dims, rng):
        """Pick a random observable from the hermitian Weyl (Pauli for d=2) basis."""
        basis = n_qudit_herm_basis(dims)
        idx = rng.integers(0, len(basis.matrix))
        return qx.Observable.from_matrix(basis.matrix[idx], (dims, dims))

    @pytest.mark.parametrize("seed", [58, 3854, 2047, 475])
    @pytest.mark.parametrize("ensemble_size", [(), (3,), (3, 4)])
    @pytest.mark.parametrize("dims", [(2,), (2, 2), (2, 2, 2), (3,), (3, 3), (2, 3)])
    def test_pure_state(self, seed, ensemble_size, dims):
        """Test the estimate function against qutip."""
        key = jax.random.key(seed)
        rng = np.random.default_rng(seed)

        state = qx.random_state_vector(dims=dims, key=key, size=ensemble_size)
        obs = self._random_observable(dims, rng)

        result = qx.estimate(state, obs)

        # Compare against qutip
        state_qobjs = state._to_qobj()
        obs_mat = np.asarray(obs.matrix)
        obs_qobj = qt.Qobj(obs_mat, dims=[list(dims), list(dims)])

        if ensemble_size == ():
            expected = qt.expect(obs_qobj, state_qobjs)
            assert jnp.allclose(result, expected, atol=1e-6)
        else:
            state_qobjs_flat = state_qobjs.ravel()
            expected_flat = np.array([qt.expect(obs_qobj, s) for s in state_qobjs_flat])
            expected = jnp.array(expected_flat).reshape(ensemble_size)
            assert jnp.allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize("seed", [58, 3854, 2047, 475])
    @pytest.mark.parametrize("ensemble_size", [(), (3,), (3, 4)])
    @pytest.mark.parametrize("dims", [(2,), (2, 2), (2, 2, 2), (3,), (3, 3), (2, 3)])
    def test_mixed_state(self, seed, ensemble_size, dims):
        """Test the estimate function against qutip."""
        key = jax.random.key(seed)
        rng = np.random.default_rng(seed)
        d = int(np.prod(dims))
        rank = d

        state = qx.random_density_matrix(rank=rank, dims=dims, key=key, size=ensemble_size)
        obs = self._random_observable(dims, rng)

        result = qx.estimate(state, obs)

        # Compare against qutip
        state_qobjs = state._to_qobj()
        obs_mat = np.asarray(obs.matrix)
        obs_qobj = qt.Qobj(obs_mat, dims=[list(dims), list(dims)])

        if ensemble_size == ():
            expected = qt.expect(obs_qobj, state_qobjs)
            assert jnp.allclose(result, expected, atol=1e-6)
        else:
            state_qobjs_flat = state_qobjs.ravel()
            expected_flat = np.array([qt.expect(obs_qobj, s) for s in state_qobjs_flat])
            expected = jnp.array(expected_flat).reshape(ensemble_size)
            assert jnp.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("seed", [58, 3854, 2047, 475])
@pytest.mark.parametrize("dims", [(2,), (2, 2), (2, 2, 2), (3,), (3, 3), (2, 3)])
def test_compute_observables_random_channel(seed, dims):
    """Test observable computation for random channels against qutip."""
    key = jax.random.key(seed)
    d = int(np.prod(dims))
    kraus_rank = d

    num_states = 10
    num_observables = 8

    choi = qx.random_choi(dims=(dims, dims), rank=kraus_rank, key=key)
    kraus_map = qx.choi_to_kraus(choi)
    superop = qx.choi_to_superop(choi)
    pl = qx.choi_to_pauli_liouville(choi)

    # Generate random input states
    states = qx.random_density_matrix(rank=kraus_rank, dims=dims, key=key, size=(num_states,))

    # Generate random Hermitian observables from the Hermitian operator basis
    basis = n_qudit_herm_basis(dims)
    basis_matrices = basis.matrix
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(basis_matrices), num_observables, replace=True)
    obs_mats = basis_matrices[indices]
    observables = qx.Observable.from_matrix(obs_mats, (dims, dims))

    # Compute with JAX implementations
    observables_choi = qx.compute_choi_observables_from_states(choi, states, observables)
    observables_kraus = qx.compute_kraus_observables_from_states(kraus_map, states, observables)
    observables_superop = qx.compute_superop_observables_from_states(superop, states, observables)
    observables_pl = qx.compute_pauli_liouville_observables_from_states(pl, states, observables)

    # Compute with qutip
    # JAX functions compute Re(Tr(O† E(ρ))), so match with qutip using obs.dag()
    dims_list = list(dims)
    dims_super = [[dims_list, dims_list], [dims_list, dims_list]]
    dims_state = [dims_list, dims_list]
    dims_obs = [dims_list, dims_list]
    super_qutip = qt.Qobj(superop.matrix, superrep="super", dims=dims_super)
    states_qutip = [qt.Qobj(rho, dims=dims_state) for rho in states.matrix]
    observables_qutip = [qt.Qobj(obs, dims=dims_obs).tidyup() for obs in observables.matrix]

    expected_observables = []
    for i, rho in enumerate(states_qutip):
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
@pytest.mark.parametrize("dims", [(2, 2), (2, 2, 2), (2, 2, 2, 2), (3, 3), (3, 3, 3), (2, 3), (2, 3, 2)])
def test_partial_trace(seed, dims):
    """Test that our partial trace implementation matches qutip's."""
    key = jax.random.key(seed)
    d = int(np.prod(dims))
    num_qudits = len(dims)
    rank = d
    rho = qx.random_density_matrix(rank=rank, dims=dims, key=key)

    # Convert the density matrix to a qutip object
    rho_qobj = rho._to_qobj()

    # Choose random subsystems to trace out (trace out half the qudits)
    subsystems = tuple(jax.random.choice(key, num_qudits, (num_qudits // 2,), replace=False).tolist())

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
@pytest.mark.parametrize("dims", [(2,), (2, 2), (2, 2, 2), (3,), (3, 3), (2, 3)])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((5,), (5,)),
        ((3, 4), (3, 4)),
        ((3, 4), ()),
    ],
)
def test_apply_superoperator_to_density_matrix(seed, dims, ensemble_size):
    """Test the application of superoperators to density matrices."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    d = int(np.prod(dims))
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
@pytest.mark.parametrize("dims", [(2,), (2, 2), (2, 2, 2), (3,), (3, 3), (2, 3)])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((5,), (5,)),
        ((3, 4), (3, 4)),
        ((3, 4), ()),
    ],
)
def test_apply_operator_to_state_vector(seed, dims, ensemble_size):
    """Test the application of superoperators to density matrices."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
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


class TestTargetedApplySuperop:
    """Tests for targeted_apply_superop."""

    def test_single_qubit_gate_on_first(self):
        """X gate as superop on qubit 0."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        reference_operator = qx.gates.X | qx.gates.I | qx.gates.I
        rho_reference = reference_operator @ initial_state
        rho_targeted = qx.targeted_apply_superop(qx.unitary_to_superop(qx.gates.X), initial_state, (0,))
        assert rho_targeted == rho_reference

    def test_single_qubit_gate_on_last(self):
        """X gate as superop on qubit 2."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        reference_operator = qx.gates.I | qx.gates.I | qx.gates.X
        rho_reference = reference_operator @ initial_state
        rho_targeted = qx.targeted_apply_superop(qx.unitary_to_superop(qx.gates.X), initial_state, (2,))
        assert rho_targeted == rho_reference

    def test_two_qubit_gate_swapped(self):
        """CNOT as superop on qubits 1, 0 (swapped order)."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        reference_operator = (qx.gates.SWAP @ qx.gates.CNOT @ qx.gates.SWAP) | qx.gates.I
        rho_reference = reference_operator @ initial_state
        rho_targeted = qx.targeted_apply_superop(qx.unitary_to_superop(qx.gates.CNOT), initial_state, (1, 0))
        assert rho_targeted == rho_reference

    def test_two_qubit_gate_non_adjacent(self):
        """CNOT as superop on qubits 0, 2 (non-adjacent)."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        reference_operator = (qx.gates.I | qx.gates.SWAP) @ (qx.gates.CNOT | qx.gates.I) @ (qx.gates.I | qx.gates.SWAP)
        rho_reference = reference_operator @ initial_state
        rho_targeted = qx.targeted_apply_superop(qx.unitary_to_superop(qx.gates.CNOT), initial_state, (0, 2))
        assert rho_targeted == rho_reference

    def test_depolarizing_channel(self):
        """Depolarizing channel as superop on qubit 1."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        s = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(2,))
        reference_operator = qx.gates.I | s | qx.gates.I
        rho_reference = reference_operator @ initial_state
        rho_targeted = qx.targeted_apply_superop(s, initial_state, (1,))
        assert rho_targeted == rho_reference

    def test_full_system_channel(self):
        """Depolarizing channel as superop on all qubits."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        s = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(2, 2, 2))
        rho_reference = s @ initial_state
        rho_targeted = qx.targeted_apply_superop(s, initial_state, (0, 1, 2))
        assert rho_targeted == rho_reference

    @pytest.mark.parametrize("seed", [90573, 42])
    @pytest.mark.parametrize("dims", [(2, 2, 2), (3, 3, 3), (2, 3, 2), (3, 2), (2, 3)])
    def test_general_qudits(self, seed, dims):
        """Test targeted apply for general qudit systems using random channels."""
        key = jax.random.key(seed)
        d_total = int(np.prod(dims))
        initial_state = qx.random_density_matrix(d_total, dims, key)

        for target_idx in range(len(dims)):
            key, subkey = jax.random.split(key)
            d_sub = dims[target_idx]
            channel = qx.random_choi(dims=((d_sub,), (d_sub,)), rank=d_sub, key=subkey)
            superop = qx.choi_to_superop(channel)

            parts = []
            for j, dj in enumerate(dims):
                if j == target_idx:
                    parts.append(superop)
                else:
                    parts.append(_identity_superop(dj))
            reference = reduce(lambda a, b: a | b, parts)
            rho_reference = reference @ initial_state
            rho_targeted = qx.targeted_apply_superop(superop, initial_state, (target_idx,))
            assert jnp.allclose(rho_targeted.matrix, rho_reference.matrix, atol=1e-6)

    @pytest.mark.parametrize("seed", [90573, 42])
    @pytest.mark.parametrize(
        "dims, subsystem",
        [
            ((3, 3), (0,)),
            ((3, 3), (1,)),
            ((3, 2, 3), (0,)),
            ((3, 2, 3), (2,)),
        ],
    )
    def test_promotion(self, seed, dims, subsystem):
        """Test auto-promotion when operator dims are smaller than target subsystem dims."""
        key = jax.random.key(seed)
        d_total = int(np.prod(dims))
        initial_state = qx.random_density_matrix(d_total, dims, key)

        key, subkey = jax.random.split(key)
        qubit_channel = qx.random_choi(dims=((2,), (2,)), rank=2, key=subkey)
        superop = qx.choi_to_superop(qubit_channel)

        rho_targeted = qx.targeted_apply_superop(superop, initial_state, subsystem)

        target_dim = dims[subsystem[0]]
        superop_promoted = qx.promote(superop, (target_dim,))
        parts = []
        for j, dj in enumerate(dims):
            if j == subsystem[0]:
                parts.append(superop_promoted)
            else:
                parts.append(_identity_superop(dj))
        reference = reduce(lambda a, b: a | b, parts)
        rho_reference = reference @ initial_state
        assert jnp.allclose(rho_targeted.matrix, rho_reference.matrix, atol=1e-6)


class TestTargetedApplyKrausMap:
    """Tests for targeted_apply_kraus_map."""

    def test_single_qubit_gate_on_first(self):
        """X gate as Kraus map on qubit 0."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        reference_operator = qx.gates.X | qx.gates.I | qx.gates.I
        rho_reference = reference_operator @ initial_state
        rho_targeted = qx.targeted_apply_kraus_map(qx.unitary_to_kraus_map(qx.gates.X), initial_state, (0,))
        assert rho_targeted == rho_reference

    def test_single_qubit_gate_on_last(self):
        """X gate as Kraus map on qubit 2."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        reference_operator = qx.gates.I | qx.gates.I | qx.gates.X
        rho_reference = reference_operator @ initial_state
        rho_targeted = qx.targeted_apply_kraus_map(qx.unitary_to_kraus_map(qx.gates.X), initial_state, (2,))
        assert rho_targeted == rho_reference

    def test_two_qubit_gate_swapped(self):
        """CNOT as Kraus map on qubits 1, 0 (swapped order)."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        reference_operator = (qx.gates.SWAP @ qx.gates.CNOT @ qx.gates.SWAP) | qx.gates.I
        rho_reference = reference_operator @ initial_state
        rho_targeted = qx.targeted_apply_kraus_map(qx.unitary_to_kraus_map(qx.gates.CNOT), initial_state, (1, 0))
        assert rho_targeted == rho_reference

    def test_two_qubit_gate_non_adjacent(self):
        """CNOT as Kraus map on qubits 0, 2 (non-adjacent)."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        reference_operator = (qx.gates.I | qx.gates.SWAP) @ (qx.gates.CNOT | qx.gates.I) @ (qx.gates.I | qx.gates.SWAP)
        rho_reference = reference_operator @ initial_state
        rho_targeted = qx.targeted_apply_kraus_map(qx.unitary_to_kraus_map(qx.gates.CNOT), initial_state, (0, 2))
        assert rho_targeted == rho_reference

    def test_depolarizing_channel(self):
        """Depolarizing channel as Kraus map on qubit 1."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        s = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(2,))
        reference_operator = qx.gates.I | s | qx.gates.I
        rho_reference = reference_operator @ initial_state
        rho_targeted = qx.targeted_apply_kraus_map(qx.superop_to_kraus(s), initial_state, (1,))
        assert rho_targeted == rho_reference

    def test_full_system_channel(self):
        """Depolarizing channel as Kraus map on all qubits."""
        key = jax.random.key(90573)
        initial_state = qx.random_density_matrix(3, (2, 2, 2), key)
        s = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(2, 2, 2))
        rho_reference = s @ initial_state
        rho_targeted = qx.targeted_apply_kraus_map(qx.superop_to_kraus(s), initial_state, (0, 1, 2))
        assert rho_targeted == rho_reference

    @pytest.mark.parametrize("seed", [90573, 42])
    @pytest.mark.parametrize("dims", [(2, 2, 2), (3, 3, 3), (2, 3, 2), (3, 2), (2, 3)])
    def test_general_qudits(self, seed, dims):
        """Test targeted apply for general qudit systems using random channels."""
        key = jax.random.key(seed)
        d_total = int(np.prod(dims))
        initial_state = qx.random_density_matrix(d_total, dims, key)

        for target_idx in range(len(dims)):
            key, subkey = jax.random.split(key)
            d_sub = dims[target_idx]
            channel = qx.random_choi(dims=((d_sub,), (d_sub,)), rank=d_sub, key=subkey)
            kraus_map = qx.choi_to_kraus(channel)
            superop = qx.choi_to_superop(channel)

            parts = []
            for j, dj in enumerate(dims):
                if j == target_idx:
                    parts.append(superop)
                else:
                    parts.append(_identity_superop(dj))
            reference = reduce(lambda a, b: a | b, parts)
            rho_reference = reference @ initial_state
            rho_targeted = qx.targeted_apply_kraus_map(kraus_map, initial_state, (target_idx,))
            assert jnp.allclose(rho_targeted.matrix, rho_reference.matrix, atol=1e-6)

    @pytest.mark.parametrize("seed", [90573, 42])
    @pytest.mark.parametrize(
        "dims, subsystem",
        [
            ((3, 3), (0,)),
            ((3, 3), (1,)),
            ((3, 2, 3), (0,)),
            ((3, 2, 3), (2,)),
        ],
    )
    def test_promotion(self, seed, dims, subsystem):
        """Test auto-promotion when operator dims are smaller than target subsystem dims."""
        key = jax.random.key(seed)
        d_total = int(np.prod(dims))
        initial_state = qx.random_density_matrix(d_total, dims, key)

        key, subkey = jax.random.split(key)
        qubit_channel = qx.random_choi(dims=((2,), (2,)), rank=2, key=subkey)
        kraus_map = qx.choi_to_kraus(qubit_channel)

        rho_targeted = qx.targeted_apply_kraus_map(kraus_map, initial_state, subsystem)

        target_dim = dims[subsystem[0]]
        superop = qx.choi_to_superop(qubit_channel)
        superop_promoted = qx.promote(superop, (target_dim,))
        parts = []
        for j, dj in enumerate(dims):
            if j == subsystem[0]:
                parts.append(superop_promoted)
            else:
                parts.append(_identity_superop(dj))
        reference = reduce(lambda a, b: a | b, parts)
        rho_reference = reference @ initial_state
        assert jnp.allclose(rho_targeted.matrix, rho_reference.matrix, atol=1e-6)


class TestTargetedApplyUnitary:
    """Tests for targeted_apply_unitary."""

    def test_single_qubit_gate_on_first(self):
        """X gate on qubit 0."""
        key = jax.random.key(90573)
        initial_state = qx.random_state_vector((2, 2, 2), key)
        reference_operator = qx.gates.X | qx.gates.I | qx.gates.I
        psi_reference = reference_operator @ initial_state
        psi_targeted = qx.targeted_apply_unitary(qx.gates.X, initial_state, (0,))
        assert psi_targeted == psi_reference

    def test_single_qubit_gate_on_last(self):
        """X gate on qubit 2."""
        key = jax.random.key(90573)
        initial_state = qx.random_state_vector((2, 2, 2), key)
        reference_operator = qx.gates.I | qx.gates.I | qx.gates.X
        psi_reference = reference_operator @ initial_state
        psi_targeted = qx.targeted_apply_unitary(qx.gates.X, initial_state, (2,))
        assert psi_targeted == psi_reference

    def test_two_qubit_gate_swapped(self):
        """CNOT on qubits 1, 0 (swapped order)."""
        key = jax.random.key(90573)
        initial_state = qx.random_state_vector((2, 2, 2), key)
        reference_operator = (qx.gates.SWAP @ qx.gates.CNOT @ qx.gates.SWAP) | qx.gates.I
        psi_reference = reference_operator @ initial_state
        psi_targeted = qx.targeted_apply_unitary(qx.gates.CNOT, initial_state, (1, 0))
        assert psi_targeted == psi_reference

    def test_two_qubit_gate_non_adjacent(self):
        """CNOT on qubits 0, 2 (non-adjacent)."""
        key = jax.random.key(90573)
        initial_state = qx.random_state_vector((2, 2, 2), key)
        reference_operator = (qx.gates.I | qx.gates.SWAP) @ (qx.gates.CNOT | qx.gates.I) @ (qx.gates.I | qx.gates.SWAP)
        psi_reference = reference_operator @ initial_state
        psi_targeted = qx.targeted_apply_unitary(qx.gates.CNOT, initial_state, (0, 2))
        assert psi_targeted == psi_reference

    @pytest.mark.parametrize("seed", [90573, 42])
    @pytest.mark.parametrize("dims", [(2, 2, 2), (3, 3, 3), (2, 3, 2), (3, 2), (2, 3)])
    def test_general_qudits(self, seed, dims):
        """Test targeted apply for general qudit systems using random unitaries."""
        key = jax.random.key(seed)
        initial_state = qx.random_state_vector(dims, key)

        for target_idx in range(len(dims)):
            key, subkey = jax.random.split(key)
            d_sub = dims[target_idx]
            unitary = qx.random_unitary(dims=((d_sub,), (d_sub,)), key=subkey)

            parts = []
            for j, dj in enumerate(dims):
                if j == target_idx:
                    parts.append(unitary)
                else:
                    parts.append(_identity_unitary(dj))
            reference = reduce(lambda a, b: a | b, parts)
            psi_reference = reference @ initial_state
            psi_targeted = qx.targeted_apply_unitary(unitary, initial_state, (target_idx,))
            assert jnp.allclose(psi_targeted.matrix, psi_reference.matrix, atol=1e-6)

    @pytest.mark.parametrize("seed", [90573, 42])
    @pytest.mark.parametrize(
        "dims, subsystem",
        [
            ((3, 3), (0,)),
            ((3, 3), (1,)),
            ((3, 2, 3), (0,)),
            ((3, 2, 3), (2,)),
        ],
    )
    def test_promotion(self, seed, dims, subsystem):
        """Test auto-promotion when operator dims are smaller than target subsystem dims."""
        key = jax.random.key(seed)
        initial_state = qx.random_state_vector(dims, key)

        key, subkey = jax.random.split(key)
        qubit_unitary = qx.random_unitary(dims=((2,), (2,)), key=subkey)

        psi_targeted = qx.targeted_apply_unitary(qubit_unitary, initial_state, subsystem)

        target_dim = dims[subsystem[0]]
        unitary_promoted = qx.promote(qubit_unitary, (target_dim,))
        parts = []
        for j, dj in enumerate(dims):
            if j == subsystem[0]:
                parts.append(unitary_promoted)
            else:
                parts.append(_identity_unitary(dj))
        reference = reduce(lambda a, b: a | b, parts)
        psi_reference = reference @ initial_state
        assert jnp.allclose(psi_targeted.matrix, psi_reference.matrix, atol=1e-6)


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
    s_1q = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(2,))

    # Build ensemble of superops if needed (stack copies with different noise params)
    if ens_op:
        keys = jax.random.split(subkey, num=reduce(lambda a, b: a * b, ens_op, 1))
        noise_params = jnp.linspace(0.01, 0.1, len(keys))
        superop_mats = jnp.stack(
            [qx.depolarizing_channel_superoperator(jnp.asarray(p), dims=(2,)).data for p in noise_params]
        )
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
@pytest.mark.parametrize("dims", [(2, 2, 2), (3, 3, 3), (2, 3, 2), (2, 3)])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((3,), ()),
        ((), (3,)),
        ((3,), (3,)),
    ],
)
def test_targeted_apply_superop_ensemble_general(seed, dims, ensemble_size):
    """Test targeted_apply_superop with ensembles for general qudit systems."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    ens_op, ens_rho = ensemble_size

    d_total = int(np.prod(dims))
    initial_state = qx.random_density_matrix(d_total, dims, key, size=ens_rho)

    # Target the last subsystem
    target = len(dims) - 1
    target_dim = dims[target]

    s_1d = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(target_dim,))

    if ens_op:
        noise_params = jnp.linspace(0.01, 0.1, reduce(lambda a, b: a * b, ens_op, 1))
        superop_mats = jnp.stack(
            [qx.depolarizing_channel_superoperator(jnp.asarray(p), dims=(target_dim,)).data for p in noise_params]
        )
        superop_mats = superop_mats.reshape(ens_op + superop_mats.shape[1:])
        s_ensemble = qx.SuperOp(data=superop_mats, num_qubits=1)
    else:
        s_ensemble = s_1d

    rho_targeted = qx.targeted_apply_superop(s_ensemble, initial_state, (target,))

    broadcast_ens = jnp.broadcast_shapes(ens_op, ens_rho)
    assert rho_targeted.ensemble_size == broadcast_ens

    # Build reference via tensor product
    parts = []
    for j, dj in enumerate(dims):
        if j == target:
            parts.append(s_ensemble)
        else:
            parts.append(_identity_superop(dj))
    reference_operator = reduce(lambda a, b: a | b, parts)
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
    s_1q = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(2,))
    k_1q = qx.superop_to_kraus(s_1q)

    # Build ensemble of kraus maps if needed
    if ens_op:
        keys = jax.random.split(subkey, num=reduce(lambda a, b: a * b, ens_op, 1))
        noise_params = jnp.linspace(0.01, 0.1, len(keys))
        kraus_list = [
            qx.superop_to_kraus(qx.depolarizing_channel_superoperator(jnp.asarray(p), dims=(2,))).data
            for p in noise_params
        ]
        kraus_mats = jnp.stack(kraus_list)
        kraus_mats = kraus_mats.reshape(ens_op + kraus_mats.shape[1:])
        k_ensemble = qx.KrausMap(data=kraus_mats, num_qubits=1)
        # Also build corresponding superops for the reference
        superop_list = [qx.depolarizing_channel_superoperator(jnp.asarray(p), dims=(2,)).data for p in noise_params]
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


@pytest.mark.parametrize("seed", [90573, 42])
@pytest.mark.parametrize("dims", [(2, 2, 2), (3, 3, 3), (2, 3, 2), (2, 3)])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((3,), ()),
        ((), (3,)),
        ((3,), (3,)),
    ],
)
def test_targeted_apply_kraus_map_ensemble_general(seed, dims, ensemble_size):
    """Test targeted_apply_kraus_map with ensembles for general qudit systems."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    ens_op, ens_rho = ensemble_size

    d_total = int(np.prod(dims))
    initial_state = qx.random_density_matrix(d_total, dims, key, size=ens_rho)

    target = len(dims) - 1
    target_dim = dims[target]

    s_1d = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(target_dim,))
    k_1d = qx.superop_to_kraus(s_1d)

    if ens_op:
        noise_params = jnp.linspace(0.01, 0.1, reduce(lambda a, b: a * b, ens_op, 1))
        kraus_list = [
            qx.superop_to_kraus(qx.depolarizing_channel_superoperator(jnp.asarray(p), dims=(target_dim,))).data
            for p in noise_params
        ]
        kraus_mats = jnp.stack(kraus_list)
        kraus_mats = kraus_mats.reshape(ens_op + kraus_mats.shape[1:])
        k_ensemble = qx.KrausMap(data=kraus_mats, num_qubits=1)
        superop_list = [
            qx.depolarizing_channel_superoperator(jnp.asarray(p), dims=(target_dim,)).data for p in noise_params
        ]
        superop_mats = jnp.stack(superop_list).reshape(ens_op + jnp.stack(superop_list).shape[1:])
        s_ensemble = qx.SuperOp(data=superop_mats, num_qubits=1)
    else:
        k_ensemble = k_1d
        s_ensemble = s_1d

    rho_targeted = qx.targeted_apply_kraus_map(k_ensemble, initial_state, (target,))

    broadcast_ens = jnp.broadcast_shapes(ens_op, ens_rho)
    assert rho_targeted.ensemble_size == broadcast_ens

    # Build reference via tensor product (using superop)
    parts = []
    for j, dj in enumerate(dims):
        if j == target:
            parts.append(s_ensemble)
        else:
            parts.append(_identity_superop(dj))
    reference_operator = reduce(lambda a, b: a | b, parts)
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


@pytest.mark.parametrize("seed", [90573, 42])
@pytest.mark.parametrize("dims", [(2, 2, 2), (3, 3, 3), (2, 3, 2), (2, 3)])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((3,), ()),
        ((), (3,)),
        ((3,), (3,)),
    ],
)
def test_targeted_apply_unitary_ensemble_general(seed, dims, ensemble_size):
    """Test targeted_apply_unitary with ensembles for general qudit systems."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    ens_op, ens_psi = ensemble_size

    initial_state = qx.random_state_vector(dims, key, size=ens_psi)

    target = len(dims) - 1
    target_dim = dims[target]

    if ens_op:
        u_ensemble = qx.random_unitary(dims=((target_dim,), (target_dim,)), key=subkey, size=ens_op)
    else:
        u_ensemble = qx.random_unitary(dims=((target_dim,), (target_dim,)), key=subkey)

    psi_targeted = qx.targeted_apply_unitary(u_ensemble, initial_state, (target,))

    broadcast_ens = jnp.broadcast_shapes(ens_op, ens_psi)
    assert psi_targeted.ensemble_size == broadcast_ens

    # Build reference via tensor product
    parts = []
    for j, dj in enumerate(dims):
        if j == target:
            parts.append(u_ensemble)
        else:
            parts.append(_identity_unitary(dj))
    reference_operator = reduce(lambda a, b: a | b, parts)
    psi_reference = reference_operator @ initial_state
    assert jnp.allclose(psi_targeted.matrix, psi_reference.matrix, atol=1e-6)


@pytest.mark.parametrize(
    "dims, gate, subsystem",
    [
        # Qubit cases
        ((2, 2, 2), qx.gates.X, (0,)),
        ((2, 2, 2), qx.gates.X, (2,)),
        ((2, 2, 2), qx.gates.CNOT, (1, 0)),
        ((2, 2, 2), qx.gates.CNOT, (0, 2)),
        # Qutrit cases
        ((3, 3, 3), qx.gates.TX, (0,)),
        ((3, 3, 3), qx.gates.TX, (2,)),
        ((3, 3, 3), qx.gates.TSWAP, (0, 1)),
        ((3, 3, 3), qx.gates.TSWAP, (1, 2)),
        # Mixed qubit-qutrit cases
        ((2, 3), qx.gates.X, (0,)),
        ((2, 3), qx.gates.TX, (1,)),
        ((3, 2, 3), qx.gates.TX, (0,)),
        ((3, 2, 3), qx.gates.X, (1,)),
        ((3, 2, 3), qx.gates.TX, (2,)),
    ],
)
def test_targeted_apply_kraus_map_trajectory_unitary(dims, gate, subsystem):
    """A single-operator Kraus map (unitary) should match targeted_apply_unitary exactly."""
    seed = 90573
    key = jax.random.key(seed)
    key, sample_key = jax.random.split(key)
    initial_state = qx.random_state_vector(dims, key)

    kraus = qx.unitary_to_kraus_map(gate)
    psi_trajectory = qx.targeted_apply_kraus_map_trajectory(kraus, initial_state, sample_key, subsystem)
    psi_reference = qx.targeted_apply_unitary(gate, initial_state, subsystem)
    # Compare via fidelity since choi_to_kraus can introduce an arbitrary global phase
    assert jnp.allclose(qx.fidelity(psi_trajectory, psi_reference), 1.0, atol=1e-6)


def test_targeted_apply_kraus_map_trajectory_normalization():
    """Output states should always be normalized."""
    key = jax.random.key(42)
    key, sample_key = jax.random.split(key)
    initial_state = qx.random_state_vector((2, 2, 2), key)

    # Depolarizing channel on qubit 1
    s = qx.depolarizing_channel_superoperator(jnp.array(0.1), dims=(2,))
    kraus = qx.superop_to_kraus(s)
    psi_out = qx.targeted_apply_kraus_map_trajectory(kraus, initial_state, sample_key, (1,))
    norm = jnp.sum(jnp.abs(psi_out.matrix) ** 2)
    assert jnp.allclose(norm, 1.0, atol=1e-6)


@pytest.mark.parametrize("seed", [42, 123, 7])
@pytest.mark.parametrize(
    "dims, target, noise_p",
    [
        # Qubit systems
        ((2, 2), (0,), 0.1),
        # Qutrit systems
        ((3, 3), (0,), 0.1),
        # Mixed qubit-qutrit
        ((2, 3), (1,), 0.1),
        ((3, 2), (0,), 0.1),
    ],
)
def test_targeted_apply_kraus_map_trajectory_statistical_convergence(seed, dims, target, noise_p):
    """Averaging |ψ⟩⟨ψ| over many trajectories should converge to ∑ K_i ρ K_i†."""
    key = jax.random.key(seed)
    key, state_key = jax.random.split(key)
    initial_state = qx.random_state_vector(dims, state_key)

    target_dims = tuple(dims[t] for t in target)
    s = qx.depolarizing_channel_superoperator(jnp.asarray(noise_p), dims=target_dims)
    kraus = qx.superop_to_kraus(s)

    # Reference: apply Kraus map to density matrix
    rho_initial = qx.promote_state_vector_to_density_matrix(initial_state)
    rho_reference = qx.targeted_apply_kraus_map(kraus, rho_initial, target)

    # Monte Carlo: use an ensemble of keys to get all trajectories at once
    n_trajectories = 5000
    sample_keys = jax.random.split(key, num=n_trajectories)

    psi_out = qx.targeted_apply_kraus_map_trajectory(kraus, initial_state, sample_keys, target)
    # psi_out.matrix has shape (n_trajectories, d) — compute |ψ⟩⟨ψ| and average
    rho_avg = jnp.mean(psi_out.matrix[:, :, None] * psi_out.matrix[:, None, :].conj(), axis=0)
    rho_avg = qx.DensityMatrix.from_matrix(rho_avg, dims)

    assert qx.fidelity(rho_avg, rho_reference) > 0.99

    # Chain two channels: 1-qudit depolarizing on last subsystem,
    # then multi-qudit depolarizing on all subsystems
    key, state_key2 = jax.random.split(key)
    initial_state_full = qx.random_state_vector(dims, state_key2)

    last = len(dims) - 1
    s_1d = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(dims[last],))
    kraus_1d = qx.superop_to_kraus(s_1d)
    s_all = qx.depolarizing_channel_superoperator(jnp.array(0.08), dims=dims)
    kraus_all = qx.superop_to_kraus(s_all)

    # Reference: apply both channels to density matrix
    rho_initial_full = qx.promote_state_vector_to_density_matrix(initial_state_full)
    rho_ref_full = qx.targeted_apply_kraus_map(kraus_1d, rho_initial_full, (last,))
    rho_ref_full = qx.targeted_apply_kraus_map(kraus_all, rho_ref_full, tuple(range(len(dims))))

    # Monte Carlo: apply both channels sequentially per trajectory
    key, key1, key2 = jax.random.split(key, 3)
    sample_keys_1 = jax.random.split(key1, num=n_trajectories)
    sample_keys_2 = jax.random.split(key2, num=n_trajectories)

    psi_mid = qx.targeted_apply_kraus_map_trajectory(kraus_1d, initial_state_full, sample_keys_1, (last,))
    psi_out_full = qx.targeted_apply_kraus_map_trajectory(kraus_all, psi_mid, sample_keys_2, tuple(range(len(dims))))

    rho_avg_full = jnp.mean(psi_out_full.matrix[:, :, None] * psi_out_full.matrix[:, None, :].conj(), axis=0)
    rho_avg_full = qx.DensityMatrix.from_matrix(rho_avg_full, dims)

    assert qx.fidelity(rho_avg_full, rho_ref_full) > 0.99


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
            qx.superop_to_kraus(qx.depolarizing_channel_superoperator(jnp.asarray(p), dims=(2,))).data
            for p in noise_params
        ]
        kraus_mats = jnp.stack(kraus_list).reshape(ens_op + jnp.stack(kraus_list).shape[1:])
        k_ensemble = qx.KrausMap(data=kraus_mats, num_qubits=1)
    else:
        k_ensemble = qx.superop_to_kraus(qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(2,)))

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


@pytest.mark.parametrize("seed", [90573, 42])
@pytest.mark.parametrize(
    "ens_op, ens_psi, ens_key",
    [
        ((3,), (), (3, 5)),  # key has more dims than op/psi ensemble
        ((), (), (3, 5)),  # only key has ensemble dims (2D)
        ((3,), (3,), (3, 5)),  # op and psi match, key has extra trailing dim
        ((3, 1), (), (3, 5)),  # 2D op broadcasts with 2D key
    ],
)
def test_targeted_apply_kraus_map_trajectory_key_extra_dims(seed, ens_op, ens_psi, ens_key):
    """Test that trajectory Kraus application broadcasts key dims beyond the op/psi ensemble."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)

    initial_state = qx.random_state_vector((2, 2, 2), key, size=ens_psi)

    # Build ensemble of Kraus maps if needed
    if ens_op:
        noise_params = jnp.linspace(0.01, 0.1, reduce(lambda a, b: a * b, ens_op, 1))
        kraus_list = [
            qx.superop_to_kraus(qx.depolarizing_channel_superoperator(jnp.asarray(p), dims=(2,))).data
            for p in noise_params
        ]
        kraus_mats = jnp.stack(kraus_list).reshape(ens_op + jnp.stack(kraus_list).shape[1:])
        k_ensemble = qx.KrausMap(data=kraus_mats, num_qubits=1)
    else:
        k_ensemble = qx.superop_to_kraus(qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(2,)))

    # Build ensemble of keys
    n_keys = reduce(lambda a, b: a * b, ens_key, 1)
    sample_keys = jax.random.split(subkey, num=n_keys).reshape(ens_key)

    psi_out = qx.targeted_apply_kraus_map_trajectory(k_ensemble, initial_state, sample_keys, (1,))

    # Check ensemble size matches broadcast.
    # The key may have more dims than the op/psi broadcast, introducing extra "samples" dims.
    ens_intermediate = jnp.broadcast_shapes(ens_op, ens_psi)
    if len(ens_key) > len(ens_intermediate):
        n_extra = len(ens_key) - len(ens_intermediate)
        ens_intermediate = ens_intermediate + (1,) * n_extra
    broadcast_ens = jnp.broadcast_shapes(ens_intermediate, ens_key)
    assert psi_out.ensemble_size == broadcast_ens

    # Check normalization for all ensemble elements
    norms = jnp.sum(jnp.abs(psi_out.matrix) ** 2, axis=-1)
    assert jnp.allclose(norms, jnp.ones_like(norms), atol=1e-6)


@pytest.mark.parametrize("seed", [42, 123])
@pytest.mark.parametrize("dims", [(3, 3), (2, 3), (3, 2, 3)])
def test_targeted_apply_kraus_map_trajectory_normalization_general(seed, dims):
    """Output states should always be normalized for general qudit systems."""
    key = jax.random.key(seed)
    key, sample_key = jax.random.split(key)
    initial_state = qx.random_state_vector(dims, key)

    target = len(dims) - 1
    target_dim = dims[target]
    s = qx.depolarizing_channel_superoperator(jnp.array(0.1), dims=(target_dim,))
    kraus = qx.superop_to_kraus(s)
    psi_out = qx.targeted_apply_kraus_map_trajectory(kraus, initial_state, sample_key, (target,))
    norm = jnp.sum(jnp.abs(psi_out.matrix) ** 2)
    assert jnp.allclose(norm, 1.0, atol=1e-6)


@pytest.mark.parametrize("seed", [90573, 42])
@pytest.mark.parametrize("dims", [(3, 3, 3), (2, 3, 2), (2, 3)])
@pytest.mark.parametrize("ens_key", [(), (3,)])
@pytest.mark.parametrize("ens_psi", [(), (3,)])
@pytest.mark.parametrize("ens_op", [(), (3,)])
def test_targeted_apply_kraus_map_trajectory_ensemble_general(seed, dims, ens_op, ens_psi, ens_key):
    """Test ensemble broadcasting for trajectory Kraus application with general dims."""
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)

    initial_state = qx.random_state_vector(dims, key, size=ens_psi)

    target = len(dims) - 1
    target_dim = dims[target]

    if ens_op:
        noise_params = jnp.linspace(0.01, 0.1, reduce(lambda a, b: a * b, ens_op, 1))
        kraus_list = [
            qx.superop_to_kraus(qx.depolarizing_channel_superoperator(jnp.asarray(p), dims=(target_dim,))).data
            for p in noise_params
        ]
        kraus_mats = jnp.stack(kraus_list).reshape(ens_op + jnp.stack(kraus_list).shape[1:])
        k_ensemble = qx.KrausMap(data=kraus_mats, num_qubits=1)
    else:
        k_ensemble = qx.superop_to_kraus(qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=(target_dim,)))

    if ens_key:
        n_keys = reduce(lambda a, b: a * b, ens_key, 1)
        sample_keys = jax.random.split(subkey, num=n_keys).reshape(ens_key)
    else:
        sample_keys = subkey

    psi_out = qx.targeted_apply_kraus_map_trajectory(k_ensemble, initial_state, sample_keys, (target,))

    broadcast_ens = jnp.broadcast_shapes(ens_op, ens_psi, ens_key)
    assert psi_out.ensemble_size == broadcast_ens

    norms = jnp.sum(jnp.abs(psi_out.matrix) ** 2, axis=-1)
    assert jnp.allclose(norms, jnp.ones_like(norms), atol=1e-6)


# ======================================================================
# state_vector_rdm tests
# ======================================================================


@pytest.mark.parametrize("dims", [(2,), (2, 2), (2, 2, 2), (2, 3)])
def test_state_vector_rdm_full_system(dims):
    """When subsystem = all qudits, RDM should equal |ψ⟩⟨ψ|."""
    psi = qx.random_state_vector(dims=dims, key=jax.random.key(42))
    subsystem = tuple(range(len(dims)))
    rdm = qx.state_vector_reduced_density_matrix(psi, subsystem)
    expected = jnp.einsum("...i,...j->...ij", psi.matrix, psi.matrix.conj())
    assert jnp.allclose(rdm.matrix, expected, atol=1e-12)


@pytest.mark.parametrize(
    "dims,subsystem",
    [
        ((2, 2), (0,)),
        ((2, 2), (1,)),
        ((2, 2, 2), (0, 2)),
        ((2, 3), (0,)),
        ((2, 3), (1,)),
    ],
)
def test_state_vector_rdm_vs_partial_trace(dims, subsystem):
    """state_vector_rdm should match partial_trace on |ψ⟩⟨ψ|."""
    psi = qx.random_state_vector(dims=dims, key=jax.random.key(123))
    rdm = qx.state_vector_reduced_density_matrix(psi, subsystem)

    # Build full density matrix and partial trace
    rho_full = qx.DensityMatrix.from_matrix(jnp.einsum("i,j->ij", psi.matrix, psi.matrix.conj()), dims)
    rdm_ref = qx.partial_trace(rho_full, subsystem)
    assert jnp.allclose(rdm.matrix, rdm_ref.matrix, atol=1e-12)


def test_state_vector_rdm_ensemble():
    """state_vector_rdm should handle ensembled state vectors."""
    dims = (2, 2)
    ens = (3,)
    psi = qx.random_state_vector(dims=dims, key=jax.random.key(99), size=ens)
    rdm = qx.state_vector_reduced_density_matrix(psi, (0,))
    assert rdm.matrix.shape == ens + (2, 2)

    # Check each element matches the scalar version
    for i in range(ens[0]):
        psi_i = qx.StateVector(data=psi.data[i], num_qubits=len(dims))
        rdm_i = qx.state_vector_reduced_density_matrix(psi_i, (0,))
        assert jnp.allclose(rdm.matrix[i], rdm_i.matrix, atol=1e-12)


def test_state_vector_rdm_trace_one():
    """RDM should have trace 1 for normalized states."""
    psi = qx.random_state_vector(dims=(2, 2, 2), key=jax.random.key(7))
    rdm = qx.state_vector_reduced_density_matrix(psi, (1,))
    assert jnp.allclose(jnp.trace(rdm.matrix), 1.0, atol=1e-12)


def test_state_vector_rdm_positive_semidefinite():
    """RDM eigenvalues should be non-negative."""
    psi = qx.random_state_vector(dims=(2, 2, 2), key=jax.random.key(11))
    rdm = qx.state_vector_reduced_density_matrix(psi, (0, 2))
    eigvals = jnp.linalg.eigvalsh(rdm.matrix)
    assert jnp.all(eigvals >= -1e-12)


# ======================================================================
# Ensembled KrausMap trajectory tests
# ======================================================================


@pytest.mark.parametrize("ens_k", [(), (3,), (1,)])
@pytest.mark.parametrize("ens_psi", [(), (3,)])
@pytest.mark.parametrize("ens_key", [(), (3,)])
def test_trajectory_ensembled_kraus_shape(ens_k, ens_psi, ens_key):
    """Ensembled KrausMap should broadcast with psi and key ensemble dims."""
    dims = (2, 2)
    target = 0
    noise_p = 0.05
    seed = 42

    try:
        broadcast_ens = jnp.broadcast_shapes(ens_k, ens_psi, ens_key)
    except Exception:
        pytest.skip("incompatible broadcast shapes")

    # Build ensembled Kraus map by stacking
    if ens_k:
        noise_ps = jnp.full(ens_k, noise_p)
        kraus_list = []
        for i in range(ens_k[0]):
            s = qx.depolarizing_channel_superoperator(noise_ps[i], dims=(2,))
            k = qx.superop_to_kraus(s)
            kraus_list.append(k.data)
        kraus_data = jnp.stack(kraus_list)
        kraus = qx.KrausMap(data=kraus_data, num_qubits=1)
    else:
        s = qx.depolarizing_channel_superoperator(jnp.array(noise_p), dims=(2,))
        kraus = qx.superop_to_kraus(s)

    psi = qx.random_state_vector(dims=dims, key=jax.random.key(seed + 1), size=ens_psi)
    if ens_key:
        sample_keys = jax.random.split(jax.random.key(seed + 2), ens_key[0])
    else:
        sample_keys = jax.random.key(seed + 2)

    psi_out = qx.targeted_apply_kraus_map_trajectory(kraus, psi, sample_keys, (target,))
    assert psi_out.ensemble_size == broadcast_ens

    norms = jnp.sum(jnp.abs(psi_out.matrix) ** 2, axis=-1)
    assert jnp.allclose(norms, jnp.ones_like(norms), atol=1e-6)


def test_trajectory_ensembled_kraus_statistical_convergence():
    """Ensembled KrausMap with different noise levels should converge to correct channels."""
    dims = (2,)
    n_samples = 5000
    seed = 314

    # Two different noise levels in ensemble
    noise_levels = jnp.array([0.01, 0.2])
    kraus_list = []
    for p in noise_levels:
        s = qx.depolarizing_channel_superoperator(p, dims=(2,))
        k = qx.superop_to_kraus(s)
        kraus_list.append(k.data)
    kraus_data = jnp.stack(kraus_list)
    kraus_ens = qx.KrausMap(data=kraus_data, num_qubits=1)

    psi = qx.zero_state_vector(dims=(2,))
    keys = jax.random.split(jax.random.key(seed), n_samples)

    # Apply to each sample
    results = []
    for i in range(n_samples):
        out = qx.targeted_apply_kraus_map_trajectory(kraus_ens, psi, keys[i], (0,))
        results.append(out.data)
    results = jnp.stack(results)  # (n_samples, 2, 2)

    # For each ensemble element, compute average density matrix
    for e in range(2):
        rho_samples = jnp.einsum("si,sj->sij", results[:, e], results[:, e].conj())
        rho_avg = jnp.mean(rho_samples, axis=0)

        s = qx.depolarizing_channel_superoperator(noise_levels[e], dims=(2,))
        rho_expected = qx.apply_superop_to_density_matrix(
            s, qx.DensityMatrix.from_matrix(jnp.array([[1, 0], [0, 0]], dtype=complex), (2,))
        )
        assert jnp.allclose(rho_avg, rho_expected.matrix, atol=0.05), (
            f"Ensemble element {e} (p={noise_levels[e]}) did not converge"
        )


# ======================================================================
# QuantumInstrument targeted apply tests
# ======================================================================


class TestInstrumentTargetedApply:
    """Test targeted application of instruments to subsystems."""

    def test_measure_first_qubit_of_two(self):
        qi = qx.gates.MEASURE()
        rho = qx.zero_state_matrix(dims=(2, 2))
        key = jax.random.key(42)
        rho_outs, probs = qx.targeted_apply_instrument_to_density_matrix(qi, rho, subsystem=(0,))
        rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
        assert int(outcome) == 0
        assert rho_out.dims == (2, 2)

    def test_measure_second_qubit_of_two(self):
        qi = qx.gates.MEASURE()
        rho = basis_dm_multi((0, 1), (2, 2))
        key = jax.random.key(42)
        rho_outs, probs = qx.targeted_apply_instrument_to_density_matrix(qi, rho, subsystem=(1,))
        rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
        assert int(outcome) == 1


# ======================================================================
# QuantumInstrument ensemble / batch application tests
# ======================================================================


class TestInstrumentEnsembleApplication:
    """Test applying instruments to batches of states and with batched instruments."""

    def test_single_instrument_batch_of_states(self):
        """Apply one ideal measurement to an ensemble of density matrices."""
        qi = qx.gates.MEASURE()
        rho0 = jnp.array([[1, 0], [0, 0]], dtype=complex)
        rho1 = jnp.array([[0, 0], [0, 1]], dtype=complex)
        rho_batch = jnp.stack([rho0, rho1], axis=0)
        rho = DensityMatrix.from_matrix(rho_batch, (2,))

        keys = jax.random.split(jax.random.key(0), 2)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        rho_out, outcomes = qx.select_outcome(rho_outs, probs, keys)

        assert outcomes.shape == (2,)
        assert int(outcomes[0]) == 0
        assert int(outcomes[1]) == 1
        assert rho_out.matrix.shape == (2, 2, 2)

    def test_batch_outcomes_deterministic_basis_states(self):
        """Batch of all basis states gives deterministic matching outcomes."""
        d = 3
        qi = qx.gates.MEASURE(d)
        rho_list = [jnp.zeros((d, d), dtype=complex).at[k, k].set(1.0) for k in range(d)]
        rho_batch = jnp.stack(rho_list, axis=0)
        rho = DensityMatrix.from_matrix(rho_batch, (d,))

        keys = jax.random.split(jax.random.key(42), d)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        rho_out, outcomes = qx.select_outcome(rho_outs, probs, keys)

        assert outcomes.shape == (d,)
        for k in range(d):
            assert int(outcomes[k]) == k
            np.testing.assert_allclose(rho_out.matrix[k], rho_batch[k], atol=1e-10)

    def test_batch_post_measurement_states(self):
        """Post-measurement states for a batch of |+> inputs are projected correctly."""
        qi = qx.gates.MEASURE()
        plus = jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        rho_batch = jnp.broadcast_to(plus, (5, 2, 2))
        rho = DensityMatrix.from_matrix(rho_batch, (2,))

        keys = jax.random.split(jax.random.key(7), 5)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        rho_out, outcomes = qx.select_outcome(rho_outs, probs, keys)

        for i in range(5):
            o = int(outcomes[i])
            expected = jnp.zeros((2, 2), dtype=complex).at[o, o].set(1.0)
            np.testing.assert_allclose(rho_out.matrix[i], expected, atol=1e-10)

    def test_batch_instrument_single_state(self):
        """Apply an ensemble of instruments (different confusions) to one state."""
        fids = jnp.array([1.0, 0.8, 0.6])
        choi_list = []
        for fid in fids:
            cm = jnp.array([[fid, 1 - fid], [1 - fid, fid]])
            qi_single = qx.instrument_from_confusion_and_transition(cm, jnp.eye(2), dims=(2,))
            choi_list.append(qi_single.matrix)
        batch_mat = jnp.stack(choi_list, axis=0)
        qi_batch = qx.QuantumInstrument.from_matrix(batch_mat, ((2,), (2,)), (0,))

        assert qi_batch.ensemble_size == (3,)
        assert qi_batch.num_outcomes == 2

        rho0 = qx.zero_state_matrix(dims=(2,))
        keys = jax.random.split(jax.random.key(0), 3)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi_batch, rho0)
        rho_out, outcomes = qx.select_outcome(rho_outs, probs, keys)

        assert outcomes.shape == (3,)
        assert rho_out.matrix.shape == (3, 2, 2)
        assert int(outcomes[0]) == 0

    def test_batch_instrument_batch_state(self):
        """Matching ensemble dims: each instrument applied to corresponding state."""
        fids = [1.0, 1.0, 1.0]
        choi_list = []
        for fid in fids:
            cm = jnp.array([[fid, 1 - fid], [1 - fid, fid]])
            qi_s = qx.instrument_from_confusion_and_transition(cm, jnp.eye(2), dims=(2,))
            choi_list.append(qi_s.matrix)
        batch_mat = jnp.stack(choi_list, axis=0)
        qi_batch = qx.QuantumInstrument.from_matrix(batch_mat, ((2,), (2,)), (0,))

        rho0 = jnp.array([[1, 0], [0, 0]], dtype=complex)
        rho1 = jnp.array([[0, 0], [0, 1]], dtype=complex)
        rho_batch = jnp.stack([rho0, rho1, rho0], axis=0)
        rho = DensityMatrix.from_matrix(rho_batch, (2,))

        keys = jax.random.split(jax.random.key(99), 3)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi_batch, rho)
        rho_out, outcomes = qx.select_outcome(rho_outs, probs, keys)

        assert outcomes.shape == (3,)
        assert int(outcomes[0]) == 0
        assert int(outcomes[1]) == 1
        assert int(outcomes[2]) == 0

    def test_batch_state_vector_input(self):
        """Apply instrument to batch of state vectors."""
        qi = qx.gates.MEASURE()
        sv0 = jnp.array([1, 0], dtype=complex)
        sv1 = jnp.array([0, 1], dtype=complex)
        sv_batch = jnp.stack([sv0, sv1], axis=0)
        psi = StateVector.from_matrix(sv_batch, (2,))

        keys = jax.random.split(jax.random.key(5), 2)
        psi_out, outcomes = qx.apply_instrument_to_state_vector(qi, psi, keys)

        assert outcomes.shape == (2,)
        assert int(outcomes[0]) == 0
        assert int(outcomes[1]) == 1
        assert isinstance(psi_out, StateVector)

    def test_batch_targeted_apply(self):
        """Targeted apply with batch of 2-qubit states, measuring first qubit."""
        qi = qx.gates.MEASURE()
        rho00 = basis_dm_multi((0, 0), (2, 2)).matrix
        rho10 = basis_dm_multi((1, 0), (2, 2)).matrix
        rho_batch = jnp.stack([rho00, rho10], axis=0)
        rho = DensityMatrix.from_matrix(rho_batch, (2, 2))

        keys = jax.random.split(jax.random.key(3), 2)
        rho_outs, probs = qx.targeted_apply_instrument_to_density_matrix(qi, rho, subsystem=(0,))
        rho_out, outcomes = qx.select_outcome(rho_outs, probs, keys)

        assert outcomes.shape == (2,)
        assert int(outcomes[0]) == 0
        assert int(outcomes[1]) == 1
        assert rho_out.dims == (2, 2)
        assert rho_out.matrix.shape == (2, 4, 4)
