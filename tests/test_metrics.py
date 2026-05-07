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

"""Tests for JAX-based distance metrics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qutip
import quax as qx


# ============================================================================
# Tests for qx.fidelity
# ============================================================================


@pytest.mark.parametrize("num_qudits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_fidelity_pure_states(seed, qudit_dim, num_qudits):
    """Test qx.fidelity between pure states against reference implementation."""
    key = jax.random.key(seed)
    d = qudit_dim**num_qudits
    key1, key2, key3, key4 = jax.random.split(key, 4)
    psi = jax.random.normal(key1, (d,)) + 1j * jax.random.normal(key2, (d,))
    psi = psi / jnp.linalg.norm(psi)
    phi = jax.random.normal(key3, (d,)) + 1j * jax.random.normal(key4, (d,))
    phi = phi / jnp.linalg.norm(phi)

    # Convert to JAX arrays
    psi_jax = qx.StateVector.from_matrix(jnp.array(psi), (qudit_dim,) * num_qudits)
    phi_jax = qx.StateVector.from_matrix(jnp.array(phi), (qudit_dim,) * num_qudits)

    # Our qx.fidelity is the square of the standard definition used in qutip
    fid_qutip = qutip.fidelity(qutip.Qobj(psi[:, jnp.newaxis]), qutip.Qobj(phi[:, jnp.newaxis])) ** 2
    fid_jax = float(qx.fidelity(psi_jax, phi_jax))

    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), "JAX qx.fidelity does not match the square of qutip qx.fidelity."


@pytest.mark.parametrize("num_qudits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_fidelity_density_matrices(seed, qudit_dim, num_qudits):
    """Test qx.fidelity between density matrices against reference implementation."""
    key = jax.random.key(seed)
    d = qudit_dim**num_qudits
    # Create random density matrices
    key1, key2, key3, key4 = jax.random.split(key, 4)
    A = jax.random.normal(key1, (d, d)) + 1j * jax.random.normal(key2, (d, d))
    rho = A @ A.conj().T
    rho = rho / jnp.trace(rho)

    B = jax.random.normal(key3, (d, d)) + 1j * jax.random.normal(key4, (d, d))
    sigma = B @ B.conj().T
    sigma = sigma / jnp.trace(sigma)

    # Convert to JAX arrays
    rho_jax = qx.DensityMatrix.from_matrix(jnp.array(rho), (qudit_dim,) * num_qudits)
    sigma_jax = qx.DensityMatrix.from_matrix(jnp.array(sigma), (qudit_dim,) * num_qudits)

    # Compute fidelities
    # Our qx.fidelity is the square of the standard definition used in qutip
    fid_qutip = qutip.fidelity(qutip.Qobj(rho), qutip.Qobj(sigma)) ** 2
    fid_jax = float(qx.fidelity(rho_jax, sigma_jax))

    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), "JAX qx.fidelity does not match the square of qutip qx.fidelity."


@pytest.mark.parametrize("num_qudits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_fidelity_mixed_pure_density(seed, qudit_dim, num_qudits):
    """Test qx.fidelity between pure state and density matrix."""
    key = jax.random.key(seed)
    d = qudit_dim**num_qudits
    key1, key2, key3, key4 = jax.random.split(key, 4)
    psi = jax.random.normal(key1, (d,)) + 1j * jax.random.normal(key2, (d,))
    psi = psi / jnp.linalg.norm(psi)

    A = jax.random.normal(key3, (d, d)) + 1j * jax.random.normal(key4, (d, d))
    sigma = A @ A.conj().T
    sigma = sigma / jnp.trace(sigma)

    # Convert to JAX arrays
    psi_jax = qx.StateVector.from_matrix(jnp.array(psi), (qudit_dim,) * num_qudits)
    sigma_jax = qx.DensityMatrix.from_matrix(jnp.array(sigma), (qudit_dim,) * num_qudits)

    # Compute fidelities
    # Our qx.fidelity is the square of the standard definition used in qutip
    fid_qutip = qutip.fidelity(qutip.Qobj(psi[:, jnp.newaxis]), qutip.Qobj(sigma)) ** 2
    fid_jax = float(qx.fidelity(psi_jax, sigma_jax))

    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), "JAX qx.fidelity does not match the square of qutip qx.fidelity."


@pytest.mark.parametrize("num_qudits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_fidelity_self_is_one(seed, qudit_dim, num_qudits):
    """Test that qx.fidelity of a state with itself is 1."""
    key = jax.random.key(seed)
    d = qudit_dim**num_qudits
    key1, key2 = jax.random.split(key, 2)
    A = jax.random.normal(key1, (d, d)) + 1j * jax.random.normal(key2, (d, d))
    rho = A @ A.conj().T
    rho = rho / jnp.trace(rho)

    rho_jax = qx.DensityMatrix.from_matrix(jnp.array(rho), (qudit_dim,) * num_qudits)
    fid_jax = float(qx.fidelity(rho_jax, rho_jax))

    assert jnp.isclose(fid_jax, 1.0, atol=1e-6)


# ============================================================================
# Tests for qx.unitary_entanglement_fidelity
# ============================================================================


@pytest.mark.parametrize("num_qudits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_unitary_entanglement_fidelity(seed, qudit_dim, num_qudits):
    """Test unitary entanglement qx.fidelity."""
    key = jax.random.key(seed)
    _, subkey = jax.random.split(key)
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    U = qx.random_unitary(dims=dims, key=key)
    V = qx.random_unitary(dims=dims, key=subkey)

    fid_qutip = qx.average_fidelity_to_process_fidelity(
        qutip.average_gate_fidelity(
            U._to_qobj(),
            V._to_qobj(),
        ),
        dims=(qudit_dim,) * num_qudits,
    )
    fid_jax = float(qx.unitary_entanglement_fidelity(U, V))

    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), (
        "JAX unitary entanglement qx.fidelity does not match qutip result."
    )

    # Check the self-qx.fidelity
    fid_self_jax = float(qx.unitary_entanglement_fidelity(U, U))
    assert jnp.isclose(fid_self_jax, 1.0, atol=1e-6), "JAX unitary entanglement self-qx.fidelity is not 1."

    # Check the phase invariance
    phase = jnp.exp(1j * jax.random.uniform(jax.random.key(seed), (), minval=0, maxval=2 * jnp.pi))
    V_phase = phase * V
    fid_phase_jax = float(qx.unitary_entanglement_fidelity(U, V_phase))
    assert jnp.isclose(fid_phase_jax, fid_jax, atol=1e-6), (
        "JAX unitary entanglement qx.fidelity is not invariant under global phase."
    )


# ============================================================================
# Tests for qx.process_fidelity
# ============================================================================


@pytest.mark.parametrize("num_qudits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_process_fidelity_unitaries(seed, qudit_dim, num_qudits):
    """Test process qx.fidelity for unitary channels."""
    key = jax.random.key(seed)
    _, subkey = jax.random.split(key)
    d = qudit_dim**num_qudits
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    U = qx.random_unitary(dims=dims, key=key)
    V = qx.random_unitary(dims=dims, key=subkey)

    # Convert to qx.Choi matrices
    choi_U = qx.unitary_to_choi(U)
    choi_V = qx.unitary_to_choi(V)
    fid_qutip = qutip.process_fidelity(
        choi_U._to_qobj(),
        choi_V._to_qobj(),
    )
    fid_jax = float(qx.process_fidelity(choi_U, choi_V))
    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), "JAX process qx.fidelity does not match qutip result."  # type: ignore[arg-type]

    # Check the self-qx.fidelity
    fid_self_jax = float(qx.process_fidelity(choi_U, choi_U))
    assert jnp.isclose(fid_self_jax, 1.0, atol=1e-6), "JAX process self-qx.fidelity is not 1."

    # Check process qx.fidelity against identity channel
    Identity_mat = jnp.eye(d)
    choi_I = qx.unitary_to_choi(qx.Unitary.from_matrix(Identity_mat, dims))
    fid_identity_jax = float(qx.process_fidelity(choi_U, choi_I))
    fid_identity_qutip = qutip.process_fidelity(
        U._to_qobj(),
        choi_I._to_qobj(),
    )
    assert jnp.isclose(fid_identity_jax, fid_identity_qutip, atol=1e-6), (  # type: ignore[arg-type]
        "JAX process qx.fidelity against identity channel does not match qutip result."
    )

    # Check that process qx.fidelity defaults to comparison against the identity channel when second argument is None
    fid_none_jax = float(qx.process_fidelity(choi_U, None))
    assert jnp.isclose(fid_none_jax, fid_identity_jax, atol=1e-6), (
        "JAX process qx.fidelity with None does not match qx.fidelity against identity channel."
    )


@pytest.mark.parametrize("num_qudits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_process_fidelity_random_maps(seed, qudit_dim, num_qudits):
    """Test process qx.fidelity for random channels."""
    d = qudit_dim**num_qudits
    kraus_rank = d
    key, subkey = jax.random.split(jax.random.key(seed))
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)

    choi_U_jax = qx.random_choi(dims=dims, rank=kraus_rank, key=key)
    choi_V_jax = qx.random_choi(dims=dims, rank=kraus_rank, key=subkey)

    fid_qutip = qutip.process_fidelity(
        choi_U_jax._to_qobj(),
        choi_V_jax._to_qobj(),
    )
    fid_jax = float(qx.process_fidelity(choi_U_jax, choi_V_jax))

    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), "JAX process qx.fidelity does not match qutip result."  # type: ignore[arg-type]

    # Check the self-qx.fidelity
    fid_self_jax = float(qx.process_fidelity(choi_U_jax, choi_U_jax))
    assert jnp.isclose(fid_self_jax, 1.0, atol=1e-6), "JAX process self-qx.fidelity is not 1."

    # Check process qx.fidelity against identity channel
    Identity_mat = jnp.eye(d)
    choi_I = qx.unitary_to_choi(qx.Unitary.from_matrix(Identity_mat, dims))
    fid_identity_jax = float(qx.process_fidelity(choi_U_jax, choi_I))
    fid_identity_qutip = qutip.process_fidelity(
        choi_U_jax._to_qobj(),
        choi_I._to_qobj(),
    )
    assert jnp.isclose(fid_identity_jax, fid_identity_qutip, atol=1e-6), (  # type: ignore[arg-type]
        "JAX process qx.fidelity against identity channel does not match qutip result."
    )

    # Check that process qx.fidelity defaults to comparison against the identity channel when second argument is None
    fid_none_jax = float(qx.process_fidelity(choi_U_jax, None))
    assert jnp.isclose(fid_none_jax, fid_identity_jax, atol=1e-6), (
        "JAX process qx.fidelity with None does not match qx.fidelity against identity channel."
    )


# ============================================================================
# Tests for metric conversion functions (dims parameter)
# ============================================================================


@pytest.mark.parametrize("num_qudits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_metric_conversion_roundtrips(qudit_dim, num_qudits):
    """Test that metric conversions are consistent round-trips using the dims parameter."""
    dims = (qudit_dim,) * num_qudits
    p = 0.95

    # p -> F_avg -> p round-trip
    F_avg = qx.depolarizing_constant_to_average_fidelity(p, dims=dims)
    p_back = qx.average_fidelity_to_depolarizing_constant(F_avg, dims=dims)
    assert jnp.isclose(p, p_back, atol=1e-10)

    # p -> F_proc -> p round-trip
    F_proc = qx.depolarizing_constant_to_process_fidelity(p, dims=dims)
    p_back2 = qx.process_fidelity_to_depolarizing_constant(F_proc, dims=dims)
    assert jnp.isclose(p, p_back2, atol=1e-10)

    # F_avg -> F_proc -> F_avg round-trip
    F_proc2 = qx.average_fidelity_to_process_fidelity(F_avg, dims=dims)
    F_avg_back = qx.process_fidelity_to_average_fidelity(F_proc2, dims=dims)
    assert jnp.isclose(F_avg, F_avg_back, atol=1e-10)

    # F_proc from p and from F_avg should agree
    assert jnp.isclose(F_proc, F_proc2, atol=1e-10)


# ============================================================================
# Tests for qx.unitarity
# ============================================================================


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (3, 4)])
def test_unitarity_unitary(seed, qudit_dim, num_qudits, ensemble_size):
    """Test that a unitary channel has qx.unitarity 1."""
    key = jax.random.key(seed)
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    U = qx.random_unitary(dims=dims, key=key, size=ensemble_size)
    u = qx.unitarity(U)
    assert jnp.allclose(u, 1.0, atol=1e-7), f"Unitary channel should have unitarity 1, got {u}"


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_unitarity_random_channel(seed, qudit_dim, num_qudits, rank):
    """Test qx.unitarity of a random channel has the expected value."""
    ensemble_size = (100, 4)
    key = jax.random.key(seed)
    d = qudit_dim**num_qudits
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    choi = qx.random_choi(dims=dims, rank=rank, key=key, size=ensemble_size)
    u = qx.unitarity(choi)

    # Expected qx.unitarity for the BCSZ distribution with Kraus rank K and
    # Hilbert space dimension d is: E[u] = (d^2 - 1) * K / (d^2 * K^2 - 1)
    expected = (d**2 - 1) * rank / (d**2 * rank**2 - 1)
    assert jnp.isclose(jnp.mean(u), expected, atol=0.05), (
        f"Expected average unitarity of {expected:.4f}, got {jnp.mean(u):.4f}"
    )


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574])
def test_unitarity_accepts_all_representations(seed, qudit_dim, num_qudits):
    """Test that qx.unitarity accepts all superoperator representations."""
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    U = qx.random_unitary(dims=dims, key=jax.random.key(seed))

    u_unitary = float(qx.unitarity(U))
    u_choi = float(qx.unitarity(qx.to_choi(U)))
    u_superop = float(qx.unitarity(qx.to_superop(U)))
    u_pl = float(qx.unitarity(qx.to_pauli_liouville(U)))
    u_kraus = float(qx.unitarity(qx.to_kraus(U)))

    assert jnp.isclose(u_unitary, u_choi, atol=1e-6)
    assert jnp.isclose(u_unitary, u_superop, atol=1e-6)
    assert jnp.isclose(u_unitary, u_pl, atol=1e-6)
    assert jnp.isclose(u_unitary, u_kraus, atol=1e-6)


# ============================================================================
# Tests for qx.stochastic_infidelity
# ============================================================================


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (3, 4)])
def test_stochastic_infidelity_unitary_channel(seed, qudit_dim, num_qudits, ensemble_size):
    """Test that a unitary channel has stochastic infidelity 0."""
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    key = jax.random.key(seed)
    U = qx.random_unitary(dims=dims, key=key, size=ensemble_size)
    e_s = qx.stochastic_infidelity(U)
    assert jnp.allclose(e_s, 0.0, atol=1e-6), f"Unitary channel should have stochastic infidelity 0, got {e_s}"


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_stochastic_infidelity_random_channel(seed, qudit_dim, num_qudits, rank):
    """Test stochastic infidelity of random channels satisfies the unitarity bound.

    For any trace-preserving map the stochastic infidelity satisfies:

        e_S <= 1 - sqrt(u * (1 - 1/d^2) + 1/d^2)

    with equality if and only if the map is unital.  This follows from the
    Pauli-Liouville decomposition: ||S||_F^2 = 1 + ||P[1:,0]||^2 + (d^2-1)*u
    where the non-unital contribution ||P[1:,0]||^2 >= 0.
    """
    ensemble_size = (100, 4)
    key = jax.random.key(seed)
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    choi = qx.random_choi(dims=dims, rank=rank, key=key, size=ensemble_size)
    e_s = qx.stochastic_infidelity(choi)
    u = qx.unitarity(choi)

    # Stochastic infidelity must be non-negative
    assert jnp.all(e_s >= -1e-7), f"Stochastic infidelity should be non-negative, got min {jnp.min(e_s):.6f}"

    # Per-channel upper bound from unitarity (equality iff unital)
    e_s_bound = qx.unitarity_to_stochastic_infidelity(u, dims=(qudit_dim,) * num_qudits)
    assert jnp.all(e_s <= e_s_bound + 1e-7), (
        f"Stochastic infidelity should be <= unitarity bound, max violation {float(jnp.max(e_s - e_s_bound)):.6f}"
    )


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574])
def test_stochastic_infidelity_accepts_all_representations(seed, qudit_dim, num_qudits):
    """Test that qx.stochastic_infidelity accepts all superoperator representations."""
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    U = qx.random_unitary(dims=dims, key=jax.random.key(seed))

    e_unitary = float(qx.stochastic_infidelity(U))
    e_choi = float(qx.stochastic_infidelity(qx.to_choi(U)))
    e_superop = float(qx.stochastic_infidelity(qx.to_superop(U)))
    e_pl = float(qx.stochastic_infidelity(qx.to_pauli_liouville(U)))
    e_kraus = float(qx.stochastic_infidelity(qx.to_kraus(U)))

    assert jnp.isclose(e_unitary, e_choi, atol=1e-6)
    assert jnp.isclose(e_unitary, e_superop, atol=1e-6)
    assert jnp.isclose(e_unitary, e_pl, atol=1e-6)
    assert jnp.isclose(e_unitary, e_kraus, atol=1e-6)


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("seed", [4865, 3574])
def test_unitarity_stochastic_infidelity_relation(seed, qudit_dim, num_qudits):
    """Test that qx.unitarity_to_stochastic_infidelity agrees with qx.stochastic_infidelity for unital TP maps."""
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    U = qx.random_unitary(dims=dims, key=jax.random.key(seed))

    u = qx.unitarity(U)
    e_s_from_u = float(qx.unitarity_to_stochastic_infidelity(u, dims=(qudit_dim,) * num_qudits))
    e_s_direct = float(qx.stochastic_infidelity(U))
    assert jnp.isclose(e_s_from_u, e_s_direct, atol=1e-6)


# ======================================================================
# QuantumInstrument fidelity function tests
# ======================================================================


class TestInstrumentFidelities:
    """Test standalone instrument fidelity functions."""

    def test_classification_ideal(self):
        np.testing.assert_allclose(qx.classification_fidelity(qx.gates.MEASURE()), 1.0, atol=1e-10)

    @pytest.mark.parametrize("fid", [0.8, 0.9, 0.95])
    def test_classification_noisy(self, fid):
        confusion = jnp.array([[fid, 1 - fid], [1 - fid, fid]])
        qi = qx.instrument_from_confusion_and_transition(confusion, jnp.eye(2), dims=(2,))
        np.testing.assert_allclose(qx.classification_fidelity(qi), fid, atol=1e-10)

    def test_non_demolition_ideal(self):
        np.testing.assert_allclose(qx.non_demolition_fidelity(qx.gates.MEASURE()), 1.0, atol=1e-10)

    def test_instrument_fidelity_ideal(self):
        np.testing.assert_allclose(qx.instrument_fidelity(qx.gates.MEASURE()), 1.0, atol=1e-10)

    def test_bitflip_backaction_zero_qnd(self):
        confusion = jnp.eye(2)
        transition = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2,))
        np.testing.assert_allclose(qx.non_demolition_fidelity(qi), 0.0, atol=1e-10)

    @pytest.mark.parametrize("fid", [0.8, 0.9, 0.95])
    def test_confusion_only_qnd_is_one(self, fid):
        """Confusion-only instrument (identity transition) has QND fidelity = 1.0."""
        confusion = jnp.array([[fid, 1 - fid], [1 - fid, fid]])
        transition = jnp.eye(2)
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2,))
        np.testing.assert_allclose(qx.non_demolition_fidelity(qi), 1.0, atol=1e-10)

    @pytest.mark.parametrize("clf_fid", [0.80, 0.90, 0.95])
    @pytest.mark.parametrize("p_flip", [0.05, 0.10, 0.30])
    def test_instrument_fidelity_bounded_by_product(self, clf_fid, p_flip):
        """Instrument fidelity <= classification_fidelity * non_demolition_fidelity."""
        confusion = jnp.array([[clf_fid, 1 - clf_fid], [1 - clf_fid, clf_fid]])
        transition = jnp.array([[1 - p_flip, p_flip], [p_flip, 1 - p_flip]])
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2,))
        f_inst = float(qx.instrument_fidelity(qi))
        f_clf = float(qx.classification_fidelity(qi))
        f_qnd = float(qx.non_demolition_fidelity(qi))
        assert f_inst <= f_clf * f_qnd + 1e-10

    @pytest.mark.parametrize("d", [2, 3])
    def test_multiqudit_ideal_fidelities(self, d):
        """Multi-qudit ideal measurement: all fidelities are 1.0."""
        qi = qx.gates.MEASURE(d) | qx.gates.MEASURE(d)
        np.testing.assert_allclose(qx.classification_fidelity(qi), 1.0, atol=1e-10)
        np.testing.assert_allclose(qx.non_demolition_fidelity(qi), 1.0, atol=1e-10)
        np.testing.assert_allclose(qx.instrument_fidelity(qi), 1.0, atol=1e-10)

    @pytest.mark.parametrize("fid", [0.80, 0.95])
    def test_multiqudit_noisy_fidelities(self, fid):
        """Two-qubit noisy instrument: fidelity bounded by clf * qnd."""
        d = 4  # total dimension for 2 qubits
        off = (1.0 - fid) / (d - 1)
        confusion = fid * jnp.eye(d) + off * (jnp.ones((d, d)) - jnp.eye(d))
        p_flip = 0.1
        transition_single = jnp.array([[1 - p_flip, p_flip], [p_flip, 1 - p_flip]])
        transition = jnp.kron(transition_single, transition_single)
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2, 2))
        f_inst = float(qx.instrument_fidelity(qi))
        f_clf = float(qx.classification_fidelity(qi))
        f_qnd = float(qx.non_demolition_fidelity(qi))
        np.testing.assert_allclose(f_clf, fid, atol=1e-10)
        assert f_inst <= f_clf * f_qnd + 1e-10
