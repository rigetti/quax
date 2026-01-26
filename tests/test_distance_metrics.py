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
import pytest
import qutip

from quax import (
    DensityMatrix,
    StateVector,
    Unitary,
    average_fidelity_to_process_fidelity,
    fidelity,
    process_fidelity,
    random_choi_BCSZ,
    random_unitary,
    unitary_entanglement_fidelity,
    unitary_to_choi,
)

# ============================================================================
# Tests for fidelity
# ============================================================================


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_fidelity_pure_states(seed, num_qubits):
    """Test fidelity between pure states against reference implementation."""
    key = jax.random.key(seed)
    d = 2**num_qubits
    key1, key2, key3, key4 = jax.random.split(key, 4)
    psi = jax.random.normal(key1, (d,)) + 1j * jax.random.normal(key2, (d,))
    psi = psi / jnp.linalg.norm(psi)
    phi = jax.random.normal(key3, (d,)) + 1j * jax.random.normal(key4, (d,))
    phi = phi / jnp.linalg.norm(phi)

    # Convert to JAX arrays
    psi_jax = StateVector.from_matrix(jnp.array(psi), (2,) * num_qubits, 0)
    phi_jax = StateVector.from_matrix(jnp.array(phi), (2,) * num_qubits, 0)

    # Our fidelity is the square of the standard definition used in qutip
    fid_qutip = qutip.fidelity(qutip.Qobj(psi[:, jnp.newaxis]), qutip.Qobj(phi[:, jnp.newaxis])) ** 2
    fid_jax = float(fidelity(psi_jax, phi_jax))

    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), "JAX fidelity does not match the square of qutip fidelity."


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_fidelity_density_matrices(seed, num_qubits):
    """Test fidelity between density matrices against reference implementation."""
    key = jax.random.key(seed)
    d = 2**num_qubits
    # Create random density matrices
    key1, key2, key3, key4 = jax.random.split(key, 4)
    A = jax.random.normal(key1, (d, d)) + 1j * jax.random.normal(key2, (d, d))
    rho = A @ A.conj().T
    rho = rho / jnp.trace(rho)

    B = jax.random.normal(key3, (d, d)) + 1j * jax.random.normal(key4, (d, d))
    sigma = B @ B.conj().T
    sigma = sigma / jnp.trace(sigma)

    # Convert to JAX arrays
    rho_jax = DensityMatrix.from_matrix(jnp.array(rho), (2,) * num_qubits, 0)
    sigma_jax = DensityMatrix.from_matrix(jnp.array(sigma), (2,) * num_qubits, 0)

    # Compute fidelities
    # Our fidelity is the square of the standard definition used in qutip
    fid_qutip = qutip.fidelity(qutip.Qobj(rho), qutip.Qobj(sigma)) ** 2
    fid_jax = float(fidelity(rho_jax, sigma_jax))

    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), "JAX fidelity does not match the square of qutip fidelity."


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_fidelity_mixed_pure_density(seed, num_qubits):
    """Test fidelity between pure state and density matrix."""
    key = jax.random.key(seed)
    d = 2**num_qubits
    key1, key2, key3, key4 = jax.random.split(key, 4)
    psi = jax.random.normal(key1, (d,)) + 1j * jax.random.normal(key2, (d,))
    psi = psi / jnp.linalg.norm(psi)

    A = jax.random.normal(key3, (d, d)) + 1j * jax.random.normal(key4, (d, d))
    sigma = A @ A.conj().T
    sigma = sigma / jnp.trace(sigma)

    # Convert to JAX arrays
    psi_jax = StateVector.from_matrix(jnp.array(psi), (2,) * num_qubits, 0)
    sigma_jax = DensityMatrix.from_matrix(jnp.array(sigma), (2,) * num_qubits, 0)

    # Compute fidelities
    # Our fidelity is the square of the standard definition used in qutip
    fid_qutip = qutip.fidelity(qutip.Qobj(psi[:, jnp.newaxis]), qutip.Qobj(sigma)) ** 2
    fid_jax = float(fidelity(psi_jax, sigma_jax))

    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), "JAX fidelity does not match the square of qutip fidelity."


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_fidelity_self_is_one(seed, num_qubits):
    """Test that fidelity of a state with itself is 1."""
    key = jax.random.key(seed)
    d = 2**num_qubits
    key1, key2 = jax.random.split(key, 2)
    A = jax.random.normal(key1, (d, d)) + 1j * jax.random.normal(key2, (d, d))
    rho = A @ A.conj().T
    rho = rho / jnp.trace(rho)

    rho_jax = DensityMatrix.from_matrix(jnp.array(rho), (2,) * num_qubits, 0)
    fid_jax = float(fidelity(rho_jax, rho_jax))

    assert jnp.isclose(fid_jax, 1.0, atol=1e-6)


# ============================================================================
# Tests for unitary_entanglement_fidelity
# ============================================================================


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_unitary_entanglement_fidelity(seed, num_qubits):
    """Test unitary entanglement fidelity."""
    key = jax.random.key(seed)
    _, subkey = jax.random.split(key)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    U = random_unitary(dims=dims, key=key)
    V = random_unitary(dims=dims, key=subkey)

    fid_qutip = average_fidelity_to_process_fidelity(
        qutip.average_gate_fidelity(
            U._to_qobj(),
            V._to_qobj(),
        ),
        num_sys=num_qubits,
    )
    fid_jax = float(unitary_entanglement_fidelity(U, V))

    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), "JAX unitary entanglement fidelity does not match qutip result."

    # Check the self-fidelity
    fid_self_jax = float(unitary_entanglement_fidelity(U, U))
    assert jnp.isclose(fid_self_jax, 1.0, atol=1e-6), "JAX unitary entanglement self-fidelity is not 1."

    # Check the phase invariance
    phase = jnp.exp(1j * jax.random.uniform(jax.random.key(seed), (), minval=0, maxval=2 * jnp.pi))
    V_phase = phase * V
    fid_phase_jax = float(unitary_entanglement_fidelity(U, V_phase))
    assert jnp.isclose(fid_phase_jax, fid_jax, atol=1e-6), (
        "JAX unitary entanglement fidelity is not invariant under global phase."
    )


# ============================================================================
# Tests for process_fidelity
# ============================================================================


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_process_fidelity_unitaries(seed, num_qubits):
    """Test process fidelity for unitary channels."""
    key = jax.random.key(seed)
    _, subkey = jax.random.split(key)
    d = 2**num_qubits
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    U = random_unitary(dims=dims, key=key)
    V = random_unitary(dims=dims, key=subkey)

    # Convert to Choi matrices
    choi_U = unitary_to_choi(U)
    choi_V = unitary_to_choi(V)
    fid_qutip = qutip.process_fidelity(
        choi_U._to_qobj(),
        choi_V._to_qobj(),
    )
    fid_jax = float(process_fidelity(choi_U, choi_V))
    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), "JAX process fidelity does not match qutip result."  # type: ignore[arg-type]

    # Check the self-fidelity
    fid_self_jax = float(process_fidelity(choi_U, choi_U))
    assert jnp.isclose(fid_self_jax, 1.0, atol=1e-6), "JAX process self-fidelity is not 1."

    # Check process fidelity against identity channel
    Identity_mat = jnp.eye(d)
    choi_I = unitary_to_choi(Unitary.from_matrix(Identity_mat, dims, 0))
    fid_identity_jax = float(process_fidelity(choi_U, choi_I))
    fid_identity_qutip = qutip.process_fidelity(
        U._to_qobj(),
        choi_I._to_qobj(),
    )
    assert jnp.isclose(fid_identity_jax, fid_identity_qutip, atol=1e-6), (  # type: ignore[arg-type]
        "JAX process fidelity against identity channel does not match qutip result."
    )

    # Check that process fidelity defaults to comparison against the identity channel when second argument is None
    fid_none_jax = float(process_fidelity(choi_U, None))
    assert jnp.isclose(fid_none_jax, fid_identity_jax, atol=1e-6), (
        "JAX process fidelity with None does not match fidelity against identity channel."
    )


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("seed", [4865, 3574, 323])
def test_process_fidelity_random_maps(seed, num_qubits):
    """Test process fidelity for random channels."""
    d = 2**num_qubits
    kraus_rank = d
    key, subkey = jax.random.split(jax.random.key(seed))
    dims = ((2,) * num_qubits, (2,) * num_qubits)

    choi_U_jax = random_choi_BCSZ(dims=dims, rank=kraus_rank, key=key)
    choi_V_jax = random_choi_BCSZ(dims=dims, rank=kraus_rank, key=subkey)

    fid_qutip = qutip.process_fidelity(
        choi_U_jax._to_qobj(),
        choi_V_jax._to_qobj(),
    )
    fid_jax = float(process_fidelity(choi_U_jax, choi_V_jax))

    assert jnp.isclose(fid_jax, fid_qutip, atol=1e-6), "JAX process fidelity does not match qutip result."  # type: ignore[arg-type]

    # Check the self-fidelity
    fid_self_jax = float(process_fidelity(choi_U_jax, choi_U_jax))
    assert jnp.isclose(fid_self_jax, 1.0, atol=1e-6), "JAX process self-fidelity is not 1."

    # Check process fidelity against identity channel
    Identity_mat = jnp.eye(d)
    choi_I = unitary_to_choi(Unitary.from_matrix(Identity_mat, dims, 0))
    fid_identity_jax = float(process_fidelity(choi_U_jax, choi_I))
    fid_identity_qutip = qutip.process_fidelity(
        choi_U_jax._to_qobj(),
        choi_I._to_qobj(),
    )
    assert jnp.isclose(fid_identity_jax, fid_identity_qutip, atol=1e-6), (  # type: ignore[arg-type]
        "JAX process fidelity against identity channel does not match qutip result."
    )

    # Check that process fidelity defaults to comparison against the identity channel when second argument is None
    fid_none_jax = float(process_fidelity(choi_U_jax, None))
    assert jnp.isclose(fid_none_jax, fid_identity_jax, atol=1e-6), (
        "JAX process fidelity with None does not match fidelity against identity channel."
    )
