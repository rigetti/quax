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

"""Tests for validation methods on quantum objects (is_herm, is_unitary, validate, etc.)."""

import jax
import jax.numpy as jnp
import pytest

import quax as qx
from quax import (
    Choi,
    KrausMap,
    SuperOp,
)


# ---------- Unitary Tests ----------


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_unitary_ensemble_is_unitary(num_qubits, ensemble_size, seed):
    """Ensembles of random unitaries should all pass the is_unitary check."""
    key = jax.random.key(seed)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    U = qx.random_unitary(dims, key, size=ensemble_size)

    assert jnp.all(qx.is_unitary(U))

    U_scaled = U * 0.5  # Scale to make non-unitary
    assert not jnp.any(qx.is_unitary(U_scaled))


def test_is_hermitian_gates():
    """Test the is_hermitian function on known Hermitian operators."""
    assert qx.is_hermitian(qx.gates.X)
    assert qx.is_hermitian(qx.gates.Y)
    assert qx.is_hermitian(qx.gates.Z)
    assert qx.is_hermitian(qx.gates.H)

    assert not qx.is_hermitian(qx.gates.S)
    assert not qx.is_hermitian(qx.gates.T)


# ---------- State Validation Tests ----------


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_state_vector_ensemble_validates(num_qubits, ensemble_size, seed):
    """Ensembles of random state vectors should all pass validate."""
    key = jax.random.key(seed)
    dims = (2,) * num_qubits
    state = qx.random_state_vector(dims, key, size=ensemble_size)

    assert jnp.all(state.validate())

    unnormalized_state = state * 2.0  # Scale to make unnormalized
    assert not jnp.any(unnormalized_state.validate())


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_density_matrix_ensemble_validates(num_qubits, ensemble_size, seed):
    """Ensembles of random density matrices should all pass validate."""
    key = jax.random.key(seed)
    dims = (2,) * num_qubits
    rank = 2**num_qubits
    rho = qx.random_density_matrix(rank, dims, key, size=ensemble_size)

    assert jnp.all(rho.validate())

    # scale to make non-unit-trace
    invalid_rho = rho * 0.5
    assert not jnp.any(invalid_rho.validate())


# ---------- Superoperator Tests (using package-level functions) ----------


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_choi_ensemble_is_cptp(num_qubits, ensemble_size, seed):
    """Ensembles of random Choi matrices should all be CPTP."""
    key = jax.random.key(seed)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    rank = 2**num_qubits
    choi = qx.random_choi_BCSZ(dims, rank, key, size=ensemble_size)

    assert jnp.all(qx.is_completely_positive(choi))
    assert jnp.all(qx.is_hermicity_preserving(choi))
    assert jnp.all(qx.is_trace_preserving(choi))
    assert jnp.all(qx.is_cptp(choi))
    assert jnp.all(choi.validate())


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_superop_is_cptp(num_qubits, ensemble_size, seed):
    """SuperOp converted from random Choi should be CPTP."""
    key = jax.random.key(seed)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    rank = 2**num_qubits
    superop = qx.choi_to_superop(qx.random_choi_BCSZ(dims, rank, key, size=ensemble_size))

    assert jnp.all(qx.is_completely_positive(superop))
    assert jnp.all(qx.is_hermicity_preserving(superop))
    assert jnp.all(qx.is_trace_preserving(superop))
    assert jnp.all(qx.is_cptp(superop))
    assert jnp.all(superop.validate())


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_kraus_is_cptp(num_qubits, ensemble_size, seed):
    """KrausMap converted from random Choi should be CPTP."""
    key = jax.random.key(seed)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    rank = 2**num_qubits
    kraus = qx.choi_to_kraus(qx.random_choi_BCSZ(dims, rank, key, size=ensemble_size))

    assert jnp.all(qx.is_completely_positive(kraus))
    assert jnp.all(qx.is_hermicity_preserving(kraus))
    assert jnp.all(qx.is_trace_preserving(kraus))
    assert jnp.all(qx.is_cptp(kraus))
    assert jnp.all(kraus.validate())


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_pauli_liouville_is_cptp(num_qubits, ensemble_size, seed):
    """PauliLiouville converted from random Choi should be CPTP."""
    key = jax.random.key(seed)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    rank = 2**num_qubits
    pauli_liouville = qx.choi_to_pauli_liouville(qx.random_choi_BCSZ(dims, rank, key, size=ensemble_size))

    assert jnp.all(qx.is_completely_positive(pauli_liouville))
    assert jnp.all(qx.is_hermicity_preserving(pauli_liouville))
    assert jnp.all(qx.is_trace_preserving(pauli_liouville))
    assert jnp.all(qx.is_cptp(pauli_liouville))
    assert jnp.all(pauli_liouville.validate())


def test_unitary_channel_validates():
    """Unitary operators should pass validate."""
    U = qx.gates.H

    assert jnp.all(U.validate())
    # Also test with package-level functions
    assert jnp.all(qx.is_cptp(U))


# ---------- Negative Tests: Random Matrices Should Fail ----------


def test_random_matrix_superop_not_cp():
    """A random matrix treated as SuperOp should not be CP."""
    key = jax.random.key(42)
    d2 = 4  # 1 qubit: d^2 = 4
    random_matrix = jax.random.normal(key, (d2, d2), dtype=complex)
    random_matrix = random_matrix + 1j * jax.random.normal(jax.random.key(43), (d2, d2))

    superop = SuperOp.from_matrix(random_matrix, dims=((2,), (2,)))

    # Random matrix is very unlikely to be CP
    assert not qx.is_completely_positive(superop)
    assert not superop.validate()


def test_random_matrix_superop_not_tp():
    """A random matrix treated as SuperOp should not be TP."""
    key = jax.random.key(42)
    d2 = 4
    random_matrix = jax.random.normal(key, (d2, d2), dtype=complex)
    random_matrix = random_matrix + 1j * jax.random.normal(jax.random.key(43), (d2, d2))

    superop = SuperOp.from_matrix(random_matrix, dims=((2,), (2,)))

    assert not qx.is_trace_preserving(superop)


def test_random_matrix_choi_not_cptp():
    """A random matrix treated as Choi should not be CPTP."""
    key = jax.random.key(42)
    d2 = 4
    random_matrix = jax.random.normal(key, (d2, d2), dtype=complex)
    random_matrix = random_matrix + 1j * jax.random.normal(jax.random.key(43), (d2, d2))

    choi = Choi.from_matrix(random_matrix, dims=((2,), (2,)))

    assert not qx.is_cptp(choi)
    assert not choi.validate()


def test_random_matrix_not_hermicity_preserving():
    """A random matrix treated as SuperOp should not be HP."""
    key = jax.random.key(42)
    d2 = 4
    random_matrix = jax.random.normal(key, (d2, d2), dtype=complex)
    random_matrix = random_matrix + 1j * jax.random.normal(jax.random.key(43), (d2, d2))

    superop = SuperOp.from_matrix(random_matrix, dims=((2,), (2,)))

    assert not qx.is_hermicity_preserving(superop)


# ---------- Edge Cases ----------


def test_identity_channel_validates():
    """Identity channel (identity unitary) should validate."""
    Id = qx.gates.I

    assert qx.is_unitary(Id)
    assert qx.is_hermitian(Id)
    assert Id.validate()
    assert qx.is_cptp(Id)


def test_cnot_validates():
    """CNOT gate should validate."""
    CNOT = qx.gates.CNOT

    assert qx.is_unitary(CNOT)
    assert CNOT.validate()
    assert qx.is_cptp(CNOT)


def test_cp_but_not_tp():
    """Construct a CP but not TP map (amplitude damping with incomplete normalization)."""
    # A simple non-TP map: just the first Kraus operator of amplitude damping
    # K0 = [[1, 0], [0, sqrt(1-gamma)]]
    gamma = 0.5
    K0 = jnp.array([[1, 0], [0, jnp.sqrt(1 - gamma)]], dtype=complex)

    # Create a KrausMap with only one Kraus operator (incomplete)
    kraus_data = K0.reshape(1, 2, 2)
    kraus = KrausMap(data=kraus_data, num_ensemble_dims=0)

    # This should be CP (Kraus operators always give CP maps) but not TP
    assert qx.is_completely_positive(kraus)
    assert not qx.is_trace_preserving(kraus)
    assert not qx.is_cptp(kraus)
    assert not kraus.validate()  # validate checks CPTP
