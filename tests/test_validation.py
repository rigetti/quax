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


# ---------- Unitary Tests ----------


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_unitary_ensemble_is_unitary(num_qudits, qudit_dim, ensemble_size, seed):
    """Ensembles of random unitaries should all pass the is_unitary check."""
    key = jax.random.key(seed)
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    U = qx.random_unitary(dims, key, size=ensemble_size)

    assert jnp.all(qx.validate(U))

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


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_state_vector_ensemble_validates(num_qudits, qudit_dim, ensemble_size, seed):
    """Ensembles of random state vectors should all pass validate."""
    key = jax.random.key(seed)
    dims = (qudit_dim,) * num_qudits
    state = qx.random_state_vector(dims, key, size=ensemble_size)

    assert jnp.all(qx.validate(state))

    unnormalized_state = state * 2.0  # Scale to make unnormalized
    assert not jnp.any(qx.validate(unnormalized_state))


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_density_matrix_ensemble_validates(num_qudits, qudit_dim, ensemble_size, seed):
    """Ensembles of random density matrices should all pass validate."""
    key = jax.random.key(seed)
    dims = (qudit_dim,) * num_qudits
    rank = qudit_dim**num_qudits
    rho = qx.random_density_matrix(rank, dims, key, size=ensemble_size)

    assert jnp.all(qx.validate(rho))

    # scale to make non-unit-trace
    invalid_rho = rho * 0.5
    assert not jnp.any(qx.validate(invalid_rho))


# ---------- Superoperator Tests (using package-level functions) ----------


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_choi_ensemble_is_cptp(num_qudits, qudit_dim, ensemble_size, seed):
    """Ensembles of random Choi matrices should all be CPTP."""
    key = jax.random.key(seed)
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    rank = qudit_dim**num_qudits
    choi = qx.random_choi(dims, rank, key, size=ensemble_size)

    assert jnp.all(qx.is_completely_positive(choi))
    assert jnp.all(qx.is_hermicity_preserving(choi))
    assert jnp.all(qx.is_trace_preserving(choi))
    assert jnp.all(qx.is_cptp(choi))
    assert jnp.all(qx.validate(choi))


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_superop_is_cptp(num_qudits, qudit_dim, ensemble_size, seed):
    """SuperOp converted from random Choi should be CPTP."""
    key = jax.random.key(seed)
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    rank = qudit_dim**num_qudits
    superop = qx.choi_to_superop(qx.random_choi(dims, rank, key, size=ensemble_size))

    assert jnp.all(qx.is_completely_positive(superop))
    assert jnp.all(qx.is_hermicity_preserving(superop))
    assert jnp.all(qx.is_trace_preserving(superop))
    assert jnp.all(qx.is_cptp(superop))
    assert jnp.all(qx.validate(superop))


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_kraus_is_cptp(num_qudits, qudit_dim, ensemble_size, seed):
    """KrausMap converted from random Choi should be CPTP."""
    key = jax.random.key(seed)
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    rank = qudit_dim**num_qudits
    kraus = qx.choi_to_kraus(qx.random_choi(dims, rank, key, size=ensemble_size))

    assert jnp.all(qx.is_completely_positive(kraus))
    assert jnp.all(qx.is_hermicity_preserving(kraus))
    assert jnp.all(qx.is_trace_preserving(kraus))
    assert jnp.all(qx.is_cptp(kraus))
    assert jnp.all(qx.validate(kraus))


@pytest.mark.parametrize("num_qudits", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_random_pauli_liouville_is_cptp(num_qudits, qudit_dim, ensemble_size, seed):
    """PauliLiouville converted from random Choi should be CPTP."""
    key = jax.random.key(seed)
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    rank = qudit_dim**num_qudits
    pauli_liouville = qx.choi_to_pauli_liouville(qx.random_choi(dims, rank, key, size=ensemble_size))

    assert jnp.all(qx.is_completely_positive(pauli_liouville))
    assert jnp.all(qx.is_hermicity_preserving(pauli_liouville))
    assert jnp.all(qx.is_trace_preserving(pauli_liouville))
    assert jnp.all(qx.is_cptp(pauli_liouville))
    assert jnp.all(qx.validate(pauli_liouville))


def test_unitary_channel_validates():
    """Unitary operators should pass validate."""
    U = qx.gates.H

    assert jnp.all(qx.validate(U))
    # Also test with package-level functions
    assert jnp.all(qx.is_cptp(U))


# ---------- Negative Tests: Random Matrices Should Fail ----------


@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_random_matrix_superop_not_cp(qudit_dim):
    """A random matrix treated as SuperOp should not be CP."""
    key = jax.random.key(42)
    d2 = qudit_dim**2
    random_matrix = jax.random.normal(key, (d2, d2), dtype=complex)
    random_matrix = random_matrix + 1j * jax.random.normal(jax.random.key(43), (d2, d2))

    superop = qx.SuperOp.from_matrix(random_matrix, dims=((qudit_dim,), (qudit_dim,)))

    # Random matrix is very unlikely to be CP
    assert not qx.is_completely_positive(superop)
    assert not qx.validate(superop)


@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_random_matrix_superop_not_tp(qudit_dim):
    """A random matrix treated as SuperOp should not be TP."""
    key = jax.random.key(42)
    d2 = qudit_dim**2
    random_matrix = jax.random.normal(key, (d2, d2), dtype=complex)
    random_matrix = random_matrix + 1j * jax.random.normal(jax.random.key(43), (d2, d2))

    superop = qx.SuperOp.from_matrix(random_matrix, dims=((qudit_dim,), (qudit_dim,)))

    assert not qx.is_trace_preserving(superop)


@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_random_matrix_choi_not_cptp(qudit_dim):
    """A random matrix treated as Choi should not be CPTP."""
    key = jax.random.key(42)
    d2 = qudit_dim**2
    random_matrix = jax.random.normal(key, (d2, d2), dtype=complex)
    random_matrix = random_matrix + 1j * jax.random.normal(jax.random.key(43), (d2, d2))

    choi = qx.Choi.from_matrix(random_matrix, dims=((qudit_dim,), (qudit_dim,)))

    assert not qx.is_cptp(choi)
    assert not qx.validate(choi)


@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_random_matrix_not_hermicity_preserving(qudit_dim):
    """A random matrix treated as SuperOp should not be HP."""
    key = jax.random.key(42)
    d2 = qudit_dim**2
    random_matrix = jax.random.normal(key, (d2, d2), dtype=complex)
    random_matrix = random_matrix + 1j * jax.random.normal(jax.random.key(43), (d2, d2))

    superop = qx.SuperOp.from_matrix(random_matrix, dims=((qudit_dim,), (qudit_dim,)))

    assert not qx.is_hermicity_preserving(superop)


# ---------- Edge Cases ----------


def test_identity_channel_validates():
    """Identity channel (identity unitary) should validate."""
    Id = qx.gates.I

    assert qx.is_unitary(Id)
    assert qx.is_hermitian(Id)
    assert qx.validate(Id)
    assert qx.is_cptp(Id)


def test_cnot_validates():
    """CNOT gate should validate."""
    CNOT = qx.gates.CNOT

    assert qx.is_unitary(CNOT)
    assert qx.validate(CNOT)
    assert qx.is_cptp(CNOT)


@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_cp_but_not_tp(qudit_dim):
    """Construct a CP but not TP map (incomplete Kraus decomposition)."""
    gamma = 0.5
    # Diagonal operator: identity on all levels except last, which is scaled by sqrt(1-gamma)
    diag = jnp.ones(qudit_dim, dtype=complex).at[-1].set(jnp.sqrt(1 - gamma))
    K0 = jnp.diag(diag)

    # Create a KrausMap with only one Kraus operator (incomplete)
    kraus_data = K0.reshape(1, qudit_dim, qudit_dim)
    kraus = qx.KrausMap.from_matrix(kraus_data, dims=((qudit_dim,), (qudit_dim,)))

    # This should be CP (Kraus operators always give CP maps) but not TP
    assert qx.is_completely_positive(kraus)
    assert not qx.is_trace_preserving(kraus)
    assert not qx.is_cptp(kraus)
    assert not qx.validate(kraus)  # validate checks CPTP


# ---------- validate() dispatch: Operator, Observable, Involution ----------


@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_validate_operator(qudit_dim, ensemble_size, seed):
    """validate(Operator) always returns True (any linear operator is well-formed)."""
    key = jax.random.key(seed)
    dims = ((qudit_dim,), (qudit_dim,))
    op = qx.random_operator(dims, key, size=ensemble_size)

    result = qx.validate(op)
    assert jnp.all(result)
    assert result.shape == ensemble_size


@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (2, 4)])
@pytest.mark.parametrize("seed", [42, 123])
def test_validate_observable(qudit_dim, ensemble_size, seed):
    """validate(Observable) checks Hermiticity; random Hermitian matrices should pass."""
    key = jax.random.key(seed)
    dims = ((qudit_dim,), (qudit_dim,))
    obs = qx.random_observable(dims, key, size=ensemble_size)

    assert jnp.all(qx.validate(obs))


def test_validate_observable_known():
    """Pauli observables should validate; a non-Hermitian operator cast as Observable should not."""
    # Paulis are Observable (actually Involution, but we can test via ensembles.PAULIS)
    assert jnp.all(qx.is_hermitian(qx.gates.X))
    assert jnp.all(qx.is_hermitian(qx.gates.Z))


def test_validate_involution():
    """validate(Involution) checks both Hermitian and unitary; Pauli gates should pass."""
    assert jnp.all(qx.validate(qx.gates.X))
    assert jnp.all(qx.validate(qx.gates.Y))
    assert jnp.all(qx.validate(qx.gates.Z))
    assert jnp.all(qx.validate(qx.gates.H))
    assert jnp.all(qx.validate(qx.gates.I))


# ---------- is_one_design / is_two_design Tests ----------


def test_is_one_design():
    """Test the is_one_design function."""
    seed = 5734
    cliffords = qx.ensembles.CLIFFORD_ENSEMBLE
    assert qx.is_one_design(cliffords)

    key = jax.random.key(seed)
    dims = ((2,), (2,))
    # 5 random unitaries are very unlikely to form a 1-design
    ensemble = qx.random_unitary(dims, key, size=(5,))
    assert not qx.is_one_design(ensemble)

    # is_one_design should raise for non-single-qubit ensembles
    key = jax.random.key(42)
    dims = ((2, 2), (2, 2))
    ensemble = qx.random_unitary(dims, key, size=(5,))
    with pytest.raises(ValueError, match="Only supports 1-qubit"):
        qx.is_one_design(ensemble)


def test_is_two_design():
    """Test the is_two_design function."""
    seed = 98573
    cliffords = qx.ensembles.CLIFFORD_ENSEMBLE
    assert qx.is_two_design(cliffords)

    key = jax.random.key(seed)
    dims = ((2,), (2,))
    ensemble = qx.random_unitary(dims, key, size=(5,))
    assert not qx.is_two_design(ensemble)

    # is_two_design should raise for non-single-qubit ensembles
    key = jax.random.key(42)
    dims = ((2, 2), (2, 2))
    ensemble = qx.random_unitary(dims, key, size=(5,))
    with pytest.raises(ValueError, match="Only supports 1-qubit"):
        qx.is_two_design(ensemble)
