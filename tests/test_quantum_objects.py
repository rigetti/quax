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

# In this file, we do not robustly test the correctness of the values
# Different test files cover implementations of the various types and transformations.
# Here, we only test that outputs are the expected shape and types

import jax
import jax.numpy as jnp
import pytest

from quax import (
    Choi,
    DensityMatrix,
    Kraus,
    KrausMap,
    PauliLiouville,
    StateVector,
    SuperOp,
    Unitary,
    choi_to_kraus,
    choi_to_pauli_liouville,
    choi_to_superop,
    random_choi_BCSZ,
    random_density_matrix,
    random_state_vector,
    random_unitary,
)


def _generate_random_object(object_type, key, size, dims, rank):
    match object_type:
        case t if t is Choi:
            return random_choi_BCSZ(dims, rank, key, size)
        case t if t is SuperOp:
            return choi_to_superop(random_choi_BCSZ(dims, rank, key, size))
        case t if t is KrausMap:
            return choi_to_kraus(random_choi_BCSZ(dims, rank, key, size))
        case t if t is PauliLiouville:
            return choi_to_pauli_liouville(random_choi_BCSZ(dims, rank, key, size))
        case t if t is StateVector:
            return random_state_vector(dims[0], key, size)
        case t if t is DensityMatrix:
            return random_density_matrix(rank, dims[0], key, size)
        case t if t is Unitary:
            return random_unitary(dims, key, size)
        case _:
            raise ValueError(f"{object_type} is not a valid quantum object type")


## Unary operations


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        (),
        (3,),
        (2, 4),
    ],
)
@pytest.mark.parametrize(
    "quantum_object", [Choi, KrausMap, SuperOp, PauliLiouville, Unitary, StateVector, DensityMatrix]
)
def test_negation(num_qubits, ensemble_size, quantum_object):
    """Test negation of the object (__neg__)."""
    key = jax.random.key(1234)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    rank = 2**num_qubits

    q_object = _generate_random_object(quantum_object, key, ensemble_size, dims, rank)
    negated_q_object = -q_object

    assert jnp.allclose(negated_q_object.data, -q_object.data)
    assert type(negated_q_object) is type(q_object)
    assert negated_q_object.ensemble_size == q_object.ensemble_size


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        (),
        (3,),
        (2, 4),
    ],
)
@pytest.mark.parametrize("scalar", [1.0, jnp.exp(1j * jnp.pi / 4), 0.5, 0.25 + 0.5j])
@pytest.mark.parametrize(
    "quantum_object", [Choi, KrausMap, SuperOp, PauliLiouville, Unitary, StateVector, DensityMatrix]
)
def test_scalar_multiplication(num_qubits, ensemble_size, scalar, quantum_object):
    """Test multiplication of the object (__mul__ and __rmul__)."""
    key = jax.random.key(1234)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    rank = 2**num_qubits
    q_object = _generate_random_object(quantum_object, key, ensemble_size, dims, rank)

    # Check forward mul
    scaled_q_object = scalar * q_object

    assert jnp.allclose(scaled_q_object.data, scalar * q_object.data)

    if isinstance(q_object, Unitary) and jnp.abs(scalar) != 1.0:
        assert type(scaled_q_object) is Kraus
    else:
        assert type(scaled_q_object) is type(q_object)
    assert scaled_q_object.ensemble_size == q_object.ensemble_size

    # Check reverse mul
    scaled_q_object = q_object * scalar

    assert jnp.allclose(scaled_q_object.data, scalar * q_object.data)

    if isinstance(q_object, Unitary) and jnp.abs(scalar) != 1.0:
        assert type(scaled_q_object) is Kraus
    else:
        assert type(scaled_q_object) is type(q_object)
    assert scaled_q_object.ensemble_size == q_object.ensemble_size


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        (),
        (3,),
        (2, 4),
    ],
)
@pytest.mark.parametrize("power", [1, 2, 3])
@pytest.mark.parametrize("quantum_object", [Choi, KrausMap, SuperOp, PauliLiouville, Unitary, DensityMatrix])
def test_integer_power(num_qubits, ensemble_size, power, quantum_object):
    """Test raising a quantum object to an integer power."""
    key = jax.random.key(1234)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    rank = 2**num_qubits
    q_object = _generate_random_object(quantum_object, key, ensemble_size, dims, rank)

    raised_object = q_object**power

    assert type(raised_object) is type(q_object)
    assert raised_object.ensemble_size == q_object.ensemble_size


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        (),
        (3,),
        (2, 4),
    ],
)
@pytest.mark.parametrize("power", [1.5, 2.2])
@pytest.mark.parametrize("quantum_object", [Choi, KrausMap, SuperOp, PauliLiouville, Unitary, DensityMatrix])
def test_fractional_power(num_qubits, ensemble_size, power, quantum_object):
    """Test raising a quantum object to an fractional power."""
    # TODO: Which objects does this make sense for?
    key = jax.random.key(1234)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    rank = 2**num_qubits
    q_object = _generate_random_object(quantum_object, key, ensemble_size, dims, rank)

    raised_object = q_object**power

    assert type(raised_object) is type(q_object)
    assert raised_object.ensemble_size == q_object.ensemble_size


## Binary operations


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((3,), (3,)),
        ((2, 4), ()),
        ((2, 4), (2, 4)),
    ],
)
@pytest.mark.parametrize(
    "object_1, object_2, expected_object",
    [
        # Choi left-hand
        (Choi, Choi, Choi),
        (Choi, KrausMap, KrausMap),
        (Choi, SuperOp, SuperOp),
        (Choi, PauliLiouville, PauliLiouville),
        (Choi, Unitary, Choi),
        (Choi, StateVector, DensityMatrix),
        (Choi, DensityMatrix, DensityMatrix),
        # SuperOp left-hand
        (SuperOp, Choi, Choi),
        (SuperOp, KrausMap, KrausMap),
        (SuperOp, SuperOp, SuperOp),
        (SuperOp, PauliLiouville, PauliLiouville),
        (SuperOp, Unitary, SuperOp),
        (SuperOp, StateVector, DensityMatrix),
        (SuperOp, DensityMatrix, DensityMatrix),
        # PauliLiouville left-hand
        (PauliLiouville, Choi, Choi),
        (PauliLiouville, KrausMap, KrausMap),
        (PauliLiouville, SuperOp, SuperOp),
        (PauliLiouville, PauliLiouville, PauliLiouville),
        (PauliLiouville, Unitary, PauliLiouville),
        (PauliLiouville, StateVector, DensityMatrix),
        (PauliLiouville, DensityMatrix, DensityMatrix),
        # KrausMap left-hand
        (KrausMap, Choi, Choi),
        (KrausMap, KrausMap, KrausMap),
        (KrausMap, SuperOp, SuperOp),
        (KrausMap, PauliLiouville, PauliLiouville),
        (KrausMap, Unitary, KrausMap),
        (KrausMap, StateVector, DensityMatrix),
        (KrausMap, DensityMatrix, DensityMatrix),
        # Unitary left-hand
        (Unitary, Choi, Choi),
        (Unitary, KrausMap, KrausMap),
        (Unitary, SuperOp, SuperOp),
        (Unitary, PauliLiouville, PauliLiouville),
        (Unitary, Unitary, Unitary),
        (Unitary, StateVector, StateVector),
        (Unitary, DensityMatrix, DensityMatrix),
        # StateVector left-hand
        # We currently don't allow ⟨ψ∣E
        # (StateVector, Choi, DensityMatrix),
        # (StateVector, KrausMap, DensityMatrix),
        # (StateVector, SuperOp, DensityMatrix),
        # (StateVector, PauliLiouville, DensityMatrix),
        # (StateVector, Unitary, DensityMatrix),
        (StateVector, StateVector, complex),  # complex scalar
        (StateVector, DensityMatrix, StateVector),
        # DensityMatrix left-hand
        # We currently don't allow ρ∣E
        # (DensityMatrix, Choi, DensityMatrix),
        # (DensityMatrix, KrausMap, DensityMatrix),
        # (DensityMatrix, SuperOp, DensityMatrix),
        # (DensityMatrix, PauliLiouville, DensityMatrix),
        # (DensityMatrix, Unitary, DensityMatrix),
        (DensityMatrix, StateVector, StateVector),
        (DensityMatrix, DensityMatrix, DensityMatrix),
    ],
)
def test_compositions(num_qubits, ensemble_size, object_1, object_2, expected_object):
    """This tests that the object compositons work, that they are the right type and the right dimension."""
    k = jax.random.key(1234)
    key_1, key_2 = jax.random.split(k)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    rank = 2**num_qubits
    ensemble_size_1, ensemble_size_2 = ensemble_size

    random_object_1 = _generate_random_object(object_1, key_1, ensemble_size_1, dims, rank)
    random_object_2 = _generate_random_object(object_2, key_2, ensemble_size_2, dims, rank)

    result = random_object_1 @ random_object_2

    broadcast_ensemble_size = jnp.broadcast_shapes(ensemble_size_1, ensemble_size_2)

    if expected_object is complex:
        assert isinstance(result, jax.Array)
        assert result.shape == broadcast_ensemble_size
    else:
        assert type(result) is expected_object
        assert result.ensemble_size == broadcast_ensemble_size

    # The shape logic is a bit tricky...
    # If the result is a Superoperator, it should have shape ensemble_size +(d^2, d^2)
    # If the result is a KrausMap it should have shape ensemble_size + (num_kraus, d, d)
    # If the result is a unitary it should have shape ensemble_size + (d, d)
    # If the result is a StateVector it should have shape ensemble_size + (d,)
    # If the results is a DensityMatrix it should have shape ensemble_size + (d, d)


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((3,), (3,)),
        ((2, 4), ()),
        ((2, 4), (2, 4)),
    ],
)
@pytest.mark.parametrize(
    "object_1, object_2, expected_object",
    [
        # Choi left-hand
        (Choi, Choi, Choi),
        (Choi, KrausMap, KrausMap),
        (Choi, SuperOp, SuperOp),
        (Choi, PauliLiouville, PauliLiouville),
        (Choi, Unitary, Choi),
        # SuperOp left-hand
        (SuperOp, Choi, Choi),
        (SuperOp, KrausMap, KrausMap),
        (SuperOp, SuperOp, SuperOp),
        (SuperOp, PauliLiouville, PauliLiouville),
        (SuperOp, Unitary, SuperOp),
        # PauliLiouville left-hand
        (PauliLiouville, Choi, Choi),
        (PauliLiouville, KrausMap, KrausMap),
        (PauliLiouville, SuperOp, SuperOp),
        (PauliLiouville, PauliLiouville, PauliLiouville),
        (PauliLiouville, Unitary, PauliLiouville),
        # KrausMap left-hand
        (KrausMap, Choi, Choi),
        (KrausMap, KrausMap, KrausMap),
        (KrausMap, SuperOp, SuperOp),
        (KrausMap, PauliLiouville, PauliLiouville),
        (KrausMap, Unitary, KrausMap),
        # Unitary left-hand
        (Unitary, Choi, Choi),
        (Unitary, KrausMap, KrausMap),
        (Unitary, SuperOp, SuperOp),
        (Unitary, PauliLiouville, PauliLiouville),
        (Unitary, Unitary, Unitary),
        # StateVector left-hand
        (StateVector, StateVector, StateVector),
        (StateVector, DensityMatrix, DensityMatrix),
        # DensityMatrix left-hand
        (DensityMatrix, StateVector, DensityMatrix),
        (DensityMatrix, DensityMatrix, DensityMatrix),
    ],
)
def test_tensor_products(num_qubits, ensemble_size, object_1, object_2, expected_object):
    """This tests that the object compositons work, that they are the right type and the right dimension."""
    k = jax.random.key(1234)
    key_1, key_2 = jax.random.split(k)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    rank = 2**num_qubits
    ensemble_size_1, ensemble_size_2 = ensemble_size

    random_object_1 = _generate_random_object(object_1, key_1, ensemble_size_1, dims, rank)
    random_object_2 = _generate_random_object(object_2, key_2, ensemble_size_2, dims, rank)

    result = random_object_1 | random_object_2

    broadcast_ensemble_size = jnp.broadcast_shapes(ensemble_size_1, ensemble_size_2)

    assert type(result) is expected_object
    assert result.ensemble_size == broadcast_ensemble_size

    # The shape logic is a bit tricky...
    # If the result is a Superoperator, it should have shape ensemble_size +(d0^2 * d1^2, d0^2 * d1^2)
    # If the result is a KrausMap it should have shape ensemble_size + (num_kraus, d0*d1, d0*d1)
    # If the result is a unitary it should have shape ensemble_size + (d0*d1, d0*d1)
    # If the result is a StateVector it should have shape ensemble_size + (d0*d1,)
    # If the results is a DensityMatrix it should have shape ensemble_size + (d0*d1, d0*d1)


@pytest.mark.parametrize("num_qubits", [1, 2])
@pytest.mark.parametrize(
    "ensemble_size",
    [
        ((), ()),
        ((3,), (3,)),
        ((2, 4), ()),
        ((2, 4), (2, 4)),
    ],
)
@pytest.mark.parametrize(
    "object_1, object_2",
    [
        # Choi left-hand
        (Choi, Choi),
        (Choi, KrausMap),
        (Choi, SuperOp),
        (Choi, PauliLiouville),
        (Choi, Unitary),
        # SuperOp left-hand
        (SuperOp, Choi),
        (SuperOp, KrausMap),
        (SuperOp, SuperOp),
        (SuperOp, PauliLiouville),
        (SuperOp, Unitary),
        # PauliLiouville left-hand
        (PauliLiouville, Choi),
        (PauliLiouville, KrausMap),
        (PauliLiouville, SuperOp),
        (PauliLiouville, PauliLiouville),
        (PauliLiouville, Unitary),
        # KrausMap left-hand
        (KrausMap, Choi),
        (KrausMap, KrausMap),
        (KrausMap, SuperOp),
        (KrausMap, PauliLiouville),
        (KrausMap, Unitary),
        # Unitary left-hand
        (Unitary, Choi),
        (Unitary, KrausMap),
        (Unitary, SuperOp),
        (Unitary, PauliLiouville),
        (Unitary, Unitary),
        # StateVector left-hand
        (StateVector, StateVector),
        (StateVector, DensityMatrix),
        # DensityMatrix left-hand
        (DensityMatrix, StateVector),
        (DensityMatrix, DensityMatrix),
    ],
)
def test_equality(num_qubits, ensemble_size, object_1, object_2):
    """Check equality between objects."""
    k = jax.random.key(1234)
    key_1, key_2 = jax.random.split(k)
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    rank = 2**num_qubits
    ensemble_size_1, ensemble_size_2 = ensemble_size

    random_object_1 = _generate_random_object(object_1, key_1, ensemble_size_1, dims, rank)
    random_object_2 = _generate_random_object(object_2, key_2, ensemble_size_2, dims, rank)

    assert random_object_1 == random_object_1
    assert random_object_1 != random_object_2
    assert random_object_2 == random_object_2


# @pytest.mark.parametrize("num_qubits", [1, 2, 3])
# def test_unitary(num_qubits):
#     """Test the behaviour of the Unitary type."""
#     d = 2**num_qubits
#     key = jax.random.key(42)
#     unitary = random_unitary(dims=((2,) * num_qubits, (2,) * num_qubits), key=key)
#     u = unitary.data

#     # Check attributes
#     assert unitary.data.shape == (d, d)
#     assert unitary.dims == ((2,) * num_qubits, (2,) * num_qubits)
#     assert jnp.allclose(unitary.data, u)

#     # Check h()
#     u_h = unitary.h
#     assert jnp.allclose(u_h.data, u.conj().T)

#     # Check phase multiplication
#     phase = jnp.exp(1j * jnp.pi / 4)
#     phased_unitary = phase * unitary
#     assert jnp.allclose(phased_unitary.data, phase * u)
#     assert type(phased_unitary) is Unitary

#     # Check scalar multiplication
#     scalar = 1.0
#     scaled_unitary = scalar * unitary
#     assert jnp.allclose(scaled_unitary.data, scalar * u)
#     assert type(scaled_unitary) is Unitary

#     # Check negation
#     negated_unitary = -unitary
#     assert jnp.allclose(negated_unitary.data, -u)
#     assert type(negated_unitary) is Unitary

#     # Check non-one scalar multiplication
#     scalar = 0.1
#     scaled_unitary = scalar * unitary
#     assert jnp.allclose(scaled_unitary.data, scalar * u)
#     assert type(scaled_unitary) is Kraus

#     # Check matmul
#     key2 = jax.random.key(43)
#     other_unitary = random_unitary(dims=((2,) * num_qubits, (2,) * num_qubits), key=key2)
#     other_u = other_unitary.data
#     prod_unitary = unitary @ other_unitary
#     assert jnp.allclose(prod_unitary.data, u @ other_u)
#     assert type(prod_unitary) is Unitary

#     # Check tensor
#     tensor_unitary = unitary | other_unitary
#     assert tensor_unitary.data.shape == (d * d, d * d)
#     assert tensor_unitary.dims == ((2,) * (2 * num_qubits), (2,) * (2 * num_qubits))
#     assert jnp.allclose(tensor_unitary.data, jnp.kron(u, other_u))
#     assert type(tensor_unitary) is Unitary

#     # Check integer power
#     power = 3
#     powered_unitary = unitary**power
#     assert jnp.isclose(unitary_entanglement_fidelity(powered_unitary, unitary @ unitary @ unitary), 1.0)
#     assert type(powered_unitary) is Unitary

#     # Check fractional power
#     fractional_power = 0.5
#     frac_powered_unitary = unitary**fractional_power
#     assert jnp.isclose(unitary_entanglement_fidelity(frac_powered_unitary @ frac_powered_unitary, unitary), 1.0)
#     assert type(frac_powered_unitary) is Unitary

#     # Check equality
#     assert unitary == unitary
#     assert unitary == phased_unitary
#     assert unitary != other_unitary

#     # Check broadcasting with unitaries
#     batch_dims = (2, 4)
#     key_batch = jax.random.key(99)
#     batched_unitaries_data = jnp.stack(
#         [
#             jnp.stack(
#                 [
#                     random_unitary(
#                         dims=((2,) * num_qubits, (2,) * num_qubits),
#                         key=jax.random.fold_in(key_batch, i * batch_dims[1] + j),
#                     ).data
#                     for j in range(batch_dims[1])
#                 ]
#             )
#             for i in range(batch_dims[0])
#         ]
#     )  # Shape (2, 4, d, d)
#     assert batched_unitaries_data.shape == batch_dims + (d, d)

#     batched_unitaries = Unitary(data=batched_unitaries_data, dims=((2,) * num_qubits, (2,) * num_qubits))
#     prod_batched = unitary @ batched_unitaries  # Should broadcast unitary
#     assert prod_batched.data.shape == batch_dims + (d, d)
#     assert type(prod_batched) is Unitary


# def test_unitary_state():
#     """Test the operations of a Unitary to a StateVector."""
#     num_qubits = 2
#     d = 2**num_qubits

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     unitary = random_unitary(dims=((2,) * num_qubits, (2,) * num_qubits), key=key1)
#     u = unitary.data

#     # Create random state vector
#     state = ginibre_matrix_complex(dim=d, k=1, key=key2)[:, 0]
#     state = state / jnp.linalg.norm(state)
#     state_vector = StateVector(data=state, dims=(2,) * num_qubits)

#     # Apply unitary to state
#     new_state_vector = unitary @ state_vector
#     expected_state = u @ state
#     assert jnp.allclose(new_state_vector.data, expected_state)
#     assert type(new_state_vector) is StateVector

#     # Apply unitary to batched states
#     batch_dims = (2, 4)
#     key_batch = jax.random.key(100)
#     batched_states = jnp.stack(
#         [
#             jnp.stack(
#                 [
#                     ginibre_matrix_complex(dim=d, k=1, key=jax.random.fold_in(key_batch, i * batch_dims[1] + j))[:, 0]
#                     / jnp.linalg.norm(
#                         ginibre_matrix_complex(dim=d, k=1, key=jax.random.fold_in(key_batch, i * batch_dims[1] + j))[
#                             :, 0
#                         ]
#                     )
#                     for j in range(batch_dims[1])
#                 ]
#             )
#             for i in range(batch_dims[0])
#         ]
#     )
#     batched_state_vectors = StateVector(data=batched_states, dims=(2,) * num_qubits)
#     new_batched_states = unitary @ batched_state_vectors
#     assert new_batched_states.data.shape == batch_dims + (d,)
#     assert type(new_batched_states) is StateVector

#     # Apply batched unitaries to state
#     key_batch2 = jax.random.key(101)
#     batched_unitaries_data = jnp.stack(
#         [
#             jnp.stack(
#                 [
#                     random_unitary(
#                         dims=((2,) * num_qubits, (2,) * num_qubits),
#                         key=jax.random.fold_in(key_batch2, i * batch_dims[1] + j),
#                     ).data
#                     for j in range(batch_dims[1])
#                 ]
#             )
#             for i in range(batch_dims[0])
#         ]
#     )
#     batched_unitaries = Unitary(data=batched_unitaries_data, dims=((2,) * num_qubits, (2,) * num_qubits))
#     new_batched_states = batched_unitaries @ state_vector
#     assert new_batched_states.data.shape == batch_dims + (d,)
#     assert type(new_batched_states) is StateVector


# def test_unitary_density_matrix():
#     """Test the operations of a Unitary to a DensityMatrix."""
#     num_qubits = 2
#     d = 2**num_qubits

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     unitary = random_unitary(dims=((2,) * num_qubits, (2,) * num_qubits), key=key1)
#     u = unitary.data

#     rho_matrix = ginibre_matrix_complex(d, d, key2)
#     rho_matrix = rho_matrix @ rho_matrix.conj().T
#     rho_matrix = rho_matrix / jnp.trace(rho_matrix)
#     density_matrix = DensityMatrix(data=rho_matrix, dims=(2,) * num_qubits)

#     # Apply unitary to density matrix
#     new_density_matrix = unitary @ density_matrix
#     expected_rho = u @ rho_matrix @ u.conj().T
#     assert jnp.allclose(new_density_matrix.data, expected_rho)
#     assert type(new_density_matrix) is DensityMatrix

#     # Apply unitary to batched density matrices
#     batch_dims = (2, 4)
#     key_batch = jax.random.key(102)
#     batched_rhos = jnp.stack(
#         [
#             jnp.stack(
#                 [
#                     (lambda m: m @ m.conj().T / jnp.trace(m @ m.conj().T))(
#                         ginibre_matrix_complex(d, d, key=jax.random.fold_in(key_batch, i * batch_dims[1] + j))
#                     )
#                     for j in range(batch_dims[1])
#                 ]
#             )
#             for i in range(batch_dims[0])
#         ]
#     )
#     batched_density_matrices = DensityMatrix(data=batched_rhos, dims=(2,) * num_qubits)
#     new_batched_rhos = unitary @ batched_density_matrices
#     assert new_batched_rhos.data.shape == batch_dims + (d, d)
#     assert type(new_batched_rhos) is DensityMatrix

#     # Apply batched unitaries to density matrix
#     key_batch2 = jax.random.key(103)
#     batched_unitaries_data = jnp.stack(
#         [
#             jnp.stack(
#                 [
#                     random_unitary(
#                         dims=((2,) * num_qubits, (2,) * num_qubits),
#                         key=jax.random.fold_in(key_batch2, i * batch_dims[1] + j),
#                     ).data
#                     for j in range(batch_dims[1])
#                 ]
#             )
#             for i in range(batch_dims[0])
#         ]
#     )
#     batched_unitaries = Unitary(data=batched_unitaries_data, dims=((2,) * num_qubits, (2,) * num_qubits))
#     new_batched_rhos = batched_unitaries @ density_matrix
#     assert new_batched_rhos.data.shape == batch_dims + (d, d)
#     assert type(new_batched_rhos) is DensityMatrix


# def test_unitary_superop():
#     """Test the operations of a Unitary to a SuperOp."""
#     num_qubits = 1
#     d = 2**num_qubits

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     unitary = random_unitary(dims=((2,) * num_qubits, (2,) * num_qubits), key=key1)

#     kraus_rank = d
#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key2)
#     superop = choi_to_superop(choi)

#     # Apply unitary to superoperator
#     new_superop = unitary @ superop
#     assert new_superop.data.shape == (d * d, d * d)
#     assert type(new_superop) is SuperOp

#     # Apply superoperator to unitary
#     new_superop = superop @ unitary
#     assert new_superop.data.shape == (d * d, d * d)
#     assert type(new_superop) is SuperOp

#     # Apply batched unitaries to superoperator
#     # batch_dims = (2, 4)
#     # batched_unitaries = jnp.asarray(
#     #     [[haar_rand_unitary(d) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_unitaries = Unitary(data=batched_unitaries, dims=((2,) * num_qubits, (2,) * num_qubits))
#     # new_superop = batched_unitaries @ superop
#     # assert new_superop.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_superop) is Choi

#     # # Apply superoperator to batched unitaries
#     # new_superop = superop @ batched_unitaries
#     # assert new_superop.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_superop) is Choi


# def test_unitary_choi():
#     """Test the operations of a Unitary to a Choi."""
#     num_qubits = 1
#     d = 2**num_qubits

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     unitary = random_unitary(dims=((2,) * num_qubits, (2,) * num_qubits), key=key1)

#     kraus_rank = d
#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key2)

#     # Apply unitary to choi
#     new_choi = unitary @ choi
#     assert new_choi.data.shape == (d * d, d * d)
#     assert type(new_choi) is Choi

#     # Apply choi to unitary
#     new_choi = choi @ unitary
#     assert new_choi.data.shape == (d * d, d * d)
#     assert type(new_choi) is Choi

#     # Apply batched unitaries to choi
#     # batch_dims = (2, 4)
#     # batched_unitaries = jnp.asarray(
#     #     [[haar_rand_unitary(d) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_unitaries = Unitary(data=batched_unitaries, dims=((2,) * num_qubits, (2,) * num_qubits))
#     # new_choi = batched_unitaries @ choi
#     # assert new_choi.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_choi) is Choi

#     # # Apply choi to batched unitaries
#     # new_choi = choi @ batched_unitaries
#     # assert new_choi.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_choi) is Choi


# def test_unitary_pauli_liouville():
#     """Test the operations of a Unitary to a PauliLiouville."""
#     num_qubits = 1
#     d = 2**num_qubits

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     unitary = random_unitary(dims=((2,) * num_qubits, (2,) * num_qubits), key=key1)

#     kraus_rank = d
#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key2)
#     pl = choi_to_pauli_liouville(choi)

#     # Apply unitary to pauli liouville
#     new_pl = unitary @ pl
#     assert new_pl.data.shape == (d * d, d * d)
#     assert type(new_pl) is PauliLiouville

#     # Apply pauli liouville to unitary
#     new_pl = pl @ unitary
#     assert new_pl.data.shape == (d * d, d * d)
#     assert type(new_pl) is PauliLiouville

#     # Apply batched unitaries to pauli liouville
#     # batch_dims = (2, 4)
#     # batched_unitaries = jnp.asarray(
#     #     [[haar_rand_unitary(d) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_unitaries = Unitary(data=batched_unitaries, dims=((2,) * num_qubits, (2,) * num_qubits))
#     # new_pl = batched_unitaries @ pl
#     # assert new_pl.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_pl) is Choi

#     # # Apply pauli liouville to batched unitaries
#     # new_pl = pl @ batched_unitaries
#     # assert new_pl.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_pl) is Choi


# def test_choi():
#     """Test the behaviour of the Choi type."""
#     num_qubits = 2
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key)
#     choi_matrix = choi.data

#     # Check attributes
#     assert choi.data.shape == (d * d, d * d)
#     assert choi.dims == ((2,) * num_qubits, (2,) * num_qubits)
#     assert jnp.allclose(choi.data, choi_matrix)

#     # Check h()
#     choi_h = choi.h
#     assert jnp.allclose(choi_h.data, choi_matrix.conj().T)

#     # Check phase multiplication
#     phase = jnp.exp(1j * jnp.pi / 4)
#     phased_choi = phase * choi
#     assert jnp.allclose(phased_choi.data, phase * choi_matrix)
#     assert type(phased_choi) is Choi

#     # Check scalar multiplication
#     scalar = 1.0
#     scaled_choi = scalar * choi
#     assert jnp.allclose(scaled_choi.data, scalar * choi_matrix)
#     assert type(scaled_choi) is Choi

#     # Check negation
#     negated_choi = -choi
#     assert jnp.allclose(negated_choi.data, -choi_matrix)
#     assert type(negated_choi) is Choi


# def test_choi_state_vector():
#     """Test the operations of a Choi to a StateVector."""
#     num_qubits = 2
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key1)

#     # Create random state vector
#     state = ginibre_matrix_complex(dim=d, k=1, key=key2)[:, 0]
#     state = state / jnp.linalg.norm(state)
#     state_vector = StateVector(data=state, dims=(2,) * num_qubits)

#     # Apply choi to state
#     new_density_matrix = choi @ state_vector
#     assert new_density_matrix.data.shape == (d, d)
#     assert type(new_density_matrix) is DensityMatrix

#     # Apply batched choi to state
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_choi = Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits))
#     # new_density_matrix = batched_choi @ state_vector
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix

#     # # Apply choi to batched states
#     # batched_states = jnp.asarray([[haar_rand_state(d) for j in range(batch_dims[1])] for i in range(batch_dims[0])])
#     # batched_states = StateVector(data=batched_states, dims=(2,) * num_qubits)
#     # new_density_matrix = choi @ batched_states
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix


# def test_choi_density_matrix():
#     """Test the operations of a Choi to a DensityMatrix."""
#     num_qubits = 2
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key1)

#     rho_matrix = ginibre_matrix_complex(d, d, key2)
#     rho_matrix = rho_matrix @ rho_matrix.conj().T
#     rho_matrix = rho_matrix / jnp.trace(rho_matrix)
#     density_matrix = DensityMatrix(data=rho_matrix, dims=(2,) * num_qubits)

#     # Apply choi to density matrix
#     new_density_matrix = choi @ density_matrix
#     assert new_density_matrix.data.shape == (d, d)
#     assert type(new_density_matrix) is DensityMatrix

#     # Apply batched choi to density matrix
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_choi = Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits))
#     # new_density_matrix = batched_choi @ density_matrix
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix

#     # # Apply choi to batched density matrices
#     # batched_rhos = jnp.asarray(
#     #     [
#     #         [
#     #             (lambda m: m @ m.conj().T / jnp.trace(m @ m.conj().T))(ginibre_matrix_complex(d, d))
#     #             for j in range(batch_dims[1])
#     #         ]
#     #         for i in range(batch_dims[0])
#     #     ]
#     # )
#     # batched_density_matrices = DensityMatrix(data=batched_rhos, dims=(2,) * num_qubits)
#     # new_density_matrix = choi @ batched_density_matrices
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix


# def test_choi_superop():
#     """Test the operations of a Choi to a SuperOp."""
#     num_qubits = 1
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key)

#     superop = choi_to_superop(choi)

#     # Apply choi to superoperator
#     new_choi = choi @ superop
#     assert new_choi.data.shape == (d * d, d * d)
#     assert type(new_choi) is Choi

#     # Apply superoperator to choi
#     new_choi = superop @ choi
#     assert new_choi.data.shape == (d * d, d * d)
#     assert type(new_choi) is Choi

#     # Apply batched choi to superoperator
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_choi = Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits))
#     # new_choi = batched_choi @ superop
#     # assert new_choi.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_choi) is Choi

#     # # Apply superoperator to batched choi
#     # new_choi = superop @ batched_choi
#     # assert new_choi.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_choi) is Choi


# def test_choi_unitary():
#     """Test the operations of a Choi to a Unitary."""
#     num_qubits = 1
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key1)

#     unitary = random_unitary(dims=((2,) * num_qubits, (2,) * num_qubits), key=key2)

#     # Apply choi to unitary
#     new_choi = choi @ unitary
#     assert new_choi.data.shape == (d * d, d * d)
#     assert type(new_choi) is Choi

#     # Apply unitary to choi
#     new_choi = unitary @ choi
#     assert new_choi.data.shape == (d * d, d * d)
#     assert type(new_choi) is Choi

#     # Apply batched choi to unitary
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_choi = Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits))
#     # new_choi = batched_choi @ unitary
#     # assert new_choi.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_choi) is Choi

#     # # Apply unitary to batched choi
#     # new_choi = unitary @ batched_choi
#     # assert new_choi.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_choi) is Choi


# def test_choi_pauli_liouville():
#     """Test the operations of a Choi to a PauliLiouville."""
#     num_qubits = 1
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key)

#     pl = choi_to_pauli_liouville(choi)

#     # Apply choi to pauli liouville
#     new_choi = choi @ pl
#     assert new_choi.data.shape == (d * d, d * d)
#     assert type(new_choi) is Choi

#     # Apply pauli liouville to choi
#     new_choi = pl @ choi
#     assert new_choi.data.shape == (d * d, d * d)
#     assert type(new_choi) is Choi

#     # Apply batched choi to pauli liouville
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_choi = Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits))
#     # new_choi = batched_choi @ pl
#     # assert new_choi.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_choi) is Choi

#     # # Apply pauli liouville to batched choi
#     # new_choi = pl @ batched_choi
#     # assert new_choi.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_choi) is Choi


# def test_pauli_liouville():
#     """Test the behaviour of the PauliLiouville type."""
#     num_qubits = 2
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key)
#     pl = choi_to_pauli_liouville(choi)
#     pl_matrix = pl.data

#     # Check attributes
#     assert pl.data.shape == (d * d, d * d)
#     assert pl.dims == ((2,) * num_qubits, (2,) * num_qubits)
#     assert jnp.allclose(pl.data, pl_matrix)

#     # Check h()
#     pl_h = pl.h
#     assert jnp.allclose(pl_h.data, pl_matrix.conj().T)

#     # Check phase multiplication
#     phase = jnp.exp(1j * jnp.pi / 4)
#     phased_pl = phase * pl
#     assert jnp.allclose(phased_pl.data, phase * pl_matrix)
#     assert type(phased_pl) is PauliLiouville

#     # Check scalar multiplication
#     scalar = 1.0
#     scaled_pl = scalar * pl
#     assert jnp.allclose(scaled_pl.data, scalar * pl_matrix)
#     assert type(scaled_pl) is PauliLiouville

#     # Check negation
#     negated_pl = -pl
#     assert jnp.allclose(negated_pl.data, -pl_matrix)
#     assert type(negated_pl) is PauliLiouville


# def test_pauli_liouville_state_vector():
#     """Test the operations of a PauliLiouville to a StateVector."""
#     num_qubits = 2
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key1)
#     pl = choi_to_pauli_liouville(choi)

#     # Create random state vector
#     state = ginibre_matrix_complex(dim=d, k=1, key=key2)[:, 0]
#     state = state / jnp.linalg.norm(state)
#     state_vector = StateVector(data=state, dims=(2,) * num_qubits)

#     # Apply pauli liouville to state
#     new_density_matrix = pl @ state_vector
#     assert new_density_matrix.data.shape == (d, d)
#     assert type(new_density_matrix) is DensityMatrix

#     # Apply batched pauli liouville to state
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_pl = choi_to_pauli_liouville(Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits)))
#     # new_density_matrix = batched_pl @ state_vector
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix

#     # # Apply pauli liouville to batched states
#     # batched_states = jnp.asarray([[haar_rand_state(d) for j in range(batch_dims[1])] for i in range(batch_dims[0])])
#     # batched_states = StateVector(data=batched_states, dims=(2,) * num_qubits)
#     # new_density_matrix = pl @ batched_states
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix


# def test_pauli_liouville_density_matrix():
#     """Test the operations of a PauliLiouville to a DensityMatrix."""
#     num_qubits = 2
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key1)
#     pl = choi_to_pauli_liouville(choi)

#     rho_matrix = ginibre_matrix_complex(d, d, key2)
#     rho_matrix = rho_matrix @ rho_matrix.conj().T
#     rho_matrix = rho_matrix / jnp.trace(rho_matrix)
#     density_matrix = DensityMatrix(data=rho_matrix, dims=(2,) * num_qubits)

#     # Apply pauli liouville to density matrix
#     new_density_matrix = pl @ density_matrix
#     assert new_density_matrix.data.shape == (d, d)
#     assert type(new_density_matrix) is DensityMatrix

#     # Apply batched pauli liouville to density matrix
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_pl = choi_to_pauli_liouville(Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits)))
#     # new_density_matrix = batched_pl @ density_matrix
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix

#     # # Apply pauli liouville to batched density matrices
#     # batched_rhos = jnp.asarray(
#     #     [
#     #         [
#     #             (lambda m: m @ m.conj().T / jnp.trace(m @ m.conj().T))(ginibre_matrix_complex(d, d))
#     #             for j in range(batch_dims[1])
#     #         ]
#     #         for i in range(batch_dims[0])
#     #     ]
#     # )
#     # batched_density_matrices = DensityMatrix(data=batched_rhos, dims=(2,) * num_qubits)
#     # new_density_matrix = pl @ batched_density_matrices
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix


# def test_pauli_liouville_superop():
#     """Test the operations of a PauliLiouville to a SuperOp."""
#     num_qubits = 1
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key)
#     pl = choi_to_pauli_liouville(choi)

#     superop = choi_to_superop(choi)

#     # Apply pauli liouville to superoperator
#     new_pl = pl @ superop
#     assert new_pl.data.shape == (d * d, d * d)
#     assert type(new_pl) is PauliLiouville

#     # Apply superoperator to pauli liouville
#     new_pl = superop @ pl
#     assert new_pl.data.shape == (d * d, d * d)
#     assert type(new_pl) is PauliLiouville

#     # Apply batched pauli liouville to superoperator
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_pl = choi_to_pauli_liouville(Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits)))
#     # new_pl = batched_pl @ superop
#     # assert new_pl.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_pl) is Choi

#     # # Apply superoperator to batched pauli liouville
#     # new_pl = superop @ batched_pl
#     # assert new_pl.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_pl) is Choi


# def test_pauli_liouville_unitary():
#     """Test the operations of a PauliLiouville to a Unitary."""
#     num_qubits = 1
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key1)
#     pl = choi_to_pauli_liouville(choi)

#     unitary = random_unitary(dims=((2,) * num_qubits, (2,) * num_qubits), key=key2)

#     # Apply pauli liouville to unitary
#     new_pl = pl @ unitary
#     assert new_pl.data.shape == (d * d, d * d)
#     assert type(new_pl) is PauliLiouville

#     # Apply unitary to pauli liouville
#     new_pl = unitary @ pl
#     assert new_pl.data.shape == (d * d, d * d)
#     assert type(new_pl) is PauliLiouville

#     # Apply batched pauli liouville to unitary


# #     batch_dims = (2, 4)
# #     batched_choi = jnp.asarray(
# #         [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
# #     )
# #     batched_pl = choi_to_pauli_liouville(Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits)))
# #     new_pl = batched_pl @ unitary
# #     assert new_pl.data.shape == batch_dims + (d * d, d * d)
# #     assert type(new_pl) is Choi

# #     # Apply unitary to batched pauli liouville
# #     new_pl = unitary @ batched_pl
# #     assert new_pl.data.shape == batch_dims + (d * d, d * d)
# #     assert type(new_pl) is Choi


# def test_superoperator():
#     """Test the behaviour of the SuperOp type."""
#     num_qubits = 2
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key)
#     superop = choi_to_superop(choi)
#     superop_matrix = superop.data

#     # Check attributes
#     assert superop.data.shape == (d * d, d * d)
#     assert superop.dims == ((2,) * num_qubits, (2,) * num_qubits)
#     assert jnp.allclose(superop.data, superop_matrix)

#     # Check h()
#     superop_h = superop.h
#     assert jnp.allclose(superop_h.data, superop_matrix.conj().T)

#     # Check phase multiplication
#     phase = jnp.exp(1j * jnp.pi / 4)
#     phased_superop = phase * superop
#     assert jnp.allclose(phased_superop.data, phase * superop_matrix)
#     assert type(phased_superop) is SuperOp

#     # Check scalar multiplication
#     scalar = 1.0
#     scaled_superop = scalar * superop
#     assert jnp.allclose(scaled_superop.data, scalar * superop_matrix)
#     assert type(scaled_superop) is SuperOp

#     # Check negation
#     negated_superop = -superop
#     assert jnp.allclose(negated_superop.data, -superop_matrix)
#     assert type(negated_superop) is SuperOp


# def test_superoperator_state_vector():
#     """Test the operations of a SuperOp to a StateVector."""
#     num_qubits = 2
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key1)
#     superop = choi_to_superop(choi)

#     # Create random state vector
#     state = ginibre_matrix_complex(dim=d, k=1, key=key2)[:, 0]
#     state = state / jnp.linalg.norm(state)
#     state_vector = StateVector(data=state, dims=(2,) * num_qubits)

#     # Apply superoperator to state
#     new_density_matrix = superop @ state_vector
#     assert new_density_matrix.data.shape == (d, d)
#     assert type(new_density_matrix) is DensityMatrix

#     # Apply batched superoperator to state
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_superop = choi_to_superop(Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits)))
#     # new_density_matrix = batched_superop @ state_vector
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix

#     # # Apply superoperator to batched states
#     # batched_states = jnp.asarray([[haar_rand_state(d) for j in range(batch_dims[1])] for i in range(batch_dims[0])])
#     # batched_states = StateVector(data=batched_states, dims=(2,) * num_qubits)
#     # new_density_matrix = superop @ batched_states
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix


# def test_superoperator_density_matrix():
#     """Test the operations of a SuperOp to a DensityMatrix."""
#     num_qubits = 2
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key1)
#     superop = choi_to_superop(choi)

#     rho_matrix = ginibre_matrix_complex(d, d, key2)
#     rho_matrix = rho_matrix @ rho_matrix.conj().T
#     rho_matrix = rho_matrix / jnp.trace(rho_matrix)
#     density_matrix = DensityMatrix(data=rho_matrix, dims=(2,) * num_qubits)

#     # Apply superoperator to density matrix
#     new_density_matrix = superop @ density_matrix
#     assert new_density_matrix.data.shape == (d, d)
#     assert type(new_density_matrix) is DensityMatrix

#     # Apply batched superoperator to density matrix
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_superop = choi_to_superop(Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits)))
#     # new_density_matrix = batched_superop @ density_matrix
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix

#     # # Apply superoperator to batched density matrices
#     # batched_rhos = jnp.asarray(
#     #     [
#     #         [
#     #             (lambda m: m @ m.conj().T / jnp.trace(m @ m.conj().T))(ginibre_matrix_complex(d, d))
#     #             for j in range(batch_dims[1])
#     #         ]
#     #         for i in range(batch_dims[0])
#     #     ]
#     # )
#     # batched_density_matrices = DensityMatrix(data=batched_rhos, dims=(2,) * num_qubits)
#     # new_density_matrix = superop @ batched_density_matrices
#     # assert new_density_matrix.data.shape == batch_dims + (d, d)
#     # assert type(new_density_matrix) is DensityMatrix


# def test_superoperator_choi():
#     """Test the operations of a SuperOp to a Choi."""
#     num_qubits = 1
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key)
#     superop = choi_to_superop(choi)

#     # Apply superoperator to choi
#     new_choi = superop @ choi
#     assert new_choi.data.shape == (d * d, d * d)
#     assert type(new_choi) is Choi

#     # Apply choi to superoperator
#     new_choi = choi @ superop
#     assert new_choi.data.shape == (d * d, d * d)
#     assert type(new_choi) is Choi

#     # Apply batched superoperator to choi
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_superop = choi_to_superop(Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits)))
#     # new_choi = batched_superop @ choi
#     # assert new_choi.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_choi) is Choi

#     # # Apply choi to batched superoperator
#     # new_choi = choi @ batched_superop
#     # assert new_choi.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_choi) is Choi


# def test_superoperator_unitary():
#     """Test the operations of a SuperOp to a Unitary."""
#     num_qubits = 1
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     key1, key2 = jax.random.split(key)

#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key1)
#     superop = choi_to_superop(choi)

#     unitary = random_unitary(dims=((2,) * num_qubits, (2,) * num_qubits), key=key2)

#     # Apply superoperator to unitary
#     new_superop = superop @ unitary
#     assert new_superop.data.shape == (d * d, d * d)
#     assert type(new_superop) is SuperOp

#     # Apply unitary to superoperator
#     new_superop = unitary @ superop
#     assert new_superop.data.shape == (d * d, d * d)
#     assert type(new_superop) is SuperOp

#     # Apply batched superoperator to unitary
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_superop = choi_to_superop(Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits)))
#     # new_superop = batched_superop @ unitary
#     # assert new_superop.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_superop) is Choi

#     # # Apply unitary to batched superoperator
#     # new_superop = unitary @ batched_superop
#     # assert new_superop.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_superop) is Choi


# def test_superoperator_pauli_liouville():
#     """Test the operations of a SuperOp to a PauliLiouville."""
#     num_qubits = 1
#     d = 2**num_qubits
#     kraus_rank = d

#     key = jax.random.key(42)
#     choi = random_choi_BCSZ(dims=((2,) * num_qubits, (2,) * num_qubits), rank=kraus_rank, key=key)
#     superop = choi_to_superop(choi)

#     pl = choi_to_pauli_liouville(choi)

#     # Apply superoperator to pauli liouville
#     new_pl = superop @ pl
#     assert new_pl.data.shape == (d * d, d * d)
#     assert type(new_pl) is PauliLiouville

#     # Apply pauli liouville to superoperator
#     new_pl = pl @ superop
#     assert new_pl.data.shape == (d * d, d * d)
#     assert type(new_pl) is PauliLiouville

#     # Apply batched superoperator to pauli liouville
#     # batch_dims = (2, 4)
#     # batched_choi = jnp.asarray(
#     #     [[rand_map_with_BCSZ_dist(d, kraus_rank) for j in range(batch_dims[1])] for i in range(batch_dims[0])]
#     # )
#     # batched_superop = choi_to_superop(Choi(data=batched_choi, dims=((2,) * num_qubits, (2,) * num_qubits)))
#     # new_pl = batched_superop @ pl
#     # assert new_pl.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_pl) is Choi

#     # # Apply pauli liouville to batched superoperator
#     # new_pl = pl @ batched_superop
#     # assert new_pl.data.shape == batch_dims + (d * d, d * d)
#     # assert type(new_pl) is Choi
