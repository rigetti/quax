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
import numpy as np
import pytest
from itertools import product

import quax as qx
from quax import (
    Choi,
    DensityMatrix,
    Involution,
    KrausMap,
    Observable,
    Operator,
    PauliLiouville,
    StateVector,
    SuperOp,
    Unitary,
    choi_to_kraus,
    choi_to_pauli_liouville,
    choi_to_superop,
    random_choi,
    random_density_matrix,
    random_state_vector,
    random_unitary,
)
from .instrument_helpers import (
    basis_dm,
    basis_dm_multi,
    basis_sv,
    superposition_dm,
)


def _generate_random_object(object_type, key, size, dims, rank):
    match object_type:
        case t if t is Choi:
            return random_choi(dims, rank, key, size)
        case t if t is SuperOp:
            return choi_to_superop(random_choi(dims, rank, key, size))
        case t if t is KrausMap:
            return choi_to_kraus(random_choi(dims, rank, key, size))
        case t if t is PauliLiouville:
            return choi_to_pauli_liouville(random_choi(dims, rank, key, size))
        case t if t is StateVector:
            return random_state_vector(dims[0], key, size)
        case t if t is DensityMatrix:
            return random_density_matrix(rank, dims[0], key, size)
        case t if t is Unitary:
            return random_unitary(dims, key, size)
        case t if t is Observable:
            return qx.random_observable(dims, key, size)
        case t if t is Operator:
            return qx.random_operator(dims, key, size)
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
@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_negation(num_qubits, ensemble_size, quantum_object, qudit_dim):
    """Test negation of the object (__neg__)."""
    key = jax.random.key(1234)
    dims = ((qudit_dim,) * num_qubits, (qudit_dim,) * num_qubits)
    rank = qudit_dim**num_qubits

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
    "quantum_object",
    [Choi, KrausMap, SuperOp, PauliLiouville, Unitary, Operator, Observable, StateVector, DensityMatrix],
)
@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_scalar_multiplication(num_qubits, ensemble_size, scalar, quantum_object, qudit_dim):
    """Test multiplication of the object (__mul__ and __rmul__)."""
    key = jax.random.key(1234)
    dims = ((qudit_dim,) * num_qubits, (qudit_dim,) * num_qubits)
    rank = qudit_dim**num_qubits
    q_object = _generate_random_object(quantum_object, key, ensemble_size, dims, rank)

    # Check forward mul
    scaled_q_object = scalar * q_object

    assert jnp.allclose(scaled_q_object.data, scalar * q_object.data)

    if isinstance(q_object, Observable) and not jnp.isreal(scalar):
        # Observable * complex scalar loses Hermitian structure → Operator
        assert type(scaled_q_object) is Operator
    elif isinstance(q_object, Unitary):
        # Unitary * any scalar → Operator (unitarity not generally preserved)
        assert type(scaled_q_object) is Operator
    else:
        assert type(scaled_q_object) is type(q_object)
    assert scaled_q_object.ensemble_size == q_object.ensemble_size

    # Check reverse mul
    scaled_q_object = q_object * scalar

    assert jnp.allclose(scaled_q_object.data, scalar * q_object.data)

    if isinstance(q_object, Observable) and not jnp.isreal(scalar):
        # Observable * complex scalar loses Hermitian structure → Operator
        assert type(scaled_q_object) is Operator
    elif isinstance(q_object, Unitary):
        # Unitary * any scalar → Operator (unitarity not generally preserved)
        assert type(scaled_q_object) is Operator
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
@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_integer_power(num_qubits, ensemble_size, power, quantum_object, qudit_dim):
    """Test raising a quantum object to an integer power."""
    key = jax.random.key(1234)
    dims = ((qudit_dim,) * num_qubits, (qudit_dim,) * num_qubits)
    rank = qudit_dim**num_qubits
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
@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_fractional_power(num_qubits, ensemble_size, power, quantum_object, qudit_dim):
    """Test raising a quantum object to an fractional power."""
    # TODO: Which objects does this make sense for?
    key = jax.random.key(1234)
    dims = ((qudit_dim,) * num_qubits, (qudit_dim,) * num_qubits)
    rank = qudit_dim**num_qubits
    q_object = _generate_random_object(quantum_object, key, ensemble_size, dims, rank)

    raised_object = q_object**power

    assert type(raised_object) is type(q_object)
    assert raised_object.ensemble_size == q_object.ensemble_size


## Binary operations


@pytest.mark.parametrize("num_qubits, qudit_dim", [(1, 2), (2, 2), (3, 2), (1, 3), (2, 3)])
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
def test_compositions(num_qubits, ensemble_size, object_1, object_2, expected_object, qudit_dim):
    """This tests that the object compositons work, that they are the right type and the right dimension."""
    k = jax.random.key(1234)
    key_1, key_2 = jax.random.split(k)
    dims = ((qudit_dim,) * num_qubits, (qudit_dim,) * num_qubits)
    rank = qudit_dim**num_qubits
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


@pytest.mark.parametrize("num_qubits, qudit_dim", [(1, 2), (2, 2), (1, 3)])
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
def test_tensor_products(num_qubits, ensemble_size, object_1, object_2, expected_object, qudit_dim):
    """This tests that the object compositons work, that they are the right type and the right dimension."""
    k = jax.random.key(1234)
    key_1, key_2 = jax.random.split(k)
    dims = ((qudit_dim,) * num_qubits, (qudit_dim,) * num_qubits)
    rank = qudit_dim**num_qubits
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


@pytest.mark.parametrize("num_qubits, qudit_dim", [(1, 2), (2, 2), (3, 2), (1, 3), (2, 3)])
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
def test_equality(num_qubits, ensemble_size, object_1, object_2, qudit_dim):
    """Check equality between objects."""
    k = jax.random.key(1234)
    key_1, key_2 = jax.random.split(k)
    dims = ((qudit_dim,) * num_qubits, (qudit_dim,) * num_qubits)
    rank = qudit_dim**num_qubits
    ensemble_size_1, ensemble_size_2 = ensemble_size

    random_object_1 = _generate_random_object(object_1, key_1, ensemble_size_1, dims, rank)
    random_object_2 = _generate_random_object(object_2, key_2, ensemble_size_2, dims, rank)

    assert random_object_1 == random_object_1
    assert random_object_1 != random_object_2
    assert random_object_2 == random_object_2


@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_indexing(qudit_dim):
    """Test that an ensemble can be indexed."""
    key = jax.random.key(1234)
    num_qudits = 4
    ensemble_size = (10, 4)
    dims = (qudit_dim,) * num_qudits
    states = qx.zero_state_vector(dims=dims, ensemble_size=ensemble_size)
    assert states.ensemble_size == ensemble_size

    state_row = states[0]
    assert state_row.ensemble_size == (4,)
    state_col = states[:, 0]
    assert state_col.ensemble_size == (10,)
    one_state = states[0, 0]
    assert one_state.ensemble_size == ()
    selected_states = states[[2, 3], :]
    assert selected_states.ensemble_size == (2, 4)
    selected_states = states[3][jnp.array([2, 3])]
    assert selected_states.ensemble_size == (2,)

    states = qx.zero_state_matrix(dims=dims, ensemble_size=ensemble_size)
    assert states.ensemble_size == ensemble_size

    state_row = states[0]
    assert state_row.ensemble_size == (4,)
    state_col = states[:, 0]
    assert state_col.ensemble_size == (10,)
    one_state = states[0, 0]
    assert one_state.ensemble_size == ()
    selected_states = states[[2, 3], :]
    assert selected_states.ensemble_size == (2, 4)
    selected_states = states[3][jnp.array([2, 3])]
    assert selected_states.ensemble_size == (2,)

    op_dims = (dims, dims)
    operators = qx.random_unitary(dims=op_dims, size=ensemble_size, key=key)
    assert operators.ensemble_size == ensemble_size

    operator_row = operators[0]
    assert operator_row.ensemble_size == (4,)
    operator_col = operators[:, 0]
    assert operator_col.ensemble_size == (10,)
    one_operator = operators[0, 0]
    assert one_operator.ensemble_size == ()
    selected_operators = operators[[2, 3], :]
    assert selected_operators.ensemble_size == (2, 4)
    selected_operators = operators[3][jnp.array([2, 3])]
    assert selected_operators.ensemble_size == (2,)

    # test overindex
    with pytest.raises(IndexError):
        states[0, 0, 0]

    with pytest.raises(IndexError):
        operators[0, 0, 0]


@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_operator_algebra(qudit_dim):
    """
    Check that the algebra of various operator types is handled correctly.

    | Operation            | Operators (O) | Observables (H)               | Unitaries (U)              | Involutions (I)                          |
    |----------------------|---------------|-------------------------------|----------------------------|-------------------------------------------|
    | Addition (+)         | O             | H                             | O                          | H                                         |
    | Subtraction (−)      | O             | H                             | O                          | H                                         |
    | Composition (·)      | O             | H if commute                  | U                          | U (H if commute)                          |
    | Tensor product (⊗)   | O             | H                             | U                          | I                                         |
    | Scalar mult. (c·A)   | O             | H if c ∈ ℝ                    | O (always)                 | H if c ∈ ℝ                                |
    """
    dims = ((qudit_dim, qudit_dim), (qudit_dim, qudit_dim))
    d_total = qudit_dim**2
    key = jax.random.key(1234)
    operator = qx.random_operator(dims=dims, key=key)
    observable = qx.random_observable(dims=dims, key=key)
    unitary = qx.random_unitary(dims=dims, key=key)
    # Build a qudit involution: a diagonal matrix with eigenvalues ±1
    diag_vals = jnp.array([(-1) ** i for i in range(d_total)], dtype=complex)
    involution = Involution.from_matrix(jnp.diag(diag_vals), dims)

    # Addition
    assert type(operator + operator) is type(operator)
    assert type(observable + observable) is type(observable)
    assert type(unitary + unitary) is Operator
    assert type(involution + involution) is Observable
    assert type(operator + observable) is Operator
    assert type(operator + unitary) is Operator
    assert type(operator + involution) is Operator
    assert type(observable + unitary) is Operator
    assert type(observable + involution) is Observable
    assert type(unitary + involution) is Operator
    # reverse order
    assert type(observable + operator) is Operator
    assert type(unitary + operator) is Operator
    assert type(involution + operator) is Operator
    assert type(unitary + observable) is Operator
    assert type(involution + observable) is Observable
    assert type(involution + unitary) is Operator

    # Subtraction
    assert type(operator - operator) is type(operator)
    assert type(observable - observable) is type(observable)
    assert type(unitary - unitary) is Operator
    assert type(involution - involution) is Observable
    assert type(operator - observable) is Operator
    assert type(operator - unitary) is Operator
    assert type(operator - involution) is Operator
    assert type(observable - unitary) is Operator
    assert type(observable - involution) is Observable
    assert type(unitary - involution) is Operator
    # reverse order
    assert type(observable - operator) is Operator
    assert type(unitary - operator) is Operator
    assert type(involution - operator) is Operator
    assert type(unitary - observable) is Operator
    assert type(involution - observable) is Observable
    assert type(involution - unitary) is Operator

    # Composition
    assert type(operator @ operator) is type(operator)
    assert type(observable @ observable) is Operator  # product of non-commuting Hermitians is not generally Hermitian
    assert type(unitary @ unitary) is Unitary
    assert type(involution @ involution) is Unitary
    assert type(operator @ observable) is Operator
    assert type(operator @ unitary) is Operator
    assert type(operator @ involution) is Operator
    assert type(observable @ unitary) is Operator
    assert type(observable @ involution) is Operator  # H·I is not generally Hermitian or Unitary
    assert type(unitary @ involution) is Unitary  # product of unitaries is unitary
    # reverse order
    assert type(observable @ operator) is Operator
    assert type(unitary @ operator) is Operator
    assert type(involution @ operator) is Operator
    assert type(unitary @ observable) is Operator
    assert type(involution @ observable) is Operator
    assert type(involution @ unitary) is Unitary  # product of unitaries is unitary

    # Tensor product
    assert type(operator | operator) is type(operator)
    assert type(observable | observable) is type(observable)
    assert type(unitary | unitary) is Unitary
    assert type(involution | involution) is Involution
    assert type(operator | observable) is Operator
    assert type(operator | unitary) is Operator
    assert type(operator | involution) is Operator
    assert type(observable | unitary) is Operator
    assert type(observable | involution) is Observable
    assert type(unitary | involution) is Unitary
    # reverse order
    assert type(observable | operator) is Operator
    assert type(unitary | operator) is Operator
    assert type(involution | operator) is Operator
    assert type(unitary | observable) is Operator
    assert type(involution | observable) is Observable
    assert type(involution | unitary) is Unitary  # unitary but not Hermitian

    # Scalar multiplication
    assert type(0.5 * operator) is type(operator)
    assert type(0.5 * observable) is Observable
    assert type(0.5 * unitary) is Operator
    assert type(0.5 * involution) is Observable

    assert type(1j * operator) is type(operator)
    assert type(1j * observable) is Operator  # complex scalar → Operator (Hermitian structure lost)
    assert type(1j * unitary) is Operator  # scalar mul never preserves Unitary
    assert type(1j * involution) is Operator  # complex scalar → Operator

    assert type(jnp.exp(1j * jnp.pi / 4) * operator) is type(operator)
    assert type(jnp.exp(1j * jnp.pi / 4) * observable) is Operator  # complex → Operator (Hermitian structure lost)
    assert type(jnp.exp(1j * jnp.pi / 4) * unitary) is Operator  # scalar mul never preserves Unitary
    assert type(jnp.exp(1j * jnp.pi / 4) * involution) is Operator  # complex scalar → Operator


## Mixed-dimension composition tests (auto-promotion)


@pytest.mark.parametrize(
    "object_1, object_2, expected_object",
    [
        # Channel-channel compositions
        (Choi, Choi, Choi),
        (SuperOp, SuperOp, SuperOp),
        (PauliLiouville, PauliLiouville, PauliLiouville),
        (KrausMap, KrausMap, KrausMap),
        (Unitary, Unitary, Unitary),
        # Channel-state compositions
        (Unitary, StateVector, StateVector),
        (Choi, DensityMatrix, DensityMatrix),
        (SuperOp, DensityMatrix, DensityMatrix),
        (KrausMap, DensityMatrix, DensityMatrix),
        (PauliLiouville, DensityMatrix, DensityMatrix),
        # State-state compositions
        (StateVector, StateVector, complex),
        (DensityMatrix, DensityMatrix, DensityMatrix),
        (DensityMatrix, StateVector, StateVector),
        (StateVector, DensityMatrix, StateVector),
    ],
)
def test_mixed_dim_compositions(object_1, object_2, expected_object):
    """Test that composing qubit and qutrit objects auto-promotes the lower-dim operand.

    Only subsystems that differ in dimension are promoted; the number of
    subsystems must match.
    """
    k = jax.random.key(42)
    key_1, key_2 = jax.random.split(k)
    # Object 1 acts on qutrits, object 2 on qubits (same number of subsystems)
    num_qudits = 2
    dims_qutrit = ((3,) * num_qudits, (3,) * num_qudits)
    dims_qubit = ((2,) * num_qudits, (2,) * num_qudits)
    rank_qutrit = 3**num_qudits
    rank_qubit = 2**num_qudits

    random_object_1 = _generate_random_object(object_1, key_1, (), dims_qutrit, rank_qutrit)
    random_object_2 = _generate_random_object(object_2, key_2, (), dims_qubit, rank_qubit)

    result = random_object_1 @ random_object_2

    if expected_object is complex:
        assert isinstance(result, jax.Array)
    else:
        assert type(result) is expected_object
        # The promoted result should live in the larger (qutrit) space
        expected_dims = (3,) * num_qudits
        if isinstance(result, (StateVector, DensityMatrix)):
            assert result.dims == expected_dims
        else:
            assert result.dims == (expected_dims, expected_dims)


@pytest.mark.parametrize(
    "object_1, object_2, expected_object",
    [
        (Choi, Choi, Choi),
        (SuperOp, SuperOp, SuperOp),
        (PauliLiouville, PauliLiouville, PauliLiouville),
        (KrausMap, KrausMap, KrausMap),
        (Unitary, Unitary, Unitary),
        (Unitary, StateVector, StateVector),
        (SuperOp, DensityMatrix, DensityMatrix),
    ],
)
def test_mixed_dim_partial_promotion(object_1, object_2, expected_object):
    """Test that only subsystems that differ in dimension are promoted.

    Object 1 has dims (3, 2) while object 2 has dims (2, 3).  After
    promotion both should operate on (3, 3).
    """
    k = jax.random.key(99)
    key_1, key_2 = jax.random.split(k)

    dims_1 = ((3, 2), (3, 2))
    dims_2 = ((2, 3), (2, 3))
    rank_1 = 6
    rank_2 = 6

    random_object_1 = _generate_random_object(object_1, key_1, (), dims_1, rank_1)
    random_object_2 = _generate_random_object(object_2, key_2, (), dims_2, rank_2)

    result = random_object_1 @ random_object_2

    assert type(result) is expected_object
    expected_dims = (3, 3)
    if isinstance(result, (StateVector, DensityMatrix)):
        assert result.dims == expected_dims
    else:
        assert result.dims == (expected_dims, expected_dims)


# ======================================================================
# QuantumInstrument tests
# ======================================================================


# ======================================================================
# Basic tests of ideal measurement
# ======================================================================


class TestIdealMeasurementSingleQudit:
    """Test ideal projective measurement on single qudits."""

    @pytest.mark.parametrize("d", [2, 3, 4])
    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 9999])
    def test_basis_state_labeled_correctly(self, d, seed):
        """Qudit in |k> is labeled k with post-measurement state |k>, for each basis state."""
        qi = qx.gates.MEASURE(d)
        for state_idx in range(d):
            rho = basis_dm(state_idx, d)
            key = jax.random.key(seed + state_idx)
            rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
            rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
            assert int(outcome) == state_idx
            np.testing.assert_allclose(rho_out.matrix, rho.matrix, atol=1e-10)

    def test_qubit_plus_state_probabilities(self):
        """Qubit in |+> has P(0)=P(1)=0.5; post-measurement state matches label."""
        qi = qx.gates.MEASURE()
        rho = superposition_dm(2)
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        np.testing.assert_allclose(probs, jnp.array([0.5, 0.5]), atol=1e-10)

        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        for label in range(2):
            key = jax.random.key(label)
            rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
            expected = basis_dm(int(outcome), 2)
            np.testing.assert_allclose(rho_out.matrix, expected.matrix, atol=1e-10)

    @pytest.mark.parametrize("d", [2, 3, 4])
    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 9999])
    def test_state_vector_input(self, d, seed):
        """Apply instrument to a state vector input |0> for dimension d."""
        qi = qx.gates.MEASURE(d)
        psi = basis_sv(0, d)
        key = jax.random.key(seed)
        psi_out, outcome = qx.apply_instrument_to_state_vector(qi, psi, key)
        assert int(outcome) == 0
        assert isinstance(psi_out, StateVector)
        expected = basis_sv(0, d)
        np.testing.assert_allclose(psi_out.matrix, expected.matrix, atol=1e-10)

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_ideal_confusion_identity(self, d):
        """Ideal measurement: confusion matrix is d×d identity."""
        qi = qx.gates.MEASURE(d)
        np.testing.assert_allclose(qi.confusion_matrix, jnp.eye(d), atol=1e-10)

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_ideal_transition_identity(self, d):
        """Ideal measurement: transition matrix is d×d identity."""
        qi = qx.gates.MEASURE(d)
        np.testing.assert_allclose(qi.transition_matrix, jnp.eye(d), atol=1e-10)

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_ideal_fidelities(self, d):
        """Ideal measurement: all fidelities are 1.0."""
        qi = qx.gates.MEASURE(d)
        np.testing.assert_allclose(qx.classification_fidelity(qi), 1.0, atol=1e-10)
        np.testing.assert_allclose(qx.non_demolition_fidelity(qi), 1.0, atol=1e-10)
        np.testing.assert_allclose(qx.instrument_fidelity(qi), 1.0, atol=1e-10)


# ======================================================================
# 2-qudit perfect measurement
# ======================================================================


class TestIdealMeasurementMultiQudit:
    """Test ideal projective measurement on pairs of qudits."""

    @pytest.mark.parametrize("seed", [0, 42, 9999])
    @pytest.mark.parametrize(
        "i,j",
        list(product(range(3), range(3))),
        ids=[f"|{i}{j}>" for i, j in product(range(3), range(3))],
    )
    def test_two_qutrit_basis_states(self, i, j, seed):
        """Pair of qutrits in |ij> is labeled ij with correct post-measurement state."""
        qi = qx.gates.MEASURE(3) | qx.gates.MEASURE(3)
        rho = basis_dm_multi((i, j), (3, 3))
        key = jax.random.key(seed)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
        expected_outcome = i * 3 + j
        assert int(outcome) == expected_outcome
        np.testing.assert_allclose(rho_out.matrix, rho.matrix, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 42, 9999])
    @pytest.mark.parametrize(
        "i,j",
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        ids=[f"|{i}{j}>" for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]],
    )
    def test_two_qubit_basis_states(self, i, j, seed):
        """Pair of qubits in |ij> is labeled ij with correct post-measurement state."""
        qi = qx.gates.MEASURE() | qx.gates.MEASURE()
        rho = basis_dm_multi((i, j), (2, 2))
        key = jax.random.key(seed)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
        expected_outcome = i * 2 + j
        assert int(outcome) == expected_outcome
        np.testing.assert_allclose(rho_out.matrix, rho.matrix, atol=1e-10)

    def test_two_qubit_plus_state_probabilities(self):
        """Two qubits both in |+> produce uniform outcome probabilities."""
        qi = qx.gates.MEASURE() | qx.gates.MEASURE()
        vec = jnp.ones(4, dtype=complex) / 2.0
        rho = DensityMatrix.from_matrix(jnp.outer(vec, jnp.conj(vec)), (2, 2))

        _, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        np.testing.assert_allclose(probs, jnp.full(4, 0.25), atol=1e-10)

    def test_two_qubit_superposition_post_state(self):
        """Post-measurement state of |++> matches the label."""
        qi = qx.gates.MEASURE() | qx.gates.MEASURE()
        vec = jnp.ones(4, dtype=complex) / 2.0
        rho = DensityMatrix.from_matrix(jnp.outer(vec, jnp.conj(vec)), (2, 2))

        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        for seed in range(50):
            key = jax.random.key(seed)
            rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
            i, j = divmod(int(outcome), 2)
            expected = basis_dm_multi((i, j), (2, 2))
            np.testing.assert_allclose(rho_out.matrix, expected.matrix, atol=1e-10)

    def test_two_qubit_minus_state_probabilities(self):
        """Two qubits in |-> states: outcome probabilities are uniform."""
        minus = jnp.array([1.0, -1.0], dtype=complex) / jnp.sqrt(2)
        vec = jnp.kron(minus, minus)
        rho = DensityMatrix.from_matrix(jnp.outer(vec, jnp.conj(vec)), (2, 2))
        qi = qx.gates.MEASURE() | qx.gates.MEASURE()

        _, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        np.testing.assert_allclose(probs, jnp.full(4, 0.25), atol=1e-10)


# ======================================================================
# QuantumInstrument operator tests
# ======================================================================


class TestInstrumentOperators:
    """Test that @ (compose) and | (tensor) operators work on QuantumInstruments."""

    def test_compose_via_matmul(self):
        qi = qx.gates.MEASURE()
        composed = qi @ qi
        assert isinstance(composed, qx.QuantumInstrument)
        assert composed.num_outcomes == 2

    def test_tensor_via_or(self):
        qi = qx.gates.MEASURE()
        tensored = qi | qi
        assert isinstance(tensored, qx.QuantumInstrument)
        assert tensored.num_outcomes == 4
        assert tensored.dims == ((2, 2), (2, 2))


# ======================================================================
# Properties and validation tests
# ======================================================================


class TestProperties:
    """Test qx.QuantumInstrument properties and methods."""

    def test_repr(self):
        qi = qx.gates.MEASURE()
        r = repr(qi)
        assert "QuantumInstrument" in r
        assert "num_outcomes=2" in r

    def test_dims(self):
        qi = qx.gates.MEASURE()
        assert qi.dims == ((2,), (2,))
        assert qi.d == (2, 2)
        assert qi.d2 == (4, 4)

    def test_mixed_dims(self):
        qi = qx.gates.MEASURE() | qx.gates.MEASURE(3)
        assert qi.dims == ((2, 3), (2, 3))
        assert qi.d == (6, 6)

    def test_total_channel_cptp(self):
        qi = qx.gates.MEASURE()
        total = qi.total_channel()
        assert isinstance(total, SuperOp)
        assert qx.is_cptp(total)

    def test_outcome_superop(self):
        qi = qx.gates.MEASURE()
        c0, coeff0 = qi.outcome_superop(0)
        c1, coeff1 = qi.outcome_superop(1)
        assert isinstance(c0, SuperOp)
        assert c0.dims == ((2,), (2,))
        np.testing.assert_allclose(coeff0, 0.5, atol=1e-10)
        np.testing.assert_allclose(coeff1, 0.5, atol=1e-10)

    def test_pytree_roundtrip(self):
        qi = qx.gates.MEASURE()
        leaves, aux = qi.tree_flatten()
        qi2 = qx.QuantumInstrument.tree_unflatten(aux, leaves)
        assert qi2.num_outcomes == qi.num_outcomes
        assert qi2.dims == qi.dims
        assert qi2.measured_qudits == qi.measured_qudits
        np.testing.assert_allclose(qi2.data, qi.data)

    def test_from_superop_constructor(self):
        P0 = jnp.array([[1, 0], [0, 0]], dtype=complex)
        P1 = jnp.array([[0, 0], [0, 1]], dtype=complex)
        superop0 = SuperOp.from_matrix(jnp.einsum("ab,cd->acbd", jnp.conj(P0), P0).reshape(4, 4), ((2,), (2,)))
        superop1 = SuperOp.from_matrix(jnp.einsum("ab,cd->acbd", jnp.conj(P1), P1).reshape(4, 4), ((2,), (2,)))
        qi = qx.QuantumInstrument.from_superop([superop0, superop1], measured_qudits=(0,))
        assert qi.num_outcomes == 2
        assert qx.validate(qi)
        np.testing.assert_allclose(qi.confusion_matrix, jnp.eye(2), atol=1e-10)

    def test_from_superop_dims_mismatch_raises(self):
        s1 = SuperOp.from_matrix(jnp.eye(4, dtype=complex) / 2, ((2,), (2,)))
        s2 = SuperOp.from_matrix(jnp.eye(9, dtype=complex) / 3, ((3,), (3,)))
        with pytest.raises(ValueError, match="dims"):
            qx.QuantumInstrument.from_superop([s1, s2], measured_qudits=(0,))

    def test_validation_ideal(self):
        assert qx.validate(qx.gates.MEASURE())

    def test_validation_noisy(self):
        confusion = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        qi = qx.instrument_from_confusion_and_transition(confusion, jnp.eye(2), dims=(2,))
        assert qx.validate(qi)

    def test_validation_bad_confusion_column_sum(self):
        confusion = jnp.array([[0.9, 0.1], [0.2, 0.8]])
        with pytest.raises(ValueError, match="columns must sum to 1"):
            qx.instrument_from_confusion_and_transition(confusion, jnp.eye(2), dims=(2,))
