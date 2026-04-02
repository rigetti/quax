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
