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

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING, Any, Iterator, Self, Tuple, overload

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    import qutip
    from numpy.typing import NDArray


# ---------- base ----------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QuantumObject:
    """Base class for all quantum objects: states, operators, and superoperators.

    Provides the shared data representation (a JAX tensor plus qubit count) and
    common operations (negation, scalar multiplication, pytree support, etc.).
    Subclasses must implement the abstract properties ``dims``, ``num_ensemble_dims``,
    ``matrix``, and the class method ``from_matrix``.
    """

    data: Array
    """The underlying JAX tensor whose trailing axes encode the quantum degrees of
    freedom.  The number and interpretation of those axes varies by subclass."""

    num_qubits: int
    """The number of qubits (qudits) described by this object."""

    # ----- arithmetic -----

    def __neg__(self) -> Self:
        """Negate the quantum object."""
        return type(self)(-self.data, self.num_qubits)

    def __mul__(self, scalar: complex | Array) -> Self:
        """Scalar multiplication of the superoperator."""
        scalar_array = jnp.asarray(scalar)
        broadcast_dims = jnp.broadcast_shapes(scalar_array.shape, self.ensemble_size)
        broadcast_scalar = jnp.broadcast_to(scalar, broadcast_dims)
        tail_ndims = self.data.ndim - self.num_ensemble_dims
        broadcast_scalar = broadcast_scalar.reshape(broadcast_scalar.shape + (1,) * tail_ndims)
        padded_data = self.data.reshape((1,) * (len(broadcast_dims) - self.num_ensemble_dims) + self.data.shape)

        return type(self)(padded_data * broadcast_scalar, self.num_qubits)

    def __rmul__(self, scalar: complex | Array) -> Self:
        """Scalar multiplication of the superoperator."""
        return self * scalar

    # ----- display / comparison -----

    def __str__(self) -> str:
        if self.ensemble_size != ():
            return (
                f"{type(self).__name__}(dims={self.dims}, ensemble_size={self.ensemble_size}, shape={self.data.shape})"
            )
        return f"{type(self).__name__}(dims={self.dims}, shape={self.data.shape})"

    def __eq__(self, other: Any) -> bool:
        """Element-wise equality check (overridden by subclasses that use fidelity)."""
        if not isinstance(other, type(self)):
            return False
        if self.dims != other.dims:
            return False
        elif jnp.allclose(self.data, other.data):
            return True
        else:
            return False

    # ----- ensemble indexing -----

    def __getitem__(self, key: Any) -> Self:
        if self.num_ensemble_dims == 0:
            raise IndexError("This quantum object is not ensembled (no ensemble dimensions), so it cannot be indexed.")

        quantum_object = type(self)(data=self.data[key], num_qubits=self.num_qubits)
        # Ensure that we didn't slice the quantum dimensions
        qubit_dims = self.data.shape[self.num_ensemble_dims :]
        quantum_object_dims = quantum_object.data.shape[quantum_object.num_ensemble_dims :]
        if qubit_dims != quantum_object_dims:
            raise IndexError(
                f"Indexing resulted in an object with dimensions {quantum_object_dims} that do not match "
                f"the original dimensions {qubit_dims}. Check that your indexing is correct and does not "
                "remove any quantum dimensions."
            )
        return quantum_object

    # ----- JAX pytree -----

    def tree_flatten(self):
        return (self.data,), self.num_qubits

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (data,) = children
        return cls(data=data, num_qubits=aux_data)

    # ----- conjugation -----

    def conj(self):
        """Complex conjugate."""
        return type(self)(data=jnp.conjugate(self.data), num_qubits=self.num_qubits)

    # ----- abstract properties -----

    @property
    def dims(self):
        """The dimensions of the quantum object.  Structure varies by subclass."""
        raise NotImplementedError("Subclasses must implement dims property.")

    @property
    def num_ensemble_dims(self) -> int:
        """The number of leading ensemble (batch) dimensions."""
        raise NotImplementedError("Subclasses must implement num_ensemble_dims property.")

    @property
    def ensemble_size(self) -> Tuple[int, ...]:
        """Shape of the ensemble (batch) dimensions, or ``()`` for a single object."""
        return self.data.shape[: self.num_ensemble_dims]

    @property
    def matrix(self) -> Array:
        """The matrix representation of the quantum object."""
        raise NotImplementedError("Subclasses must implement matrix property.")

    @classmethod
    def from_matrix(cls, matrix: Array, dims) -> Self:
        """Construct from a matrix representation."""
        raise NotImplementedError("Subclasses must implement from_matrix class method.")


# ---------- states base ----------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class State(QuantumObject):
    """Base class for a quantum state."""

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the state."""
        raise NotImplementedError(f"Exponentiation not implemented for {type(self)}.")

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication of the state with another object."""
        raise NotImplementedError(f"Matrix multiplication not implemented between {type(self)} and {type(other)}.")

    def __or__(self, other: Any) -> Any:
        """Tensor product of the state with another object."""
        raise NotImplementedError(f"Tensor product not implemented between {type(self)} and {type(other)}.")

    @property
    def T(self):
        """Transpose of the state(s)."""
        raise NotImplementedError(f"Transpose not implemented for {type(self)}.")

    @property
    def h(self):
        """Hermitian conjugate of the state(s)."""
        raise NotImplementedError(f"Hermitian conjugate not implemented for {type(self)}.")

    @property
    def d(self) -> int:
        return reduce(mul, self.dims)

    @property
    def d2(self) -> int:
        return self.d**2


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Operator(QuantumObject):
    """Base class for a quantum operator."""

    def __mul__(self, scalar: complex | Array) -> Self:
        """Scalar multiplication of the superoperator."""
        scalar_array = jnp.asarray(scalar)
        broadcast_dims = jnp.broadcast_shapes(scalar_array.shape, self.ensemble_size)
        broadcast_scalar = jnp.broadcast_to(scalar, broadcast_dims)
        tail_ndims = self.data.ndim - self.num_ensemble_dims
        broadcast_scalar = broadcast_scalar.reshape(broadcast_scalar.shape + (1,) * tail_ndims)
        padded_data = self.data.reshape((1,) * (len(broadcast_dims) - self.num_ensemble_dims) + self.data.shape)

        return type(self)(padded_data * broadcast_scalar, self.num_qubits)

    def __rmul__(self, scalar: complex | Array) -> Self:
        """Scalar multiplication of the superoperator."""
        return self * scalar

    def __add__(self, other: Any) -> "Operator":
        """Add an operator with another operator. Returns an Operator.

        Subclasses override to return a more specific type where valid.
        """
        if isinstance(other, Operator) and self.num_qubits == other.num_qubits:
            return Operator(self.data + other.data, self.num_qubits)
        return NotImplemented

    def __radd__(self, other: Any) -> "Operator":
        """Right-hand addition of two operators."""
        if isinstance(other, Operator) and self.num_qubits == other.num_qubits:
            return Operator(other.data + self.data, self.num_qubits)
        return NotImplemented

    def __sub__(self, other: Any) -> "Operator":
        """Subtract an operator from another. Returns an Operator.

        Subclasses override to return a more specific type where valid.
        """
        if isinstance(other, Operator) and self.num_qubits == other.num_qubits:
            return Operator(self.data - other.data, self.num_qubits)
        return NotImplemented

    def __rsub__(self, other: Any) -> "Operator":
        """Right-hand subtraction of two operators."""
        if isinstance(other, Operator) and self.num_qubits == other.num_qubits:
            return Operator(other.data - self.data, self.num_qubits)
        return NotImplemented

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the operator"""
        # By default, only support integer exponents
        # matrix_power is not _correct_ for all operators
        if jnp.allclose(int(jnp.abs(exponent)), exponent):
            new_data = jnp.linalg.matrix_power(self.matrix, int(exponent))
            return type(self).from_matrix(new_data, self.dims)
        else:
            raise TypeError(f"Exponent must be an integer, but got {type(exponent)}.")

    def _to_qobj(self) -> "qutip.Qobj | NDArray":
        """Convert to a QuTiP Qobj for interoperability testing."""
        import numpy as np
        import qutip as qt

        matrix = self.matrix

        if self.ensemble_size == ():
            return qt.Qobj(
                np.array(matrix),
                dims=[[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]],
            )

        flat_shape = (-1,) + matrix.shape[-2:]
        flat_kraus = np.asarray(matrix).reshape(flat_shape)
        dims = [[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]]
        qobjs = np.asarray(
            [
                qt.Qobj(
                    np.asarray(kraus),
                    dims=dims,
                )
                for kraus in flat_kraus
            ],
            dtype=object,
        )
        return qobjs.reshape(self.ensemble_size)

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication of a quantum object with the unitary."""
        match other:
            case Unitary() | Operator():
                # U @ U -> Unitary
                # O1 @ O2 -> Operator
                from ._compose import compose_operator

                return compose_operator(self, other)
            case _:
                return NotImplemented

    def __or__(self, other: Any) -> Any:
        """Tensor product of the operator with another operator."""
        match other:
            case Unitary() | Operator():
                # O ⊗ U -> Operator
                # O1 ⊗ O2 -> Operator
                from ._tensor import tensor_operator

                return tensor_operator(self, other)
            case _:
                return NotImplemented

    def __ror__(self, other: Any) -> Any:
        """Tensor product with the operator on the right (reflected)."""
        match other:
            case Unitary() | Operator():
                from ._tensor import tensor_operator

                return tensor_operator(other, self)
            case _:
                return NotImplemented

    @property
    def dims(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The (output, input) dimensions of each qudit, inferred from data shape."""
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        return (qudit_shape[: self.num_qubits], qudit_shape[self.num_qubits :])

    @property
    def T(self):
        """Transpose of the operator(s)"""
        # use the swapaxes function to swap the last two axes
        matrix_t = jnp.swapaxes(self.matrix, -1, -2)
        return type(self).from_matrix(matrix_t, (self.dims[1], self.dims[0]))

    @property
    def h(self):
        """Hermitian conjugate of the operator(s)"""
        matrix_h = jnp.conjugate(jnp.swapaxes(self.matrix, -1, -2))
        return type(self).from_matrix(matrix_h, (self.dims[1], self.dims[0]))

    @property
    def d(self) -> Tuple[int, int]:
        return tuple(reduce(mul, dim) for dim in self.dims)

    @property
    def d2(self) -> Tuple[int, int]:
        return self.d[0] ** 2, self.d[1] ** 2

    @property
    def num_ensemble_dims(self) -> int:
        """The number of leading ensemble dimensions, derived from data shape and num_qubits."""
        return self.data.ndim - 2 * self.num_qubits

    @property
    def matrix(self) -> Array:
        """Returns the matrix representation (*ensemble, d_out, d_in) of the operator."""
        ensemble_shape = self.data.shape[: self.num_ensemble_dims]
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 2
        d_out = reduce(mul, qudit_shape[:n_qudits], 1)
        d_in = reduce(mul, qudit_shape[n_qudits:], 1)
        return self.data.reshape(ensemble_shape + (d_out, d_in))

    @classmethod
    def from_matrix(cls, matrix: Array, dims: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Self:
        """Construct from matrix representation.

        :param matrix: Array with shape (*ensemble, d_out, d_in)
        :param dims: Tuple of (dims_out, dims_in) where each is a tuple of qudit dimensions
        :return: Operator with tensor data
        """
        num_qubits = len(dims[0])
        ensemble_shape = matrix.shape[:-2]
        tensor = matrix.reshape(ensemble_shape + dims[0] + dims[1])
        return cls(data=tensor, num_qubits=num_qubits)

    def __iter__(self) -> Iterator[Self]:
        if self.ensemble_size == ():
            raise TypeError(
                "This Operator is not ensembled (no ensemble dimensions), so it cannot be iterated. "
                "If you intended an ensemble, make data shape (N, d0_out, d1_out, ..., d0_in, d1_in, ...) (or higher-rank)."
            )
        # iterate over axis 0 only; remaining ensemble axes (if any) remain on each item
        for i in range(self.data.shape[0]):
            yield type(self)(data=self.data[i], num_qubits=self.num_qubits)


# This class is basically for typing purposes
# Some methods work on any sort of Superoperator
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SuperOperator(QuantumObject):
    """Base class for a quantum superoperator.

    SuperOperators have a 4-group tensor structure:
    (*ensemble, d0_out_bra, d1_out_bra, ..., d0_out_ket, d1_out_ket, ...,
            d0_in_bra, d1_in_bra, ..., d0_in_ket, d1_in_ket, ...)

    SuperOperators represent CPTP maps; ``+`` and ``-`` are intentionally not
    defined because the sum/difference of CPTP maps is not generally CPTP.
    """

    @property
    def num_ensemble_dims(self) -> int:
        """The number of leading ensemble dimensions, derived from data shape and num_qubits."""
        return self.data.ndim - 4 * self.num_qubits

    @property
    def matrix(self) -> Array:
        """Returns the matrix representation (*ensemble, d_out^2, d_in^2) of the superoperator."""
        ensemble_shape = self.data.shape[: self.num_ensemble_dims]
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        # 4 groups of dimensions: out_bra, out_ket, in_bra, in_ket
        n_qudits = len(qudit_shape) // 4
        d_out_bra = reduce(mul, qudit_shape[:n_qudits], 1)
        d_out_ket = reduce(mul, qudit_shape[n_qudits : 2 * n_qudits], 1)
        d_in_bra = reduce(mul, qudit_shape[2 * n_qudits : 3 * n_qudits], 1)
        d_in_ket = reduce(mul, qudit_shape[3 * n_qudits :], 1)
        d_out = d_out_bra * d_out_ket
        d_in = d_in_bra * d_in_ket
        return self.data.reshape(ensemble_shape + (d_out, d_in))

    @classmethod
    def from_matrix(cls, matrix: Array, dims: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Self:
        """Construct from matrix representation.

        :param matrix: Array with shape (*ensemble, d_out^2, d_in^2)
        :param dims: Tuple of (dims_out, dims_in) where each is a tuple of qudit dimensions
        :return: SuperOperator with tensor data
        """
        num_qubits = len(dims[0])
        ensemble_shape = matrix.shape[:-2]
        # Tensor shape is: out_bra_dims + out_ket_dims + in_bra_dims + in_ket_dims
        tensor_shape = dims[0] + dims[0] + dims[1] + dims[1]
        tensor = matrix.reshape(ensemble_shape + tensor_shape)
        return cls(data=tensor, num_qubits=num_qubits)

    @property
    def dims(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The (output, input) qudit dimensions, inferred from data shape.

        Returns the qudit dimensions, not the doubled superoperator dimensions.
        """
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 4
        dims_out = qudit_shape[:n_qudits]
        dims_in = qudit_shape[2 * n_qudits : 3 * n_qudits]
        return (dims_out, dims_in)

    @property
    def d(self) -> Tuple[int, int]:
        return tuple(reduce(mul, dim) for dim in self.dims)

    @property
    def d2(self) -> Tuple[int, int]:
        return self.d[0] ** 2, self.d[1] ** 2

    @property
    def T(self):
        """Transpose of the superoperator."""
        raise NotImplementedError(f"Transpose not implemented for {type(self)}.")

    @property
    def h(self):
        """Hermitian conjugate of the superoperator."""
        raise NotImplementedError(f"Hermitian conjugate not implemented for {type(self)}.")

    def _to_qobj(self) -> "qutip.Qobj | NDArray":
        """Convert to a QuTiP Qobj for interoperability testing."""
        raise NotImplementedError("_to_qobj not implemented for the base SuperOperator class.")

    def __iter__(self) -> Iterator[Self]:
        if self.ensemble_size == ():
            raise TypeError(
                "This SuperOperator is not ensembled (no ensemble dimensions), so it cannot be iterated. "
                "If you intended an ensemble, ensure the data has leading batch dimensions."
            )
        # iterate over axis 0 only; remaining ensemble axes (if any) remain on each item
        for i in range(self.data.shape[0]):
            yield type(self)(data=self.data[i], num_qubits=self.num_qubits)


# ---------- states ----------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StateVector(State):
    """State vector |psi>, shape (*ensemble, d0, d1, ...) in tensor form or (*ensemble, d) in matrix form."""

    @property
    def num_ensemble_dims(self) -> int:
        """The number of leading ensemble dimensions, derived from data shape and num_qubits."""
        return self.data.ndim - self.num_qubits

    @property
    def dims(self) -> Tuple[int, ...]:
        """The dimensions of each qudit, inferred from data shape."""
        return self.data.shape[-self.num_qubits :] if self.num_qubits > 0 else ()

    @property
    def ensemble_size(self) -> Tuple[int, ...]:
        """Returns the size of the ensemble if the state represents an ensemble of states."""
        return self.data.shape[: self.num_ensemble_dims]

    @property
    def matrix(self) -> Array:
        """Returns the vector representation (*ensemble, d) of the state."""
        ensemble_shape = self.data.shape[: self.num_ensemble_dims]
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        d = reduce(mul, qudit_shape, 1)
        return self.data.reshape(ensemble_shape + (d,))

    @classmethod
    def from_matrix(cls, matrix: Array, dims: Tuple[int, ...]) -> "StateVector":
        """Construct from vector representation.

        :param matrix: Array with shape (*ensemble, d) where d = prod(dims)
        :param dims: Tuple of qudit dimensions (d0, d1, ...)
        :return: StateVector with tensor data
        """
        num_qubits = len(dims)
        ensemble_shape = matrix.shape[:-1] if num_qubits > 0 else matrix.shape
        tensor = matrix.reshape(ensemble_shape + dims)
        return cls(data=tensor, num_qubits=num_qubits)

    @property
    def T(self):
        """Transpose of the operator(s)"""
        # The state vector is a 1D array, so transpose does nothing
        return self

    @property
    def h(self):
        """Hermitian conjugate of the operator(s)"""
        # The state vector is a 1D array, so hermitian conjugate is just complex conjugate
        return self.conj()

    def __matmul__(self, other):
        """Left multiply the state by another."""
        match other:
            case StateVector():  # <𝜓|𝜙> -> p
                return jnp.einsum("...a,...a->...", self.matrix.conj(), other.matrix)
            case DensityMatrix():  #  <𝜓|𝜌 -> <𝜙|
                result = jnp.einsum("...b,...ba->...a", self.matrix.conj(), other.matrix)
                return StateVector.from_matrix(result, self.dims)
            case _:
                return NotImplemented

    def __or__(self, other: Any) -> Any:
        """Tensor product of the state vector with another state."""
        match other:
            case StateVector():
                # |ψ1⟩ ⊗ |ψ2⟩ -> StateVector
                from ._tensor import tensor_state_vector

                return tensor_state_vector(self, other)
            case DensityMatrix():
                # |ψ⟩ ⊗ ρ -> DensityMatrix (promote state vector)
                from ._promotion import promote_state_vector_to_density_matrix
                from ._tensor import tensor_density_matrix

                return tensor_density_matrix(promote_state_vector_to_density_matrix(self), other)
            case Unitary() | SuperOp() | Choi() | PauliLiouville() | KrausMap():
                # State | Operator -> NotImplemented
                return NotImplemented
            case _:
                return NotImplemented

    def __eq__(self, other: Any) -> bool:
        """Check equality using quantum state fidelity."""
        match other:
            case StateVector():
                # Compare two state vectors using fidelity
                from ._distance_metrics import fidelity

                return bool(jnp.allclose(fidelity(self, other), 1.0))
            case DensityMatrix():
                # Promote self to density matrix and compare
                from ._distance_metrics import fidelity
                from ._promotion import promote_state_vector_to_density_matrix

                return bool(jnp.allclose(fidelity(promote_state_vector_to_density_matrix(self), other), 1.0))
            case Unitary() | SuperOp() | Choi() | PauliLiouville() | KrausMap():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented

    def _to_qobj(self) -> "qutip.Qobj | NDArray":
        """Convert to a QuTiP Qobj (or array of Qobjs for ensembles) for interoperability testing."""
        import numpy as np
        import qutip as qt

        dims = [list(self.dims), [1] * len(self.dims)]
        matrix = self.matrix

        if self.ensemble_size == ():
            # Scalar case - return single Qobj
            return qt.Qobj(np.asarray(matrix), dims=dims)

        flat_shape = (-1,) + matrix.shape[-1:]
        flat_vectors = np.asarray(matrix).reshape(flat_shape)
        qobjs = np.asarray(
            [
                qt.Qobj(
                    np.asarray(vec),
                    dims=dims,
                )
                for vec in flat_vectors
            ],
            dtype=object,
        )
        return qobjs.reshape(self.ensemble_size)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DensityMatrix(State):
    """Density matrix ρ, shape (*ensemble, d0_out, d1_out, ..., d0_in, d1_in, ...) in tensor form
    or (*ensemble, d, d) in matrix form."""

    @property
    def num_ensemble_dims(self) -> int:
        """The number of leading ensemble dimensions, derived from data shape and num_qubits."""
        return self.data.ndim - 2 * self.num_qubits

    @property
    def dims(self) -> Tuple[int, ...]:
        """The dimensions of each qudit, inferred from data shape.

        For DensityMatrix, dims returns just the qudit dimensions (same for in/out).
        """
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 2
        return qudit_shape[:n_qudits]

    @property
    def ensemble_size(self) -> Tuple[int, ...]:
        """Returns the size of the ensemble if the state represents an ensemble of states."""
        return self.data.shape[: self.num_ensemble_dims]

    @property
    def matrix(self) -> Array:
        """Returns the matrix representation (*ensemble, d, d) of the density matrix."""
        ensemble_shape = self.data.shape[: self.num_ensemble_dims]
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 2
        d_out = reduce(mul, qudit_shape[:n_qudits], 1)
        d_in = reduce(mul, qudit_shape[n_qudits:], 1)
        return self.data.reshape(ensemble_shape + (d_out, d_in))

    @classmethod
    def from_matrix(cls, matrix: Array, dims: Tuple[int, ...]) -> "DensityMatrix":
        """Construct from matrix representation.

        :param matrix: Array with shape (*ensemble, d, d) where d = prod(dims)
        :param dims: Tuple of qudit dimensions (d0, d1, ...)
        :return: DensityMatrix with tensor data
        """
        # For density matrices, dims_out = dims_in = dims
        num_qubits = len(dims)
        ensemble_shape = matrix.shape[:-2]
        tensor = matrix.reshape(ensemble_shape + dims + dims)
        return cls(data=tensor, num_qubits=num_qubits)

    @property
    def T(self):
        """Transpose of the operator(s)"""
        matrix_t = jnp.swapaxes(self.matrix, -1, -2)
        return DensityMatrix.from_matrix(matrix_t, self.dims)

    @property
    def h(self):
        """Hermitian conjugate of the operator(s)"""
        matrix_h = jnp.conjugate(jnp.swapaxes(self.matrix, -1, -2))
        return DensityMatrix.from_matrix(matrix_h, self.dims)

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the density matrix using eigendecomposition (ensemble-compatible)."""
        from ._power import density_matrix_power

        return density_matrix_power(self, exponent)

    def __matmul__(self, other):
        """Left multiply the density matrix by another object."""
        match other:
            case StateVector():  # 𝜌|𝜓> -> |𝜙>
                result = jnp.einsum("...ab,...b->...a", self.matrix, other.matrix)
                return StateVector.from_matrix(result, other.dims)
            case DensityMatrix():  # 𝜌𝜎 -> 𝜏
                result = jnp.einsum("...ab,...bc->...ac", self.matrix, other.matrix)
                return DensityMatrix.from_matrix(result, self.dims)
            case _:
                return NotImplemented

    def __or__(self, other: Any) -> Any:
        """Tensor product of the density matrix with another state."""
        match other:
            case StateVector():
                # ρ ⊗ |ψ⟩ -> DensityMatrix (promote state vector)
                from ._promotion import promote_state_vector_to_density_matrix
                from ._tensor import tensor_density_matrix

                return tensor_density_matrix(self, promote_state_vector_to_density_matrix(other))
            case DensityMatrix():
                # ρ1 ⊗ ρ2 -> DensityMatrix
                from ._tensor import tensor_density_matrix

                return tensor_density_matrix(self, other)
            case Unitary() | SuperOp() | Choi() | PauliLiouville() | KrausMap():
                # State | Operator -> NotImplemented
                return NotImplemented
            case _:
                return NotImplemented

    def __eq__(self, other: Any) -> bool:
        """Check equality using quantum state fidelity."""
        match other:
            case StateVector():
                # Promote other to density matrix and compare
                from ._distance_metrics import fidelity
                from ._promotion import promote_state_vector_to_density_matrix

                return bool(jnp.allclose(fidelity(self, promote_state_vector_to_density_matrix(other)), 1.0))
            case DensityMatrix():
                # Compare two density matrices using fidelity
                from ._distance_metrics import fidelity

                return bool(jnp.allclose(fidelity(self, other), 1.0))
            case Unitary() | SuperOp() | Choi() | PauliLiouville() | KrausMap():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented

    def _to_qobj(self) -> "qutip.Qobj | NDArray":
        """Convert to a QuTiP Qobj (or array of Qobjs for ensembles) for interoperability testing."""
        import numpy as np
        import qutip as qt

        # Density matrices are operators: dims = [input_dims, output_dims] = [dims, dims]
        dims = [list(self.dims), list(self.dims)]  # operator on the same space :contentReference[oaicite:1]{index=1}
        matrix = self.matrix

        if self.ensemble_size == ():
            # Scalar case - return single density-matrix Qobj
            return qt.Qobj(np.asarray(matrix), dims=dims)

        # Ensemble case - matrix has shape ensemble_size + (dim, dim)
        flat_shape = (-1,) + matrix.shape[-2:]
        flat_rhos = np.asarray(matrix).reshape(flat_shape)

        qobjs = np.asarray(
            [qt.Qobj(rho, dims=dims) for rho in flat_rhos],
            dtype=object,
        )
        return qobjs.reshape(self.ensemble_size)


# ---------- operators ----------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Unitary(Operator):
    """Unitary operator U, shape (*ensemble, d0_out, d1_out, ..., d0_in, d1_in, ...) in tensor form
    or (*ensemble, d, d) in matrix form."""

    @property
    def dims(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The (output, input) dimensions of each qudit, inferred from data shape."""
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 2
        return (qudit_shape[:n_qudits], qudit_shape[n_qudits:])

    def _to_qobj(self) -> "qutip.Qobj | NDArray":
        """Convert to a QuTiP Qobj for interoperability testing."""
        import numpy as np
        import qutip as qt

        dims = [list(self.dims[0]), list(self.dims[1])]
        matrix = self.matrix

        if self.ensemble_size == ():
            return qt.Qobj(
                np.array(matrix),
                dims=dims,
            )

        flat_shape = (-1,) + matrix.shape[-2:]
        flat_unitary = np.asarray(matrix).reshape(flat_shape)
        qobjs = np.asarray(
            [
                qt.Qobj(
                    np.asarray(unitary),
                    dims=dims,
                )
                for unitary in flat_unitary
            ],
            dtype=object,
        )
        return qobjs.reshape(self.ensemble_size)

    def __mul__(self, scalar: complex | Array) -> "Unitary | Operator":
        """Scalar multiplication of the unitary.

        - |scalar| = 1: result is unitary → returns ``Unitary``.
        - Otherwise: returns ``Operator``.
        """
        scalar_array = jnp.asarray(scalar)
        broadcast_dims = jnp.broadcast_shapes(scalar_array.shape, self.ensemble_size)
        broadcast_scalar = jnp.broadcast_to(scalar, broadcast_dims)
        tail_ndims = self.data.ndim - self.num_ensemble_dims
        broadcast_scalar = broadcast_scalar.reshape(broadcast_scalar.shape + (1,) * tail_ndims)
        padded_data = self.data.reshape((1,) * (len(broadcast_dims) - self.num_ensemble_dims) + self.data.shape)
        new_data = padded_data * broadcast_scalar

        # For Python scalars or 0-dim JAX arrays, check unit modulus precisely
        if isinstance(scalar, (int, float, complex)):
            if abs(abs(scalar) - 1.0) < 1e-10:
                return Unitary(new_data, self.num_qubits)
        elif scalar_array.ndim == 0:
            if abs(abs(complex(scalar_array)) - 1.0) < 1e-10:
                return Unitary(new_data, self.num_qubits)
        return Operator(new_data, self.num_qubits)

    def __rmul__(self, scalar: complex | Array) -> "Unitary | Operator":
        """Scalar multiplication of the unitary."""
        return self * scalar

    def __pow__(self, exponent: float) -> "Unitary":
        """Exponentiation of the unitary using eigendecomposition (ensemble-compatible)."""
        from ._power import power_unitary

        return power_unitary(self, exponent)

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication of a quantum object with the unitary."""
        match other:
            case Unitary():
                # U @ U -> Unitary
                from ._compose import compose_unitary

                return compose_unitary(self, other)
            case Choi():
                # J @ U -> Choi (promotion)
                from ._compose import compose_choi
                from ._superoperator_transformations import unitary_to_choi

                return compose_choi(other, unitary_to_choi(self))
            case PauliLiouville():
                # P @ U -> PauliLiouville (promotion)
                from ._compose import compose_pauli_liouville
                from ._superoperator_transformations import unitary_to_pauli_liouville

                return compose_pauli_liouville(other, unitary_to_pauli_liouville(self))
            case SuperOp():
                # S @ U -> SuperOp (promotion)
                from ._compose import compose_superop
                from ._superoperator_transformations import unitary_to_superop

                return compose_superop(other, unitary_to_superop(self))
            case KrausMap():
                # K @ U -> KrausMap (promotion)
                from ._compose import compose_kraus_map
                from ._superoperator_transformations import unitary_to_kraus

                return compose_kraus_map(other, unitary_to_kraus(self))
            case StateVector():
                # <psi|U = <phi| -> StateVector (apply unitary to state vector)
                from ._apply import apply_unitary_to_state_vector

                return apply_unitary_to_state_vector(self, other)
            case DensityMatrix():
                # ρU is an Operator product
                from ._apply import apply_superop_to_density_matrix
                from ._superoperator_transformations import unitary_to_superop

                return apply_superop_to_density_matrix(unitary_to_superop(self), other)
            case Operator():
                # U @ O -> Operator (catches Observable, Involution, plain Operator)
                from ._compose import compose_operator

                return compose_operator(self, other)
            case _:
                return NotImplemented

    def __or__(self, other: Any) -> Any:
        """Tensor product of the unitary with another operator."""
        match other:
            case Unitary():
                # U ⊗ V -> Unitary (tensor product)
                from ._tensor import tensor_unitary

                return tensor_unitary(self, other)
            case SuperOp():
                # U ⊗ S -> SuperOp (promote unitary)
                from ._superoperator_transformations import unitary_to_superop
                from ._tensor import tensor_superop

                return tensor_superop(unitary_to_superop(self), other)
            case Choi():
                # U ⊗ J -> Choi (promote unitary)
                from ._superoperator_transformations import unitary_to_choi
                from ._tensor import tensor_choi

                return tensor_choi(unitary_to_choi(self), other)
            case PauliLiouville():
                # U ⊗ P -> PauliLiouville (promote unitary)
                from ._superoperator_transformations import unitary_to_pauli_liouville
                from ._tensor import tensor_pauli_liouville

                return tensor_pauli_liouville(unitary_to_pauli_liouville(self), other)
            case KrausMap():
                # U ⊗ K -> KrausMap (promote unitary)
                from ._superoperator_transformations import unitary_to_kraus
                from ._tensor import tensor_kraus

                return tensor_kraus(unitary_to_kraus(self), other)
            case StateVector() | DensityMatrix():
                # Operator | State -> NotImplemented
                return NotImplemented
            case _:
                return NotImplemented

    def __eq__(self, other: Any) -> bool:
        """Check equality using unitary entanglement fidelity."""
        match other:
            case Unitary():
                # Compare two unitaries using entanglement fidelity
                from ._distance_metrics import unitary_entanglement_fidelity

                return bool(jnp.allclose(unitary_entanglement_fidelity(self, other), 1.0))
            case SuperOp() | Choi() | PauliLiouville() | KrausMap():
                # Promote self to superoperator and compare using process fidelity
                from ._distance_metrics import process_fidelity
                from ._superoperator_transformations import unitary_to_superop

                return bool(jnp.allclose(process_fidelity(unitary_to_superop(self), other), 1.0))
            case StateVector() | DensityMatrix():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented


# ---------- observables ----------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Observable(Operator):
    """
    Hermitian operator A = A†, used to represent a quantum observable.

    As a Hermitian operator it has real eigenvalues and an orthonormal eigenbasis.
    Observables are closed under addition, subtraction, and *real* scalar multiplication.
    Complex scalar multiplication and composition with non-Hermitian operators can break
    Hermiticity, so those operations return the more general ``Operator`` type.

    Tensor shape: (*ensemble, d0_out, d1_out, ..., d0_in, d1_in, ...)
    Matrix shape: (*ensemble, d, d)
    """

    @property
    def dims(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The (output, input) dimensions of each qudit, inferred from data shape.

        For an Observable the output and input spaces are identical (it is a square matrix
        on one Hilbert space).
        """
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 2
        space_dims = qudit_shape[:n_qudits]
        return (space_dims, space_dims)

    @property
    def h(self) -> "Observable":
        """Hermitian conjugate of the observable — returns ``self`` because A† = A."""
        return self

    def __neg__(self) -> "Observable":
        """Negate the observable; -A is still Hermitian (and still Involution if self is an Involution)."""
        return type(self)(-self.data, self.num_qubits)

    @overload
    def __mul__(self, scalar: float) -> "Observable": ...

    @overload
    def __mul__(self, scalar: complex) -> "Operator": ...

    @overload
    def __mul__(self, scalar: Array) -> "Observable | Operator": ...

    def __mul__(self, scalar: float | complex | Array) -> "Observable | Operator":
        """Scalar multiplication with ensemble broadcasting.

        - Real scalar: result is Hermitian → returns ``Observable``.
        - Complex scalar: result is not generally Hermitian → returns ``Operator``.
        """
        scalar_array = jnp.asarray(scalar)
        broadcast_dims = jnp.broadcast_shapes(scalar_array.shape, self.ensemble_size)
        broadcast_scalar = jnp.broadcast_to(scalar, broadcast_dims)
        tail_ndims = self.data.ndim - self.num_ensemble_dims
        broadcast_scalar = broadcast_scalar.reshape(broadcast_scalar.shape + (1,) * tail_ndims)
        padded_data = self.data.reshape((1,) * (len(broadcast_dims) - self.num_ensemble_dims) + self.data.shape)
        new_data = padded_data * broadcast_scalar

        if isinstance(scalar, (int, float)):
            return Observable(new_data, self.num_qubits)
        elif isinstance(scalar, complex):
            if abs(scalar.imag) < 1e-10:
                return Observable(new_data, self.num_qubits)
            return Operator(new_data, self.num_qubits)
        elif scalar_array.ndim == 0:
            # 0-dim JAX array — extract Python value for precise narrowing
            if abs(complex(scalar_array).imag) < 1e-10:
                return Observable(new_data, self.num_qubits)
            return Operator(new_data, self.num_qubits)
        else:
            # JAX array with shape — use dtype for JIT-safe type narrowing
            if jnp.issubdtype(scalar_array.dtype, jnp.floating):
                return Observable(new_data, self.num_qubits)
            return Operator(new_data, self.num_qubits)

    @overload
    def __rmul__(self, scalar: float) -> "Observable": ...

    @overload
    def __rmul__(self, scalar: complex) -> "Operator": ...

    @overload
    def __rmul__(self, scalar: Array) -> "Observable | Operator": ...

    def __rmul__(self, scalar: float | complex | Array) -> "Observable | Operator":
        """Scalar multiplication (scalar on the left)."""
        return self * scalar

    def __pow__(self, exponent: float) -> "Observable":
        """Power of a Hermitian matrix is Hermitian → returns ``Observable``.

        Supports both integer and fractional exponents via eigendecomposition.
        """
        from ._power import power_observable

        return power_observable(self, exponent)

    def __add__(self, other: Any) -> "Observable | Operator":
        """Addition of observables.

        - ``Observable + Observable`` → ``Observable`` (sum of Hermitian matrices is Hermitian).
        - ``Observable + Operator`` → ``Operator`` (not guaranteed Hermitian).
        """
        match other:
            case Observable():
                if self.dims != other.dims:
                    raise ValueError(f"Cannot add observables with different dims: {self.dims} vs {other.dims}")
                return Observable.from_matrix(self.matrix + other.matrix, self.dims)
            case Operator():
                if self.dims != other.dims:
                    raise ValueError(f"Cannot add operators with different dims: {self.dims} vs {other.dims}")
                return Operator.from_matrix(self.matrix + other.matrix, self.dims)
            case _:
                return NotImplemented

    def __radd__(self, other: Any) -> "Observable | Operator":
        """Right-hand addition."""
        match other:
            case Observable() | Operator():
                return other.__add__(self)
            case _:
                return NotImplemented

    def __sub__(self, other: Any) -> "Observable | Operator":
        """Subtraction of observables.

        - ``Observable - Observable`` → ``Observable`` (difference of Hermitian matrices is Hermitian).
        - ``Observable - Operator`` → ``Operator`` (not guaranteed Hermitian).
        """
        match other:
            case Observable():
                if self.dims != other.dims:
                    raise ValueError(f"Cannot subtract observables with different dims: {self.dims} vs {other.dims}")
                return Observable.from_matrix(self.matrix - other.matrix, self.dims)
            case Operator():
                if self.dims != other.dims:
                    raise ValueError(f"Cannot subtract operators with different dims: {self.dims} vs {other.dims}")
                return Operator.from_matrix(self.matrix - other.matrix, self.dims)
            case _:
                return NotImplemented

    def __rsub__(self, other: Any) -> "Observable | Operator":
        """Right-hand subtraction."""
        match other:
            case Observable():
                return other.__sub__(self)
            case Operator():
                return other.__sub__(self)
            case _:
                return NotImplemented

    def __or__(self, other: Any) -> Any:
        """Tensor product of the observable with another operator.

        - ``Observable ⊗ Observable`` → ``Observable``
        - ``Observable ⊗ Operator / Unitary`` → ``Operator``
        """
        match other:
            case Observable():
                # O1 ⊗ O2 -> Observable (tensor product of Hermitian matrices is Hermitian)
                from ._tensor import tensor_observable

                return tensor_observable(self, other)
            case Unitary():
                # Obs ⊗ U -> Operator
                from ._tensor import tensor_operator

                return tensor_operator(self, other)
            case Operator():
                # Obs ⊗ Op -> Operator
                from ._tensor import tensor_operator

                return tensor_operator(self, other)
            case _:
                return NotImplemented


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Involution(Observable, Unitary):
    """An operator that is simultaneously Hermitian (A = A†) and Unitary (A A† = I).

    Equivalently characterised by A² = I with eigenvalues in {±1}.  Canonical examples
    are the Pauli matrices X, Y, Z and the Hadamard gate H.

    ``Involution`` inherits from ``Observable`` rather than ``Unitary`` because the
    Hermitian property is the structurally important one; the unitary property is an
    additional constraint.

    Algebra rules
    -------------
    - Real scalar × Involution: ±1 → ``Involution``; |s|=1 non-real → ``Unitary``;
      real |s| ≠ 1 → ``Observable``; otherwise → ``Operator``.
    - Involution + Involution → ``Observable`` (sum not generally unitary).
    - Involution ⊗ Involution → ``Involution``.
    - Involution ⊗ Observable → ``Observable``.
    - Involution ⊗ Operator → ``Operator``.
    """

    @overload
    def __mul__(self, scalar: float) -> "Observable": ...

    @overload
    def __mul__(self, scalar: complex) -> "Operator": ...

    @overload
    def __mul__(self, scalar: Array) -> "Involution | Unitary | Observable | Operator": ...

    def __mul__(self, scalar: float | complex | Array) -> "Involution | Unitary | Observable | Operator":
        """Scalar multiplication with ensemble broadcasting.

        - ±1: stays ``Involution``.
        - |scalar| = 1, complex: unitary but not Hermitian → ``Unitary``.
        - Real, |scalar| ≠ 1: Hermitian but not unitary → ``Observable``.
        - Otherwise: general ``Operator``.
        """
        scalar_array = jnp.asarray(scalar)
        broadcast_dims = jnp.broadcast_shapes(scalar_array.shape, self.ensemble_size)
        broadcast_scalar = jnp.broadcast_to(scalar, broadcast_dims)
        tail_ndims = self.data.ndim - self.num_ensemble_dims
        broadcast_scalar = broadcast_scalar.reshape(broadcast_scalar.shape + (1,) * tail_ndims)
        padded_data = self.data.reshape((1,) * (len(broadcast_dims) - self.num_ensemble_dims) + self.data.shape)
        new_data = padded_data * broadcast_scalar

        if isinstance(scalar, (int, float)):
            # Python real scalar — precise narrowing
            if abs(abs(scalar) - 1.0) < 1e-10:
                return Involution(new_data, self.num_qubits)
            return Observable(new_data, self.num_qubits)
        elif isinstance(scalar, complex):
            # Python complex — precise narrowing
            is_real = abs(scalar.imag) < 1e-10
            is_unit = abs(abs(scalar) - 1.0) < 1e-10
            if is_real and is_unit:
                return Involution(new_data, self.num_qubits)
            if is_unit:
                return Unitary(new_data, self.num_qubits)
            if is_real:
                return Observable(new_data, self.num_qubits)
            return Operator(new_data, self.num_qubits)
        elif scalar_array.ndim == 0:
            # 0-dim JAX array — extract Python value for precise narrowing
            scalar_c = complex(scalar_array)
            is_real = abs(scalar_c.imag) < 1e-10
            is_unit = abs(abs(scalar_c) - 1.0) < 1e-10
            if is_real and is_unit:
                return Involution(new_data, self.num_qubits)
            if is_unit:
                return Unitary(new_data, self.num_qubits)
            if is_real:
                return Observable(new_data, self.num_qubits)
            return Operator(new_data, self.num_qubits)
        else:
            # JAX array with shape — use dtype for JIT-safe type narrowing
            if jnp.issubdtype(scalar_array.dtype, jnp.floating):
                return Observable(new_data, self.num_qubits)
            return Operator(new_data, self.num_qubits)

    @overload
    def __rmul__(self, scalar: float) -> "Observable": ...

    @overload
    def __rmul__(self, scalar: complex) -> "Operator": ...

    @overload
    def __rmul__(self, scalar: Array) -> "Involution | Unitary | Observable | Operator": ...

    def __rmul__(self, scalar: float | complex | Array) -> "Involution | Unitary | Observable | Operator":
        """Scalar multiplication (scalar on the left)."""
        return self * scalar

    def __pow__(self, exponent: float) -> "Involution | Unitary":
        """Power of an Involution.

        - Integer exponents: eigenvalues stay in {±1} → returns ``Involution``.
        - Fractional exponents: eigenvalues move onto the unit circle → returns ``Unitary``.
        """
        if jnp.allclose(int(jnp.abs(exponent)), exponent):
            new_data = jnp.linalg.matrix_power(self.matrix, int(exponent))
            return Involution.from_matrix(new_data, self.dims)

        from ._power import power_unitary

        return power_unitary(self, exponent)

    def __or__(self, other: Any) -> Any:
        """Tensor product, preserving the most specific correct type.

        - ``Involution ⊗ Involution`` → ``Involution``
        - ``Involution ⊗ Observable`` → ``Observable``
        - ``Involution ⊗ Unitary / Operator`` → ``Operator``
        """
        match other:
            case Involution():
                # I1 ⊗ I2 -> Involution (tensor product of Involutions is an Involution)
                from ._tensor import tensor_involution

                return tensor_involution(self, other)
            case Observable():
                # I ⊗ Obs -> Observable
                from ._tensor import tensor_observable

                return tensor_observable(self, other)
            case Unitary():
                # I ⊗ U -> Operator (unitary, but not generally Hermitian)
                from ._tensor import tensor_operator

                return tensor_operator(self, other)
            case Operator():
                # I ⊗ Op -> Operator
                from ._tensor import tensor_operator

                return tensor_operator(self, other)
            case _:
                return NotImplemented


# ---------- superoperators ----------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SuperOp(SuperOperator):
    """SuperOp matrix (also known as Superoperator) S.

    Tensor shape: (*ensemble, d0_out_bra, d1_out_bra, ..., d0_out_ket, d1_out_ket, ...,
                          d0_in_bra, d1_in_bra, ..., d0_in_ket, d1_in_ket, ...)
    Matrix shape: (*ensemble, d_out^2, d_in^2)
    """

    @property
    def dims(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The (output, input) dimensions of each qudit, inferred from data shape.

        Returns the qudit dimensions, not the doubled superoperator dimensions.
        """
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 4
        # dims_out is the first n_qudits dimensions (out_bra = out_ket for valid superops)
        # dims_in is the third group of n_qudits dimensions (in_bra = in_ket for valid superops)
        dims_out = qudit_shape[:n_qudits]
        dims_in = qudit_shape[2 * n_qudits : 3 * n_qudits]
        return (dims_out, dims_in)

    def _to_qobj(self) -> "qutip.Qobj | NDArray":
        """Convert to a QuTiP Qobj for interoperability testing."""
        import numpy as np
        import qutip as qt

        matrix = self.matrix

        if self.ensemble_size == ():
            return qt.Qobj(
                np.array(matrix),
                dims=[[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]],
                superrep="super",
            )

        flat_shape = (-1,) + matrix.shape[-2:]
        flat_superop = np.asarray(matrix).reshape(flat_shape)
        dims = [[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]]
        qobjs = np.asarray(
            [
                qt.Qobj(
                    np.asarray(superop),
                    dims=dims,
                    superrep="super",
                )
                for superop in flat_superop
            ],
            dtype=object,
        )
        return qobjs.reshape(self.ensemble_size)

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the superoperator using eigendecomposition (ensemble-compatible)."""
        from ._power import power_superop

        return power_superop(self, exponent)

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication of the superoperator with another superoperator."""
        match other:
            case SuperOp():
                # S1 @ S2 -> SuperOp (composition)
                from ._compose import compose_superop

                return compose_superop(self, other)
            case Choi():
                # S @ J -> Choi (convert to Choi and compose)
                from ._compose import compose_choi
                from ._superoperator_transformations import superop_to_choi

                return compose_choi(superop_to_choi(self), other)
            case PauliLiouville():
                # S @ P -> PauliLiouville (convert to PauliLiouville and compose)
                from ._compose import compose_pauli_liouville
                from ._superoperator_transformations import superop_to_pauli_liouville

                return compose_pauli_liouville(superop_to_pauli_liouville(self), other)
            case KrausMap():
                # S @ K -> KrausMap (convert to KrausMap and compose)
                from ._compose import compose_kraus_map
                from ._superoperator_transformations import superop_to_kraus

                return compose_kraus_map(superop_to_kraus(self), other)
            case Unitary():
                # S @ U -> SuperOp (promotion)
                from ._compose import compose_superop
                from ._superoperator_transformations import unitary_to_superop

                return compose_superop(self, unitary_to_superop(other))
            case StateVector():
                # S @ |ψ⟩ -> DensityMatrix (promotion)
                from ._apply import apply_superop_to_density_matrix
                from ._promotion import promote_state_vector_to_density_matrix

                return apply_superop_to_density_matrix(self, promote_state_vector_to_density_matrix(other))
            case DensityMatrix():
                # S @ ρ -> DensityMatrix (apply channel to density matrix)
                from ._apply import apply_superop_to_density_matrix

                return apply_superop_to_density_matrix(self, other)
            case _:
                return NotImplemented

    def __or__(self, other: Any) -> Any:
        """Tensor product of the superoperator with another operator."""
        match other:
            case SuperOp():
                # S1 ⊗ S2 -> SuperOp
                from ._tensor import tensor_superop

                return tensor_superop(self, other)
            case Choi():
                # S ⊗ J -> Choi (convert to Choi)
                from ._superoperator_transformations import superop_to_choi
                from ._tensor import tensor_choi

                return tensor_choi(superop_to_choi(self), other)
            case PauliLiouville():
                # S ⊗ P -> PauliLiouville (convert to PauliLiouville)
                from ._superoperator_transformations import superop_to_pauli_liouville
                from ._tensor import tensor_pauli_liouville

                return tensor_pauli_liouville(superop_to_pauli_liouville(self), other)
            case KrausMap():
                # S ⊗ K -> KrausMap (convert to KrausMap)
                from ._superoperator_transformations import superop_to_kraus
                from ._tensor import tensor_kraus

                return tensor_kraus(superop_to_kraus(self), other)
            case Unitary():
                # S ⊗ U -> SuperOp (promote unitary)
                from ._superoperator_transformations import unitary_to_superop
                from ._tensor import tensor_superop

                return tensor_superop(self, unitary_to_superop(other))
            case StateVector() | DensityMatrix():
                # Operator | State -> NotImplemented
                return NotImplemented
            case _:
                return NotImplemented

    def __eq__(self, other: Any) -> bool:
        """Check equality using process fidelity."""
        match other:
            case SuperOp():
                # Compare two superoperators using process fidelity
                from ._distance_metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case Choi() | PauliLiouville() | KrausMap():
                # Convert other to SuperOp and compare
                from ._distance_metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case Unitary():
                # Promote Unitary to SuperOp and compare
                from ._distance_metrics import process_fidelity
                from ._superoperator_transformations import unitary_to_superop

                return bool(jnp.allclose(process_fidelity(self, unitary_to_superop(other)), 1.0))
            case StateVector() | DensityMatrix():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KrausMap(SuperOperator):
    """Kraus channel.

    Tensor shape: (*ensemble, n_kraus, d0_out, d1_out, ..., d0_in, d1_in, ...)
    Matrix shape: (*ensemble, n_kraus, d_out, d_in)
    """

    @property
    def num_ensemble_dims(self) -> int:
        """The number of leading ensemble dimensions, derived from data shape and num_qubits."""
        return self.data.ndim - 2 * self.num_qubits - 1

    @property
    def dims(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The (output, input) dimensions of each qudit, inferred from data shape."""
        # After ensemble dims, first is n_kraus, then qudit dims
        qudit_shape = self.data.shape[self.num_ensemble_dims + 1 :]
        n_qudits = len(qudit_shape) // 2
        return (qudit_shape[:n_qudits], qudit_shape[n_qudits:])

    @property
    def ensemble_size(self) -> Tuple[int, ...]:
        """Returns the size of the ensemble if the operator represents an ensemble of operators."""
        return self.data.shape[: self.num_ensemble_dims]

    @property
    def matrix(self) -> Array:
        """Returns the matrix representation (*ensemble, n_kraus, d_out, d_in) of the Kraus map."""
        ensemble_shape = self.data.shape[: self.num_ensemble_dims]
        n_kraus = self.data.shape[self.num_ensemble_dims]
        qudit_shape = self.data.shape[self.num_ensemble_dims + 1 :]
        n_qudits = len(qudit_shape) // 2
        d_out = reduce(mul, qudit_shape[:n_qudits], 1)
        d_in = reduce(mul, qudit_shape[n_qudits:], 1)
        return self.data.reshape(ensemble_shape + (n_kraus, d_out, d_in))

    @classmethod
    def from_matrix(cls, matrix: Array, dims: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> "KrausMap":
        """Construct from matrix representation.

        :param matrix: Array with shape (*ensemble, n_kraus, d_out, d_in)
        :param dims: Tuple of (dims_out, dims_in) where each is a tuple of qudit dimensions
        :return: KrausMap with tensor data
        """
        num_qubits = len(dims[0])
        ensemble_shape = matrix.shape[:-3]
        n_kraus = matrix.shape[-3]
        tensor = matrix.reshape(ensemble_shape + (n_kraus,) + dims[0] + dims[1])
        return cls(data=tensor, num_qubits=num_qubits)

    def _to_qobj(self) -> list["qutip.Qobj"]:
        """Convert to a QuTiP Qobj for interoperability testing.

        Returns a list of QuTiP Qobjs, one for each Kraus operator.
        """
        import numpy as np
        import qutip as qt

        matrix = self.matrix
        # KrausMap is always an ensemble of Kraus operators, shape (K, d_out, d_in)
        return [qt.Qobj(np.array(k), dims=[[list(self.dims[0])], [list(self.dims[1])]]) for k in matrix]

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the Kraus channel using eigendecomposition (ensemble-compatible)."""
        from ._power import power_kraus

        return power_kraus(self, exponent)

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication of the Kraus channel with another superoperator."""
        match other:
            case KrausMap():
                # K1 @ K2 -> KrausMap (composition)
                from ._compose import compose_kraus_map

                return compose_kraus_map(self, other)
            case SuperOp():
                # K @ S -> SuperOp (convert to SuperOp and compose)
                from ._compose import compose_superop
                from ._superoperator_transformations import kraus_to_superop

                return compose_superop(kraus_to_superop(self), other)
            case Choi():
                # K @ J -> Choi (convert to Choi and compose)
                from ._compose import compose_choi
                from ._superoperator_transformations import kraus_to_choi

                return compose_choi(kraus_to_choi(self), other)
            case PauliLiouville():
                # K @ P -> PauliLiouville (convert to PauliLiouville and compose)
                from ._compose import compose_pauli_liouville
                from ._superoperator_transformations import kraus_to_pauli_liouville

                return compose_pauli_liouville(kraus_to_pauli_liouville(self), other)
            case Unitary():
                # K @ U -> KrausMap (promotion)
                from ._compose import compose_kraus_map
                from ._superoperator_transformations import unitary_to_kraus

                return compose_kraus_map(self, unitary_to_kraus(other))
            case StateVector():
                # K @ |ψ⟩ -> DensityMatrix (promotion)
                from ._apply import apply_kraus_to_density_matrix
                from ._promotion import promote_state_vector_to_density_matrix

                return apply_kraus_to_density_matrix(self, promote_state_vector_to_density_matrix(other))
            case DensityMatrix():
                # K @ ρ -> DensityMatrix (apply channel to density matrix)
                from ._apply import apply_kraus_to_density_matrix

                return apply_kraus_to_density_matrix(self, other)
            case _:
                return NotImplemented

    def __or__(self, other: Any) -> Any:
        """Tensor product of the Kraus channel with another quantum object."""
        match other:
            case KrausMap():
                # K1 | K2 -> KrausMap
                from ._tensor import tensor_kraus

                return tensor_kraus(self, other)
            case SuperOp():
                # K | S -> SuperOp (convert to SuperOp and tensor)
                from ._superoperator_transformations import kraus_to_superop
                from ._tensor import tensor_superop

                return tensor_superop(kraus_to_superop(self), other)
            case Choi():
                # K | J -> Choi (convert to Choi and tensor)
                from ._superoperator_transformations import kraus_to_choi
                from ._tensor import tensor_choi

                return tensor_choi(kraus_to_choi(self), other)
            case PauliLiouville():
                # K | P -> PauliLiouville (convert to PauliLiouville and tensor)
                from ._superoperator_transformations import kraus_to_pauli_liouville
                from ._tensor import tensor_pauli_liouville

                return tensor_pauli_liouville(kraus_to_pauli_liouville(self), other)
            case Unitary():
                # K | U -> KrausMap (promote Unitary to KrausMap and tensor)
                from ._superoperator_transformations import unitary_to_kraus
                from ._tensor import tensor_kraus

                return tensor_kraus(self, unitary_to_kraus(other))
            case StateVector() | DensityMatrix():
                # K | |ψ⟩ or K | ρ -> NotImplemented (operator | state)
                return NotImplemented
            case _:
                return NotImplemented

    def __eq__(self, other: Any) -> bool:
        """Check equality using process fidelity."""
        match other:
            case KrausMap():
                # Compare two KrausMaps using process fidelity
                from ._distance_metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case SuperOp() | Choi() | PauliLiouville():
                # Compare using process fidelity (handles conversions internally)
                from ._distance_metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case Unitary():
                # Promote Unitary to KrausMap and compare
                from ._distance_metrics import process_fidelity
                from ._superoperator_transformations import unitary_to_kraus

                return bool(jnp.allclose(process_fidelity(self, unitary_to_kraus(other)), 1.0))
            case StateVector() | DensityMatrix():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Choi(SuperOperator):
    """Choi matrix C.

    Tensor shape: (*ensemble, d0_out_bra, d1_out_bra, ..., d0_out_ket, d1_out_ket, ...,
                          d0_in_bra, d1_in_bra, ..., d0_in_ket, d1_in_ket, ...)
    Matrix shape: (*ensemble, d_out^2, d_in^2)
    """

    @property
    def dims(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The (output, input) dimensions of each qudit, inferred from data shape."""
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 4
        dims_out = qudit_shape[:n_qudits]
        dims_in = qudit_shape[2 * n_qudits : 3 * n_qudits]
        return (dims_out, dims_in)

    def _to_qobj(self) -> "qutip.Qobj | NDArray":
        """Convert to a QuTiP Qobj for interoperability testing."""
        import numpy as np
        import qutip as qt

        matrix = self.matrix

        if self.ensemble_size == ():
            return qt.Qobj(
                np.array(matrix),
                dims=[[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]],
                superrep="choi",
            )

        flat_shape = (-1,) + matrix.shape[-2:]
        flat_choi = np.asarray(matrix).reshape(flat_shape)
        dims = [[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]]
        qobjs = np.asarray(
            [
                qt.Qobj(
                    np.asarray(choi),
                    dims=dims,
                    superrep="choi",
                )
                for choi in flat_choi
            ],
            dtype=object,
        )
        return qobjs.reshape(self.ensemble_size)

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the Choi matrix using eigendecomposition (ensemble-compatible)."""
        from ._power import power_choi

        return power_choi(self, exponent)

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication of the Choi with another Superoperator."""
        match other:
            case Choi():
                # J1 @ J2 -> Choi (composition)
                from ._compose import compose_choi

                return compose_choi(self, other)
            case SuperOp():
                # J @ S -> SuperOp (convert to SuperOp and compose)
                from ._compose import compose_superop
                from ._superoperator_transformations import choi_to_superop

                return compose_superop(choi_to_superop(self), other)
            case PauliLiouville():
                # J @ P -> PauliLiouville (convert to PauliLiouville and compose)
                from ._compose import compose_pauli_liouville
                from ._superoperator_transformations import choi_to_pauli_liouville

                return compose_pauli_liouville(choi_to_pauli_liouville(self), other)
            case KrausMap():
                # J @ K -> KrausMap (convert to KrausMap and compose)
                from ._compose import compose_kraus_map
                from ._superoperator_transformations import choi_to_kraus

                return compose_kraus_map(choi_to_kraus(self), other)
            case Unitary():
                # J @ U -> Choi (promotion)
                from ._compose import compose_choi
                from ._superoperator_transformations import unitary_to_choi

                return compose_choi(self, unitary_to_choi(other))
            case StateVector():
                # J @ |ψ⟩ -> DensityMatrix (apply channel to state)
                from ._apply import apply_choi_to_density_matrix
                from ._promotion import promote_state_vector_to_density_matrix

                return apply_choi_to_density_matrix(self, promote_state_vector_to_density_matrix(other))
            case DensityMatrix():
                # J @ ρ -> DensityMatrix (apply channel to density matrix)
                from ._apply import apply_choi_to_density_matrix

                return apply_choi_to_density_matrix(self, other)
            case _:
                return NotImplemented

    def __or__(self, other: Any) -> Any:
        """Tensor product of the Choi matrix with another quantum object."""
        match other:
            case Choi():
                # J1 | J2 -> Choi
                from ._tensor import tensor_choi

                return tensor_choi(self, other)
            case SuperOp():
                # J | S -> SuperOp (convert to SuperOp and tensor)
                from ._superoperator_transformations import choi_to_superop
                from ._tensor import tensor_superop

                return tensor_superop(choi_to_superop(self), other)
            case PauliLiouville():
                # J | P -> PauliLiouville (convert to PauliLiouville and tensor)
                from ._superoperator_transformations import choi_to_pauli_liouville
                from ._tensor import tensor_pauli_liouville

                return tensor_pauli_liouville(choi_to_pauli_liouville(self), other)
            case KrausMap():
                # J | K -> KrausMap (convert to KrausMap and tensor)
                from ._superoperator_transformations import choi_to_kraus
                from ._tensor import tensor_kraus

                return tensor_kraus(choi_to_kraus(self), other)
            case Unitary():
                # J | U -> Choi (promote Unitary to Choi and tensor)
                from ._superoperator_transformations import unitary_to_choi
                from ._tensor import tensor_choi

                return tensor_choi(self, unitary_to_choi(other))
            case StateVector() | DensityMatrix():
                # J | |ψ⟩ or J | ρ -> NotImplemented (operator | state)
                return NotImplemented
            case _:
                return NotImplemented

    def __eq__(self, other: Any) -> bool:
        """Check equality using process fidelity."""
        match other:
            case Choi():
                # Compare two Choi matrices using process fidelity
                from ._distance_metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case SuperOp() | PauliLiouville() | KrausMap():
                # Compare using process fidelity (handles conversions internally)
                from ._distance_metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case Unitary():
                # Promote Unitary to Choi and compare
                from ._distance_metrics import process_fidelity
                from ._superoperator_transformations import unitary_to_choi

                return bool(jnp.allclose(process_fidelity(self, unitary_to_choi(other)), 1.0))
            case StateVector() | DensityMatrix():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Chi(SuperOperator):
    """Chi matrix Χ.

    Tensor shape: (*ensemble, d0_out_bra, d1_out_bra, ..., d0_out_ket, d1_out_ket, ...,
                          d0_in_bra, d1_in_bra, ..., d0_in_ket, d1_in_ket, ...)
    Matrix shape: (*ensemble, d_out^2, d_in^2)
    """

    @property
    def dims(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The (output, input) dimensions of each qudit, inferred from data shape."""
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 4
        dims_out = qudit_shape[:n_qudits]
        dims_in = qudit_shape[2 * n_qudits : 3 * n_qudits]
        return (dims_out, dims_in)

    def _to_qobj(self) -> "qutip.Qobj | list[qutip.Qobj]":
        """Convert to a QuTiP Qobj for interoperability testing."""
        import numpy as np
        import qutip as qt

        matrix = self.matrix

        if self.ensemble_size != ():
            # Batched Chi matrices - return list
            chi_qobjs = []
            for c in matrix:
                base_qobj = qt.Qobj(
                    np.array(c),
                    dims=[[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]],
                )
                chi_qobjs.append(qt.to_chi(base_qobj))
            return chi_qobjs
        # Single Chi matrix
        base_qobj = qt.Qobj(
            np.array(matrix),
            dims=[[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]],
        )
        return qt.to_chi(base_qobj)

    def __pow__(self, exponent: float) -> Self:
        return NotImplemented

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication of the Chi matrix with another superoperator."""
        match other:
            case Chi():
                # Composition of Chi matrices is not directly supported.
                # One would need to convert to another representation (e.g., SuperOp),
                # compose, and then convert back.
                # No chi transformations are implemented
                raise NotImplementedError("Composition of Chi matrices is not implemented.")
            case SuperOp():
                # χ @ S -> SuperOp (convert Chi to SuperOp and compose)
                raise NotImplementedError("Chi to SuperOp conversion not implemented.")
            case Choi():
                # χ @ J -> Choi (convert Chi to Choi and compose)
                raise NotImplementedError("Chi to Choi conversion not implemented.")
            case PauliLiouville():
                # χ @ P -> PauliLiouville (convert Chi to PauliLiouville and compose)
                raise NotImplementedError("Chi to PauliLiouville conversion not implemented.")
            case KrausMap():
                # χ @ K -> KrausMap (convert Chi to KrausMap and compose)
                raise NotImplementedError("Chi to KrausMap conversion not implemented.")
            case Unitary():
                # χ @ U -> Chi (promotion)
                raise NotImplementedError("Unitary to Chi conversion not implemented.")
            case StateVector():
                # χ @ |ψ⟩ -> DensityMatrix (convert to applicable representation)
                raise NotImplementedError("Applying Chi to StateVector is not implemented.")
            case DensityMatrix():
                # χ @ ρ -> DensityMatrix (convert to applicable representation)
                raise NotImplementedError("Applying Chi to DensityMatrix is not implemented.")
            case _:
                return NotImplemented

    def __or__(self, other: Any) -> Any:
        """Tensor product of the Chi matrix with another quantum object."""
        # Since Chi transformations are not implemented, we can only handle Chi | Chi
        match other:
            case Chi():
                # χ1 | χ2 -> Chi
                raise NotImplementedError("Tensor product of Chi matrices is not implemented.")
            case SuperOp() | Choi() | PauliLiouville() | KrausMap() | Unitary():
                # χ | S -> would require Chi transformations (not implemented)
                raise NotImplementedError(f"Tensor product not implemented between Chi and {type(other).__name__}.")
            case StateVector() | DensityMatrix():
                # χ | |ψ⟩ or χ | ρ -> NotImplemented (operator | state)
                return NotImplemented
            case _:
                return NotImplemented

    def __eq__(self, other: Any) -> bool:
        """Check equality for Chi matrices."""
        # Since Chi transformations are not implemented, we can only compare Chi with Chi
        match other:
            case Chi():
                # Direct comparison not possible without Chi transformations
                raise NotImplementedError("Equality comparison for Chi matrices is not implemented.")
            case SuperOp() | Choi() | PauliLiouville() | KrausMap() | Unitary():
                # Would require Chi transformations (not implemented)
                raise NotImplementedError(
                    "Equality comparison between Chi and other superoperators is not implemented."
                )
            case StateVector() | DensityMatrix():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PauliLiouville(SuperOperator):
    """Pauli-Liouville matrix P.

    Tensor shape: (*ensemble, d0_out_bra, d1_out_bra, ..., d0_out_ket, d1_out_ket, ...,
                          d0_in_bra, d1_in_bra, ..., d0_in_ket, d1_in_ket, ...)
    Matrix shape: (*ensemble, d_out^2, d_in^2)
    """

    @property
    def dims(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The (output, input) dimensions of each qudit, inferred from data shape."""
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 4
        dims_out = qudit_shape[:n_qudits]
        dims_in = qudit_shape[2 * n_qudits : 3 * n_qudits]
        return (dims_out, dims_in)

    def _to_qobj(self):
        """Convert to a QuTiP Qobj for interoperability testing.

        Note: QuTiP doesn't have native Pauli-Liouville representation,
        so this will look like a SuperOp Qobj, but it will be the Pauli-Liouville matrix.
        """
        raise NotImplementedError("Conversion to QuTiP Qobj not implemented for PauliLiouville.")

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the Pauli-Liouville matrix using eigendecomposition (ensemble-compatible)."""
        from ._power import power_pauli_liouville

        return power_pauli_liouville(self, exponent)

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication of the PauliLiouville with another PauliLiouville."""
        match other:
            case PauliLiouville():
                # P1 @ P2 -> PauliLiouville (composition)
                from ._compose import compose_pauli_liouville

                return compose_pauli_liouville(self, other)
            case SuperOp():
                # P @ S -> SuperOp (convert to SuperOp and compose)
                from ._compose import compose_superop
                from ._superoperator_transformations import pauli_liouville_to_superop

                return compose_superop(pauli_liouville_to_superop(self), other)
            case Choi():
                # P @ J -> Choi (convert to Choi and compose)
                from ._compose import compose_choi
                from ._superoperator_transformations import pauli_liouville_to_choi

                return compose_choi(pauli_liouville_to_choi(self), other)
            case KrausMap():
                # P @ K -> KrausMap (convert to KrausMap and compose)
                from ._compose import compose_kraus_map
                from ._superoperator_transformations import pauli_liouville_to_kraus

                return compose_kraus_map(pauli_liouville_to_kraus(self), other)
            case Unitary():
                # P @ U -> PauliLiouville (promotion)
                from ._compose import compose_pauli_liouville
                from ._superoperator_transformations import unitary_to_pauli_liouville

                return compose_pauli_liouville(self, unitary_to_pauli_liouville(other))
            case StateVector():
                # P @ |ψ⟩ -> DensityMatrix (apply channel to state)
                from ._apply import apply_pauli_liouville_to_density_matrix
                from ._promotion import promote_state_vector_to_density_matrix

                return apply_pauli_liouville_to_density_matrix(self, promote_state_vector_to_density_matrix(other))
            case DensityMatrix():
                # P @ ρ -> DensityMatrix (apply channel to density matrix)
                from ._apply import apply_pauli_liouville_to_density_matrix

                return apply_pauli_liouville_to_density_matrix(self, other)
            case _:
                return NotImplemented

    def __or__(self, other: Any) -> Any:
        """Tensor product of the Pauli-Liouville matrix with another quantum object."""
        match other:
            case PauliLiouville():
                # P1 | P2 -> PauliLiouville
                from ._tensor import tensor_pauli_liouville

                return tensor_pauli_liouville(self, other)
            case SuperOp():
                # P | S -> PauliLiouville (convert to PauliLiouville and tensor)
                from ._superoperator_transformations import pauli_liouville_to_superop
                from ._tensor import tensor_superop

                return tensor_superop(pauli_liouville_to_superop(self), other)
            case Choi():
                # P | J -> PauliLiouville (convert to PauliLiouville and tensor)
                from ._superoperator_transformations import pauli_liouville_to_choi
                from ._tensor import tensor_choi

                return tensor_choi(pauli_liouville_to_choi(self), other)
            case KrausMap():
                # P | K -> PauliLiouville (convert to PauliLiouville and tensor)
                from ._superoperator_transformations import pauli_liouville_to_kraus
                from ._tensor import tensor_kraus

                return tensor_kraus(pauli_liouville_to_kraus(self), other)
            case Unitary():
                # P | U -> PauliLiouville (promote Unitary to PauliLiouville and tensor)
                from ._superoperator_transformations import unitary_to_pauli_liouville
                from ._tensor import tensor_pauli_liouville

                return tensor_pauli_liouville(self, unitary_to_pauli_liouville(other))
            case StateVector() | DensityMatrix():
                # P | |ψ⟩ or P | ρ -> NotImplemented (operator | state)
                return NotImplemented
            case _:
                return NotImplemented

    def __eq__(self, other: Any) -> bool:
        """Check equality using process fidelity."""
        match other:
            case PauliLiouville():
                # Compare two PauliLiouville matrices using process fidelity
                from ._distance_metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case SuperOp() | Choi() | KrausMap():
                # Compare using process fidelity (handles conversions internally)
                from ._distance_metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case Unitary():
                # Promote Unitary to PauliLiouville and compare
                from ._distance_metrics import process_fidelity
                from ._superoperator_transformations import unitary_to_pauli_liouville

                return bool(jnp.allclose(process_fidelity(self, unitary_to_pauli_liouville(other)), 1.0))
            case StateVector() | DensityMatrix():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented
