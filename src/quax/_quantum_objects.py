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
from functools import cached_property, reduce
from operator import mul
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Self, Sequence, Tuple, overload

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
        """Scalar multiplication of the quantum object."""
        from ._mul import mul

        return mul(self, scalar)  # type: ignore[return-value]

    def __rmul__(self, scalar: complex | Array) -> Self:
        """Scalar multiplication of the quantum object."""
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

    def __eq__(self, other: Any) -> bool:
        """Element-wise equality check with numerical tolerance."""
        if not isinstance(other, Operator):
            return NotImplemented
        if self.dims != other.dims:
            return False
        return bool(jnp.allclose(self.data, other.data))

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

        dims = [list(self.dims[0]), list(self.dims[1])]
        if self.ensemble_size == ():
            return qt.Qobj(
                np.array(matrix),
                dims=dims,
            )

        flat_shape = (-1,) + matrix.shape[-2:]
        flat_kraus = np.asarray(matrix).reshape(flat_shape)
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
    def d(self) -> Tuple[int, ...]:
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
        """Returns the matrix representation ``(*ensemble, d_out, d_in)`` of the operator."""
        ensemble_shape = self.data.shape[: self.num_ensemble_dims]
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 2
        d_out = reduce(mul, qudit_shape[:n_qudits], 1)
        d_in = reduce(mul, qudit_shape[n_qudits:], 1)
        return self.data.reshape(ensemble_shape + (d_out, d_in))

    @classmethod
    def from_matrix(cls, matrix: Array, dims: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Self:
        """Construct from matrix representation.

        :param matrix: Array with shape ``(*ensemble, d_out, d_in)``
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
    ``(*ensemble, d0_out_bra, d1_out_bra, ..., d0_out_ket, d1_out_ket, ..., d0_in_bra, d1_in_bra, ..., d0_in_ket, d1_in_ket, ...)``

    SuperOperators represent CPTP maps; ``+`` and ``-`` are intentionally not
    defined because the sum/difference of CPTP maps is not generally CPTP.
    """

    @property
    def num_ensemble_dims(self) -> int:
        """The number of leading ensemble dimensions, derived from data shape and num_qubits."""
        return self.data.ndim - 4 * self.num_qubits

    @property
    def matrix(self) -> Array:
        """Returns the matrix representation ``(*ensemble, d_out^2, d_in^2)`` of the superoperator."""
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

        :param matrix: Array with shape ``(*ensemble, d_out^2, d_in^2)``
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
    def d(self) -> Tuple[int, ...]:
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
    """State vector ``|psi>``, shape ``(*ensemble, d0, d1, ...)`` in tensor form or ``(*ensemble, d)`` in matrix form."""

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
        """Returns the vector representation ``(*ensemble, d)`` of the state."""
        ensemble_shape = self.data.shape[: self.num_ensemble_dims]
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        d = reduce(mul, qudit_shape, 1)
        return self.data.reshape(ensemble_shape + (d,))

    @classmethod
    def from_matrix(cls, matrix: Array, dims: Tuple[int, ...]) -> "StateVector":
        """Construct from vector representation.

        :param matrix: Array with shape ``(*ensemble, d)`` where d = prod(dims)
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
                from ._promotion import promote_hilbert_space

                self_p, other_p = promote_hilbert_space(self, other)
                return jnp.einsum("...a,...a->...", self_p.matrix.conj(), other_p.matrix)
            case DensityMatrix():  #  <𝜓|𝜌 -> <𝜙|
                from ._promotion import promote_hilbert_space

                self_p, other_p = promote_hilbert_space(self, other)
                result = jnp.einsum("...b,...ba->...a", self_p.matrix.conj(), other_p.matrix)
                return StateVector.from_matrix(result, self_p.dims)
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
                from ._metrics import fidelity

                return bool(jnp.allclose(fidelity(self, other), 1.0))
            case DensityMatrix():
                # Promote self to density matrix and compare
                from ._metrics import fidelity
                from ._promotion import promote_state_vector_to_density_matrix

                return bool(jnp.allclose(fidelity(promote_state_vector_to_density_matrix(self), other), 1.0))
            case Unitary() | SuperOp() | Choi() | PauliLiouville() | KrausMap():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented

    def pretty_print(self, decimals: int = 3, atol: float = 1e-6) -> str:
        """Return a human-readable Unicode Dirac-notation string for this state vector.

        Basis states whose amplitude magnitude is below *atol* are omitted.
        Amplitudes are rounded to *decimals* decimal places with trailing zeros stripped.
        For ensembled states each element is printed on its own labelled line.

        :param decimals: Number of decimal places for each amplitude (default: 3).
        :param atol: Amplitude magnitude threshold below which terms are dropped (default: 1e-6).
        :return: A Unicode string, e.g. ``0.707|0\u27e9 + 0.707|1\u27e9``.

        Example::

            >>> sv = StateVector.from_matrix(jnp.array([0.0, 1.0], dtype=complex), dims=(2,))
            >>> sv.pretty_print()
            '1|1\u27e9'
        """
        from ._state import _format_state_vector_str
        import itertools

        matrix = self.matrix
        dims = self.dims

        if self.ensemble_size == ():
            return _format_state_vector_str(matrix, dims, decimals, atol)

        lines = []
        for idx in itertools.product(*[range(s) for s in self.ensemble_size]):
            label = "[" + ", ".join(str(i) for i in idx) + "]"
            vec = matrix[idx]
            lines.append(f"{label}: {_format_state_vector_str(vec, dims, decimals, atol)}")
        return "\n".join(lines)

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
    """Density matrix ρ, shape ``(*ensemble, d0_out, d1_out, ..., d0_in, d1_in, ...)`` in tensor form
    or ``(*ensemble, d, d)`` in matrix form."""

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
        """Returns the matrix representation ``(*ensemble, d, d)`` of the density matrix."""
        ensemble_shape = self.data.shape[: self.num_ensemble_dims]
        qudit_shape = self.data.shape[self.num_ensemble_dims :]
        n_qudits = len(qudit_shape) // 2
        d_out = reduce(mul, qudit_shape[:n_qudits], 1)
        d_in = reduce(mul, qudit_shape[n_qudits:], 1)
        return self.data.reshape(ensemble_shape + (d_out, d_in))

    @classmethod
    def from_matrix(cls, matrix: Array, dims: Tuple[int, ...]) -> "DensityMatrix":
        """Construct from matrix representation.

        :param matrix: Array with shape ``(*ensemble, d, d)`` where d = prod(dims)
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
        from ._exponentiation import density_matrix_power

        return density_matrix_power(self, exponent)

    def __matmul__(self, other):
        """Left multiply the density matrix by another object."""
        match other:
            case StateVector():  # 𝜌|𝜓> -> |𝜙>
                from ._promotion import promote_hilbert_space

                self_p, other_p = promote_hilbert_space(self, other)
                result = jnp.einsum("...ab,...b->...a", self_p.matrix, other_p.matrix)
                return StateVector.from_matrix(result, other_p.dims)
            case DensityMatrix():  # 𝜌𝜎 -> 𝜏
                from ._promotion import promote_hilbert_space

                self_p, other_p = promote_hilbert_space(self, other)
                result = jnp.einsum("...ab,...bc->...ac", self_p.matrix, other_p.matrix)
                return DensityMatrix.from_matrix(result, self_p.dims)
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
                from ._metrics import fidelity
                from ._promotion import promote_state_vector_to_density_matrix

                return bool(jnp.allclose(fidelity(self, promote_state_vector_to_density_matrix(other)), 1.0))
            case DensityMatrix():
                # Compare two density matrices using fidelity
                from ._metrics import fidelity

                return bool(jnp.allclose(fidelity(self, other), 1.0))
            case Unitary() | SuperOp() | Choi() | PauliLiouville() | KrausMap():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented

    def pretty_print(self, decimals: int = 3, atol: float = 1e-6) -> str:
        """Return a human-readable Unicode string in the ``|i⟩⟨j|`` operator basis.

        Matrix elements whose magnitude is below *atol* are omitted.
        Values are rounded to *decimals* decimal places with trailing zeros stripped.
        For ensembled density matrices each element is printed on its own labelled line.

        :param decimals: Number of decimal places for each element (default: 3).
        :param atol: Magnitude threshold below which terms are dropped (default: 1e-6).
        :return: A Unicode string, e.g. ``0.5|0⟩⟨0| + 0.5|1⟩⟨1|``.

        Example::

            >>> rho = DensityMatrix.from_matrix(jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex), dims=(2,))
            >>> rho.pretty_print()
            '0.5|0⟩⟨0| + 0.5|0⟩⟨1| + 0.5|1⟩⟨0| + 0.5|1⟩⟨1|'
        """
        from ._state import _format_density_matrix_str
        import itertools

        matrix = self.matrix
        dims = self.dims

        if self.ensemble_size == ():
            return _format_density_matrix_str(matrix, dims, decimals, atol)

        lines = []
        for idx in itertools.product(*[range(s) for s in self.ensemble_size]):
            label = "[" + ", ".join(str(i) for i in idx) + "]"
            mat = matrix[idx]
            lines.append(f"{label}: {_format_density_matrix_str(mat, dims, decimals, atol)}")
        return "\n".join(lines)

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
    """Unitary operator U, shape ``(*ensemble, d0_out, d1_out, ..., d0_in, d1_in, ...)`` in tensor form
    or ``(*ensemble, d, d)`` in matrix form."""

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

    def __mul__(self, scalar: complex | Array) -> "Operator":
        """Scalar multiplication of the unitary.

        Always returns ``Operator`` since scalar multiplication does not,
        in general, preserve unitarity.
        """
        from ._mul import mul

        return mul(self, scalar)

    def __rmul__(self, scalar: complex | Array) -> "Operator":
        """Scalar multiplication of the unitary."""
        return self * scalar

    def __pow__(self, exponent: float) -> "Unitary":
        """Exponentiation of the unitary using eigendecomposition (ensemble-compatible)."""
        from ._exponentiation import power_unitary

        return power_unitary(self, exponent)

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication of a quantum object with the unitary."""
        match other:
            case Unitary():
                # U @ U -> Unitary
                from ._compose import compose_unitary

                return compose_unitary(self, other)
            case Choi():
                # U @ J -> Choi (promotion): U applied second, J first
                from ._compose import compose_choi
                from ._superoperator_transformations import unitary_to_choi

                return compose_choi(unitary_to_choi(self), other)
            case PauliLiouville():
                # U @ P -> PauliLiouville (promotion): U applied second, P first
                from ._compose import compose_pauli_liouville
                from ._superoperator_transformations import unitary_to_pauli_liouville

                return compose_pauli_liouville(unitary_to_pauli_liouville(self), other)
            case SuperOp():
                # U @ S -> SuperOp (promotion): U applied second, S first
                from ._compose import compose_superop
                from ._superoperator_transformations import unitary_to_superop

                return compose_superop(unitary_to_superop(self), other)
            case KrausMap():
                # U @ K -> KrausMap: U applied second, K first
                from ._compose import compose_kraus_map
                from ._superoperator_transformations import unitary_to_kraus_map

                return compose_kraus_map(unitary_to_kraus_map(self), other)
            case StateVector():
                # <psi|U = <phi| -> StateVector (apply unitary to state vector)
                from ._apply import apply_unitary_to_state_vector

                return apply_unitary_to_state_vector(self, other)
            case DensityMatrix():
                # U ρ U† → DensityMatrix (apply unitary channel to density matrix)
                from ._apply import apply_unitary_to_density_matrix

                return apply_unitary_to_density_matrix(self, other)
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
                from ._superoperator_transformations import unitary_to_kraus_map
                from ._tensor import tensor_kraus

                return tensor_kraus(unitary_to_kraus_map(self), other)
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
                from ._metrics import unitary_entanglement_fidelity

                return bool(jnp.allclose(unitary_entanglement_fidelity(self, other), 1.0))
            case SuperOp() | Choi() | PauliLiouville() | KrausMap():
                # Promote self to superoperator and compare using process fidelity
                from ._metrics import process_fidelity
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

    Tensor shape: ``(*ensemble, d0_out, d1_out, ..., d0_in, d1_in, ...)``
    Matrix shape: ``(*ensemble, d, d)``
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
        from ._mul import mul

        return mul(self, scalar)

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
        from ._exponentiation import power_observable

        return power_observable(self, exponent)

    @overload
    def __add__(self, other: "Observable") -> "Observable": ...

    @overload
    def __add__(self, other: "Operator") -> "Operator": ...

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

    @overload
    def __radd__(self, other: "Observable") -> "Observable": ...

    @overload
    def __radd__(self, other: "Operator") -> "Operator": ...

    def __radd__(self, other: Any) -> "Observable | Operator":
        """Right-hand addition."""
        match other:
            case Observable() | Operator():
                return other.__add__(self)
            case _:
                return NotImplemented

    @overload
    def __sub__(self, other: "Observable") -> "Observable": ...

    @overload
    def __sub__(self, other: "Operator") -> "Operator": ...

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

    @overload
    def __rsub__(self, other: "Observable") -> "Observable": ...

    @overload
    def __rsub__(self, other: "Operator") -> "Operator": ...

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
    - Real scalar × Involution → ``Observable``; complex scalar → ``Operator``.
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
    def __mul__(self, scalar: Array) -> "Observable | Operator": ...

    def __mul__(self, scalar: float | complex | Array) -> "Observable | Operator":
        """Scalar multiplication with ensemble broadcasting.

        - Real scalar/dtype: Hermitian is preserved → ``Observable``.
        - Complex scalar/dtype: → ``Operator``.
        """
        from ._mul import mul

        return mul(self, scalar)

    @overload
    def __rmul__(self, scalar: float) -> "Observable": ...

    @overload
    def __rmul__(self, scalar: complex) -> "Operator": ...

    @overload
    def __rmul__(self, scalar: Array) -> "Observable | Operator": ...

    def __rmul__(self, scalar: float | complex | Array) -> "Observable | Operator":
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

        from ._exponentiation import power_unitary

        return power_unitary(self, exponent)

    def __or__(self, other: Any) -> Any:
        """Tensor product, preserving the most specific correct type.

        - ``Involution ⊗ Involution`` → ``Involution``
        - ``Involution ⊗ Observable`` → ``Observable``
        - ``Involution ⊗ Unitary / Operator`` → ``Operator``
        - ``Involution ⊗ SuperOp / Choi / PauliLiouville / KrausMap`` → (delegates to Unitary)
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
                from ._tensor import tensor_unitary

                return tensor_unitary(self, other)
            case Operator():
                # I ⊗ Op -> Operator
                from ._tensor import tensor_operator

                return tensor_operator(self, other)
            case _:
                # Superoperator types (SuperOp, Choi, PauliLiouville, KrausMap)
                # are handled by the Unitary parent class.
                return Unitary.__or__(self, other)


# ---------- superoperators ----------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SuperOp(SuperOperator):
    """SuperOp matrix (also known as Superoperator) S.

    Tensor shape: ``(*ensemble, d0_out_bra, d1_out_bra, ..., d0_out_ket, d1_out_ket, ..., d0_in_bra, d1_in_bra, ..., d0_in_ket, d1_in_ket, ...)``

    Matrix shape: ``(*ensemble, d_out^2, d_in^2)``
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
        from ._exponentiation import power_superop

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
                from ._metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case Choi() | PauliLiouville() | KrausMap():
                # Convert other to SuperOp and compare
                from ._metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case Unitary():
                # Promote Unitary to SuperOp and compare
                from ._metrics import process_fidelity
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

    Tensor shape: ``(*ensemble, n_kraus, d0_out, d1_out, ..., d0_in, d1_in, ...)``

    Matrix shape: ``(*ensemble, n_kraus, d_out, d_in)``
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
        """Returns the matrix representation ``(*ensemble, n_kraus, d_out, d_in)`` of the Kraus map."""
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

        :param matrix: Array with shape ``(*ensemble, n_kraus, d_out, d_in)``
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
        from ._exponentiation import power_kraus

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
                # K @ U -> KrausMap: compose each Kraus operator with U (applied first)
                from ._compose import compose_kraus_map
                from ._superoperator_transformations import unitary_to_kraus_map

                return compose_kraus_map(self, unitary_to_kraus_map(other))
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
                from ._superoperator_transformations import unitary_to_kraus_map
                from ._tensor import tensor_kraus

                return tensor_kraus(self, unitary_to_kraus_map(other))
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
                from ._metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case SuperOp() | Choi() | PauliLiouville():
                # Compare using process fidelity (handles conversions internally)
                from ._metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case Unitary():
                # Promote Unitary to KrausMap and compare
                from ._metrics import process_fidelity
                from ._superoperator_transformations import unitary_to_kraus_map

                return bool(jnp.allclose(process_fidelity(self, unitary_to_kraus_map(other)), 1.0))
            case StateVector() | DensityMatrix():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Choi(SuperOperator):
    """Choi matrix C.

    Tensor shape: ``(*ensemble, d0_out_bra, d1_out_bra, ..., d0_out_ket, d1_out_ket, ..., d0_in_bra, d1_in_bra, ..., d0_in_ket, d1_in_ket, ...)``

    Matrix shape: ``(*ensemble, d_out^2, d_in^2)``
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
        from ._exponentiation import power_choi

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
                from ._metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case SuperOp() | PauliLiouville() | KrausMap():
                # Compare using process fidelity (handles conversions internally)
                from ._metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case Unitary():
                # Promote Unitary to Choi and compare
                from ._metrics import process_fidelity
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

    Tensor shape: ``(*ensemble, d0_out_bra, d1_out_bra, ..., d0_out_ket, d1_out_ket, ..., d0_in_bra, d1_in_bra, ..., d0_in_ket, d1_in_ket, ...)``

    Matrix shape: ``(*ensemble, d_out^2, d_in^2)``
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

    Tensor shape: ``(*ensemble, d0_out_bra, d1_out_bra, ..., d0_out_ket, d1_out_ket, ..., d0_in_bra, d1_in_bra, ..., d0_in_ket, d1_in_ket, ...)``

    Matrix shape: ``(*ensemble, d_out^2, d_in^2)``
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
        from ._exponentiation import power_pauli_liouville

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
                from ._metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case SuperOp() | Choi() | KrausMap():
                # Compare using process fidelity (handles conversions internally)
                from ._metrics import process_fidelity

                return bool(jnp.allclose(process_fidelity(self, other), 1.0))
            case Unitary():
                # Promote Unitary to PauliLiouville and compare
                from ._metrics import process_fidelity
                from ._superoperator_transformations import unitary_to_pauli_liouville

                return bool(jnp.allclose(process_fidelity(self, unitary_to_pauli_liouville(other)), 1.0))
            case StateVector() | DensityMatrix():
                # States and operators are never equal
                return False
            case _:
                return NotImplemented


# ======================================================================
# Lindbladian
# ======================================================================


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Lindbladian(QuantumObject):
    """Lindbladian generator of a quantum dynamical semigroup.

    Stores the GKSL master-equation ingredients directly: a Hamiltonian ``hamiltonian``
    (:class:`Observable`, or ``None`` for pure dissipation) and the ``jump_operators``
    (:class:`Operator`, with the jump operators stacked along a leading ``n_ops`` axis).
    The d²×d² generator matrix :math:`\\mathcal{L}` is computed on demand from these and cached
    (see :attr:`matrix`).  Because the operators are the source of truth, :meth:`to_operators`
    returns exactly what was supplied — no gauge canonicalization.

    Exponentiating via :func:`evolve` produces a CPTP quantum channel for :math:`t \\geq 0`.
    This is NOT a CPTP map — it is the generator. Use :func:`evolve` to obtain the channel.

    Only physically valid (CPTP-generating) Lindbladians are representable, so operations that
    could produce a non-CP generator — :meth:`__neg__`, :meth:`__sub__`, and multiplication by a
    negative or complex scalar — are not supported.

    Matrix shape: ``(*ensemble, d_out^2, d_in^2)`` (via :attr:`matrix`).
    """

    hamiltonian: "Observable | None"
    jump_operators: "Operator"
    # ``data`` and ``num_qubits`` from the base are redundant here — both are derived from
    # ``jump_operators`` — so drop them as dataclass fields and provide them as properties.
    data: ClassVar[Array]  # pyright: ignore[reportRedeclaration]
    num_qubits: ClassVar[int]  # pyright: ignore[reportRedeclaration]

    # ----- derived metadata (from the stored operators) -----

    @property
    def num_qubits(self) -> int:  # type: ignore[override]
        """The number of qudits, taken from the jump operators."""
        return self.jump_operators.num_qubits

    @property
    def num_ensemble_dims(self) -> int:
        """Leading ensemble dimensions (the jump operators' ``n_ops`` axis is not one of them)."""
        return self.jump_operators.num_ensemble_dims - 1

    @property
    def ensemble_size(self) -> Tuple[int, ...]:
        """Shape of the ensemble (batch) dimensions, or ``()`` for a single generator."""
        return self.jump_operators.ensemble_size[:-1]

    @property
    def dims(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The (output, input) qudit dimensions, taken from the jump operators."""
        return self.jump_operators.dims

    @cached_property
    def matrix(self) -> Array:
        """The generator matrix ``(*ensemble, d_out^2, d_in^2)``, computed once and cached."""
        return _gksl_generator(self.hamiltonian, self.jump_operators)

    @property
    def data(self) -> Array:  # type: ignore[override]
        """Tensor form of the generator ``(*ensemble, out_bra…, out_ket…, in_bra…, in_ket…)``."""
        dims_out, dims_in = self.dims
        ensemble_shape = self.matrix.shape[:-2]
        return self.matrix.reshape(ensemble_shape + dims_out + dims_out + dims_in + dims_in)

    # ----- pytree -----

    def tree_flatten(self):
        return (self.hamiltonian, self.jump_operators), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        hamiltonian, jump_operators = children
        return cls(hamiltonian=hamiltonian, jump_operators=jump_operators)

    # ----- construction -----

    @classmethod
    def from_operators(
        cls,
        hamiltonian: "Observable | None",
        jump_operators: "Operator",
    ) -> "Lindbladian":
        """Construct the GKSL Lindbladian generator from a Hamiltonian and jump operators.

        Rates must be pre-absorbed into ``jump_operators``: pass ``sqrt(γ) * L_physical``.
        Stack multiple jump operators along a leading ``n_ops`` axis.  The operators are stored
        verbatim and are recoverable via :meth:`to_operators`.

        :param hamiltonian: The Hamiltonian (``Observable``), or ``None`` for pure dissipation.
        :param jump_operators: An ``Operator`` with shape ``(*ensemble, n_ops, d, d)``.
        :return: The Lindbladian generator as a :class:`Lindbladian`.
        """
        return cls(hamiltonian=hamiltonian, jump_operators=jump_operators)

    def to_operators(self) -> "Tuple[Observable | None, Operator]":
        """Return the stored ``(hamiltonian, jump_operators)`` — the exact inputs to
        :meth:`from_operators`.

        Unlike a Kossakowski reconstruction, there is no gauge freedom: the physical operators you
        supplied come back unchanged (``hamiltonian`` is ``None`` for a purely dissipative generator).

        :return: ``(hamiltonian, jump_operators)`` with matrix shapes ``(*ensemble, d, d)`` and
            ``(*ensemble, n_ops, d, d)``.
        """
        return self.hamiltonian, self.jump_operators

    # ----- generator algebra -----

    def __add__(self, other: Any) -> "Lindbladian":
        """Add two Lindbladian generators (combine independent noise sources acting in parallel).

        The jump operators are concatenated and the Hamiltonians summed — the operator-level form
        of adding the two generators.
        """
        if not isinstance(other, Lindbladian) or self.dims != other.dims:
            return NotImplemented
        combined_jumps = Operator.from_matrix(
            jnp.concatenate([self.jump_operators.matrix, other.jump_operators.matrix], axis=-3),
            self.jump_operators.dims,
        )
        return Lindbladian.from_operators(_add_hamiltonians(self.hamiltonian, other.hamiltonian), combined_jumps)

    def __radd__(self, other: Any) -> "Lindbladian":
        """Right-hand addition of two Lindbladian generators."""
        if not isinstance(other, Lindbladian) or self.dims != other.dims:
            return NotImplemented
        return other.__add__(self)

    def __or__(self, other: Any) -> "Lindbladian":
        """Tensor product of two Lindbladian generators (independent subsystems).

        ``L_A | L_B`` gives the combined generator for the joint system A⊗B, built at the operator
        level: the jump operators are ``{L_k^A ⊗ I_B} ∪ {I_A ⊗ L_j^B}`` and the Hamiltonian is
        ``H_A ⊗ I_B + I_A ⊗ H_B``, using quax's index-interleaving convention so that
        ``evolve(L_A | L_B, t) == evolve(L_A, t) | evolve(L_B, t)``.
        """
        if not isinstance(other, Lindbladian):
            return NotImplemented
        if self.num_ensemble_dims != 0 or other.num_ensemble_dims != 0:
            raise NotImplementedError("__or__ is not yet implemented for ensemble Lindbladians.")

        from ._tensor import tensor_operator

        dims_A, dims_B = self.dims[0], other.dims[0]
        dA = reduce(mul, dims_A, 1)
        dB = reduce(mul, dims_B, 1)
        I_A = Operator.from_matrix(jnp.eye(dA, dtype=complex), (dims_A, dims_A))
        I_B = Operator.from_matrix(jnp.eye(dB, dtype=complex), (dims_B, dims_B))
        joint_dims = (dims_A + dims_B, dims_A + dims_B)

        # Jump operators embedded into the joint space, then stacked along the n_ops axis.
        jumps_A = tensor_operator(self.jump_operators, I_B)  # {L_k^A ⊗ I_B}
        jumps_B = tensor_operator(I_A, other.jump_operators)  # {I_A ⊗ L_j^B}
        combined_jumps = Operator.from_matrix(jnp.concatenate([jumps_A.matrix, jumps_B.matrix], axis=-3), joint_dims)

        # Joint Hamiltonian H_A ⊗ I_B + I_A ⊗ H_B (skipping absent coherent terms).
        h_terms = []
        if self.hamiltonian is not None:
            h_terms.append(tensor_operator(self.hamiltonian, I_B).matrix)
        if other.hamiltonian is not None:
            h_terms.append(tensor_operator(I_A, other.hamiltonian).matrix)
        hamiltonian = Observable.from_matrix(reduce(jnp.add, h_terms), joint_dims) if h_terms else None

        return Lindbladian.from_operators(hamiltonian, combined_jumps)

    def __sub__(self, other: Any) -> "Lindbladian":
        """Not supported: generator subtraction can yield a non-CP generator (see class docstring)."""
        raise NotImplementedError(
            "Lindbladian subtraction is not supported: the result can be a non-CP generator that "
            "cannot be represented by jump operators. Work with evolve() + superoperator arithmetic "
            "if you need general linear combinations."
        )

    def __neg__(self) -> "Lindbladian":
        """Not supported: negating a generator yields a non-CP generator (see class docstring)."""
        raise NotImplementedError(
            "Negating a Lindbladian is not supported: -L is a non-CP generator that cannot be "
            "represented by jump operators."
        )

    # ----- conjugation / indexing / display -----

    def conj(self) -> "Lindbladian":
        """Complex conjugate (conjugates the Hamiltonian and jump operators)."""
        hamiltonian = self.hamiltonian.conj() if self.hamiltonian is not None else None
        return Lindbladian(hamiltonian=hamiltonian, jump_operators=self.jump_operators.conj())

    def __getitem__(self, key: Any) -> "Lindbladian":
        if self.num_ensemble_dims == 0:
            raise IndexError("This Lindbladian is not ensembled (no ensemble dimensions), so it cannot be indexed.")
        hamiltonian = self.hamiltonian[key] if self.hamiltonian is not None else None
        return Lindbladian(hamiltonian=hamiltonian, jump_operators=self.jump_operators[key])

    def __str__(self) -> str:
        n_ops = self.jump_operators.matrix.shape[-3]
        if self.ensemble_size != ():
            return f"Lindbladian(dims={self.dims}, ensemble_size={self.ensemble_size}, n_jump_operators={n_ops})"
        return f"Lindbladian(dims={self.dims}, n_jump_operators={n_ops})"

    def __eq__(self, other: Any) -> bool:
        """Element-wise equality of the generators (gauge-invariant: compares ``matrix``)."""
        if not isinstance(other, Lindbladian):
            return NotImplemented
        if self.dims != other.dims:
            return False
        return bool(jnp.allclose(self.matrix, other.matrix))


def _add_hamiltonians(h1: "Observable | None", h2: "Observable | None") -> "Observable | None":
    """Sum two optional Hamiltonians, treating ``None`` as the zero operator."""
    if h1 is None:
        return h2
    if h2 is None:
        return h1
    return Observable.from_matrix(h1.matrix + h2.matrix, h1.dims)


@jax.jit
def _gksl_generator(hamiltonian: "Observable | None", jump_operators: "Operator") -> Array:
    """GKSL generator matrix ``-i[H,ρ] + Σ_k D[L_k]`` as ``(*ensemble, d², d²)``.

    Rates are pre-absorbed into ``jump_operators``. Engine for :attr:`Lindbladian.matrix`.
    """
    dims = jump_operators.dims
    d = reduce(mul, dims[1], 1)
    I = jnp.eye(d, dtype=complex)
    L = jump_operators.matrix  # (*ensemble, n_ops, d, d)
    ensemble_shape = L.shape[:-3]

    # Build the GKSL generator as a rank-4 tensor with indices
    # [out_bra a, out_ket c, in_bra b, in_ket d], reshaped at the end to a (d², d²)
    # superoperator acting on vec(ρ).
    #
    # Tensor products via einsum: an einsum whose output indices are the union of the
    # (disjoint) input indices, with none summed away, is an outer/tensor product.  E.g.
    # ``einsum("ab,cd->acbd", A, B)`` computes A ⊗ B and then interleaves the axes so that
    # (a, c) form the superoperator's row multi-index and (b, d) its column multi-index —
    # exactly the layout a (d², d²) matrix on vec(ρ) needs.  ``einsum("...kab,...kcd->...acbd")``
    # is the same tensor product but additionally summed over the jump index k.

    # No-jump rate operator  G = Σ_k L_k† L_k  (Hermitian):
    #   G_{ab} = Σ_{k,c} conj(L_k)_{ca} (L_k)_{cb}
    G = jnp.einsum("...kca,...kcb->...ab", jnp.conj(L), L)
    # Jump (dissipative gain) term  Σ_k L_k ρ L_k†  — tensor product Σ_k conj(L_k) ⊗ L_k:
    #   sandwich_{acbd} = Σ_k conj(L_k)_{ab} (L_k)_{cd}
    sandwich = jnp.einsum("...kab,...kcd->...acbd", jnp.conj(L), L)
    # Two δ-structured halves of the anticommutator  −½{G, ρ} = −½(G ρ + ρ G), each an I ⊗ G tensor product:
    #   G_rho_{acbd} = δ_{ab} G_{cd}          (the G ρ half)
    G_rho = jnp.einsum("ab,...cd->...acbd", I, G)
    #   rho_G_{acbd} = conj(G)_{ab} δ_{cd}    (the ρ G half; G Hermitian ⇒ conj(G)_{ab} = G_{ba})
    rho_G = jnp.einsum("...ab,cd->...acbd", jnp.conj(G), I)
    # Dissipator  D[ρ] = Σ_k ( L_k ρ L_k† − ½{L_k† L_k, ρ} )
    gen_data = sandwich - 0.5 * G_rho - 0.5 * rho_G

    if hamiltonian is not None:
        H = hamiltonian.matrix
        # Two δ-structured halves of the commutator  −i[H, ρ] = −i(H ρ − ρ H), each an I ⊗ H tensor product:
        #   H_rho_{acbd} = δ_{ab} H_{cd}          (the H ρ half)
        H_rho = jnp.einsum("ab,...cd->...acbd", I, H)
        #   rho_H_{acbd} = conj(H)_{ab} δ_{cd}    (the ρ H half; H Hermitian ⇒ conj(H)_{ab} = H_{ba})
        rho_H = jnp.einsum("...ab,cd->...acbd", jnp.conj(H), I)
        gen_data = gen_data + (-1j * (H_rho - rho_H))

    return gen_data.reshape(ensemble_shape + (d * d, d * d))


# ======================================================================
# QuantumInstrument
# ======================================================================


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QuantumInstrument(QuantumObject):
    """A quantum instrument modeling mid-circuit measurement.

    Stores per-outcome superoperator matrices (CP but not TP) whose sum is CPTP,
    plus metadata about which qudits produce classical output.

    Data tensor shape:
        ``(*ensemble, num_outcomes, d_out_0, …, d_out_{n-1},
           d_out_0, …, d_out_{n-1}, d_in_0, …, d_in_{n-1},
           d_in_0, …, d_in_{n-1})``

    The first qudit axis after ``num_outcomes`` is the outcome axis.
    """

    measured_qudits: Tuple[int, ...]
    """Indices of the qudits that produce a classical outcome."""

    # ---- pytree support ----

    def tree_flatten(self):
        return (self.data,), (self.num_qubits, self.measured_qudits)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (data,) = children
        num_qubits, measured_qudits = aux
        return cls(data=data, num_qubits=num_qubits, measured_qudits=measured_qudits)

    # ---- core properties ----

    @property
    def num_outcomes(self) -> int:
        """Number of classical measurement outcomes."""
        return self.data.shape[self.num_ensemble_dims]

    @property
    def num_ensemble_dims(self) -> int:
        """Number of leading ensemble / batch dimensions."""
        n_qudit = self.num_qubits
        qudit_dims = 4 * n_qudit  # bra/ket for output & input
        return self.data.ndim - 1 - qudit_dims  # subtract 1 for outcome axis

    @property
    def ensemble_size(self) -> Tuple[int, ...]:
        return self.data.shape[: self.num_ensemble_dims]

    @property
    def dims(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        qudit_shape = self.data.shape[self.num_ensemble_dims + 1 :]
        n = len(qudit_shape) // 4
        dims_out = qudit_shape[:n]
        dims_in = qudit_shape[2 * n : 3 * n]
        return (dims_out, dims_in)

    @property
    def measured_dims(self) -> Tuple[int, ...]:
        """Per-qudit dimensions of the measured subsystems."""
        return tuple(self.dims[0][i] for i in self.measured_qudits)

    @property
    def d_measured(self) -> int:
        """Total Hilbert-space dimension of the measured subsystems."""
        return reduce(mul, self.measured_dims, 1)

    @property
    def d(self) -> Tuple[int, int]:
        """Total (output, input) Hilbert-space dimensions."""
        return tuple(reduce(mul, dim) for dim in self.dims)  # type: ignore[return-value]

    @property
    def d2(self) -> Tuple[int, int]:
        """Squared (output, input) dimensions."""
        return self.d[0] ** 2, self.d[1] ** 2

    @property
    def matrix(self) -> Array:
        """Flattened superoperator matrices: ``(*ensemble, num_outcomes, d_out², d_in²)``."""
        ensemble_shape = self.data.shape[: self.num_ensemble_dims]
        n_outcomes = self.num_outcomes
        qudit_shape = self.data.shape[self.num_ensemble_dims + 1 :]
        n = len(qudit_shape) // 4
        d_out_bra = reduce(mul, qudit_shape[:n], 1)
        d_out_ket = reduce(mul, qudit_shape[n : 2 * n], 1)
        d_in_bra = reduce(mul, qudit_shape[2 * n : 3 * n], 1)
        d_in_ket = reduce(mul, qudit_shape[3 * n :], 1)
        d_out = d_out_bra * d_out_ket
        d_in = d_in_bra * d_in_ket
        return self.data.reshape(ensemble_shape + (n_outcomes, d_out, d_in))

    @classmethod
    def from_matrix(
        cls,
        matrix: Array,
        dims: Tuple[Tuple[int, ...], Tuple[int, ...]],
        measured_qudits: Tuple[int, ...],
    ) -> "QuantumInstrument":
        """Construct from flattened superoperator matrices.

        :param matrix: ``(*ensemble, num_outcomes, d_out², d_in²)``
        :param dims: ``(dims_out, dims_in)`` per-qudit dimensions.
        :param measured_qudits: Indices of measured qudits.
        """
        num_qubits = len(dims[0])
        ensemble_shape = matrix.shape[:-3]
        n_outcomes = matrix.shape[-3]
        tensor_shape = dims[0] + dims[0] + dims[1] + dims[1]
        tensor = matrix.reshape(ensemble_shape + (n_outcomes,) + tensor_shape)
        return cls(data=tensor, num_qubits=num_qubits, measured_qudits=measured_qudits)

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------

    def outcome_superop(self, i: int) -> Tuple["SuperOp", Array]:
        """Return the superoperator for outcome *i* and its normalization coefficient.

        Per-outcome maps are CP but *not* TP, so the trace of
        ``E_i(ρ)`` gives the probability of outcome *i*.  The returned
        coefficient is that probability for a maximally mixed input,
        i.e. ``Tr[S_i] / d_in``.

        :returns: ``(superop, coeff)`` where *superop* is the (un-normalised)
            superoperator and *coeff* is the scalar normalization factor.
        """
        mat = self.matrix[..., i, :, :]
        superop = SuperOp.from_matrix(mat, self.dims)
        coeff = jnp.real(jnp.trace(mat, axis1=-2, axis2=-1)) / self.d[1]
        return superop, coeff

    def total_channel(self) -> "SuperOp":
        """Return the CPTP channel obtained by summing over all outcomes."""
        total = jnp.sum(self.matrix, axis=-3)
        return SuperOp.from_matrix(total, self.dims)

    # ------------------------------------------------------------------
    # Ensemble indexing
    # ------------------------------------------------------------------

    def __getitem__(self, key: Any) -> "QuantumInstrument":
        if self.num_ensemble_dims == 0:
            raise IndexError("This QuantumInstrument has no ensemble dimensions.")
        new_data = self.data[key]
        obj = QuantumInstrument(data=new_data, num_qubits=self.num_qubits, measured_qudits=self.measured_qudits)
        qubit_dims = self.data.shape[self.num_ensemble_dims :]
        if obj.data.shape[obj.num_ensemble_dims :] != qubit_dims:
            raise IndexError("Indexing removed quantum dimensions.")
        return obj

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_superop(
        cls,
        superop_matrices: Sequence["SuperOp"],
        measured_qudits: Tuple[int, ...],
    ) -> "QuantumInstrument":
        """Construct from a sequence of superoperator matrices (one per outcome).

        :param superop_matrices: CP maps, one per measurement outcome.  Their
            sum must be trace-preserving.
        :param measured_qudits: Indices of the qudits that are measured.
        """
        if len(superop_matrices) == 0:
            raise ValueError("At least one superoperator matrix is required.")

        dims = superop_matrices[0].dims
        for j, s in enumerate(superop_matrices):
            if s.dims != dims:
                raise ValueError(f"Superoperator {j} has dims {s.dims}, expected {dims}.")

        matrices = jnp.stack([s.matrix for s in superop_matrices], axis=-3)
        return cls.from_matrix(matrices, dims, measured_qudits)

    # ------------------------------------------------------------------
    # Properties: confusion matrix, transition matrix
    # ------------------------------------------------------------------

    @property
    def confusion_matrix(self) -> Array:
        r"""
        Extract the confusion matrix.

        Shape ``(*ensemble, num_outcomes, d_measured)``.  Entry ``[i, j]`` is the
        probability of reporting outcome *i* when the measured subsystem is in
        computational basis state *j*.

        In the Liouville representation, applying superoperator :math:`S_i` to a
        computational basis state :math:`|j\\rangle\\langle j|` reduces to selecting
        column :math:`j(d+1)` of :math:`S_i`, because
        :math:`\\operatorname{vec}(|j\\rangle\\langle j|)` is a unit vector at that
        position.  Taking the trace then sums over the diagonal rows:

        .. math::

            C[i, j] = \\operatorname{Tr}[\\mathcal{E}_i(|j\\rangle\\langle j|)]
                     = \\sum_k S_i[k(d+1),\\, j(d+1)]

        When not all qudits are measured, entries are averaged over the unmeasured subsystem
        states that share the same measured-subsystem index.
        """
        d_total = self.d[0]
        d_measured = self.d_measured
        dims = self.dims[0]

        # Diagonal positions in the d_total^2-dimensional Liouville space.
        # vec(|j><j|) is non-zero only at index j*(d_total+1).
        diag_idx = jnp.arange(d_total) * (d_total + 1)  # (d_total,)

        # raw_probs[..., i, j_full] = Tr[E_i(|j_full><j_full|)]
        # self.matrix: (*ensemble, n_outcomes, d^2, d^2)
        # submatrix at diagonal rows and cols: (*ensemble, n_outcomes, d_total, d_total)
        # sum over k (output diagonal): (*ensemble, n_outcomes, d_total)
        raw_probs = jnp.real(self.matrix[..., diag_idx[:, None], diag_idx[None, :]].sum(axis=-2))

        # Map each full-space index j_full to its measured-subsystem index j_meas.
        j_meas_array = jnp.array(
            [_extract_measured_index(j, dims, self.measured_qudits) for j in range(d_total)]
        )  # (d_total,)

        # Sum contributions into measured-subsystem columns, then normalize.
        # one_hot: (d_total, d_measured); matmul broadcasts over leading ensemble/outcome dims.
        n_per_meas = d_total // d_measured
        one_hot = jax.nn.one_hot(j_meas_array, d_measured)  # (d_total, d_measured)
        return raw_probs @ one_hot / n_per_meas  # (*ensemble, n_outcomes, d_measured)

    @property
    def transition_matrix(self) -> Array:
        """Extract the transition matrix over the full Hilbert space.

        Shape ``(d_total, d_total)``.  Entry ``[k, j]`` is the probability of
        ending in computational basis state *k* given input *j*, marginalised
        over all measurement outcomes.
        """
        from ._apply import apply_superop_to_density_matrix

        d = self.d[0]
        dims = self.dims[0]
        total_superop = SuperOp.from_matrix(jnp.sum(self.matrix, axis=-3), self.dims)

        transition = jnp.zeros((d, d))
        for j in range(d):
            rho_j = DensityMatrix.from_matrix(jnp.zeros((d, d), dtype=jnp.complex128).at[j, j].set(1.0), dims)
            rho_out = apply_superop_to_density_matrix(total_superop, rho_j)
            for k in range(d):
                transition = transition.at[k, j].set(jnp.real(rho_out.matrix[k, k]))

        return transition

    # ------------------------------------------------------------------
    # Composition and tensor product operators
    # ------------------------------------------------------------------

    def __matmul__(self, other: Any) -> Any:
        """Compose two instruments (or apply to a state)."""
        match other:
            case QuantumInstrument():
                from ._compose import compose_instrument

                return compose_instrument(self, other)
            case DensityMatrix():
                raise TypeError(
                    "Use select_outcome(*apply_instrument_to_density_matrix(instrument, rho), key) "
                    "to apply a QuantumInstrument to a DensityMatrix."
                )
            case _:
                return NotImplemented

    def __or__(self, other: Any) -> Any:
        """Tensor product of two instruments."""
        match other:
            case QuantumInstrument():
                from ._tensor import tensor_instrument

                return tensor_instrument(self, other)
            case _:
                return NotImplemented

    def __ror__(self, other: Any) -> Any:
        match other:
            case QuantumInstrument():
                from ._tensor import tensor_instrument

                return tensor_instrument(other, self)
            case _:
                return NotImplemented

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mq = ",".join(str(q) for q in self.measured_qudits)
        return f"QuantumInstrument(dims={self.dims}, num_outcomes={self.num_outcomes}, measured_qudits=({mq}))"


# ======================================================================
# QuantumInstrument private helpers
# ======================================================================


def _decode_index(flat: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Decode a flat index into a multi-dimensional index (row-major)."""
    indices: list[int] = []
    for s in reversed(shape):
        indices.append(flat % s)
        flat //= s
    return tuple(reversed(indices))


def _encode_index(indices: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    """Encode a multi-dimensional index into a flat index (row-major)."""
    flat = 0
    for idx, s in zip(indices, shape):
        flat = flat * s + idx
    return flat


def _build_partial_projector(
    dims: Tuple[int, ...],
    measured_qudits: Tuple[int, ...],
    measured_values: Tuple[int, ...],
) -> Array:
    """Build a projector that fixes measured qudits to given values and acts
    as identity on unmeasured qudits.
    """
    d_total = reduce(mul, dims, 1)
    projector = jnp.zeros((d_total, d_total), dtype=jnp.complex128)
    for idx in range(d_total):
        qudit_indices = _decode_index(idx, dims)
        match = all(qudit_indices[mq] == mv for mq, mv in zip(measured_qudits, measured_values))
        if match:
            projector = projector.at[idx, idx].set(1.0)
    return projector


def _extract_measured_index(
    full_index: int,
    dims: Tuple[int, ...],
    measured_qudits: Tuple[int, ...],
) -> int:
    """Given a full computational-basis index, extract the measured subsystem index."""
    qudit_indices = _decode_index(full_index, dims)
    measured_indices = tuple(qudit_indices[mq] for mq in measured_qudits)
    measured_dims = tuple(dims[mq] for mq in measured_qudits)
    return _encode_index(measured_indices, measured_dims)


def _count_full_states_per_measured(
    j_meas: int,
    dims: Tuple[int, ...],
    measured_qudits: Tuple[int, ...],
) -> int:
    """Count how many full-space basis states map to a given measured-subsystem index."""
    d_total = reduce(mul, dims, 1)
    d_measured = reduce(mul, (dims[mq] for mq in measured_qudits), 1)
    return d_total // d_measured
