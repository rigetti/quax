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
from typing import TYPE_CHECKING, Any, Iterator, Self, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Array as JaxArray
from jaxtyping import Complex, Float

if TYPE_CHECKING:
    import qutip
    from numpy.typing import NDArray


# ---------- base mixin ----------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class State:
    """Base class for a quantum state."""

    data: (
        Array  # Complex[JaxArray, "*ensemble d"] for StateVector, Complex[JaxArray, "*ensemble d d"] for DensityMatrix
    )
    """The operator or state."""

    dims: Tuple[int, ...]
    """The dimensions of each qudit."""

    def __neg__(self) -> Self:
        """Multiply the unitary by -1."""
        return type(self)(-self.data, self.dims)

    def __mul__(self, scalar: complex) -> Self:
        """Scalar multiplication of the superoperator."""
        return type(self)(self.data * scalar, self.dims)

    def __rmul__(self, scalar: complex) -> Self:
        """Scalar multiplication of the superoperator."""
        return self * scalar

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the state"""
        raise NotImplementedError(f"Exponentiation not implemented for {type(self)}.")

    def __str__(self) -> str:
        if self.ensemble_size != ():
            return (
                f"{type(self).__name__}(dims={self.dims}, ensemble_size={self.ensemble_size}, shape={self.data.shape})"
            )
        return f"{type(self).__name__}(dims={self.dims}, shape={self.data.shape})"

    def __eq__(self, other: Any) -> bool:
        """Check equality of two Operators."""
        # This eq method will never be _wrong_ but some operators are equivalent
        # up to a global phase or other transformations
        if not isinstance(other, type(self)):
            return False
        if self.dims != other.dims:
            return False
        elif jnp.allclose(self.data, other.data):
            return True
        else:
            return False

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication of the operator with another operator."""
        # composition is very different between different operator types and should be delegated
        raise NotImplementedError(f"Matrix multiplication not implemented between {type(self)} and {type(other)}.")

    def __or__(self, other: Any) -> Any:
        """Tensor product of the operator with another operator."""
        # tensor product is very differentse between different operator types and should be delegated
        raise NotImplementedError(f"Tensor product not implemented between {type(self)} and {type(other)}.")

    def tree_flatten(self):
        return (self.data,), self.dims

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (data,) = children
        return cls(data=data, dims=aux_data)

    def conj(self):
        """Complex conjugate of the states(s)"""
        return type(self)(data=jnp.conjugate(self.data), dims=self.dims)

    @property
    def T(self):
        """Transpose of the operator(s)"""
        raise NotImplementedError(f"Transpose not implemented for {type(self)}.")

    @property
    def h(self):
        """Hermitian conjugate of the operator(s)"""
        raise NotImplementedError(f"Hermitian conjugate not implemented for {type(self)}.")

    @property
    def d(self) -> int:
        return reduce(mul, self.dims)

    @property
    def d2(self) -> int:
        return self.d**2

    @property
    def ensemble_size(self) -> Tuple[int, ...]:
        """Returns the size of the ensemble if the state represents an ensemble of states."""
        raise NotImplementedError("Subclasses must implement ensemble_size property.")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Operator:
    """Base class for a quantum operator."""

    data: Array  # Complex[JaxArray, "*ensemble d_out d_in"] for most operators
    """The operator or state."""

    dims: Tuple[Tuple[int, ...], Tuple[int, ...]]
    """The (input, output) dimensions of each qudit."""

    def __neg__(self) -> Self:
        """Multiply the unitary by -1."""
        return type(self)(-self.data, self.dims)

    def __mul__(self, scalar: complex) -> Self:
        """Scalar multiplication of the superoperator."""
        return type(self)(self.data * scalar, self.dims)

    def __rmul__(self, scalar: complex) -> Self:
        """Scalar multiplication of the superoperator."""
        return self * scalar

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the operator"""
        # By default, only support integer exponents
        # matrix_power is not _correct_ for all operators
        if jnp.isclose(int(jnp.abs(exponent)), exponent):
            new_data = jnp.linalg.matrix_power(self.data, exponent)
            return type(self)(new_data, self.dims)
        else:
            raise TypeError(f"Exponent must be an integer, but got {type(exponent)}.")

    def __str__(self) -> str:
        if self.ensemble_size != ():
            return (
                f"{type(self).__name__}(dims={self.dims}, ensemble_size={self.ensemble_size}, shape={self.data.shape})"
            )
        return f"{type(self).__name__}(dims={self.dims}, shape={self.data.shape})"

    def __eq__(self, other: Any) -> bool:
        """Check equality of two Operators."""
        # This eq method will never be _wrong_ but some operators are equivalent
        # up to a global phase or other transformations
        if not isinstance(other, type(self)):
            return False
        if self.dims != other.dims:
            return False
        elif jnp.allclose(self.data, other.data):
            return True
        else:
            return False

    def __rmatmul__(self, other: Any) -> Any:
        """Matrix multiplication of the operator with another operator."""
        # composition is very different between different operator types and should be delegated
        raise NotImplementedError(f"Matrix multiplication not implemented between {type(self)} and {type(other)}.")

    def __or__(self, other: Any) -> Any:
        """Tensor product of the operator with another operator."""
        # tensor product is very different between different operator types and should be delegated
        raise NotImplementedError(f"Tensor product not implemented between {type(self)} and {type(other)}.")

    def tree_flatten(self):
        return (self.data,), self.dims

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (data,) = children
        return cls(data=data, dims=aux_data)

    def conj(self):
        """Complex conjugate of the operator(s)"""
        return type(self)(data=jnp.conjugate(self.data), dims=self.dims)

    @property
    def T(self):
        """Transpose of the operator(s)"""
        # use the swapaxes function to swap the last two axes
        return type(self)(data=jnp.swapaxes(self.data, -1, -2), dims=(self.dims[1], self.dims[0]))

    @property
    def h(self):
        """Hermitian conjugate of the operator(s)"""
        return type(self)(data=jnp.conjugate(jnp.swapaxes(self.data, -1, -2)), dims=(self.dims[1], self.dims[0]))

    @property
    def d(self) -> Tuple[int, int]:
        return tuple(reduce(mul, dim) for dim in self.dims)

    @property
    def d2(self) -> Tuple[int, int]:
        return self.d[0] ** 2, self.d[1] ** 2

    @property
    def ensemble_size(self) -> Tuple[int, ...]:
        """Returns the size of the ensemble if the operator represents an ensemble of operators."""
        if len(self.data.shape) > 2:
            return self.data.shape[:-2]
        else:
            return ()

    def __iter__(self) -> Iterator[Self]:
        if self.data.ndim == 2:
            raise TypeError(
                "This Operator is not batched (data.ndim == 2), so it cannot be iterated. "
                "If you intended a batch, make data shape (N, d_out, d_in) (or higher-rank)."
            )
        # iterate over axis 0 only; remaining batch axes (if any) remain on each item
        for i in range(self.data.shape[0]):
            yield type(self)(data=self.data[i], dims=self.dims)


# This class is basically for typing purposes
# Some methods works on any sort of Superoperator
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SuperOperator(Operator):
    """Base class for a quantum superoperator."""

    pass


# ---------- states ----------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StateVector(State):
    """State vector |psi>, shape (*ensemble, d)."""

    data: Complex[JaxArray, "*ensemble d"]

    @property
    def ensemble_size(self) -> Tuple[int, ...]:
        """Returns the size of the ensemble if the state represents an ensemble of states."""
        if len(self.data.shape) > 1:
            return self.data.shape[:-1]
        else:
            return ()

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
                return jnp.einsum("...a,...a->...", self.data.conj(), other.data)
            case DensityMatrix():  #  <𝜓|𝜌 -> <𝜙|
                return StateVector(data=jnp.einsum("...b,...ba->...a", self.data.conj(), other.data), dims=self.dims)
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

        dims = [list(self.dims), [1]]

        if self.ensemble_size == ():
            # Scalar case - return single Qobj
            return qt.Qobj(self.data, dims=dims)

        flat_shape = (-1,) + self.data.shape[-1:]
        flat_unitary = self.data.reshape(flat_shape)
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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DensityMatrix(State):
    """Density matrix ρ, shape (*ensemble, d, d)."""

    data: Complex[JaxArray, "*ensemble d d"]

    @property
    def ensemble_size(self) -> Tuple[int, ...]:
        """Returns the size of the ensemble if the state represents an ensemble of states."""
        if len(self.data.shape) > 2:
            return self.data.shape[:-2]
        else:
            return ()

    @property
    def T(self):
        """Transpose of the operator(s)"""
        return type(self)(data=jnp.swapaxes(self.data, -1, -2), dims=(self.dims[1], self.dims[0]))

    @property
    def h(self):
        """Hermitian conjugate of the operator(s)"""
        return type(self)(data=jnp.conjugate(jnp.swapaxes(self.data, -1, -2)), dims=(self.dims[1], self.dims[0]))

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the density matrix using eigendecomposition (batch-compatible)."""
        from ._power import density_matrix_power

        return density_matrix_power(self, exponent)

    def __matmul__(self, other):
        """Left multiply the density matrix by another object."""
        match other:
            case StateVector():  # 𝜌|𝜓> -> |𝜙>
                return StateVector(data=jnp.einsum("...ab,...b->...a", self.data, other.data), dims=other.dims)
            case DensityMatrix():  # 𝜌𝜎 -> 𝜏
                return DensityMatrix(data=jnp.einsum("...ab,...bc->...ac", self.data, other.data), dims=self.dims)
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

        if self.ensemble_size == ():
            # Scalar case - return single density-matrix Qobj
            return qt.Qobj(np.asarray(self.data), dims=dims)

        # Ensemble case - self.data has shape ensemble_size + (dim, dim)
        flat_shape = (-1,) + self.data.shape[-2:]
        flat_rhos = np.asarray(self.data).reshape(flat_shape)

        qobjs = np.asarray(
            [qt.Qobj(rho, dims=dims) for rho in flat_rhos],
            dtype=object,
        )
        return qobjs.reshape(self.ensemble_size)


# ---------- operators ----------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Unitary(Operator):
    """Unitary operator U, shape (*ensemble, d_out, d_in)."""

    data: Complex[JaxArray, "*ensemble d_out d_in"]

    def _to_qobj(self) -> "qutip.Qobj | NDArray":
        """Convert to a QuTiP Qobj for interoperability testing."""
        import numpy as np
        import qutip as qt

        dims = [list(self.dims[0]), list(self.dims[1])]

        if self.ensemble_size == ():
            return qt.Qobj(
                np.array(self.data),
                dims=dims,
            )

        flat_shape = (-1,) + self.data.shape[-2:]
        flat_unitary = self.data.reshape(flat_shape)
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

    def __mul__(self, scalar: complex) -> "Unitary | Kraus":
        """Scalar multiplication of the unitary."""
        if jnp.isclose(jnp.abs(scalar), 1.0):
            return Unitary(self.data * scalar, self.dims)
        else:
            return Kraus(self.data * scalar, self.dims)

    def __rmul__(self, scalar: complex) -> "Unitary | Kraus":
        """Scalar multiplication of the unitary."""
        return self * scalar

    def __pow__(self, exponent: float) -> "Unitary":
        """Exponentiation of the unitary using eigendecomposition (batch-compatible)."""
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
                from ._compose import compose_kraus
                from ._superoperator_transformations import unitary_to_kraus

                return compose_kraus(other, unitary_to_kraus(self))
            case StateVector():
                # <psi|U = <phi| -> StateVector (apply unitary to state vector)
                from ._apply import apply_unitary_to_state_vector

                return apply_unitary_to_state_vector(self, other)
            case DensityMatrix():
                # ρU is an Operator product
                from ._apply import apply_superop_to_density_matrix
                from ._superoperator_transformations import unitary_to_superop

                return apply_superop_to_density_matrix(unitary_to_superop(self), other)
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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Kraus(Operator):
    """Kraus operator, shape (*ensemble, d_out, d_in)."""

    data: Complex[JaxArray, "*ensemble d_out d_in"]

    def _to_qobj(self) -> "qutip.Qobj | NDArray":
        """Convert to a QuTiP Qobj for interoperability testing."""
        import numpy as np
        import qutip as qt

        if self.ensemble_size == ():
            return qt.Qobj(
                np.array(self.data),
                dims=[[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]],
            )

        flat_shape = (-1,) + self.data.shape[-2:]
        flat_kraus = self.data.reshape(flat_shape)
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


# ---------- superoperators ----------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SuperOp(SuperOperator):
    """SuperOp matrix (also known as Superoperator) S, shape (*ensemble, d2, d2)."""

    data: Complex[JaxArray, "*ensemble d2 d2"]

    def _to_qobj(self) -> "qutip.Qobj | NDArray":
        """Convert to a QuTiP Qobj for interoperability testing."""
        import numpy as np
        import qutip as qt

        if self.ensemble_size == ():
            return qt.Qobj(
                np.array(self.data),
                dims=[[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]],
                superrep="super",
            )

        flat_shape = (-1,) + self.data.shape[-2:]
        flat_superop = self.data.reshape(flat_shape)
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
        """Exponentiation of the superoperator using eigendecomposition (batch-compatible)."""
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
                from ._compose import compose_kraus
                from ._superoperator_transformations import superop_to_kraus

                return compose_kraus(superop_to_kraus(self), other)
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
    """Kraus channel, shape (*ensemble, n_kraus, d_out, d_in)."""

    data: Complex[JaxArray, "*ensemble n_kraus d_out d_in"]

    def _to_qobj(self) -> list["qutip.Qobj"]:
        """Convert to a QuTiP Qobj for interoperability testing.

        Returns a list of QuTiP Qobjs, one for each Kraus operator.
        """
        import numpy as np
        import qutip as qt

        # KrausMap is always a batch of Kraus operators, shape (K, d_out, d_in)
        return [qt.Qobj(np.array(k), dims=[[list(self.dims[0])], [list(self.dims[1])]]) for k in self.data]

    @property
    def ensemble_size(self) -> Tuple[int, ...]:
        """Returns the size of the ensemble if the operator represents an ensemble of operators."""
        if len(self.data.shape) > 3:
            return self.data.shape[:-3]
        else:
            return ()

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the Kraus channel using eigendecomposition (batch-compatible)."""
        from ._power import power_kraus

        return power_kraus(self, exponent)

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication of the Kraus channel with another superoperator."""
        match other:
            case KrausMap():
                # K1 @ K2 -> KrausMap (composition)
                from ._compose import compose_kraus

                return compose_kraus(self, other)
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
                from ._compose import compose_kraus
                from ._superoperator_transformations import unitary_to_kraus

                return compose_kraus(self, unitary_to_kraus(other))
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
    """Choi matrix C, shape (*ensemble, d2, d2)."""

    data: Complex[JaxArray, "*ensemble d2 d2"]
    dims: Tuple[Tuple[int, ...], Tuple[int, ...]]

    def _to_qobj(self) -> "qutip.Qobj | NDArray":
        """Convert to a QuTiP Qobj for interoperability testing."""
        import numpy as np
        import qutip as qt

        if self.ensemble_size == ():
            return qt.Qobj(
                np.array(self.data),
                dims=[[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]],
                superrep="choi",
            )

        flat_shape = (-1,) + self.data.shape[-2:]
        flat_choi = self.data.reshape(flat_shape)
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
        """Exponentiation of the Choi matrix using eigendecomposition (batch-compatible)."""
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
                from ._compose import compose_kraus
                from ._superoperator_transformations import choi_to_kraus

                return compose_kraus(choi_to_kraus(self), other)
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
    """Chi matrix Χ, shape (d^2, d^2)."""

    def _to_qobj(self) -> "qutip.Qobj | list[qutip.Qobj]":
        """Convert to a QuTiP Qobj for interoperability testing."""
        import numpy as np
        import qutip as qt

        if self.ensemble_size != ():
            # Batched Chi matrices - return list
            chi_qobjs = []
            for c in self.data:
                base_qobj = qt.Qobj(
                    np.array(c),
                    dims=[[list(self.dims[0]), list(self.dims[0])], [list(self.dims[1]), list(self.dims[1])]],
                )
                chi_qobjs.append(qt.to_chi(base_qobj))
            return chi_qobjs
        # Single Chi matrix
        base_qobj = qt.Qobj(
            np.array(self.data),
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
    """Pauli-Liouville matrix P, shape (*ensemble, d2, d2)."""

    data: Float[JaxArray, "*ensemble d2 d2"]
    dims: Tuple[Tuple[int, ...], Tuple[int, ...]]

    def _to_qobj(self):
        """Convert to a QuTiP Qobj for interoperability testing.

        Note: QuTiP doesn't have native Pauli-Liouville representation,
        so this will look like a SuperOp Qobj, but it will be the Pauli-Liouville matrix.
        """
        raise NotImplementedError("Conversion to QuTiP Qobj not implemented for PauliLiouville.")

    def __pow__(self, exponent: float) -> Self:
        """Exponentiation of the Pauli-Liouville matrix using eigendecomposition (batch-compatible)."""
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
                from ._compose import compose_kraus
                from ._superoperator_transformations import pauli_liouville_to_kraus

                return compose_kraus(pauli_liouville_to_kraus(self), other)
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
