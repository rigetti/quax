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

"""Promotion utilities for embedding quantum objects in larger Hilbert spaces."""

from functools import partial, reduce, singledispatch
from operator import mul
from typing import Tuple, TypeVar, cast

import jax
import jax.numpy as jnp

from ._quantum_objects import (
    Choi,
    DensityMatrix,
    KrausMap,
    Operator,
    PauliLiouville,
    StateVector,
    SuperOp,
    Unitary,
    SuperOperator,
)
from ._superoperator_transformations import (
    choi_to_kraus,
    kraus_to_choi,
    kraus_to_pauli_liouville,
    kraus_to_superop,
    pauli_liouville_to_kraus,
    superop_to_kraus,
    unitary_to_superop,
)


def promote_state_vector_to_density_matrix(
    state: StateVector,
) -> DensityMatrix:
    """
    Promote a state vector to a density matrix.

    :param state: State vector to promote.
    :return: Density matrix corresponding to the state vector.
    """
    state_vec = state.matrix  # shape (*ensemble, d)
    rho_matrix = jnp.einsum("...a,...b->...ab", state_vec, jnp.conj(state_vec))
    return DensityMatrix.from_matrix(rho_matrix, state.dims)


def _validate_promote_dims(current_dims, target_dims):
    """
    Validate that target dims are compatible with current dims for promotion.

    Each target dimension must be >= the corresponding current dimension,
    and the number of subsystems must match.
    """
    if len(target_dims) != len(current_dims):
        raise ValueError(
            f"Number of subsystems must match: got {len(current_dims)} subsystems but target has {len(target_dims)}."
        )
    for i, (c, t) in enumerate(zip(current_dims, target_dims)):
        if t < c:
            raise ValueError(
                f"Target dimension {t} at subsystem {i} is smaller than "
                f"current dimension {c}. Use demote for truncation."
            )


# ---------------------------------------------------------------------------
# Single-dispatch ``promote`` and type-specific implementations
# ---------------------------------------------------------------------------


@singledispatch
def promote(obj, dims):
    """Embed a quantum object into a larger Hilbert space.

    The object is placed in the low-index subspace of the target space;
    higher basis states act as identity / zero as appropriate for the
    object type.  The embedding respects the tensor product structure,
    promoting each qudit independently.

    :param obj: A quax quantum object (StateVector, DensityMatrix,
        Unitary, Operator, SuperOp, KrausMap, Choi, or PauliLiouville).
    :param dims: Target per-subsystem dimensions, e.g. ``(3,)`` for a
        single qutrit.  Must have the same number of subsystems as *obj*
        and each dimension must be >= the corresponding current dimension.
    :return: A new object of the same type with the given *dims*.
    :raises TypeError: If *obj* is not a supported quantum object type.
    :raises ValueError: If *dims* is incompatible with the current dimensions.
    """
    raise TypeError(f"promote is not implemented for {type(obj).__name__}.")


@promote.register(StateVector)
@partial(jax.jit, static_argnames=("dims",))
def _promote_state_vector(state: StateVector, dims: tuple[int, ...]) -> StateVector:
    """Embed a state vector in a larger Hilbert space (zero-padded)."""
    current_dims = state.dims
    _validate_promote_dims(current_dims, dims)

    n_ensemble = len(state.ensemble_size)
    pw = [(0, 0)] * n_ensemble + [(0, D - d) for d, D in zip(current_dims, dims)]
    promoted = jnp.pad(state.data, pw)
    return StateVector(promoted, num_qubits=len(dims))


@promote.register(DensityMatrix)
@partial(jax.jit, static_argnames=("dims",))
def _promote_density_matrix(dm: DensityMatrix, dims: tuple[int, ...]) -> DensityMatrix:
    """Embed a density matrix in a larger Hilbert space (zero-padded)."""
    current_dims = dm.dims
    _validate_promote_dims(current_dims, dims)

    n_ensemble = len(dm.ensemble_size)
    # DensityMatrix tensor has shape (*ensemble, *dims, *dims)
    pw = [(0, 0)] * n_ensemble + [(0, D - d) for d, D in zip(current_dims + current_dims, dims + dims)]
    promoted = jnp.pad(dm.data, pw)
    return DensityMatrix(promoted, num_qubits=len(dims))


@promote.register(Unitary)
@partial(jax.jit, static_argnames=("dims",))
def _promote_unitary(unitary: Unitary, dims: tuple[int, ...]) -> Unitary:
    """Embed a unitary in a larger Hilbert space (identity on higher states)."""
    current_dims = unitary.dims[0]
    _validate_promote_dims(current_dims, dims)

    batch_shape = unitary.ensemble_size
    n_ensemble = len(batch_shape)
    D = reduce(mul, dims, 1)
    identity = jnp.broadcast_to(jnp.eye(D, dtype=complex).reshape(dims + dims), batch_shape + dims + dims)
    slices = tuple([slice(None)] * n_ensemble + [slice(0, d) for d in current_dims] * 2)
    promoted = identity.at[slices].set(unitary.data)
    return Unitary(promoted, num_qubits=len(dims))


@promote.register(Operator)
@partial(jax.jit, static_argnames=("dims",))
def _promote_operator(op: Operator, dims: tuple[int, ...]) -> Operator:
    """Embed an operator in a larger Hilbert space (identity on higher states)."""
    current_dims = op.dims[0]
    _validate_promote_dims(current_dims, dims)

    batch_shape = op.ensemble_size
    n_ensemble = len(batch_shape)
    D = reduce(mul, dims, 1)
    identity = jnp.broadcast_to(jnp.eye(D, dtype=complex).reshape(dims + dims), batch_shape + dims + dims)
    slices = tuple([slice(None)] * n_ensemble + [slice(0, d) for d in current_dims] * 2)
    promoted = identity.at[slices].set(op.data)
    return Operator(promoted, num_qubits=len(dims))


@promote.register(SuperOp)
@partial(jax.jit, static_argnames=("dims",))
def _promote_superop(superop: SuperOp, dims: tuple[int, ...]) -> SuperOp:
    """Embed a superoperator in a larger Liouville space via coherent extension.

    The promoted channel applies the original channel on the original
    subspace and acts as identity on the complement subspace, preserving
    coherences between the two subspaces.  This is achieved by converting
    to a Kraus representation, performing the coherent Kraus promotion,
    and converting back.
    """
    current_dims = superop.dims[0]
    _validate_promote_dims(current_dims, dims)

    kraus = superop_to_kraus(superop)
    promoted_kraus = _promote_kraus_map(kraus, dims)
    return kraus_to_superop(promoted_kraus)


@promote.register(Choi)
@partial(jax.jit, static_argnames=("dims",))
def _promote_choi(choi: Choi, dims: tuple[int, ...]) -> Choi:
    """Embed a Choi matrix in a larger Hilbert space via coherent extension.

    The promoted Choi matrix corresponds to a channel that acts as
    identity on the higher basis states while preserving coherences
    between the original and complement subspaces.
    """
    current_dims = choi.dims[0]
    _validate_promote_dims(current_dims, dims)

    kraus = choi_to_kraus(choi)
    promoted_kraus = _promote_kraus_map(kraus, dims)
    return kraus_to_choi(promoted_kraus)


@promote.register(KrausMap)
@partial(jax.jit, static_argnames=("dims",))
def _promote_kraus_map(kraus: KrausMap, dims: tuple[int, ...]) -> KrausMap:
    """Embed Kraus operators in a larger Hilbert space via coherent extension.

    Each Kraus operator K_i is zero-padded in the larger tensor space.
    The complement projector (identity on higher basis states) is added
    to the first Kraus operator K_0, preserving coherences between the
    original and complement subspaces.
    """
    current_dims = kraus.dims[0]
    _validate_promote_dims(current_dims, dims)

    batch_shape = kraus.ensemble_size
    n_ensemble = len(batch_shape)

    # Zero-pad each existing Kraus operator in tensor form.
    # Kraus tensor: (*ensemble, n_kraus, *dims_out, *dims_in)
    pw = [(0, 0)] * (n_ensemble + 1) + [(0, D - d) for d, D in zip(current_dims + current_dims, dims + dims)]
    padded = jnp.pad(kraus.data, pw)

    # Complement projector: identity on the target space with the original sub-block zeroed out.
    D = reduce(mul, dims, 1)
    complement = jnp.eye(D, dtype=complex).reshape(dims + dims)
    orig_slices = tuple(slice(0, d) for d in current_dims) * 2
    complement = complement.at[orig_slices].set(0.0)

    # Add complement identity to the first Kraus operator (coherent extension).
    k0_slices = tuple([slice(None)] * n_ensemble + [0])
    padded = padded.at[k0_slices].add(complement)

    return KrausMap(padded, num_qubits=len(dims))


@promote.register(PauliLiouville)
@partial(jax.jit, static_argnames=("dims",))
def _promote_pauli_liouville(pl: PauliLiouville, dims: tuple[int, ...]) -> PauliLiouville:
    """Embed a Pauli-Liouville matrix in a larger Hilbert space via coherent extension.

    Converts to Kraus, promotes coherently, and converts back.
    """
    current_dims = pl.dims[0]
    _validate_promote_dims(current_dims, dims)

    kraus = pauli_liouville_to_kraus(pl)
    promoted_kraus = _promote_kraus_map(kraus, dims)
    return kraus_to_pauli_liouville(promoted_kraus)


def _get_subsystem_dims(obj):
    """Return the per-subsystem dimensions of any quantum object.

    For states the dims are a flat tuple ``(d0, d1, ...)``.
    For operators and channels the dims are nested ``((d0, ...), (d0, ...))``;
    the output (first) tuple is returned.
    """
    if isinstance(obj, (StateVector, DensityMatrix)):
        return obj.dims
    return obj.dims[0]


def broadcast_qudits(obj1, obj2) -> tuple:
    """
    Determine the target per-subsystem dimensions from a pair of quantum objects.

    Analogous to NumPy broadcasting: for each subsystem the target dimension
    is the maximum of the two objects' dimensions.

    :param obj1: First quantum object.
    :param obj2: Second quantum object.
    :return: Tuple of target per-subsystem dimensions.
    :raises ValueError: If the objects have different numbers of subsystems.
    """
    dims1 = _get_subsystem_dims(obj1)
    dims2 = _get_subsystem_dims(obj2)

    if len(dims1) != len(dims2):
        raise ValueError(f"Number of subsystems must match for broadcasting: {len(dims1)} vs {len(dims2)}.")

    return tuple(max(a, b) for a, b in zip(dims1, dims2))


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def promote_hilbert_space(obj1: _T1, obj2: _T2) -> Tuple[_T1, _T2]:
    """
    Promote two quantum objects to compatible per-subsystem dimensions.

    For each subsystem index the target dimension is the maximum of the
    two objects' dimensions.  Only the subsystems that actually differ
    are enlarged; other subsystems are left untouched.

    When a :class:`Unitary` is paired with a channel representation
    (:class:`SuperOp`, :class:`Choi`, :class:`KrausMap`, or
    :class:`PauliLiouville`) and promotion is required, the ``Unitary``
    is first converted to ``SuperOp``.  This ensures the global phase —
    which is unobservable for channels but becomes a physical relative
    phase after embedding in a larger Hilbert space — is stripped
    *before* promotion, so that ``promote(to_superop(U))`` and
    ``to_superop(promote(U))`` agree.

    .. note::
        This function is not JIT-compiled because its control flow is
        inherently dimension-dependent (branches on static shape metadata).

    Returns the (possibly promoted) pair.
    """
    dims1 = _get_subsystem_dims(obj1)
    dims2 = _get_subsystem_dims(obj2)

    if dims1 == dims2:
        return obj1, obj2

    if isinstance(obj1, Unitary) and isinstance(obj2, SuperOperator):
        obj1 = unitary_to_superop(obj1)
    elif isinstance(obj2, Unitary) and isinstance(obj1, SuperOperator):
        obj2 = unitary_to_superop(obj2)

    target = broadcast_qudits(obj1, obj2)

    if dims1 != target:
        obj1 = promote(obj1, target)
    if dims2 != target:
        obj2 = promote(obj2, target)

    return cast(_T1, obj1), cast(_T2, obj2)
