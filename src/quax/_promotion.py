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
    """Validate that target dims are compatible with current dims for promotion.

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

    The object is placed in the upper-left block (low-index subspace) of
    the target space; higher basis states act as identity / zero as
    appropriate for the object type.

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

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, dims, 1)
    batch_shape = state.ensemble_size
    vec = state.matrix  # (*ensemble, d_in)
    padded = jnp.zeros(batch_shape + (d_target,), dtype=complex)
    padded = padded.at[..., :d_in].set(vec)
    return StateVector.from_matrix(padded, dims)


@promote.register(DensityMatrix)
@partial(jax.jit, static_argnames=("dims",))
def _promote_density_matrix(dm: DensityMatrix, dims: tuple[int, ...]) -> DensityMatrix:
    """Embed a density matrix in a larger Hilbert space (zero-padded)."""
    current_dims = dm.dims
    _validate_promote_dims(current_dims, dims)

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, dims, 1)
    batch_shape = dm.ensemble_size
    mat = dm.matrix  # (*ensemble, d_in, d_in)
    padded = jnp.zeros(batch_shape + (d_target, d_target), dtype=complex)
    padded = padded.at[..., :d_in, :d_in].set(mat)
    return DensityMatrix.from_matrix(padded, dims)


@promote.register(Unitary)
@partial(jax.jit, static_argnames=("dims",))
def _promote_unitary(unitary: Unitary, dims: tuple[int, ...]) -> Unitary:
    """Embed a unitary in a larger Hilbert space (identity on higher states)."""
    current_dims = unitary.dims[0]
    _validate_promote_dims(current_dims, dims)

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, dims, 1)
    batch_shape = unitary.ensemble_size
    mat = unitary.matrix  # (*ensemble, d_in, d_in)
    eye = jnp.broadcast_to(jnp.eye(d_target, dtype=complex), batch_shape + (d_target, d_target))
    promoted = eye.at[..., :d_in, :d_in].set(mat)
    return Unitary.from_matrix(promoted, (dims, dims))


@promote.register(Operator)
@partial(jax.jit, static_argnames=("dims",))
def _promote_operator(op: Operator, dims: tuple[int, ...]) -> Operator:
    """Embed an operator in a larger Hilbert space (zero-padded)."""
    current_dims = op.dims[0]
    _validate_promote_dims(current_dims, dims)

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, dims, 1)
    batch_shape = op.ensemble_size
    mat = op.matrix  # (*ensemble, d_in, d_in)
    padded = jnp.zeros(batch_shape + (d_target, d_target), dtype=complex)
    padded = padded.at[..., :d_in, :d_in].set(mat)
    return Operator.from_matrix(padded, (dims, dims))


def _promote_superop_matrix(mat, d_in_sq, d_target_sq, batch_shape):
    """Embed a superoperator matrix in a larger Liouville space.

    The promoted channel is the direct sum of the original channel on
    the qubit subspace and the identity channel on the complement.
    Cross-terms (basis elements |i><j| where exactly one of i, j is
    in the complement) are mapped to zero.

    Uses column-stacking vectorization: vec(rho)[j*d + i] = rho[i, j].
    """
    import numpy as np

    d_in = int(round(d_in_sq**0.5))
    d_target = int(round(d_target_sq**0.5))

    # Qubit-subspace Liouville indices (column-stacking: vec index = j * d + i)
    qubit_idxs = np.array([j * d_target + i for j in range(d_in) for i in range(d_in)])
    # Complement-subspace Liouville indices (both row and column in complement)
    comp_idxs = np.array([j * d_target + i for j in range(d_in, d_target) for i in range(d_in, d_target)])

    promoted = jnp.zeros(batch_shape + (d_target_sq, d_target_sq), dtype=complex)
    promoted = promoted.at[..., qubit_idxs[:, None], qubit_idxs[None, :]].set(mat)
    comp_eye = jnp.broadcast_to(jnp.eye(len(comp_idxs), dtype=complex), batch_shape + (len(comp_idxs), len(comp_idxs)))
    promoted = promoted.at[..., comp_idxs[:, None], comp_idxs[None, :]].set(comp_eye)
    return promoted


@promote.register(SuperOp)
@partial(jax.jit, static_argnames=("dims",))
def _promote_superop(superop: SuperOp, dims: tuple[int, ...]) -> SuperOp:
    """Embed a superoperator in a larger Liouville space (identity on higher states)."""
    current_dims = superop.dims[0]
    _validate_promote_dims(current_dims, dims)

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, dims, 1)
    batch_shape = superop.ensemble_size
    mat = superop.matrix  # (*ensemble, d_in^2, d_in^2)
    promoted = _promote_superop_matrix(mat, d_in**2, d_target**2, batch_shape)
    return SuperOp.from_matrix(promoted, (dims, dims))


@promote.register(Choi)
@partial(jax.jit, static_argnames=("dims",))
def _promote_choi(choi: Choi, dims: tuple[int, ...]) -> Choi:
    """Embed a Choi matrix in a larger Hilbert space.

    The promoted Choi matrix corresponds to a channel that acts as
    identity on the higher basis states.
    """
    from ._superoperator_transformations import choi_to_superop, superop_to_choi

    current_dims = choi.dims[0]
    _validate_promote_dims(current_dims, dims)

    superop = choi_to_superop(choi)
    promoted_superop = _promote_superop(superop, dims)
    return superop_to_choi(promoted_superop)


@promote.register(KrausMap)
@partial(jax.jit, static_argnames=("dims",))
def _promote_kraus_map(kraus: KrausMap, dims: tuple[int, ...]) -> KrausMap:
    """Embed Kraus operators in a larger Hilbert space.

    Each Kraus operator K_i is embedded as the upper-left block.  An
    additional Kraus operator is appended that acts as identity on the
    higher basis states to preserve trace preservation.
    """
    current_dims = kraus.dims[0]
    _validate_promote_dims(current_dims, dims)

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, dims, 1)
    batch_shape = kraus.ensemble_size
    mat = kraus.matrix  # (*ensemble, n_kraus, d_in, d_in)
    n_kraus = mat.shape[-3]

    # Embed each existing Kraus operator: upper-left block, zeros elsewhere
    padded = jnp.zeros(batch_shape + (n_kraus, d_target, d_target), dtype=complex)
    padded = padded.at[..., :d_in, :d_in].set(mat)

    # Identity on the complement subspace as an extra Kraus operator
    complement = jnp.eye(d_target, dtype=complex)
    complement = complement.at[:d_in, :d_in].set(0.0)
    complement = jnp.broadcast_to(complement, batch_shape + (d_target, d_target))
    extra = complement[..., None, :, :]  # (*ensemble, 1, d_target, d_target)

    promoted = jnp.concatenate([padded, extra], axis=-3)
    return KrausMap.from_matrix(promoted, (dims, dims))


@promote.register(PauliLiouville)
@partial(jax.jit, static_argnames=("dims",))
def _promote_pauli_liouville(pl: PauliLiouville, dims: tuple[int, ...]) -> PauliLiouville:
    """Embed a Pauli-Liouville matrix in a larger Hilbert space.

    Converts to SuperOp, promotes, and converts back.
    """
    from ._superoperator_transformations import pauli_liouville_to_superop, superop_to_pauli_liouville

    current_dims = pl.dims[0]
    _validate_promote_dims(current_dims, dims)

    superop = pauli_liouville_to_superop(pl)
    promoted_superop = _promote_superop(superop, dims)
    return superop_to_pauli_liouville(promoted_superop)


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


def promote_if_necessary(obj1, obj2):
    """
    Promote two quantum objects to compatible per-subsystem dimensions.

    For each subsystem index the target dimension is the maximum of the
    two objects' dimensions.  Only the subsystems that actually differ
    are enlarged; other subsystems are left untouched.

    .. note::
        This function is not JIT-compiled because its control flow is
        inherently dimension-dependent (branches on static shape metadata).

    Returns the (possibly promoted) pair.
    """
    dims1 = _get_subsystem_dims(obj1)
    dims2 = _get_subsystem_dims(obj2)

    if dims1 == dims2:
        return obj1, obj2

    target = broadcast_qudits(obj1, obj2)

    if dims1 != target:
        obj1 = promote(obj1, target)
    if dims2 != target:
        obj2 = promote(obj2, target)

    return obj1, obj2
