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
    """Embed a superoperator in a larger Liouville space (CPTP).

    The promoted channel applies the original channel on the original
    subspace, acts as identity on the complement subspace, and maps
    cross-term coherences (between original and complement) to zero.
    This is the unique CPTP extension and corresponds to:

    .. math:: S' = \\mathcal{E}_s S \\mathcal{E}_s^\\dagger + Q \\otimes Q

    where *Q = I − P* is the complement projector.
    """
    current_dims = superop.dims[0]
    _validate_promote_dims(current_dims, dims)

    batch_shape = superop.ensemble_size
    n_ensemble = len(batch_shape)
    n = len(dims)
    D = reduce(mul, dims, 1)

    # Zero-pad the original superoperator tensor (equivalent to E_s S E_s†)
    pw = [(0, 0)] * n_ensemble + [(0, D_t - d_c) for d_c, D_t in zip(current_dims * 4, dims * 4)]
    padded = jnp.pad(superop.data, pw)

    # Complement indicator: 1 for multi-indices outside the original subspace
    q = jnp.ones(dims, dtype=complex)
    q = q.at[tuple(slice(0, d) for d in current_dims)].set(0.0)

    # Q⊗Q in superop tensor form: I_s[α,β,γ,δ] · q[α] · q[β]
    I_s = jnp.eye(D**2, dtype=complex).reshape(dims * 4)
    q_bra = q.reshape(dims + (1,) * (3 * n))
    q_ket = q.reshape((1,) * n + dims + (1,) * (2 * n))
    QQ = I_s * q_bra * q_ket

    promoted = padded + jnp.broadcast_to(QQ, batch_shape + dims * 4)
    return SuperOp(promoted, num_qubits=n)


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

    Each Kraus operator K_i is zero-padded in the larger tensor space.
    An additional Kraus operator is appended that acts as identity on
    the complement subspace to preserve trace preservation.
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
    # Shape: (*batch, 1, *dims, *dims)
    target_shape = batch_shape + (1,) + dims + dims
    extra = jnp.broadcast_to(complement, target_shape)

    promoted = jnp.concatenate([padded, extra], axis=n_ensemble)
    return KrausMap(promoted, num_qubits=len(dims))


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


def promote_hilbert_space(obj1, obj2):
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
