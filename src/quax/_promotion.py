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

from functools import reduce, singledispatch
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
    QuantumInstrument,
    StateVector,
    SuperOp,
    Unitary,
    SuperOperator,
)
from ._state import zero_state_matrix, zero_state_vector
from ._superoperator_transformations import (
    choi_to_kraus,
    choi_to_superop,
    kraus_to_choi,
    kraus_to_pauli_liouville,
    kraus_to_superop,
    pauli_liouville_to_kraus,
    pauli_liouville_to_superop,
    superop_to_choi,
    superop_to_kraus,
    superop_to_pauli_liouville,
    unitary_to_superop,
)


@jax.jit
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
def promote(obj, dims: Tuple[int, ...]):
    """Promote each qudit of a quantum object to a larger dimension.

    The number of subsystems must match, and each target dimension must
    be >= the corresponding current dimension.  For states, higher
    dimensions are zero-padded.  For operators and channels, the
    promoted object acts as identity on the higher basis states.

    For channels (SuperOp, KrausMap, Choi, PauliLiouville) the
    **coherent** Kraus extension is used: the complement projector is
    distributed entirely to the first Kraus operator. The coherent extension
    can have unpredictable effects on cross-subspace coherences, please
    see the docs.

    :param obj: A quax quantum object (StateVector, DensityMatrix,
        Unitary, Operator, SuperOp, KrausMap, Choi, PauliLiouville,
        or QuantumInstrument).
    :param dims: Target per-subsystem dimensions.  Must have the same
        length as the object's qudit count.
    :return: A new object of the same type with the given *dims*.
    :raises TypeError: If *obj* is not a supported quantum object type.
    :raises ValueError: If *dims* is incompatible.
    """
    raise TypeError(f"promote is not implemented for {type(obj).__name__}.")


@promote.register(StateVector)
@jax.jit(static_argnames=("dims",))
def _promote_state_vector(state: StateVector, dims: Tuple[int, ...]) -> StateVector:
    """Embed a state vector in a larger Hilbert space (zero-padded)."""
    current_dims = state.dims
    _validate_promote_dims(current_dims, dims)

    n_ensemble = len(state.ensemble_size)
    pw = [(0, 0)] * n_ensemble + [(0, D - d) for d, D in zip(current_dims, dims)]
    promoted = jnp.pad(state.data, pw)
    return StateVector(promoted, num_qubits=len(dims))


@promote.register(DensityMatrix)
@jax.jit(static_argnames=("dims",))
def _promote_density_matrix(dm: DensityMatrix, dims: Tuple[int, ...]) -> DensityMatrix:
    """Embed a density matrix in a larger Hilbert space (zero-padded)."""
    current_dims = dm.dims
    _validate_promote_dims(current_dims, dims)

    n_ensemble = len(dm.ensemble_size)
    # DensityMatrix tensor has shape (*ensemble, *dims, *dims)
    pw = [(0, 0)] * n_ensemble + [(0, D - d) for d, D in zip(current_dims + current_dims, dims + dims)]
    promoted = jnp.pad(dm.data, pw)
    return DensityMatrix(promoted, num_qubits=len(dims))


def _promote_operator_like(obj, dims: Tuple[int, ...]):
    """Embed an operator-like object in a larger Hilbert space (identity on higher states)."""
    current_dims = obj.dims[0]
    _validate_promote_dims(current_dims, dims)
    batch_shape = obj.ensemble_size
    n_ensemble = len(batch_shape)
    D = reduce(mul, dims, 1)
    identity = jnp.broadcast_to(jnp.eye(D, dtype=complex).reshape(dims + dims), batch_shape + dims + dims)
    slices = tuple([slice(None)] * n_ensemble + [slice(0, d) for d in current_dims] * 2)
    promoted = identity.at[slices].set(obj.data)
    return type(obj)(promoted, num_qubits=len(dims))


@promote.register(Unitary)
@jax.jit(static_argnames=("dims",))
def _promote_unitary(unitary: Unitary, dims: Tuple[int, ...]) -> Unitary:
    """Embed a unitary in a larger Hilbert space (identity on higher states)."""
    return _promote_operator_like(unitary, dims)


@promote.register(Operator)
@jax.jit(static_argnames=("dims",))
def _promote_operator(op: Operator, dims: Tuple[int, ...]) -> Operator:
    """Embed an operator in a larger Hilbert space (identity on higher states)."""
    return _promote_operator_like(op, dims)


def _promote_via_kraus(obj, dims, to_kraus, from_kraus):
    """Promote a channel representation via the (default) coherent Kraus extension."""
    _validate_promote_dims(obj.dims[0], dims)
    return from_kraus(_promote_kraus_map(to_kraus(obj), dims))


@promote.register(SuperOp)
@jax.jit(static_argnames=("dims",))
def _promote_superop(superop: SuperOp, dims: Tuple[int, ...]) -> SuperOp:
    """Embed a superoperator via the coherent Kraus extension."""
    return _promote_via_kraus(superop, dims, superop_to_kraus, kraus_to_superop)


@promote.register(Choi)
@jax.jit(static_argnames=("dims",))
def _promote_choi(choi: Choi, dims: Tuple[int, ...]) -> Choi:
    """Embed a Choi matrix via the coherent Kraus extension."""
    return _promote_via_kraus(choi, dims, choi_to_kraus, kraus_to_choi)


@promote.register(KrausMap)
@jax.jit(static_argnames=("dims",))
def _promote_kraus_map(kraus: KrausMap, dims: Tuple[int, ...]) -> KrausMap:
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

    # Canonicalize the phase of K_0 before adding P_⊥.  Batched jnp.linalg.eigh can
    # return eigenvectors with different global phases than per-element eigh on the same
    # matrix, which would otherwise make the promoted channel depend on batch size.
    # We fix this by rotating K_0 so its largest-magnitude element is real and positive.
    k0_slices = tuple([slice(None)] * n_ensemble + [0])
    k0 = padded[k0_slices]  # (*batch_shape, *dims, *dims)
    k0_flat = k0.reshape(*batch_shape, -1)  # (*batch_shape, D²)
    max_idx = jnp.argmax(jnp.abs(k0_flat), axis=-1, keepdims=True)  # (*batch_shape, 1)
    max_elem = jnp.take_along_axis(k0_flat, max_idx, axis=-1)  # (*batch_shape, 1)
    max_abs = jnp.abs(max_elem)
    phase = jnp.where(max_abs > 1e-30, jnp.conj(max_elem) / max_abs, jnp.ones_like(max_elem))
    phase_shape = batch_shape + (1,) * len(dims + dims)
    k0 = k0 * phase.reshape(phase_shape)
    padded = padded.at[k0_slices].set(k0)

    # Add complement projector to K_0 (coherent extension).
    padded = padded.at[k0_slices].add(complement)

    return KrausMap(padded, num_qubits=len(dims))


@promote.register(PauliLiouville)
@jax.jit(static_argnames=("dims",))
def _promote_pauli_liouville(pl: PauliLiouville, dims: Tuple[int, ...]) -> PauliLiouville:
    """Embed a Pauli-Liouville matrix via the coherent Kraus extension."""
    return _promote_via_kraus(pl, dims, pauli_liouville_to_kraus, kraus_to_pauli_liouville)


@promote.register(QuantumInstrument)
@jax.jit(static_argnames=("dims",))
def _promote_quantum_instrument(inst: QuantumInstrument, dims: Tuple[int, ...]) -> QuantumInstrument:
    """Embed a quantum instrument in a larger Hilbert space via incoherent Kraus extension.

    Each outcome's CP map is zero-padded into the larger space.  The
    complement projector (identity on higher basis states) is added as
    a **separate** Kraus operator to the *last* outcome, so that the
    sum over outcomes remains CPTP while cross-subspace coherences are
    fully decohered.

    For a measurement instrument this means higher basis states (e.g.
    the transmon |2⟩ level) are assigned to the highest-index
    computational outcome — matching the typical dispersive-readout
    behaviour where leakage IQ points land near the |1⟩ blob — but
    coherences between the original and complement subspaces are
    destroyed, as a physical measurement would.
    """
    current_dims = inst.dims[0]
    _validate_promote_dims(current_dims, dims)

    n_outcomes = inst.num_outcomes

    # Complement projector superoperator: P_perp ⊗ P_perp* (incoherent).
    D = reduce(mul, dims, 1)
    complement = jnp.eye(D, dtype=complex)
    d = reduce(mul, current_dims, 1)
    complement = complement.at[:d, :d].set(0.0)
    complement_superop = jnp.kron(complement, complement.conj())

    promoted_outcome_mats = []
    for k in range(n_outcomes):
        superop_k, _ = inst.outcome_superop(k)
        kraus_k = superop_to_kraus(superop_k)

        promoted_kraus_k = _zero_pad_kraus(kraus_k, dims)
        promoted_superop_k = kraus_to_superop(promoted_kraus_k).matrix

        if k == n_outcomes - 1:
            promoted_superop_k = promoted_superop_k + complement_superop

        promoted_outcome_mats.append(promoted_superop_k)

    result_mat = jnp.stack(promoted_outcome_mats, axis=-3)
    return QuantumInstrument.from_matrix(result_mat, (dims, dims), inst.measured_qudits)


def _zero_pad_kraus(kraus: KrausMap, dims: Tuple[int, ...]) -> KrausMap:
    """Zero-pad Kraus operators to target dims without adding complement projector."""
    current_dims = kraus.dims[0]
    n_ensemble = len(kraus.ensemble_size)
    pw = [(0, 0)] * (n_ensemble + 1) + [(0, D - d) for d, D in zip(current_dims + current_dims, dims + dims)]
    return KrausMap(jnp.pad(kraus.data, pw), num_qubits=len(dims))


# ---------------------------------------------------------------------------
# Incoherent promotion: ``promote_incoherent``
# ---------------------------------------------------------------------------


@singledispatch
def promote_incoherent(obj, dims: Tuple[int, ...]):
    """Promote a superoperator to a larger Hilbert space via incoherent extension.

    Unlike :func:`promote`, which uses the coherent Kraus extension that
    preserves cross-subspace coherences, this function destroys all
    coherences between the original and complement subspaces.  This is
    the physically correct extension for measurements and resets, where
    the higher basis states do not maintain coherence with the
    computational subspace.

    Only dispatches on superoperator types: :class:`SuperOp`,
    :class:`KrausMap`, :class:`Choi`, and :class:`PauliLiouville`.

    :param obj: A quax superoperator object.
    :param dims: Target per-subsystem dimensions.
    :return: A new object of the same type with the given *dims*.
    :raises TypeError: If *obj* is not a supported superoperator type.
    :raises ValueError: If *dims* is incompatible.
    """
    raise TypeError(
        f"promote_incoherent is not implemented for {type(obj).__name__}. "
        f"Only superoperator types (SuperOp, KrausMap, Choi, PauliLiouville) are supported."
    )


@promote_incoherent.register(SuperOp)
@jax.jit(static_argnames=("dims",))
def _promote_incoherent_superop(superop: SuperOp, dims: Tuple[int, ...]) -> SuperOp:
    """Incoherent extension of a superoperator: ``E S E† + P_⊥ ⊗ P_⊥*``."""
    current_dims = superop.dims[0]
    _validate_promote_dims(current_dims, dims)
    d = reduce(mul, current_dims, 1)
    D = reduce(mul, dims, 1)
    # Embedding isometry E: D×d
    E = jnp.eye(D, d, dtype=complex)
    # For row-vectorized superops: E_vec = E* ⊗ E
    E_vec = jnp.kron(E.conj(), E)
    padded = E_vec @ superop.matrix @ E_vec.conj().T
    # Complement projector superoperator
    complement = jnp.eye(D, dtype=complex)
    complement = complement.at[:d, :d].set(0.0)
    complement_superop = jnp.kron(complement, complement.conj())
    padded = padded + complement_superop
    return SuperOp.from_matrix(padded, (dims, dims))


@promote_incoherent.register(KrausMap)
@jax.jit(static_argnames=("dims",))
def _promote_incoherent_kraus_map(kraus: KrausMap, dims: Tuple[int, ...]) -> KrausMap:
    """Incoherent extension of Kraus operators: zero-pad + P_⊥ as separate operator."""
    current_dims = kraus.dims[0]
    _validate_promote_dims(current_dims, dims)

    batch_shape = kraus.ensemble_size
    n_ensemble = len(batch_shape)

    # Zero-pad each existing Kraus operator.
    pw = [(0, 0)] * (n_ensemble + 1) + [(0, D - d) for d, D in zip(current_dims + current_dims, dims + dims)]
    padded = jnp.pad(kraus.data, pw)

    # Complement projector as a separate Kraus operator.
    D = reduce(mul, dims, 1)
    complement = jnp.eye(D, dtype=complex).reshape(dims + dims)
    orig_slices = tuple(slice(0, d) for d in current_dims) * 2
    complement = complement.at[orig_slices].set(0.0)
    complement_shape = batch_shape + (1,) + dims + dims
    complement_op = jnp.broadcast_to(complement, complement_shape)
    padded = jnp.concatenate([padded, complement_op], axis=n_ensemble)

    return KrausMap(padded, num_qubits=len(dims))


@promote_incoherent.register(Choi)
@jax.jit(static_argnames=("dims",))
def _promote_incoherent_choi(choi: Choi, dims: Tuple[int, ...]) -> Choi:
    """Incoherent extension of a Choi matrix (via SuperOp round-trip)."""
    _validate_promote_dims(choi.dims[0], dims)
    superop = choi_to_superop(choi)
    promoted_superop = _promote_incoherent_superop(superop, dims)
    return superop_to_choi(promoted_superop)


@promote_incoherent.register(PauliLiouville)
@jax.jit(static_argnames=("dims",))
def _promote_incoherent_pauli_liouville(pl: PauliLiouville, dims: Tuple[int, ...]) -> PauliLiouville:
    """Incoherent extension of a Pauli-Liouville matrix (via SuperOp round-trip)."""
    _validate_promote_dims(pl.dims[0], dims)
    superop = pauli_liouville_to_superop(pl)
    promoted_superop = _promote_incoherent_superop(superop, dims)
    return superop_to_pauli_liouville(promoted_superop)


# ---------------------------------------------------------------------------
# Positional embedding: ``_validate_embed``, ``permute``, ``embed``
# ---------------------------------------------------------------------------


def _validate_embed(current_dims: Tuple[int, ...], target_dims: Tuple[int, ...], positions: Tuple[int, ...]):
    """Validate arguments for positional embedding.

    :param current_dims: Per-subsystem dims of the object being embedded.
    :param target_dims: Per-subsystem dims of the target space.
    :param positions: Mapping from object qudits to target qudits.
    """
    n_obj = len(current_dims)
    n_target = len(target_dims)

    if len(positions) != n_obj:
        raise ValueError(f"positions has length {len(positions)} but the object has {n_obj} qudits.")
    if len(set(positions)) != len(positions):
        raise ValueError(f"positions must be distinct, got {positions}.")
    for p in positions:
        if p < 0 or p >= n_target:
            raise ValueError(f"Position {p} is out of range for a target space with {n_target} qudits.")
    for i, p in enumerate(positions):
        if target_dims[p] < current_dims[i]:
            raise ValueError(
                f"Target dimension {target_dims[p]} at position {p} is smaller than "
                f"current dimension {current_dims[i]} at qudit {i}."
            )


# ---- permute singledispatch ----


@singledispatch
def permute(obj, perm: Tuple[int, ...]):
    """Permute the qudit ordering of a quantum object.

    ``perm[i] = j`` means "qudit currently at position *i* moves to
    position *j* in the output".

    :param obj: A quax quantum object.
    :param perm: A permutation tuple of length equal to the number of
        qudits.  Must be a valid permutation of ``range(n_qudits)``.
    :return: A new object of the same type with reordered qudits.
    :raises TypeError: If *obj* is not a supported quantum object type.
    """
    raise TypeError(f"permute is not implemented for {type(obj).__name__}.")


def _inverse_perm(perm: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute the inverse of a permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


@permute.register(StateVector)
@jax.jit(static_argnames=("perm",))
def _permute_state_vector(state: StateVector, perm: Tuple[int, ...]) -> StateVector:
    n_ens = len(state.ensemble_size)
    inv = _inverse_perm(perm)
    axes = tuple(range(n_ens)) + tuple(n_ens + inv[i] for i in range(len(perm)))
    data = jnp.transpose(state.data, axes)
    return StateVector(data, num_qubits=state.num_qubits)


@permute.register(DensityMatrix)
@jax.jit(static_argnames=("perm",))
def _permute_density_matrix(dm: DensityMatrix, perm: Tuple[int, ...]) -> DensityMatrix:
    return _permute_operator_like(dm, perm)


def _permute_operator_like(obj, perm: Tuple[int, ...]):
    """Shared implementation for Operator-like types (2 groups of qudit axes)."""
    n = len(perm)
    n_ens = len(obj.ensemble_size)
    inv = _inverse_perm(perm)
    axes = tuple(range(n_ens)) + tuple(n_ens + inv[i] for i in range(n)) + tuple(n_ens + n + inv[i] for i in range(n))
    data = jnp.transpose(obj.data, axes)
    return type(obj)(data, num_qubits=obj.num_qubits)


@permute.register(Unitary)
@jax.jit(static_argnames=("perm",))
def _permute_unitary(u: Unitary, perm: Tuple[int, ...]) -> Unitary:
    return _permute_operator_like(u, perm)


@permute.register(Operator)
@jax.jit(static_argnames=("perm",))
def _permute_op(op: Operator, perm: Tuple[int, ...]) -> Operator:
    return _permute_operator_like(op, perm)


def _permute_superop_like(obj, perm: Tuple[int, ...]):
    """Shared implementation for SuperOperator-like types (4 groups of qudit axes)."""
    n = len(perm)
    n_ens = len(obj.ensemble_size)
    inv = _inverse_perm(perm)
    axes = (
        tuple(range(n_ens))
        + tuple(n_ens + inv[i] for i in range(n))  # out_bra
        + tuple(n_ens + n + inv[i] for i in range(n))  # out_ket
        + tuple(n_ens + 2 * n + inv[i] for i in range(n))  # in_bra
        + tuple(n_ens + 3 * n + inv[i] for i in range(n))  # in_ket
    )
    data = jnp.transpose(obj.data, axes)
    return type(obj)(data, num_qubits=obj.num_qubits)


@permute.register(SuperOp)
@jax.jit(static_argnames=("perm",))
def _permute_superop(s: SuperOp, perm: Tuple[int, ...]) -> SuperOp:
    return _permute_superop_like(s, perm)


@permute.register(Choi)
@jax.jit(static_argnames=("perm",))
def _permute_choi(c: Choi, perm: Tuple[int, ...]) -> Choi:
    return _permute_superop_like(c, perm)


@permute.register(PauliLiouville)
@jax.jit(static_argnames=("perm",))
def _permute_pauli_liouville(pl: PauliLiouville, perm: Tuple[int, ...]) -> PauliLiouville:
    # PauliLiouville data lives in the generalized Pauli basis, so a simple
    # tensor transpose does NOT correctly permute qudits.  Round-trip through
    # the computational-basis SuperOp representation where transposition is valid.
    return superop_to_pauli_liouville(_permute_superop(pauli_liouville_to_superop(pl), perm))


@permute.register(KrausMap)
@jax.jit(static_argnames=("perm",))
def _permute_kraus(k: KrausMap, perm: Tuple[int, ...]) -> KrausMap:
    n = len(perm)
    n_ens = len(k.ensemble_size)
    inv = _inverse_perm(perm)
    # Kraus tensor: (*ensemble, n_kraus, *dims_out, *dims_in)
    axes = (
        tuple(range(n_ens))
        + (n_ens,)  # n_kraus axis
        + tuple(n_ens + 1 + inv[i] for i in range(n))
        + tuple(n_ens + 1 + n + inv[i] for i in range(n))
    )
    data = jnp.transpose(k.data, axes)
    return KrausMap(data, num_qubits=k.num_qubits)


@permute.register(QuantumInstrument)
@jax.jit(static_argnames=("perm",))
def _permute_instrument(inst: QuantumInstrument, perm: Tuple[int, ...]) -> QuantumInstrument:
    n = len(perm)
    n_ens = len(inst.ensemble_size)
    inv = _inverse_perm(perm)
    # Instrument tensor: (*ensemble, n_outcomes, *out_bra, *out_ket, *in_bra, *in_ket)
    axes = (
        tuple(range(n_ens))
        + (n_ens,)  # n_outcomes axis
        + tuple(n_ens + 1 + inv[i] for i in range(n))  # out_bra
        + tuple(n_ens + 1 + n + inv[i] for i in range(n))  # out_ket
        + tuple(n_ens + 1 + 2 * n + inv[i] for i in range(n))  # in_bra
        + tuple(n_ens + 1 + 3 * n + inv[i] for i in range(n))  # in_ket
    )
    data = jnp.transpose(inst.data, axes)
    # Remap measured_qudits through the permutation
    new_measured = tuple(perm[m] for m in inst.measured_qudits)
    return QuantumInstrument(data, num_qubits=inst.num_qubits, measured_qudits=new_measured)


# ---- embed: positional embedding ----


@jax.jit(static_argnames=("target_dims", "positions"))
def embed(obj, target_dims: Tuple[int, ...], positions: Tuple[int, ...]):
    """Embed a quantum object into a larger Hilbert space at specified positions.

    The object's qudits are placed at the target positions given by
    *positions*.  Remaining target qudits act as identity (for operators/
    channels) or |0⟩ (for states).  If a target dimension is larger than
    the object's corresponding qudit, per-subsystem promotion is applied
    first.

    :param obj: A quax quantum object (StateVector, DensityMatrix,
        Unitary, Operator, SuperOp, KrausMap, Choi, PauliLiouville,
        or QuantumInstrument).
    :param target_dims: Per-subsystem dimensions of the full target space.
    :param positions: Tuple mapping each of the object's qudits to a
        target qudit index.  Must have the same length as the number of
        qudits in *obj*, contain distinct non-negative integers, each
        less than ``len(target_dims)``.
    :return: A new object of the same type embedded in the target space.
    :raises TypeError: If *obj* is not a supported quantum object type.
    :raises ValueError: If *target_dims* or *positions* are incompatible.
    """
    current_dims = _get_subsystem_dims(obj)
    _validate_embed(current_dims, target_dims, positions)

    n_target = len(target_dims)

    # Step 1: per-subsystem promote the positioned qudits if any dim increases
    positioned_dims = tuple(target_dims[p] for p in positions)
    if positioned_dims != current_dims:
        obj = promote(obj, positioned_dims)

    # Step 2: tensor with identity/zero for extra qudits
    extras = [i for i in range(n_target) if i not in positions]
    for idx in extras:
        d = target_dims[idx]
        obj = _tensor_with_identity(obj, d)

    # Step 3: permute into correct order
    # After steps 1-2, the qudit ordering is: [positions[0], ..., positions[n_obj-1], extras[0], ...]
    # We want standard order [0, 1, ..., n_target-1]
    current_order = list(positions) + extras
    perm = tuple(current_order)

    if perm != tuple(range(n_target)):
        obj = permute(obj, perm)

    return obj


def _tensor_with_identity(obj, d: int):
    """Tensor a quantum object with a single-qudit identity/zero of dimension *d*.

    For states, tensors with |0⟩ (or |0⟩⟨0|).
    For operators/channels, tensors with the identity operator/channel.
    Uses the ``|`` operator for the tensor product.
    """
    eye = Unitary.from_matrix(jnp.eye(d, dtype=complex), ((d,), (d,)))

    match obj:
        case StateVector():
            return obj | zero_state_vector(dims=(d,))

        case DensityMatrix():
            return obj | zero_state_matrix(dims=(d,))

        case Unitary() | Operator() | KrausMap() | SuperOp() | Choi() | PauliLiouville():
            return obj | eye

        case QuantumInstrument():
            id_superop = SuperOp.from_matrix(jnp.eye(d * d, dtype=complex), ((d,), (d,)))
            id_inst = QuantumInstrument.from_matrix(
                id_superop.matrix[..., None, :, :],
                ((d,), (d,)),
                measured_qudits=(),
            )
            return obj | id_inst

        case _:
            raise TypeError(f"_tensor_with_identity is not implemented for {type(obj).__name__}.")


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
        obj1 = promote(obj1, target)  # type: ignore[assignment]
    if dims2 != target:
        obj2 = promote(obj2, target)  # type: ignore[assignment]

    return cast(_T1, obj1), cast(_T2, obj2)
