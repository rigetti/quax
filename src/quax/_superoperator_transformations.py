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

from functools import lru_cache, reduce, singledispatch
from operator import mul
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from ._operator_basis import n_qudit_herm_basis
from ._quantum_objects import Choi, KrausMap, PauliLiouville, SuperOp, Unitary

if TYPE_CHECKING:
    from ._quantum_objects import Lindbladian

# ============================================================================
# Choi <-> Superoperator transformations
# ============================================================================


@jax.jit
def choi_to_superop(choi: Choi) -> SuperOp:
    """
    Convert Choi to superoperator.

    This operation is its own inverse (bijection).
    Following the reference implementation: reshape to (d, d, d, d), swap axes 0 and 3, reshape back.
    """
    d_out, d_in = choi.d
    assert d_in == d_out, "Choi to Superoperator conversion only supports square operators"
    d = d_in
    d2 = choi.d2[0]

    ensemble_size = choi.ensemble_size

    J4 = choi.matrix.reshape(*ensemble_size, d, d, d, d)  # (..., d, d, d, d)
    S4 = jnp.swapaxes(J4, -4, -1)  # swap first/last of the last-4 axes
    S = S4.reshape(*ensemble_size, d2, d2)  # (..., d^2, d^2)

    return SuperOp.from_matrix(S, choi.dims)


@jax.jit
def superop_to_choi(superop: SuperOp) -> Choi:
    """
    Convert superoperator to Choi.

    :param superop: Superoperator
    :return: equivalent Choi

    This operation is its own inverse (bijection), identical to choi_to_superop.
    """
    d_out, d_in = superop.d
    assert d_in == d_out, "Superoperator to Choi conversion only supports square operators"
    d = d_in
    d2 = superop.d2[0]

    ensemble_size = superop.ensemble_size

    S4 = superop.matrix.reshape(*ensemble_size, d, d, d, d)  # (..., d, d, d, d)
    J4 = jnp.swapaxes(S4, -4, -1)  # same swap
    J = J4.reshape(*ensemble_size, d2, d2)  # (..., d^2, d^2)
    return Choi.from_matrix(J, superop.dims)


# ============================================================================
# Pauli-Liouville transformations
# ============================================================================


@lru_cache(maxsize=32)
def _pauli2computational_basis_matrix(dims: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Array:
    """
    Produce basis transform matrix from Hermitian operator basis to computational basis.

    For qubits this is the Pauli-to-computational transform.
    For general qudits this is the Hermitian-Weyl-to-computational transform.

    Returns a d^2 by d^2 matrix where column i is vec(B_i).
    """
    basis_mats = n_qudit_herm_basis(dims[0]).matrix
    return jnp.stack([op.T.ravel() for op in basis_mats], axis=1)


@lru_cache(maxsize=32)
def _computational2pauli_basis_matrix(dims: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Array:
    """
    Produce basis transform matrix from computational basis to Hermitian operator basis.

    This is (1/dim) * conjugate transpose of _pauli2computational_basis_matrix.
    """
    d = int(reduce(mul, dims[0]))
    p2c = _pauli2computational_basis_matrix(dims)
    return jnp.conj(p2c).T / d


@jax.jit
def superop_to_pauli_liouville(superop: SuperOp) -> PauliLiouville:
    """
    Convert superoperator to Pauli-Liouville matrix.

    This is achieved by a linear change of basis.
    """
    d_out, d_in = superop.d
    assert d_in == d_out, "Superoperator to Pauli-Liouville conversion only supports square operators"

    # ensure_compile_time_eval forces eager evaluation so that the @lru_cache
    # in _operator_basis stores concrete arrays, not JIT tracers that would
    # leak across compilation scopes.
    with jax.ensure_compile_time_eval():
        c2p = _computational2pauli_basis_matrix(superop.dims)
    data = c2p @ superop.matrix @ jnp.conj(c2p).T * d_in
    # The Hermitian operator basis guarantees real PL matrices for CP maps.
    data = data.real
    return PauliLiouville.from_matrix(data, superop.dims)


@jax.jit
def pauli_liouville_to_superop(pauli_liouville: PauliLiouville) -> SuperOp:
    """
    Convert Pauli-Liouville matrix to superoperator.

    This is achieved by a linear change of basis.
    """
    d_out, d_in = pauli_liouville.d
    assert d_in == d_out, "Pauli-Liouville to Superoperator conversion only supports square operators"

    # ensure_compile_time_eval forces eager evaluation so that the @lru_cache
    # in _operator_basis stores concrete arrays, not JIT tracers that would
    # leak across compilation scopes.
    with jax.ensure_compile_time_eval():
        p2c = _pauli2computational_basis_matrix(pauli_liouville.dims)
    data = p2c @ pauli_liouville.matrix @ jnp.conj(p2c).T / d_in
    return SuperOp.from_matrix(data, pauli_liouville.dims)


@jax.jit
def choi_to_pauli_liouville(choi: Choi) -> PauliLiouville:
    """
    Convert Choi matrix to Pauli-Liouville matrix.

    Composed transformation: Choi -> Superop -> Pauli-Liouville
    """
    S = choi_to_superop(choi)
    return superop_to_pauli_liouville(S)


@jax.jit
def pauli_liouville_to_choi(pauli_liouville: PauliLiouville) -> Choi:
    """
    Convert Pauli-Liouville matrix to Choi matrix.

    Composed transformation: Pauli-Liouville -> Superop -> Choi
    """
    S = pauli_liouville_to_superop(pauli_liouville)
    return superop_to_choi(S)


# ============================================================================
# Kraus transformations
# ============================================================================


@jax.jit
def kraus_to_choi(kraus_map: KrausMap) -> Choi:
    """
    Convert Kraus operators to Choi matrix using column-stacking vec:
      vec(M) = M.T.reshape(-1)

    If kraus_map.matrix has shape (N, d_out, d_in),
    result has shape (d_out*d_in, d_out*d_in).
    """
    K = kraus_map.matrix  # (..., N, d_out, d_in)

    assert kraus_map.d[0] == kraus_map.d[1], "Kraus to Choi conversion only supports square operators"
    d2 = kraus_map.d2[0]

    ensemble_size = kraus_map.ensemble_size
    N = K.shape[-3]

    # Vectorize all Kraus operators: V[i] = vec(K[i]) has shape (d_out*d_in,)
    V = jnp.reshape(jnp.swapaxes(K, -1, -2), (*ensemble_size, N, d2))  # (N, d_out*d_in)

    # J = sum_i vec(Ki) vec(Ki)†  => V^H V as an outer-sum
    # einsum: (N, a) (N, b)* -> (a, b)
    J = jnp.einsum("...na,...nb->...ab", V, jnp.conj(V))

    return Choi.from_matrix(J, kraus_map.dims)


@jax.jit
def kraus_to_superop(kraus_map: KrausMap) -> SuperOp:
    """
    Convert Kraus operators to superoperator.

    This implements: S = sum_i conj(K_i) ⊗ K_i
    where conj is elementwise complex conjugate (no transpose).
    """
    K = kraus_map.matrix  # (..., N, d_out, d_in)

    assert kraus_map.d[0] == kraus_map.d[1], "Kraus to Choi conversion only supports square operators"
    d_out, d_in = kraus_map.d

    ensemble_size = kraus_map.ensemble_size
    N = K.shape[-3]

    # Build each kron term as a matrix of shape (d_out*d_out, d_in*d_in) or (d_out*d_in, d_out*d_in)
    # depending on convention. For the expression conj(K) ⊗ K with K:(d_out,d_in),
    # the kron is (d_out*d_out, d_in*d_in).
    #
    # Compute kron for each i via einsum:
    # (a,b) x (c,d) -> (a,c,b,d) then reshape -> (a*c, b*d)
    terms = jnp.einsum("...iab,...icd->...iacbd", jnp.conj(K), K)
    terms = terms.reshape(*ensemble_size, N, d_out * d_out, d_in * d_in)

    S = jnp.sum(terms, axis=-3)  # (d_out^2, d_in^2)
    return SuperOp.from_matrix(S, kraus_map.dims)


@jax.jit
def kraus_to_pauli_liouville(kraus_ops: KrausMap) -> PauliLiouville:
    """
    Convert Kraus operators to Pauli-Liouville matrix.

    Composed transformation: Kraus -> Superop -> Pauli-Liouville
    """
    S = kraus_to_superop(kraus_ops)
    return superop_to_pauli_liouville(S)


@jax.jit
def choi_to_kraus(choi: Choi, tol: float = 1e-6) -> KrausMap:
    """
    Convert Choi matrix to a fixed-size Kraus representation (jit-compatible).

    Returns KrausMap.matrix with shape (Nmax, d_out, d_in) where
      Nmax = d_out * d_in  (i.e., Choi dimension).

    Construction:
      J = sum_i λ_i ``|v_i><v_i|``
      K_i = sqrt(max(λ_i, 0)) * unvec(v_i)

    The Kraus operators are returned in descending order of their Choi
    eigenvalues. Eigenvalues below tol are clamped to 0, yielding zero
    Kraus operators that can be removed with :func:`truncate_kraus`.
    """
    d_out, d_in = choi.d
    ensemble_size = choi.ensemble_size
    n = d_in * d_out  # dimension of vec space; also max Kraus count

    # Hermitian symmetrization helps numerical stability
    J = 0.5 * (choi.matrix + jnp.conj(jnp.swapaxes(choi.matrix, -1, -2)))

    # Eigendecomposition (J is Hermitian)
    eigvals, eigvecs = jnp.linalg.eigh(J)  # eigvecs columns

    # Batched sort descending
    order = jnp.argsort(eigvals, axis=-1)
    order = jnp.flip(order, axis=-1)

    eigvals = jnp.take_along_axis(eigvals, order, axis=-1)  # (..., n)
    eigvecs = jnp.take_along_axis(eigvecs, order[..., None, :], axis=-1)  # (..., n, n)

    # Clamp + sqrt
    eigvals = jnp.where(eigvals > tol, eigvals, 0.0)
    coeffs = jnp.sqrt(eigvals).astype(J.dtype)  # (..., n)

    # Scale columns: W[..., :, i] = coeffs[..., i] * v_i
    W = eigvecs * coeffs[..., None, :]  # (..., n, n)

    # Now interpret each (scaled) eigenvector as a Kraus operator.
    # We want K shape: (..., n, d_out, d_in)
    # Each kraus vector is length n; with vec(M)=M.T.reshape(-1):
    # unvec(v) = v.reshape(d_out, d_in).T  -> (d_in, d_out), then swap to get (d_out, d_in)
    WT = jnp.swapaxes(W, -1, -2)  # (..., n, n)
    K = WT.reshape(*ensemble_size, n, d_out, d_in)  # (..., n, d_out, d_in)
    K = jnp.swapaxes(K, -1, -2)  # (..., n, d_in, d_out) -> swap to get (d_out, d_in)

    return KrausMap.from_matrix(K, choi.dims)


def truncate_kraus(kraus_map: KrausMap, atol: float = 1e-6) -> KrausMap:
    """
    Sort Kraus operators by descending Frobenius norm and remove near-zero operators.

    This is useful for Monte Carlo simulations where the number of Kraus operators
    directly impacts performance. Many physical noise channels have only a few
    significant Kraus operators, so truncating near-zero operators can dramatically
    reduce computational cost.

    The Frobenius norm is used as the ordering and truncation criterion because
    it equals the square root of the corresponding Choi eigenvalue, providing a
    basis-independent measure of each operator's contribution to the channel.
    For low-error-rate channels the dominant Kraus operator (typically close to
    the identity) will have the largest norm, so ordering by descending norm
    places the most probable trajectory first — ideal for Monte Carlo sampling.

    For ensembles, the number of significant operators may vary across ensemble
    members. In this case, the maximum count across the ensemble is used so that
    the output array has a uniform shape.

    .. note::

        This function is **not** JIT-compatible because the output shape depends
        on the data. Use it outside of JIT-compiled regions.

    :param kraus_map: KrausMap to truncate
    :param atol: Absolute tolerance. Kraus operators with Frobenius norm <= atol are removed.
    :return: KrausMap with operators sorted by descending norm and near-zero operators removed
    """
    K = kraus_map.matrix  # (..., n_kraus, d_out, d_in)

    # Frobenius norm of each Kraus operator: (..., n_kraus)
    norms = jnp.linalg.norm(K, axis=(-2, -1))

    # Sort by descending norm
    order = jnp.flip(jnp.argsort(norms, axis=-1), axis=-1)
    K_sorted = jnp.take_along_axis(K, order[..., None, None], axis=-3)

    # Recompute norms in sorted order for threshold check
    norms_sorted = jnp.take_along_axis(norms, order, axis=-1)

    # Count significant operators per ensemble member, then take global max
    n_sig = jnp.sum(norms_sorted > atol, axis=-1)
    max_kraus = int(jnp.max(n_sig))
    max_kraus = max(max_kraus, 1)  # keep at least one operator

    # Truncate
    K_truncated = K_sorted[..., :max_kraus, :, :]

    return KrausMap.from_matrix(K_truncated, kraus_map.dims)


@jax.jit
def superop_to_kraus(superop: SuperOp) -> KrausMap:
    """
    Convert superoperator to Kraus operators.

    Composed transformation: Superop -> Choi -> Kraus
    """
    J = superop_to_choi(superop)
    return choi_to_kraus(J)


@jax.jit
def pauli_liouville_to_kraus(pauli_liouville: PauliLiouville) -> KrausMap:
    """
    Convert Pauli-Liouville matrix to Kraus operators.

    Composed transformation: Pauli-Liouville -> Choi -> Kraus
    """
    J = pauli_liouville_to_choi(pauli_liouville)
    return choi_to_kraus(J)


# ============================================================================
# Unitary transformations
# ============================================================================


@jax.jit
def unitary_to_choi(unitary: Unitary) -> Choi:
    """
    Convert unitary operator to Choi matrix.

    For a unitary channel E(ρ) = U ρ U†, the Choi matrix is ``|U>> <<U|``
    where ``|U>>`` = vec(U) using column-stacking convention.

    This is more efficient than treating U as a single-element Kraus operator list.

    Args:
        unitary: Unitary object

    Returns:
        Choi object
    """

    unitary_data = unitary.matrix
    ensemble_size = unitary.ensemble_size
    d = unitary.d[0]
    assert d == unitary.d[1], "Unitary to Choi conversion only supports square operators"

    # vec(U): (..., d^2, 1)
    vec_u = jnp.swapaxes(unitary_data, -1, -2).reshape(*ensemble_size, d * d, 1)

    # |U⟩⟩⟨⟨U|: (..., d^2, d^2)
    choi_data = vec_u @ jnp.conj(jnp.swapaxes(vec_u, -1, -2))

    return Choi.from_matrix(choi_data, unitary.dims)


@jax.jit
def unitary_to_superop(unitary: Unitary) -> SuperOp:
    """
    Convert unitary operator to superoperator.

    For a unitary channel E(ρ) = U ρ U†, the superoperator is U^* ⊗ U
    where U^* is complex conjugate (not conjugate transpose).

    This implements the standard vec convention where vec(A B C) = (C^T ⊗ A) vec(B).

    :param unitary: Unitary object
    :return: SuperOp object
    """
    unitary_data = unitary.matrix
    ensemble_size = unitary.ensemble_size
    d = unitary.d[0]
    assert d == unitary.d[1], "Unitary to SuperOp conversion only supports square operators"

    # (U^*)_{a b} (U)_{c d} -> S_{(a,c),(b,d)}
    S4 = jnp.einsum("...ab,...cd->...acbd", jnp.conj(unitary_data), unitary_data)  # (..., d, d, d, d)
    superop_data = S4.reshape(*ensemble_size, d * d, d * d)

    return SuperOp.from_matrix(superop_data, unitary.dims)


@jax.jit
def unitary_to_pauli_liouville(unitary: Unitary) -> PauliLiouville:
    """
    Convert unitary operator to Pauli-Liouville matrix.

    Composed transformation: Unitary -> Superop -> Pauli-Liouville

    Args:
        unitary: Unitary object

    Returns:
        PauliLiouville object
    """
    S = unitary_to_superop(unitary)
    return superop_to_pauli_liouville(S)


@jax.jit
def unitary_to_kraus_map(unitary: Unitary) -> KrausMap:
    """
    Convert unitary operator to Kraus operators.

    Composed transformation: Unitary -> Choi -> Kraus

    Args:
        unitary: Unitary object

    Returns:
        KrausMap object
    """
    J = unitary_to_choi(unitary)
    return choi_to_kraus(J)


# ============================================================================
# Dispatchers for converting superoperators
# ============================================================================


@singledispatch
def to_choi(operator) -> Choi:
    """
    Convert any superoperator type to Choi representation.

    This is a single-dispatch function that automatically selects the appropriate
    conversion based on the input type.

    :param operator: A superoperator (Choi, SuperOp, PauliLiouville, KrausMap, or Unitary)
    :return: Choi representation of the operator
    :raises NotImplementedError: If the operator type is not supported
    """
    raise NotImplementedError(f"Conversion to Choi not implemented for type {type(operator)}")


@to_choi.register(Choi)
def _choi_to_choi(choi: Choi) -> Choi:
    """Identity conversion for Choi matrices."""
    return choi


@to_choi.register(SuperOp)
def _superop_to_choi_dispatch(superop: SuperOp) -> Choi:
    """Convert SuperOp to Choi."""
    return superop_to_choi(superop)


@to_choi.register(PauliLiouville)
def _pauli_liouville_to_choi_dispatch(pauli_liouville: PauliLiouville) -> Choi:
    """Convert PauliLiouville to Choi."""
    return pauli_liouville_to_choi(pauli_liouville)


@to_choi.register(KrausMap)
def _kraus_map_to_choi_dispatch(kraus_map: KrausMap) -> Choi:
    """Convert KrausMap to Choi."""
    return kraus_to_choi(kraus_map)


@to_choi.register(Unitary)
def _unitary_to_choi_dispatch(unitary: Unitary) -> Choi:
    """Convert Unitary to Choi."""
    return unitary_to_choi(unitary)


@singledispatch
def to_superop(operator) -> SuperOp:
    """
    Convert any superoperator type to SuperOp representation.

    This is a single-dispatch function that automatically selects the appropriate
    conversion based on the input type.

    :param operator: A superoperator (Choi, SuperOp, PauliLiouville, KrausMap, or Unitary)
    :return: SuperOp representation of the operator
    :raises NotImplementedError: If the operator type is not supported
    """
    raise NotImplementedError(f"Conversion to SuperOp not implemented for type {type(operator)}")


@to_superop.register(Choi)
def _choi_to_superop(choi: Choi) -> SuperOp:
    """Convert Choi to SuperOp."""
    return choi_to_superop(choi)


@to_superop.register(SuperOp)
def _superop_to_superop(superop: SuperOp) -> SuperOp:
    """Identity conversion for SuperOp matrices."""
    return superop


@to_superop.register(PauliLiouville)
def _pauli_liouville_to_superop(pauli_liouville: PauliLiouville) -> SuperOp:
    """Convert PauliLiouville to SuperOp."""
    return pauli_liouville_to_superop(pauli_liouville)


@to_superop.register(KrausMap)
def _kraus_map_to_superop(kraus_map: KrausMap) -> SuperOp:
    """Convert KrausMap to SuperOp."""
    return kraus_to_superop(kraus_map)


@to_superop.register(Unitary)
def _unitary_to_superop(unitary: Unitary) -> SuperOp:
    """Convert Unitary to SuperOp."""
    return unitary_to_superop(unitary)


@singledispatch
def to_pauli_liouville(operator) -> PauliLiouville:
    """
    Convert any superoperator type to PauliLiouville representation.

    This is a single-dispatch function that automatically selects the appropriate
    conversion based on the input type.

    :param operator: A superoperator (Choi, SuperOp, PauliLiouville, KrausMap, or Unitary)
    :return: PauliLiouville representation of the operator
    :raises NotImplementedError: If the operator type is not supported
    """
    raise NotImplementedError(f"Conversion to SuperOp not implemented for type {type(operator)}")


@to_pauli_liouville.register(Choi)
def _choi_to_pauli_liouville(choi: Choi) -> PauliLiouville:
    """Identity conversion for Choi matrices."""
    return choi_to_pauli_liouville(choi)


@to_pauli_liouville.register(SuperOp)
def _superop_to_pauli_liouville(superop: SuperOp) -> PauliLiouville:
    """Convert SuperOp to PauliLiouville."""
    return superop_to_pauli_liouville(superop)


@to_pauli_liouville.register(PauliLiouville)
def _pauli_liouville_to_pauli_liouville(pauli_liouville: PauliLiouville) -> PauliLiouville:
    """Identity conversion for PauliLiouville matrices."""
    return pauli_liouville


@to_pauli_liouville.register(KrausMap)
def _kraus_map_to_pauli_liouville(kraus_map: KrausMap) -> PauliLiouville:
    """Convert KrausMap to PauliLiouville."""
    return kraus_to_pauli_liouville(kraus_map)


@to_pauli_liouville.register(Unitary)
def _unitary_to_pauli_liouville(unitary: Unitary) -> PauliLiouville:
    """Convert Unitary to PauliLiouville."""
    return unitary_to_pauli_liouville(unitary)


@singledispatch
def to_kraus(operator) -> KrausMap:
    """
    Convert any superoperator type to KrausMap representation.

    This is a single-dispatch function that automatically selects the appropriate
    conversion based on the input type.

    :param operator: A superoperator (Choi, SuperOp, PauliLiouville, KrausMap, or Unitary)
    :return: KrausMap representation of the operator
    :raises NotImplementedError: If the operator type is not supported
    """
    raise NotImplementedError(f"Conversion to SuperOp not implemented for type {type(operator)}")


@to_kraus.register(Choi)
def _choi_to_kraus_map(choi: Choi) -> KrausMap:
    """Identity conversion for Choi matrices."""
    return choi_to_kraus(choi)


@to_kraus.register(SuperOp)
def _superop_to_kraus_map(superop: SuperOp) -> KrausMap:
    """Convert SuperOp to KrausMap."""
    return superop_to_kraus(superop)


@to_kraus.register(PauliLiouville)
def _pauli_liouville_to_kraus_map(pauli_liouville: PauliLiouville) -> KrausMap:
    """Convert PauliLiouville to KrausMap."""
    return pauli_liouville_to_kraus(pauli_liouville)


@to_kraus.register(KrausMap)
def _kraus_map_to_kraus_map(kraus_map: KrausMap) -> KrausMap:
    """Identity conversion for KrausMap matrices."""
    return kraus_map


@to_kraus.register(Unitary)
def _unitary_to_kraus_map(unitary: Unitary) -> KrausMap:
    """Convert Unitary to KrausMap."""
    return unitary_to_kraus_map(unitary)


def superop_to_lindbladian(superop: SuperOp) -> "Lindbladian":
    """Recover a Lindbladian generator whose ``evolve(·, 1)`` reproduces the channel ``superop``.

    Computes the principal matrix logarithm of the superoperator (its generator) and decomposes it
    into a canonical ``(H, jump_operators)`` GKSL representation.

    .. note::
        This is exact only for **infinitely divisible** channels (those of the form ``exp(L)`` for a
        valid GKSL generator).  For a general channel the principal log need not be a CP generator;
        the reconstruction clamps to the nearest valid one (best-effort), so the round-trip is not
        guaranteed.  A singular ``superop`` (zero eigenvalue) has no logarithm.

    :param superop: The channel superoperator.
    :return: A :class:`~quax.Lindbladian` generator.
    """
    from ._generators import reconstruct_gksl
    from ._quantum_objects import Lindbladian, Observable, Operator

    dims = superop.dims[1]
    d = reduce(mul, dims, 1)
    matrix = superop.matrix  # (*ensemble, d², d²)
    ensemble_shape = matrix.shape[:-2]
    flat = matrix.reshape((-1, d * d, d * d))

    def _log_then_reconstruct(m: Array):
        eigvals, eigvecs = jnp.linalg.eig(m)
        log_gen = (eigvecs * jnp.log(eigvals)[..., None, :]) @ jnp.linalg.inv(eigvecs)  # principal log
        return reconstruct_gksl(log_gen, d)

    h_flat, l_flat = jax.vmap(_log_then_reconstruct)(flat)
    hamiltonian = Observable.from_matrix(h_flat.reshape(ensemble_shape + (d, d)), (dims, dims))
    jump_operators = Operator.from_matrix(l_flat.reshape(ensemble_shape + (d * d, d, d)), (dims, dims))
    return Lindbladian(hamiltonian=hamiltonian, jump_operators=jump_operators)
