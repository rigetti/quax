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

from functools import reduce, singledispatch, lru_cache
from operator import mul
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import (
    Choi,
    DensityMatrix,
    KrausMap,
    Observable,
    Operator,
    PauliLiouville,
    QuantumInstrument,
    State,
    StateVector,
    SuperOp,
    Unitary,
)
from ._superoperator_transformations import choi_to_superop, pauli_liouville_to_superop, superop_to_kraus
from ._promotion import promote, promote_hilbert_space

CHARS = "ijklmnopqrstuvwxyzabcdefghIJKLMNOPQRSTUVWXYZABCDEFGH123456789"


@jax.jit
def apply_superop_to_density_matrix(superop: SuperOp, rho: DensityMatrix) -> DensityMatrix:
    """Apply a superoperator to a density matrix.

    Supports ensemble broadcasting: superop and rho can have different ensemble sizes,
    and standard NumPy broadcasting rules apply.

    :param superop: Superoperator with shape ensemble_size + (d², d²)
    :param rho: Density matrix with shape ensemble_size + (d, d)
    :return: Transformed density matrix with broadcasted ensemble_size + (d, d)
    """
    superop, rho = promote_hilbert_space(superop, rho)

    # Get dimension
    d = rho.d  # Linear dimension
    d2 = rho.d2  # Hilbert-Schmidt dimension

    # Get matrix representations
    superop_mat = superop.matrix
    rho_mat = rho.matrix

    # vec_col(rho) via transpose-last-axes then reshape
    rho_vec = jnp.reshape(jnp.swapaxes(rho_mat, -1, -2), rho.ensemble_size + (d2,))

    # Apply superoperator: vec(ρ_out) = S @ vec(ρ_in)
    # Use einsum with ellipsis for automatic broadcasting
    rho_out_vec = jnp.einsum("...ij,...j->...i", superop_mat, rho_vec)

    # Reshape back to matrix form (un-vectorize)
    # Column-stacking means we need to transpose back
    ensemble_size = jnp.broadcast_shapes(superop.ensemble_size, rho.ensemble_size)

    # Unvec_col: reshape to (..., d, d) then transpose back
    rho_out = jnp.reshape(rho_out_vec, ensemble_size + (d, d))
    rho_out = jnp.swapaxes(rho_out, -1, -2)

    return DensityMatrix.from_matrix(rho_out, rho.dims)


@jax.jit
def apply_choi_to_density_matrix(choi: Choi, rho: DensityMatrix) -> DensityMatrix:
    """Apply a Choi matrix to a density matrix.

    Supports ensemble broadcasting: choi and rho can have different ensemble sizes,
    and standard NumPy broadcasting rules apply.

    :param choi: Choi matrix with shape ensemble_size + (d², d²)
    :param rho: Density matrix with shape ensemble_size + (d, d)
    :return: Transformed density matrix with broadcasted ensemble_size + (d, d)
    """
    # Convert to superoperator and apply
    superop = choi_to_superop(choi)
    return apply_superop_to_density_matrix(superop, rho)


@jax.jit
def apply_kraus_to_density_matrix(kraus_map: KrausMap, rho: DensityMatrix) -> DensityMatrix:
    """Apply a Kraus map to a density matrix.

    Computes E(ρ) = Σ_i K_i ρ K_i†

    Supports ensemble broadcasting: kraus_map and rho can have different ensemble sizes,
    and standard NumPy broadcasting rules apply.

    :param kraus_map: Kraus operators with shape ensemble_size + (n_kraus, d, d)
    :param rho: Density matrix with shape ensemble_size + (d, d)
    :return: Transformed density matrix with broadcasted ensemble_size + (d, d)
    """
    kraus_map, rho = promote_hilbert_space(kraus_map, rho)

    # Get matrix representations
    kraus_mat = kraus_map.matrix
    rho_mat = rho.matrix

    # Apply each Kraus operator: K_i @ rho @ K_i†
    # Use einsum with ellipsis for broadcasting
    # kraus_mat has shape (..., n_kraus, d, d)
    # rho_mat has shape (..., d, d)
    # Result: sum over i of K[..., i, :, :] @ rho[..., :, :] @ K[..., i, :, :].conj().T

    rho_out = jnp.einsum("...iab,...bc,...idc->...ad", kraus_mat, rho_mat, kraus_mat.conj())

    return DensityMatrix.from_matrix(rho_out, rho.dims)


@jax.jit
def apply_pauli_liouville_to_density_matrix(pl: PauliLiouville, rho: DensityMatrix) -> DensityMatrix:
    """Apply a Pauli Liouville representation to a density matrix.

    Supports ensemble broadcasting: pl and rho can have different ensemble sizes,
    and standard NumPy broadcasting rules apply.

    :param pl: Pauli-Liouville matrix with shape ensemble_size + (d², d²)
    :param rho: Density matrix with shape ensemble_size + (d, d)
    :return: Transformed density matrix with broadcasted ensemble_size + (d, d)
    """
    # Convert to superoperator and apply
    superop = pauli_liouville_to_superop(pl)
    return apply_superop_to_density_matrix(superop, rho)


@jax.jit
def apply_unitary_to_state_vector(unitary: Unitary, state: StateVector) -> StateVector:
    r"""Apply a unitary operator to a state vector.

    Computes \|ψ_out⟩ = U \|ψ_in⟩

    Supports ensemble broadcasting: unitary and state can have different ensemble sizes,
    and standard NumPy broadcasting rules apply.

    :param unitary: Unitary operator with shape ensemble_size + (d, d)
    :param state: State vector with shape ensemble_size + (d,)
    :return: Transformed state vector with broadcasted ensemble_size + (d,)
    """
    unitary, state = promote_hilbert_space(unitary, state)

    # Get matrix representations
    unitary_mat = unitary.matrix
    state_mat = state.matrix

    # Apply unitary: |ψ_out⟩ = U |ψ_in⟩
    # Use einsum with ellipsis for broadcasting
    state_out = jnp.einsum("...ab,...b->...a", unitary_mat, state_mat)

    return StateVector.from_matrix(state_out, state.dims)


@jax.jit
def apply_unitary_to_density_matrix(unitary: Unitary, rho: DensityMatrix) -> DensityMatrix:
    r"""Apply a unitary operator to a density matrix.

    Computes :math:`\rho' = U \rho U^\dagger`.

    Supports ensemble broadcasting: unitary and rho can have different ensemble sizes,
    and standard NumPy broadcasting rules apply.

    :param unitary: Unitary operator
    :param rho: Density matrix
    :return: Transformed density matrix
    """
    unitary, rho = promote_hilbert_space(unitary, rho)
    subsystem = tuple(range(len(rho.dims)))
    return targeted_apply_unitary_to_density_matrix(unitary, rho, subsystem)


@jax.jit
def apply_kraus_to_state_vector(kraus_map: Operator, state: StateVector) -> DensityMatrix:
    """Apply a Kraus map to a state vector, resulting in a density matrix."""
    raise NotImplementedError("Applying Kraus operators to state vectors is not yet implemented.")


@singledispatch
def partial_trace(rho, indices: Tuple[int, ...]) -> DensityMatrix:
    raise TypeError("rho must be a DensityMatrix or Choi.")


@partial_trace.register
def _(rho: DensityMatrix, indices: Tuple[int, ...]) -> DensityMatrix:
    dims = tuple(rho.dims)  # e.g. (2,2)
    keep = tuple(indices)

    out_data = _partial_trace_data(rho.matrix, dims=dims, keep=keep)
    out_dims = tuple(dims[i] for i in sorted(keep))
    return DensityMatrix.from_matrix(out_data, out_dims)


@partial_trace.register
def _(rho: Choi, indices: Tuple[int, ...]) -> DensityMatrix:
    dims_out, dims_in = rho.dims  # e.g. ((2,2),(2,2))
    dims_all = tuple(dims_out) + tuple(dims_in)  # e.g. (2,2,2,2)
    keep = tuple(indices)

    out_data = _partial_trace_data(rho.matrix, dims=dims_all, keep=keep)
    out_dims = tuple(dims_all[i] for i in sorted(keep))

    # After tracing arbitrary subsystems, "input vs output" split may not be meaningful,
    # so store dims as a flat tuple by default.
    return DensityMatrix.from_matrix(out_data, out_dims)


@jax.jit(static_argnames=("dims", "keep"))
def _partial_trace_data(data: Array, dims: Tuple[int, ...], keep: Tuple[int, ...]) -> Array:
    """
    Batched partial trace of a matrix.

    data: (..., D, D)
    dims: (d0, d1, ..., dn-1) with prod(dims) == D
    keep: indices of subsystems to keep (in [0, n-1])

    returns: (..., D_keep, D_keep)
    """
    ensemble_shape = data.shape[:-2]
    D = data.shape[-1]
    n = len(dims)

    if reduce(mul, dims, 1) != D:
        raise ValueError(f"prod(dims)={reduce(mul, dims, 1)} must equal matrix dim D={D}.")

    if len(keep) != len(set(keep)):
        raise ValueError("Duplicate indices in keep.")
    if any((k < 0 or k >= n) for k in keep):
        raise IndexError(f"keep must be in [0, {n - 1}].")

    keep = tuple(sorted(keep))
    trace = tuple(i for i in range(n) if i not in keep)

    if len(trace) == 0:
        return data

    d_keep = reduce(mul, (dims[i] for i in keep), 1)
    d_trace = reduce(mul, (dims[i] for i in trace), 1)

    # reshape to (..., dims_row..., dims_col...)
    data_nd = data.reshape(ensemble_shape + dims + dims)

    # permute to (..., keep_row, trace_row, keep_col, trace_col)
    b = len(ensemble_shape)
    perm = (
        tuple(range(b))
        + tuple(b + i for i in keep)
        + tuple(b + i for i in trace)
        + tuple(b + n + i for i in keep)
        + tuple(b + n + i for i in trace)
    )
    data_perm = jnp.transpose(data_nd, perm)

    # group into (..., d_keep, d_trace, d_keep, d_trace)
    data_grp = data_perm.reshape(ensemble_shape + (d_keep, d_trace, d_keep, d_trace))

    # trace over traced subsystem
    out = jnp.trace(data_grp, axis1=-3, axis2=-1)  # (..., d_keep, d_keep)
    return out


@jax.jit
def compute_kraus_observables_from_states(
    kraus_map: KrausMap, input_states: DensityMatrix, observables: Unitary
) -> Array:
    """
    Compute the provided observables for the given input density matrices and process.

    :param kraus_ops: A Kraus Channel.
    :param input_states: A (num_states, d, d) array of density matrices.
    :param observables: A (num_observables, d, d) array of observables.
    :return: ``(num_states, num_observables)`` array of expectation values.
    """
    # Get matrix representations
    input_mat = input_states.matrix
    obs_mat = observables.matrix
    kraus_mat = kraus_map.matrix

    # Compute obs: Tr[ρ Σ K_i† O K_i] = Σ Tr[K_i ρ K_i† O] -> ikj, kl, ilh
    # Kraus ops K†, K (num_kraus, d, d) -> ikj, ilh
    # Observable (num_observables, d, d) -> mkl
    # Input states 𝜌 (num_states, d, d) -> nhj
    # Result (num_states, num_observable) -> nm
    assert len(input_mat.shape) == 3, "input_states must be a (num_states, d, d) array"
    assert len(obs_mat.shape) == 3, "observables must be a (num_observables, d, d) array"
    assert len(kraus_mat.shape) == 3, "kraus_map must be a (num_kraus, d, d) array"
    predicted_expectations = jnp.real(
        jnp.einsum("nhj,ikj,mkl,ilh->nm", input_mat, kraus_mat.conj(), obs_mat, kraus_mat)
    )
    return predicted_expectations


@jax.jit
def compute_choi_observables_from_states(choi: Choi, input_states: DensityMatrix, observables: Unitary) -> Array:
    """
    Compute the provided observables for the given input density matrices and process.

    Uses the relation: Tr[O · E(ρ)] = vec(O)† · S · vec(ρ)
    where S is the superoperator (converted from Choi), E is the channel,
    ρ is the input state, O is the observable, and vec() is column-stacking vectorization.

    :param choi: A Choi matrix object.
    :param input_states: A (num_states, d, d) array of density matrices.
    :param observables: A (num_observables, d, d) array of observables.
    :return: A (num_states, num_observables) array of expectation values.
    """
    # Get matrix representations
    input_mat = input_states.matrix
    obs_mat = observables.matrix
    choi_mat = choi.matrix

    assert len(input_mat.shape) == 3, "input_states must be a (num_states, d, d) array"
    assert len(obs_mat.shape) == 3, "observables must be a (num_observables, d, d) array"
    assert len(choi_mat.shape) == 2, "choi must be a (d², d²) array"

    # Convert Choi to superoperator using existing function
    superop = choi_to_superop(choi)

    # Use the superoperator implementation
    return compute_superop_observables_from_states(superop, input_states, observables)


@jax.jit
def compute_superop_observables_from_states(
    superop: SuperOp, input_states: DensityMatrix, observables: Unitary
) -> Array:
    """
    Compute the provided observables for the given input density matrices and process.

    Uses the relation: Tr[O · E(ρ)] = vec(O)† · S · vec(ρ)
    where S is the superoperator, E is the channel, ρ is the input state,
    O is the observable, and vec() is column-stacking vectorization.

    :param superop: A superoperator matrix object.
    :param input_states: A (num_states, d, d) array of density matrices.
    :param observables: A (num_observables, d, d) array of observables.
    :return: A (num_states, num_observables) array of expectation values.
    """
    # Get matrix representations
    input_mat = input_states.matrix
    obs_mat = observables.matrix
    superop_mat = superop.matrix

    assert len(input_mat.shape) == 3, "input_states must be a (num_states, d, d) array"
    assert len(obs_mat.shape) == 3, "observables must be a (num_observables, d, d) array"
    assert len(superop_mat.shape) == 2, "superop must be a (d², d²) array"

    # Get dimension from superoperator matrix
    d2 = superop.d2[0]

    # Vectorize states and observables using column-stacking convention
    # vec(M)[i*d+j] = M[j,i] for column-stacking
    # So M.T.reshape(-1) gives the column-stacked vector
    vec_states = input_mat.transpose(0, 2, 1).reshape(-1, d2)  # (num_states, d²)
    vec_obs = obs_mat.transpose(0, 2, 1).reshape(-1, d2)  # (num_observables, d²)

    # Compute: Tr[O · E(ρ)] = vec(O)† · S · vec(ρ)
    # result[n,m] = Σ_rs vec_obs[m,r]^* · superop[r,s] · vec_states[n,s]
    predicted_expectations = jnp.real(
        vec_obs.conj() @ superop_mat @ vec_states.T
    ).T  # Transpose to get (num_states, num_observables)

    return predicted_expectations


@jax.jit
def compute_pauli_liouville_observables_from_states(
    pauli_liouville: PauliLiouville, input_states: DensityMatrix, observables: Unitary
) -> Array:
    """
    Compute the provided observables for the given input density matrices and process.

    Converts the Pauli-Liouville representation to superoperator form, then uses:
    Tr[O · E(ρ)] = vec(O)† · S · vec(ρ)
    where S is the superoperator, E is the channel, ρ is the input state,
    O is the observable, and vec() is column-stacking vectorization.

    :param pauli_liouville: A Pauli-Liouville matrix object.
    :param input_states: A (num_states, d, d) array of density matrices.
    :param observables: A (num_observables, d, d) array of observables.
    :return: A (num_states, num_observables) array of expectation values.
    """
    # Get matrix representations
    input_mat = input_states.matrix
    obs_mat = observables.matrix
    pl_mat = pauli_liouville.matrix

    assert len(input_mat.shape) == 3, "input_states must be a (num_states, d, d) array"
    assert len(obs_mat.shape) == 3, "observables must be a (num_observables, d, d) array"
    assert len(pl_mat.shape) == 2, "pauli_liouville must be a (d², d²) array"

    # Convert Pauli-Liouville to superoperator
    superop = pauli_liouville_to_superop(pauli_liouville)

    # Use the superoperator implementation
    return compute_superop_observables_from_states(superop, input_states, observables)


@singledispatch
def estimate(state: State, observable: "Observable") -> Array:
    """Compute the expectation value of a Hermitian observable for a quantum state.

    For a state vector ``|ψ⟩``:  ⟨A⟩ = ⟨ψ|A|ψ⟩

    For a density matrix ``ρ``:   ⟨A⟩ = Tr[A ρ]

    Dispatches on the type of *state*.  Supports arbitrary ensemble broadcasting
    between ``observable`` and ``state``; the result has shape equal to the broadcast
    of their ensemble sizes.

    :param state: A ``StateVector`` or ``DensityMatrix`` (may be ensembled).
    :param observable: A Hermitian ``Observable`` (may be ensembled).
    :return: Real-valued ``Array`` of expectation values with shape
        ``jnp.broadcast_shapes(state.ensemble_size, observable.ensemble_size)``.
    """
    raise TypeError(f"estimate() does not support state type {type(state)!r}. Expected StateVector or DensityMatrix.")


@estimate.register(StateVector)
def _estimate_state_vector(state: StateVector, observable: "Observable") -> Array:
    broadcast_ens = jnp.broadcast_shapes(observable.ensemble_size, state.ensemble_size)
    obs_mat = jnp.broadcast_to(observable.matrix, broadcast_ens + observable.matrix.shape[-2:])
    sv_mat = jnp.broadcast_to(state.matrix, broadcast_ens + state.matrix.shape[-1:])
    # ⟨ψ|A|ψ⟩ = Σ_{ab} ψ*_a A_{ab} ψ_b
    result = jnp.einsum("...a,...ab,...b->...", sv_mat.conj(), obs_mat, sv_mat)
    return jnp.real(result)


@estimate.register(DensityMatrix)
def _estimate_density_matrix(state: DensityMatrix, observable: "Observable") -> Array:
    broadcast_ens = jnp.broadcast_shapes(observable.ensemble_size, state.ensemble_size)
    obs_mat = jnp.broadcast_to(observable.matrix, broadcast_ens + observable.matrix.shape[-2:])
    dm_mat = jnp.broadcast_to(state.matrix, broadcast_ens + state.matrix.shape[-2:])
    # Tr[A ρ] = Σ_{ab} A_{ab} ρ_{ba}
    result = jnp.einsum("...ab,...ba->...", obs_mat, dm_mat)
    return jnp.real(result)


@lru_cache(maxsize=1000)
def _generate_superop_contraction(qubits: Tuple[int, ...], n: int) -> str:
    """
    Generate the einsum string for operating on a density matrix which performs
    Sρ where S is a superoperator.

    :param qubits: The qubit indices of the operator.
    :param n: The number of qubits in the density tensor.
    :return: The einsum string.
    """
    num_qubits = n
    d = 2 * n
    operator_support = len(qubits)
    right_qubits = [q + num_qubits for q in qubits]
    return (
        "..."
        + CHARS[: 2 * operator_support]
        + "".join(CHARS[2 * operator_support + q] for q in right_qubits)
        + "".join(CHARS[2 * operator_support + q] for q in qubits)
        + ","
        + "..."
        + CHARS[2 * operator_support : 2 * operator_support + d]
        + "->"
        + "..."
        + "".join(
            CHARS[qubits.index(i) + operator_support]
            if i in qubits
            else CHARS[right_qubits.index(i)]
            if i in right_qubits
            else CHARS[2 * operator_support + i]
            for i in range(d)
        )
    )


@jax.jit(static_argnames=("subsystem",))
def targeted_apply_superop(superoperator: SuperOp, rho: DensityMatrix, subsystem: Tuple[int, ...]) -> DensityMatrix:
    """
    Apply a superoperator to a density matrix. Sρ

    :param superoperator: The superoperator.
    :param rho: The density matrix.
    :param subsystem: The qubit indices of the operator.
    :return: A density matrix.
    """
    target_dims = tuple(rho.dims[i] for i in subsystem)
    if superoperator.dims[0] != target_dims:
        superoperator = promote(superoperator, target_dims)
    n = len(rho.dims)
    einsum_str = _generate_superop_contraction(subsystem, n)
    super_tensor = superoperator.data
    output_density_tensor = jnp.einsum(
        einsum_str,
        super_tensor,
        rho.data,
        optimize="optimal",
    )
    return DensityMatrix(data=output_density_tensor, num_qubits=rho.num_qubits)


@jax.jit(static_argnames=("subsystem",))
def targeted_apply_kraus_map(kraus_map: KrausMap, rho: DensityMatrix, subsystem: Tuple[int, ...]) -> DensityMatrix:
    """
    Apply a Kraus map to a density matrix. ∑_i K_i ρ K_i†

    :param operator: The unitary or Kraus operator.
    :param rho: The density matrix.
    :param subsystem: The qubit indices of the operator.
    :return: A density matrix.
    """
    target_dims = tuple(rho.dims[i] for i in subsystem)
    if kraus_map.dims[0] != target_dims:
        kraus_map = promote(kraus_map, target_dims)
    n = len(rho.dims)
    einsum_str = _generate_kraus_map_contraction(subsystem, n)
    kraus_tensor = kraus_map.data
    output_density_tensor = jnp.einsum(
        einsum_str,
        kraus_tensor,
        rho.data,
        jnp.conjugate(kraus_tensor),
        optimize="optimal",
    )
    return DensityMatrix(data=output_density_tensor, num_qubits=rho.num_qubits)


@jax.jit(static_argnames=("subsystem",))
def targeted_apply_unitary(unitary: Unitary, psi: StateVector, subsystem: Tuple[int, ...]) -> StateVector:
    """
    Apply a unitary to state vector. U|𝜓⟩

    :param unitary: The unitary.
    :param psi: The state vector.
    :param subsystem: The qubit indices of the operator.
    :return: A state vector.
    """
    target_dims = tuple(psi.dims[i] for i in subsystem)
    if unitary.dims[0] != target_dims:
        unitary = promote(unitary, target_dims)
    n = len(psi.dims)
    einsum_str = _generate_unitary_contraction(subsystem, n)
    output_state_vector = jnp.einsum(
        einsum_str,
        unitary.data,
        psi.data,
        optimize="optimal",
    )
    return StateVector(data=output_state_vector, num_qubits=psi.num_qubits)


@jax.jit(static_argnames=("subsystem",))
def targeted_apply_unitary_to_density_matrix(
    unitary: Unitary, rho: DensityMatrix, subsystem: Tuple[int, ...]
) -> DensityMatrix:
    r"""Apply a unitary to a density matrix on a subsystem. :math:`U \rho U^\dagger`

    Computes :math:`\rho' = U \rho U^\dagger`
    acting only on the qudits specified by *subsystem*.

    :param unitary: The unitary operator.
    :param rho: The density matrix.
    :param subsystem: The qudit indices the operator acts on.
    :return: A density matrix.
    """
    target_dims = tuple(rho.dims[i] for i in subsystem)
    if unitary.dims[0] != target_dims:
        unitary = promote(unitary, target_dims)
    n = len(rho.dims)
    einsum_str = _generate_unitary_density_matrix_contraction(subsystem, n)
    output_density_tensor = jnp.einsum(
        einsum_str,
        unitary.data,
        rho.data,
        jnp.conjugate(unitary.data),
        optimize="optimal",
    )
    return DensityMatrix(data=output_density_tensor, num_qubits=rho.num_qubits)


def _batched_categorical(key: Array, logits: Array) -> Array:
    """
    Sample from categorical distribution, supporting both scalar and batched keys.

    ``jax.random.categorical`` requires a scalar key. When ``key`` has ensemble
    dimensions we vmap over them, broadcasting against ``logits``.

    :param key: A JAX PRNG key with shape ``()`` or ``(*ens_key,)``.
    :param logits: Log-probabilities with shape ``(*ens_logits, n_categories)``.
    :return: Sampled indices with shape ``(*broadcast_ens,)``.
    """
    key_ndim = key.ndim
    if key_ndim == 0:
        # Scalar key: categorical handles batched logits natively.
        return jax.random.categorical(key, logits, axis=-1)

    # Batched keys: broadcast key and logits ensemble shapes, then vmap.
    # key shape: (*ens_key,), logits shape: (*ens_logits, n_kraus)
    ens_key = key.shape
    ens_logits = logits.shape[:-1]
    broadcast_ens = jnp.broadcast_shapes(ens_key, ens_logits)

    key = jnp.broadcast_to(key, broadcast_ens)
    logits = jnp.broadcast_to(logits, broadcast_ens + logits.shape[-1:])

    # Flatten all ensemble dims, vmap over them, then reshape back.
    flat_size = 1
    for s in broadcast_ens:
        flat_size *= s
    flat_keys = key.reshape(flat_size)
    flat_logits = logits.reshape(flat_size, logits.shape[-1])

    flat_samples = jax.vmap(lambda k, l: jax.random.categorical(k, l, axis=-1))(flat_keys, flat_logits)  # noqa: E741
    return flat_samples.reshape(broadcast_ens)


@jax.jit(static_argnames=("subsystem",))
def state_vector_reduced_density_matrix(psi: StateVector, subsystem: Tuple[int, ...]) -> DensityMatrix:
    r"""Compute the reduced density matrix of a state vector on a subsystem.

    Computes :math:`\rho_\text{sub} = \text{Tr}_\text{comp}(|\psi\rangle\langle\psi|)`
    directly from the state vector, without forming the full density matrix.
    The state vector is reshaped into a bipartite matrix
    :math:`\Psi \in \mathbb{C}^{d_\text{sub} \times d_\text{comp}}` and the
    reduced density matrix is computed as :math:`\rho_\text{sub} = \Psi \Psi^\dagger`.

    This uses :math:`O(2^n)` memory (the state vector) instead of the
    :math:`O(2^{2n})` required by forming :math:`|\psi\rangle\langle\psi|`
    and calling :func:`partial_trace`.

    :param psi: State vector with data shape ``(*ens, d0, d1, ...)``.
    :param subsystem: The qudit indices to keep (trace out the complement).
    :return: Reduced density matrix on the subsystem with shape
        ``(*ens, d_sub, d_sub)`` in matrix form.
    """
    num_qubits = len(psi.dims)
    n_ens = psi.num_ensemble_dims

    subsystem_set = set(subsystem)
    complement = tuple(i for i in range(num_qubits) if i not in subsystem_set)
    d_sub = reduce(mul, (psi.dims[q] for q in subsystem), 1)
    d_complement = reduce(mul, (psi.dims[q] for q in complement), 1)

    # Permute: (*ens, subsystem_qubits..., complement_qubits...)
    perm = tuple(range(n_ens)) + tuple(n_ens + q for q in subsystem) + tuple(n_ens + q for q in complement)
    psi_perm = jnp.transpose(psi.data, perm)
    psi_bipartite = psi_perm.reshape(psi.ensemble_size + (d_sub, d_complement))

    # ρ_sub = Ψ Ψ† (trace over complement via matrix multiply)
    rho_sub = jnp.einsum("...ij,...kj->...ik", psi_bipartite, psi_bipartite.conj())

    sub_dims = tuple(psi.dims[q] for q in subsystem)
    return DensityMatrix.from_matrix(rho_sub, sub_dims)


@jax.jit(static_argnames=("subsystem",))
def targeted_apply_kraus_map_trajectory(
    kraus_map: KrausMap, psi: StateVector, key: Array, subsystem: Tuple[int, ...]
) -> StateVector:
    """
    Apply a Kraus map to a state vector probabilistically (Monte Carlo trajectory).

    This function computes Born probabilities via the subsystem
    reduced density matrix instead of materializing all K_i|ψ⟩ simultaneously.

    Algorithm:
      1. Compute M_i = K_i†K_i for each Kraus operator.
      2. Compute ρ_sub = Tr_complement(|ψ⟩⟨ψ|), the reduced density matrix
         on the subsystem qubits.
      3. Born probabilities: p_i = Tr[M_i · ρ_sub].
      4. Sample outcome.
      5. Apply the selected operator.

    Peak memory: O(ens × 2^n) instead of O(ens × n_kraus × 2^n).
    This enables effective batch parallelism that the standard implementation
    cannot achieve, because each of the two full-state passes (steps 2 and 5)
    is a batched operation that scales like a unitary application.

    Supports ensemble broadcasting between psi, key, and kraus_map.

    :param kraus_map: The Kraus map with data shape (*ens_k, n_kraus, d_out..., d_in...).
    :param psi: The state vector with data shape (*ens_s, d0, d1, ...).
    :param key: A JAX PRNG key (scalar or ensemble of keys) for sampling.
    :param subsystem: The qubit indices the operator acts on.
    :return: A state vector with data shape (*broadcast_ens, d0, d1, ...).
    """
    target_dims = tuple(psi.dims[i] for i in subsystem)
    if kraus_map.dims[0] != target_dims:
        kraus_map = promote(kraus_map, target_dims)

    num_qubits = len(psi.dims)
    n_ens_k = kraus_map.num_ensemble_dims

    # --- Step 1: Precompute K†K for each Kraus operator ---
    # kraus_mat: (*ens_k, n_kraus, d_out, d_in)
    # KdagK: (*ens_k, n_kraus, d_in, d_in)
    kraus_mat = kraus_map.matrix
    KdagK = jnp.einsum("...xja,...xjb->...xab", kraus_mat.conj(), kraus_mat)

    # --- Step 2: Compute subsystem reduced density matrix ---
    rho_sub = state_vector_reduced_density_matrix(psi, subsystem)

    # --- Step 3: Born probabilities p_i = Tr[M_i · ρ_sub] ---
    # KdagK: (*ens_k, n_kraus, d_in, d_in), rho_sub: (*ens_s, d_sub, d_sub)
    # Broadcast ensemble dims, then contract: -> (*broadcast_ens, n_kraus)
    rho_sub_mat = rho_sub.matrix
    probs = jnp.real(jnp.einsum("...xab,...ba->...x", KdagK, rho_sub_mat))

    # --- Step 4: Broadcast probs with key ensemble dims ---
    ens_probs = probs.shape[:-1]
    ens_key = key.shape if key.ndim > 0 else ()
    # Key may have more dims than probs (extra "sample" dimensions appended).
    # Pad ens_probs with trailing 1s so broadcast_shapes can align them.
    if len(ens_key) > len(ens_probs):
        ens_probs = ens_probs + (1,) * (len(ens_key) - len(ens_probs))
    broadcast_ens = jnp.broadcast_shapes(ens_probs, ens_key)
    probs = jnp.broadcast_to(probs.reshape(ens_probs + probs.shape[-1:]), broadcast_ens + probs.shape[-1:])

    # --- Step 5: Sample one Kraus outcome per ensemble element ---
    logits = jnp.log(jnp.clip(probs, min=1e-30))
    sampled_idx = _batched_categorical(key, logits)

    # --- Step 6: Gather selected Kraus operator per batch element ---
    # kraus_map.data: (*ens_k, n_kraus, d_out..., d_in...)
    # sampled_idx: (*broadcast_ens,) → selected: (*broadcast_ens, d_out..., d_in...)
    if n_ens_k == 0:
        selected_kraus = kraus_map.data[sampled_idx]
    else:
        # Advanced indexing for ensembled Kraus maps.
        # Pad ens_k with trailing 1s so it broadcasts against broadcast_ens.
        ens_k = kraus_map.ensemble_size
        if len(broadcast_ens) > len(ens_k):
            padded_ens_k = ens_k + (1,) * (len(broadcast_ens) - len(ens_k))
        else:
            padded_ens_k = ens_k
        ens_broadcast = jnp.broadcast_shapes(padded_ens_k, broadcast_ens)
        non_ens_shape = kraus_map.data.shape[n_ens_k:]
        kraus_data = jnp.broadcast_to(
            kraus_map.data.reshape(padded_ens_k + non_ens_shape),
            ens_broadcast + non_ens_shape,
        )
        idx = tuple(
            jnp.arange(s).reshape((1,) * i + (s,) + (1,) * (len(ens_broadcast) - i - 1))
            for i, s in enumerate(ens_broadcast)
        )
        selected_kraus = kraus_data[idx + (sampled_idx,)]

    # --- Step 7: Apply selected operators (batched, like unitary application) ---
    # Broadcast psi data to match broadcast_ens (key may introduce extra dims).
    psi_ens = psi.ensemble_size
    if len(broadcast_ens) > len(psi_ens):
        psi_ens = psi_ens + (1,) * (len(broadcast_ens) - len(psi_ens))
    psi_data = jnp.broadcast_to(
        psi.data.reshape(psi_ens + psi.dims),
        jnp.broadcast_shapes(psi_ens, broadcast_ens) + psi.dims,
    )
    einsum_str = _generate_unitary_contraction(subsystem, num_qubits)
    selected_state = jnp.einsum(
        einsum_str,
        selected_kraus,
        psi_data,
        optimize="optimal",
    )

    # --- Step 8: Normalize ---
    p_selected = jnp.take_along_axis(probs, sampled_idx[..., jnp.newaxis], axis=-1).squeeze(-1)
    norm = jnp.sqrt(jnp.clip(p_selected, min=1e-30))
    selected_state = selected_state / norm.reshape(norm.shape + (1,) * num_qubits)

    return StateVector(data=selected_state, num_qubits=psi.num_qubits)


@lru_cache(maxsize=1000)
def _generate_kraus_map_contraction(qubits: Tuple[int], n: int) -> str:
    """
    Generate the einsum string for operating on a density matrix.

    :param qubits: The qubit indices of the operator.
    :param n: The number of qubits in the density tensor.
    :return: The einsum string.
    """
    d = n * 2
    operator_support = len(qubits)
    right_qubits = [q + n for q in qubits]
    return (
        "..."
        + CHARS[0]
        + CHARS[1 : operator_support + 1]
        + "".join(CHARS[1 + operator_support + q] for q in qubits)
        + ","
        + "..."
        + CHARS[1 + operator_support : (1 + operator_support + d)]
        + ","
        + "..."
        + CHARS[0]
        + CHARS[1 + d + operator_support : 1 + 2 * operator_support + d]
        + "".join(CHARS[1 + operator_support + q] for q in right_qubits)
        + "->"
        + "..."
        + "".join(
            CHARS[1 + qubits.index(i)]
            if i in qubits
            else CHARS[1 + right_qubits.index(i) + operator_support + d]
            if i in right_qubits
            else CHARS[1 + operator_support + i]
            for i in range(d)
        )
    )


@lru_cache(maxsize=1000)
def _generate_unitary_contraction(qubits: Tuple[int], n: int) -> str:
    """
    Generate the einsum string for operating on a state tensor.

    :param qubits: The qubit indices of the operator.
    :param n: The number of dimensions in the state tensor.
    :return: The einsum string.
    """
    operator_support = len(qubits)
    return (
        "..."
        + CHARS[:operator_support]
        + "".join(CHARS[operator_support + q] for q in qubits)
        + ","
        + "..."
        + CHARS[operator_support : (operator_support + n)]
        + "->"
        + "..."
        + "".join(CHARS[qubits.index(i)] if i in qubits else CHARS[operator_support + i] for i in range(n))
    )


@lru_cache(maxsize=1000)
def _generate_unitary_density_matrix_contraction(qubits: Tuple[int, ...], n: int) -> str:
    r"""Generate the einsum string for :math:`U \rho U^\dagger` on a density tensor.

    The density tensor has ``2*n`` qudit axes (bra then ket).  The unitary acts on
    the qudits specified by *qubits*, contracting with the corresponding bra (left)
    and ket (right) indices of the density matrix.

    Three operands: ``U``, ``rho``, ``conj(U)``.

    :param qubits: The qudit indices the operator acts on.
    :param n: The number of qudits in the density tensor.
    :return: The einsum string.
    """
    d = n * 2
    operator_support = len(qubits)
    right_qubits = [q + n for q in qubits]
    # Index layout (no Kraus index, unlike _generate_kraus_map_contraction):
    # CHARS[0 : operator_support]                     = U output (bra) indices
    # CHARS[operator_support + q] for q in qubits     = U input indices (contracted with rho bra)
    # CHARS[operator_support : operator_support + d]   = rho tensor indices
    # CHARS[d + operator_support : d + 2*operator_support] = conj(U) output (ket) indices
    # CHARS[operator_support + q] for q in right_qubits = conj(U) input indices (contracted with rho ket)
    return (
        "..."
        + CHARS[:operator_support]
        + "".join(CHARS[operator_support + q] for q in qubits)
        + ","
        + "..."
        + CHARS[operator_support : operator_support + d]
        + ","
        + "..."
        + CHARS[d + operator_support : d + 2 * operator_support]
        + "".join(CHARS[operator_support + q] for q in right_qubits)
        + "->"
        + "..."
        + "".join(
            CHARS[qubits.index(i)]
            if i in qubits
            else CHARS[right_qubits.index(i) + operator_support + d]
            if i in right_qubits
            else CHARS[operator_support + i]
            for i in range(d)
        )
    )


@lru_cache(maxsize=1000)
def _generate_kraus_trajectory_contraction(qubits: Tuple[int, ...], n: int) -> str:
    """
    Generate the einsum string for applying Kraus operators to a state tensor,
    preserving the Kraus index in the output.

    Produces all K_i|ψ⟩ at once: (*ens_k, n_kraus, d_out..., d_in...), (*ens_s, d...) -> (*ens, n_kraus, d...)

    :param qubits: The qubit indices of the operator.
    :param n: The number of dimensions in the state tensor.
    :return: The einsum string.
    """
    operator_support = len(qubits)
    # CHARS[0] = Kraus index (preserved in output)
    # CHARS[1:1+op_support] = operator output qubit indices
    # CHARS[1+op_support+q] for q in qubits = operator input qubit indices (contracted with state)
    # CHARS[1+op_support : 1+op_support+n] = state tensor indices
    return (
        "..."
        + CHARS[0]
        + CHARS[1 : 1 + operator_support]
        + "".join(CHARS[1 + operator_support + q] for q in qubits)
        + ","
        + "..."
        + CHARS[1 + operator_support : 1 + operator_support + n]
        + "->"
        + "..."
        + CHARS[0]
        + "".join(CHARS[1 + qubits.index(i)] if i in qubits else CHARS[1 + operator_support + i] for i in range(n))
    )


# ======================================================================
# Quantum instrument application
# ======================================================================


def apply_instrument_to_density_matrix(
    instrument: QuantumInstrument,
    rho: DensityMatrix,
) -> tuple[DensityMatrix, Array]:
    r"""Apply a quantum instrument to a density matrix.

    Computes :math:`\mathcal{E}_i(\rho)` for every outcome and returns
    the un-normalised outcome density matrices together with their
    probabilities.

    Supports broadcasting between ensembled states and instruments
    following standard NumPy broadcasting rules.

    :param instrument: The quantum instrument.
    :param rho: Input density matrix, possibly ensembled.
    :return: ``(rho_outs, probs)`` where *rho_outs* has an outcome axis
        in its ensemble dimensions and *probs* has shape
        ``(*ens, n_outcomes)``.
    """
    subsystem = tuple(range(len(rho.dims)))
    return targeted_apply_instrument_to_density_matrix(instrument, rho, subsystem)


def apply_instrument_to_state_vector(
    instrument: QuantumInstrument,
    psi: StateVector,
    key: Array,
) -> tuple[StateVector, Array]:
    r"""Apply a quantum instrument to a state vector (Monte Carlo trajectory).

    Converts per-outcome superoperators to Kraus operators, applies all
    operators to :math:`|\psi\rangle`, computes Born-rule probabilities,
    samples one Kraus operator, and returns the normalised post-measurement
    state vector together with the corresponding outcome index.

    Supports broadcasting between ensembled states and keys.

    :param instrument: The quantum instrument.
    :param psi: Input state vector, possibly ensembled.
    :param key: JAX PRNG key (scalar or ensemble of keys).
    :return: ``(psi_out, outcome)``
    """
    subsystem = tuple(range(len(psi.dims)))
    return targeted_apply_instrument_to_state_vector(instrument, psi, key, subsystem)


def targeted_apply_instrument_to_density_matrix(
    instrument: QuantumInstrument,
    rho: DensityMatrix,
    subsystem: Tuple[int, ...],
) -> tuple[DensityMatrix, Array]:
    """Apply a quantum instrument to specific qudits of a density matrix.

    The instrument acts on the qudits specified by *subsystem*; the remaining
    qudits are left unchanged (identity channel on the complement).

    Returns the un-normalised outcome density matrices and their probabilities.

    Supports broadcasting between ensembled states and instruments.

    :param instrument: The quantum instrument.
    :param rho: Input density matrix.
    :param subsystem: The qudit indices the instrument acts on.
    :return: ``(rho_outs, probs)`` where *rho_outs* has an outcome axis
        in its ensemble dimensions and *probs* has shape
        ``(*ens, n_outcomes)``.
    """
    # Instrument data is already in SuperOp form (outcome axis sits in the ensemble)
    superop = SuperOp(instrument.data, instrument.num_qubits)

    # Expand rho so the outcome axis broadcasts independently of rho's batch axes
    rho_expanded = DensityMatrix(jnp.expand_dims(rho.data, axis=rho.num_ensemble_dims), rho.num_qubits)

    # Apply superoperator on the target subsystem for all outcomes at once
    rho_outs = targeted_apply_superop(superop, rho_expanded, subsystem)

    # Probabilities: Tr[ρ_out] for each outcome
    rho_outs_mat = rho_outs.matrix  # (*ens, n_outcomes, d_total, d_total)
    probs = jnp.real(jnp.trace(rho_outs_mat, axis1=-2, axis2=-1))

    return rho_outs, probs


def select_outcome(
    rho_outs: DensityMatrix,
    probs: Array,
    key: Array,
) -> tuple[DensityMatrix, Array]:
    """Select an outcome from instrument results using a random key.

    Samples an outcome index from the probability distribution and returns
    the normalised post-measurement density matrix.

    :param rho_outs: Un-normalised outcome density matrices (with outcome
        axis in the ensemble dimensions), as returned by
        :func:`apply_instrument_to_density_matrix` or
        :func:`targeted_apply_instrument_to_density_matrix`.
    :param probs: Outcome probabilities, shape ``(*ens, n_outcomes)``.
    :param key: JAX PRNG key (scalar or ensemble of keys).
    :return: ``(rho_out, outcome)`` — the normalised post-measurement
        state and the sampled outcome index.
    """
    n_outcomes = probs.shape[-1]
    rho_outs_mat = rho_outs.matrix  # (*ens, n_outcomes, d_total, d_total)

    # Expand probs/rho_outs_mat so ensemble dims broadcast with the key.
    n_ens_probs = probs.ndim - 1
    if key.ndim > n_ens_probs:
        n_extra = key.ndim - n_ens_probs
        expand_axes = tuple(range(n_ens_probs, n_ens_probs + n_extra))
        probs = jnp.expand_dims(probs, axis=expand_axes)
        rho_outs_mat = jnp.expand_dims(rho_outs_mat, axis=expand_axes)

    logits = jnp.log(jnp.clip(probs, min=1e-30))
    sampled_idx = _batched_categorical(key, logits)

    one_hot = jax.nn.one_hot(sampled_idx, n_outcomes)[..., :, jnp.newaxis, jnp.newaxis]
    selected = jnp.sum(rho_outs_mat * one_hot, axis=-3)

    p_selected = jnp.sum(probs * jax.nn.one_hot(sampled_idx, n_outcomes), axis=-1)
    selected = selected / jnp.clip(p_selected, min=1e-30)[..., jnp.newaxis, jnp.newaxis]

    return DensityMatrix.from_matrix(selected, rho_outs.dims), sampled_idx


def targeted_apply_instrument_to_state_vector(
    instrument: QuantumInstrument,
    psi: StateVector,
    key: Array,
    subsystem: Tuple[int, ...],
) -> tuple[StateVector, Array]:
    """Apply a quantum instrument to specific qudits of a state vector (trajectory).

    The instrument acts on the qudits specified by *subsystem*; the remaining
    qudits are left unchanged.

    Supports broadcasting between ensembled states, instruments, and keys.

    :param instrument: The quantum instrument.
    :param psi: Input state vector.
    :param key: JAX PRNG key for sampling.
    :param subsystem: The qudit indices the instrument acts on.
    :return: ``(psi_out, outcome)``.
    """
    n = len(psi.dims)
    n_outcomes = instrument.num_outcomes
    d_out, d_in = instrument.d
    n_kraus_per_outcome = d_out * d_in
    n_total_kraus = n_outcomes * n_kraus_per_outcome

    # Convert instrument SuperOp data → KrausMap (outcome axis sits in the ensemble)
    kraus_map = superop_to_kraus(SuperOp(instrument.data, instrument.num_qubits))

    # Promote if needed
    target_dims = tuple(psi.dims[i] for i in subsystem)
    if kraus_map.dims[0] != target_dims:
        kraus_map = promote(kraus_map, target_dims)

    # Merge the outcome axis (last instrument ensemble dim) with the Kraus axis
    # kraus_map.data: (*ens_i, n_outcomes, n_kraus_per_outcome, d_out..., d_in...)
    # → (*ens_i, n_total_kraus, d_out..., d_in...)
    kraus_data = kraus_map.data
    n_ens_i = len(instrument.ensemble_size)
    shape = kraus_data.shape
    kraus_data = kraus_data.reshape(shape[:n_ens_i] + (n_total_kraus,) + shape[n_ens_i + 2 :])

    einsum_str = _generate_kraus_trajectory_contraction(subsystem, n)
    all_outcomes = jnp.einsum(einsum_str, kraus_data, psi.data, optimize="optimal")

    # Expand all_outcomes so its ensemble dims broadcast with the key.
    n_ens_outcomes = all_outcomes.ndim - (n + 1)
    if key.ndim > n_ens_outcomes:
        n_extra = key.ndim - n_ens_outcomes
        expand_axes = tuple(range(n_ens_outcomes, n_ens_outcomes + n_extra))
        all_outcomes = jnp.expand_dims(all_outcomes, axis=expand_axes)

    # Born-rule probabilities
    qubit_axes = tuple(range(-n, 0))
    per_kraus_probs = jnp.sum(jnp.abs(all_outcomes) ** 2, axis=qubit_axes)

    # Sample, select, normalize
    logits = jnp.log(jnp.clip(per_kraus_probs, min=1e-30))
    sampled_kraus_idx = _batched_categorical(key, logits)

    one_hot = jax.nn.one_hot(sampled_kraus_idx, n_total_kraus).reshape(
        jax.nn.one_hot(sampled_kraus_idx, n_total_kraus).shape + (1,) * n
    )
    selected = jnp.sum(all_outcomes * one_hot, axis=-(n + 1))

    p_selected = jnp.sum(per_kraus_probs * jax.nn.one_hot(sampled_kraus_idx, n_total_kraus), axis=-1)
    norm = jnp.sqrt(jnp.clip(p_selected, min=1e-30))
    selected = selected / norm.reshape(norm.shape + (1,) * n)

    sampled_outcome = sampled_kraus_idx // n_kraus_per_outcome

    return StateVector(data=selected, num_qubits=psi.num_qubits), sampled_outcome
