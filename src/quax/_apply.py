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
    State,
    StateVector,
    SuperOp,
    Unitary,
)
from ._superoperator_transformations import choi_to_superop, pauli_liouville_to_superop

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
    assert superop.dims[0] == rho.dims, "Superoperator and density matrix must have the same dims."

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
    assert kraus_map.dims[0] == rho.dims, "Kraus map and density matrix must have the same dims."

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
    """Apply a unitary operator to a state vector.

    Computes |ψ_out⟩ = U |ψ_in⟩

    Supports ensemble broadcasting: unitary and state can have different ensemble sizes,
    and standard NumPy broadcasting rules apply.

    :param unitary: Unitary operator with shape ensemble_size + (d, d)
    :param state: State vector with shape ensemble_size + (d,)
    :return: Transformed state vector with broadcasted ensemble_size + (d,)
    """
    assert unitary.dims[0] == state.dims, "Unitary and state vector must have the same dims."

    # Get matrix representations
    unitary_mat = unitary.matrix
    state_mat = state.matrix

    # Apply unitary: |ψ_out⟩ = U |ψ_in⟩
    # Use einsum with ellipsis for broadcasting
    state_out = jnp.einsum("...ab,...b->...a", unitary_mat, state_mat)

    return StateVector.from_matrix(state_out, state.dims)


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
    :return (num_states, num_observables) array of expectation values.
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
    n = len(psi.dims)
    einsum_str = _generate_unitary_contraction(subsystem, n)
    output_state_vector = jnp.einsum(
        einsum_str,
        unitary.data,
        psi.data,
        optimize="optimal",
    )
    return StateVector(data=output_state_vector, num_qubits=psi.num_qubits)


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
def targeted_apply_kraus_map_trajectory(
    kraus_map: KrausMap, psi: StateVector, key: Array, subsystem: Tuple[int, ...]
) -> StateVector:
    """
    Apply a Kraus map to a state vector probabilistically (Monte Carlo trajectory).

    Applies all Kraus operators K_i to |ψ⟩, computes Born-rule probabilities
    p_i = ⟨ψ|K_i†K_i|ψ⟩, samples one outcome per ensemble element, and returns
    the normalized post-measurement state |ψ'_i⟩ / √p_i.

    Supports ensemble broadcasting between kraus_map, psi, and key.

    :param kraus_map: The Kraus map with data shape (*ens_k, n_kraus, d_out..., d_in...).
    :param psi: The state vector with data shape (*ens_s, d0, d1, ...).
    :param key: A JAX PRNG key (scalar or ensemble of keys with shape (*ens_key,)) for sampling.
    :param subsystem: The qubit indices the operator acts on.
    :return: A state vector with data shape (*broadcast_ens, d0, d1, ...).
    """
    n = len(psi.dims)
    n_kraus = kraus_map.data.shape[kraus_map.num_ensemble_dims]

    # Step 1: Apply all Kraus operators to the state vector.
    # Result shape: (*broadcast_ens, n_kraus, d0, d1, ..., dn-1)
    einsum_str = _generate_kraus_trajectory_contraction(subsystem, n)
    all_outcomes = jnp.einsum(
        einsum_str,
        kraus_map.data,
        psi.data,
        optimize="optimal",
    )

    # Step 2: Compute Born-rule probabilities p_i = ⟨ψ'_i|ψ'_i⟩
    # Sum |amplitude|^2 over qubit dimensions (last n axes), keeping Kraus axis.
    qubit_axes = tuple(range(-n, 0))
    probs = jnp.sum(jnp.abs(all_outcomes) ** 2, axis=qubit_axes)  # (*ens, n_kraus)

    # Step 3: Sample one Kraus outcome per ensemble element.
    # categorical requires a scalar key, so we vmap over key ensemble dims if present.
    logits = jnp.log(jnp.clip(probs, min=1e-30))  # guard against log(0)
    sampled_idx = _batched_categorical(key, logits)  # (*broadcast_ens,)

    # Step 4: Select the sampled states via one-hot multiply.
    one_hot = jax.nn.one_hot(sampled_idx, n_kraus)  # (*ens, n_kraus)
    # Reshape to (*ens, n_kraus, 1, 1, ..., 1) for broadcasting over qubit dims
    one_hot = one_hot.reshape(one_hot.shape + (1,) * n)
    selected = jnp.sum(all_outcomes * one_hot, axis=-(n + 1))  # (*ens, d0, ..., dn-1)

    # Step 5: Normalize by 1/sqrt(p_selected).
    p_selected = jnp.sum(probs * jax.nn.one_hot(sampled_idx, n_kraus), axis=-1)  # (*ens,)
    norm = jnp.sqrt(jnp.clip(p_selected, min=1e-30))
    selected = selected / norm.reshape(norm.shape + (1,) * n)

    return StateVector(data=selected, num_qubits=psi.num_qubits)


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
