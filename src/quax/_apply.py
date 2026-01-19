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

from functools import reduce, singledispatch
from operator import mul
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Choi, DensityMatrix, Kraus, KrausMap, PauliLiouville, StateVector, SuperOp, Unitary
from ._superoperator_transformations import choi_to_superop, pauli_liouville_to_superop


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

    # vec_col(rho) via transpose-last-axes then reshape
    rho_vec = jnp.reshape(jnp.swapaxes(rho.data, -1, -2), rho.ensemble_size + (d2,))

    # Apply superoperator: vec(ρ_out) = S @ vec(ρ_in)
    # Use einsum with ellipsis for automatic broadcasting
    rho_out_vec = jnp.einsum("...ij,...j->...i", superop.data, rho_vec)

    # Reshape back to matrix form (un-vectorize)
    # Column-stacking means we need to transpose back
    ensemble_size = jnp.broadcast_shapes(superop.ensemble_size, rho.ensemble_size)

    # Unvec_col: reshape to (..., d, d) then transpose back
    rho_out = jnp.reshape(rho_out_vec, ensemble_size + (d, d))
    rho_out = jnp.swapaxes(rho_out, -1, -2)

    return DensityMatrix(data=rho_out, dims=rho.dims)


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

    # Apply each Kraus operator: K_i @ rho @ K_i†
    # Use einsum with ellipsis for broadcasting
    # kraus_map.data has shape (..., n_kraus, d, d)
    # rho.data has shape (..., d, d)
    # Result: sum over i of K[..., i, :, :] @ rho[..., :, :] @ K[..., i, :, :].conj().T

    rho_out = jnp.einsum("...iab,...bc,...idc->...ad", kraus_map.data, rho.data, kraus_map.data.conj())

    return DensityMatrix(data=rho_out, dims=rho.dims)


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

    # Apply unitary: |ψ_out⟩ = U |ψ_in⟩
    # Use einsum with ellipsis for broadcasting
    state_out = jnp.einsum("...ab,...b->...a", unitary.data, state.data)

    return StateVector(data=state_out, dims=state.dims)


def apply_kraus_to_state_vector(kraus_map: Kraus, state: StateVector) -> DensityMatrix:
    """Apply a Kraus map to a state vector, resulting in a density matrix."""
    raise NotImplementedError("Applying Kraus operators to state vectors is not yet implemented.")


@singledispatch
def partial_trace(rho, indices: Tuple[int, ...]) -> DensityMatrix:
    raise TypeError("rho must be a DensityMatrix or Choi.")


@partial_trace.register
def _(rho: DensityMatrix, indices: Tuple[int, ...]) -> DensityMatrix:
    dims = tuple(rho.dims)  # e.g. (2,2)
    keep = tuple(indices)

    out_data = _partial_trace_data(rho.data, dims=dims, keep=keep)
    out_dims = tuple(dims[i] for i in sorted(keep))
    return DensityMatrix(data=out_data, dims=out_dims)


@partial_trace.register
def _(rho: Choi, indices: Tuple[int, ...]) -> DensityMatrix:
    dims_in, dims_out = rho.dims  # e.g. ((2,2),(2,2))
    dims_all = tuple(dims_in) + tuple(dims_out)  # e.g. (2,2,2,2)
    keep = tuple(indices)

    out_data = _partial_trace_data(rho.data, dims=dims_all, keep=keep)
    out_dims = tuple(dims_all[i] for i in sorted(keep))

    # After tracing arbitrary subsystems, "input vs output" split may not be meaningful,
    # so store dims as a flat tuple by default.
    return DensityMatrix(data=out_data, dims=out_dims)


@jax.jit(static_argnames=("dims", "keep"))
def _partial_trace_data(data: Array, dims: Tuple[int, ...], keep: Tuple[int, ...]) -> Array:
    """
    Batched partial trace of a matrix.

    data: (..., D, D)
    dims: (d0, d1, ..., dn-1) with prod(dims) == D
    keep: indices of subsystems to keep (in [0, n-1])

    returns: (..., D_keep, D_keep)
    """
    batch_shape = data.shape[:-2]
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
    data_nd = data.reshape(batch_shape + dims + dims)

    # permute to (..., keep_row, trace_row, keep_col, trace_col)
    b = len(batch_shape)
    perm = (
        tuple(range(b))
        + tuple(b + i for i in keep)
        + tuple(b + i for i in trace)
        + tuple(b + n + i for i in keep)
        + tuple(b + n + i for i in trace)
    )
    data_perm = jnp.transpose(data_nd, perm)

    # group into (..., d_keep, d_trace, d_keep, d_trace)
    data_grp = data_perm.reshape(batch_shape + (d_keep, d_trace, d_keep, d_trace))

    # trace over traced subsystem
    out = jnp.trace(data_grp, axis1=-3, axis2=-1)  # (..., d_keep, d_keep)
    return out


# @jax.jit(static_argnames=("indices",))
# def partial_trace(rho: DensityMatrix | Choi, indices: Tuple[int, ...]):
#     r"""
#     Calculate the partial trace.

#     Consider a joint state ρ on the Hilbert space :math:`H_a \otimes H_b`. We wish to trace out
#     :math:`H_b`

#     .. math::

#         ρ_a = Tr_b(ρ)

#     :param rho: 2D array, the matrix to trace.
#     :param indices: An tuple of indices of the spaces to keep after being traced. For instance,
#                  if the space is A x B x C x D and we want to trace out B and D, keep = [0, 2].
#     :return:  ρ_a, a 2D array i.e. the traced matrix
#     """
#     # Dimension handling and validation (static)
#     if isinstance(rho, Choi):
#         dims: Tuple[int, ...] = rho.dims[0]  # e.g. (2,2,4)
#         new_subdims = (tuple(dims[i] for i in indices), tuple(dims[i] for i in indices))
#     elif isinstance(rho, DensityMatrix):
#         dims: Tuple[int, ...] = rho.dims  # e.g. (2,2,4)
#         new_subdims = tuple(dims[i] for i in indices)
#     else:
#         raise TypeError("rho must be a DensityMatrix or Choi object.")

#     n = len(dims)

#     if len(indices) != len(set(indices)):
#         raise ValueError("Duplicate indices in indices.")
#     if any((k < 0 or k >= n) for k in indices):
#         raise IndexError(f"indices must be in [0, {n - 1}].")

#     indices = tuple(sorted(indices))  # sort the indices
#     trace = tuple(i for i in range(n) if i not in indices)

#     if len(trace) == 0:
#         return rho  # keep everything

#     d_keep = reduce(mul, [dims[i] for i in indices], 1)
#     d_trace = reduce(mul, [dims[i] for i in trace], 1)

#     # Permute axes to: keep_row, trace_row, keep_col, trace_col
#     perm = indices + trace + tuple(i + n for i in indices) + tuple(i + n for i in trace)

#     new_subdims = tuple(dims[i] for i in indices)

#     # Core math (jitted) -----------------
#     rho_nd = rho.data.reshape(dims + dims)

#     rho_perm = jnp.transpose(rho_nd, perm)

#     # Group into (d_keep, d_trace, d_keep, d_trace)
#     rho_grp = rho_perm.reshape((d_keep, d_trace, d_keep, d_trace))

#     # Trace over the traced subsystem: sum over matching trace indices
#     # result shape: (d_keep, d_keep)
#     rho_red = jnp.trace(rho_grp, axis1=1, axis2=3)
#     return type(rho)(data=rho_red, dims=new_subdims)


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
    # Compute obs: Tr[ρ Σ K_i† O K_i] = Σ Tr[K_i ρ K_i† O] -> ikj, kl, ilh
    # Kraus ops K†, K (num_kraus, d, d) -> ikj, ilh
    # Observable (num_observables, d, d) -> mkl
    # Input states 𝜌 (num_states, d, d) -> nhj
    # Result (num_states, num_observable) -> nm
    assert len(input_states.data.shape) == 3, "input_states must be a (num_states, d, d) array"
    assert len(observables.data.shape) == 3, "observables must be a (num_observables, d, d) array"
    assert len(kraus_map.data.shape) == 3, "kraus_map must be a (num_kraus, d, d) array"
    predicted_expectations = jnp.real(
        jnp.einsum("nhj,ikj,mkl,ilh->nm", input_states.data, kraus_map.data.conj(), observables.data, kraus_map.data)
    )
    return predicted_expectations


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
    assert len(input_states.data.shape) == 3, "input_states must be a (num_states, d, d) array"
    assert len(observables.data.shape) == 3, "observables must be a (num_observables, d, d) array"
    assert len(choi.data.shape) == 2, "choi must be a (d², d²) array"

    # Convert Choi to superoperator using existing function
    superop = choi_to_superop(choi)

    # Use the superoperator implementation
    return compute_superop_observables_from_states(superop, input_states, observables)


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
    assert len(input_states.data.shape) == 3, "input_states must be a (num_states, d, d) array"
    assert len(observables.data.shape) == 3, "observables must be a (num_observables, d, d) array"
    assert len(superop.data.shape) == 2, "superop must be a (d², d²) array"

    # Get dimension from superoperator matrix
    d2 = superop.d2[0]

    # Vectorize states and observables using column-stacking convention
    # vec(M)[i*d+j] = M[j,i] for column-stacking
    # So M.T.reshape(-1) gives the column-stacked vector
    vec_states = input_states.data.transpose(0, 2, 1).reshape(-1, d2)  # (num_states, d²)
    vec_obs = observables.data.transpose(0, 2, 1).reshape(-1, d2)  # (num_observables, d²)

    # Compute: Tr[O · E(ρ)] = vec(O)† · S · vec(ρ)
    # result[n,m] = Σ_rs vec_obs[m,r]^* · superop[r,s] · vec_states[n,s]
    predicted_expectations = jnp.real(
        vec_obs.conj() @ superop.data @ vec_states.T
    ).T  # Transpose to get (num_states, num_observables)

    return predicted_expectations


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
    assert len(input_states.data.shape) == 3, "input_states must be a (num_states, d, d) array"
    assert len(observables.data.shape) == 3, "observables must be a (num_observables, d, d) array"
    assert len(pauli_liouville.data.shape) == 2, "pauli_liouville must be a (d², d²) array"

    # Convert Pauli-Liouville to superoperator
    superop = pauli_liouville_to_superop(pauli_liouville)

    # Use the superoperator implementation
    return compute_superop_observables_from_states(superop, input_states, observables)
