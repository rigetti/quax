# Copyright 2021-2023 Rigetti & Co, LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""
operator_tools.apply_superoperator module
-----------------------------------------
A module containing tools for applying superoperators to states.

We have arbitrarily decided to use a column stacking convention.

For more information about the conventions used, look at the file in
/docs/Superoperator representations.md

Further references include:

.. [GRAPTN] Tensor networks and graphical calculus for open quantum systems.
         Wood et al.
         Quant. Inf. Comp. 15, 0579-0811 (2015).
         (no DOI)
         https://arxiv.org/abs/1111.6950

.. [MATQO] On the Matrix Representation of Quantum Operations.
        Nambu et al.
        arXiv: 0504091 (2005).
        https://arxiv.org/abs/quant-ph/0504091

.. [DUAL] On duality between quantum maps and quantum states.
       Zyczkowski et al.
       Open Syst. Inf. Dyn. 11, 3 (2004).
       https://dx.doi.org/10.1023/B:OPSY.0000024753.05661.c2
       https://arxiv.org/abs/quant-ph/0401119

"""

from functools import reduce
from operator import mul
from typing import Tuple

import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Choi, DensityMatrix, KrausMap, PauliLiouville, SuperOp, Unitary
from ._superoperator_transformations import choi_to_superop, pauli_liouville_to_superop


# @jax.jit(static_argnames=("indices",))
def partial_trace(rho: DensityMatrix, indices: Tuple[int, ...]):
    r"""
    Calculate the partial trace.

    Consider a joint state ρ on the Hilbert space :math:`H_a \otimes H_b`. We wish to trace out
    :math:`H_b`

    .. math::

        ρ_a = Tr_b(ρ)

    :param rho: 2D array, the matrix to trace.
    :param indices: An tuple of indices of the spaces to keep after being traced. For instance,
                 if the space is A x B x C x D and we want to trace out B and D, keep = [0, 2].
    :return:  ρ_a, a 2D array i.e. the traced matrix
    """
    # Dimension handling and validation (static)
    dims: Tuple[int, ...] = rho.dims  # e.g. (2,2,4)
    n = len(dims)

    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate indices in indices.")
    if any((k < 0 or k >= n) for k in indices):
        raise IndexError(f"indices must be in [0, {n - 1}].")

    indices = tuple(sorted(indices))  # sort the indices
    trace = tuple(i for i in range(n) if i not in indices)

    if len(trace) == 0:
        return rho  # keep everything

    d_keep = reduce(mul, [dims[i] for i in indices], 1)
    d_trace = reduce(mul, [dims[i] for i in trace], 1)

    # Permute axes to: keep_row, trace_row, keep_col, trace_col
    perm = indices + trace + tuple(i + n for i in indices) + tuple(i + n for i in trace)

    new_subdims = tuple(dims[i] for i in indices)

    # Core math (jitted) -----------------
    rho_nd = rho.data.reshape(dims + dims)

    rho_perm = jnp.transpose(rho_nd, perm)

    # Group into (d_keep, d_trace, d_keep, d_trace)
    rho_grp = rho_perm.reshape((d_keep, d_trace, d_keep, d_trace))

    # Trace over the traced subsystem: sum over matching trace indices
    # result shape: (d_keep, d_keep)
    rho_red = jnp.trace(rho_grp, axis1=1, axis2=3)
    return type(rho)(data=rho_red, dims=new_subdims)


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
