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

"""Module implementing composition for quantum objects."""

import jax
import jax.numpy as jnp

from ._quantum_objects import Choi, KrausMap, PauliLiouville, SuperOp, Unitary
from ._superoperator_transformations import (
    choi_to_superop,
    superop_to_choi,
)


@jax.jit
def compose_kraus(k1: KrausMap, k2: KrausMap) -> KrausMap:
    """
    Given two channels, E1 and E2, acting on the same system in the Kraus representation this
    function return the Kraus operators representing the composition of the channels E1 o E2.

    :param k1: The list of Kraus operators for channel E1 (applied second).
    :param k2: The list of Kraus operators for channel E2 (applied first).
    :return: A combinatorially generated list of composed Kraus operators.
    """
    assert k1.dims == k2.dims, "Kraus operators must act on the same space to be composed."
    dims = k1.dims
    d_out, d_in = k1.d
    n1 = k1.matrix.shape[-3]
    n2 = k2.matrix.shape[-3]

    ensemble_size = jnp.broadcast_shapes(k1.ensemble_size, k2.ensemble_size)
    num_ensemble_dims = len(ensemble_size)
    k1_mat = jnp.broadcast_to(k1.matrix, ensemble_size + (n1, d_out, d_in))
    k2_mat = jnp.broadcast_to(k2.matrix, ensemble_size + (n2, d_out, d_in))

    kraus_data = jnp.einsum("...iab,...jbc->...ijac", k1_mat, k2_mat)
    kraus_data = kraus_data.reshape(ensemble_size + (n1 * n2, d_out, d_in))

    return KrausMap.from_matrix(kraus_data, dims, num_ensemble_dims)


@jax.jit
def compose_unitary(U1: Unitary, U2: Unitary) -> Unitary:
    """Compute the composition of two unitary operators.

    For two unitaries U1 and U2 acting on the same systems, this returns the unitary
    representing U1 ∘ U2 (U2 applied first, then U1).

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles compose element-wise.

    :param U1: Unitary matrix for the first system
    :param U2: Unitary matrix for the second system
    :returns: Unitary matrix for the composed system
    """
    assert U1.dims == U2.dims, "Unitaries must act on the same space to be composed."

    # Use einsum with ellipsis to handle arbitrary ensemble dimensions
    data = jnp.einsum("...ab,...bc->...ac", U1.matrix, U2.matrix)

    ensemble_size = jnp.broadcast_shapes(U1.ensemble_size, U2.ensemble_size)
    num_ensemble_dims = len(ensemble_size)

    return Unitary.from_matrix(data, U1.dims, num_ensemble_dims)


@jax.jit
def compose_superop(S1: SuperOp, S2: SuperOp) -> SuperOp:
    """Compute the composition of two superoperators.

    For two channels E1 and E2 acting on the same system, this returns the superoperator
    representing E1 ∘ E2 (E2 applied first, then E1).

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles compose element-wise.

    :param S1: Superoperator matrix for the first channel (applied second)
    :param S2: Superoperator matrix for the second channel (applied first)
    :returns: Superoperator matrix for the composed channel
    """
    assert S1.dims == S2.dims, "Superoperators must act on the same space to be composed."

    # Use matmul which automatically broadcasts ensemble dimensions
    data = S1.matrix @ S2.matrix

    ensemble_size = jnp.broadcast_shapes(S1.ensemble_size, S2.ensemble_size)
    num_ensemble_dims = len(ensemble_size)

    return SuperOp.from_matrix(data, S1.dims, num_ensemble_dims)


@jax.jit
def compose_choi(J1: Choi, J2: Choi) -> Choi:
    """
    Compute the composition of two Choi matrices.

    For two channels E1 and E2 acting on the same system, this returns the Choi matrix
    representing E1 ∘ E2 (E2 applied first, then E1).

    The composition rule in the Choi representation is:
    J_{E1 ∘ E2} = Tr_B[(J_E1 ⊗ I) (I ⊗ J_E2^T)]

    where the partial trace is over the intermediate system.

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles compose element-wise.

    Note: Due to the complexity of the Choi matrix index conventions (column-stacking),
    this implementation converts to superoperator representation, composes, and converts back.

    :param J1: Choi matrix for the first channel (applied second).
    :param J2: Choi matrix for the second channel (applied first).
    :return: Choi matrix for the composed channel.
    """
    assert J1.dims == J2.dims, "Choi matrices must act on the same space to be composed."

    # Convert to superoperators, compose, and convert back
    # This avoids the complexity of handling column-stacking index conventions
    # The compose_superop function handles ensemble broadcasting
    S1 = choi_to_superop(J1)
    S2 = choi_to_superop(J2)

    S_composed = compose_superop(S1, S2)

    # Convert back to Choi
    J_composed = superop_to_choi(S_composed)

    return J_composed


@jax.jit
def compose_pauli_liouville(P1: PauliLiouville, P2: PauliLiouville) -> PauliLiouville:
    """Compute the composition of two Pauli-Liouville representations.

    For two channels E1 and E2 acting on the same system, this returns the Pauli-Liouville
    matrix representing E1 ∘ E2 (E2 applied first, then E1).

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles compose element-wise.

    :param P1: Pauli-Liouville matrix for the first channel (applied second)
    :param P2: Pauli-Liouville matrix for the second channel (applied first)
    :returns: Pauli-Liouville matrix for the composed channel
    """
    assert P1.dims == P2.dims, "Pauli-Liouville matrices must act on the same space to be composed."

    # Use matmul which automatically broadcasts ensemble dimensions
    data = P1.matrix @ P2.matrix

    ensemble_size = jnp.broadcast_shapes(P1.ensemble_size, P2.ensemble_size)
    num_ensemble_dims = len(ensemble_size)

    return PauliLiouville.from_matrix(data, P1.dims, num_ensemble_dims)
