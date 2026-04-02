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

from ._quantum_objects import Choi, KrausMap, PauliLiouville, SuperOp, Unitary, Operator
from ._promotion import promote_hilbert_space
from ._superoperator_transformations import (
    choi_to_superop,
    superop_to_choi,
)


@jax.jit
def compose_kraus_map(k1: KrausMap, k2: KrausMap) -> KrausMap:
    """
    Given two channels, E1 and E2, acting on the same system in the Kraus representation this
    function returns the KrausMap representing the composition of the channels E1 o E2.

    :param k1: KrausMap for channel E1 (applied second).
    :param k2: KrausMap for channel E2 (applied first).
    :return: A KrausMap containing the composed Kraus decomposition.
    """
    k1, k2 = promote_hilbert_space(k1, k2)
    dims = k1.dims
    d_out, d_in = k1.d
    n1 = k1.matrix.shape[-3]
    n2 = k2.matrix.shape[-3]

    ensemble_size = jnp.broadcast_shapes(k1.ensemble_size, k2.ensemble_size)
    k1_mat = jnp.broadcast_to(k1.matrix, ensemble_size + (n1, d_out, d_in))
    k2_mat = jnp.broadcast_to(k2.matrix, ensemble_size + (n2, d_out, d_in))

    kraus_data = jnp.einsum("...iab,...jbc->...ijac", k1_mat, k2_mat)
    kraus_data = kraus_data.reshape(ensemble_size + (n1 * n2, d_out, d_in))

    return KrausMap.from_matrix(kraus_data, dims)


@jax.jit
def compose_operator(O1: Operator, O2: Operator) -> Operator:
    """Compute the composition of two operators.

    For two operators O1 and O2 acting on the same systems, this returns the operator
    representing O1 ∘ O2 (O2 applied first, then O1).

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles compose element-wise.

    :param O1: Operator matrix for the first system
    :param O2: Operator matrix for the second system
    :returns: Operator matrix for the composed system
    """
    O1, O2 = promote_hilbert_space(O1, O2)

    # Use einsum with ellipsis to handle arbitrary ensemble dimensions
    data = jnp.einsum("...ab,...bc->...ac", O1.matrix, O2.matrix)

    return Operator.from_matrix(data, O1.dims)


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
    op = compose_operator(U1, U2)
    return Unitary(data=op.data, num_qubits=op.num_qubits)


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
    S1, S2 = promote_hilbert_space(S1, S2)

    # Use matmul which automatically broadcasts ensemble dimensions
    data = S1.matrix @ S2.matrix

    return SuperOp.from_matrix(data, S1.dims)


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
    J1, J2 = promote_hilbert_space(J1, J2)

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
    P1, P2 = promote_hilbert_space(P1, P2)

    # Use matmul which automatically broadcasts ensemble dimensions
    data = P1.matrix @ P2.matrix

    return PauliLiouville.from_matrix(data, P1.dims)
