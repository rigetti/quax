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

"""Module implementing tensor product for quantum objects."""

from functools import reduce
from operator import mul
from typing import List

import jax
import jax.numpy as jnp

from ._quantum_objects import Choi, DensityMatrix, Kraus, KrausMap, PauliLiouville, StateVector, SuperOp, Unitary
from ._superoperator_transformations import (
    choi_to_pauli_liouville,
    choi_to_superop,
    pauli_liouville_to_choi,
    superop_to_choi,
)


@jax.jit
def tensor_choi(choi_0: Choi, choi_1: Choi) -> Choi:
    """
    Compute the tensor product of two Choi matrices.

    Choi tensor product for product channel E0 ⊗ E1 in the convention
      J[(a,i),(b,j)] = <a| E(|i><j|) |b>
    i.e. J reshapes as (a, i, b, j).

    Returns the two-qubit Choi with grouped indices:
      (a0,a1, i0,i1, b0,b1, j0,j1) -> (a, i, b, j) -> matrix.

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles tensor element-wise.
    """

    d0_out_dims, d0_in_dims = choi_0.dims
    d1_out_dims, d1_in_dims = choi_1.dims
    d0_in = reduce(mul, d0_in_dims)
    d0_out = reduce(mul, d0_out_dims)
    d1_in = reduce(mul, d1_in_dims)
    d1_out = reduce(mul, d1_out_dims)

    d_in = d0_in * d1_in
    d_out = d0_out * d1_out
    new_dims = (d0_out_dims + d1_out_dims, d0_in_dims + d1_in_dims)

    # In our convention, the Choi matrix J is such that J.reshape(d_out, d_in, d_out, d_in)
    # has indices (a, i, b, j) where ``|a><b|`` is in L(H_out) and ``|i><j|`` is in L(H_in).
    # This corresponds to J_{ai,bj}
    J0 = choi_0.matrix.reshape(choi_0.ensemble_size + (d0_out, d0_in, d0_out, d0_in))
    J1 = choi_1.matrix.reshape(choi_1.ensemble_size + (d1_out, d1_in, d1_out, d1_in))

    # Build J_tensored_{a0a1,i0i1,b0b1,j0j1} using einsum with ellipsis for ensemble dims
    J = jnp.einsum("...aibj,...ckdl->...acikbdjl", J0, J1)

    ensemble_size = jnp.broadcast_shapes(choi_0.ensemble_size, choi_1.ensemble_size)
    num_ensemble_dims = len(ensemble_size)
    data = J.reshape(ensemble_size + (d_out * d_in, d_out * d_in))

    return Choi.from_matrix(data, new_dims, num_ensemble_dims)


def tensor_channel_kraus(k1: List[Kraus], k2: List[Kraus]) -> List[Kraus]:
    r"""
    Given the Kraus representation for two channels, :math:`\mathcal E_1` and :math:`\mathcal E_2`,
    acting on different systems this function returns the Kraus operators representing the
    tensor product of these channels, :math:`\mathcal E_2 \otimes \mathcal E_1`.

    Suppose :math:`\mathcal E_1` and :math:`\mathcal E_2` each have one Kraus operator,
    :math:`K_1 = X` and :math:`K_2 = H`. Then this function returns a single Kraus operator
    for the tensor product channel:

    .. math::

        K_{\rm tot} = H \otimes X

    :param k1: The list of Kraus operators on the first system.
    :param k2: The list of Kraus operators on the second system.
    :return: A list of tensored Kraus operators.
    """
    assert len(k1) > 0 and len(k2) > 0
    dims1 = k1[0].dims
    dims2 = k2[0].dims
    # dims are (out, in), and each is a tuple of dimensions for subsystems
    new_dims = (dims2[0] + dims1[0], dims2[1] + dims1[1])

    kraus_data = [jnp.kron(k2l.matrix, k1j.matrix) for k1j in k1 for k2l in k2]
    return [Kraus.from_matrix(kd, new_dims, 0) for kd in kraus_data]


@jax.jit
def tensor_kraus(k1: KrausMap, k2: KrausMap) -> KrausMap:
    """
    Generate Kraus map for the tensor product channel E = E1 ⊗ E2.

    If E1 has Kraus {A_i} and E2 has Kraus {B_j}, then E has Kraus {A_i ⊗ B_j}.
    """
    new_dims = (k1.dims[0] + k2.dims[0], k1.dims[1] + k2.dims[1])

    K1 = k1.matrix
    K2 = k2.matrix

    ensemble_size_1 = k1.ensemble_size
    ensemble_size_2 = k2.ensemble_size

    d1_out, d1_in = k1.d
    d2_out, d2_in = k2.d

    n1 = K1.shape[-3]
    n2 = K2.shape[-3]

    # Broadcast ensemble dims to a common leading shape
    ensemble_size_12 = jnp.broadcast_shapes(ensemble_size_1, ensemble_size_2)
    num_ensemble_dims = len(ensemble_size_12)
    K1b = jnp.broadcast_to(K1, ensemble_size_12 + (n1, d1_out, d1_in))
    K2b = jnp.broadcast_to(K2, ensemble_size_12 + (n2, d2_out, d2_in))

    # Introduce pair axes i and j in an ensemble-safe way:
    # A: (..., N1, 1, d1_out, d1_in)
    # B: (..., 1, N2, d2_out, d2_in)
    A = K1b[..., :, None, :, :]
    B = K2b[..., None, :, :, :]

    # Kronecker per (i,j), preserving ensemble dims:
    # (..., N1, N2, d1_out, d1_in) x (..., N1, N2, d2_out, d2_in)
    # -> (..., N1, N2, d1_out, d2_out, d1_in, d2_in)
    tensor6 = jnp.einsum("...ijab,...ijcd->...ijacbd", A, B)

    # Reshape to (..., N1, N2, d_out, d_in)
    d_out = d1_out * d2_out
    d_in = d1_in * d2_in
    tensor4 = tensor6.reshape(ensemble_size_12 + (n1 * n2, d_out, d_in))

    # Collapse (N1, N2) -> (N1*N2)
    tensor_data = tensor4.reshape(ensemble_size_12 + (n1 * n2, d_out, d_in))

    return KrausMap.from_matrix(tensor_data, new_dims, num_ensemble_dims)


@jax.jit
def tensor_unitary(U1: Unitary, U2: Unitary) -> Unitary:
    """
    Compute the tensor product of two unitary operators.

    For two unitaries U1 and U2 acting on different systems, this returns the unitary
    representing U1 ⊗ U2.

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles tensor element-wise.

    :param U1: Unitary matrix for the first system.
    :param U2: Unitary matrix for the second system.
    :return: Unitary matrix for the tensor product system.
    """
    new_dims = (U1.dims[0] + U2.dims[0], U1.dims[1] + U2.dims[1])
    m, n = U1.d
    p, q = U2.d

    # Use einsum with ellipsis to handle arbitrary ensemble dimensions
    out = jnp.einsum("...ab,...cd->...acbd", U1.matrix, U2.matrix)

    ensemble_size = jnp.broadcast_shapes(U1.ensemble_size, U2.ensemble_size)
    num_ensemble_dims = len(ensemble_size)
    data = out.reshape(ensemble_size + (m * p, n * q))

    return Unitary.from_matrix(data, new_dims, num_ensemble_dims)


@jax.jit
def tensor_superop(S1: SuperOp, S2: SuperOp) -> SuperOp:
    """
    Compute the tensor product of two superoperators.

    For two channels E1 and E2 acting on different systems, this returns the superoperator
    representing E1 ⊗ E2.

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles tensor element-wise.

    :param S1: Superoperator matrix for the first system.
    :param S2: Superoperator matrix for the second system.
    :return: Superoperator matrix for the tensor product system.
    """
    # In order to do this directly we need a reshuffle function.
    # Instead, we delegate to tensor_choi which handles the reshuffle correctly.
    return choi_to_superop(tensor_choi(superop_to_choi(S1), superop_to_choi(S2)))


@jax.jit
def tensor_pauli_liouville(P1: PauliLiouville, P2: PauliLiouville) -> PauliLiouville:
    """
    Compute the tensor product of two Pauli-Liouville representations.

    For two channels E1 and E2 acting on different systems, this returns the Pauli-Liouville
    matrix representing E2 ⊗ E1.

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles tensor element-wise.

    :param P1: Pauli-Liouville matrix for the first channel.
    :param P2: Pauli-Liouville matrix for the second channel.
    :return: Pauli-Liouville matrix for the tensor product channel.
    """
    # Convert to Choi, tensor, and convert back
    # The tensor_choi function handles ensemble broadcasting
    return choi_to_pauli_liouville(tensor_choi(pauli_liouville_to_choi(P1), pauli_liouville_to_choi(P2)))


@jax.jit
def tensor_state_vector(psi1: StateVector, psi2: StateVector) -> StateVector:
    """
    Compute the tensor product of two state vectors.

    For two state vectors ``|ψ1⟩`` and ``|ψ2⟩`` acting on different systems, this returns the state vector
    representing ``|ψ1⟩ ⊗ |ψ2⟩``.

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles tensor element-wise.

    :param psi1: State vector for the first system.
    :param psi2: State vector for the second system.
    :return: State vector for the tensor product system.
    """
    new_dims = psi1.dims + psi2.dims
    d1 = reduce(mul, psi1.dims)
    d2 = reduce(mul, psi2.dims)

    # Use einsum with ellipsis to handle arbitrary ensemble dimensions
    # |ψ1⟩ ⊗ |ψ2⟩ -> concatenate the tensor factors
    out = jnp.einsum("...a,...b->...ab", psi1.matrix, psi2.matrix)

    ensemble_size = jnp.broadcast_shapes(psi1.ensemble_size, psi2.ensemble_size)
    num_ensemble_dims = len(ensemble_size)
    data = out.reshape(ensemble_size + (d1 * d2,))

    return StateVector.from_matrix(data, new_dims, num_ensemble_dims)


@jax.jit
def tensor_density_matrix(rho1: DensityMatrix, rho2: DensityMatrix) -> DensityMatrix:
    """
    Compute the tensor product of two density matrices.

    For two density matrices ρ1 and ρ2 acting on different systems, this returns the density matrix
    representing ρ1 ⊗ ρ2.

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles tensor element-wise.

    :param rho1: Density matrix for the first system.
    :param rho2: Density matrix for the second system.
    :return: Density matrix for the tensor product system.
    """
    new_dims = rho1.dims + rho2.dims
    d1 = reduce(mul, rho1.dims)
    d2 = reduce(mul, rho2.dims)

    # Use einsum with ellipsis to handle arbitrary ensemble dimensions
    # ρ1 ⊗ ρ2: (d1, d1) ⊗ (d2, d2) -> (d1*d2, d1*d2)
    # The tensor product of matrices is: (ρ1 ⊗ ρ2)_{(i1,i2),(j1,j2)} = ρ1_{i1,j1} * ρ2_{i2,j2}
    out = jnp.einsum("...ab,...cd->...acbd", rho1.matrix, rho2.matrix)

    ensemble_size = jnp.broadcast_shapes(rho1.ensemble_size, rho2.ensemble_size)
    num_ensemble_dims = len(ensemble_size)
    data = out.reshape(ensemble_size + (d1 * d2, d1 * d2))

    return DensityMatrix.from_matrix(data, new_dims, num_ensemble_dims)
