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

import jax
import jax.numpy as jnp

from ._quantum_objects import (
    Choi,
    DensityMatrix,
    Involution,
    KrausMap,
    Lindbladian,
    Observable,
    Operator,
    PauliLiouville,
    QuantumInstrument,
    StateVector,
    SuperOp,
    Unitary,
)
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
    ``J[(a,i),(b,j)] = <a| E(|i><j|) |b>``,
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
    data = J.reshape(ensemble_size + (d_out * d_in, d_out * d_in))

    return Choi.from_matrix(data, new_dims)


def tensor_channel_kraus(k1: list[Operator], k2: list[Operator]) -> list[Operator]:
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
    return [Operator.from_matrix(kd, new_dims) for kd in kraus_data]


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

    return KrausMap.from_matrix(tensor_data, new_dims)


@jax.jit
def tensor_operator(O1: Operator, O2: Operator) -> Operator:
    """
    Compute the tensor product of two operators.

    For two operators O1 and O2 acting on different systems, this returns the operator
    representing O1 ⊗ O2.

    Supports ensemble broadcasting: empty ensemble broadcasts with any ensemble,
    and matching ensembles tensor element-wise.

    :param O1: Operator for the first system.
    :param O2: Operator for the second system.
    :return: Operator for the tensor product system.
    """
    new_dims = (O1.dims[0] + O2.dims[0], O1.dims[1] + O2.dims[1])
    m, n = O1.d
    p, q = O2.d

    # Use einsum with ellipsis to handle arbitrary ensemble dimensions
    out = jnp.einsum("...ab,...cd->...acbd", O1.matrix, O2.matrix)

    ensemble_size = jnp.broadcast_shapes(O1.ensemble_size, O2.ensemble_size)
    data = out.reshape(ensemble_size + (m * p, n * q))

    return Operator.from_matrix(data, new_dims)


def tensor_lindbladian(L1: Lindbladian, L2: Lindbladian) -> Lindbladian:
    """Tensor product of two Lindbladian generators on independent subsystems.

    ``L_A | L_B`` gives the combined generator for the joint system A⊗B, built at the operator
    level: the jump operators are ``{L_k^A ⊗ I_B} ∪ {I_A ⊗ L_j^B}`` and the Hamiltonian is
    ``H_A ⊗ I_B + I_A ⊗ H_B``, using quax's index-interleaving convention so that
    ``evolve(L_A | L_B, t) == evolve(L_A, t) | evolve(L_B, t)``.

    Supports ensemble broadcasting: an empty ensemble broadcasts with any ensemble, and matching
    ensembles tensor element-wise (following the same convention as the other ``tensor_*``
    functions). The leading ``n_ops`` axis of the jump operators is kept distinct from the ensemble
    axes throughout.

    :param L1: Lindbladian generator for the first system.
    :param L2: Lindbladian generator for the second system.
    :return: Lindbladian generator for the tensor product system.
    """
    dims_A, dims_B = L1.dims[0], L2.dims[0]
    dA = reduce(mul, dims_A, 1)
    dB = reduce(mul, dims_B, 1)
    dAB = dA * dB
    I_A = Operator.from_matrix(jnp.eye(dA, dtype=complex), (dims_A, dims_A))
    I_B = Operator.from_matrix(jnp.eye(dB, dtype=complex), (dims_B, dims_B))
    joint_dims = (dims_A + dims_B, dims_A + dims_B)

    # Common ensemble shape the two operands broadcast to (the jump operators' own ensemble axes,
    # excluding the leading n_ops axis that tensor_operator otherwise folds into the batch).
    ensemble = jnp.broadcast_shapes(L1.ensemble_size, L2.ensemble_size)

    # Jump operators embedded into the joint space, then stacked along the n_ops axis.  Each stack
    # has shape (*ensemble_i, n_ops_i, dAB, dAB); broadcast the ensemble axes (leaving n_ops
    # distinct) before concatenating, since jnp.concatenate does not broadcast its other axes.
    jumps_A = tensor_operator(L1.jump_operators, I_B).matrix  # {L_k^A ⊗ I_B}
    jumps_B = tensor_operator(I_A, L2.jump_operators).matrix  # {I_A ⊗ L_j^B}
    n_ops_A = jumps_A.shape[-3]
    n_ops_B = jumps_B.shape[-3]
    jumps_A = jnp.broadcast_to(jumps_A, ensemble + (n_ops_A, dAB, dAB))
    jumps_B = jnp.broadcast_to(jumps_B, ensemble + (n_ops_B, dAB, dAB))
    combined_jumps = Operator.from_matrix(jnp.concatenate([jumps_A, jumps_B], axis=-3), joint_dims)

    # Joint Hamiltonian H_A ⊗ I_B + I_A ⊗ H_B (skipping absent coherent terms), each broadcast to
    # the common ensemble shape before summing.
    h_terms = []
    if L1.hamiltonian is not None:
        h_terms.append(jnp.broadcast_to(tensor_operator(L1.hamiltonian, I_B).matrix, ensemble + (dAB, dAB)))
    if L2.hamiltonian is not None:
        h_terms.append(jnp.broadcast_to(tensor_operator(I_A, L2.hamiltonian).matrix, ensemble + (dAB, dAB)))
    hamiltonian = Observable.from_matrix(reduce(jnp.add, h_terms), joint_dims) if h_terms else None

    return Lindbladian(hamiltonian=hamiltonian, jump_operators=combined_jumps)


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
    op = tensor_operator(U1, U2)
    return Unitary(data=op.data, num_qubits=op.num_qubits)


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
    data = out.reshape(ensemble_size + (d1 * d2,))

    return StateVector.from_matrix(data, new_dims)


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
    data = out.reshape(ensemble_size + (d1 * d2, d1 * d2))

    return DensityMatrix.from_matrix(data, new_dims)


@jax.jit
def tensor_observable(O1: Observable, O2: Observable) -> Observable:
    """Compute the tensor product of two observables.

    The tensor product of Hermitian matrices is Hermitian, so the result is an ``Observable``.

    :param O1: Observable for the first system.
    :param O2: Observable for the second system.
    :return: Observable for the tensor product system.
    """
    op = tensor_operator(O1, O2)
    return Observable(data=op.data, num_qubits=op.num_qubits)


@jax.jit
def tensor_involution(I1: Involution, I2: Involution) -> Involution:
    """Compute the tensor product of two Involution operators.

    The tensor product of Involutions is again an Involution
    (e.g. X ⊗ Z is both Hermitian and Unitary).

    :param I1: Involution for the first system.
    :param I2: Involution for the second system.
    :return: Involution for the tensor product system.
    """
    op = tensor_operator(I1, I2)
    return Involution(data=op.data, num_qubits=op.num_qubits)


def tensor_instrument(
    i1: QuantumInstrument,
    i2: QuantumInstrument,
) -> QuantumInstrument:
    """Tensor product of two quantum instruments on independent subsystems.

    The result has ``n1 * n2`` outcomes encoding the joint pair ``(i, j)``.
    Outcome ordering: ``flat_index = i * n2 + j``.

    :param i1: Instrument for the first subsystem.
    :param i2: Instrument for the second subsystem.
    :return: Tensor-product instrument.
    """
    from ._quantum_objects import QuantumInstrument

    n1 = i1.num_outcomes
    n2 = i2.num_outcomes

    mat1 = i1.matrix  # (*ens1, n1, d1², d1²)
    mat2 = i2.matrix  # (*ens2, n2, d2², d2²)

    # For each pair (i, j), compute the SuperOp tensor product using tensor_superop
    superop_list = []
    for i in range(n1):
        s1 = SuperOp.from_matrix(mat1[..., i, :, :], i1.dims)
        for j in range(n2):
            s2 = SuperOp.from_matrix(mat2[..., j, :, :], i2.dims)
            s_tensor = tensor_superop(s1, s2)
            superop_list.append(s_tensor.matrix)

    result_mat = jnp.stack(superop_list, axis=-3)

    new_dims = (i1.dims[0] + i2.dims[0], i1.dims[1] + i2.dims[1])
    new_measured = i1.measured_qudits + tuple(m + i1.num_qubits for m in i2.measured_qudits)

    return QuantumInstrument.from_matrix(result_mat, new_dims, new_measured)
