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
JAX-based implementations of quantum distance metrics.

This module provides JIT-compiled implementations of quantum fidelity measures
for use in differentiable quantum algorithms and high-performance computing.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ._quantum_objects import Choi, DensityMatrix, KrausMap, PauliLiouville, State, StateVector, SuperOp, Unitary
from ._superoperator_transformations import to_choi


@jax.jit
def fidelity(rho: State, sigma: State) -> Array:
    r"""
    Compute the Jozsa fidelity between two quantum states rho and sigma using JAX.

    The fidelity is defined as:

    .. math::

        F(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2

    For pure states |ψ⟩ and |φ⟩, this reduces to:

    .. math::

        F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|^2

    :param rho: A State object (StateVector or DensityMatrix).
    :param sigma: A State object (StateVector or DensityMatrix).
    :return: Fidelity value in [0, 1]
    """
    # --- Convert to density matrices (batched) ---
    rho_data = rho.matrix
    if isinstance(rho, StateVector):
        # (..., d) -> (..., d, d)
        rho_data = jnp.einsum("...i,...j->...ij", rho_data, jnp.conj(rho_data))

    sigma_data = sigma.matrix
    if isinstance(sigma, StateVector):
        sigma_data = jnp.einsum("...i,...j->...ij", sigma_data, jnp.conj(sigma_data))

    w, v = jnp.linalg.eigh(rho_data)  # w: (..., d), v: (..., d, d)
    w = jnp.maximum(w, 0.0)
    sqrt_w = jnp.sqrt(w)

    # sqrt_rho = v @ diag(sqrt_w) @ v†  (batched, no explicit diag)
    # v_scaled[..., :, k] = v[..., :, k] * sqrt_w[..., k]
    v_scaled = v * sqrt_w[..., None, :]
    sqrt_rho = v_scaled @ jnp.swapaxes(jnp.conj(v), -1, -2)

    M = sqrt_rho @ sigma_data @ sqrt_rho

    # --- Fidelity = (Tr sqrt(M))^2, using batched eigvalsh ---
    m = jnp.linalg.eigvalsh(M)  # (..., d)
    m = jnp.maximum(m, 0.0)
    tr_sqrt = jnp.sum(jnp.sqrt(m), axis=-1)  # (...,)

    return jnp.real(tr_sqrt**2)


@jax.jit
def unitary_entanglement_fidelity(unitary_e: Unitary, unitary_f: Unitary) -> Array:
    r"""
    Return the entanglement fidelity between two unitary operators using JAX.

    The entanglement fidelity is:

    .. math::

        F_e(E,F) = \left|\frac{\text{Tr}[E^\dagger F]}{d}\right|^2

    where d is the dimension of the Hilbert space.

    :param unitary_e: A Unitary object.
    :param unitary_f: A Unitary object.
    :return: Entanglement fidelity in [0, 1]
    """
    d = unitary_f.d[0]
    # Compute Tr[E^† F] = Tr[E^H F] using einsum
    # For matrices: einsum('...ij,...jk->...ik', E^H, F) then trace with '...ii'
    trace = jnp.einsum(
        "...ii",
        jnp.einsum("...ij,...jk->...ik", jnp.moveaxis(unitary_e.matrix.conj(), -1, -2), unitary_f.matrix),
    )
    return jnp.abs(trace / d) ** 2


def process_fidelity(
    superoperator_0: Choi | SuperOp | PauliLiouville | KrausMap | Unitary,
    superoperator_1: Choi | SuperOp | PauliLiouville | KrausMap | Unitary | None = None,
) -> Array:
    r"""
    Return the process fidelity between two superoperators.

    The process fidelity is defined as:

    .. math::

        F_{\text{process}} = \left(\frac{F_{\text{state}}(J_0, J_1)}{d}\right)^2

    where d is the dimension of the Hilbert space and F_state is the Jozsa fidelity
    between the Choi matrices treated as quantum states.

    This follows the definition from:
    A. Gilchrist, N.K. Langford, M.A. Nielsen, Phys. Rev. A 71, 062310 (2005).

    It is the square of the one implemented in Nielsen & Chuang,
    "Quantum Computation and Quantum Information"

    :param choi_0: Any superoperator type (Choi, SuperOp, PauliLiouville, KrausMap, Unitary).
    :param choi_1: Optional second operator. If None, identity channel is assumed.
    :return: Process fidelity in [0, 1]
    """

    # Convert inputs to Choi representation
    choi_0 = to_choi(superoperator_0)

    d2 = choi_0.d2[0]
    dims_out = choi_0.dims[0]
    dims_in = choi_0.dims[1]

    if dims_out != dims_in:
        raise NotImplementedError("Process fidelity only implemented for dimension-preserving operators.")

    if superoperator_1 is None:
        omega = jnp.eye(choi_0.d[0], dtype=choi_0.matrix.dtype).reshape(-1)
        id_choi_data = jnp.outer(omega, jnp.conj(omega))  # Tr = d
        choi_1 = Choi.from_matrix(id_choi_data, choi_0.dims, 0)
    else:
        choi_1 = to_choi(superoperator_1)
        if choi_1.dims != choi_0.dims:
            raise ValueError("Choi matrices must have the same dimensions for process fidelity.")

    # The definition of fidelity assumes trace 1 states. Choi matrices have trace d.
    # So we should normalize them before passing to fidelity.

    # We treat J/d as a density matrix. The Choi matrix is (d_out^2 x d_in^2) and we
    # treat it as a single-system density matrix with dimension d^2.
    choi_dm_dims = (choi_0.d2[0],)  # e.g., (16,) for 2-qubit
    rho = DensityMatrix.from_matrix(choi_0.matrix, choi_dm_dims, choi_0.num_ensemble_dims)
    sigma = DensityMatrix.from_matrix(choi_1.matrix, choi_dm_dims, choi_1.num_ensemble_dims)

    # Compute state fidelity between normalized Choi matrices
    state_fid = fidelity(rho, sigma)

    return state_fid / d2


# Convert between process fidelity, average fidelity and depolarizing constant
# https://arxiv.org/abs/1610.05296 table 1


@jax.jit
def depolarizing_constant_to_average_fidelity(p: ArrayLike, num_sys: int = 1) -> Array:
    """
    Convert the depolarizing constant to the average fidelity.

    :param p: Depolarizing constant. Defined so that a 1% depolarizing error corresponds to p=0.99.
    :param num_sys: The number of qubits.
    :return: Average fidelity in [0, 1]
    """
    d = 2**num_sys
    F = ((d - 1) * p + 1) / d
    return F


@jax.jit
def depolarizing_constant_to_process_fidelity(p: ArrayLike, num_sys: int) -> Array:
    """
    Convert the depolarizing constant to the process fidelity.

    :param p: Depolarizing constant. Defined so that a 1% depolarizing error corresponds to p=0.99.
    :param num_sys: The number of qubits.
    :return: Process fidelity in [0, 1]
    """
    d = 2**num_sys
    chi_00 = ((d**2 - 1) * p + 1) / (d**2)
    return chi_00


@jax.jit
def average_fidelity_to_process_fidelity(F: ArrayLike, num_sys: int) -> Array:
    """
    Convert the average gate fidelity to the process fidelity.

    :param F: The average fidelity.
    :param num_sys: The number of qubits.
    :return: Process fidelity in [0, 1]
    """
    d = 2**num_sys
    chi_00 = (F * (d + 1) - 1) / d
    return chi_00


@jax.jit
def process_fidelity_to_average_fidelity(chi_00: ArrayLike, num_sys: int) -> Array:
    """
    Convert the process fidelity to the average fidelity.

    :param chi_00: The process fidelity.
    :param num_sys: The number of qubits.
    :return: Average fidelity in [0, 1]
    """
    d = 2**num_sys
    F = (d * chi_00 + 1) / (d + 1)
    return F


@jax.jit
def process_fidelity_to_depolarizing_constant(chi_00: ArrayLike, num_sys: int) -> Array:
    """
    Convert the process fidelity to a depolarizing constant.
    Defined so that a 1% depolarizing error corresponds to p=0.99.

    :param chi_00: The process fidelity.
    :param num_sys: The number of qubits.
    :return: Depolarizing constant
    """
    d = 2**num_sys
    p = (d**2 * chi_00 - 1) / (d**2 - 1)
    return p


@jax.jit
def average_fidelity_to_depolarizing_constant(F: ArrayLike, num_sys: int) -> Array:
    """
    Convert the average fidelity to a depolarizing constant.
    Defined so that a 1% depolarizing error corresponds to p=0.99.

    :param F: The average fidelity.
    :param num_sys: The number of qubits.
    :return: Depolarizing constant
    """
    d = 2**num_sys
    p = (d * F - 1) / (d - 1)
    return p


@jax.jit
def unitarity_to_stochastic_infidelity(unitarity: ArrayLike, num_sys: int) -> Array:
    """
    Convert a unitarity to a stochastic infidelity.

    Valid for unital trace-preserving maps.

    :param unitarity: The unitarity of the channel.
    :param num_sys: The number of qubits.
    :return: Stochastic infidelity in [0, 1]
    """
    d = 2**num_sys
    return 1 - jnp.sqrt(unitarity * (1 - 1 / d**2) + (1 / d**2))
