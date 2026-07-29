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

from ._apply import apply_superop_to_density_matrix
from ._promotion import promote_hilbert_space
from ._quantum_objects import (
    Choi,
    DensityMatrix,
    QuantumInstrument,
    State,
    StateVector,
    SuperOperator,
    Unitary,
    _extract_measured_index,
)
from ._superoperator_transformations import to_choi, to_pauli_liouville, to_superop


@jax.jit
def fidelity(rho: State, sigma: State) -> Array:
    r"""
    Compute the Jozsa fidelity between two quantum states rho and sigma using JAX.

    The fidelity is defined as:

    .. math::

        F(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2

    For pure states \|ψ⟩ and \|φ⟩, this reduces to:

    .. math::

        F(|\psi\rangle, |\phi\rangle) = |\langle\psi|\phi\rangle|^2

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
    superoperator_0: SuperOperator | Unitary,
    superoperator_1: SuperOperator | Unitary | None = None,
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

    :param superoperator_0: Any superoperator type (SuperOperator, Unitary).
    :param superoperator_1: Optional second operator. If None, identity channel is assumed.
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
        choi_1 = Choi.from_matrix(id_choi_data, choi_0.dims)
    else:
        choi_1 = to_choi(superoperator_1)
        if choi_1.dims != choi_0.dims:
            choi_0, choi_1 = promote_hilbert_space(choi_0, choi_1)
            d2 = choi_0.d2[0]

    # The definition of fidelity assumes trace 1 states. Choi matrices have trace d.
    # So we should normalize them before passing to fidelity.

    # We treat J/d as a density matrix. The Choi matrix is (d_out^2 x d_in^2) and we
    # treat it as a single-system density matrix with dimension d^2.
    choi_dm_dims = (choi_0.d2[0],)  # e.g., (16,) for 2-qubit
    rho = DensityMatrix.from_matrix(choi_0.matrix, choi_dm_dims)
    sigma = DensityMatrix.from_matrix(choi_1.matrix, choi_dm_dims)

    # Compute state fidelity between normalized Choi matrices
    state_fid = fidelity(rho, sigma)

    return state_fid / d2


# Convert between process fidelity, average fidelity and depolarizing constant
# https://arxiv.org/abs/1610.05296 table 1


@jax.jit
def depolarizing_constant_to_average_fidelity(p: ArrayLike, dims: tuple[int, ...] = (2,)) -> Array:
    """
    Convert the depolarizing constant to the average fidelity.

    :param p: Depolarizing constant. Defined so that a 1% depolarizing error corresponds to p=0.99.
    :param dims: Tuple of qudit dimensions, one entry per subsystem (e.g. ``(2,)`` for a single qubit).
    :return: Average fidelity in [0, 1]

    See :cite:`BAGFU`, Table 1.
    """
    d = jnp.prod(jnp.array(dims))
    F = ((d - 1) * p + 1) / d
    return jnp.asarray(F)


@jax.jit
def depolarizing_constant_to_process_fidelity(p: ArrayLike, dims: tuple[int, ...] = (2,)) -> Array:
    """
    Convert the depolarizing constant to the process fidelity.

    :param p: Depolarizing constant. Defined so that a 1% depolarizing error corresponds to p=0.99.
    :param dims: Tuple of qudit dimensions, one entry per subsystem (e.g. ``(2,)`` for a single qubit).
    :return: Process fidelity in [0, 1]

    See :cite:`BAGFU`, Table 1.
    """
    d = jnp.prod(jnp.array(dims))
    chi_00 = ((d**2 - 1) * p + 1) / (d**2)
    return jnp.asarray(chi_00)


@jax.jit
def average_fidelity_to_process_fidelity(F: ArrayLike, dims: tuple[int, ...] = (2,)) -> Array:
    """
    Convert the average gate fidelity to the process fidelity.

    :param F: The average fidelity.
    :param dims: Tuple of qudit dimensions, one entry per subsystem (e.g. ``(2,)`` for a single qubit).
    :return: Process fidelity in [0, 1]

    See :cite:`BAGFU`, Table 1.
    """
    d = jnp.prod(jnp.array(dims))
    chi_00 = (F * (d + 1) - 1) / d
    return jnp.asarray(chi_00)


@jax.jit
def process_fidelity_to_average_fidelity(chi_00: ArrayLike, dims: tuple[int, ...] = (2,)) -> Array:
    """
    Convert the process fidelity to the average fidelity.

    :param chi_00: The process fidelity.
    :param dims: Tuple of qudit dimensions, one entry per subsystem (e.g. ``(2,)`` for a single qubit).
    :return: Average fidelity in [0, 1]

    See :cite:`BAGFU`, Table 1.
    """
    d = jnp.prod(jnp.array(dims))
    F = (d * chi_00 + 1) / (d + 1)
    return jnp.asarray(F)


@jax.jit
def process_fidelity_to_depolarizing_constant(chi_00: ArrayLike, dims: tuple[int, ...] = (2,)) -> Array:
    """
    Convert the process fidelity to a depolarizing constant.
    Defined so that a 1% depolarizing error corresponds to p=0.99.

    :param chi_00: The process fidelity.
    :param dims: Tuple of qudit dimensions, one entry per subsystem (e.g. ``(2,)`` for a single qubit).
    :return: Depolarizing constant

    See :cite:`BAGFU`, Table 1.
    """
    d = jnp.prod(jnp.array(dims))
    p = (d**2 * chi_00 - 1) / (d**2 - 1)
    return jnp.asarray(p)


@jax.jit
def average_fidelity_to_depolarizing_constant(F: ArrayLike, dims: tuple[int, ...] = (2,)) -> Array:
    """
    Convert the average fidelity to a depolarizing constant.
    Defined so that a 1% depolarizing error corresponds to p=0.99.

    :param F: The average fidelity.
    :param dims: Tuple of qudit dimensions, one entry per subsystem (e.g. ``(2,)`` for a single qubit).
    :return: Depolarizing constant

    See :cite:`BAGFU`, Table 1.
    """
    d = jnp.prod(jnp.array(dims))
    p = (d * F - 1) / (d - 1)
    return jnp.asarray(p)


@jax.jit
def unitarity_to_stochastic_infidelity(u: ArrayLike, dims: tuple[int, ...] = (2,)) -> Array:
    """
    Convert a unitarity to a stochastic infidelity.

    Valid for unital trace-preserving maps.

    :param u: The unitarity of the channel.
    :param dims: Tuple of qudit dimensions, one entry per subsystem (e.g. ``(2,)`` for a single qubit).
    :return: Stochastic infidelity in [0, 1]

    See :cite:`BAGFU`.
    """
    d = jnp.prod(jnp.array(dims))
    return 1 - jnp.sqrt(u * (1 - 1 / d**2) + (1 / d**2))


def unitarity(
    superoperator: SuperOperator,
) -> Array:
    r"""
    Compute the unitarity of a quantum channel.

    The unitarity is defined as:

    .. math::

        u(\mathcal{E}) = \frac{1}{d^2 - 1} \| M_{1:,1:} \|_F^2

    where *M* is the Pauli-Liouville representation of the channel and the
    subscript :math:`1:` indicates that the first row and column (corresponding
    to the identity component) are removed.

    :param superoperator: Any superoperator type.
    :return: Unitarity in [0, 1], scalar or array for ensembles.
    """
    pl = to_pauli_liouville(superoperator)
    mat = pl.matrix  # (*ensemble, d^2, d^2)
    unitary_block = mat[..., 1:, 1:]
    return jnp.real(jnp.sum(jnp.abs(unitary_block) ** 2, axis=(-2, -1)) / unitary_block.shape[-1])


def stochastic_infidelity(
    superoperator: SuperOperator,
) -> Array:
    r"""
    Compute the stochastic infidelity :math:`e_S` of a quantum channel.

    The stochastic infidelity is defined via the superoperator (standard form)
    representation *S*:

    .. math::

        e_S = 1 - \frac{\operatorname{Re}\!\sqrt{\operatorname{Tr}(S S^\dagger)}}{d}


    :param superoperator: Any superoperator type.
    :return: Stochastic infidelity, scalar or array for ensembles.
    """
    sop = to_superop(superoperator)
    mat = sop.matrix  # (*ensemble, d^2, d^2)
    d = sop.d[0]
    # Tr(S @ S†) = sum of |S_ij|^2
    tr_sst = jnp.sum(jnp.abs(mat) ** 2, axis=(-2, -1))
    return 1 - jnp.real(jnp.sqrt(tr_sst)) / d


# ======================================================================
# Quantum instrument fidelities
# ======================================================================


def classification_fidelity(instrument: QuantumInstrument) -> Array:
    """
    Average classification fidelity of a quantum instrument.

    The confusion matrix :math:`C` has shape ``(num_outcomes, d_measured)``, where entry
    :math:`C[i, j]` is the probability of reporting outcome *i* when the measured subsystem
    is prepared in computational basis state :math:`|j\\rangle`.

    The classification fidelity is the diagonal average of this matrix — the mean
    probability of obtaining the *correct* (matching) outcome:

    .. math::

        F_\\text{class} = \\frac{1}{d} \\sum_j C[j, j] = \\frac{1}{d} \\sum_j p(\\text{outcome} = j \\mid \\text{input} = j)

    This measures readout accuracy alone and is insensitive to the post-measurement state.
    Supports ensembles — returns a scalar per ensemble element.

    See :cite:`DICQI`.
    """
    cm = instrument.confusion_matrix
    d = min(cm.shape[-2], cm.shape[-1])
    return jnp.sum(jnp.diagonal(cm, axis1=-2, axis2=-1)[..., :d], axis=-1) / d


def non_demolition_fidelity(instrument: QuantumInstrument) -> Array:
    """
    Quantum non-demolition (QND) fidelity of a quantum instrument.

    For each input basis state :math:`|j\\rangle` and *every* outcome *i*, we apply the
    corresponding instrument branch superoperator :math:`\\mathcal{E}_i` to
    :math:`|j\\rangle\\langle j|` and extract two quantities from the (unnormalized)
    output :math:`\\tilde{\\rho}_{ij} = \\mathcal{E}_i(|j\\rangle\\langle j|)`:

    - :math:`p(i \\mid j) = \\operatorname{Tr}(\\tilde{\\rho}_{ij})` — probability of outcome *i*.
    - :math:`p(\\text{post} = j \\mid i, j) = \\langle j | \\tilde{\\rho}_{ij} | j \\rangle \\,/\\, p(i \\mid j)` — probability that the post-measurement state is still :math:`|j\\rangle`, given outcome *i*.

    The QND fidelity accumulates these joint contributions over **all** outcomes and input
    states:

    .. math::

        F_\\text{QND} = \\frac{1}{d} \\sum_j \\sum_i p(i \\mid j) \\cdot p(\\text{post} = j \\mid i,\\, j)

    Unlike :func:`instrument_fidelity`, wrong outcomes can contribute as long as the
    post-measurement state is preserved.  This makes the QND fidelity sensitive to
    state preservation independent of readout accuracy.
    Supports ensembles — returns a scalar per ensemble element.

    See :cite:`DICQI`.
    """
    d_total = instrument.d[0]
    n_outcomes = instrument.num_outcomes
    dims = instrument.dims[0]

    # TODO: Replace Python loops with vectorised implementation for large systems.
    total = jnp.array(0.0)
    count = 0
    for j_full in range(d_total):
        rho_j_mat = jnp.zeros((d_total, d_total), dtype=jnp.complex128).at[j_full, j_full].set(1.0)
        rho_j = DensityMatrix.from_matrix(rho_j_mat, dims)

        for i in range(n_outcomes):
            superop_i, _ = instrument.outcome_superop(i)
            rho_out = apply_superop_to_density_matrix(superop_i, rho_j)
            prob = jnp.real(jnp.trace(rho_out.matrix, axis1=-2, axis2=-1))
            fid = jnp.where(prob > 1e-12, jnp.real(rho_out.matrix[..., j_full, j_full]) / prob, 0.0)
            total = total + prob * fid
        count += 1

    return total / count


def instrument_fidelity(instrument: QuantumInstrument) -> Array:
    r"""
    Overall instrument fidelity w.r.t. ideal QND measurement.

    For each input basis state, j, we apply the conditional instrument superoperator to the state |j⟩⟨j|.

    We compute the probability, p, of the correct outcome which is just the trace of the un-normalized output state
    and the fidelity, f, of the post-measurement state with the input state |j⟩⟨j|, normalized by p.

    The instrument fidelity is the cumulative sum of the product of p and f for each input state.

    .. math::

        F_\text{inst} = \frac{1}{d} \sum_j \underbrace{p(j_\text{meas} \mid j)}_{\text{correct outcome}} \cdot \underbrace{p(\text{post} = j \mid j_\text{meas},\, j)}_{\text{state preserved}}


    Only "correct" outcomes (outcome *i* matches input basis state *j*
    on the measured subsystem) contribute.

    Supports ensembles — returns a scalar per ensemble element.

    See :cite:`DICQI`.
    """
    d_total = instrument.d[0]
    n_outcomes = instrument.num_outcomes
    dims = instrument.dims[0]

    # TODO: Replace Python loops with vectorised implementation for large systems.
    total = jnp.array(0.0)
    count = 0
    for j_full in range(d_total):
        j_meas = _extract_measured_index(j_full, dims, instrument.measured_qudits)
        rho_j_mat = jnp.zeros((d_total, d_total), dtype=jnp.complex128).at[j_full, j_full].set(1.0)
        rho_j = DensityMatrix.from_matrix(rho_j_mat, dims)

        if j_meas < n_outcomes:
            superop_i, _ = instrument.outcome_superop(j_meas)
            rho_out = apply_superop_to_density_matrix(superop_i, rho_j)
            prob = jnp.real(jnp.trace(rho_out.matrix, axis1=-2, axis2=-1))
            fid = jnp.where(prob > 1e-12, jnp.real(rho_out.matrix[..., j_full, j_full]) / prob, 0.0)
            total = total + prob * fid
        count += 1

    return total / count
