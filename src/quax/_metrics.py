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
import numpy as np
from jax import Array
from jax.typing import ArrayLike

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
    # Tr[E^† F] = sum_ij conj(E_ij) F_ij: an elementwise product and a sum, not a matrix product
    # whose diagonal alone is used.
    trace = jnp.sum(jnp.conj(unitary_e.matrix) * unitary_f.matrix, axis=(-2, -1))
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


@jax.jit
def non_demolition_fidelity(instrument: QuantumInstrument) -> Array:
    r"""
    Quantum non-demolition (QND) fidelity of a quantum instrument.

    For each input basis state :math:`|j\rangle` and *every* outcome *i*, consider the
    (unnormalized) output :math:`\tilde{\rho}_{ij} = \mathcal{E}_i(|j\rangle\langle j|)` of the
    instrument branch :math:`\mathcal{E}_i`, and two quantities derived from it:

    - :math:`p(i \mid j) = \operatorname{Tr}(\tilde{\rho}_{ij})` — probability of outcome *i*.
    - :math:`p(\text{post} = j \mid i, j) = \langle j | \tilde{\rho}_{ij} | j \rangle \,/\, p(i \mid j)` — probability that the post-measurement state is still :math:`|j\rangle`, given outcome *i*.

    The QND fidelity accumulates these joint contributions over **all** outcomes and input
    states:

    .. math::

        F_\text{QND} = \frac{1}{d} \sum_j \sum_i p(i \mid j) \cdot p(\text{post} = j \mid i,\, j)

    Unlike :func:`instrument_fidelity`, wrong outcomes can contribute as long as the
    post-measurement state is preserved.  This makes the QND fidelity sensitive to
    state preservation independent of readout accuracy.
    Supports ensembles — returns a scalar per ensemble element.

    **Computation.**  With :math:`T_i[k, j] = \langle k | \tilde{\rho}_{ij} | k \rangle`, a gather
    from the superoperator diagonal (see ``QuantumInstrument._basis_state_transitions``), both
    quantities are entries of :math:`T`:

    .. math::

        p(i \mid j) = \sum_k T_i[k, j],
        \qquad
        p(i \mid j) \cdot p(\text{post} = j \mid i, j) = T_i[j, j].

    The conditional probability is undefined where :math:`p(i \mid j) = 0`; such terms are dropped
    below the threshold :math:`\varepsilon = 10^{-12}`, giving

    .. math::

        F_\text{QND} = \frac{1}{d} \sum_j \sum_i \bigl[\, p(i \mid j) > \varepsilon \,\bigr]\, T_i[j, j].

    Since :math:`\tilde{\rho}_{ij} \succeq 0`, every :math:`T_i[k, j] \ge 0`, so each dropped term
    satisfies :math:`T_i[j, j] \le p(i \mid j) \le \varepsilon`, and the threshold moves
    :math:`F_\text{QND}` by at most :math:`n \varepsilon` for :math:`n` outcomes.  The product is
    evaluated as the single entry :math:`T_i[j, j]`, with no division by :math:`p(i \mid j)`, so
    the gradient stays finite where an outcome is impossible.

    See :cite:`DICQI`.
    """
    # transitions[..., i, k, j] = <k|E_i(|j><j|)|k>
    transitions = instrument._basis_state_transitions
    prob = transitions.sum(axis=-2)  # p(i | j), (*ensemble, n_outcomes, d)
    stay = jnp.diagonal(transitions, axis1=-2, axis2=-1)  # <j|E_i(|j><j|)|j>, (*ensemble, n_outcomes, d)
    # p(i | j) * p(post = j | i, j) = stay, counted where outcome i is possible for input j
    return jnp.sum(jnp.where(prob > 1e-12, stay, 0.0), axis=(-2, -1)) / instrument.d[0]


@jax.jit
def instrument_fidelity(instrument: QuantumInstrument) -> Array:
    r"""
    Overall instrument fidelity w.r.t. ideal QND measurement.

    For each input basis state :math:`|j\rangle`, let :math:`j_\text{meas}` be its index on the
    measured subsystem, which names the correct outcome.  Apply that outcome's branch to
    :math:`|j\rangle\langle j|`: the probability :math:`p` of the correct outcome is the trace of the
    unnormalized output, and the fidelity :math:`f` of the post-measurement state with
    :math:`|j\rangle\langle j|` is normalized by :math:`p`.  The instrument fidelity is the average of
    :math:`p f` over input states:

    .. math::

        F_\text{inst} = \frac{1}{d} \sum_j \underbrace{p(j_\text{meas} \mid j)}_{\text{correct outcome}} \cdot \underbrace{p(\text{post} = j \mid j_\text{meas},\, j)}_{\text{state preserved}}

    Only "correct" outcomes (outcome *i* matches input basis state *j*
    on the measured subsystem) contribute.

    Supports ensembles — returns a scalar per ensemble element.

    **Computation.**  As in :func:`non_demolition_fidelity`, with
    :math:`T_i[k, j] = \langle k | \mathcal{E}_i(|j\rangle\langle j|) | k \rangle`,

    .. math::

        p(j_\text{meas} \mid j) = \sum_k T_{j_\text{meas}}[k, j],
        \qquad
        p(j_\text{meas} \mid j) \cdot p(\text{post} = j \mid j_\text{meas}, j) = T_{j_\text{meas}}[j, j].

    Inputs whose :math:`j_\text{meas}` is not an outcome of the instrument contribute nothing but
    still count in :math:`d`.  With :math:`J` the remaining inputs and the same threshold
    :math:`\varepsilon = 10^{-12}`,

    .. math::

        F_\text{inst} = \frac{1}{d} \sum_{j \in J} \bigl[\, p(j_\text{meas} \mid j) > \varepsilon \,\bigr]\, T_{j_\text{meas}}[j, j].

    See :cite:`DICQI`.
    """
    d_total = instrument.d[0]
    dims = instrument.dims[0]

    # The correct outcome for each input basis state j; inputs without one contribute nothing.
    j_meas = np.array([_extract_measured_index(j, dims, instrument.measured_qudits) for j in range(d_total)])
    j_full = np.flatnonzero(j_meas < instrument.num_outcomes)
    outcome = j_meas[j_full]

    # transitions[..., i, k, j] = <k|E_i(|j><j|)|k>
    transitions = instrument._basis_state_transitions
    prob = transitions.sum(axis=-2)[..., outcome, j_full]  # p(j_meas | j)
    stay = transitions[..., outcome, j_full, j_full]  # <j|E_{j_meas}(|j><j|)|j>
    # p(j_meas | j) * p(post = j | j_meas, j) = stay, counted where the outcome is possible
    return jnp.sum(jnp.where(prob > 1e-12, stay, 0.0), axis=-1) / d_total
