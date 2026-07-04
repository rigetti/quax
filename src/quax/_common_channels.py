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
operator_tools.common_channels module
-----------------------------------------
A module containing common channels.
"""

from functools import partial, reduce
from operator import mul
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit

from ._exponentiation import evolve
from ._generators import unitary_to_hamiltonian
from ._quantum_objects import (
    Choi,
    KrausMap,
    Lindbladian,
    Operator,
    QuantumInstrument,
    SuperOp,
    Unitary,
    _extract_measured_index,
)
from ._tensor import tensor_choi
from .gates import Y, I, GELLMANN4, GELLMANN5, GELLMANN6, GELLMANN7, TP0, TP1, TP2
from .lindbladians import thermal_relaxation


@partial(jit, static_argnums=())
def _thermal_relaxation_choi_1q(t1: float, tphi: float, duration: float) -> Choi:
    """
    Construct the Choi matrix for a single qubit thermal relaxation channel.

    :param t1: The T1 time.
    :param tphi: The pure dephasing time.
    :param duration: The duration.
    :return: The 4x4 Choi matrix for the single-qubit thermal relaxation channel.
    """
    e1 = jnp.exp(-duration / t1)
    # 1/tphi = 1/t2 - 1/2t1
    t2 = 1 / (1 / tphi + 1 / (2 * t1))
    e2 = jnp.exp(-duration / t2)

    return Choi.from_matrix(
        jnp.array(
            [
                [1, 0, 0, e2],
                [0, 0, 0, 0],
                [0, 0, 1 - e1, 0],
                [e2, 0, 0, e1],
            ],
            dtype=jnp.complex64,
        ),
        ((2,), (2,)),
    )


def _thermal_relaxation_choi(t1s: Array, tphis: Array, duration: float) -> Choi:
    """
    Construct a multi-qubit thermal relaxation channel.

    :param t1s: (num_qubits,) array of t1s.
    :param tphis: (num_qubits,) array of tphis.
    :param duration: The duration (same for all qubits).
    :return: The Choi matrix for the multi-qubit thermal relaxation channel.
    """
    num_qubits = len(t1s)
    if num_qubits == 1:
        return _thermal_relaxation_choi_1q(t1s[0], tphis[0], duration)

    # Use Python loop with static dimensions (unrolls at compile time when num_qubits is static)
    result = _thermal_relaxation_choi_1q(t1s[0], tphis[0], duration)

    for i in range(1, num_qubits):
        single_qubit_choi = _thermal_relaxation_choi_1q(t1s[i], tphis[i], duration)
        result = tensor_choi(result, single_qubit_choi)

    return result


@jax.jit(static_argnames=("dims",))
def depolarizing_channel_superoperator(
    depolarizing_prob: Array,
    dims: Tuple[int, ...],
) -> SuperOp:
    """
    Construct the superoperator for a multi-qudit depolarizing channel.

    :param depolarizing_prob: The depolarizing probability (between 0 and 1).
        Can be a scalar or an array for batched construction, e.g. an array
        of shape ``(n,)`` produces an ensemble of *n* superoperators.
    :param dims: Per-subsystem dimensions, e.g. ``(2,)`` for a single qubit
        or ``(2, 2)`` for two qubits.
    :return: The superoperator matrix for the depolarizing channel.

    .. note::
        This closed form ``(1 - p)·I + p·P_mix`` is exactly the evolved depolarizing Lindbladian
        ``exp(t·L)`` with ``L = -Γ(I - P_mix)`` and ``1 - p = e^{-Γt}`` (``P_mix`` is the projector
        onto the maximally mixed state).  It is written analytically rather than via
        :func:`~quax.evolve` so that the fully-depolarizing endpoint ``p = 1`` (the ``t → ∞``
        limit) is exact.
    """
    depolarizing_prob = jnp.asarray(depolarizing_prob)
    d = reduce(mul, dims, 1)
    identity_super = jnp.eye(d * d, dtype=complex)
    vec_identity = jnp.ravel(jnp.eye(d, dtype=complex))
    max_mixed_super = jnp.outer(vec_identity, vec_identity.conj()) / d

    p = depolarizing_prob[..., None, None]
    depolarizing_super_data = (1 - p) * identity_super + p * max_mixed_super
    full_dims = (dims, dims)
    return SuperOp.from_matrix(depolarizing_super_data, full_dims)


@jax.custom_vjp
def fractional_unitary_power(unitary: Unitary, exponent: float) -> Unitary:
    """
    Compute U^exponent for a unitary matrix with gradient support.

    This function computes fractional powers of unitary matrices and supports
    automatic differentiation using a custom VJP rule.

    The forward pass uses eigendecomposition:
    U = V @ diag(λ) @ V^(-1), so U^exponent = V @ diag(λ^exponent) @ V^(-1)

    The backward pass uses the Fréchet derivative of matrix powers, computed
    via a Sylvester equation solver for efficiency.

    :param unitary: The unitary object.
    :param exponent: The exponent (e.g., 1/n for the n-th root).
    :return: The matrix U^exponent as a Unitary object.
    """
    eigvals, eigvecs = jnp.linalg.eig(unitary.matrix)
    fractional_eigvals = jnp.power(eigvals, exponent)
    result_data = eigvecs @ jnp.diag(fractional_eigvals) @ jnp.linalg.inv(eigvecs)
    return Unitary.from_matrix(result_data, unitary.dims)


def _fractional_unitary_power_fwd(unitary: Unitary, exponent: float):
    """Forward pass: compute U^exponent and save values for backward pass."""
    eigvals, eigvecs = jnp.linalg.eig(unitary.matrix)
    fractional_eigvals = jnp.power(eigvals, exponent)
    V_inv = jnp.linalg.inv(eigvecs)
    result_data = eigvecs @ jnp.diag(fractional_eigvals) @ V_inv
    result = Unitary.from_matrix(result_data, unitary.dims)
    # Save for backward pass
    return result, (eigvals, eigvecs, V_inv, exponent, unitary.dims)


def _fractional_unitary_power_bwd(residuals, g: Unitary):
    """Backward pass: compute VJP using Fréchet derivative of matrix power.

    For f(U) = U^α, the Fréchet derivative is:
    Df(U)[V] = ∫₀^α U^(α-t) V U^(t-1) dt

    We compute this using a more efficient formulation with eigenvalues.
    """
    eigvals, eigvecs, V_inv, exponent, dims = residuals

    # g is the gradient w.r.t. output U^α, as a Unitary object
    # We need to compute gradient w.r.t. input U

    # Transform g to eigenspace: G = V^(-1) @ g @ V
    G = V_inv @ g.matrix @ eigvecs

    # Compute gradient in eigenspace using Loewner matrix
    # For diagonal matrices, D[diag(λ)^α] @ diag(δλ) involves (λ_i^α - λ_j^α)/(λ_i - λ_j)
    λ = eigvals
    λα = jnp.power(λ, exponent)

    # Loewner matrix: L[i,j] = (λ_i^α - λ_j^α) / (λ_i - λ_j) if i≠j, else α*λ_i^(α-1)
    λ_diff = λ[:, None] - λ[None, :]
    λα_diff = λα[:, None] - λα[None, :]

    # Avoid division by zero: use limit formula for i=j
    safe_denom = jnp.where(jnp.abs(λ_diff) < 1e-8, 1.0, λ_diff)
    L = jnp.where(
        jnp.abs(λ_diff) < 1e-8,
        exponent * jnp.power(λ[:, None], exponent - 1),  # Limit as λ_i -> λ_j
        λα_diff / safe_denom,  # type: ignore[operator]
    )

    # Gradient in eigenspace: element-wise multiplication
    G_eigen = L * G

    # Transform back to original space
    grad_unitary_data = eigvecs @ G_eigen @ V_inv
    grad_unitary = Unitary.from_matrix(grad_unitary_data, dims)

    return (grad_unitary, None)  # None for exponent (not differentiable)


# Register the custom VJP
fractional_unitary_power.defvjp(_fractional_unitary_power_fwd, _fractional_unitary_power_bwd)


# Wrap with JIT
fractional_unitary_power = partial(jit, static_argnames=("exponent",))(fractional_unitary_power)  # type: ignore[assignment]


def integrated_thermal_superoperator(
    unitary: Unitary,
    t1s: Array,
    tphis: Array,
    duration: float,
) -> SuperOp:
    """
    Construct a thermal channel integrated over a unitary rotation.

    Models a gate ``U`` applied while thermal relaxation acts continuously for the gate
    ``duration``: it solves the GKSL master equation ``dρ/dt = -i[H, ρ] + D_thermal(ρ)`` over
    ``[0, duration]``, where ``H`` is the Hamiltonian generating ``U`` (``exp(-iH) = U``) and
    ``D_thermal`` is the per-qubit thermal-relaxation dissipator (T1 + pure dephasing).

    The result is the *exact* integrated channel (no Trotter step count) and is differentiable
    with respect to ``unitary`` and the times.

    :param unitary: The gate unitary.
    :param t1s: (num_qubits,) array of T1 relaxation times.
    :param tphis: (num_qubits,) array of pure dephasing times.
    :param duration: The gate duration.
    :return: The superoperator integrated over the unitary rotation.
    """
    # H generates the gate: exp(-iH) = U, so the coherent part integrates to U over `duration`.
    hamiltonian = unitary_to_hamiltonian(unitary)

    # Per-qubit thermal generators (unit-time rates 1/T1, 1/Tφ) combined into the full system.
    thermal = reduce(lambda a, b: a | b, [thermal_relaxation(t1s[i], tphis[i]) for i in range(len(t1s))])

    # Scaling the jump operators by sqrt(duration) scales the dissipator by `duration`, so the
    # generator below is  -i[H, ·] + duration · D_thermal  and  evolve(·, 1) applies U while the
    # thermal noise acts over the gate duration.  At duration = 0 the jumps vanish → pure U channel.
    scaled_jumps = thermal.jump_operators * jnp.sqrt(duration)
    generator = Lindbladian(hamiltonian=hamiltonian, jump_operators=scaled_jumps)
    return evolve(generator, 1.0)


def bitphase_flip_operators(p: float) -> KrausMap:
    """Generate the KrausMap for a bit-phase flip channel.

    The bit-phase flip channel applies Y with probability p and I with probability 1-p.

    :param p: Probability of bit-phase flip error (0 <= p <= 1)
    :return: KrausMap with two 2x2 operator terms
    """
    k0 = Operator.from_matrix(jnp.sqrt(1.0 - p) * I.matrix, ((2,), (2,)))
    k1 = Operator.from_matrix(jnp.sqrt(p) * Y.matrix, ((2,), (2,)))
    data = jnp.stack([k0.matrix, k1.matrix], axis=0)
    return KrausMap.from_matrix(data, ((2,), (2,)))


def stochastic_leakage_operators(gamma: float) -> KrausMap:
    """
    Generate the KrausMap for a leakage channel on a single qutrit.

    Models population transfer from both computational states (|0⟩ and
    |1⟩) to the leaked |2⟩ state with probability gamma per gate.
    This matches the standard leakage RB definition where gamma is the
    probability of leaving the computational subspace, independent of
    the input state.

    :param gamma: Leakage probability per gate (0 <= gamma <= 1)
    :return: KrausMap with three 3x3 operator terms
    """
    # Transition operators from Gell-Mann raising/lowering:
    # |2⟩⟨0| = (λ₄ - iλ₅)/2,  |2⟩⟨1| = (λ₆ - iλ₇)/2
    raise_20 = (GELLMANN4.matrix - 1j * GELLMANN5.matrix) / 2
    raise_21 = (GELLMANN6.matrix - 1j * GELLMANN7.matrix) / 2

    k0_data = jnp.sqrt(1.0 - gamma) * (TP0.matrix + TP1.matrix) + TP2.matrix
    k1_data = jnp.sqrt(gamma) * raise_20
    k2_data = jnp.sqrt(gamma) * raise_21
    data = jnp.stack([k0_data, k1_data, k2_data], axis=0)
    return KrausMap.from_matrix(data, ((3,), (3,)))


def instrument_from_confusion_and_transition(
    confusion_matrix: Array,
    transition_matrix: Array,
    dims: Tuple[int, ...],
    measured_qudits: Optional[Tuple[int, ...]] = None,
) -> QuantumInstrument:
    r"""
    Construct a quantum instrument from a confusion matrix and a transition matrix.

    The confusion matrix is defined as the probability of observing each measurement outcome,
    given the true state of the measured qudits or Tij = P(measurement outcome i | true state j).
    Note that the convention is that the confusion matrix is column-stochastic, i.e. columns sum to 1.

    The transition matrix describes the "backaction" of the measurement on the post-measurement state,
    i.e. Tij = P(post-measurement state i | pre-measurement state j). The convention is that the
    transition matrix is column-stochastic, i.e. columns sum to 1.

    The instrument is constructed by creating a Kraus operator of each outcome i, basis state j
    and post-measurement state k. Each Kraus operator is

    .. math::
        K_{i,j,k} = \sqrt{C_{i, j} T_{k, j}} |k\rangle\langle j|

    The overall instrument is then

    .. math::
        \mathcal{M}_i(\rho) = \sum_{j, k} C_{i,j} T_{k, j} \langle j|\rho|k\rangle |k\rangle\langle k|


    :param confusion_matrix: ``(num_outcomes, d_measured)`` column-stochastic
        matrix.
    :param transition_matrix: ``(d_total, d_total)`` column-stochastic matrix.
    :param dims: Per-qudit dimensions.
    :param measured_qudits: Indices of measured qudits.  Defaults to all.
    """
    if measured_qudits is None:
        measured_qudits = tuple(range(len(dims)))

    d_total = reduce(mul, dims, 1)
    d_measured = reduce(mul, (dims[i] for i in measured_qudits), 1)
    num_outcomes = confusion_matrix.shape[0]

    if confusion_matrix.shape != (num_outcomes, d_measured):
        raise ValueError(
            f"Confusion matrix shape {confusion_matrix.shape} does not match "
            f"(num_outcomes={num_outcomes}, d_measured={d_measured})."
        )
    if transition_matrix.shape != (d_total, d_total):
        raise ValueError(
            f"Transition matrix shape {transition_matrix.shape} does not match (d_total={d_total}, d_total={d_total})."
        )
    if not jnp.all(confusion_matrix >= -1e-14):
        raise ValueError("Confusion matrix entries must be non-negative.")
    if not jnp.all(transition_matrix >= -1e-14):
        raise ValueError("Transition matrix entries must be non-negative.")
    if not jnp.allclose(jnp.sum(confusion_matrix, axis=0), 1.0, atol=1e-6):
        raise ValueError("Confusion matrix columns must sum to 1.")
    if not jnp.allclose(jnp.sum(transition_matrix, axis=0), 1.0, atol=1e-6):
        raise ValueError("Transition matrix columns must sum to 1.")

    superop_list: list[Array] = []
    for i in range(num_outcomes):
        kraus_ops: list[Array] = []
        for j_full in range(d_total):
            j_meas = _extract_measured_index(j_full, dims, measured_qudits)
            p_measure = confusion_matrix[i, j_meas]
            for k in range(d_total):
                p_transition = transition_matrix[k, j_full]
                amplitude = jnp.sqrt(jnp.clip(p_measure * p_transition, min=0.0))
                K = jnp.zeros((d_total, d_total), dtype=jnp.complex128)
                K = K.at[k, j_full].set(amplitude)
                kraus_ops.append(K)
        # SuperOp = Σ conj(K_i) ⊗ K_i
        kraus_stack = jnp.stack(kraus_ops, axis=0)  # (n_kraus, d, d)
        superop_mat = jnp.einsum("iab,icd->acbd", jnp.conj(kraus_stack), kraus_stack)
        superop_mat = superop_mat.reshape(d_total * d_total, d_total * d_total)
        superop_list.append(superop_mat)

    matrices = jnp.stack(superop_list, axis=0)
    op_dims = (dims, dims)
    return QuantumInstrument.from_matrix(matrices, op_dims, measured_qudits)


def instrument_from_axis(
    theta: float = 0.0,
    phi: float = 0.0,
    sharpness: float = 1.0,
) -> QuantumInstrument:
    """Create a single-qubit instrument from a Bloch-sphere measurement axis.

    The angles follow standard Bloch sphere notation.  ``theta=0, phi=0``
    is the Z-axis (computational basis measurement).

    :param theta: Colatitude with respect to the z-axis.
    :param phi: Longitude with respect to the x-axis.
    :param sharpness: Measurement sharpness.  1.0 is projective, 0.0 is no
        measurement.
    :return: A single-qubit :class:`QuantumInstrument`.
    """
    eye = jnp.eye(2, dtype=jnp.complex128)
    sig_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    sig_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
    sig_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

    nx = jnp.sin(theta) * jnp.cos(phi)
    ny = jnp.sin(theta) * jnp.sin(phi)
    nz = jnp.cos(theta)

    lambda_plus = jnp.sqrt((1 + sharpness) / 2)
    lambda_minus = jnp.sqrt((1 - sharpness) / 2)
    c = (lambda_plus + lambda_minus) / 2
    d = (lambda_plus - lambda_minus) / 2

    n_dot_sigma = nx * sig_x + ny * sig_y + nz * sig_z
    K_plus = c * eye + d * n_dot_sigma
    K_minus = c * eye - d * n_dot_sigma

    # SuperOp = conj(K) ⊗ K for each single-Kraus outcome
    superop_plus = jnp.einsum("ab,cd->acbd", jnp.conj(K_plus), K_plus).reshape(4, 4)
    superop_minus = jnp.einsum("ab,cd->acbd", jnp.conj(K_minus), K_minus).reshape(4, 4)

    matrices = jnp.stack([superop_plus, superop_minus], axis=0)
    return QuantumInstrument.from_matrix(matrices, ((2,), (2,)), (0,))
