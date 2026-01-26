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

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, jit

from ._compose import compose_superop
from ._quantum_objects import Choi, SuperOp, Unitary, Kraus
from ._superoperator_transformations import (
    choi_to_superop,
    unitary_to_superop,
)
from ._tensor import tensor_choi
from .gates import X, Y, Z, I


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
        0,
    )


def thermal_relaxation_choi(t1s: Array, tphis: Array, duration: float) -> Choi:
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


def depolarizing_channel_superoperator(depolarizing_prob: float, num_qubits: int) -> SuperOp:
    """
    Construct the superoperator for a multi-qubit depolarizing channel.

    :param depolarizing_prob: The depolarizing probability (between 0 and 1).
    :param num_qubits: The number of qubits.
    :return: The superoperator matrix for the depolarizing channel.
    """
    d = 2**num_qubits
    identity_super = jnp.eye(d * d, dtype=complex)
    # The channel is defined as rho -> (1-p) rho + p * Tr(rho) * I / d
    # The superoperator for rho -> Tr(rho) * I / d is |I>> <<I| / d
    vec_identity = jnp.ravel(jnp.eye(d, dtype=complex))
    max_mixed_super = jnp.outer(vec_identity, vec_identity.conj()) / d

    depolarizing_super_data = (1 - depolarizing_prob) * identity_super + depolarizing_prob * max_mixed_super
    dims = (tuple([2] * num_qubits), tuple([2] * num_qubits))
    return SuperOp.from_matrix(depolarizing_super_data, dims, 0)


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
    return Unitary.from_matrix(result_data, unitary.dims, unitary.num_ensemble_dims)


def _fractional_unitary_power_fwd(unitary: Unitary, exponent: float):
    """Forward pass: compute U^exponent and save values for backward pass."""
    eigvals, eigvecs = jnp.linalg.eig(unitary.matrix)
    fractional_eigvals = jnp.power(eigvals, exponent)
    V_inv = jnp.linalg.inv(eigvecs)
    result_data = eigvecs @ jnp.diag(fractional_eigvals) @ V_inv
    result = Unitary.from_matrix(result_data, unitary.dims, unitary.num_ensemble_dims)
    # Save for backward pass
    return result, (eigvals, eigvecs, V_inv, exponent, unitary.dims, unitary.num_ensemble_dims)


def _fractional_unitary_power_bwd(residuals, g: Unitary):
    """Backward pass: compute VJP using Fréchet derivative of matrix power.

    For f(U) = U^α, the Fréchet derivative is:
    Df(U)[V] = ∫₀^α U^(α-t) V U^(t-1) dt

    We compute this using a more efficient formulation with eigenvalues.
    """
    eigvals, eigvecs, V_inv, exponent, dims, num_ensemble_dims = residuals

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
    grad_unitary = Unitary.from_matrix(grad_unitary_data, dims, num_ensemble_dims)

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
    num_steps: int = 100,
) -> SuperOp:
    """
    Construct a thermal channel which is integrated over a unitary rotation.

    For example, we may know accurate decoherence times for a qubit, and wish to calculate
    the channel over a unitary rotation by integrating the thermal relaxation during the gate.

    :param unitary: The unitary rotation object.
    :param t1s: (num_qubits,) array of T1 relaxation times.
    :param tphis: (num_qubits,) array of pure dephasing times.
    :param duration: The gate duration.
    :param num_steps: The number of steps to integrate over (static argument for JIT).
    :return: The superoperator integrated over the unitary rotation.
    """

    # 1. Compute fractional unitary: U^(1/num_steps)
    dU = fractional_unitary_power(unitary, 1.0 / num_steps)

    # Convert fractional unitary to superoperator
    dU_super = unitary_to_superop(dU)

    # 2. Compute thermal relaxation for fractional duration
    fractional_duration = duration / num_steps
    thermal_choi = thermal_relaxation_choi(t1s, tphis, fractional_duration)
    thermal_super = choi_to_superop(thermal_choi)

    # 3. Compute one step: thermal @ unitary
    superstep = compose_superop(thermal_super, dU_super)

    # 4. Compute the integrated channel by repeated matrix multiplication
    # Result = (S_thermal @ S_U)^num_steps
    # Use lax.scan for efficient repeated composition
    def scan_fn(acc, _):
        """Scan function: repeatedly apply the superstep."""
        return compose_superop(superstep, acc), None

    # Apply superstep (num_steps) times
    result, _ = jax.lax.scan(scan_fn, superstep, None, length=num_steps - 1)

    return result


def bit_flip_operators(p: float) -> tuple[Kraus, Kraus]:
    """Generate Kraus operators for a bit flip channel.

    The bit flip channel applies X with probability p and I with probability 1-p.

    :param p: Probability of bit flip error (0 <= p <= 1)
    :return: Tuple of two 2x2 Kraus operators (K0, K1)
    """
    return (
        Kraus.from_matrix(jnp.sqrt(1.0 - p) * I.matrix, ((2,), (2,)), 0),
        Kraus.from_matrix(jnp.sqrt(p) * X.matrix, ((2,), (2,)), 0),
    )


def phase_flip_operators(p: float) -> tuple[Kraus, Kraus]:
    """Generate Kraus operators for a phase flip channel.

    The phase flip channel applies Z with probability p and I with probability 1-p.

    :param p: Probability of phase flip error (0 <= p <= 1)
    :return: Tuple of two 2x2 Kraus operators (K0, K1)
    """
    return (
        Kraus.from_matrix(jnp.sqrt(1.0 - p) * I.matrix, ((2,), (2,)), 0),
        Kraus.from_matrix(jnp.sqrt(p) * Z.matrix, ((2,), (2,)), 0),
    )


def bitphase_flip_operators(p: float) -> tuple[Kraus, Kraus]:
    """Generate Kraus operators for a bit-phase flip channel.

    The bit-phase flip channel applies Y with probability p and I with probability 1-p.

    :param p: Probability of bit-phase flip error (0 <= p <= 1)
    :return: Tuple of two 2x2 Kraus operators (K0, K1)
    """
    return (
        Kraus.from_matrix(jnp.sqrt(1.0 - p) * I.matrix, ((2,), (2,)), 0),
        Kraus.from_matrix(jnp.sqrt(p) * Y.matrix, ((2,), (2,)), 0),
    )


def dephasing_operators(p: float) -> tuple[Kraus, Kraus]:
    """Generate Kraus operators for a dephasing (phase damping) channel.

    The dephasing channel causes loss of quantum information without energy dissipation.

    :param p: Dephasing probability (0 <= p <= 1)
    :return: Tuple of two 2x2 Kraus operators (K0, K1)
    """
    sqrt_p2 = jnp.sqrt(p / 2.0)
    sqrt_1mp2 = jnp.sqrt(1.0 - p / 2.0)
    return (
        Kraus.from_matrix(sqrt_1mp2 * I.matrix, ((2,), (2,)), 0),
        Kraus.from_matrix(sqrt_p2 * Z.matrix, ((2,), (2,)), 0),
    )


def depolarizing_operators(p: float) -> tuple[Kraus, Kraus, Kraus, Kraus]:
    """Generate Kraus operators for a depolarizing channel.

    The depolarizing channel applies I, X, Y, or Z each with equal probability p/4,
    and I with probability 1-p.

    :param p: Depolarizing probability (0 <= p <= 1)
    :return: Tuple of four 2x2 Kraus operators (K0, K1, K2, K3)
    """
    return (
        Kraus.from_matrix(jnp.sqrt(1.0 - p) * I.matrix, ((2,), (2,)), 0),
        Kraus.from_matrix(jnp.sqrt(p / 3.0) * X.matrix, ((2,), (2,)), 0),
        Kraus.from_matrix(jnp.sqrt(p / 3.0) * Y.matrix, ((2,), (2,)), 0),
        Kraus.from_matrix(jnp.sqrt(p / 3.0) * Z.matrix, ((2,), (2,)), 0),
    )


def relaxation_operators(p: float) -> tuple[Kraus, Kraus]:
    """Generate Kraus operators for an amplitude damping (relaxation) channel.

    The amplitude damping channel models energy dissipation (T1 decay).

    :param p: Relaxation probability (0 <= p <= 1)
    :return: Tuple of two 2x2 Kraus operators (K0, K1)
    """
    return (
        Kraus.from_matrix(jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1.0 - p)]], dtype=complex), ((2,), (2,)), 0),
        Kraus.from_matrix(jnp.array([[0.0, jnp.sqrt(p)], [0.0, 0.0]], dtype=complex), ((2,), (2,)), 0),
    )


# Dictionary mapping channel names to their Kraus operator functions
KRAUS_OPS = {
    "bit_flip": bit_flip_operators,
    "phase_flip": phase_flip_operators,
    "bitphase_flip": bitphase_flip_operators,
    "dephasing": dephasing_operators,
    "depolarizing": depolarizing_operators,
    "relaxation": relaxation_operators,
}
