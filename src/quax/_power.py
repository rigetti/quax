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

"""Integer and fractional powers of quantum objects.

This module provides power functions for quantum states, unitaries, and superoperators.
All implementations use eigendecomposition matching scipy.linalg.fractional_matrix_power:

    M^p = V @ diag(λ^p) @ V^(-1)

where V are eigenvectors and λ are eigenvalues.

**For quantum states (density matrices, state vectors) and unitaries:**
- Fractional matrix powers are well-defined and preserve physical properties
- Density matrices remain positive semidefinite and Hermitian
- Unitaries remain unitary (eigenvalues on unit circle)

**For quantum channels (superoperators):**
- Integer powers correspond to channel composition (always preserves CPTP)
- Fractional powers may NOT preserve complete positivity or trace preservation
- Result may not represent a valid quantum channel for non-integer powers

**Note on fractional channel powers:**
The eigenvalue-based approach implemented here matches scipy but is mathematically
naive for quantum channels. For physically meaningful fractional channel interpolation,
consider:
- Using Lindbladian generators: Φ^α = exp(α·log(Φ)) via matrix logarithm
- Restricting to infinitely divisible channel families (depolarizing, damping, etc.)
- Projecting results to CPTP space using operator_tools.project_superoperators
- Verifying CPTP properties with operator_tools.validate_superoperator

The eigenvalue approach is provided for compatibility with scipy and numerical testing.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Choi, DensityMatrix, KrausMap, PauliLiouville, StateVector, SuperOp, Unitary
from ._superoperator_transformations import choi_to_superop, kraus_to_superop, superop_to_choi, superop_to_kraus


def _matrix_power_via_eig(matrix: Array, power: float) -> Array:
    """
    Compute matrix power using eigendecomposition.

    For matrix M, computes M^power = V @ diag(λ^power) @ V^(-1)
    where V are eigenvectors and λ are eigenvalues.

    Used for states, unitaries, and integer powers of superoperators.

    :param matrix: Matrix or ensemble of matrices with shape (..., n, n)
    :param power: The power to raise the matrix to
    :return: Matrix power with same shape as input
    """
    # Eigendecomposition
    eigvals, eigvecs = jnp.linalg.eig(matrix)

    # Raise eigenvalues to the power
    powered_eigvals = jnp.power(eigvals, power)

    # Reconstruct: M^p = V @ diag(λ^p) @ V^(-1)
    # eigvecs: (..., n, n), powered_eigvals: (..., n)
    V_scaled = eigvecs * powered_eigvals[..., None, :]
    V_inv = jnp.linalg.inv(eigvecs)
    result = V_scaled @ V_inv

    return result


def _matrix_power_via_lindbladian(matrix: Array, power: float) -> Array:
    """
    Compute matrix power using Lindbladian approach: M^α = exp(α·log(M)).

    This is the mathematically correct approach for fractional powers of quantum channels,
    as it corresponds to interpolating along a continuous-time semigroup.

    For a channel Φ that can be written as Φ = exp(t·ℒ) for some generator ℒ,
    the fractional power is Φ^α = exp(α·t·ℒ) = exp(α·log(Φ)).

    :param matrix: Matrix or ensemble of matrices with shape (..., n, n)
    :param power: The power to raise the matrix to
    :return: Matrix power with same shape as input
    """
    # Matrix logarithm via eigendecomposition: log(M) = V @ diag(log(λ)) @ V^(-1)
    eigvals, eigvecs = jnp.linalg.eig(matrix)
    log_eigvals = jnp.log(eigvals)

    # Construct log(M) = V @ diag(log(λ)) @ V^(-1)
    V_scaled = eigvecs * log_eigvals[..., None, :]
    V_inv = jnp.linalg.inv(eigvecs)
    log_matrix = V_scaled @ V_inv

    # Scale by power: α·log(M)
    scaled_log = power * log_matrix

    # Matrix exponential
    result = jax.scipy.linalg.expm(scaled_log)

    return result


@jax.jit
def power_choi(choi: Choi, power: float) -> Choi:
    """
    Compute the power of a Choi matrix.

    Uses eigendecomposition (M^p = V @ diag(λ^p) @ V^(-1)) matching scipy.linalg.fractional_matrix_power.

    Note: For fractional powers of quantum channels, this may not preserve CPTP properties.
    For physically meaningful fractional channel interpolation, use the Lindbladian generator
    approach or restrict to infinitely divisible channel families.

    :param choi: The Choi matrix to exponentiate.
    :param power: The power to raise the Choi matrix to.
    :return: The Choi matrix raised to the specified power.
    """
    powered_superop = power_superop(choi_to_superop(choi), power)
    return superop_to_choi(powered_superop)


@jax.jit
def power_superop(superop: SuperOp, power: float) -> SuperOp:
    """
    Compute the power of a superoperator matrix.

    Uses eigendecomposition (M^p = V @ diag(λ^p) @ V^(-1)) matching scipy.linalg.fractional_matrix_power.

    Note: For fractional powers of quantum channels, this may not preserve CPTP properties.
    For physically meaningful fractional channel interpolation, use the Lindbladian generator
    approach or restrict to infinitely divisible channel families.

    :param superop: The superoperator to exponentiate.
    :param power: The power to raise the superoperator to.
    :return: The superoperator raised to the specified power.
    """
    powered_data = _matrix_power_via_eig(superop.matrix, power)
    return SuperOp.from_matrix(powered_data, superop.dims, superop.num_ensemble_dims)


@jax.jit
def power_pauli_liouville(pauli_liouville: PauliLiouville, power: float) -> PauliLiouville:
    """
    Compute the power of a Pauli-Liouville matrix.

    Uses eigendecomposition (M^p = V @ diag(λ^p) @ V^(-1)) matching scipy.linalg.fractional_matrix_power.

    Note: For fractional powers of quantum channels, this may not preserve CPTP properties.

    :param pauli_liouville: The Pauli-Liouville matrix to exponentiate.
    :param power: The power to raise the Pauli-Liouville matrix to.
    :return: The Pauli-Liouville matrix raised to the specified power.
    """
    powered_data = _matrix_power_via_eig(pauli_liouville.matrix, power)
    return PauliLiouville.from_matrix(powered_data, pauli_liouville.dims, pauli_liouville.num_ensemble_dims)


@jax.jit
def power_kraus(kraus_map: KrausMap, power: float) -> KrausMap:
    """
    Compute the power of a Kraus map via Choi representation.

    This converts to Choi, takes the matrix power, and converts back to Kraus form.

    Note: For integer powers n > 1, this corresponds to composing the channel n times,
    but the resulting Kraus decomposition may have more operators than the original.
    The number of Kraus operators can grow as K → K*d² after conversion through Choi.

    For fractional powers, this may not preserve CPTP properties.

    :param kraus_map: The Kraus map to exponentiate.
    :param power: The power to raise the Kraus map to.
    :return: The Kraus map raised to the specified power.
    """
    # Convert to SuperOp, take power, convert back
    powered_superop = power_superop(kraus_to_superop(kraus_map), power)
    return superop_to_kraus(powered_superop)


@jax.jit
def density_matrix_power(density_matrix: DensityMatrix, power: float) -> DensityMatrix:
    """
    Compute the fractional power of a density matrix.

    :param density_matrix: The density matrix to exponentiate.
    :param power: The fractional power to raise the density matrix to.
    :return: The density matrix raised to the specified fractional power.
    """
    powered_data = _matrix_power_via_eig(density_matrix.matrix, power)
    return DensityMatrix.from_matrix(powered_data, density_matrix.dims, density_matrix.num_ensemble_dims)


@jax.jit
def state_vector_power(state_vector: StateVector, power: float) -> StateVector:
    """
    Compute the element-wise power of a state vector.

    Note: This raises each element to the power, not a matrix power.
    The result is re-normalized to maintain unit norm.

    :param state_vector: The state vector to exponentiate.
    :param power: The fractional power to raise the state vector elements to.
    :return: The state vector with elements raised to the specified power (normalized).
    """
    # Element-wise power
    powered_data = jnp.power(state_vector.matrix, power)

    # Re-normalize to maintain unit norm
    norm = jnp.linalg.norm(powered_data, axis=-1, keepdims=True)
    normalized_data = powered_data / norm

    return StateVector.from_matrix(normalized_data, state_vector.dims, state_vector.num_ensemble_dims)


@jax.jit
def power_unitary(unitary: Unitary, power: float) -> Unitary:
    """
    Compute the fractional power of a unitary matrix.

    :param unitary: The unitary object.
    :param power: The power to raise the unitary to.
    :return: The unitary raised to the specified power.
    """
    powered_data = _matrix_power_via_eig(unitary.matrix, power)
    return Unitary.from_matrix(powered_data, unitary.dims, unitary.num_ensemble_dims)


@jax.custom_vjp
def fractional_power_unitary(unitary: Unitary, exponent: float) -> Unitary:
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


def _fractional_power_unitary_fwd(unitary: Unitary, exponent: float):
    """Forward pass: compute U^exponent and save values for backward pass."""
    eigvals, eigvecs = jnp.linalg.eig(unitary.matrix)
    fractional_eigvals = jnp.power(eigvals, exponent)
    V_inv = jnp.linalg.inv(eigvecs)
    result_data = eigvecs @ jnp.diag(fractional_eigvals) @ V_inv
    result = Unitary.from_matrix(result_data, unitary.dims, unitary.num_ensemble_dims)
    # Save for backward pass
    return result, (eigvals, eigvecs, V_inv, exponent, unitary.dims, unitary.num_ensemble_dims)


def _fractional_power_unitary_bwd(residuals, g: Unitary):
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
fractional_power_unitary.defvjp(_fractional_power_unitary_fwd, _fractional_power_unitary_bwd)


# Wrap with JIT
fractional_power_unitary = partial(jax.jit, static_argnames=("exponent",))(fractional_power_unitary)  # type: ignore[assignment]
