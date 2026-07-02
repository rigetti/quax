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

"""Matrix exponentials and integer/fractional powers of quantum objects.

This module provides:

- :func:`evolve` — single-dispatch evolution of generators to quantum objects.
  Hamiltonian → Unitary via ``exp(-i·t·H)``; Lindbladian → SuperOp via ``exp(t·L)``.
- :func:`cis` — deprecated alias for ``evolve(H, t=1.0)``.
- :func:`exp` — bare matrix exponential (no global-phase correction).
- ``power_*`` family — integer and fractional matrix powers for each representation.

All power implementations use eigendecomposition matching scipy.linalg.fractional_matrix_power:

    M^p = V @ diag(λ^p) @ V^(-1)

**For quantum channels (superoperators):**
- Integer powers correspond to channel composition (always CPTP).
- Fractional powers preserve CPTP for *infinitely divisible* channels — those of the form
  ``e^{tL}`` for a valid Lindbladian ``L`` (e.g. anything returned by ``evolve``), since
  ``M^s = e^{(s·t)L}`` is then itself a Lindbladian evolution. For a *general* channel this
  need not hold: the principal power computed here (via eigendecomposition) can leave the
  CPTP set.
- For guaranteed CPTP at any fractional power, build the channel from a Lindbladian and
  scale the time: ``evolve(L, s·t)`` is CPTP for ``s·t ≥ 0``.
"""

from functools import partial, singledispatch
from typing import overload

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import (
    Choi,
    DensityMatrix,
    KrausMap,
    Lindbladian,
    Observable,
    Operator,
    PauliLiouville,
    StateVector,
    SuperOp,
    Unitary,
)
from ._superoperator_transformations import choi_to_superop, kraus_to_superop, superop_to_choi, superop_to_kraus


@overload
def exp(operator: Observable) -> Observable: ...


@overload
def exp(operator: Operator) -> Operator: ...


def exp(operator: Operator) -> Operator:
    """Compute the matrix exponential of a quantum object.

    Dispatches on the type of the input:

    - ``Observable`` → ``Observable`` (exp of Hermitian is Hermitian positive-definite)
    - ``Operator``   → ``Operator``   (general matrix exponential with global-phase removal)
    """
    if isinstance(operator, Observable):
        return _exp_observable(operator)
    return _exp_operator(operator)


def _exp_operator(operator: Operator) -> Operator:
    """Matrix exponential of a general operator, with global-phase removal."""
    dims = operator.dims
    eiH = jax.scipy.linalg.expm(operator.matrix)
    phase = jnp.exp(-1j * jnp.angle(eiH[..., 0, 0]))
    return Operator.from_matrix(phase[..., None, None] * eiH, dims)


def _exp_observable(operator: Observable) -> Observable:
    """Matrix exponential of an observable (Hermitian): result is Hermitian positive-definite."""
    dims = operator.dims
    result = jax.scipy.linalg.expm(operator.matrix)
    return Observable.from_matrix(result, dims)


def cis(operator: Operator) -> Unitary:
    """Compute the complex exponential ``exp(i·operator)`` of a quantum object, returning a Unitary.

    .. deprecated::
        Use :func:`evolve` instead. Note the sign convention differs: :func:`evolve`
        uses the Schrödinger convention ``exp(-i·t·H)``, so ``cis(H)`` is equivalent to
        ``evolve(H, t=-1.0)``.

    :param operator: The operator (typically Hermitian/Observable) acting as the generator.
    :return: The unitary exp(iH) with determinant-phase correction.
    """
    dims = operator.dims
    result = jax.scipy.linalg.expm(1j * operator.matrix)
    phase = jnp.exp(-1j * jnp.angle(result[..., 0, 0]))
    return Unitary.from_matrix(phase[..., None, None] * result, dims)


@singledispatch
def evolve(operator, t: float = 1.0):
    """Evolve a quantum generator for time ``t``, returning the corresponding quantum object.

    Dispatches on the type of ``operator``:

    - :class:`Observable` (Hamiltonian) → :class:`Unitary` via ``exp(-i·t·H)``
    - :class:`Lindbladian` (open-system generator) → :class:`SuperOp` via ``exp(t·L)``

    The Hamiltonian branch follows the standard Schrödinger convention
    ``U(t) = exp(-i·t·H)``, consistent with the coherent term ``-i[H, ρ]`` of the
    GKSL generator (so ``evolve`` agrees on both branches for closed-system dynamics).

    The Lindbladian branch returns a CPTP map for ``t ≥ 0``.

    .. note::
        **Precondition:** ``t >= 0`` for the Lindbladian branch.  A negative ``t`` runs the
        semigroup backwards and yields a non-CPTP map.  This is *not* validated — it cannot
        be asserted cleanly under JAX tracing — so it is the caller's responsibility.

    :param operator: A Hamiltonian (:class:`Observable`) or Lindbladian generator.
    :param t: Evolution time (default 1.0). Must be ``>= 0`` for a CPTP Lindbladian channel.
    :return: The evolved quantum object.

    Examples::

        import quax as qx

        # Hamiltonian evolution (replaces cis)
        U = qx.evolve(qx.gates.H_obs, t=jnp.pi / 2)

        # Lindbladian channel (guaranteed CPTP for t >= 0)
        L = qx.amplitude_damping_lindbladian(gamma=0.1)
        channel = qx.evolve(L, t=1.0)
    """
    raise TypeError(f"evolve is not implemented for {type(operator).__name__}.")


@evolve.register(Observable)
@jax.jit
def _evolve_observable(observable: Observable, t: float = 1.0) -> Unitary:
    """Compute the Schrödinger propagator ``exp(-i·t·H)``, returning a Unitary.

    Unlike :func:`cis`, no global-phase normalization is applied: the result is the literal
    propagator, so it composes correctly with other phase-sensitive objects and carries the
    true dynamical phase.  (The ``[0,0]``-anchored normalization used by ``cis`` is also
    ill-defined when that entry is ~0, e.g. ``exp(-iπX/2) = -iX``.)
    """
    result = jax.scipy.linalg.expm(-1j * t * observable.matrix)
    return Unitary.from_matrix(result, observable.dims)


@evolve.register(Lindbladian)
@jax.jit
def _evolve_lindbladian(generator: Lindbladian, t: float = 1.0) -> SuperOp:
    """Compute exp(t·L) for a Lindbladian generator, returning a CPTP SuperOp."""
    matrix = generator.matrix  # (*ensemble, d², d²)

    if generator.num_ensemble_dims == 0:
        result = jax.scipy.linalg.expm(t * matrix)
    else:
        shape = matrix.shape
        flat = matrix.reshape(-1, shape[-2], shape[-1])
        result = jax.vmap(lambda m: jax.scipy.linalg.expm(t * m))(flat).reshape(shape)

    return SuperOp.from_matrix(result, generator.dims)


def _matrix_power_via_eig(matrix: Array, power: float) -> Array:
    """Compute matrix power using eigendecomposition: M^p = V @ diag(λ^p) @ V^(-1)."""
    eigvals, eigvecs = jnp.linalg.eig(matrix)
    powered_eigvals = jnp.power(eigvals, power)
    V_scaled = eigvecs * powered_eigvals[..., None, :]
    V_inv = jnp.linalg.inv(eigvecs)
    return V_scaled @ V_inv


def _matrix_power_via_lindbladian(matrix: Array, power: float) -> Array:
    """Compute matrix power via M^α = exp(α·log(M)) — correct for infinitely divisible channels."""
    eigvals, eigvecs = jnp.linalg.eig(matrix)
    log_eigvals = jnp.log(eigvals)
    V_scaled = eigvecs * log_eigvals[..., None, :]
    V_inv = jnp.linalg.inv(eigvecs)
    log_matrix = V_scaled @ V_inv
    return jax.scipy.linalg.expm(power * log_matrix)


@jax.jit
def power_choi(choi: Choi, power: float) -> Choi:
    """Compute the power of a Choi matrix via eigendecomposition.

    Note: For fractional powers of quantum channels, this may not preserve CPTP properties.

    :param choi: The Choi matrix to exponentiate.
    :param power: The power to raise the Choi matrix to.
    :return: The Choi matrix raised to the specified power.
    """
    powered_superop = power_superop(choi_to_superop(choi), power)
    return superop_to_choi(powered_superop)


@jax.jit
def power_superop(superop: SuperOp, power: float) -> SuperOp:
    """Compute the power of a superoperator matrix via eigendecomposition.

    Note: For fractional powers of quantum channels, this may not preserve CPTP properties.
    For physically meaningful fractional powers, use ``evolve(Lindbladian.from_operators(...), alpha*t)``.

    :param superop: The superoperator to exponentiate.
    :param power: The power to raise the superoperator to.
    :return: The superoperator raised to the specified power.
    """
    powered_data = _matrix_power_via_eig(superop.matrix, power)
    return SuperOp.from_matrix(powered_data, superop.dims)


@jax.jit
def power_pauli_liouville(pauli_liouville: PauliLiouville, power: float) -> PauliLiouville:
    """Compute the power of a Pauli-Liouville matrix via eigendecomposition.

    :param pauli_liouville: The Pauli-Liouville matrix to exponentiate.
    :param power: The power to raise the Pauli-Liouville matrix to.
    :return: The Pauli-Liouville matrix raised to the specified power.
    """
    powered_data = _matrix_power_via_eig(pauli_liouville.matrix, power)
    return PauliLiouville.from_matrix(powered_data, pauli_liouville.dims)


@jax.jit
def power_kraus(kraus_map: KrausMap, power: float) -> KrausMap:
    """Compute the power of a Kraus map via the superoperator representation.

    Note: For integer powers n > 1, this corresponds to composing the channel n times.
    For fractional powers, this may not preserve CPTP properties.

    :param kraus_map: The Kraus map to exponentiate.
    :param power: The power to raise the Kraus map to.
    :return: The Kraus map raised to the specified power.
    """
    powered_superop = power_superop(kraus_to_superop(kraus_map), power)
    return superop_to_kraus(powered_superop)


@jax.jit
def density_matrix_power(density_matrix: DensityMatrix, power: float) -> DensityMatrix:
    """Compute the fractional power of a density matrix.

    :param density_matrix: The density matrix to exponentiate.
    :param power: The fractional power to raise the density matrix to.
    :return: The density matrix raised to the specified fractional power.
    """
    powered_data = _matrix_power_via_eig(density_matrix.matrix, power)
    return DensityMatrix.from_matrix(powered_data, density_matrix.dims)


@jax.jit
def state_vector_power(state_vector: StateVector, power: float) -> StateVector:
    """Compute the element-wise power of a state vector (re-normalized).

    :param state_vector: The state vector to exponentiate.
    :param power: The fractional power to raise the state vector elements to.
    :return: The state vector with elements raised to the specified power (normalized).
    """
    powered_data = jnp.power(state_vector.matrix, power)
    norm = jnp.linalg.norm(powered_data, axis=-1, keepdims=True)
    normalized_data = powered_data / norm
    return StateVector.from_matrix(normalized_data, state_vector.dims)


@jax.jit
def power_unitary(unitary: Unitary, power: float) -> Unitary:
    """Compute the fractional power of a unitary matrix.

    :param unitary: The unitary object.
    :param power: The power to raise the unitary to.
    :return: The unitary raised to the specified power.
    """
    powered_data = _matrix_power_via_eig(unitary.matrix, power)
    return Unitary.from_matrix(powered_data, unitary.dims)


@jax.jit
def power_observable(observable: Observable, power: float) -> Observable:
    """Compute the fractional power of an observable (Hermitian matrix).

    :param observable: The observable object.
    :param power: The power to raise the observable to.
    :return: The observable raised to the specified power.
    """
    powered_data = _matrix_power_via_eig(observable.matrix, power)
    return Observable.from_matrix(powered_data, observable.dims)


@jax.custom_vjp
def fractional_power_unitary(unitary: Unitary, exponent: float) -> Unitary:
    """Compute U^exponent for a unitary matrix with gradient support via custom VJP.

    :param unitary: The unitary object.
    :param exponent: The exponent (e.g., 1/n for the n-th root).
    :return: The matrix U^exponent as a Unitary object.
    """
    eigvals, eigvecs = jnp.linalg.eig(unitary.matrix)
    fractional_eigvals = jnp.power(eigvals, exponent)
    result_data = eigvecs @ jnp.diag(fractional_eigvals) @ jnp.linalg.inv(eigvecs)
    return Unitary.from_matrix(result_data, unitary.dims)


def _fractional_power_unitary_fwd(unitary: Unitary, exponent: float):
    """Forward pass: compute U^exponent and save values for backward pass."""
    eigvals, eigvecs = jnp.linalg.eig(unitary.matrix)
    fractional_eigvals = jnp.power(eigvals, exponent)
    V_inv = jnp.linalg.inv(eigvecs)
    result_data = eigvecs @ jnp.diag(fractional_eigvals) @ V_inv
    result = Unitary.from_matrix(result_data, unitary.dims)
    return result, (eigvals, eigvecs, V_inv, exponent, unitary.dims)


def _fractional_power_unitary_bwd(residuals, g: Unitary):
    """Backward pass: compute VJP using Fréchet derivative of matrix power."""
    eigvals, eigvecs, V_inv, exponent, dims = residuals
    G = V_inv @ g.matrix @ eigvecs
    λ = eigvals
    λα = jnp.power(λ, exponent)
    λ_diff = λ[:, None] - λ[None, :]
    λα_diff = λα[:, None] - λα[None, :]
    safe_denom = jnp.where(jnp.abs(λ_diff) < 1e-8, 1.0, λ_diff)
    L = jnp.where(
        jnp.abs(λ_diff) < 1e-8,
        exponent * jnp.power(λ[:, None], exponent - 1),
        λα_diff / safe_denom,  # type: ignore[operator]
    )
    G_eigen = L * G
    grad_unitary_data = eigvecs @ G_eigen @ V_inv
    grad_unitary = Unitary.from_matrix(grad_unitary_data, dims)
    return (grad_unitary, None)


fractional_power_unitary.defvjp(_fractional_power_unitary_fwd, _fractional_power_unitary_bwd)
fractional_power_unitary = partial(jax.jit, static_argnames=("exponent",))(fractional_power_unitary)  # type: ignore[assignment]
