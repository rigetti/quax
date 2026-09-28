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

"""Error generators of quantum channels (Blume-Kohout et al., arXiv:2103.01928).

Engines for :class:`~quax.ErrorGenerator`: taking the logarithm of a channel, reading off the
Hamiltonian and dissipator coordinates, building elementary error generators, and converting a
completely positive generator to a :class:`~quax.Lindbladian`. Kept out of ``_quantum_objects`` so
the class definition stays lean; imported lazily by the class.

Throughout, :math:`F_i` is the Hermitian operator basis of :func:`n_qudit_herm_basis` (the Paulis
for qubits), with :math:`F_0 = I` and :math:`\\mathrm{Tr}[F_i F_j] = d\\,\\delta_{ij}`. A
superoperator matrix acts on column-stacked ``vec(ρ)``, so the map :math:`\\rho \\mapsto A \\rho B`
has matrix :math:`B^T \\otimes A`.
"""

from __future__ import annotations

from functools import lru_cache, reduce, singledispatch
from operator import mul
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ._operator_basis import n_qudit_herm_basis
from ._quantum_objects import (
    Choi,
    ErrorGenerator,
    KrausMap,
    Lindbladian,
    Observable,
    Operator,
    PauliLiouville,
    SuperOp,
)
from ._superoperator_transformations import to_superop

ErrorGeneratorKind = Literal["H", "S", "C", "A"]
"""The four sectors of elementary error generators: Hamiltonian, stochastic, correlation, active."""


# --------------------------------------------------------------------------- #
# Basis constants
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=32)
def _herm_basis(dims: tuple[int, ...]) -> np.ndarray:
    """The Hermitian operator basis ``(d², d, d)`` as a concrete NumPy constant (``F_0 = I``)."""
    return np.asarray(n_qudit_herm_basis(dims).matrix)


@lru_cache(maxsize=32)
def error_generator_labels(dims: tuple[int, ...]) -> tuple[str, ...]:
    """Labels of the non-identity Hermitian basis elements, e.g. ``("X", "Y", "Z")`` for a qubit."""
    from ._visualization import _weyl_labels

    return tuple(_weyl_labels(dims)[1:])


def _label_index(label: str, dims: tuple[int, ...]) -> int:
    """Index of a non-identity basis label into the rate arrays."""
    labels = error_generator_labels(dims)
    if label not in labels:
        raise ValueError(f"Unknown basis label {label!r} for dims {dims}; expected one of {labels}.")
    return labels.index(label)


@lru_cache(maxsize=32)
def _dissipator_basis(dims: tuple[int, ...]) -> np.ndarray:
    """Superoperator matrices of ``ρ ↦ F_i ρ F_j − ½{F_j F_i, ρ}`` for ``i, j ≥ 1``: ``(n, n, d², d²)``."""
    basis = _herm_basis(dims)[1:]
    d = basis.shape[-1]
    identity = np.eye(d)
    # F_i ρ F_j  ->  F_j^T ⊗ F_i
    sandwich = np.einsum("jba,icd->ijacbd", basis, basis)
    # F_j F_i ρ  ->  I ⊗ (F_j F_i);   ρ F_j F_i  ->  (F_j F_i)^T ⊗ I
    products = np.einsum("jab,ibc->ijac", basis, basis)
    left = np.einsum("ab,ijcd->ijacbd", identity, products)
    right = np.einsum("ijba,cd->ijacbd", products, identity)
    n = basis.shape[0]
    return (sandwich - 0.5 * (left + right)).reshape(n, n, d * d, d * d)


@lru_cache(maxsize=32)
def _commutator_basis(dims: tuple[int, ...]) -> np.ndarray:
    """Superoperator matrices of ``ρ ↦ −i[F_i, ρ]`` for ``i ≥ 1``: ``(n, d², d²)``."""
    basis = _herm_basis(dims)[1:]
    d = basis.shape[-1]
    identity = np.eye(d)
    left = np.einsum("ab,icd->iacbd", identity, basis)
    right = np.einsum("iba,cd->iacbd", basis, identity)
    return (-1j * (left - right)).reshape(basis.shape[0], d * d, d * d)


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #


@jax.jit
def hamiltonian_and_dissipator(generator: ErrorGenerator) -> tuple[Array, Array]:
    """Hamiltonian rates ``h`` and dissipator matrix ``K`` of an error generator.

    Expands the generator in the χ (process-matrix) form :math:`\\mathcal{L}[\\rho] =
    \\sum_{ij} \\chi_{ij} F_i \\rho F_j`, whose coefficients are orthogonal projections,
    :math:`\\chi_{ij} = \\mathrm{Tr}[(F_j^T \\otimes F_i)^\\dagger \\mathcal{L}] / d^2`. Hermiticity
    and trace preservation then give :math:`K = \\chi_{i,j \\geq 1}` and
    :math:`h_i = -\\mathrm{Im}\\,\\chi_{i0}`.
    """
    dims = generator.dims[0]
    with jax.ensure_compile_time_eval():
        basis = jnp.asarray(_herm_basis(dims))
    d = basis.shape[-1]
    generator_4 = generator.matrix.reshape(generator.ensemble_size + (d, d, d, d))
    # χ_ij = Σ conj((F_j^T)_{ab}) conj((F_i)_{cd}) L_{(a,c),(b,d)} / d², with (F_j^T)_{ab} = (F_j)_{ba}
    chi = jnp.einsum("jba,icd,...acbd->...ij", jnp.conj(basis), jnp.conj(basis), generator_4) / d**2
    hamiltonian_rates = -jnp.imag(chi[..., 1:, 0])
    dissipator = chi[..., 1:, 1:]
    dissipator = 0.5 * (dissipator + jnp.conj(jnp.swapaxes(dissipator, -1, -2)))
    return hamiltonian_rates, dissipator


def hamiltonian_from_rates(hamiltonian_rates: Array, dims: tuple[int, ...]) -> Observable:
    """The traceless Hamiltonian :math:`\\sum_i h_i F_i` from its rates."""
    with jax.ensure_compile_time_eval():
        basis = jnp.asarray(_herm_basis(dims)[1:])
    matrix = jnp.einsum("...i,iab->...ab", hamiltonian_rates.astype(complex), basis)
    return Observable.from_matrix(matrix, (dims, dims))


def error_generator_from_coordinates(
    hamiltonian_rates: Array, dissipator_matrix: Array, dims: tuple[int, ...]
) -> ErrorGenerator:
    """Assemble an error generator from Hamiltonian rates ``h`` and a dissipator matrix ``K``.

    The inverse of :func:`hamiltonian_and_dissipator`.
    """
    with jax.ensure_compile_time_eval():
        commutators = jnp.asarray(_commutator_basis(dims))
        dissipators = jnp.asarray(_dissipator_basis(dims))
    matrix = jnp.einsum("...i,ixy->...xy", hamiltonian_rates.astype(complex), commutators) + jnp.einsum(
        "...ij,ijxy->...xy", dissipator_matrix.astype(complex), dissipators
    )
    return ErrorGenerator.from_matrix(matrix, (dims, dims))


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #


def elementary_error_generator(
    kind: ErrorGeneratorKind, p: str, q: str | None = None, dims: tuple[int, ...] | None = None
) -> ErrorGenerator:
    """An elementary error generator of arXiv:2103.01928.

    For basis labels ``p`` and ``q`` (e.g. ``"XI"``, ``"ZZ"``; see :attr:`ErrorGenerator.labels`):

    - ``"H"``: :math:`H_P[\\rho] = -i[P, \\rho]`
    - ``"S"``: :math:`S_P[\\rho] = P \\rho P - \\tfrac{1}{2}\\{P^2, \\rho\\}`
    - ``"C"``: :math:`C_{P,Q}[\\rho] = P \\rho Q + Q \\rho P - \\tfrac{1}{2}\\{\\{P, Q\\}, \\rho\\}`
    - ``"A"``: :math:`A_{P,Q}[\\rho] = i\\left(P \\rho Q - Q \\rho P + \\tfrac{1}{2}\\{[P, Q], \\rho\\}\\right)`

    For Paulis, :math:`P^2 = I` and these are exactly the paper's definitions. Combine them with
    ``+`` and real scalars to build a general error generator.

    :param kind: The sector: ``"H"``, ``"S"``, ``"C"`` or ``"A"``.
    :param p: The first basis label.
    :param q: The second basis label, required (and distinct from ``p``) for ``"C"`` and ``"A"``.
    :param dims: Per-qudit dimensions. Defaults to qubits, one per character of ``p``.
    :return: The elementary generator.
    """
    if dims is None:
        dims = (2,) * len(p)
    n = len(error_generator_labels(dims))
    i = _label_index(p, dims)
    hamiltonian_rates = jnp.zeros(n)
    dissipator = jnp.zeros((n, n), dtype=complex)
    if kind == "H":
        hamiltonian_rates = hamiltonian_rates.at[i].set(1.0)
    elif kind == "S":
        dissipator = dissipator.at[i, i].set(1.0)
    elif kind in ("C", "A"):
        if q is None or q == p:
            raise ValueError(f"A {kind!r} error generator needs two distinct labels, got p={p!r}, q={q!r}.")
        j = _label_index(q, dims)
        value = 1.0 if kind == "C" else 1j
        dissipator = dissipator.at[i, j].set(value).at[j, i].set(jnp.conj(value))
    else:
        raise ValueError(f"Unknown error generator kind {kind!r}; expected 'H', 'S', 'C' or 'A'.")
    return error_generator_from_coordinates(hamiltonian_rates, dissipator, dims)


@jax.jit
def _matrix_log_via_eig(matrix: Array) -> Array:
    """Principal matrix logarithm via eigendecomposition: log M = V @ diag(log λ) @ V^(-1)."""
    eigvals, eigvecs = jnp.linalg.eig(matrix)
    return (eigvecs * jnp.log(eigvals)[..., None, :]) @ jnp.linalg.inv(eigvecs)


@singledispatch
def error_generator(channel) -> ErrorGenerator:
    """The error generator :math:`\\mathcal{L} = \\log \\mathcal{E}` of a channel.

    For a superoperator this is the principal matrix logarithm, computed by eigendecomposition,
    so :func:`~quax.evolve` inverts it: ``evolve(error_generator(E)) ≈ E``. For a
    :class:`Lindbladian` it is the generator itself, exactly.

    To get the error generator of a noisy *gate*, pass the error alone, with the ideal gate
    factored out. With the post-gate convention of arXiv:2103.01928, :math:`\\mathcal{E} =
    e^{\\mathcal{L}} \\mathcal{U}`, that is ``error_generator(noisy @ ideal.h)``.

    .. note::
        The principal logarithm is the right generator for small and moderate errors. It is
        ill-defined when the channel has an eigenvalue on the negative real axis (or zero), and
        inaccurate for a non-diagonalizable superoperator. A large error, such as a near-π
        rotation, lands on a different branch than the physical generator.

    :param channel: A :class:`SuperOp`, :class:`PauliLiouville`, :class:`Choi`, :class:`KrausMap`
        or :class:`Lindbladian`.
    :return: The error generator.
    """
    raise TypeError(
        f"error_generator() does not support type {type(channel)!r}. "
        "Expected SuperOp, PauliLiouville, Choi, KrausMap or Lindbladian."
    )


@error_generator.register(SuperOp)
@jax.jit
def _superop_error_generator(channel: SuperOp) -> ErrorGenerator:
    return ErrorGenerator.from_matrix(_matrix_log_via_eig(channel.matrix), channel.dims)


@error_generator.register(PauliLiouville)
@error_generator.register(Choi)
@error_generator.register(KrausMap)
def _superoperator_error_generator(channel: PauliLiouville | Choi | KrausMap) -> ErrorGenerator:
    return _superop_error_generator(to_superop(channel))


@error_generator.register(Lindbladian)
def _lindbladian_error_generator(channel: Lindbladian) -> ErrorGenerator:
    return ErrorGenerator.from_matrix(channel.matrix, channel.dims)


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #


def error_generator_to_lindbladian(generator: ErrorGenerator, atol: float = 1e-8) -> Lindbladian:
    """Convert a completely positive error generator to a :class:`Lindbladian`.

    Diagonalizes :math:`K = V \\mathrm{diag}(\\gamma) V^\\dagger` and uses the jump operators
    :math:`L_k = \\sqrt{\\gamma_k} \\sum_i V_{ik} F_i`, so :math:`\\sum_{ij} K_{ij} F_i \\rho F_j =
    \\sum_k L_k \\rho L_k^\\dagger`. The Hamiltonian is kept as is.

    :raises ValueError: If :math:`K` has an eigenvalue below ``-atol`` (the generator is not CP).
    """
    dims = generator.dims[0]
    gammas, vectors = jnp.linalg.eigh(generator.dissipator_matrix)
    min_gamma = float(jnp.min(gammas))
    if min_gamma < -atol:
        raise ValueError(
            f"The error generator is not completely positive (dissipator eigenvalue {min_gamma:.3g}), "
            "so it has no Lindbladian form."
        )
    basis = jnp.asarray(_herm_basis(dims)[1:])
    weights = vectors * jnp.sqrt(jnp.clip(gammas, 0.0))[..., None, :]  # column k: √γ_k V_{·k}
    jumps = jnp.einsum("...ik,iab->...kab", weights, basis)
    d = reduce(mul, dims, 1)
    jump_operators = Operator.from_matrix(jumps.reshape(generator.ensemble_size + (-1, d, d)), (dims, dims))
    return Lindbladian(hamiltonian=generator.hamiltonian, jump_operators=jump_operators)
