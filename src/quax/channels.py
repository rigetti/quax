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

r"""Common noise channels and instruments as :class:`~quax.SuperOp` / :class:`~quax.QuantumInstrument`.

Most channels here are thin wrappers that evolve the corresponding :mod:`quax.lindbladians`
generator for a time ``t`` (default ``1.0``) to produce the CPTP channel::

    channel = qx.channels.depolarizing(0.1)            # SuperOp
    # equivalent to:
    channel = qx.evolve(qx.lindbladians.depolarizing(0.1), 1.0)

These are parameterized by a *rate* (folded into the generator); the amount of noise is the rate
integrated over ``t``.  For fine-grained control (Hamiltonian + noise, custom generators) build a
:class:`~quax.Lindbladian` and call :func:`~quax.evolve` directly, or add a gate to a Lindbladian
(``gate + lindbladian``).

The module also hosts the measurement instruments :func:`instrument_from_axis` and
:func:`instrument_from_confusion_and_transition`, which have no Lindbladian generator.
"""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from . import lindbladians
from ._exponentiation import evolve
from ._quantum_objects import QuantumInstrument, SuperOp, _extract_measured_index


def depolarizing(rate: float | Array, dims: Tuple[int, ...] = (2,), t: float = 1.0) -> SuperOp:
    """Depolarizing channel on ``dims`` (default a single qubit). See :func:`quax.lindbladians.depolarizing`."""
    return evolve(lindbladians.depolarizing(rate, dims), t)


def amplitude_damping(rate: float | Array, dims: Tuple[int, ...] = (2,), t: float = 1.0) -> SuperOp:
    """Amplitude-damping (T1) channel on a qudit. See :func:`quax.lindbladians.amplitude_damping`."""
    return evolve(lindbladians.amplitude_damping(rate, dims), t)


def dephasing(rate: float | Array, t: float = 1.0) -> SuperOp:
    """Dephasing channel. See :func:`quax.lindbladians.dephasing`."""
    return evolve(lindbladians.dephasing(rate), t)


def bit_flip(rate: float | Array, t: float = 1.0) -> SuperOp:
    """Bit-flip channel. See :func:`quax.lindbladians.bit_flip`."""
    return evolve(lindbladians.bit_flip(rate), t)


def phase_flip(rate: float | Array, t: float = 1.0) -> SuperOp:
    """Phase-flip channel. See :func:`quax.lindbladians.phase_flip`."""
    return evolve(lindbladians.phase_flip(rate), t)


def leakage(rate: float | Array, t: float = 1.0) -> SuperOp:
    """Leakage channel on a qutrit (|1⟩ → |2⟩). See :func:`quax.lindbladians.leakage`."""
    return evolve(lindbladians.leakage(rate), t)


def seepage(rate: float | Array, t: float = 1.0) -> SuperOp:
    """Seepage channel on a qutrit (|2⟩ → |1⟩). See :func:`quax.lindbladians.seepage`."""
    return evolve(lindbladians.seepage(rate), t)


def thermal_relaxation(t1: float | Array, tphi: float | Array, p1: float | Array = 0.0, t: float = 1.0) -> SuperOp:
    """Finite-temperature thermal-relaxation channel. See :func:`quax.lindbladians.thermal_relaxation`.

    ``p1`` is the equilibrium excited-state population (default 0 = zero temperature).  Note ``tphi``
    is the pure-dephasing time ``Tφ``, not ``T₂`` (``1/T₂ = 1/(2·T₁) + 1/Tφ``).
    """
    return evolve(lindbladians.thermal_relaxation(t1, tphi, p1), t)


# ---------------------------------------------------------------------------
# Measurement instruments (no Lindbladian generator)
# ---------------------------------------------------------------------------


def instrument_from_confusion_and_transition(
    confusion_matrix: Array,
    transition_matrix: Array,
    dims: Tuple[int, ...],
    measured_qudits: Optional[Tuple[int, ...]] = None,
) -> QuantumInstrument:
    r"""Construct a quantum instrument from a confusion matrix and a transition matrix.

    The confusion matrix is the probability of observing each measurement outcome given the true
    state of the measured qudits, ``C_{ij} = P(outcome i | true state j)`` (column-stochastic).

    The transition matrix describes the measurement backaction on the post-measurement state,
    ``T_{ij} = P(post-measurement state i | pre-measurement state j)`` (column-stochastic).

    Each Kraus operator is :math:`K_{i,j,k} = \sqrt{C_{i, j} T_{k, j}} |k\rangle\langle j|`, giving
    the instrument :math:`\mathcal{M}_i(\rho) = \sum_{j, k} C_{i,j} T_{k, j} \langle j|\rho|k\rangle |k\rangle\langle k|`.

    :param confusion_matrix: ``(num_outcomes, d_measured)`` column-stochastic matrix.
    :param transition_matrix: ``(d_total, d_total)`` column-stochastic matrix.
    :param dims: Per-qudit dimensions.
    :param measured_qudits: Indices of measured qudits.  Defaults to all.
    """
    if measured_qudits is None:
        measured_qudits = tuple(range(len(dims)))

    d_total = reduce(mul, dims, 1)
    d_measured = reduce(mul, (dims[i] for i in measured_qudits), 1)
    num_outcomes = confusion_matrix.shape[0]

    # Shape checks use static shapes (always run; jit-safe).
    if confusion_matrix.shape != (num_outcomes, d_measured):
        raise ValueError(
            f"Confusion matrix shape {confusion_matrix.shape} does not match "
            f"(num_outcomes={num_outcomes}, d_measured={d_measured})."
        )
    if transition_matrix.shape != (d_total, d_total):
        raise ValueError(
            f"Transition matrix shape {transition_matrix.shape} does not match (d_total={d_total}, d_total={d_total})."
        )
    # Value checks (non-negativity, column-stochasticity) run only for concrete inputs; they are
    # skipped under `jax.jit` tracing (branching on traced arrays is not possible) so the function
    # stays jit-compatible.
    for label, m in (("Confusion", confusion_matrix), ("Transition", transition_matrix)):
        if not isinstance(m, jax.core.Tracer):  # pyright: ignore[reportAttributeAccessIssue]
            if not bool(jnp.all(m >= -1e-14)):
                raise ValueError(f"{label} matrix entries must be non-negative.")
            if not bool(jnp.allclose(jnp.sum(m, axis=0), 1.0, atol=1e-6)):
                raise ValueError(f"{label} matrix columns must sum to 1.")

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
    return QuantumInstrument.from_matrix(matrices, (dims, dims), measured_qudits)


def instrument_from_axis(theta: float = 0.0, phi: float = 0.0, sharpness: float = 1.0) -> QuantumInstrument:
    """Create a single-qubit instrument from a Bloch-sphere measurement axis.

    The angles follow standard Bloch sphere notation.  ``theta=0, phi=0`` is the Z-axis
    (computational-basis measurement).

    :param theta: Colatitude with respect to the z-axis.
    :param phi: Longitude with respect to the x-axis.
    :param sharpness: Measurement sharpness.  1.0 is projective, 0.0 is no measurement.
    :return: A single-qubit :class:`~quax.QuantumInstrument`.
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
