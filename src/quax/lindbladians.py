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

r"""Lindbladian factories: GKSL generators for common open-system noise channels.

Each factory returns a :class:`~quax.Lindbladian` generator ``L``; the corresponding
CPTP channel is obtained with ``quax.evolve(L, t)``.  Rates are the arguments to the
factories (they are absorbed into the jump operators as :math:`\sqrt{\gamma}\,L`).

All factories are jittable and broadcast over their rate arguments: passing an array of
shape ``(n,)`` (or any leading batch shape) yields an ensemble of ``n`` generators.

Example::

    import jax
    import jax.numpy as jnp
    import quax as qx

    L = qx.lindbladians.amplitude_damping(0.1)          # single generator
    channel = qx.evolve(L, 0.5)                          # CPTP SuperOp

    gammas = jnp.array([0.1, 0.2, 0.3])
    ensemble = qx.lindbladians.amplitude_damping(gammas)  # ensemble_size == (3,)
    fast = jax.jit(qx.lindbladians.amplitude_damping)(0.1)
"""

import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Lindbladian, Operator
from .gates import X, Y, Z

# Qubit operators
_SIGMA_MINUS = Operator.from_matrix(jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex), ((2,), (2,)))

# Qutrit operators
_SIGMA_12 = Operator.from_matrix(
    jnp.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=complex), ((3,), (3,))
)  # |2⟩⟨1| — leakage out of computational subspace
_SIGMA_21 = Operator.from_matrix(
    jnp.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=complex), ((3,), (3,))
)  # |1⟩⟨2| — seepage back into computational subspace


def amplitude_damping(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for the amplitude damping (T1 relaxation) channel.

    Jump operator: :math:`L = \\sqrt{\\gamma}\\,|0\\rangle\\langle 1|`.

    The resulting CPTP channel ``evolve(L, t)`` matches
    ``relaxation_operators(1 - exp(-gamma * t))`` converted to a SuperOp.

    :param gamma: Relaxation rate (1/T1). Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the amplitude damping channel.
    """
    # scale[..., None, None, None] adds the (n_ops=1, d, d) axes so a scalar rate gives
    # shape (1, 2, 2) and a batched rate of shape (n,) gives (n, 1, 2, 2).
    scale = jnp.sqrt(gamma)
    L = scale[..., None, None, None] * _SIGMA_MINUS.matrix
    return Lindbladian.from_operators(None, Operator.from_matrix(L, ((2,), (2,))))


def dephasing(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for the dephasing channel.

    Jump operator: :math:`L = \\sqrt{\\gamma/2}\\,Z`.

    The resulting CPTP channel ``evolve(L, t)`` matches
    ``dephasing_operators(1 - exp(-gamma * t))`` converted to a SuperOp.

    :param gamma: Dephasing rate. Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the dephasing channel.
    """
    scale = jnp.sqrt(gamma / 2.0)
    L = scale[..., None, None, None] * Z.matrix
    return Lindbladian.from_operators(None, Operator.from_matrix(L, ((2,), (2,))))


def depolarizing(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for the depolarizing channel.

    Jump operators: :math:`L_k = \\sqrt{\\gamma/3}\\,\\sigma_k` for k ∈ {X, Y, Z}.

    The resulting CPTP channel ``evolve(L, t)`` matches
    ``depolarizing_operators(p)`` with ``p = 3/4 * (1 - exp(-4*gamma*t/3))``
    converted to a SuperOp.

    :param gamma: Depolarizing rate. Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the depolarizing channel.
    """
    # scale[..., None, None] broadcasts the rate over each Pauli; stacking on axis=-3
    # inserts the n_ops axis, giving (3, 2, 2) for a scalar rate and (n, 3, 2, 2) for a batch.
    scale = jnp.sqrt(gamma / 3.0)[..., None, None]
    L_stack = jnp.stack([scale * X.matrix, scale * Y.matrix, scale * Z.matrix], axis=-3)
    return Lindbladian.from_operators(None, Operator.from_matrix(L_stack, ((2,), (2,))))


def thermal_relaxation(t1: float | Array, tphi: float | Array) -> Lindbladian:
    """Lindbladian generator for the thermal relaxation channel.

    Combines amplitude damping (1/T1) and pure dephasing (1/Tφ):

    - :math:`L_1 = \\sqrt{1/T_1}\\,|0\\rangle\\langle 1|`
    - :math:`L_2 = \\sqrt{1/T_\\varphi}\\,Z/\\sqrt{2}`

    The resulting channel ``evolve(L, t)`` matches
    ``thermal_relaxation_choi([t1], [tphi], t)`` converted to a SuperOp.

    :param t1: T1 relaxation time (energy decay). Must be positive. Arrays produce an ensemble.
    :param tphi: Pure dephasing time (Tφ, not T2). Must be positive. Arrays produce an ensemble.
    :return: Lindbladian generator for the thermal relaxation channel.
    """
    scale_t1 = jnp.sqrt(1.0 / t1)[..., None, None]
    scale_tphi = (jnp.sqrt(1.0 / tphi) / jnp.sqrt(2.0))[..., None, None]
    L_stack = jnp.stack([scale_t1 * _SIGMA_MINUS.matrix, scale_tphi * Z.matrix], axis=-3)
    return Lindbladian.from_operators(None, Operator.from_matrix(L_stack, ((2,), (2,))))


def bit_flip(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for the bit-flip channel.

    Jump operator: :math:`L = \\sqrt{\\gamma}\\,X`.

    :param gamma: Bit-flip rate. Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the bit-flip channel.
    """
    scale = jnp.sqrt(gamma)
    L = scale[..., None, None, None] * X.matrix
    return Lindbladian.from_operators(None, Operator.from_matrix(L, ((2,), (2,))))


def phase_flip(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for the phase-flip channel.

    Jump operator: :math:`L = \\sqrt{\\gamma}\\,Z`.

    :param gamma: Phase-flip rate. Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the phase-flip channel.
    """
    scale = jnp.sqrt(gamma)
    L = scale[..., None, None, None] * Z.matrix
    return Lindbladian.from_operators(None, Operator.from_matrix(L, ((2,), (2,))))


def leakage(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for leakage out of the computational subspace (qutrit).

    Models population loss from :math:`|1\\rangle` to the leakage state :math:`|2\\rangle`:

    Jump operator: :math:`L = \\sqrt{\\gamma}\\,|2\\rangle\\langle 1|`.

    :param gamma: Leakage rate. Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the leakage channel (qutrit space).
    """
    scale = jnp.sqrt(gamma)
    L = scale[..., None, None, None] * _SIGMA_12.matrix
    return Lindbladian.from_operators(None, Operator.from_matrix(L, ((3,), (3,))))


def seepage(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for seepage back into the computational subspace (qutrit).

    Models population return from the leakage state :math:`|2\\rangle` to :math:`|1\\rangle`:

    Jump operator: :math:`L = \\sqrt{\\gamma}\\,|1\\rangle\\langle 2|`.

    :param gamma: Seepage rate. Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the seepage channel (qutrit space).
    """
    scale = jnp.sqrt(gamma)
    L = scale[..., None, None, None] * _SIGMA_21.matrix
    return Lindbladian.from_operators(None, Operator.from_matrix(L, ((3,), (3,))))
