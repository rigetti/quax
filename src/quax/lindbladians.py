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

from functools import reduce
from operator import mul
from typing import Tuple

import jax.numpy as jnp
from jax import Array

from ._operator_basis import n_qudit_herm_basis
from ._quantum_objects import Lindbladian, Operator
from .gates import GELLMANN6, GELLMANN7, X, Y, Z

# Qubit lowering/raising operators |0⟩⟨1| = (X + iY)/2 and |1⟩⟨0| = (X − iY)/2.
_SIGMA_MINUS = Operator.from_matrix((X.matrix + 1j * Y.matrix) / 2, ((2,), (2,)))
_SIGMA_PLUS = Operator.from_matrix((X.matrix - 1j * Y.matrix) / 2, ((2,), (2,)))

# Qutrit transition operators from the Gell-Mann generators:
#   |2⟩⟨1| = (λ₆ − iλ₇)/2  (leakage out of the computational subspace),
#   |1⟩⟨2| = (λ₆ + iλ₇)/2  (seepage back into it).
_SIGMA_12 = Operator.from_matrix((GELLMANN6.matrix - 1j * GELLMANN7.matrix) / 2, ((3,), (3,)))
_SIGMA_21 = Operator.from_matrix((GELLMANN6.matrix + 1j * GELLMANN7.matrix) / 2, ((3,), (3,)))


def amplitude_damping(gamma: float | Array, dims: Tuple[int, ...] = (2,)) -> Lindbladian:
    """Lindbladian generator for amplitude damping (T1 relaxation) of a qudit.

    Single jump operator :math:`L = \\sqrt{\\gamma}\\,a`, where ``a`` is the harmonic-oscillator
    annihilation operator on the ``d``-level system (``d = prod(dims)``), :math:`a|n\\rangle =
    \\sqrt{n}\\,|n-1\\rangle`.  In this harmonic approximation a single rate ``gamma`` sets the
    decay of every level: the relaxation rate out of level ``n`` is ``n·gamma``.

    For a qubit (``dims=(2,)``) this is :math:`L = \\sqrt{\\gamma}\\,|0\\rangle\\langle 1|` and
    ``evolve(L, t)`` is amplitude damping with probability ``p = 1 - exp(-gamma * t)``.

    :param gamma: Relaxation rate (1/T1) of the first excited level. Non-negative; arrays produce an ensemble.
    :param dims: Per-subsystem dimensions of the qudit (default a single qubit ``(2,)``).
    :return: Lindbladian generator for the amplitude damping channel.
    """
    d = reduce(mul, dims, 1)
    # Harmonic annihilation operator a[n-1, n] = sqrt(n); a|n> = sqrt(n)|n-1>.
    a = (
        jnp.zeros((d, d), dtype=complex)
        .at[jnp.arange(d - 1), jnp.arange(1, d)]
        .set(jnp.sqrt(jnp.arange(1, d, dtype=float)))
    )
    # scale[..., None, None, None] adds the (n_ops=1, d, d) axes so a scalar rate gives
    # shape (1, d, d) and a batched rate of shape (n,) gives (n, 1, d, d).
    scale = jnp.sqrt(gamma)
    L = scale[..., None, None, None] * a
    return Lindbladian(hamiltonian=None, jump_operators=Operator.from_matrix(L, (dims, dims)))


def dephasing(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for the dephasing channel.

    Jump operator: :math:`L = \\sqrt{\\gamma/2}\\,Z`.

    The resulting CPTP channel ``evolve(L, t)`` is dephasing with probability
    ``p = 1 - exp(-gamma * t)``.

    :param gamma: Dephasing rate. Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the dephasing channel.
    """
    scale = jnp.sqrt(gamma / 2.0)
    L = scale[..., None, None, None] * Z.matrix
    return Lindbladian(hamiltonian=None, jump_operators=Operator.from_matrix(L, ((2,), (2,))))


def depolarizing(gamma: float | Array, dims: Tuple[int, ...] = (2,)) -> Lindbladian:
    """Lindbladian generator for the (uniform, global) depolarizing channel on ``dims``.

    The jump operators are the ``D²−1`` traceless Hermitian basis operators of the ``D``-dimensional
    space (``D = prod(dims)``), each scaled by :math:`\\sqrt{\\gamma/(D^2-1)}`.  For a qubit
    (``dims=(2,)``) these are :math:`\\sqrt{\\gamma/3}\\,\\{X, Y, Z\\}` and ``evolve(L, t)`` is the
    depolarizing channel with probability ``p = 3/4 * (1 - exp(-4*gamma*t/3))``; higher-dimensional
    or multi-qudit ``dims`` give the analogous uniform depolarizer toward the maximally mixed state.

    :param gamma: Depolarizing rate. Must be non-negative. Arrays produce an ensemble.
    :param dims: Per-subsystem dimensions of the space to depolarize (default a single qubit).
    :return: Lindbladian generator for the depolarizing channel.
    """
    d = reduce(mul, dims, 1)
    # Traceless Hermitian basis (drop the leading identity element). Uniform norm ⇒ isotropic.
    traceless = n_qudit_herm_basis(dims).matrix[1:]  # (D²−1, D, D)
    scale = jnp.sqrt(gamma / (d * d - 1))[..., None, None, None]  # broadcast over (n_ops, D, D)
    L_stack = scale * traceless
    return Lindbladian(hamiltonian=None, jump_operators=Operator.from_matrix(L_stack, (dims, dims)))


def thermal_relaxation(t1: float | Array, tphi: float | Array, p1: float | Array = 0.0) -> Lindbladian:
    """Lindbladian generator for the finite-temperature thermal relaxation channel.

    Combines T1 relaxation (energy exchange with a bath at finite temperature) and pure dephasing
    (1/Tφ).  At equilibrium the qubit relaxes toward an excited-state population ``p1``:

    - downward (decay)      :math:`L_\\downarrow = \\sqrt{(1-p_1)/T_1}\\,|0\\rangle\\langle 1|`
    - upward (excitation)   :math:`L_\\uparrow = \\sqrt{p_1/T_1}\\,|1\\rangle\\langle 0|`
    - pure dephasing        :math:`L_\\varphi = \\sqrt{1/(2 T_\\varphi)}\\,Z`

    The total energy-relaxation rate is ``1/T1`` (``= γ↓ + γ↑``).  With the default ``p1 = 0``
    (zero temperature) the excitation jump vanishes and this reduces to pure amplitude damping.

    .. note::
        This takes the **pure-dephasing** time ``Tφ``, not the coherence time ``T₂`` usually
        reported for hardware.  They are related by ``1/T₂ = 1/(2·T₁) + 1/Tφ``, so convert with

        .. math:: T_\\varphi = \\frac{1}{\\,1/T_2 - 1/(2 T_1)\\,}.

        This requires ``T₂ ≤ 2·T₁`` (equivalently ``1/T₂ ≥ 1/(2 T₁)``); the pure-dephasing rate is
        otherwise negative and unphysical.

    :param t1: T1 relaxation time (total energy relaxation). Must be positive. Arrays produce an ensemble.
    :param tphi: Pure dephasing time (Tφ, not T2). Must be positive. Arrays produce an ensemble.
    :param p1: Equilibrium excited-state population ``∈ [0, 1]`` (finite temperature). Default ``0``.
    :return: Lindbladian generator for the thermal relaxation channel.
    """
    # Broadcast the (possibly mixed scalar/array) rate arguments to a common ensemble shape so the
    # three jump operators stack cleanly even when, e.g., t1 is batched but tphi is a scalar.
    t1, tphi, p1 = jnp.broadcast_arrays(
        jnp.asarray(t1, dtype=float), jnp.asarray(tphi, dtype=float), jnp.asarray(p1, dtype=float)
    )
    # Factor each rate as sqrt(population) / sqrt(time) rather than sqrt(population / time): the two
    # are identical in value but the factored form has a finite gradient w.r.t. t1 even at p1 = 0
    # (sqrt(p1 / t1) hits a 0/0 → NaN gradient there, whereas sqrt(p1) / sqrt(t1) → 0).
    inv_sqrt_t1 = (1.0 / jnp.sqrt(t1))[..., None, None]
    scale_down = jnp.sqrt(1.0 - p1)[..., None, None] * inv_sqrt_t1
    scale_up = jnp.sqrt(p1)[..., None, None] * inv_sqrt_t1
    scale_tphi = (jnp.sqrt(1.0 / tphi) / jnp.sqrt(2.0))[..., None, None]
    L_stack = jnp.stack(
        [
            scale_down * _SIGMA_MINUS.matrix,
            scale_up * _SIGMA_PLUS.matrix,
            scale_tphi * Z.matrix,
        ],
        axis=-3,
    )
    return Lindbladian(hamiltonian=None, jump_operators=Operator.from_matrix(L_stack, ((2,), (2,))))


def bit_flip(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for the bit-flip channel.

    Jump operator: :math:`L = \\sqrt{\\gamma}\\,X`.

    :param gamma: Bit-flip rate. Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the bit-flip channel.
    """
    scale = jnp.sqrt(gamma)
    L = scale[..., None, None, None] * X.matrix
    return Lindbladian(hamiltonian=None, jump_operators=Operator.from_matrix(L, ((2,), (2,))))


def phase_flip(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for the phase-flip channel.

    Jump operator: :math:`L = \\sqrt{\\gamma}\\,Z`.

    :param gamma: Phase-flip rate. Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the phase-flip channel.
    """
    scale = jnp.sqrt(gamma)
    L = scale[..., None, None, None] * Z.matrix
    return Lindbladian(hamiltonian=None, jump_operators=Operator.from_matrix(L, ((2,), (2,))))


def leakage(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for leakage out of the computational subspace (qutrit).

    Models population loss from :math:`|1\\rangle` to the leakage state :math:`|2\\rangle`:

    Jump operator: :math:`L = \\sqrt{\\gamma}\\,|2\\rangle\\langle 1|`.

    :param gamma: Leakage rate. Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the leakage channel (qutrit space).
    """
    scale = jnp.sqrt(gamma)
    L = scale[..., None, None, None] * _SIGMA_12.matrix
    return Lindbladian(hamiltonian=None, jump_operators=Operator.from_matrix(L, ((3,), (3,))))


def seepage(gamma: float | Array) -> Lindbladian:
    """Lindbladian generator for seepage back into the computational subspace (qutrit).

    Models population return from the leakage state :math:`|2\\rangle` to :math:`|1\\rangle`:

    Jump operator: :math:`L = \\sqrt{\\gamma}\\,|1\\rangle\\langle 2|`.

    :param gamma: Seepage rate. Must be non-negative. Arrays produce an ensemble.
    :return: Lindbladian generator for the seepage channel (qutrit space).
    """
    scale = jnp.sqrt(gamma)
    L = scale[..., None, None, None] * _SIGMA_21.matrix
    return Lindbladian(hamiltonian=None, jump_operators=Operator.from_matrix(L, ((3,), (3,))))
