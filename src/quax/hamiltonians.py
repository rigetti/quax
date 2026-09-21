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
r"""Hamiltonian generators of common gates, for building noisy gate models.

A gate model in which coherent errors (angle offsets) and incoherent ones (relaxation during the
gate) act concurrently needs the gate's *generator*: the Hamiltonian :math:`H` with
``evolve(H, 1.0) == gate``, so that ``Lindbladian(hamiltonian=H, jump_operators=...)`` evolved over one
gate is the noisy gate.  The generators here are dimensionless: the "time" is one gate.

Example::

    import quax as qx

    H = qx.hamiltonians.fsim(theta=0.0, phi=jnp.pi)                   # an ideal CZ
    noise = qx.lindbladians.conditional_relaxation(decay, dephasing)  # partner-dependent T1, Tphi
    gate = qx.evolve(qx.Lindbladian(hamiltonian=H, jump_operators=noise.jump_operators), 1.0)
"""

import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Observable
from .gates import P1, I, X, Y, Z

_XX_YY = ((X | X) + (Y | Y)).matrix
_P11 = (P1 | P1).matrix
_ZI = (Z | I).matrix
_IZ = (I | Z).matrix
_DIMS = ((2, 2), (2, 2))


def fsim(
    theta: float | Array, phi: float | Array, phi_0: float | Array = 0.0, phi_1: float | Array = 0.0
) -> Observable:
    r"""The generator of the fSim gate in PhasedFSim form, so that ``evolve(fsim(theta, phi), 1.0)`` is
    ``gates.FSIM(theta, phi)``.

    .. math:: H = -\frac{\theta}{4}(XX + YY) - \phi\,|11\rangle\langle 11|
                  + \frac{\phi_0}{2}\, Z \otimes I + \frac{\phi_1}{2}\, I \otimes Z

    An ideal ``CZ`` is ``theta=0, phi=pi``; ``theta`` is the exchange (iSWAP) angle, ``phi`` the
    controlled phase and ``phi_0, phi_1`` single-qubit phases acquired during the gate.  Equal
    single-qubit phases commute with the exchange term, and then ``evolve(fsim(theta, phi, -g, -g))``
    is ``gates.PHASEDFSIM(theta, 0, 0, g, phi)`` up to a global phase.

    The angles broadcast: arrays produce an ensemble of generators.  Jittable and differentiable.

    :param theta: The exchange angle.
    :param phi: The controlled phase.
    :param phi_0: The single-qubit phase on the first qubit.
    :param phi_1: The single-qubit phase on the second qubit.
    :return: The Hermitian generator as an ``Observable`` on two qubits.
    """
    theta, phi, phi_0, phi_1 = jnp.broadcast_arrays(
        jnp.asarray(theta, dtype=float),
        jnp.asarray(phi, dtype=float),
        jnp.asarray(phi_0, dtype=float),
        jnp.asarray(phi_1, dtype=float),
    )
    expand = lambda angle: angle[..., None, None]
    hamiltonian = -expand(theta / 4) * _XX_YY - expand(phi) * _P11 + expand(phi_0 / 2) * _ZI + expand(phi_1 / 2) * _IZ
    return Observable.from_matrix(hamiltonian, _DIMS)
