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

r"""Standard gate set, as detailed in Quil whitepaper (arXiV:1608:03355v2).

Currently includes:
    I - identity :math:`\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}`

    X - Pauli-X :math:`\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}`

    Y - Pauli-Y :math:`\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}`

    Z - Pauli-Z :math:`\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}`

    H - Hadamard
    :math:`\frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}`

    S - PHASE(pi/2)
    :math:`\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}`

    T - PHASE(pi/4)
    :math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{pmatrix}`

    PHASE(:math:`\phi`) - PHASE
    :math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{i \phi} \end{pmatrix}`

    RX(:math:`\phi`) - RX
    :math:`\begin{pmatrix} \cos(\phi / 2) & -i \sin(\phi/2) \\ -i \sin(\phi/2) & \cos(\phi/2) \end{pmatrix}`

    RY(:math:`\phi`) - RY
    :math:`\begin{pmatrix} \cos(\phi / 2) & -\sin(\phi / 2) \\ \sin(\phi/2) & \cos(\phi/2) \end{pmatrix}`

    RZ(:math:`\phi`) - RZ
    :math:`\begin{pmatrix} \cos(\phi/2) - i \sin(\phi/2) & 0 \\ 0 & \cos(\phi/2) + i \sin(\phi/2) \end{pmatrix}`

    PHASEDRX(:math:`\theta, \phi`) - PHASEDRX
    :math:`\begin{pmatrix} e^{i \theta / 2} \cos\left(\frac{\theta}{2}\right) & -i e^{i\left(\frac{\theta}{2} - \phi\right)} \sin\left(\frac{\theta}{2}\right) \\
-i e^{i\left(\frac{\theta}{2} + \phi\right)} \sin\left(\frac{\theta}{2}\right) & e^{i \theta / 2} \cos\left(\frac{\theta}{2}\right) \end{pmatrix}`

    U(:math:`\theta, \phi, \lambda`) - U3
    :math:`\begin{pmatrix} \cos(\theta/2) & - e^{i\lambda} \sin(\theta/2) \\ e^{i\phi} \sin(\theta/2) & e^{i(\phi + \lambda)} \cos(\theta/2) \end{pmatrix}`

    CZ - controlled-Z
    :math:`P_0 \otimes I + P_1 \otimes Z = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&1&0 \\ 0&0&0&-1 \end{pmatrix}`

    CNOT - controlled-X / controlled-NOT
    :math:`P_0 \otimes I + P_1 \otimes X = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&0&1 \\ 0&0&1&0 \end{pmatrix}`

    CCNOT - double-controlled-X
    :math:`P_0 \otimes P_0 \otimes I + P_0 \otimes P_1 \otimes I + P_1 \otimes P_0 \otimes I + P_1 \otimes P_1 \otimes X`

    CPHASE00(:math:`\phi`) - controlled-phase-on-\|00\>
    :math:`\text{diag}(e^{i \phi}, 1, 1, 1,)`

    CPHASE01(:math:`\phi`) - controlled-phase-on-\|01\>
    :math:`\text{diag}(1, e^{i \phi}, 1, 1,)`

    CPHASE10(:math:`\phi`) - controlled-phase-on-\|10\>
    :math:`\text{diag}(1, 1, e^{i \phi}, 1)`

    CPHASE(:math:`\phi`) - controlled-phase-on-\|11\>
    :math:`\text{diag}(1, 1, 1, e^{i \phi})`

    SWAP - swap
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&0&1&0 \\ 0&1&0&0 \\ 0&0&0&1 \end{pmatrix}`

    CSWAP - controlled-swap
    :math:`P_0 \otimes I_2 + P_1 \otimes \text{SWAP}`

    ISWAP - i-phase-swap
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&0&i&0 \\ 0&i&0&0 \\ 0&0&0&1 \end{pmatrix}`

    PSWAP(:math:`\phi`) - phi-phase-swap
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&0&e^{i\phi}&0 \\ 0&e^{i\phi}&0&0 \\ 0&0&0&1 \end{pmatrix}`

    XY(:math:`\phi`) - XY-interaction
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&\cos(\phi/2)&i\sin(\phi/2)&0 \\ 0&i\sin(\phi/2)&\cos(\phi/2)&0 \\  0&0&0&1 \end{pmatrix}`

    SQISW - XY(:math:`\pi/2`)-interaction
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&\frac{1}{\sqrt{2}}&\frac{i}{\sqrt{2}}&0 \\ 0&\frac{i}{\sqrt{2}}&\frac{1}{\sqrt{2}}&0 \\  0&0&0&1 \end{pmatrix}`

    FSIM(:math:`\theta, \phi`) - XX+YY interaction with conditional phase on \|11\>
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&\cos(\frac{\theta}{2})&i\sin(\frac{\theta}{2})&0 \\ 0&i\sin(\frac{\theta}{2})&\cos(\frac{\theta}{2})&0 \\  0&0&0&e^{i \phi} \end{pmatrix}`

    PHASEDFSIM(:math:`\theta, \zeta, \chi, \gamma, \phi`) - XX+YY interaction with conditional phase on \|11\>
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&\ e^{-i(\gamma+\zeta)}\cos(\frac{\theta}{2})&ie^{-i(\gamma-\chi)}\sin(\frac{\theta}{2})&0 \\ 0&ie^{-i(\gamma+\chi)}\sin(\frac{\theta}{2})&e^{-i(\gamma-\zeta)}\cos(\frac{\theta}{2})&0 \\  0&0&0&e^{ i\phi - 2i\gamma} \end{pmatrix}`

    RXX(:math:`\phi`) - XX-interaction
    :math:`\begin{pmatrix} \cos(\phi/2)&0&0&-i\sin(\phi/2) \\ 0&\cos(\phi/2)&-i\sin(\phi/2)&0 \\ 0&-i\sin(\phi/2)&\cos(\phi/2)&0 \\  -i\sin(\phi/2)&0&0&cos(\phi/2) \end{pmatrix}`

    RYY(:math:`\phi`) - YY-interaction
    :math:`\begin{pmatrix} \cos(\phi/2)&0&0&i\sin(\phi/2) \\ 0&\cos(\phi/2)&-i\sin(\phi/2)&0 \\ 0&-i\sin(\phi/2)&\cos(\phi/2)&0 \\  i\sin(\phi/2)&0&0&cos(\phi/2) \end{pmatrix}`

    RZZ(:math:`\phi`) - ZZ-interaction
    :math:`\text{diag}(e^{-i\phi/2}, e^{i\phi/2}, e^{i\phi/2}, e^{-i\phi/2})`


Each parametric gate's docstring gives its Hamiltonian generator :math:`H`, with the gate equal to
:math:`e^{-iH}` (the convention of :func:`quax.evolve`).

Specialized gates / internal utility gates:
    BARENCO(:math:`\alpha, \phi, \theta`) - Barenco gate
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&e^{i\phi} \cos\theta & -i e^{i(\alpha-\phi)} \sin\theta \\ 0&0&-i e^{i(\alpha+\phi)} \sin\theta & e^{i\alpha} \cos\theta \end{pmatrix}`

    P0 - project-onto-zero
    :math:`\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}`

    P1 - project-onto-one
    :math:`\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}`
"""

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Involution, Observable, Operator, QuantumInstrument, SuperOp, Unitary

# Parametric gates are written in closed form rather than as ``evolve(H)``: a matrix exponential is far slower than
# a handful of trigonometric entries, especially over an ensemble of angles.  Each docstring records the Hamiltonian
# generator ``H`` in the Schrödinger convention of :func:`quax.evolve`, i.e. the gate is ``exp(-i H)``.

_QUBIT_DIMS = ((2,), (2,))
_TWO_QUBIT_DIMS = ((2, 2), (2, 2))
_QUTRIT_DIMS = ((3,), (3,))


def _matrix_from_entries(entries: list[list]) -> Array:
    """Assemble a square matrix from nested rows of scalar or array entries.

    Entries broadcast against one another, so array-valued entries produce an ensemble of matrices with the
    broadcast shape as the leading ensemble dimensions.
    """
    size = len(entries)
    flat = jnp.broadcast_arrays(*(jnp.asarray(entry, dtype=complex) for row in entries for entry in row))
    return jnp.stack(flat, axis=-1).reshape(flat[0].shape + (size, size))


def _unitary_from_entries(entries: list[list], dims: tuple[tuple[int, ...], tuple[int, ...]]) -> Unitary:
    """A (possibly ensemble) unitary from nested rows of scalar or array entries."""
    return Unitary.from_matrix(_matrix_from_entries(entries), dims)


def _diagonal_unitary(diagonal: list, dims: tuple[tuple[int, ...], tuple[int, ...]]) -> Unitary:
    """A (possibly ensemble) diagonal unitary from a list of scalar or array entries."""
    flat = jnp.broadcast_arrays(*(jnp.asarray(entry, dtype=complex) for entry in diagonal))
    matrix = jnp.stack(flat, axis=-1)[..., :, None] * jnp.eye(len(diagonal), dtype=complex)
    return Unitary.from_matrix(matrix, dims)


def _rx_entries(phi) -> list[list]:
    """Entries of ``exp(-i φ X / 2)``."""
    cos, sin = jnp.cos(phi / 2.0), jnp.sin(phi / 2.0)
    return [[cos, -1j * sin], [-1j * sin, cos]]


def _ry_entries(phi) -> list[list]:
    """Entries of ``exp(-i φ Y / 2)``."""
    cos, sin = jnp.cos(phi / 2.0), jnp.sin(phi / 2.0)
    return [[cos, -sin], [sin, cos]]


def _rz_entries(phi) -> list[list]:
    """Entries of ``exp(-i φ Z / 2)``."""
    return [[jnp.exp(-0.5j * phi), 0.0], [0.0, jnp.exp(0.5j * phi)]]


I = Involution.from_matrix(jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex), ((2,), (2,)))

X = Involution.from_matrix(jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), ((2,), (2,)))

Y = Involution.from_matrix(jnp.array([[0.0, 0.0 - 1.0j], [0.0 + 1.0j, 0.0]], dtype=complex), ((2,), (2,)))

Z = Involution.from_matrix(jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex), ((2,), (2,)))

H = Involution.from_matrix((1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex), ((2,), (2,)))

S = Unitary.from_matrix(jnp.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex), ((2,), (2,)))

T = Unitary.from_matrix(jnp.array([[1.0, 0.0], [0.0, jnp.exp(1.0j * jnp.pi / 4.0)]], dtype=complex), ((2,), (2,)))


@jax.jit
def PHASE(phi: float) -> Unitary:
    r"""Phase shift of :math:`|1\rangle` by :math:`e^{i\phi}`.

    Generator: :math:`H = -\phi\,|1\rangle\langle 1| = -\frac{\phi}{2}(I - Z)`.
    """
    return _diagonal_unitary([1.0, jnp.exp(1j * phi)], _QUBIT_DIMS)


@jax.jit
def RX(phi: float) -> Unitary:
    r"""Rotation by :math:`\phi` about the X axis.

    Generator: :math:`H = \frac{\phi}{2} X`.
    """
    return _unitary_from_entries(_rx_entries(phi), _QUBIT_DIMS)


@jax.jit
def RY(phi: float) -> Unitary:
    r"""Rotation by :math:`\phi` about the Y axis.

    Generator: :math:`H = \frac{\phi}{2} Y`.
    """
    return _unitary_from_entries(_ry_entries(phi), _QUBIT_DIMS)


@jax.jit
def RZ(phi: float) -> Unitary:
    r"""Rotation by :math:`\phi` about the Z axis.

    Generator: :math:`H = \frac{\phi}{2} Z`.
    """
    return _unitary_from_entries(_rz_entries(phi), _QUBIT_DIMS)


@jax.jit
def PHASEDRX(theta: float, phi: float) -> Unitary:
    r"""Rotation by :math:`\theta` about the equatorial axis at azimuth :math:`\phi`, with a global phase
    :math:`e^{i\theta/2}`, i.e. :math:`e^{i\theta/2}\, RZ(\phi)\, RX(\theta)\, RZ(-\phi)`.

    Generator: :math:`H = \frac{\theta}{2}\left(\cos\phi\, X + \sin\phi\, Y - I\right)`.
    """
    phase = jnp.exp(0.5j * theta)
    cos, sin = jnp.cos(theta / 2.0), jnp.sin(theta / 2.0)
    return _unitary_from_entries(
        [
            [phase * cos, -1j * phase * jnp.exp(-1j * phi) * sin],
            [-1j * phase * jnp.exp(1j * phi) * sin, phase * cos],
        ],
        _QUBIT_DIMS,
    )


@jax.jit
def U(theta: float, phi: float, lam: float) -> Unitary:
    r"""General single-qubit gate :math:`e^{i(\phi + \lambda)/2}\, RZ(\phi)\, RY(\theta)\, RZ(\lambda)`.

    Generators: a product of three evolutions (applied right to left)

    .. math::
        U(\theta, \phi, \lambda) = e^{i(\phi + \lambda)/2}\, e^{-i \frac{\phi}{2} Z}\, e^{-i \frac{\theta}{2} Y}\,
        e^{-i \frac{\lambda}{2} Z}.
    """
    cos, sin = jnp.cos(theta / 2.0), jnp.sin(theta / 2.0)
    return _unitary_from_entries(
        [
            [cos, -jnp.exp(1j * lam) * sin],
            [jnp.exp(1j * phi) * sin, jnp.exp(1j * (phi + lam)) * cos],
        ],
        _QUBIT_DIMS,
    )


CZ = Involution.from_matrix(
    jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex), ((2, 2), (2, 2))
)

CNOT = Involution.from_matrix(
    jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex), ((2, 2), (2, 2))
)

CCNOT = Involution.from_matrix(
    jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
    ),
    ((2, 2, 2), (2, 2, 2)),
)


@jax.jit
def CPHASE00(phi: float) -> Unitary:
    r"""Phase shift of :math:`|00\rangle` by :math:`e^{i\phi}`.

    Generator: :math:`H = -\phi\,|00\rangle\langle 00| = -\frac{\phi}{4}(II + ZI + IZ + ZZ)`.
    """
    return _diagonal_unitary([jnp.exp(1j * phi), 1.0, 1.0, 1.0], _TWO_QUBIT_DIMS)


@jax.jit
def CPHASE01(phi: float) -> Unitary:
    r"""Phase shift of :math:`|01\rangle` by :math:`e^{i\phi}`.

    Generator: :math:`H = -\phi\,|01\rangle\langle 01| = -\frac{\phi}{4}(II + ZI - IZ - ZZ)`.
    """
    return _diagonal_unitary([1.0, jnp.exp(1j * phi), 1.0, 1.0], _TWO_QUBIT_DIMS)


@jax.jit
def CPHASE10(phi: float) -> Unitary:
    r"""Phase shift of :math:`|10\rangle` by :math:`e^{i\phi}`.

    Generator: :math:`H = -\phi\,|10\rangle\langle 10| = -\frac{\phi}{4}(II - ZI + IZ - ZZ)`.
    """
    return _diagonal_unitary([1.0, 1.0, jnp.exp(1j * phi), 1.0], _TWO_QUBIT_DIMS)


@jax.jit
def CPHASE(phi: float) -> Unitary:
    r"""Phase shift of :math:`|11\rangle` by :math:`e^{i\phi}`.

    Generator: :math:`H = -\phi\,|11\rangle\langle 11| = -\frac{\phi}{4}(II - ZI - IZ + ZZ)`.
    """
    return _diagonal_unitary([1.0, 1.0, 1.0, jnp.exp(1j * phi)], _TWO_QUBIT_DIMS)


SWAP = Involution.from_matrix(
    jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex), ((2, 2), (2, 2))
)

CSWAP = Involution.from_matrix(
    jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=complex,
    ),
    ((2, 2, 2), (2, 2, 2)),
)

ISWAP = Unitary.from_matrix(
    jnp.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex), ((2, 2), (2, 2))
)


@jax.jit
def PSWAP(phi: float) -> Unitary:
    r"""SWAP with a phase :math:`e^{i\phi}` on the swapped :math:`|01\rangle, |10\rangle` amplitudes.

    Generator: :math:`H = \left(\frac{\pi}{4} - \frac{\phi}{2}\right)(II - ZZ) - \frac{\pi}{4}(XX + YY)`
    (the two terms commute).
    """
    phase = jnp.exp(1j * phi)
    return _unitary_from_entries(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, phase, 0.0], [0.0, phase, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        _TWO_QUBIT_DIMS,
    )


def _xy_entries(theta, phase_11) -> list[list]:
    """Entries of ``exp(i θ (XX + YY) / 4)`` with ``phase_11`` on the ``|11>`` diagonal."""
    cos, sin = jnp.cos(theta / 2.0), jnp.sin(theta / 2.0)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos, 1j * sin, 0.0],
        [0.0, 1j * sin, cos, 0.0],
        [0.0, 0.0, 0.0, phase_11],
    ]


@jax.jit
def XY(phi: float) -> Unitary:
    r"""XY (iSWAP-family) interaction by angle :math:`\phi` in the :math:`|01\rangle, |10\rangle` subspace.

    Generator: :math:`H = -\frac{\phi}{4}(XX + YY)`.
    """
    return _unitary_from_entries(_xy_entries(phi, 1.0), _TWO_QUBIT_DIMS)


@jax.jit
def FSIM(theta: float, phi: float) -> Unitary:
    r"""XY interaction by :math:`\theta` followed by a conditional phase :math:`e^{i\phi}` on :math:`|11\rangle`,
    i.e. :math:`CPHASE(\phi)\, XY(\theta)`.

    Generator: :math:`H = -\frac{\theta}{4}(XX + YY) - \phi\,|11\rangle\langle 11|` (the two terms commute).
    """
    return _unitary_from_entries(_xy_entries(theta, jnp.exp(1j * phi)), _TWO_QUBIT_DIMS)


@jax.jit
def PHASEDFSIM(theta: float, zeta: float, chi: float, gamma: float, phi: float) -> Unitary:
    r"""FSIM with additional single-qubit phases :math:`\zeta, \chi, \gamma`.

    Generators: a product of evolutions (applied right to left)
    :math:`e^{-i H_\text{diag}}\, e^{-i H_\zeta}\, e^{-i H_\text{int}}\, e^{-i H_\zeta}` with

    .. math::
        H_\text{diag} &= \left(\gamma - \frac{\phi}{4}\right) II - \frac{2\gamma - \phi}{4}(ZI + IZ)
            - \frac{\phi}{4} ZZ, \\
        H_\zeta &= \frac{\zeta}{4}(ZI - IZ), \\
        H_\text{int} &= -\frac{\theta}{4}\left[\cos\chi\,(XX + YY) - \sin\chi\,(YX - XY)\right].
    """
    cos, sin = jnp.cos(theta / 2.0), jnp.sin(theta / 2.0)
    return _unitary_from_entries(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, jnp.exp(-1j * (gamma + zeta)) * cos, 1j * jnp.exp(-1j * (gamma - chi)) * sin, 0.0],
            [0.0, 1j * jnp.exp(-1j * (gamma + chi)) * sin, jnp.exp(-1j * (gamma - zeta)) * cos, 0.0],
            [0.0, 0.0, 0.0, jnp.exp(1j * (phi - 2.0 * gamma))],
        ],
        _TWO_QUBIT_DIMS,
    )


@jax.jit
def RZZ(phi: float) -> Unitary:
    r"""ZZ interaction by angle :math:`\phi`.

    Generator: :math:`H = \frac{\phi}{2} ZZ`.
    """
    even, odd = jnp.exp(-0.5j * phi), jnp.exp(0.5j * phi)
    return _diagonal_unitary([even, odd, odd, even], _TWO_QUBIT_DIMS)


@jax.jit
def RXX(phi: float) -> Unitary:
    r"""XX interaction by angle :math:`\phi`.

    Generator: :math:`H = \frac{\phi}{2} XX`.
    """
    cos, sin = jnp.cos(phi / 2.0), -1j * jnp.sin(phi / 2.0)
    return _unitary_from_entries(
        [[cos, 0.0, 0.0, sin], [0.0, cos, sin, 0.0], [0.0, sin, cos, 0.0], [sin, 0.0, 0.0, cos]],
        _TWO_QUBIT_DIMS,
    )


@jax.jit
def RYY(phi: float) -> Unitary:
    r"""YY interaction by angle :math:`\phi`.

    Generator: :math:`H = \frac{\phi}{2} YY`.
    """
    cos, sin = jnp.cos(phi / 2.0), -1j * jnp.sin(phi / 2.0)
    return _unitary_from_entries(
        [[cos, 0.0, 0.0, -sin], [0.0, cos, sin, 0.0], [0.0, sin, cos, 0.0], [-sin, 0.0, 0.0, cos]],
        _TWO_QUBIT_DIMS,
    )


SQISWAP = SQISW = Unitary.from_matrix(
    jnp.array(
        [
            [1, 0, 0, 0],
            [0, 1 / jnp.sqrt(2), 1j / jnp.sqrt(2), 0],
            [0, 1j / jnp.sqrt(2), 1 / jnp.sqrt(2), 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    ),
    ((2, 2), (2, 2)),
)

# Utility gates for internal QVM use
P0 = Operator.from_matrix(jnp.array([[1, 0], [0, 0]], dtype=complex), ((2,), (2,)))

P1 = Operator.from_matrix(jnp.array([[0, 0], [0, 1]], dtype=complex), ((2,), (2,)))


# Specialized useful gates; not officially in standard gate set
def BARENCO(alpha: float, phi: float, theta: float) -> Unitary:
    lower_unitary = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.exp(1j * phi) * jnp.cos(theta),
                    -1j * jnp.exp(1j * (alpha - phi)) * jnp.sin(theta),
                ],
                axis=-1,
            ),
            jnp.stack(
                [
                    -1j * jnp.exp(1j * (alpha + phi)) * jnp.sin(theta),
                    jnp.exp(1j * alpha) * jnp.cos(theta),
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    ensemble_shape = lower_unitary.shape[:-2]
    top_left = jnp.broadcast_to(jnp.eye(2, dtype=complex), ensemble_shape + (2, 2))
    zeros = jnp.zeros(ensemble_shape + (2, 2), dtype=complex)
    top_block = jnp.concatenate([top_left, zeros], axis=-1)
    bottom_block = jnp.concatenate([zeros, lower_unitary], axis=-1)
    return Unitary.from_matrix(
        jnp.concatenate([top_block, bottom_block], axis=-2),
        ((2, 2), (2, 2)),
    )


@jax.jit
def CAN(tx: float, ty: float, tz: float) -> Unitary:
    r"""Canonical gate.

    Generator: :math:`H = -\frac{1}{2}\left(t_x XX + t_y YY + t_z ZZ\right)` (the three terms commute).
    """
    even_phase, odd_phase = jnp.exp(0.5j * tz), jnp.exp(-0.5j * tz)
    even_cos, even_sin = even_phase * jnp.cos((tx - ty) / 2.0), 1j * even_phase * jnp.sin((tx - ty) / 2.0)
    odd_cos, odd_sin = odd_phase * jnp.cos((tx + ty) / 2.0), 1j * odd_phase * jnp.sin((tx + ty) / 2.0)
    return _unitary_from_entries(
        [
            [even_cos, 0.0, 0.0, even_sin],
            [0.0, odd_cos, odd_sin, 0.0],
            [0.0, odd_sin, odd_cos, 0.0],
            [even_sin, 0.0, 0.0, even_cos],
        ],
        _TWO_QUBIT_DIMS,
    )


B = CAN(jnp.pi / 2.0, jnp.pi / 4.0, 0.0)
r"""Berkeley gate :math:`CAN(\pi/2, \pi/4, 0)`."""


ECR = Unitary.from_matrix(
    (1.0 / jnp.sqrt(2.0)) * jnp.array([[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]], dtype=complex),
    ((2, 2), (2, 2)),
)
r"""Echoed cross-resonance gate :math:`e^{-iH}\, (X \otimes I)` with generator :math:`H = -\frac{\pi}{4} ZX`."""


@jax.jit
def GIVENS(theta: float) -> Unitary:
    r"""Givens rotation by :math:`\theta` on the subspace spanned by :math:`|01\rangle` and :math:`|10\rangle`.

    Generator: :math:`H = \frac{\theta}{2}(YX - XY)`.
    """
    cos, sin = jnp.cos(theta), jnp.sin(theta)
    return _unitary_from_entries(
        [[1.0, 0.0, 0.0, 0.0], [0.0, cos, -sin, 0.0], [0.0, sin, cos, 0.0], [0.0, 0.0, 0.0, 1.0]],
        _TWO_QUBIT_DIMS,
    )


SYCAMORE = FSIM(-jnp.pi, -jnp.pi / 6.0)
r"""Sycamore gate :math:`FSIM(-\pi, -\pi/6)`, with generator
:math:`H = \frac{\pi}{4}(XX + YY) + \frac{\pi}{6}\,|11\rangle\langle 11|`."""


# =============================================================================
# Qutrit gates
# =============================================================================

# References:
# .. [TERN] Elementary gates for ternary quantum logic circuit.
#         Di, Y.-M. & Wei, H.-R.
#         https://doi.org/10.48550/arXiv.1105.5485
# .. [ASYQT] Asymptotic Improvements to Quantum Circuits via Qutrits.
#         Gokhale, P et al.
#         https://arxiv.org/abs/1905.10481


TX = Unitary.from_matrix(
    jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex),
    ((3,), (3,)),
)
"""Generalized qutrit X gate (cyclic shift) :cite:`ASYQT`."""

TY = Unitary.from_matrix(
    jnp.array([[0, -1j, 0], [0, 0, -1j], [1j, 0, 0]], dtype=complex),
    ((3,), (3,)),
)
"""Generalized qutrit Y gate :cite:`ASYQT`."""

TZ = Unitary.from_matrix(
    jnp.array(
        [[1, 0, 0], [0, jnp.exp(2j * jnp.pi / 3), 0], [0, 0, jnp.exp(4j * jnp.pi / 3)]],
        dtype=complex,
    ),
    ((3,), (3,)),
)
"""Generalized qutrit Z gate (clock matrix) :cite:`ASYQT`."""

TH = Unitary.from_matrix(
    (1.0 / jnp.sqrt(3.0))
    * jnp.array(
        [
            [1, 1, 1],
            [1, jnp.exp(2j * jnp.pi / 3), jnp.exp(4j * jnp.pi / 3)],
            [1, jnp.exp(4j * jnp.pi / 3), jnp.exp(2j * jnp.pi / 3)],
        ],
        dtype=complex,
    ),
    ((3,), (3,)),
)
"""Qutrit Hadamard (QFT on 3 levels)."""


TSHIFT = TX
"""Qutrit shift gate (alias for TX)."""

TSWAP = Involution.from_matrix(
    jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=complex,
    ),
    ((3, 3), (3, 3)),
)
"""Qutrit SWAP gate."""

# Qutrit level projectors
TP0 = Observable.from_matrix(jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex), ((3,), (3,)))
"""Qutrit projector onto |0⟩."""

TP1 = Observable.from_matrix(jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex), ((3,), (3,)))
"""Qutrit projector onto |1⟩."""

TP2 = Observable.from_matrix(jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=complex), ((3,), (3,)))
"""Qutrit projector onto |2⟩."""

# =============================================================================
# Gell-Mann observables (qutrit analogue of Pauli matrices)
# =============================================================================
# The eight Gell-Mann matrices λ_1 … λ_8 form a basis for traceless
# Hermitian 3×3 matrices (the Lie algebra su(3)).  Together with the
# identity they span all 3×3 Hermitian matrices.
# They satisfy Tr(λ_i λ_j) = 2 δ_{ij}.
# However, they are not unitary, as they must span a 8-dimension Lie algebra and
# cannot square to the identity
GELLMANN1 = Observable.from_matrix(jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex), ((3,), (3,)))
GELLMANN2 = Observable.from_matrix(jnp.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex), ((3,), (3,)))
GELLMANN3 = Observable.from_matrix(jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex), ((3,), (3,)))
GELLMANN4 = Observable.from_matrix(jnp.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex), ((3,), (3,)))
GELLMANN5 = Observable.from_matrix(jnp.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex), ((3,), (3,)))
GELLMANN6 = Observable.from_matrix(jnp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex), ((3,), (3,)))
GELLMANN7 = Observable.from_matrix(jnp.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex), ((3,), (3,)))
GELLMANN8 = Observable.from_matrix(
    (1 / jnp.sqrt(3)) * jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex),
    ((3,), (3,)),
)

GELLMANN_MATRICES = Observable.from_matrix(
    jnp.stack(
        [
            GELLMANN1.matrix,
            GELLMANN2.matrix,
            GELLMANN3.matrix,
            GELLMANN4.matrix,
            GELLMANN5.matrix,
            GELLMANN6.matrix,
            GELLMANN7.matrix,
            GELLMANN8.matrix,
        ]
    ),
    ((3,), (3,)),
)

# =============================================================================
# Qutrit Weyl operators W_{x,z} = X^x Z^z  (indexed by shift x, clock z)
# =============================================================================
# The nine operators form a complete orthonormal basis for 3×3 matrices:
#   Tr(W_{x,z}^† W_{x',z'}) = 3 δ_{xx'} δ_{zz'}
# For d=3: ω = exp(2πi/3).  Each W_{x,z} is the matrix whose (i, (i−x) mod 3)
# entry equals ω^(z·(i−x) mod 3), with all other entries zero.
_omega3 = jnp.exp(2j * jnp.pi / 3)

W00 = Unitary.from_matrix(
    jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=complex),
    ((3,), (3,)),
)
"""Qutrit Weyl W_{0,0}: identity."""

W01 = Unitary.from_matrix(
    jnp.array([[1, 0, 0], [0, _omega3, 0], [0, 0, _omega3**2]], dtype=complex),
    ((3,), (3,)),
)
"""Qutrit Weyl W_{0,1}: clock (Z) operator."""

W02 = Unitary.from_matrix(
    jnp.array([[1, 0, 0], [0, _omega3**2, 0], [0, 0, _omega3]], dtype=complex),
    ((3,), (3,)),
)
"""Qutrit Weyl W_{0,2}: clock-squared (Z²) operator."""

W10 = Unitary.from_matrix(
    jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex),
    ((3,), (3,)),
)
"""Qutrit Weyl W_{1,0}: shift (X) operator."""

W11 = Unitary.from_matrix(
    jnp.array([[0, 0, _omega3**2], [1, 0, 0], [0, _omega3, 0]], dtype=complex),
    ((3,), (3,)),
)
"""Qutrit Weyl W_{1,1}: X·Z."""

W12 = Unitary.from_matrix(
    jnp.array([[0, 0, _omega3], [1, 0, 0], [0, _omega3**2, 0]], dtype=complex),
    ((3,), (3,)),
)
"""Qutrit Weyl W_{1,2}: X·Z²."""

W20 = Unitary.from_matrix(
    jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex),
    ((3,), (3,)),
)
"""Qutrit Weyl W_{2,0}: shift-squared (X²) operator."""

W21 = Unitary.from_matrix(
    jnp.array([[0, _omega3, 0], [0, 0, _omega3**2], [1, 0, 0]], dtype=complex),
    ((3,), (3,)),
)
"""Qutrit Weyl W_{2,1}: X²·Z."""

W22 = Unitary.from_matrix(
    jnp.array([[0, _omega3**2, 0], [0, 0, _omega3], [1, 0, 0]], dtype=complex),
    ((3,), (3,)),
)
"""Qutrit Weyl W_{2,2}: X²·Z²."""

WEYLS3 = Unitary.from_matrix(
    jnp.stack(
        [W00.matrix, W01.matrix, W02.matrix, W10.matrix, W11.matrix, W12.matrix, W20.matrix, W21.matrix, W22.matrix]
    ),
    ((3,), (3,)),
)
"""All nine qutrit Weyl operators as an ensemble, ordered (0,0), (0,1), …, (2,2)."""

# =============================================================================
# Qutrit rotation gates (qubit rotations embedded in a two-level subspace)
# =============================================================================
# The Gell-Mann matrices directly provide X and Y generators for each 2-level
# subspace.  The Z generators for (0,2) and (1,2) are linear combinations of
# λ₃ and λ₈:
#   Z₀₂ = diag(1,0,−1) = ½λ₃ + (√3/2)λ₈
#   Z₁₂ = diag(0,1,−1) = (√3/2)λ₈ − ½λ₃


def _qutrit_subspace_unitary(block: list[list], levels: tuple[int, int]) -> Unitary:
    """Embed 2×2 entries on the qutrit ``levels``, acting as the identity on the remaining level."""
    entries: list[list] = [[1.0 if row == col else 0.0 for col in range(3)] for row in range(3)]
    for block_row, row in enumerate(levels):
        for block_col, col in enumerate(levels):
            entries[row][col] = block[block_row][block_col]
    return _unitary_from_entries(entries, _QUTRIT_DIMS)


@jax.jit
def TRX01(phi: float) -> Unitary:
    r"""Qutrit X-rotation in the |0⟩–|1⟩ subspace.

    Generator: :math:`H = \frac{\phi}{2} \lambda_1`.
    """
    return _qutrit_subspace_unitary(_rx_entries(phi), (0, 1))


@jax.jit
def TRY01(phi: float) -> Unitary:
    r"""Qutrit Y-rotation in the |0⟩–|1⟩ subspace.

    Generator: :math:`H = \frac{\phi}{2} \lambda_2`.
    """
    return _qutrit_subspace_unitary(_ry_entries(phi), (0, 1))


@jax.jit
def TRZ01(phi: float) -> Unitary:
    r"""Qutrit Z-rotation in the |0⟩–|1⟩ subspace.

    Generator: :math:`H = \frac{\phi}{2} \lambda_3 = \frac{\phi}{2} \operatorname{diag}(1, -1, 0)`.
    """
    return _qutrit_subspace_unitary(_rz_entries(phi), (0, 1))


@jax.jit
def TRX02(phi: float) -> Unitary:
    r"""Qutrit X-rotation in the |0⟩–|2⟩ subspace.

    Generator: :math:`H = \frac{\phi}{2} \lambda_4`.
    """
    return _qutrit_subspace_unitary(_rx_entries(phi), (0, 2))


@jax.jit
def TRY02(phi: float) -> Unitary:
    r"""Qutrit Y-rotation in the |0⟩–|2⟩ subspace.

    Generator: :math:`H = \frac{\phi}{2} \lambda_5`.
    """
    return _qutrit_subspace_unitary(_ry_entries(phi), (0, 2))


@jax.jit
def TRZ02(phi: float) -> Unitary:
    r"""Qutrit Z-rotation in the |0⟩–|2⟩ subspace.

    Generator: :math:`H = \frac{\phi}{2} \operatorname{diag}(1, 0, -1)
    = \frac{\phi}{2}\left(\frac{1}{2}\lambda_3 + \frac{\sqrt{3}}{2}\lambda_8\right)`.
    """
    return _qutrit_subspace_unitary(_rz_entries(phi), (0, 2))


@jax.jit
def TRX12(phi: float) -> Unitary:
    r"""Qutrit X-rotation in the |1⟩–|2⟩ subspace.

    Generator: :math:`H = \frac{\phi}{2} \lambda_6`.
    """
    return _qutrit_subspace_unitary(_rx_entries(phi), (1, 2))


@jax.jit
def TRY12(phi: float) -> Unitary:
    r"""Qutrit Y-rotation in the |1⟩–|2⟩ subspace.

    Generator: :math:`H = \frac{\phi}{2} \lambda_7`.
    """
    return _qutrit_subspace_unitary(_ry_entries(phi), (1, 2))


@jax.jit
def TRZ12(phi: float) -> Unitary:
    r"""Qutrit Z-rotation in the |1⟩–|2⟩ subspace.

    Generator: :math:`H = \frac{\phi}{2} \operatorname{diag}(0, 1, -1)
    = \frac{\phi}{2}\left(\frac{\sqrt{3}}{2}\lambda_8 - \frac{1}{2}\lambda_3\right)`.
    """
    return _qutrit_subspace_unitary(_rz_entries(phi), (1, 2))


# Convenience aliases
RX12 = TRX12
RY12 = TRY12
RZ12 = TRZ12


# ---- Measurement instruments ----


def MEASURE(
    dim: int = 2,
    ensemble_size: int | None = None,
) -> QuantumInstrument:
    """
    Ideal projective measurement of a single qudit in the computational basis.

    :param dim: Dimension of the qudit, e.g. ``2`` for a qubit, ``3`` for a qutrit.
    :param ensemble_size: If provided, returns an ensemble of identical instruments.
    :return: A :class:`QuantumInstrument` with ``dim`` outcomes.
    """
    # Each outcome k has a single Kraus operator P_k = |k⟩⟨k|.
    # SuperOp_k = conj(P_k) ⊗ P_k, computed via S_k[ab,cd] = conj(P_k[a,c]) * P_k[b,d].
    eye = jnp.eye(dim, dtype=complex)  # rows are basis vectors e_k
    # P_k = |k⟩⟨k| as matrix: P_k[a,c] = e_k[a] * e_k[c]
    # S_k[ab,cd] = conj(e_k[a]*e_k[c]) * e_k[b]*e_k[d] = e_k[a]*e_k[b]*e_k[c]*e_k[d]
    projectors = eye[:, :, None] * eye[:, None, :]  # (dim, dim, dim) = P_k[a,c]
    # S_k as rank-4: S_k[a,b,c,d] = conj(P_k[a,c]) * P_k[b,d]
    S4 = projectors[:, :, None, :, None] * projectors[:, None, :, None, :]  # (dim, dim, dim, dim, dim)
    matrices = S4.reshape(dim, dim * dim, dim * dim)  # (dim, dim², dim²)
    if ensemble_size is not None:
        matrices = jnp.broadcast_to(matrices, (ensemble_size,) + matrices.shape)
    return QuantumInstrument.from_matrix(matrices, ((dim,), (dim,)), (0,))


def RESET(dim: int = 2) -> SuperOp:
    """
    Ideal reset channel that maps every state to the ground state |0⟩⟨0|.

    The channel has Kraus operators :math:`K_k = |0\\rangle\\langle k|` for
    :math:`k = 0, \\ldots, d-1`, giving the superoperator
    :math:`S = \\sum_k \\bar{K}_k \\otimes K_k`.

    :param dim: Dimension of the qudit, e.g. ``2`` for a qubit, ``3`` for a qutrit.
    :return: A :class:`SuperOp` representing the ideal reset channel.
    """
    # K_k = |0⟩⟨k|  →  K_k[a,b] = delta_{a,0} * delta_{b,k}
    # S = sum_k conj(K_k) ⊗ K_k  via einsum
    eye = jnp.eye(dim, dtype=complex)
    kraus = eye[0:1, :, None] * eye[None, :, :]  # (dim, dim, dim): K_k[a,b] = e_0[a]*e_k[b]
    # kraus shape: (dim, dim, dim) but we need (dim, dim, dim) = (k, a, b)
    kraus = jnp.zeros((dim, dim, dim), dtype=complex).at[:, 0, :].set(eye)
    S = jnp.einsum("kab,kcd->acbd", jnp.conj(kraus), kraus).reshape(dim * dim, dim * dim)
    return SuperOp.from_matrix(S, ((dim,), (dim,)))


QUANTUM_GATES = {
    "RZ": RZ,
    "RX": RX,
    "RY": RY,
    "CZ": CZ,
    "XY": XY,
    "CPHASE": CPHASE,
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": H,
    "S": S,
    "T": T,
    "PHASE": PHASE,
    "CNOT": CNOT,
    "CCNOT": CCNOT,
    "CPHASE00": CPHASE00,
    "CPHASE01": CPHASE01,
    "CPHASE10": CPHASE10,
    "SWAP": SWAP,
    "CSWAP": CSWAP,
    "ISWAP": ISWAP,
    "PSWAP": PSWAP,
    "BARENCO": BARENCO,
    "FSIM": FSIM,
    "PHASEDFSIM": PHASEDFSIM,
    "RXX": RXX,
    "RYY": RYY,
    "RZZ": RZZ,
    "U": U,
    "PHASEDRX": PHASEDRX,
    "CAN": CAN,
    "B": B,
    "ECR": ECR,
    "GIVENS": GIVENS,
    "SYCAMORE": SYCAMORE,
    # Qutrit gates
    "TX": TX,
    "TY": TY,
    "TZ": TZ,
    "TH": TH,
    "TSHIFT": TSHIFT,
    "TSWAP": TSWAP,
    "TP0": TP0,
    "TP1": TP1,
    "TP2": TP2,
    "TRX01": TRX01,
    "TRY01": TRY01,
    "TRZ01": TRZ01,
    "TRX02": TRX02,
    "TRY02": TRY02,
    "TRZ02": TRZ02,
    "TRX12": TRX12,
    "TRY12": TRY12,
    "TRZ12": TRZ12,
    # Qutrit Weyl operators W_{x,z}
    "W00": W00,
    "W01": W01,
    "W02": W02,
    "W10": W10,
    "W11": W11,
    "W12": W12,
    "W20": W20,
    "W21": W21,
    "W22": W22,
    # MEASURE and RESET
    "MEASURE": MEASURE,
    "RESET": RESET,
}
