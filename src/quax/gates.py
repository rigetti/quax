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

Fixed gates (``X``, ``H``, ``CNOT``, the qutrit gates, …) are **built lazily on first access** at
the current JAX precision.  They remain plain attributes — ``qx.gates.X`` returns an operator, not a
function — but constructing them on demand means enabling x64 *after* importing ``quax`` yields
correctly-typed constants (see :mod:`quax._lazy`).  Parametric gates (``RX``, ``CAN``, …) are
ordinary functions and already build at call time.

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
    :math:`\begin{pmatrix} e^{i \theta / 2} \cos\left(\frac{\theta}{2}\right) & -i, e^{i\left(\frac{\theta}{2} - \phi\right)} \sin\left(\frac{\theta}{2}\right) \
-i, e^{i\left(\frac{\theta}{2} + \phi\right)} \sin\left(\frac{\theta}{2}\right) & e^{i \theta / 2} \cos\left(\frac{\theta}{2}\right) \end{pmatrix}`

    U(:math:`\theta, \phi, \lambda`) - U3
    :math:`\begin{pmatrix} \cos(\theta/2) & - \exp{i\lambda} \sin(\theta/2) \\ \exp{i\phi} \sin(\theta/2) & \exp{i\phi + \lambda} \cos(\theta/2) \end{pmatrix}`

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

    SQISW - XY(:math: `\pi/2`)-interaction
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&\frac{1}{\sqrt{2}}&\frac{i}{\sqrt{2}}&0 \\ \frac{i}{\sqrt{2}}&\frac{1}{\sqrt{2}} \\  0&0&0&1 \end{pmatrix}`

    FSIM(:math:`\theta, \phi`) - XX+YY interaction with conditional phase on \|11\>
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&\cos(\frac{\theta}{2})&i\sin(\frac{\theta}{2})&0 \\ 0&i\sin(\frac{\theta}{2})&\cos(\frac{\theta}{2})&0 \\  0&0&0&e^{i \phi} \end{pmatrix}`

    PHASEDFSIM(:math:`\theta, \zeta, \chi, \gamma, \phi`) - XX+YY interaction with conditional phase on \|11\>
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&\ e^{-i(\gamma+\zeta)}\cos(\frac{\theta}{2})&ie^{-i(\gamma-\chi)}\sin(\frac{\theta}{2})&0 \\ 0&ie^{-i(\gamma+\chi)}\sin(\frac{\theta}{2})&e^{-i(\gamma-\zeta)}\cos(\frac{\theta}{2})&0 \\  0&0&0&e^{ i\phi - 2i\gamma} \end{pmatrix}`

    RXX(:math:`\phi`) - XX-interaction
    :math:`\begin{pmatrix} \cos(\phi/2)&0&0&-i\sin(\phi/2) \\ 0&\cos(\phi/2)&-i\sin(\phi/2)&0 \\ 0&-i\sin(\phi/2)&\cos(\phi/2)&0 \\  -i\sin(\phi/2)&0&0&cos(\phi/2) \end{pmatrix}`

    RYY(:math:`\phi`) - YY-interaction
    :math:`\begin{pmatrix} \cos(\phi/2)&0&0&i\sin(\phi/2) \\ 0&\cos(\phi/2)&-i\sin(\phi/2)&0 \\ 0&-i\sin(\phi/2)&\cos(\phi/2)&0 \\  i\sin(\phi/2)&0&0&cos(\phi/2) \end{pmatrix}`

    RZZ(:math:`\phi`) - ZZ-interaction
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&\cos(\phi/2)&-i\sin(\phi/2)&0 \\ 0&-i\sin(\phi/2)&\cos(\phi/2)&0 \\  0&0&0&1 \end{pmatrix}`


Specialized gates / internal utility gates:
    BARENCO(:math:`\alpha, \phi, \theta`) - Barenco gate
    :math:`\begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&e^{i\phi} \cos\theta & -i e^{i(\alpha-\phi)} \sin\theta \\ 0&0&-i e^{i(\alpha+\phi)} \sin\theta & e^{i\alpha} \cos\theta \end{pmatrix}`

    P0 - project-onto-zero
    :math:`\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}`

    P1 - project-onto-one
    :math:`\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}`
"""  # noqa: E501

from typing import Any, Callable

import jax.numpy as jnp

from ._exponentiation import evolve
from ._lazy import make_lazy_getter
from ._quantum_objects import Involution, Observable, Operator, QuantumInstrument, SuperOp, Unitary

# ``_get(name)`` returns the current-precision instance of a lazily-built constant.  It is bound at
# the bottom of this module; builders and parametric gates below reference it at call time.
_get: Callable[[str], Any]


# =============================================================================
# Single-qubit constants
# =============================================================================


def _make_I() -> Involution:
    return Involution.from_matrix(jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex), ((2,), (2,)))


def _make_X() -> Involution:
    return Involution.from_matrix(jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), ((2,), (2,)))


def _make_Y() -> Involution:
    return Involution.from_matrix(jnp.array([[0.0, 0.0 - 1.0j], [0.0 + 1.0j, 0.0]], dtype=complex), ((2,), (2,)))


def _make_Z() -> Involution:
    return Involution.from_matrix(jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex), ((2,), (2,)))


def _make_H() -> Involution:
    return Involution.from_matrix(
        (1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex), ((2,), (2,))
    )


def _make_S() -> Unitary:
    return Unitary.from_matrix(jnp.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex), ((2,), (2,)))


def _make_T() -> Unitary:
    return Unitary.from_matrix(
        jnp.array([[1.0, 0.0], [0.0, jnp.exp(1.0j * jnp.pi / 4.0)]], dtype=complex), ((2,), (2,))
    )


def PHASE(phi: float) -> Unitary:
    I, Z = _get("I"), _get("Z")  # noqa: E741
    return evolve((I - Z) * (-0.5 * phi))


def RX(phi) -> Unitary:
    return evolve(_get("X") * (0.5 * phi))


def RY(phi: float) -> Unitary:
    return evolve(_get("Y") * (0.5 * phi))


def RZ(phi: float) -> Unitary:
    result = PHASE(phi) * jnp.exp(-1j * phi / 2.0)
    return Unitary(result.data, result.num_qubits)


def PHASEDRX(theta: float, phi: float) -> Unitary:
    result = jnp.exp(1j * theta / 2.0) * (RZ(phi) @ RX(theta) @ RZ(-phi))
    return Unitary(result.data, result.num_qubits)


def U(theta: float, phi: float, lam: float) -> Unitary:
    result = jnp.exp(1j * (phi + lam) / 2.0) * (RZ(phi) @ RY(theta) @ RZ(lam))
    return Unitary(result.data, result.num_qubits)


# =============================================================================
# Two-qubit constants
# =============================================================================


def _make_CZ() -> Involution:
    return Involution.from_matrix(
        jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex), ((2, 2), (2, 2))
    )


def _make_CNOT() -> Involution:
    return Involution.from_matrix(
        jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex), ((2, 2), (2, 2))
    )


def _make_CCNOT() -> Involution:
    return Involution.from_matrix(
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


def CPHASE00(phi: float) -> Unitary:
    I, Z = _get("I"), _get("Z")  # noqa: E741
    result = jnp.exp(0.5j * phi) * evolve((((I | I) - (Z | I) - (I | Z)) - (Z | Z)) * (0.25 * phi))
    return Unitary(result.data, result.num_qubits)


def CPHASE01(phi: float) -> Unitary:
    I, Z = _get("I"), _get("Z")  # noqa: E741
    return evolve((((I | I) + (Z | I) - (I | Z)) - (Z | Z)) * (-0.25 * phi))


def CPHASE10(phi: float) -> Unitary:
    I, Z = _get("I"), _get("Z")  # noqa: E741
    return evolve((((I | I) - (Z | I) + (I | Z)) - (Z | Z)) * (-0.25 * phi))


def CPHASE(phi: float) -> Unitary:
    I, Z = _get("I"), _get("Z")  # noqa: E741
    return evolve((((I | I) - (Z | I) - (I | Z)) + (Z | Z)) * (-0.25 * phi))


def _make_SWAP() -> Involution:
    return Involution.from_matrix(
        jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex), ((2, 2), (2, 2))
    )


def _make_CSWAP() -> Involution:
    return Involution.from_matrix(
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


def _make_ISWAP() -> Unitary:
    return Unitary.from_matrix(
        jnp.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex), ((2, 2), (2, 2))
    )


def PSWAP(phi: float) -> Unitary:
    I, X, Y, Z = _get("I"), _get("X"), _get("Y"), _get("Z")  # noqa: E741
    return evolve(((I | I) - (Z | Z)) * (-0.5 * (phi - jnp.pi / 2.0))) @ evolve(((X | X) + (Y | Y)) * (-jnp.pi / 4.0))


def XY(phi: float) -> Unitary:
    X, Y = _get("X"), _get("Y")
    return evolve(((X | X) + (Y | Y)) * (-phi / 4.0))


def FSIM(theta: float, phi: float) -> Unitary:
    return CPHASE(phi) @ XY(theta)


def PHASEDFSIM(theta: float, zeta: float, chi: float, gamma: float, phi: float) -> Unitary:
    I, X, Y, Z = _get("I"), _get("X"), _get("Y"), _get("Z")  # noqa: E741
    diagonal_generator = (
        (I | I) * ((phi / 4.0) - gamma) + ((Z | I) + (I | Z)) * ((2.0 * gamma - phi) / 4.0) + (Z | Z) * (phi / 4.0)
    )
    zeta_generator = ((Z | I) - (I | Z)) * (-zeta / 4.0)
    interaction_generator = (((X | X) + (Y | Y)) * (theta / 4.0) * jnp.cos(chi)) - (
        ((Y | X) - (X | Y)) * (theta / 4.0) * jnp.sin(chi)
    )

    return (
        evolve(-diagonal_generator) @ evolve(-zeta_generator) @ evolve(-interaction_generator) @ evolve(-zeta_generator)
    )


def RZZ(phi: float) -> Unitary:
    Z = _get("Z")
    return evolve((Z | Z) * (0.5 * phi))


def RXX(phi: float) -> Unitary:
    X = _get("X")
    return evolve((X | X) * (0.5 * phi))


def RYY(phi: float) -> Unitary:
    Y = _get("Y")
    return evolve((Y | Y) * (0.5 * phi))


def _make_SQISWAP() -> Unitary:
    return Unitary.from_matrix(
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
def _make_P0() -> Operator:
    return Operator.from_matrix(jnp.array([[1, 0], [0, 0]], dtype=complex), ((2,), (2,)))


def _make_P1() -> Operator:
    return Operator.from_matrix(jnp.array([[0, 0], [0, 1]], dtype=complex), ((2,), (2,)))


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


def CAN(tx: float, ty: float, tz: float) -> Unitary:
    """Canonical gate."""
    X, Y, Z = _get("X"), _get("Y"), _get("Z")
    return evolve((X | X) * (-0.5 * tx) + (Y | Y) * (-0.5 * ty) + (Z | Z) * (-0.5 * tz))


def _make_B() -> Unitary:
    return CAN(jnp.pi / 2.0, jnp.pi / 4.0, 0.0)


def _make_ECR() -> Unitary:
    I, X, Z = _get("I"), _get("X"), _get("Z")  # noqa: E741
    return evolve((-jnp.pi / 4.0) * (Z | X)) @ (X | I)


def GIVENS(theta: float) -> Unitary:
    """Givens rotation on the subspace spanned by |01> and |10>."""
    X, Y = _get("X"), _get("Y")
    return evolve(((Y | X) - (X | Y)) * (0.5 * theta))


def _make_SYCAMORE() -> Unitary:
    return CPHASE(-jnp.pi / 6.0) @ XY(-jnp.pi)


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


def _make_TX() -> Unitary:
    """Generalized qutrit X gate (cyclic shift) :cite:`ASYQT`."""
    return Unitary.from_matrix(
        jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex),
        ((3,), (3,)),
    )


def _make_TY() -> Unitary:
    """Generalized qutrit Y gate :cite:`ASYQT`."""
    return Unitary.from_matrix(
        jnp.array([[0, -1j, 0], [0, 0, -1j], [1j, 0, 0]], dtype=complex),
        ((3,), (3,)),
    )


def _make_TZ() -> Unitary:
    """Generalized qutrit Z gate (clock matrix) :cite:`ASYQT`."""
    return Unitary.from_matrix(
        jnp.array(
            [[1, 0, 0], [0, jnp.exp(2j * jnp.pi / 3), 0], [0, 0, jnp.exp(4j * jnp.pi / 3)]],
            dtype=complex,
        ),
        ((3,), (3,)),
    )


def _make_TH() -> Unitary:
    """Qutrit Hadamard (QFT on 3 levels)."""
    return Unitary.from_matrix(
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


def _make_TSWAP() -> Involution:
    """Qutrit SWAP gate."""
    return Involution.from_matrix(
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


# Qutrit level projectors
def _make_TP0() -> Observable:
    """Qutrit projector onto |0⟩."""
    return Observable.from_matrix(jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex), ((3,), (3,)))


def _make_TP1() -> Observable:
    """Qutrit projector onto |1⟩."""
    return Observable.from_matrix(jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex), ((3,), (3,)))


def _make_TP2() -> Observable:
    """Qutrit projector onto |2⟩."""
    return Observable.from_matrix(jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=complex), ((3,), (3,)))


# =============================================================================
# Gell-Mann observables (qutrit analogue of Pauli matrices)
# =============================================================================
# The eight Gell-Mann matrices λ_1 … λ_8 form a basis for traceless
# Hermitian 3×3 matrices (the Lie algebra su(3)).  Together with the
# identity they span all 3×3 Hermitian matrices.
# They satisfy Tr(λ_i λ_j) = 2 δ_{ij}.
# However, they are not unitary, as they must span a 8-dimension Lie algebra and
# cannot square to the identity
def _make_GELLMANN1() -> Observable:
    return Observable.from_matrix(jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex), ((3,), (3,)))


def _make_GELLMANN2() -> Observable:
    return Observable.from_matrix(jnp.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex), ((3,), (3,)))


def _make_GELLMANN3() -> Observable:
    return Observable.from_matrix(jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex), ((3,), (3,)))


def _make_GELLMANN4() -> Observable:
    return Observable.from_matrix(jnp.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex), ((3,), (3,)))


def _make_GELLMANN5() -> Observable:
    return Observable.from_matrix(jnp.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex), ((3,), (3,)))


def _make_GELLMANN6() -> Observable:
    return Observable.from_matrix(jnp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex), ((3,), (3,)))


def _make_GELLMANN7() -> Observable:
    return Observable.from_matrix(jnp.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex), ((3,), (3,)))


def _make_GELLMANN8() -> Observable:
    return Observable.from_matrix(
        (1 / jnp.sqrt(3)) * jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex),
        ((3,), (3,)),
    )


def _make_GELLMANN_MATRICES() -> Observable:
    return Observable.from_matrix(
        jnp.stack(
            [
                _get("GELLMANN1").matrix,
                _get("GELLMANN2").matrix,
                _get("GELLMANN3").matrix,
                _get("GELLMANN4").matrix,
                _get("GELLMANN5").matrix,
                _get("GELLMANN6").matrix,
                _get("GELLMANN7").matrix,
                _get("GELLMANN8").matrix,
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
def _make__omega3():
    return jnp.exp(2j * jnp.pi / 3)


def _make_W00() -> Unitary:
    """Qutrit Weyl W_{0,0}: identity."""
    return Unitary.from_matrix(
        jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=complex),
        ((3,), (3,)),
    )


def _make_W01() -> Unitary:
    """Qutrit Weyl W_{0,1}: clock (Z) operator."""
    w = _get("_omega3")
    return Unitary.from_matrix(
        jnp.array([[1, 0, 0], [0, w, 0], [0, 0, w**2]], dtype=complex),
        ((3,), (3,)),
    )


def _make_W02() -> Unitary:
    """Qutrit Weyl W_{0,2}: clock-squared (Z²) operator."""
    w = _get("_omega3")
    return Unitary.from_matrix(
        jnp.array([[1, 0, 0], [0, w**2, 0], [0, 0, w]], dtype=complex),
        ((3,), (3,)),
    )


def _make_W10() -> Unitary:
    """Qutrit Weyl W_{1,0}: shift (X) operator."""
    return Unitary.from_matrix(
        jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex),
        ((3,), (3,)),
    )


def _make_W11() -> Unitary:
    """Qutrit Weyl W_{1,1}: X·Z."""
    w = _get("_omega3")
    return Unitary.from_matrix(
        jnp.array([[0, 0, w**2], [1, 0, 0], [0, w, 0]], dtype=complex),
        ((3,), (3,)),
    )


def _make_W12() -> Unitary:
    """Qutrit Weyl W_{1,2}: X·Z²."""
    w = _get("_omega3")
    return Unitary.from_matrix(
        jnp.array([[0, 0, w], [1, 0, 0], [0, w**2, 0]], dtype=complex),
        ((3,), (3,)),
    )


def _make_W20() -> Unitary:
    """Qutrit Weyl W_{2,0}: shift-squared (X²) operator."""
    return Unitary.from_matrix(
        jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex),
        ((3,), (3,)),
    )


def _make_W21() -> Unitary:
    """Qutrit Weyl W_{2,1}: X²·Z."""
    w = _get("_omega3")
    return Unitary.from_matrix(
        jnp.array([[0, w, 0], [0, 0, w**2], [1, 0, 0]], dtype=complex),
        ((3,), (3,)),
    )


def _make_W22() -> Unitary:
    """Qutrit Weyl W_{2,2}: X²·Z²."""
    w = _get("_omega3")
    return Unitary.from_matrix(
        jnp.array([[0, w**2, 0], [0, 0, w], [1, 0, 0]], dtype=complex),
        ((3,), (3,)),
    )


def _make_WEYLS3() -> Unitary:
    """All nine qutrit Weyl operators as an ensemble, ordered (0,0), (0,1), …, (2,2)."""
    return Unitary.from_matrix(
        jnp.stack(
            [
                _get("W00").matrix,
                _get("W01").matrix,
                _get("W02").matrix,
                _get("W10").matrix,
                _get("W11").matrix,
                _get("W12").matrix,
                _get("W20").matrix,
                _get("W21").matrix,
                _get("W22").matrix,
            ]
        ),
        ((3,), (3,)),
    )


# =============================================================================
# Qutrit rotation gates (built from Gell-Mann generators)
# =============================================================================
# The Gell-Mann matrices directly provide X and Y generators for each 2-level
# subspace.  The Z generators for (0,2) and (1,2) are linear combinations of
# λ₃ and λ₈:
#   Z₀₂ = diag(1,0,−1) = ½λ₃ + (√3/2)λ₈
#   Z₁₂ = diag(0,1,−1) = (√3/2)λ₈ − ½λ₃
def _make__Z02() -> Observable:
    return _get("GELLMANN3") * 0.5 + _get("GELLMANN8") * (jnp.sqrt(3.0) * 0.5)


def _make__Z12() -> Observable:
    return _get("GELLMANN8") * (jnp.sqrt(3.0) * 0.5) - _get("GELLMANN3") * 0.5


def TRX01(phi) -> Unitary:
    """Qutrit X-rotation in the |0⟩–|1⟩ subspace."""
    return evolve(_get("GELLMANN1") * (0.5 * phi))


def TRY01(phi) -> Unitary:
    """Qutrit Y-rotation in the |0⟩–|1⟩ subspace."""
    return evolve(_get("GELLMANN2") * (0.5 * phi))


def TRZ01(phi) -> Unitary:
    """Qutrit Z-rotation in the |0⟩–|1⟩ subspace."""
    return evolve(_get("GELLMANN3") * (0.5 * phi))


def TRX02(phi) -> Unitary:
    """Qutrit X-rotation in the |0⟩–|2⟩ subspace."""
    return evolve(_get("GELLMANN4") * (0.5 * phi))


def TRY02(phi) -> Unitary:
    """Qutrit Y-rotation in the |0⟩–|2⟩ subspace."""
    return evolve(_get("GELLMANN5") * (0.5 * phi))


def TRZ02(phi) -> Unitary:
    """Qutrit Z-rotation in the |0⟩–|2⟩ subspace."""
    return evolve(_get("_Z02") * (0.5 * phi))


def TRX12(phi) -> Unitary:
    """Qutrit X-rotation in the |1⟩–|2⟩ subspace."""
    return evolve(_get("GELLMANN6") * (0.5 * phi))


def TRY12(phi) -> Unitary:
    """Qutrit Y-rotation in the |1⟩–|2⟩ subspace."""
    return evolve(_get("GELLMANN7") * (0.5 * phi))


def TRZ12(phi) -> Unitary:
    """Qutrit Z-rotation in the |1⟩–|2⟩ subspace."""
    return evolve(_get("_Z12") * (0.5 * phi))


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


def _make_QUANTUM_GATES() -> dict[str, Any]:
    """Registry of gates by name.

    Constant gates resolve to their current-precision instance; parametric gates map to the
    factory function.  Built lazily so the constant instances respect the active precision.
    """
    return {
        "RZ": RZ,
        "RX": RX,
        "RY": RY,
        "CZ": _get("CZ"),
        "XY": XY,
        "CPHASE": CPHASE,
        "I": _get("I"),
        "X": _get("X"),
        "Y": _get("Y"),
        "Z": _get("Z"),
        "H": _get("H"),
        "S": _get("S"),
        "T": _get("T"),
        "PHASE": PHASE,
        "CNOT": _get("CNOT"),
        "CCNOT": _get("CCNOT"),
        "CPHASE00": CPHASE00,
        "CPHASE01": CPHASE01,
        "CPHASE10": CPHASE10,
        "SWAP": _get("SWAP"),
        "CSWAP": _get("CSWAP"),
        "ISWAP": _get("ISWAP"),
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
        "B": _get("B"),
        "ECR": _get("ECR"),
        "GIVENS": GIVENS,
        "SYCAMORE": _get("SYCAMORE"),
        # Qutrit gates
        "TX": _get("TX"),
        "TY": _get("TY"),
        "TZ": _get("TZ"),
        "TH": _get("TH"),
        "TSHIFT": _get("TSHIFT"),
        "TSWAP": _get("TSWAP"),
        "TP0": _get("TP0"),
        "TP1": _get("TP1"),
        "TP2": _get("TP2"),
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
        "W00": _get("W00"),
        "W01": _get("W01"),
        "W02": _get("W02"),
        "W10": _get("W10"),
        "W11": _get("W11"),
        "W12": _get("W12"),
        "W20": _get("W20"),
        "W21": _get("W21"),
        "W22": _get("W22"),
        # MEASURE and RESET
        "MEASURE": MEASURE,
        "RESET": RESET,
    }


# Registry of lazily-built constant gates: name -> zero-argument builder.  Aliases (e.g. TSHIFT/TX)
# share a builder.  Accessing ``qx.gates.<NAME>`` builds the object at the current precision.
_BUILDERS: dict[str, Callable[[], Any]] = {
    "I": _make_I,
    "X": _make_X,
    "Y": _make_Y,
    "Z": _make_Z,
    "H": _make_H,
    "S": _make_S,
    "T": _make_T,
    "CZ": _make_CZ,
    "CNOT": _make_CNOT,
    "CCNOT": _make_CCNOT,
    "SWAP": _make_SWAP,
    "CSWAP": _make_CSWAP,
    "ISWAP": _make_ISWAP,
    "SQISWAP": _make_SQISWAP,
    "SQISW": _make_SQISWAP,
    "P0": _make_P0,
    "P1": _make_P1,
    "B": _make_B,
    "ECR": _make_ECR,
    "SYCAMORE": _make_SYCAMORE,
    # Qutrit gates
    "TX": _make_TX,
    "TY": _make_TY,
    "TZ": _make_TZ,
    "TH": _make_TH,
    "TSHIFT": _make_TX,  # alias for TX
    "TSWAP": _make_TSWAP,
    "TP0": _make_TP0,
    "TP1": _make_TP1,
    "TP2": _make_TP2,
    # Gell-Mann observables
    "GELLMANN1": _make_GELLMANN1,
    "GELLMANN2": _make_GELLMANN2,
    "GELLMANN3": _make_GELLMANN3,
    "GELLMANN4": _make_GELLMANN4,
    "GELLMANN5": _make_GELLMANN5,
    "GELLMANN6": _make_GELLMANN6,
    "GELLMANN7": _make_GELLMANN7,
    "GELLMANN8": _make_GELLMANN8,
    "GELLMANN_MATRICES": _make_GELLMANN_MATRICES,
    # Qutrit Weyl operators
    "_omega3": _make__omega3,
    "W00": _make_W00,
    "W01": _make_W01,
    "W02": _make_W02,
    "W10": _make_W10,
    "W11": _make_W11,
    "W12": _make_W12,
    "W20": _make_W20,
    "W21": _make_W21,
    "W22": _make_W22,
    "WEYLS3": _make_WEYLS3,
    # Qutrit Z generators for rotation gates
    "_Z02": _make__Z02,
    "_Z12": _make__Z12,
    # Gate registry
    "QUANTUM_GATES": _make_QUANTUM_GATES,
}

_get = make_lazy_getter(_BUILDERS)


def __getattr__(name: str) -> Any:
    if name in _BUILDERS:
        return _get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _BUILDERS.keys())
