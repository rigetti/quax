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

import jax.numpy as jnp
from typing import cast

from ._quantum_objects import Involution, Observable, Operator, Unitary
from ._power import cis

I = Involution.from_matrix(jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex), ((2,), (2,)))  # noqa: E741

X = Involution.from_matrix(jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), ((2,), (2,)))

Y = Involution.from_matrix(jnp.array([[0.0, 0.0 - 1.0j], [0.0 + 1.0j, 0.0]], dtype=complex), ((2,), (2,)))

Z = Involution.from_matrix(jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex), ((2,), (2,)))

H = Involution.from_matrix((1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex), ((2,), (2,)))

S = Unitary.from_matrix(jnp.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex), ((2,), (2,)))

T = Unitary.from_matrix(jnp.array([[1.0, 0.0], [0.0, jnp.exp(1.0j * jnp.pi / 4.0)]], dtype=complex), ((2,), (2,)))


def PHASE(phi: float) -> Unitary:
    return cis((I - Z) * (0.5 * phi))


def RX(phi) -> Unitary:
    return cis(X * (-0.5 * phi))


def RY(phi: float) -> Unitary:
    return cis(Y * (-0.5 * phi))


def RZ(phi: float) -> Unitary:
    # Multiplication by a unit-modulus phase preserves unitarity; cast narrows static type.
    return cast(Unitary, PHASE(phi) * jnp.exp(-1j * phi / 2.0))


def PHASEDRX(theta: float, phi: float) -> Unitary:
    return jnp.exp(1j * theta / 2.0) * (RZ(phi) @ RX(theta) @ RZ(-phi))


def U(theta: float, phi: float, lam: float) -> Unitary:
    return jnp.exp(1j * (phi + lam) / 2.0) * (RZ(phi) @ RY(theta) @ RZ(lam))


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


def CPHASE00(phi: float) -> Unitary:
    # Multiplication by a unit-modulus phase preserves unitarity; cast narrows static type.
    return cast(Unitary, jnp.exp(1j * phi) * cis((((I | I) - (Z | I) - (I | Z)) - (Z | Z)) * (-0.25 * phi)))


def CPHASE01(phi: float) -> Unitary:
    return cis((((I | I) + (Z | I) - (I | Z)) - (Z | Z)) * (0.25 * phi))


def CPHASE10(phi: float) -> Unitary:
    return cis((((I | I) - (Z | I) + (I | Z)) - (Z | Z)) * (0.25 * phi))


def CPHASE(phi: float) -> Unitary:
    return cis((((I | I) - (Z | I) - (I | Z)) + (Z | Z)) * (0.25 * phi))


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


def PSWAP(phi: float) -> Unitary:
    return cis(((I | I) - (Z | Z)) * (0.5 * (phi - jnp.pi / 2.0))) @ cis(((X | X) + (Y | Y)) * (jnp.pi / 4.0))


def XY(phi: float) -> Unitary:
    return cis(((X | X) + (Y | Y)) * (phi / 4.0))


def FSIM(theta: float, phi: float) -> Unitary:
    return CPHASE(phi) @ XY(theta)


def PHASEDFSIM(theta: float, zeta: float, chi: float, gamma: float, phi: float) -> Unitary:
    diagonal_generator = (
        (I | I) * ((phi / 4.0) - gamma) + ((Z | I) + (I | Z)) * ((2.0 * gamma - phi) / 4.0) + (Z | Z) * (phi / 4.0)
    )
    zeta_generator = ((Z | I) - (I | Z)) * (-zeta / 4.0)
    interaction_generator = (((X | X) + (Y | Y)) * (theta / 4.0) * jnp.cos(chi)) - (
        ((Y | X) - (X | Y)) * (theta / 4.0) * jnp.sin(chi)
    )

    return cis(diagonal_generator) @ cis(zeta_generator) @ cis(interaction_generator) @ cis(zeta_generator)


def RZZ(phi: float) -> Unitary:
    # Multiplication by a unit-modulus phase preserves unitarity; cast narrows static type.
    return cast(Unitary, jnp.exp(-1j * phi / 2.0) * cis((Z | Z) * (-0.5 * phi)))


def RXX(phi: float) -> Unitary:
    return cis((X | X) * (-0.5 * phi))


def RYY(phi: float) -> Unitary:
    return cis((Y | Y) * (-0.5 * phi))


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


def CAN(tx: float, ty: float, tz: float) -> Unitary:
    """Canonical gate."""
    # Multiplication by a unit-modulus phase preserves unitarity; cast narrows static type.
    return cast(
        Unitary, jnp.exp(1j * tz / 2.0) * cis((X | X) * (0.5 * tx) + (Y | Y) * (0.5 * ty) + (Z | Z) * (0.5 * tz))
    )


B = CAN(jnp.pi / 2.0, jnp.pi / 4.0, 0.0)


ECR = cis((jnp.pi / 4.0) * (Z | X)) @ (X | I)


def GIVENS(theta: float) -> Unitary:
    """Givens rotation on the subspace spanned by |01> and |10>."""
    return cis(((Y | X) - (X | Y)) * (-0.5 * theta))


SYCAMORE = CPHASE(-jnp.pi / 6.0) @ XY(-jnp.pi)


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
"""Generalized qutrit X gate (cyclic shift) [ASYQT]_."""

TY = Unitary.from_matrix(
    jnp.array([[0, -1j, 0], [0, 0, -1j], [1j, 0, 0]], dtype=complex),
    ((3,), (3,)),
)
"""Generalized qutrit Y gate [ASYQT]_."""

TZ = Unitary.from_matrix(
    jnp.array(
        [[1, 0, 0], [0, jnp.exp(2j * jnp.pi / 3), 0], [0, 0, jnp.exp(4j * jnp.pi / 3)]],
        dtype=complex,
    ),
    ((3,), (3,)),
)
"""Generalized qutrit Z gate (clock matrix) [ASYQT]_."""

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
# Qutrit rotation gates (built from Gell-Mann generators)
# =============================================================================
# The Gell-Mann matrices directly provide X and Y generators for each 2-level
# subspace.  The Z generators for (0,2) and (1,2) are linear combinations of
# λ₃ and λ₈:
#   Z₀₂ = diag(1,0,−1) = ½λ₃ + (√3/2)λ₈
#   Z₁₂ = diag(0,1,−1) = (√3/2)λ₈ − ½λ₃
_Z02 = GELLMANN3 * 0.5 + GELLMANN8 * (jnp.sqrt(3.0) * 0.5)
_Z12 = GELLMANN8 * (jnp.sqrt(3.0) * 0.5) - GELLMANN3 * 0.5


def TRX01(phi) -> Unitary:
    """Qutrit X-rotation in the |0⟩–|1⟩ subspace."""
    return cis(GELLMANN1 * (-0.5 * phi))


def TRY01(phi) -> Unitary:
    """Qutrit Y-rotation in the |0⟩–|1⟩ subspace."""
    return cis(GELLMANN2 * (-0.5 * phi))


def TRZ01(phi) -> Unitary:
    """Qutrit Z-rotation in the |0⟩–|1⟩ subspace."""
    return cis(GELLMANN3 * (-0.5 * phi))


def TRX02(phi) -> Unitary:
    """Qutrit X-rotation in the |0⟩–|2⟩ subspace."""
    return cis(GELLMANN4 * (-0.5 * phi))


def TRY02(phi) -> Unitary:
    """Qutrit Y-rotation in the |0⟩–|2⟩ subspace."""
    return cis(GELLMANN5 * (-0.5 * phi))


def TRZ02(phi) -> Unitary:
    """Qutrit Z-rotation in the |0⟩–|2⟩ subspace."""
    return cis(_Z02 * (-0.5 * phi))


def TRX12(phi) -> Unitary:
    """Qutrit X-rotation in the |1⟩–|2⟩ subspace."""
    return cis(GELLMANN6 * (-0.5 * phi))


def TRY12(phi) -> Unitary:
    """Qutrit Y-rotation in the |1⟩–|2⟩ subspace."""
    return cis(GELLMANN7 * (-0.5 * phi))


def TRZ12(phi) -> Unitary:
    """Qutrit Z-rotation in the |1⟩–|2⟩ subspace."""
    return cis(_Z12 * (-0.5 * phi))


# Convenience aliases
RX12 = TRX12
RY12 = TRY12
RZ12 = TRZ12


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
}
