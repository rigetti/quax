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

from functools import cache, reduce
from itertools import permutations, product

import jax.numpy as jnp
from jax.numpy import arccos, pi, sqrt

from ._quantum_objects import Unitary
from .gates import RX, RZ, I, RY, X, Y, Z


SIC_PREP = Unitary.from_matrix(
    jnp.array(
        [
            I.matrix,  # SIC0
            (RX(-pi / 2) @ RZ(float(2 * arccos(1 / sqrt(3)) - pi)) @ RX(-pi / 2)).matrix,  # SIC1
            (
                RZ(float(-2 * pi / 3)) @ RX(-pi / 2) @ RZ(float(2 * arccos(1 / sqrt(3)) - pi)) @ RX(-pi / 2)
            ).matrix,  # SIC2
            (
                RZ(float(+2 * pi / 3)) @ RX(-pi / 2) @ RZ(float(2 * arccos(1 / sqrt(3)) - pi)) @ RX(-pi / 2)
            ).matrix,  # SIC3
        ]
    ),
    ((2,), (2,)),
    1,
)
"""The unitary operators that prepare the SIC states from |0> in the order SIC0, SIC1, SIC2, SIC3."""

SIC_STATE_PREP_OPERATORS = {
    "SIC0": SIC_PREP.matrix[0],
    "SIC1": SIC_PREP.matrix[1],
    "SIC2": SIC_PREP.matrix[2],
    "SIC3": SIC_PREP.matrix[3],
}
"""The unitary operators that prepare the SIC states from |0>."""

PAULI_PREP = Unitary.from_matrix(
    jnp.array(
        [
            RY(pi / 2).matrix,  # X+
            RY(-pi / 2).matrix,  # X-
            RX(-pi / 2).matrix,  # Y+
            RX(pi / 2).matrix,  # Y-
            I.matrix,  # Z+
            RX(pi).matrix,  # Z-
        ]
    ),
    ((2,), (2,)),
    1,
)
"""The unitary operators that prepare the Pauli states from |0> in the order X+, X-, Y+, Y-, Z+, Z-."""

PAULI_STATE_PREP_OPERATORS = {
    "X+": PAULI_PREP.matrix[0],
    "X-": PAULI_PREP.matrix[1],
    "Y+": PAULI_PREP.matrix[2],
    "Y-": PAULI_PREP.matrix[3],
    "Z+": PAULI_PREP.matrix[4],
    "Z-": PAULI_PREP.matrix[5],
}
"""The unitary operators that prepare the Pauli states from |0>."""


# Ensembles and Groups
PAULI_ENSEMBLE = PAULIS = Unitary.from_matrix(
    jnp.asarray([I.matrix, X.matrix, Y.matrix, Z.matrix], dtype=complex), ((2,), (2,)), 1
)
"""The ensemble of Pauli operators."""

TETRAHEDRAL_ENSEMBLE = Unitary.from_matrix(
    jnp.asarray(
        [
            I.matrix,
            X.matrix,
            (Z @ X).matrix,  # Y
            Z.matrix,
            (RX(+pi / 2) @ RZ(+pi / 2)).matrix,
            (RX(+pi / 2) @ RZ(-pi / 2)).matrix,
            (RX(-pi / 2) @ RZ(+pi / 2)).matrix,
            (RX(-pi / 2) @ RZ(-pi / 2)).matrix,
            (RZ(+pi / 2) @ RX(+pi / 2)).matrix,
            (RZ(+pi / 2) @ RX(-pi / 2)).matrix,
            (RZ(-pi / 2) @ RX(+pi / 2)).matrix,
            (RZ(-pi / 2) @ RX(-pi / 2)).matrix,
        ],
        dtype=complex,
    ),
    ((2,), (2,)),
    1,
)
"""The tetrahedral ensemble of operators."""

CLIFFORD_ENSEMBLE = OCTAHEDRAL_ENSEMBLE = Unitary.from_matrix(
    jnp.asarray(
        [
            # 0: Identity
            I.matrix,
            # 1..3: Paulis
            X.matrix,
            (Z @ X).matrix,  # Y
            Z.matrix,
            # sX, sZ
            RX(+pi / 2).matrix,
            RX(-pi / 2).matrix,
            RZ(+pi / 2).matrix,
            RZ(-pi / 2).matrix,
            # ZsX, XsZ
            (Z @ RX(+pi / 2)).matrix,
            (Z @ RX(-pi / 2)).matrix,
            (X @ RZ(+pi / 2)).matrix,
            (X @ RZ(-pi / 2)).matrix,
            # sZsX
            (RX(+pi / 2) @ RZ(+pi / 2)).matrix,
            (RX(+pi / 2) @ RZ(-pi / 2)).matrix,
            (RX(-pi / 2) @ RZ(+pi / 2)).matrix,
            (RX(-pi / 2) @ RZ(-pi / 2)).matrix,
            (RZ(+pi / 2) @ RX(+pi / 2)).matrix,
            (RZ(+pi / 2) @ RX(-pi / 2)).matrix,
            (RZ(-pi / 2) @ RX(+pi / 2)).matrix,
            (RZ(-pi / 2) @ RX(-pi / 2)).matrix,
            # sZsXsZ
            (RZ(+pi / 2) @ RX(+pi / 2) @ RZ(-pi / 2)).matrix,  # sY
            (RZ(-pi / 2) @ RX(+pi / 2) @ RZ(-pi / 2)).matrix,  # -H
            (RZ(+pi / 2) @ RX(-pi / 2) @ RZ(-pi / 2)).matrix,  # -sY
            (RZ(-pi / 2) @ RX(-pi / 2) @ RZ(-pi / 2)).matrix,  # H
        ],
        dtype=complex,
    ),
    ((2,), (2,)),
    1,
)
"""The ensemble of single-qubit Clifford operators."""


def _is_even_permutation(p: tuple) -> bool:
    """Checks if a permutation is even by counting cycles."""
    n = len(p)
    visited = [False] * n
    cycles = 0
    for i in range(n):
        if not visited[i]:
            cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = p[j]
    # The parity of a permutation is the parity of (n - number of cycles)
    return (n - cycles) % 2 == 0


@cache
def _generate_binary_icosahedral_group() -> Unitary:
    """
    Generate the binary isosahedral group of rotations.

    https://en.wikipedia.org/wiki/Binary_icosahedral_group
    """

    def quaternion_to_unitary(a, b, c, d):
        return jnp.array([[a - 1j * d, -c - 1j * b], [c - 1j * b, a + 1j * d]], dtype=complex)
        # return jnp.array([[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]], dtype=complex)

    phi = (1 + sqrt(5)) / 2
    generators = [
        (0, +1 / 2, +1 / (2 * phi), +phi / 2),
        (0, +1 / 2, +1 / (2 * phi), -phi / 2),
        (0, +1 / 2, -1 / (2 * phi), +phi / 2),
        (0, +1 / 2, -1 / (2 * phi), -phi / 2),
        (0, -1 / 2, +1 / (2 * phi), +phi / 2),
        (0, -1 / 2, +1 / (2 * phi), -phi / 2),
        (0, -1 / 2, -1 / (2 * phi), +phi / 2),
        (0, -1 / 2, -1 / (2 * phi), -phi / 2),
    ]

    even_permutations = [p for p in permutations(range(4)) if _is_even_permutation(p)]

    icosahedral_quaternions = jnp.array(
        [
            # 8 permutations of (+/-1, 0, 0, 0)
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, -1.0),
            # 16 permutations of (1/2, 1/2, 1/2, 1/2)
            (+0.5, +0.5, -0.5, +0.5),  # ++-+
            (+0.5, +0.5, +0.5, -0.5),  # +++-
            (+0.5, -0.5, +0.5, +0.5),  # +-++
            (+0.5, -0.5, -0.5, -0.5),  # +---
            (+0.5, +0.5, +0.5, +0.5),  # ++++
            (+0.5, -0.5, -0.5, +0.5),  # +--+
            (+0.5, +0.5, -0.5, -0.5),  # ++--
            (+0.5, -0.5, +0.5, -0.5),  # +-+-
            (-0.5, -0.5, +0.5, -0.5),  # --+-
            (-0.5, -0.5, -0.5, +0.5),  # ---+
            (-0.5, +0.5, -0.5, -0.5),  # -+--
            (-0.5, +0.5, +0.5, +0.5),  # -+++
            (-0.5, -0.5, -0.5, -0.5),  # ----
            (-0.5, +0.5, +0.5, -0.5),  # -++-
            (-0.5, -0.5, +0.5, +0.5),  # --++
            (-0.5, +0.5, -0.5, +0.5),  # -+-+
            # 96 permutations of (0, 1/2, 1/2 phi, phi/2)
        ]
        + [tuple(generator[pi] for pi in p) for generator in generators for p in even_permutations]
    )

    icosahedral_unitaries = jnp.array([quaternion_to_unitary(*q) for q in icosahedral_quaternions])
    return Unitary.from_matrix(icosahedral_unitaries, ((2,), (2,)), 1)


@cache
def _generate_icosahedral_rotation_group() -> Unitary:
    """
    Generate the binary isosahedral group of rotations.

    https://en.wikipedia.org/wiki/Binary_icosahedral_group
    """

    def quaternion_to_unitary(a, b, c, d):
        return jnp.array([[a - 1j * d, -c - 1j * b], [c - 1j * b, a + 1j * d]], dtype=complex)
        # return jnp.array([[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]], dtype=complex)

    phi = (1 + sqrt(5)) / 2
    generators = [
        (0, +1 / 2, +1 / (2 * phi), +phi / 2),  # equal to -, -, -
        (0, -1 / 2, +1 / (2 * phi), -phi / 2),  # equal to +, -, +
        (0, +1 / 2, +1 / (2 * phi), -phi / 2),  # equal to +, +, -
        (0, +1 / 2, -1 / (2 * phi), -phi / 2),  # equal to -, +, +
    ]

    even_permutations = [p for p in permutations(range(4)) if _is_even_permutation(p)]

    icosahedral_quaternions = jnp.array(
        [
            # 4 permutations of (+/-1, 0, 0, 0)
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            # 8 permutations of (1/2, 1/2, 1/2, 1/2)
            (+0.5, +0.5, -0.5, +0.5),  # ++-+
            (+0.5, +0.5, +0.5, -0.5),  # +++-
            (+0.5, -0.5, +0.5, +0.5),  # +-++
            (+0.5, -0.5, -0.5, -0.5),  # +---
            (+0.5, +0.5, +0.5, +0.5),  # ++++
            (+0.5, -0.5, -0.5, +0.5),  # +--+
            (+0.5, +0.5, -0.5, -0.5),  # ++--
            (+0.5, -0.5, +0.5, -0.5),  # +-+-
            # 48 permutations of (0, 1/2, 1/2 phi, phi/2)
        ]
        + [tuple(generator[pi] for pi in p) for generator in generators for p in even_permutations]
    )

    icosahedral_unitaries = jnp.array([quaternion_to_unitary(*q) for q in icosahedral_quaternions])
    return Unitary.from_matrix(icosahedral_unitaries, ((2,), (2,)), 1)


ICOSAHEDRAL_ENSEMBLE = ICOSAHEDRAL_GROUP = _generate_icosahedral_rotation_group()
"""The ensemble of single-qubit icosahedral rotation operators."""

BINARY_ICOSAHEDRAL_ENSEMBLE = BINARY_ICOSAHEDRAL_GROUP = _generate_binary_icosahedral_group()
"""The ensemble of single-qubit binary icosahedral operators."""


# All 24 single-qubit Clifford gates
CLIFFORDS_1Q = CLIFFORD_ENSEMBLE


def n_qubit_pauli_operators(n: int = 1) -> Unitary:
    """Generate all n-qubit Pauli operators.

    For n qubits, generates all 4^n Pauli operators as a tensor product
    of I, X, Y, Z operators.

    :param n: Number of qubits (default: 1)
    :return: Array of shape (4^n, 2^n, 2^n) containing all n-qubit Pauli operators
    """
    if n == 1:
        return PAULI_ENSEMBLE

    # Build tensor products recursively

    paulis_1q = [I.matrix, X.matrix, Y.matrix, Z.matrix]
    n_qubit_paulis = []

    for pauli_tuple in product(paulis_1q, repeat=n):
        # Compute tensor product
        result = reduce(jnp.kron, pauli_tuple)
        n_qubit_paulis.append(result)

    return Unitary.from_matrix(jnp.array(n_qubit_paulis, dtype=complex), ((2,) * n, (2,) * n), 1)
