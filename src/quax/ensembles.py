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

"""Predefined ensembles and groups of single-qubit operators.

The fixed ensembles here (Paulis, Cliffords, icosahedral groups, SIC/Pauli preparation unitaries)
are **built lazily on first access** at the current JAX precision (see :mod:`quax._lazy`).  This
both keeps them precision-correct when x64 is toggled after import and keeps the (relatively
expensive) group generation off the ``import quax`` path until something actually uses it.
"""

from functools import reduce
from itertools import permutations, product
from typing import Any, Callable

import jax.numpy as jnp
from jax.numpy import arccos, pi, sqrt

from . import gates
from ._lazy import make_lazy_getter
from ._quantum_objects import Unitary

_get: Callable[[str], Any]


def _make_SIC_PREP() -> Unitary:
    """The unitary operators that prepare the SIC states from |0> in the order SIC0, SIC1, SIC2, SIC3."""
    RX, RZ, I = gates.RX, gates.RZ, gates.I  # noqa: E741
    return Unitary.from_matrix(
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
    )


def _make_SIC_STATE_PREP_OPERATORS() -> dict[str, Any]:
    """The unitary operators that prepare the SIC states from |0>."""
    sic_prep = _get("SIC_PREP")
    return {
        "SIC0": sic_prep.matrix[0],
        "SIC1": sic_prep.matrix[1],
        "SIC2": sic_prep.matrix[2],
        "SIC3": sic_prep.matrix[3],
    }


def _make_PAULI_PREP() -> Unitary:
    """The unitary operators that prepare the Pauli states from |0> in the order X+, X-, Y+, Y-, Z+, Z-."""
    RX, RY, I = gates.RX, gates.RY, gates.I  # noqa: E741
    return Unitary.from_matrix(
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
    )


def _make_PAULI_STATE_PREP_OPERATORS() -> dict[str, Any]:
    """The unitary operators that prepare the Pauli states from |0>."""
    pauli_prep = _get("PAULI_PREP")
    return {
        "X+": pauli_prep.matrix[0],
        "X-": pauli_prep.matrix[1],
        "Y+": pauli_prep.matrix[2],
        "Y-": pauli_prep.matrix[3],
        "Z+": pauli_prep.matrix[4],
        "Z-": pauli_prep.matrix[5],
    }


def _make_PAULI_ENSEMBLE() -> Unitary:
    """The ensemble of Pauli operators."""
    I, X, Y, Z = gates.I, gates.X, gates.Y, gates.Z  # noqa: E741
    return Unitary.from_matrix(jnp.asarray([I.matrix, X.matrix, Y.matrix, Z.matrix], dtype=complex), ((2,), (2,)))


def _make_TETRAHEDRAL_ENSEMBLE() -> Unitary:
    """The tetrahedral ensemble of operators."""
    RX, RZ, I, X, Z = gates.RX, gates.RZ, gates.I, gates.X, gates.Z  # noqa: E741
    return Unitary.from_matrix(
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
    )


def _make_CLIFFORD_ENSEMBLE() -> Unitary:
    """The ensemble of single-qubit Clifford operators."""
    RX, RZ, I, X, Z = gates.RX, gates.RZ, gates.I, gates.X, gates.Z  # noqa: E741
    return Unitary.from_matrix(
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
    )


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
    return Unitary.from_matrix(icosahedral_unitaries, ((2,), (2,)))


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
    return Unitary.from_matrix(icosahedral_unitaries, ((2,), (2,)))


def n_qubit_pauli_operators(n: int = 1) -> Unitary:
    """Generate all n-qubit Pauli operators.

    For n qubits, generates all 4^n Pauli operators as a tensor product
    of I, X, Y, Z operators.

    :param n: Number of qubits (default: 1)
    :return: Array of shape (4^n, 2^n, 2^n) containing all n-qubit Pauli operators
    """
    if n == 1:
        return _get("PAULI_ENSEMBLE")

    # Build tensor products recursively

    paulis_1q = [gates.I.matrix, gates.X.matrix, gates.Y.matrix, gates.Z.matrix]
    n_qubit_paulis = []

    for pauli_tuple in product(paulis_1q, repeat=n):
        # Compute tensor product
        result = reduce(jnp.kron, pauli_tuple)
        n_qubit_paulis.append(result)

    return Unitary.from_matrix(jnp.array(n_qubit_paulis, dtype=complex), ((2,) * n, (2,) * n))


# Registry of lazily-built constants: name -> builder.  Aliases share a builder.
_BUILDERS: dict[str, Callable[[], Any]] = {
    "SIC_PREP": _make_SIC_PREP,
    "SIC_STATE_PREP_OPERATORS": _make_SIC_STATE_PREP_OPERATORS,
    "PAULI_PREP": _make_PAULI_PREP,
    "PAULI_STATE_PREP_OPERATORS": _make_PAULI_STATE_PREP_OPERATORS,
    "PAULI_ENSEMBLE": _make_PAULI_ENSEMBLE,
    "PAULIS": _make_PAULI_ENSEMBLE,
    "TETRAHEDRAL_ENSEMBLE": _make_TETRAHEDRAL_ENSEMBLE,
    "CLIFFORD_ENSEMBLE": _make_CLIFFORD_ENSEMBLE,
    "OCTAHEDRAL_ENSEMBLE": _make_CLIFFORD_ENSEMBLE,
    "CLIFFORDS_1Q": _make_CLIFFORD_ENSEMBLE,
    "ICOSAHEDRAL_ENSEMBLE": _generate_icosahedral_rotation_group,
    "ICOSAHEDRAL_GROUP": _generate_icosahedral_rotation_group,
    "BINARY_ICOSAHEDRAL_ENSEMBLE": _generate_binary_icosahedral_group,
    "BINARY_ICOSAHEDRAL_GROUP": _generate_binary_icosahedral_group,
}

_get = make_lazy_getter(_BUILDERS)


def __getattr__(name: str) -> Any:
    if name in _BUILDERS:
        return _get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _BUILDERS.keys())
