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

"""Predefined state vectors.

Like :mod:`quax.gates`, the fixed states here are **built lazily on first access** at the current
JAX precision (see :mod:`quax._lazy`), so ``qx.states.KET0`` picks up an x64 setting enabled after
``quax`` was imported.  They remain plain attributes, not functions.
"""

from typing import Any, Callable

import jax.numpy as jnp
from jax.numpy import exp, pi, sqrt

from ._lazy import make_lazy_getter
from ._quantum_objects import StateVector

_get: Callable[[str], Any]


def _make_KET0() -> StateVector:
    """|0> state vector."""
    return StateVector.from_matrix(jnp.array([1, 0], dtype=complex), (2,))


def _make_KET1() -> StateVector:
    """|1> state vector."""
    return StateVector.from_matrix(jnp.array([0, 1], dtype=complex), (2,))


def _make_KETPLUS() -> StateVector:
    """|+> = (|0> + |1>) / sqrt(2) state vector."""
    return StateVector.from_matrix(jnp.array([1, 1], dtype=complex) / sqrt(2), (2,))


def _make_KETMINUS() -> StateVector:
    """|-> = (|0> - |1>) / sqrt(2) state vector."""
    return StateVector.from_matrix(jnp.array([1, -1], dtype=complex) / sqrt(2), (2,))


def _make_KETPLUSI() -> StateVector:
    """|+i> = (|0> + i|1>) / sqrt(2) state vector."""
    return StateVector.from_matrix(jnp.array([1, 1j], dtype=complex) / sqrt(2), (2,))


def _make_KETMINUSI() -> StateVector:
    """|-i> = (|0> - i|1>) / sqrt(2) state vector."""
    return StateVector.from_matrix(jnp.array([1, -1j], dtype=complex) / sqrt(2), (2,))


def _make_SIC1() -> StateVector:
    KET0, KET1 = _get("KET0"), _get("KET1")
    return StateVector.from_matrix(
        exp(1j * pi / 2) * (1 / sqrt(3) * KET0.matrix) + exp(1j * pi / 2) * sqrt(2 / 3) * KET1.matrix, (2,)
    )


def _make_SIC2() -> StateVector:
    KET0, KET1 = _get("KET0"), _get("KET1")
    return StateVector.from_matrix(
        exp(1j * 5 * pi / 6) * (1 / sqrt(3) * KET0.matrix) + exp(1j * 1 * pi / 6) * sqrt(2 / 3) * KET1.matrix, (2,)
    )


def _make_SIC3() -> StateVector:
    KET0, KET1 = _get("KET0"), _get("KET1")
    return StateVector.from_matrix(
        exp(1j * pi / 6) * (1 / sqrt(3) * KET0.matrix) + exp(1j * 5 * pi / 6) * sqrt(2 / 3) * KET1.matrix, (2,)
    )


def _make_SIC_STATES() -> dict[str, StateVector]:
    """The symmetric informationally complete POVMs for a qubit.

    These can reduce the number of experiments to perform quantum process tomography.
    For more information, please see http://info.phys.unm.edu/~caves/reports/infopovm.pdf
    """
    return {
        "SIC0": _get("SIC1"),
        "SIC1": _get("SIC1"),
        "SIC2": _get("SIC2"),
        "SIC3": _get("SIC3"),
    }


def _make_PAULI_STATES() -> dict[str, StateVector]:
    """The six eigenstates of the Pauli operators X, Y, and Z."""
    return {
        "X+": _get("XPLUS"),
        "X-": _get("XMINUS"),
        "Y+": _get("YPLUS"),
        "Y-": _get("YMINUS"),
        "Z+": _get("ZPLUS"),
        "Z-": _get("ZMINUS"),
    }


def _make_STATES() -> dict[str, list[StateVector]]:
    return {
        "X": [
            StateVector.from_matrix(jnp.array([1, 1], dtype=complex) / jnp.sqrt(2), (2,)),
            StateVector.from_matrix(jnp.array([1, -1], dtype=complex) / jnp.sqrt(2), (2,)),
        ],
        "Y": [
            StateVector.from_matrix(jnp.array([1, 1j], dtype=complex) / jnp.sqrt(2), (2,)),
            StateVector.from_matrix(jnp.array([1, -1j], dtype=complex) / jnp.sqrt(2), (2,)),
        ],
        "Z": [
            StateVector.from_matrix(jnp.array([1, 0], dtype=complex), (2,)),
            StateVector.from_matrix(jnp.array([0, 1], dtype=complex), (2,)),
        ],
        "SIC": [_get("SIC0"), _get("SIC1"), _get("SIC2"), _get("SIC3")],
    }


# Registry of lazily-built constants: name -> builder.  Aliases share a builder (e.g. KET0/ZPLUS,
# and SIC0 which equals KET0).
_BUILDERS: dict[str, Callable[[], Any]] = {
    "KET0": _make_KET0,
    "ZPLUS": _make_KET0,
    "SIC0": _make_KET0,
    "KET1": _make_KET1,
    "ZMINUS": _make_KET1,
    "KETPLUS": _make_KETPLUS,
    "XPLUS": _make_KETPLUS,
    "KETMINUS": _make_KETMINUS,
    "XMINUS": _make_KETMINUS,
    "KETPLUSI": _make_KETPLUSI,
    "YPLUS": _make_KETPLUSI,
    "KETMINUSI": _make_KETMINUSI,
    "YMINUS": _make_KETMINUSI,
    "SIC1": _make_SIC1,
    "SIC2": _make_SIC2,
    "SIC3": _make_SIC3,
    "SIC_STATES": _make_SIC_STATES,
    "PAULI_STATES": _make_PAULI_STATES,
    "STATES": _make_STATES,
}

_get = make_lazy_getter(_BUILDERS)


def __getattr__(name: str) -> Any:
    if name in _BUILDERS:
        return _get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _BUILDERS.keys())
