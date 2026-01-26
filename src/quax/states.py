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

import jax.numpy as jnp
from jax.numpy import exp, pi, sqrt

from ._quantum_objects import StateVector

KET0 = ZPLUS = StateVector.from_matrix(jnp.array([1, 0], dtype=complex), (2,), 0)
"""|0> state vector."""

KET1 = ZMINUS = StateVector.from_matrix(jnp.array([0, 1], dtype=complex), (2,), 0)
"""|1> state vector."""

KETPLUS = XPLUS = StateVector.from_matrix(jnp.array([1, 1], dtype=complex) / sqrt(2), (2,), 0)
"""|+> = (|0> + |1>) / sqrt(2) state vector."""

KETMINUS = XMINUS = StateVector.from_matrix(jnp.array([1, -1], dtype=complex) / sqrt(2), (2,), 0)
"""|-> = (|0> - |1>) / sqrt(2) state vector."""

KETPLUSI = YPLUS = StateVector.from_matrix(jnp.array([1, 1j], dtype=complex) / sqrt(2), (2,), 0)
"""|+i> = (|0> + i|1>) / sqrt(2) state vector."""

KETMINUSI = YMINUS = StateVector.from_matrix(jnp.array([1, -1j], dtype=complex) / sqrt(2), (2,), 0)
"""|-i> = (|0> - i|1>) / sqrt(2) state vector."""


SIC0 = KET0
SIC1 = StateVector.from_matrix(
    exp(1j * pi / 2) * (1 / sqrt(3) * KET0.matrix) + exp(1j * pi / 2) * sqrt(2 / 3) * KET1.matrix, (2,), 0
)
SIC2 = StateVector.from_matrix(
    exp(1j * 5 * pi / 6) * (1 / sqrt(3) * KET0.matrix) + exp(1j * 1 * pi / 6) * sqrt(2 / 3) * KET1.matrix, (2,), 0
)
SIC3 = StateVector.from_matrix(
    exp(1j * pi / 6) * (1 / sqrt(3) * KET0.matrix) + exp(1j * 5 * pi / 6) * sqrt(2 / 3) * KET1.matrix, (2,), 0
)

SIC_STATES = {
    "SIC0": SIC1,
    "SIC1": SIC1,
    "SIC2": SIC2,
    "SIC3": SIC3,
}
"""
The symmetric informationally complete POVMs for a qubit.

These can reduce the number of experiments to perform quantum process tomography.
For more information, please see http://info.phys.unm.edu/~caves/reports/infopovm.pdf
"""

PAULI_STATES = {
    "X+": XPLUS,
    "X-": XMINUS,
    "Y+": YPLUS,
    "Y-": YMINUS,
    "Z+": ZPLUS,
    "Z-": ZMINUS,
}
"""The six eigenstates of the Pauli operators X, Y, and Z."""

STATES = {
    "X": [
        StateVector.from_matrix(jnp.array([1, 1], dtype=complex) / jnp.sqrt(2), (2,), 0),
        StateVector.from_matrix(jnp.array([1, -1], dtype=complex) / jnp.sqrt(2), (2,), 0),
    ],
    "Y": [
        StateVector.from_matrix(jnp.array([1, 1j], dtype=complex) / jnp.sqrt(2), (2,), 0),
        StateVector.from_matrix(jnp.array([1, -1j], dtype=complex) / jnp.sqrt(2), (2,), 0),
    ],
    "Z": [
        StateVector.from_matrix(jnp.array([1, 0], dtype=complex), (2,), 0),
        StateVector.from_matrix(jnp.array([0, 1], dtype=complex), (2,), 0),
    ],
    "SIC": [SIC0, SIC1, SIC2, SIC3],
}
