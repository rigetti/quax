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
"""Enumeration of pure stabilizer states."""

from collections import Counter
from functools import lru_cache, reduce
from math import prod

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from ._quantum_objects import StateVector


def _is_prime(d: int) -> bool:
    return d >= 2 and all(d % k for k in range(2, int(d**0.5) + 1))


def _stabilizer_state_count(dims: tuple[int, ...]) -> int:
    """Gross's stabilizer-state count, multiplied over the distinct primes in ``dims``."""
    return prod(p**n * prod(p**i + 1 for i in range(1, n + 1)) for p, n in Counter(dims).items())


def _clifford_generators(dims: tuple[int, ...]) -> list[npt.NDArray[np.complex128]]:
    """Clifford generators: Fourier, phase and ``Z`` per qudit, ``SUM`` between equal qudits."""

    def on(qudit: int, gate: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        return np.asarray(
            reduce(np.kron, [gate if k == qudit else np.eye(d) for k, d in enumerate(dims)]), dtype=complex
        )

    generators = []
    for qudit, d in enumerate(dims):
        omega = np.exp(2j * np.pi / d)
        j = np.arange(d)
        tau = (-1) ** d * np.exp(1j * np.pi / d)
        generators.append(on(qudit, omega ** np.outer(j, j) / np.sqrt(d)))  # Fourier
        generators.append(on(qudit, np.diag(tau ** (j * j))))  # phase: X -> XZ up to phase
        generators.append(on(qudit, np.diag(omega**j)))  # Z
    total = prod(dims)
    digits = np.array(list(np.ndindex(*dims)))  # row k: the digits of basis state k
    for control, target in [(a, b) for a in range(len(dims)) for b in range(len(dims)) if a != b]:
        if dims[control] != dims[target]:
            continue
        shifted = digits.copy()
        shifted[:, target] = (digits[:, target] + digits[:, control]) % dims[target]
        rows = np.ravel_multi_index(tuple(shifted.T), dims)
        sum_gate = np.zeros((total, total), dtype=complex)
        sum_gate[rows, np.arange(total)] = 1
        generators.append(sum_gate)
    return generators


def _canonical_keys(states: npt.NDArray[np.complex128]) -> list[bytes]:
    """Phase-free keys: each row rotated so its first nonzero amplitude is real and positive."""
    first = np.argmax(np.abs(states) > 1e-6, axis=-1)
    phase = states[np.arange(len(states)), first]
    canonical = states * (np.abs(phase) / phase)[:, None]
    rounded = np.round(np.stack([canonical.real, canonical.imag], axis=-1) * 1e6).astype(np.int64)
    return [row.tobytes() for row in rounded]


@lru_cache(maxsize=32)
def stabilizer_states(dims: tuple[int, ...]) -> StateVector:
    r"""Every pure stabilizer state of qudits with prime dimensions ``dims``, up to global phase.

    A stabilizer state is the common eigenvector of :math:`D` commuting Weyl operators. In
    phase-space language it is :math:`|M, v\rangle`, fixed by a maximal isotropic subspace
    :math:`M` (the operators' labels, :math:`|M| = D`) and a vector :math:`v` setting their
    eigenvalues, with projector :math:`D^{-1}\sum_{m \in M} \chi([v, m])\, w(m)` ([HUDSON]_
    Lemma 8, Eq. (17)). For qubits these are the familiar stabilizer states: :math:`|0\rangle`,
    :math:`|+\rangle`, :math:`|{+i}\rangle`, GHZ and graph states, and their Pauli and Clifford
    images.

    How many there are ([HUDSON]_ Corollary 21): :math:`n` qudits of prime dimension :math:`d`
    have :math:`d^n \prod_{i=1}^n (d^i + 1)` stabilizer states. Qudits of different primes do not
    share stabilizer states, so for mixed ``dims`` the counts of each prime's qudits multiply. One
    qubit has 6, two have 60, three 1,080 and four 36,720; two qutrits have 360.

    Five qubits give 2,423,520 states, about 1.2 GB as state vectors; nothing stops a larger
    request, so check the count first.

    Uses: ground truth for tests of magic measures (each state has :func:`stabilizer_norm` 1 and
    :func:`stabilizer_renyi_entropy` 0), and the vertices of the stabilizer polytope over which
    the robustness of magic and the stabilizer fidelity are defined.

    The states are found as the orbit of :math:`|0\cdots0\rangle` under the Clifford group, which
    for prime :math:`d` acts transitively on them. The orbit is searched breadth first with the
    Fourier, phase and :math:`Z` gates on each qudit and the ``SUM`` gate between qudits of equal
    dimension, and its size is checked against the count above. The first state is
    :math:`|0\cdots0\rangle`; the rest follow in search order, each with its first nonzero
    amplitude real and positive. Results are cached per ``dims``.

    :param dims: Prime local dimensions, e.g. ``(2, 2)`` or ``(2, 3)``; must be a tuple.
    :return: A ``StateVector`` ensemble of shape ``(N, *dims)``.
    :raises ValueError: If any dimension is not prime.

    The full citation for [HUDSON]_ is under :func:`characteristic_function`.
    """
    if not dims or not all(_is_prime(d) for d in dims):
        raise ValueError(f"Stabilizer states are enumerated for prime local dimensions only, got {dims}.")
    generators = _clifford_generators(dims)
    start = np.zeros((1, prod(dims)), dtype=complex)
    start[0, 0] = 1
    found = dict(zip(_canonical_keys(start), start))
    frontier = start
    while len(frontier):
        candidates = np.concatenate([frontier @ gate.T for gate in generators])
        new = {}
        for key, state in zip(_canonical_keys(candidates), candidates):
            if key not in found and key not in new:
                new[key] = state
        found.update(new)
        frontier = np.array(list(new.values())).reshape(-1, prod(dims))
    expected = _stabilizer_state_count(dims)
    if len(found) != expected:
        raise RuntimeError(f"Found {len(found)} stabilizer states for dims {dims}, expected {expected}.")
    states = np.array(list(found.values()))
    first = np.argmax(np.abs(states) > 1e-6, axis=-1)
    phase = states[np.arange(len(states)), first]
    states = states * (np.abs(phase) / phase)[:, None]
    return StateVector.from_matrix(jnp.asarray(states), dims)
