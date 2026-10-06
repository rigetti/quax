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

"""Performance benchmarks for targeted apply operations.

Run with:
    make benchmark

These benchmarks are NOT included in the default test suite (``make test-package``).
"""

from functools import reduce
from itertools import product
from operator import mul

import jax
import pytest

import quax as qx

from .conftest import (
    make_density_matrix,
    make_depolarizing_kraus,
    make_ideal_instrument,
    make_keys,
    make_kraus_map,
    make_noisy_instrument,
    make_state_vector,
    make_superop,
    make_unitary,
)

# ---------------------------------------------------------------------------
# System configurations
# ---------------------------------------------------------------------------

# (label, dims) for state-vector-safe systems
SV_SYSTEMS = [
    ("6Q", (2,) * 6),
    ("10Q", (2,) * 10),
    ("16Q", (2,) * 16),
    ("20Q", (2,) * 20),
]

# (label, dims) for density-matrix-safe systems (capped for memory)
DM_SYSTEMS = [
    ("2Q", (2, 2)),
    ("4Q", (2,) * 4),
    ("8Q", (2,) * 8),
    ("10Q", (2,) * 10),
]

ENSEMBLE_SIZES = [(), (4,), (8,), (16,)]

# (label, dims) for building the tensor-product operator bases (D**4 entries each)
BASIS_SYSTEMS = [
    ("2Q", (2, 2)),
    ("4Q", (2,) * 4),
    ("6Q", (2,) * 6),
    ("3Qt", (3,) * 3),
    ("2Qt2Q", (3, 3, 2, 2)),
]


def _subsystem_patterns(dims: tuple[int, ...]) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    """Return applicable (label, gate_dims, subsystem) tuples for given system dims."""
    n = len(dims)
    patterns: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    patterns.append(("1Qop", (dims[0],), (0,)))
    if n >= 2:
        patterns.append(("2Qop", dims[:2], (0, 1)))
    if n >= 3:
        patterns.append(("3Qop", dims[:3], (0, 1, 2)))
    if n >= 4:
        patterns.append(("4Qop", dims[:4], (0, 1, 2, 3)))
    return patterns


def _get_instrument(inst_type: str, dim: int = 2) -> qx.QuantumInstrument:
    """Get a single-qudit instrument by type."""
    if inst_type == "ideal":
        return make_ideal_instrument(dim=dim)
    elif inst_type == "noisy":
        return make_noisy_instrument(dim=dim)
    raise ValueError(f"Unknown instrument type: {inst_type}")


# Kraus source descriptors: (label, factory_fn)
# factory_fn(gate_dims, truncate) -> KrausMap
KRAUS_SOURCES = {
    "rank2": lambda gate_dims, tr: make_kraus_map(gate_dims, rank=2, truncate=tr),
    "depolarizing": lambda gate_dims, tr: make_depolarizing_kraus(gate_dims, truncate=tr),
}


# ---------------------------------------------------------------------------
# 1. targeted_apply_unitary  (state vectors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dims,ensemble_size,gate_dims,subsystem",
    [
        pytest.param(dims, ens, gdims, sub, id=f"{lbl}-ens{ens}-{slbl}")
        for (lbl, dims), ens in product(SV_SYSTEMS, ENSEMBLE_SIZES)
        for slbl, gdims, sub in _subsystem_patterns(dims)
    ],
)
def test_targeted_apply_unitary(benchmark, dims, ensemble_size, gate_dims, subsystem):
    """Benchmark targeted_apply_unitary on state vectors."""
    psi = make_state_vector(dims, ensemble_size)
    gate = make_unitary(gate_dims)

    # JIT warmup — first call triggers compilation
    result = qx.targeted_apply_unitary(gate, psi, subsystem)
    result.data.block_until_ready()

    def fn():
        r = qx.targeted_apply_unitary(gate, psi, subsystem)
        r.data.block_until_ready()

    benchmark(fn)


# ---------------------------------------------------------------------------
# 2. targeted_apply_superop  (density matrices)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dims,ensemble_size,gate_dims,subsystem",
    [
        pytest.param(dims, ens, gdims, sub, id=f"{lbl}-ens{ens}-{slbl}")
        for (lbl, dims), ens in product(DM_SYSTEMS, ENSEMBLE_SIZES)
        for slbl, gdims, sub in _subsystem_patterns(dims)
    ],
)
def test_targeted_apply_superop(benchmark, dims, ensemble_size, gate_dims, subsystem):
    """Benchmark targeted_apply_superop on density matrices."""
    rho = make_density_matrix(dims, ensemble_size)
    superop = make_superop(gate_dims)

    # JIT warmup
    result = qx.targeted_apply_superop(superop, rho, subsystem)
    result.data.block_until_ready()

    def fn():
        r = qx.targeted_apply_superop(superop, rho, subsystem)
        r.data.block_until_ready()

    benchmark(fn)


# ---------------------------------------------------------------------------
# 3. targeted_apply_kraus_map_trajectory  (state vectors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dims,ensemble_size,source_label,truncated,gate_dims,subsystem",
    [
        pytest.param(
            dims,
            ens,
            src,
            trunc,
            gdims,
            sub,
            id=f"{lbl}-ens{ens}-{src}-{'trunc' if trunc else 'full'}-{slbl}",
        )
        for (lbl, dims), ens, src, trunc in product(SV_SYSTEMS, ENSEMBLE_SIZES, KRAUS_SOURCES.keys(), [False, True])
        for slbl, gdims, sub in _subsystem_patterns(dims)
    ],
)
def test_kraus_trajectory(benchmark, dims, ensemble_size, source_label, truncated, gate_dims, subsystem):
    """Benchmark targeted_apply_kraus_map_trajectory on state vectors."""
    psi = make_state_vector(dims, ensemble_size)
    kraus = KRAUS_SOURCES[source_label](gate_dims, truncated)
    key = make_keys(1)

    # JIT warmup
    result = qx.targeted_apply_kraus_map_trajectory(kraus, psi, key, subsystem)
    result.data.block_until_ready()

    def fn():
        r = qx.targeted_apply_kraus_map_trajectory(kraus, psi, key, subsystem)
        r.data.block_until_ready()

    benchmark(fn)


# ---------------------------------------------------------------------------
# 4. targeted_apply_instrument  (density matrices)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dims,ensemble_size,inst_type",
    [
        pytest.param(dims, ens, itype, id=f"{lbl}-ens{ens}-{itype}")
        for (lbl, dims), ens, itype in product(DM_SYSTEMS, ENSEMBLE_SIZES, ["ideal", "noisy"])
    ],
)
def test_instrument_dm(benchmark, dims, ensemble_size, inst_type):
    """Benchmark targeted_apply_instrument_to_density_matrix."""
    rho = make_density_matrix(dims, ensemble_size)
    inst = _get_instrument(inst_type, dim=dims[0])
    subsystem = (0,)

    # JIT warmup
    rho_outs, probs = qx.targeted_apply_instrument_to_density_matrix(inst, rho, subsystem)
    rho_outs.data.block_until_ready()
    probs.block_until_ready()

    def fn():
        ro, p = qx.targeted_apply_instrument_to_density_matrix(inst, rho, subsystem)
        ro.data.block_until_ready()
        p.block_until_ready()

    benchmark(fn)


# ---------------------------------------------------------------------------
# 5. targeted_apply_instrument  (state vectors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dims,ensemble_size,inst_type",
    [
        pytest.param(dims, ens, itype, id=f"{lbl}-ens{ens}-{itype}")
        for (lbl, dims), ens, itype in product(SV_SYSTEMS, ENSEMBLE_SIZES, ["ideal", "noisy"])
    ],
)
def test_instrument_sv(benchmark, dims, ensemble_size, inst_type):
    """Benchmark targeted_apply_instrument_to_state_vector."""
    psi = make_state_vector(dims, ensemble_size)
    inst = _get_instrument(inst_type, dim=dims[0])
    key = make_keys(1)
    subsystem = (0,)

    # JIT warmup
    psi_out, _outcome = qx.targeted_apply_instrument_to_state_vector(inst, psi, key, subsystem)
    psi_out.data.block_until_ready()

    def fn():
        r, o = qx.targeted_apply_instrument_to_state_vector(inst, psi, key, subsystem)
        r.data.block_until_ready()
        o.block_until_ready()

    benchmark(fn)


# ---------------------------------------------------------------------------
# 6. n_qudit_basis / n_qudit_herm_basis  (cold-cache construction)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dims", [pytest.param(dims, id=lbl) for lbl, dims in BASIS_SYSTEMS])
@pytest.mark.parametrize("build", [qx.n_qudit_basis, qx.n_qudit_herm_basis], ids=["weyl", "hermitian"])
def test_tensor_basis_build(benchmark, build, dims):
    """Benchmark building n_qudit_basis / n_qudit_herm_basis from a cold cache."""

    def fn():
        build.cache_clear()
        build(dims)

    benchmark(fn)


# ---------------------------------------------------------------------------
# Pauli-Liouville change of basis
# ---------------------------------------------------------------------------

PAULI_LIOUVILLE_SYSTEMS = [
    ("1Q", (2,)),
    ("3Q", (2,) * 3),
    ("5Q", (2,) * 5),
    ("2Qt", (3, 3)),
]


# Each conversion with the representation it takes, built from a random unitary channel.
PAULI_LIOUVILLE_CONVERSIONS = [
    ("superop_to_pauli_liouville", qx.to_superop),
    ("pauli_liouville_to_superop", qx.to_pauli_liouville),
    ("choi_to_pauli_liouville", qx.to_choi),
    ("pauli_liouville_to_choi", qx.to_pauli_liouville),
    ("kraus_to_pauli_liouville", lambda unitary: qx.truncate_kraus(qx.to_kraus(unitary))),
    ("unitary_to_pauli_liouville", lambda unitary: unitary),
]


@pytest.mark.parametrize("dims", [pytest.param(dims, id=lbl) for lbl, dims in PAULI_LIOUVILLE_SYSTEMS])
@pytest.mark.parametrize(
    ("name", "make_input"), [pytest.param(name, make, id=name) for name, make in PAULI_LIOUVILLE_CONVERSIONS]
)
def test_pauli_liouville_conversion(benchmark, name, make_input, dims):
    """Benchmark the conversions that change to or from the Pauli-Liouville basis."""
    convert = getattr(qx, name)
    operator = make_input(make_unitary(dims))
    convert(operator).data.block_until_ready()
    benchmark(lambda: convert(operator).data.block_until_ready())


# ---------------------------------------------------------------------------
# Kraus to superoperator
# ---------------------------------------------------------------------------

KRAUS_TO_SUPEROP_SYSTEMS = [
    ("1Q", (2,)),
    ("3Q", (2,) * 3),
    ("4Q", (2,) * 4),
]


@pytest.mark.parametrize("dims", [pytest.param(dims, id=lbl) for lbl, dims in KRAUS_TO_SUPEROP_SYSTEMS])
@pytest.mark.parametrize("full_rank", [True, False], ids=["full_rank", "rank1"])
def test_kraus_to_superop(benchmark, dims, full_rank):
    """Benchmark kraus_to_superop with D² Kraus operators (full rank) and with a single one."""
    d = reduce(mul, dims)
    kraus_map = make_kraus_map(dims, rank=d * d if full_rank else 1, truncate=not full_rank)
    qx.kraus_to_superop(kraus_map).data.block_until_ready()
    benchmark(lambda: qx.kraus_to_superop(kraus_map).data.block_until_ready())


# ---------------------------------------------------------------------------
# Unitary entanglement fidelity
# ---------------------------------------------------------------------------

ENTANGLEMENT_FIDELITY_SYSTEMS = [
    ("1Q", (2,)),
    ("6Q", (2,) * 6),
    ("9Q", (2,) * 9),
    ("11Q", (2,) * 11),
]


@pytest.mark.parametrize("dims", [pytest.param(dims, id=lbl) for lbl, dims in ENTANGLEMENT_FIDELITY_SYSTEMS])
def test_unitary_entanglement_fidelity(benchmark, dims):
    """Benchmark unitary_entanglement_fidelity, which also backs ``Unitary.__eq__``."""
    unitary_e = make_unitary(dims)
    unitary_f = qx.random_unitary(dims=(dims, dims), key=jax.random.key(7))
    qx.unitary_entanglement_fidelity(unitary_e, unitary_f).block_until_ready()
    benchmark(lambda: qx.unitary_entanglement_fidelity(unitary_e, unitary_f).block_until_ready())


# ---------------------------------------------------------------------------
# Instrument diagnostics: basis-state transitions per outcome
# ---------------------------------------------------------------------------

INSTRUMENT_DIAGNOSTIC_QUBITS = [1, 2, 4]

INSTRUMENT_DIAGNOSTICS = [
    ("confusion_matrix", lambda instrument: instrument.confusion_matrix),
    ("transition_matrix", lambda instrument: instrument.transition_matrix),
    ("non_demolition_fidelity", qx.non_demolition_fidelity),
    ("instrument_fidelity", qx.instrument_fidelity),
]


@pytest.mark.parametrize("n_qubits", [pytest.param(n, id=f"{n}Q") for n in INSTRUMENT_DIAGNOSTIC_QUBITS])
@pytest.mark.parametrize(("name", "diagnostic"), [pytest.param(name, f, id=name) for name, f in INSTRUMENT_DIAGNOSTICS])
def test_instrument_diagnostic(benchmark, name, diagnostic, n_qubits):
    """Benchmark the instrument diagnostics on a noisy measurement of every qubit."""
    instrument = reduce(lambda a, b: a | b, [make_noisy_instrument(dim=2)] * n_qubits)
    jax.block_until_ready(diagnostic(instrument))
    benchmark(lambda: jax.block_until_ready(diagnostic(instrument)))
