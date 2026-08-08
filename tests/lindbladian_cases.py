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

"""A battery of physically motivated Lindbladian-learning test cases.

Each :class:`LearningCase` bundles

* a **structurally correct** ansatz (:class:`quax.learning.LindbladModel`),
* the **true** coefficients, drawn at random from realistic superconducting-qubit ranges, and
* an **initial guess** whose coefficients sit within a bounded fraction (default 50%) of the truth.

The cases progress from one to four qubits and from decoherence only, through coherent detuning, ZZ
crosstalk and Rabi drives, up to time-dependent XY exchange; two "exotic" cases cover correlated
(joint) decay and XZ drive crosstalk.

Coherent couplings are deliberately **orders of magnitude faster** than the dissipative rates — a
Rabi drive of ~50 rad/us against a T1 rate of ~0.04 /us — which is the regime that makes learning
hard.

An :class:`Experiment` wraps a case with the two experimental imperfections we want to sweep over:
finite **shot noise** and an unknown turn-on **ramp** before the measurement window.

Units: time in microseconds, dissipative rates in 1/us, coherent couplings in rad/us (= 2*pi*f).
"""

import itertools
from dataclasses import dataclass, field
from functools import reduce

import jax
import jax.numpy as jnp
from jax import Array

import quax as qx
from quax.learning import (
    LindbladModel,
    Term,
    amplitude_damping_term,
    dephasing_term,
    detuning_term,
    drive_term,
    exchange_term,
    excitation_term,
    joint_decay_term,
    xz_term,
    zz_term,
)

# Angular frequency in rad/us for a frequency quoted in MHz (or kHz): omega = 2*pi*f[Hz]*1e-6.
MHZ = 2 * jnp.pi  # rad/us per MHz
KHZ = MHZ * 1e-3  # rad/us per kHz

#: Ranges for the physical parameters, in working units (1/us for rates, rad/us for couplings).
T1_RANGE = (15.0, 40.0)  # us
TPHI_RANGE = (20.0, 60.0)  # us
EXCITED_POPULATION_RANGE = (0.01, 0.05)  # dimensionless
DETUNING_RANGE = (100.0 * KHZ, 1000.0 * KHZ)
ZZ_RANGE = (20.0 * KHZ, 200.0 * KHZ)
XZ_RANGE = (50.0 * KHZ, 500.0 * KHZ)
RABI_PERIOD_RANGE = (0.08, 0.20)  # us -> Omega = 2*pi/period, ~30-80 rad/us
EXCHANGE_RANGE = (1.0 * MHZ, 5.0 * MHZ)
QUBIT_SPACING_RANGE = (100.0 * MHZ, 400.0 * MHZ)  # frequency differences for modulated exchange
JOINT_DECAY_RANGE = (0.005, 0.02)  # 1/us

SQUARE_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


def _uniform(key: Array, low: float, high: float, shape: tuple[int, ...] = ()) -> Array:
    return jax.random.uniform(key, shape, minval=low, maxval=high)


@dataclass(frozen=True)
class LearningCase:
    """A Lindbladian-learning problem: a model, its true coefficients, and a nearby initial guess.

    :param name: Unique identifier, e.g. ``"2q-detuning-zz"``.
    :param description: One-line physical description.
    :param model: The structurally correct ansatz.
    :param true_coefficients: The generating coefficients (what a learner must recover).
    :param initial_guess: Structurally correct starting coefficients, within ``guess_fraction``.
    :param guess_fraction: The bound used when perturbing the truth to make the guess.
    """

    name: str
    description: str
    model: LindbladModel
    true_coefficients: Array
    initial_guess: Array
    guess_fraction: float = 0.5

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the case."""
        return self.model.num_qubits

    @property
    def names(self) -> tuple[str, ...]:
        """Term labels, in coefficient order."""
        return self.model.names

    def relative_error(self, estimate: Array) -> Array:
        """Relative error of ``estimate`` against the truth, per coefficient."""
        return jnp.abs(jnp.asarray(estimate) - self.true_coefficients) / jnp.abs(self.true_coefficients)


def _make_guess(key: Array, truth: Array, fraction: float) -> Array:
    """Perturb every coefficient by up to ``fraction`` (relative), preserving sign."""
    perturbation = _uniform(key, -fraction, fraction, truth.shape)
    return truth * (1.0 + perturbation)


def _case(
    name: str,
    description: str,
    num_qubits: int,
    terms: tuple[Term, ...],
    values: Array,
    key: Array,
    guess_fraction: float,
) -> LearningCase:
    model = LindbladModel(num_qubits=num_qubits, terms=terms)
    assert len(terms) == values.shape[0], f"{name}: {len(terms)} terms but {values.shape[0]} values"
    return LearningCase(
        name=name,
        description=description,
        model=model,
        true_coefficients=values,
        initial_guess=_make_guess(key, values, guess_fraction),
        guess_fraction=guess_fraction,
    )


def _decoherence_terms(num_qubits: int) -> tuple[Term, ...]:
    return tuple(amplitude_damping_term(q, num_qubits) for q in range(num_qubits)) + tuple(
        dephasing_term(q, num_qubits) for q in range(num_qubits)
    )


def _decoherence_values(key: Array, num_qubits: int) -> Array:
    k1, k2 = jax.random.split(key)
    damping = 1.0 / _uniform(k1, *T1_RANGE, (num_qubits,))
    dephasing = 1.0 / _uniform(k2, *TPHI_RANGE, (num_qubits,))
    return jnp.concatenate([damping, dephasing])


def build_cases(seed: int = 0, guess_fraction: float = 0.5) -> tuple[LearningCase, ...]:
    """Build the full battery of learning cases.

    :param seed: Seed for the random true coefficients and the guess perturbation.
    :param guess_fraction: Maximum relative perturbation of the initial guess (default 50%).
    :return: The cases, ordered from simplest to hardest.
    """
    keys = iter(jax.random.split(jax.random.key(seed), 64))
    cases: list[LearningCase] = []

    def new(name, description, n, terms, values):
        cases.append(_case(name, description, n, terms, values, next(keys), guess_fraction))

    # ---- one qubit ---------------------------------------------------------
    k = next(keys)
    new(
        "1q-amplitude-damping",
        "T1 relaxation only",
        1,
        (amplitude_damping_term(0, 1),),
        1.0 / _uniform(k, *T1_RANGE, (1,)),
    )

    new("1q-t1-t2", "T1 relaxation plus pure dephasing", 1, _decoherence_terms(1), _decoherence_values(next(keys), 1))

    k = next(keys)
    k1, k2 = jax.random.split(k)
    damping = 1.0 / _uniform(k1, *T1_RANGE, (1,))
    new(
        "1q-thermal",
        "finite-temperature relaxation: decay, excitation and dephasing",
        1,
        (amplitude_damping_term(0, 1), excitation_term(0, 1), dephasing_term(0, 1)),
        jnp.concatenate(
            [
                damping,
                damping * _uniform(k2, *EXCITED_POPULATION_RANGE, (1,)),
                1.0 / _uniform(next(keys), *TPHI_RANGE, (1,)),
            ]
        ),
    )

    new(
        "1q-detuned",
        "decoherence plus a residual Z detuning (coherent >> dissipative)",
        1,
        _decoherence_terms(1) + (detuning_term(0, 1),),
        jnp.concatenate([_decoherence_values(next(keys), 1), _uniform(next(keys), *DETUNING_RANGE, (1,))]),
    )

    new(
        "1q-driven",
        "decoherence plus a resonant Rabi drive (~1000x faster than T1)",
        1,
        _decoherence_terms(1) + (drive_term(0, 1),),
        jnp.concatenate(
            [_decoherence_values(next(keys), 1), 2 * jnp.pi / _uniform(next(keys), *RABI_PERIOD_RANGE, (1,))]
        ),
    )

    new(
        "1q-driven-detuned",
        "driven qubit with a residual detuning and decoherence",
        1,
        _decoherence_terms(1) + (detuning_term(0, 1), drive_term(0, 1)),
        jnp.concatenate(
            [
                _decoherence_values(next(keys), 1),
                _uniform(next(keys), *DETUNING_RANGE, (1,)),
                2 * jnp.pi / _uniform(next(keys), *RABI_PERIOD_RANGE, (1,)),
            ]
        ),
    )

    # ---- two qubits --------------------------------------------------------
    new(
        "2q-decoherence",
        "independent T1/T2 on a qubit pair",
        2,
        _decoherence_terms(2),
        _decoherence_values(next(keys), 2),
    )

    new(
        "2q-detuning-zz",
        "detunings plus static ZZ crosstalk",
        2,
        _decoherence_terms(2) + (detuning_term(0, 2), detuning_term(1, 2), zz_term(0, 1, 2)),
        jnp.concatenate(
            [
                _decoherence_values(next(keys), 2),
                _uniform(next(keys), *DETUNING_RANGE, (2,)),
                _uniform(next(keys), *ZZ_RANGE, (1,)),
            ]
        ),
    )

    new(
        "2q-driven-zz",
        "both qubits driven, with detunings and ZZ crosstalk",
        2,
        _decoherence_terms(2)
        + (detuning_term(0, 2), detuning_term(1, 2), zz_term(0, 1, 2), drive_term(0, 2), drive_term(1, 2)),
        jnp.concatenate(
            [
                _decoherence_values(next(keys), 2),
                _uniform(next(keys), *DETUNING_RANGE, (2,)),
                _uniform(next(keys), *ZZ_RANGE, (1,)),
                2 * jnp.pi / _uniform(next(keys), *RABI_PERIOD_RANGE, (2,)),
            ]
        ),
    )

    new(
        "2q-joint-decay",
        "exotic: correlated (collective) decay alongside independent decoherence",
        2,
        _decoherence_terms(2) + (joint_decay_term(0, 1, 2),),
        jnp.concatenate([_decoherence_values(next(keys), 2), _uniform(next(keys), *JOINT_DECAY_RANGE, (1,))]),
    )

    new(
        "2q-xz-crosstalk",
        "exotic: XZ drive crosstalk (a drive on one qubit conditioned on the other)",
        2,
        _decoherence_terms(2) + (detuning_term(0, 2), detuning_term(1, 2), xz_term(0, 1, 2)),
        jnp.concatenate(
            [
                _decoherence_values(next(keys), 2),
                _uniform(next(keys), *DETUNING_RANGE, (2,)),
                _uniform(next(keys), *XZ_RANGE, (1,)),
            ]
        ),
    )

    new(
        "2q-exchange-resonant",
        "static XY exchange between resonant qubits",
        2,
        _decoherence_terms(2) + (exchange_term(0, 1, 2),),
        jnp.concatenate([_decoherence_values(next(keys), 2), _uniform(next(keys), *EXCHANGE_RANGE, (1,))]),
    )

    spacing = float(_uniform(next(keys), *QUBIT_SPACING_RANGE))
    new(
        "2q-exchange-detuned",
        "time-dependent XY exchange between qubits at different frequencies",
        2,
        _decoherence_terms(2) + (exchange_term(0, 1, 2, frequency=spacing),),
        jnp.concatenate([_decoherence_values(next(keys), 2), _uniform(next(keys), *EXCHANGE_RANGE, (1,))]),
    )

    # ---- four qubits (square lattice) --------------------------------------
    new(
        "4q-decoherence",
        "independent T1/T2 across a four-qubit square",
        4,
        _decoherence_terms(4),
        _decoherence_values(next(keys), 4),
    )

    new(
        "4q-detuning-zz-ring",
        "detunings plus nearest-neighbour ZZ on the square (not all-to-all)",
        4,
        _decoherence_terms(4)
        + tuple(detuning_term(q, 4) for q in range(4))
        + tuple(zz_term(i, j, 4) for i, j in SQUARE_EDGES),
        jnp.concatenate(
            [
                _decoherence_values(next(keys), 4),
                _uniform(next(keys), *DETUNING_RANGE, (4,)),
                _uniform(next(keys), *ZZ_RANGE, (4,)),
            ]
        ),
    )

    new(
        "4q-driven-ring",
        "all four qubits driven, with detunings and nearest-neighbour ZZ",
        4,
        _decoherence_terms(4)
        + tuple(detuning_term(q, 4) for q in range(4))
        + tuple(zz_term(i, j, 4) for i, j in SQUARE_EDGES)
        + tuple(drive_term(q, 4) for q in range(4)),
        jnp.concatenate(
            [
                _decoherence_values(next(keys), 4),
                _uniform(next(keys), *DETUNING_RANGE, (4,)),
                _uniform(next(keys), *ZZ_RANGE, (4,)),
                2 * jnp.pi / _uniform(next(keys), *RABI_PERIOD_RANGE, (4,)),
            ]
        ),
    )

    # Commensurate frequency spacings keep the modulated generator periodic (integer multiples of a base).
    base = float(_uniform(next(keys), *QUBIT_SPACING_RANGE)) / 4.0
    multiples = (1, 2, 3, 4)
    new(
        "4q-exchange-ring-timedep",
        "time-dependent XY exchange on every edge of the square",
        4,
        _decoherence_terms(4)
        + tuple(detuning_term(q, 4) for q in range(4))
        + tuple(exchange_term(i, j, 4, frequency=base * m) for (i, j), m in zip(SQUARE_EDGES, multiples)),
        jnp.concatenate(
            [
                _decoherence_values(next(keys), 4),
                _uniform(next(keys), *DETUNING_RANGE, (4,)),
                _uniform(next(keys), *EXCHANGE_RANGE, (4,)),
            ]
        ),
    )

    return tuple(cases)


# ---------------------------------------------------------------------------
# Experiments: a case plus the imperfections we sweep over.
# ---------------------------------------------------------------------------


def _pauli_eigenstates() -> Array:
    """The six single-qubit Pauli eigenstate density matrices, shape ``(6, 2, 2)``."""
    labels = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
    return jnp.stack([qx.promote_state_vector_to_density_matrix(qx.states.PAULI_STATES[k]).matrix for k in labels])


def product_states(num_qubits: int) -> Array:
    """All ``6**num_qubits`` product Pauli eigenstates (qubit 0 most significant)."""
    single = _pauli_eigenstates()
    product = single
    for _ in range(num_qubits - 1):
        count, dim = product.shape[0], product.shape[1]
        product = jnp.einsum("pij,qkl->pqikjl", product, single).reshape(count * 6, dim * 2, dim * 2)
    return product


def pauli_observables(num_qubits: int) -> Array:
    """All ``4**num_qubits`` Pauli-string observables (base-4 order over I, X, Y, Z)."""
    return qx.ensembles.n_qubit_pauli_operators(num_qubits).matrix


@dataclass(frozen=True)
class Experiment:
    """A learning case exposed as measurable data, with optional shot noise and turn-on ramp.

    The learner is given the model *structure* and the initial guess, and may request measurements at
    times of its choosing via :meth:`measure`; it is never told the ramp duration or the true
    coefficients.

    :param case: The underlying learning case.
    :param shots: Shots per measurement setting, or ``None`` for noiseless expectation values.
    :param ramp: Duration of an unknown turn-on transient before ``t = 0`` of the measurement window.
    :param seed: Seed for the shot noise and the (unknown) ramp generator.
    :param ramp_profile: ``"transient"`` for the adversarial default (every coefficient mis-scaled plus
        strong extra dephasing), or ``"coherent"`` for the physically realistic case: the Hamiltonian
        rising linearly from zero to full strength while the dissipators act throughout.
    """

    case: LearningCase
    shots: int | None = None
    ramp: float = 0.0
    seed: int = 0
    ramp_profile: str = "transient"
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        return self.case.num_qubits

    @property
    def noise_sigma(self) -> float:
        """Standard deviation of a single expectation-value estimate."""
        return 0.0 if self.shots is None else float(1.0 / jnp.sqrt(self.shots))

    def _initial_states(self) -> Array:
        """Product Pauli eigenstates, pushed through the unknown ramp if there is one."""
        if "states" not in self._cache:
            states = product_states(self.num_qubits)
            if self.ramp > 0.0:
                count, dim = states.shape[0], states.shape[1]
                vectors = states.transpose(0, 2, 1).reshape(count, -1)  # column-stacked vec
                evolved = vectors @ self._ramp_propagator().T
                states = evolved.reshape(count, dim, dim).transpose(0, 2, 1)
            self._cache["states"] = states
        return self._cache["states"]

    def _ramp_propagator(self) -> Array:
        """The turn-on transient, either adversarial (default) or a realistic coherent ramp."""
        model, truth = self.case.model, self.case.true_coefficients
        if self.ramp_profile == "coherent":
            # The physical picture: the Hamiltonian rises linearly to full strength over `ramp` while
            # decoherence acts the whole time.  Time-ordered, since the generator is time dependent.
            coherent = jnp.array([term.coherent for term in model.terms])
            steps = 64
            step = self.ramp / steps
            propagator = jnp.eye(4**self.num_qubits, dtype=complex)
            for index in range(steps):
                fraction = (index + 0.5) / steps
                scaled = jnp.where(coherent, truth * fraction, truth)
                propagator = jax.scipy.linalg.expm(step * model.generator_at(scaled)) @ propagator
            return propagator
        k_scale, k_extra = jax.random.split(jax.random.key(self.seed + 991))
        scale = _uniform(k_scale, 0.3, 0.8, truth.shape)
        extra_rate = float(_uniform(k_extra, 0.5, 2.0))
        extra = reduce(
            lambda a, b: a + b,
            [dephasing_term(q, self.num_qubits).generator for q in range(self.num_qubits)],
        )
        generator = model.generator_at(truth * scale, 0.0) + extra_rate * extra
        return jax.scipy.linalg.expm(self.ramp * generator)

    def measure(self, times: Array) -> Array:
        """Measure every Pauli observable on every product Pauli eigenstate at the requested times.

        :param times: Times (microseconds) relative to the start of the measurement window.
        :return: Real array of shape ``(len(times), 6**n, 4**n)`` of expectation values.
        """
        times = jnp.atleast_1d(jnp.asarray(times, dtype=float))
        states = self._initial_states()
        observables = pauli_observables(self.num_qubits)
        propagators = self.case.model.propagators(self.case.true_coefficients, times)

        state_vectors = states.transpose(0, 2, 1).reshape(states.shape[0], -1)
        observable_vectors = observables.transpose(0, 2, 1).reshape(observables.shape[0], -1)
        values = jnp.real(jnp.einsum("or,trs,ks->tko", observable_vectors.conj(), propagators, state_vectors))

        if self.shots is not None:
            key = jax.random.fold_in(jax.random.key(self.seed), times.shape[0])
            noise = self.noise_sigma * jax.random.normal(key, values.shape)
            # The all-identity observable is fixed by normalisation and is never actually measured.
            noise = noise.at[:, :, 0].set(0.0)
            values = values + noise
        return values


def experiment_variants(case: LearningCase, seed: int = 0) -> tuple[Experiment, ...]:
    """The standard sweep of imperfections for a case: shot noise x turn-on ramp."""
    return tuple(
        Experiment(case=case, shots=shots, ramp=ramp, seed=seed)
        for shots, ramp in itertools.product((None, 100_000, 4_096), (0.0, 0.2))
    )
