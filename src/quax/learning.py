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

r"""Structured Lindbladian models for learning generators from data.

A :class:`LindbladModel` is a **linear** ansatz for a GKSL generator: a fixed list of unit
:class:`Term` generators together with a vector of coefficients,

.. math:: \mathcal{L}(\theta, t) = \sum_k \theta_k \left[\cos(\omega_k t)\,G_k + \sin(\omega_k t)\,Q_k\right],

where :math:`G_k` is the term's unit generator and :math:`(\omega_k, Q_k)` describe an optional
periodic modulation (used for transverse couplings between qubits at different frequencies, which
become time dependent in the rotating frame).  Static terms have :math:`\omega_k = 0` and
:math:`Q_k = 0`.

Because the generator is *linear* in ``theta``, the short-time derivative of any observable is linear
in the coefficients too — the basis for convex generator estimation — while the finite-time
propagator :math:`e^{t\mathcal{L}}` is exact at any step and gives access to slow dynamics.

Units convention: times are microseconds, so dissipative rates are in :math:`\mu s^{-1}` and coherent
couplings are angular frequencies in :math:`\mathrm{rad}\,\mu s^{-1}` (i.e. :math:`2\pi f`).
"""

import itertools
import math
from dataclasses import dataclass, field
from functools import reduce
from math import prod

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Lindbladian, Observable, Operator
from .gates import X, Y, Z

__all__ = [
    "LearningResult",
    "LindbladModel",
    "Term",
    "amplitude_damping_term",
    "coherent_term",
    "dephasing_term",
    "detuning_term",
    "dissipative_term",
    "drive_term",
    "embed_operator",
    "exchange_term",
    "excitation_term",
    "joint_decay_term",
    "learn",
    "learn_local",
    "measurement_schedule",
    "neighbourhood",
    "patch_model",
    "rebuild_term",
    "xz_term",
    "zz_term",
]

_SIGMA_MINUS = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # |0><1|
_SIGMA_PLUS = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)  # |1><0|


def _embed(operators: dict[int, Array], num_qubits: int) -> Array:
    """Tensor single-qubit matrices into an ``num_qubits``-qubit matrix (identity elsewhere)."""
    eye = jnp.eye(2, dtype=complex)
    return reduce(jnp.kron, [operators.get(k, eye) for k in range(num_qubits)])


def _dims(num_qubits: int) -> tuple[int, ...]:
    return (2,) * num_qubits


def coherent_term(
    name: str,
    hamiltonian: Array,
    num_qubits: int,
    frequency: float = 0.0,
    quadrature_hamiltonian: Array | None = None,
    support: tuple[int, ...] = (),
    local_hamiltonian: Array | None = None,
    local_quadrature: Array | None = None,
) -> "Term":
    """Build a purely coherent unit term from a Hamiltonian matrix.

    :param name: Human-readable label for the term.
    :param hamiltonian: The (possibly cos-quadrature) Hamiltonian matrix.
    :param num_qubits: Number of qubits the term acts on.
    :param frequency: Modulation angular frequency in rad/us (0 for a static term).
    :param quadrature_hamiltonian: The sin-quadrature Hamiltonian for a modulated term.
    :return: The corresponding :class:`Term`.
    """
    dims = _dims(num_qubits)
    d = prod(dims)
    zero_jump = Operator.from_matrix(jnp.zeros((1, d, d), dtype=complex), (dims, dims))

    def generator(matrix: Array) -> Array:
        return Lindbladian(hamiltonian=Observable.from_matrix(matrix, (dims, dims)), jump_operators=zero_jump).matrix

    quadrature = None if quadrature_hamiltonian is None else generator(quadrature_hamiltonian)
    return Term(
        name=name,
        generator=generator(hamiltonian),
        quadrature=quadrature,
        frequency=frequency,
        coherent=True,
        support=tuple(sorted(support)),
        local_hamiltonian=local_hamiltonian,
        local_quadrature=local_quadrature,
    )


def dissipative_term(
    name: str,
    jump_operator: Array,
    num_qubits: int,
    support: tuple[int, ...] = (),
    local_jump: Array | None = None,
) -> "Term":
    """Build a unit dissipator term ``D[L]`` from a single jump operator ``L``.

    The coefficient multiplying this term is the **rate** ``gamma``, since
    :math:`D[\\sqrt{\\gamma} L] = \\gamma\\, D[L]`.

    :param name: Human-readable label for the term.
    :param jump_operator: The jump operator matrix ``L``.
    :param num_qubits: Number of qubits the term acts on.
    :return: The corresponding :class:`Term`.
    """
    dims = _dims(num_qubits)
    jumps = Operator.from_matrix(jump_operator[jnp.newaxis], (dims, dims))
    return Term(
        name=name,
        generator=Lindbladian(hamiltonian=None, jump_operators=jumps).matrix,
        coherent=False,
        support=tuple(sorted(support)),
        local_jump=local_jump,
    )


def amplitude_damping_term(qubit: int, num_qubits: int) -> "Term":
    """T1 relaxation on ``qubit``: jump operator :math:`|0\\rangle\\langle 1|`."""
    return dissipative_term(
        f"amp_damp[{qubit}]", _embed({qubit: _SIGMA_MINUS}, num_qubits), num_qubits, (qubit,), _SIGMA_MINUS
    )


def excitation_term(qubit: int, num_qubits: int) -> "Term":
    """Thermal excitation on ``qubit``: jump operator :math:`|1\\rangle\\langle 0|`."""
    return dissipative_term(
        f"excite[{qubit}]", _embed({qubit: _SIGMA_PLUS}, num_qubits), num_qubits, (qubit,), _SIGMA_PLUS
    )


def dephasing_term(qubit: int, num_qubits: int) -> "Term":
    """Pure dephasing on ``qubit``: jump operator :math:`Z/\\sqrt{2}` (so the rate matches ``1/T_phi``)."""
    local = Z.matrix / jnp.sqrt(2.0)
    jump = _embed({qubit: Z.matrix}, num_qubits) / jnp.sqrt(2.0)
    return dissipative_term(f"dephase[{qubit}]", jump, num_qubits, (qubit,), local)


def joint_decay_term(first: int, second: int, num_qubits: int) -> "Term":
    """Correlated (collective) decay of a qubit pair: jump operator :math:`\\sigma^-_i + \\sigma^-_j`.

    Unlike two independent dampers this dissipator is *coherent* across the pair — it decays the
    symmetric (bright) combination while leaving the antisymmetric (dark) state untouched.
    """
    eye = jnp.eye(2, dtype=complex)
    local = jnp.kron(_SIGMA_MINUS, eye) + jnp.kron(eye, _SIGMA_MINUS)
    jump = _embed({first: _SIGMA_MINUS}, num_qubits) + _embed({second: _SIGMA_MINUS}, num_qubits)
    return dissipative_term(f"joint_decay[{first},{second}]", jump, num_qubits, (first, second), local)


def detuning_term(qubit: int, num_qubits: int) -> "Term":
    """Residual Z precession of ``qubit``: :math:`H = Z_i/2`, so the coefficient is the angular detuning."""
    return coherent_term(
        f"detuning[{qubit}]",
        0.5 * _embed({qubit: Z.matrix}, num_qubits),
        num_qubits,
        support=(qubit,),
        local_hamiltonian=0.5 * Z.matrix,
    )


def drive_term(qubit: int, num_qubits: int, axis: str = "x") -> "Term":
    """Resonant Rabi drive on ``qubit``: :math:`H = X_i/2` (or ``Y``), coefficient = Rabi rate."""
    matrix = {"x": X.matrix, "y": Y.matrix}[axis]
    return coherent_term(
        f"drive_{axis}[{qubit}]",
        0.5 * _embed({qubit: matrix}, num_qubits),
        num_qubits,
        support=(qubit,),
        local_hamiltonian=0.5 * matrix,
    )


def zz_term(first: int, second: int, num_qubits: int) -> "Term":
    """Static ZZ crosstalk: :math:`H = Z_i Z_j`."""
    return coherent_term(
        f"zz[{first},{second}]",
        _embed({first: Z.matrix, second: Z.matrix}, num_qubits),
        num_qubits,
        support=(first, second),
        local_hamiltonian=jnp.kron(Z.matrix, Z.matrix),
    )


def xz_term(first: int, second: int, num_qubits: int) -> "Term":
    """Classical drive crosstalk: :math:`H = X_i Z_j / 2` — a drive on ``first`` conditioned on ``second``."""
    matrix = 0.5 * _embed({first: X.matrix, second: Z.matrix}, num_qubits)
    local = 0.5 * (jnp.kron(X.matrix, Z.matrix) if first < second else jnp.kron(Z.matrix, X.matrix))
    return coherent_term(f"xz[{first},{second}]", matrix, num_qubits, support=(first, second), local_hamiltonian=local)


def exchange_term(first: int, second: int, num_qubits: int, frequency: float = 0.0) -> "Term":
    """Transverse (XY) exchange coupling :math:`\\tfrac{1}{2}(X_iX_j + Y_iY_j)`.

    With ``frequency = 0`` this is a static, resonant exchange.  For qubits separated by
    :math:`\\Delta = \\omega_i - \\omega_j` the rotating frame makes it **time dependent**,

    .. math:: \\tfrac{1}{2}\\left[(X_iX_j + Y_iY_j)\\cos\\Delta t + (X_iY_j - Y_iX_j)\\sin\\Delta t\\right],

    which is captured by passing ``frequency = Delta``.

    :param first: First qubit index.
    :param second: Second qubit index.
    :param num_qubits: Total number of qubits.
    :param frequency: Modulation angular frequency in rad/us (the qubit frequency difference).
    :return: The corresponding :class:`Term`.
    """
    local_cos = 0.5 * (jnp.kron(X.matrix, X.matrix) + jnp.kron(Y.matrix, Y.matrix))
    local_sin = 0.5 * (jnp.kron(X.matrix, Y.matrix) - jnp.kron(Y.matrix, X.matrix))
    if first > second:  # local operators are ordered by ascending qubit index
        local_sin = -local_sin
    cos_part = 0.5 * (
        _embed({first: X.matrix, second: X.matrix}, num_qubits)
        + _embed({first: Y.matrix, second: Y.matrix}, num_qubits)
    )
    sin_part = 0.5 * (
        _embed({first: X.matrix, second: Y.matrix}, num_qubits)
        - _embed({first: Y.matrix, second: X.matrix}, num_qubits)
    )
    return coherent_term(
        f"exchange[{first},{second}]",
        cos_part,
        num_qubits,
        frequency=frequency,
        quadrature_hamiltonian=sin_part,
        support=(first, second),
        local_hamiltonian=local_cos,
        local_quadrature=local_sin,
    )


@dataclass(frozen=True)
class Term:
    """A single unit generator of a Lindbladian ansatz.

    :param name: Human-readable label, e.g. ``"amp_damp[0]"``.
    :param generator: The ``(d², d²)`` unit generator matrix (the cos quadrature if modulated).
    :param quadrature: The sin-quadrature generator for a modulated term, or ``None``.
    :param frequency: Modulation angular frequency in rad/us; ``0`` marks a static term.
    :param coherent: ``True`` for Hamiltonian terms, ``False`` for dissipators (whose coefficients are
        rates and must stay non-negative).
    :param support: The qubits the term acts on non-trivially; the basis of local-marginal fitting.
    :param local_hamiltonian: The Hamiltonian restricted to ``support`` (coherent terms).
    :param local_quadrature: The sin-quadrature Hamiltonian restricted to ``support``.
    :param local_jump: The jump operator restricted to ``support`` (dissipative terms).

    The ``local_*`` matrices are what make patch-local evaluation possible: they let the term be
    rebuilt on any register containing its support, without ever forming a full-system operator.
    """

    name: str
    generator: Array
    quadrature: Array | None = None
    frequency: float = 0.0
    coherent: bool = True
    support: tuple[int, ...] = ()
    local_hamiltonian: Array | None = None
    local_quadrature: Array | None = None
    local_jump: Array | None = None

    @property
    def is_time_dependent(self) -> bool:
        """Whether this term carries a non-zero modulation frequency."""
        return self.frequency != 0.0


@dataclass(frozen=True)
class LindbladModel:
    """A linear Lindbladian ansatz: an ordered list of unit :class:`Term` generators.

    :param num_qubits: Number of qubits the model acts on.
    :param terms: The unit generators, in the order matching the coefficient vector.
    """

    num_qubits: int
    terms: tuple[Term, ...]
    _stacked: tuple[Array, Array, Array] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        generators = jnp.stack([t.generator for t in self.terms])
        quadratures = jnp.stack(
            [jnp.zeros_like(t.generator) if t.quadrature is None else t.quadrature for t in self.terms]
        )
        frequencies = jnp.array([t.frequency for t in self.terms], dtype=float)
        object.__setattr__(self, "_stacked", (generators, quadratures, frequencies))

    @property
    def names(self) -> tuple[str, ...]:
        """The term labels, in coefficient order."""
        return tuple(t.name for t in self.terms)

    @property
    def size(self) -> int:
        """Number of coefficients (terms) in the model."""
        return len(self.terms)

    @property
    def dim(self) -> int:
        """Hilbert-space dimension ``2**num_qubits``."""
        return 2**self.num_qubits

    @property
    def generators(self) -> Array:
        """Stacked unit generator matrices, shape ``(n_terms, d², d²)``."""
        return self._stacked[0]

    @property
    def quadratures(self) -> Array:
        """Stacked sin-quadrature generators, shape ``(n_terms, d², d²)`` (zeros for static terms)."""
        return self._stacked[1]

    @property
    def frequencies(self) -> Array:
        """Modulation angular frequencies, shape ``(n_terms,)``."""
        return self._stacked[2]

    @property
    def is_time_dependent(self) -> bool:
        """Whether any term is modulated. Pure Python, so it stays usable inside traced code."""
        return any(term.frequency != 0.0 for term in self.terms)

    @property
    def period(self) -> float | None:
        """The common period of the modulated terms, or ``None`` if the model is static.

        Requires the non-zero frequencies to be commensurate (integer multiples of their minimum),
        which is how the modulated test models are constructed.
        """
        active = [abs(term.frequency) for term in self.terms if term.frequency != 0.0]
        if not active:
            return None
        return float(2 * math.pi / min(active))

    def generator_at(self, coefficients: Array, time: float | Array = 0.0) -> Array:
        """The generator matrix ``L(theta, t)``.

        :param coefficients: Coefficient vector of length ``size``.
        :param time: Evaluation time in microseconds (only matters for modulated terms).
        :return: The ``(d², d²)`` generator matrix.
        """
        coefficients = jnp.asarray(coefficients)
        phases = self.frequencies * time
        weighted_cos = coefficients * jnp.cos(phases)
        weighted_sin = coefficients * jnp.sin(phases)
        return jnp.einsum("k,kij->ij", weighted_cos, self.generators) + jnp.einsum(
            "k,kij->ij", weighted_sin, self.quadratures
        )

    def propagators(self, coefficients: Array, times: Array, substeps: int = 256) -> Array:
        """Superoperator propagators ``U(t)`` at each requested time.

        For a static model this is ``expm(t L)``.  For a modulated (periodic) model the time-ordered
        propagator is built by splitting ``t = n T + r``: one period ``U(T)`` is integrated once with a
        product of short-step propagators, and ``U(t) = U(r) · U(T)^n`` follows from periodicity.

        :param coefficients: Coefficient vector of length ``size``.
        :param times: Times in microseconds (may be negative for a model anchored mid-experiment).
        :param substeps: Sub-steps per period used for the time-ordered integration.
        :return: Array of shape ``(len(times), d², d²)``.
        """
        times = jnp.asarray(times, dtype=float)
        if not self.is_time_dependent:
            generator = self.generator_at(coefficients)
            return jax.vmap(lambda t: jax.scipy.linalg.expm(t * generator))(times)

        period = self.period
        assert period is not None
        identity = jnp.eye(self.dim**2, dtype=complex)

        def integrate(duration: Array, steps: int) -> Array:
            """Time-ordered propagator over ``[0, duration)`` using ``steps`` midpoint sub-steps."""
            delta = duration / steps

            def advance(propagator: Array, index: Array) -> tuple[Array, None]:
                matrix = delta * self.generator_at(coefficients, (index + 0.5) * delta)
                step = identity + matrix + matrix @ matrix / 2 + matrix @ matrix @ matrix / 6
                return step @ propagator, None

            result, _ = jax.lax.scan(advance, identity, jnp.arange(steps))
            return result

        period_propagator = integrate(jnp.asarray(period), substeps)
        inverse_period_propagator = jnp.linalg.inv(period_propagator)

        def single(time: float) -> Array:
            cycles = int(time // period)  # concrete: modulated propagators need concrete times
            residual = time - cycles * period
            base = period_propagator if cycles >= 0 else inverse_period_propagator
            whole = jnp.linalg.matrix_power(base, abs(cycles))
            return integrate(jnp.asarray(residual), substeps) @ whole

        return jnp.stack([single(float(t)) for t in times])


# ---------------------------------------------------------------------------
# Learning a generator from measured Pauli tomography.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearningResult:
    """Outcome of :func:`learn`.

    :param coefficients: The fitted coefficient vector, in the model's term order.
    :param times: The absolute times that were measured.
    :param offsets: Those times relative to the anchor.
    :param losses: The residual after each progressive stage.
    """

    coefficients: Array
    times: Array
    offsets: Array
    losses: tuple[float, ...]


def measurement_schedule(
    model: "LindbladModel",
    coefficients: Array,
    *,
    core_points: int = 9,
    tail_points: int = 8,
    phase_resolution: float = 0.05,
    horizon_factor: float = 2.0,
) -> Array:
    """Choose measurement offsets around an anchor: a dense core plus a geometric tail.

    The **core** resolves the fastest coherent scale in the ansatz (spacing set so the fastest phase
    advances by ``phase_resolution`` radians per step), which is what removes frequency aliasing.  The
    **tail** spreads geometrically out to a few decay times, where the slow dissipative rates finally
    leave a signal above the shot noise.  Offsets are symmetric about the anchor, so both sides of it
    are used.

    :param model: The ansatz being learned (its modulation frequencies count as fast scales).
    :param coefficients: A rough guess of the coefficients, used only to set the timescales.
    :param core_points: Number of densely spaced points (odd; includes the anchor itself).
    :param tail_points: Number of geometric points per side.
    :param phase_resolution: Phase (radians) advanced by the fastest term across one core step.  Large
        enough that the signal stands above the shot noise, small enough that a coefficient known to
        ~50% cannot alias across the core.
    :param horizon_factor: Tail extent, in units of the fastest decay time.
    :return: Sorted, de-duplicated offsets in microseconds, always including ``0``.
    """
    magnitudes = jnp.abs(jnp.asarray(coefficients))
    coherent = jnp.array([t.coherent for t in model.terms])
    fast = float(jnp.max(jnp.where(coherent, magnitudes, 0.0), initial=0.0))
    fast = max(fast, float(jnp.max(jnp.abs(model.frequencies), initial=0.0)))
    rates = float(jnp.max(jnp.where(coherent, 0.0, magnitudes), initial=0.0))

    horizon = horizon_factor / rates if rates > 0 else 1.0
    core_step = phase_resolution / fast if fast > 0 else horizon / 100.0
    half = core_points // 2
    core = core_step * jnp.arange(-half, half + 1)
    tail = jnp.geomspace(max(core_step * 10.0, horizon * 1e-3), horizon, tail_points)
    offsets = jnp.concatenate([core, tail, -tail])
    return jnp.unique(jnp.round(offsets, 12))


def _pauli_vectors(num_qubits: int) -> Array:
    """Column-stacked vectorisations of every Pauli string, shape ``(4**n, d²)``."""
    from .ensembles import n_qubit_pauli_operators

    paulis = n_qubit_pauli_operators(num_qubits).matrix
    return paulis.transpose(0, 2, 1).reshape(paulis.shape[0], -1)


def _reconstruct_states(anchor_data: Array, num_qubits: int) -> Array:
    """Linear-inversion tomography of the anchor states: ``rho = (1/d) sum_a <P_a> P_a``.

    :param anchor_data: Expectation values at the anchor, shape ``(n_states, 4**n)``.
    :param num_qubits: Number of qubits.
    :return: Column-stacked state vectors, shape ``(n_states, d²)``.
    """
    return (anchor_data.astype(complex) @ _pauli_vectors(num_qubits)) / (2**num_qubits)


def _reconstruct_propagators(data: Array, state_vectors: Array, num_qubits: int) -> Array:
    """Recover the superoperator taking the anchor states to each measured time.

    Everything is expressed *relative to the anchor*, so an unknown turn-on ramp (or any state
    preparation error) is absorbed into the anchor states and never enters the model.

    :param data: Expectations, shape ``(n_times, n_states, n_observables)``.
    :param state_vectors: Anchor states as column-stacked vectors, shape ``(n_states, d²)``.
    :param num_qubits: Number of qubits.
    :return: Superoperator matrices, shape ``(n_times, d², d²)``.
    """
    observable_vectors = _pauli_vectors(num_qubits)
    left = jnp.linalg.pinv(observable_vectors.conj())
    right = jnp.linalg.pinv(state_vectors.T)
    return jnp.einsum("ro,jko,ks->jrs", left, data.astype(complex), right)


def _coherent_mask(model: "LindbladModel") -> Array:
    return jnp.array([t.coherent for t in model.terms])


def _to_unconstrained(model: "LindbladModel", coefficients: Array) -> Array:
    """Map coefficients to an unconstrained vector (dissipative rates through an inverse softplus)."""
    positive = jnp.maximum(jnp.asarray(coefficients), 1e-12)
    return jnp.where(_coherent_mask(model), coefficients, jnp.log(jnp.expm1(positive)))


def _from_unconstrained(model: "LindbladModel", raw: Array) -> Array:
    """Inverse of :func:`_to_unconstrained`; keeps dissipative rates non-negative."""
    return jnp.where(_coherent_mask(model), raw, jax.nn.softplus(raw))


def _adam(loss_fn, raw: Array, steps: int, learning_rate: float) -> Array:
    """A minimal Adam loop, so the library needs no optimiser dependency."""
    gradient = jax.grad(loss_fn)

    def body(index: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        value, first, second = carry
        grad = gradient(value)
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad**2
        step = index + 1
        corrected_first = first / (1 - 0.9**step)
        corrected_second = second / (1 - 0.999**step)
        return value - learning_rate * corrected_first / (jnp.sqrt(corrected_second) + 1e-12), first, second

    result, _, _ = jax.lax.fori_loop(0, steps, body, (raw, jnp.zeros_like(raw), jnp.zeros_like(raw)))
    return result


def learn(
    model: "LindbladModel",
    measure,
    initial_guess: Array,
    *,
    settling: float = 0.5,
    noise_floor: float = 0.0,
    offsets: Array | None = None,
    stages: int = 8,
    steps: int = 400,
    learning_rate: float = 0.03,
    **schedule_options,
) -> LearningResult:
    """Learn a Lindbladian's coefficients from Pauli tomography of its dynamics.

    The learner chooses *when* to measure (see :func:`measurement_schedule`) and consumes the complete
    set of product Pauli eigenstates and Pauli-string observables at those times.  It then

    1. reconstructs the **anchor states** by linear inversion, so all subsequent modelling is relative
       to measured states — this is what makes it robust to an unknown turn-on ramp and to state
       preparation errors;
    2. reconstructs the **propagator** to each measured time from the complete tomographic data;
    3. fits the ansatz to those propagators with a **progressive** widening of the time window: short
       windows have a broad basin of convergence and lock the fast coherent terms, and each widening
       brings in slower dynamics while warm-starting from the previous stage.

    :param model: The (structurally correct) ansatz to fit.
    :param measure: Callable taking an array of times and returning ``(n_times, 6**n, 4**n)`` expectations.
    :param initial_guess: Starting coefficients; also sets the measurement timescales.
    :param settling: Delay before the first measurement, covering any unknown turn-on transient (us).
    :param noise_floor: Expected measurement noise per expectation value, i.e. ``1/sqrt(shots)``; leave
        at 0 for noiseless data.  Used to discard baselines whose signal has decayed into the noise.
    :param offsets: Explicit measurement offsets, overriding the automatic schedule.
    :param stages: Number of progressive window-widening stages.
    :param steps: Optimiser iterations per stage.
    :param learning_rate: Adam step size.
    :param schedule_options: Extra keyword arguments forwarded to :func:`measurement_schedule`.
    :return: The fitted coefficients and the measurement times that were used.
    """
    initial_guess = jnp.asarray(initial_guess, dtype=float)
    if offsets is None:
        offsets = measurement_schedule(model, initial_guess, **schedule_options)
    offsets = jnp.asarray(offsets, dtype=float)

    times = settling + float(jnp.max(offsets)) + offsets
    data = jnp.asarray(measure(times))

    anchor_index = int(jnp.argmin(jnp.abs(offsets)))
    state_vectors = _reconstruct_states(data[anchor_index], model.num_qubits)
    targets = _reconstruct_propagators(data, state_vectors, model.num_qubits)

    if model.is_time_dependent:
        coefficients, losses = _fit_by_propagator_matching(
            model, targets, offsets, initial_guess, stages=stages, steps=steps, learning_rate=learning_rate
        )
    else:
        coefficients, losses = _fit_by_generator_logarithm(
            model, targets, offsets, initial_guess, rounds=stages, noise_floor=noise_floor
        )
        if noise_floor > 0.0:
            # The logarithm inverts the measured states, which amplifies shot noise badly for the fast
            # coherent terms.  Use it only as a candidate start, then fit the measured expectations
            # directly -- a forward model, so noise is never amplified.
            candidates = (coefficients, jnp.asarray(initial_guess, dtype=float))
            best = min(candidates, key=lambda c: _propagator_residual(model, c, offsets, targets))
            coefficients, refinement = _fit_by_local_marginals(
                model,
                data,
                offsets,
                state_vectors,
                best,
                stages=stages,
                steps=steps,
                learning_rate=learning_rate,
            )
            losses = losses + refinement

    return LearningResult(coefficients=coefficients, times=times, offsets=offsets, losses=tuple(losses))


def _solve_coefficients(model: "LindbladModel", generator: Array, floor: Array | None = None) -> Array:
    """Least-squares projection of a generator matrix onto the model's unit generators.

    Dissipative coefficients are kept strictly positive: clipping one to exactly zero would trap the
    subsequent relative-scale refinement, which can only rescale what it is given.
    """
    basis = model.generators.reshape(model.size, -1)
    design = jnp.concatenate([basis.real, basis.imag], axis=1).T
    target = jnp.concatenate([generator.reshape(-1).real, generator.reshape(-1).imag])
    coefficients, *_ = jnp.linalg.lstsq(design, target, rcond=None)
    lower = jnp.full_like(coefficients, 1e-9) if floor is None else jnp.abs(floor) * 1e-3
    return jnp.where(_coherent_mask(model), coefficients, jnp.maximum(coefficients, lower))


def _fit_by_generator_logarithm(
    model: "LindbladModel",
    propagators: Array,
    offsets: Array,
    initial_guess: Array,
    rounds: int = 4,
    noise_floor: float = 0.0,
) -> tuple[Array, list[float]]:
    """Estimate a static generator by inverting ``S = exp(tau L)`` with a branch-corrected logarithm.

    A principal matrix logarithm is only correct while the fastest phase satisfies
    ``|Im(lambda) tau| < pi``; beyond that the coherent eigenvalues alias.  Projecting each measured
    propagator onto the eigenbasis of a *reference* generator lets us pick the right branch for every
    eigenvalue, so long baselines — where the slow dissipative rates finally have signal — become
    usable.  Each round re-derives the reference from the previous estimate, and the estimates are
    combined with inverse-variance weights ``tau**2`` (logarithm noise scales as ``1/tau``).

    :param model: The ansatz being fitted.
    :param propagators: Measured propagators, shape ``(n_times, d², d²)``.
    :param offsets: The corresponding time offsets.
    :param initial_guess: Starting coefficients, used to seed the first reference generator.
    :param rounds: Number of reference-refinement rounds.
    :param noise_floor: Expected measurement noise per expectation value (``1/sqrt(shots)``); baselines
        whose weakest mode has decayed below it carry no information and are dropped.
    :return: The fitted coefficients and the residual after each round.
    """
    taus = offsets[jnp.abs(offsets) > 0]
    measured = propagators[jnp.abs(offsets) > 0]
    magnitudes = jnp.abs(taus)

    # Diagonalise every measured propagator once; the loop below only re-selects logarithm branches.
    spectra = [jnp.linalg.eig(matrix) for matrix in measured]
    inverses = [jnp.linalg.inv(vectors) for _, vectors in spectra]

    # A mode that has decayed into the noise carries no information but a huge logarithmic error,
    # so drop any baseline whose weakest mode has fallen below the measurement floor.
    floor = max(10.0 * noise_floor, 1e-3)
    survives = jnp.array([bool(jnp.min(jnp.abs(values)) > floor) for values, _ in spectra])
    if not bool(jnp.any(survives)):
        survives = magnitudes <= jnp.min(magnitudes)

    coefficients = jnp.asarray(initial_guess, dtype=float)
    losses: list[float] = []
    # Widen the window progressively: a short baseline has an unambiguous logarithm even from a poor
    # reference, and each widened stage starts from the sharper estimate the previous one produced.
    for cutoff in jnp.quantile(magnitudes, jnp.linspace(1.0 / rounds, 1.0, rounds)):
        included = [int(i) for i in jnp.nonzero((magnitudes <= cutoff) & survives)[0]]
        if not included:
            continue
        for _ in range(2):
            reference = model.generator_at(coefficients)
            reference_values, reference_vectors = jnp.linalg.eig(reference)
            reference_inverse = jnp.linalg.inv(reference_vectors)

            estimates, weights = [], []
            for index in included:
                tau = taus[index]
                eigenvalues, vectors = spectra[index]
                # Pair each measured eigenvector with a reference one, then undo the 2*pi wrapping.
                overlap = jnp.abs(reference_inverse @ vectors)
                partner = jnp.argmax(overlap, axis=0)
                principal = jnp.log(eigenvalues)
                winding = jnp.round((reference_values.imag[partner] * tau - principal.imag) / (2 * jnp.pi))
                unwrapped = (principal + 2j * jnp.pi * winding) / tau
                estimates.append(vectors @ jnp.diag(unwrapped) @ inverses[index])
                # Logarithm noise scales as 1/(tau * |mu|): weight by the inverse variance.
                weights.append(float(tau**2 * jnp.min(jnp.abs(eigenvalues)) ** 2))

            stacked = jnp.stack(estimates)
            weight = jnp.array(weights)
            # Reject baselines whose logarithm branch went astray: they would otherwise dominate.
            deviation = jnp.array([float(jnp.linalg.norm(item - reference)) for item in stacked])
            keep = deviation <= 3.0 * jnp.median(deviation) + 1e-12
            weight = jnp.where(keep, weight, 0.0)
            if float(jnp.sum(weight)) <= 0.0:
                weight = jnp.ones_like(weight)

            average = jnp.einsum("j,jab->ab", weight / jnp.sum(weight), stacked)
            coefficients = _solve_coefficients(model, average, floor=initial_guess)

        losses.append(float(jnp.mean(jnp.abs(model.generator_at(coefficients) - average) ** 2)))
    return coefficients, losses


def _marginalise_data(data: Array, support: tuple[int, ...], num_qubits: int) -> Array:
    """Reduce complete tomography ``(T, 6**n, 4**n)`` to the experiment on ``support``.

    Averaging a qubit over its six Pauli eigenstates is exactly the maximally mixed state, so averaging
    the complement's preparations gives the data for inputs ``rho_S (x) (I/2)**env``; keeping only the
    observables that are identity outside ``support`` restricts the measurement to that support.
    """
    complement = [k for k in range(num_qubits) if k not in support]
    reshaped = data.reshape((data.shape[0],) + (6,) * num_qubits + (4,) * num_qubits)
    reshaped = reshaped.mean(axis=tuple(1 + k for k in complement), keepdims=True)
    selector: list[slice] = [slice(None)] * reshaped.ndim
    for k in complement:
        selector[1 + num_qubits + k] = slice(0, 1)
    reshaped = reshaped[tuple(selector)]
    drop = tuple(1 + k for k in complement) + tuple(1 + num_qubits + k for k in complement)
    reshaped = jnp.squeeze(reshaped, axis=drop)
    return reshaped.reshape(data.shape[0], 6 ** len(support), 4 ** len(support))


def _marginalise_states(states: Array, support: tuple[int, ...], num_qubits: int) -> Array:
    """Average reconstructed anchor states ``(6**n, d²)`` over the complement's preparations.

    Applying the *same* linear average to the states and to the data keeps the estimator exact -- the
    model is linear in its input state -- and so it stays valid even after an unknown turn-on ramp,
    where the anchor states are no longer products.
    """
    complement = [k for k in range(num_qubits) if k not in support]
    reshaped = states.reshape((6,) * num_qubits + (states.shape[-1],))
    reshaped = reshaped.mean(axis=tuple(complement), keepdims=True)
    reshaped = jnp.squeeze(reshaped, axis=tuple(complement))
    return reshaped.reshape(6 ** len(support), states.shape[-1])


def _support_observable_indices(support: tuple[int, ...], num_qubits: int) -> Array:
    """Indices of the Pauli strings that act as identity outside ``support`` (base-4 ordering)."""
    indices = []
    for combination in itertools.product(range(4), repeat=len(support)):
        assignment = dict(zip(support, combination))
        index = 0
        for qubit in range(num_qubits):
            index = index * 4 + assignment.get(qubit, 0)
        indices.append(index)
    return jnp.array(indices)


def _fit_by_local_marginals(
    model: "LindbladModel",
    data: Array,
    offsets: Array,
    anchor_states: Array,
    start: Array,
    *,
    stages: int,
    steps: int,
    learning_rate: float,
    coherent_periods: float = 40.0,
) -> tuple[Array, list[float]]:
    """Fit each term against the reduced dynamics on its own support.

    A ``k``-local term only shows up in the reduced dynamics of the qubits it acts on, so its residual
    can be evaluated from the ``6**k x 4**k`` tomography of that support with the rest of the register
    maximally mixed.  Compared with matching the whole ``6**n x 4**n`` data set this is

    * far better conditioned -- a weak two-qubit coupling is no longer one small signal buried among
      exponentially many irrelevant ones, and
    * far cheaper -- the number of settings entering the objective grows linearly in the number of
      terms rather than exponentially in the register size.

    The marginals are formed by averaging the complete data over the complement's preparations, and the
    *same* average is applied to the reconstructed anchor states, so the estimator stays exact (the
    model is linear in its input state) and remains valid after an unknown turn-on ramp.
    """
    num_qubits = model.num_qubits
    supports = sorted({term.support for term in model.terms if term.support})
    if not supports:
        supports = [tuple(range(num_qubits))]
    pauli_vectors = _pauli_vectors(num_qubits)

    local_states, local_observables, local_targets = [], [], []
    for support in supports:
        local_states.append(_marginalise_states(anchor_states, support, num_qubits))
        local_observables.append(pauli_vectors[_support_observable_indices(support, num_qubits)])
        local_targets.append(_marginalise_data(data, support, num_qubits))

    start = jnp.where(jnp.abs(start) > 1e-12, start, 1e-12)
    magnitudes = jnp.abs(offsets)
    scaled = jnp.zeros_like(start)
    losses: list[float] = []

    coherent = _coherent_mask(model)
    fastest = float(jnp.max(jnp.where(coherent, jnp.abs(start), 0.0), initial=0.0))
    fastest = max(fastest, float(jnp.max(jnp.abs(model.frequencies), initial=0.0)))
    coherent_horizon = float("inf") if fastest <= 0.0 else coherent_periods * 2 * jnp.pi / fastest

    for cutoff in jnp.quantile(magnitudes, jnp.linspace(1.0 / stages, 1.0, stages)):
        keep = jnp.nonzero(magnitudes <= cutoff)[0]
        kept_offsets = offsets[keep]
        kept_targets = [target[keep] for target in local_targets]
        free = jnp.ones_like(scaled) if float(cutoff) <= coherent_horizon else (~coherent).astype(scaled.dtype)

        def objective(
            value: Array,
            kept_offsets: Array = kept_offsets,
            kept_targets: list = kept_targets,
            free: Array = free,
            frozen: Array = scaled,
        ) -> Array:
            active = jnp.where(free > 0, value, frozen)
            propagators = model.propagators(start * jnp.exp(active), kept_offsets)
            total = 0.0
            for observables, states, target in zip(local_observables, local_states, kept_targets):
                predicted = jnp.real(jnp.einsum("ar,trs,ks->tka", observables.conj(), propagators, states))
                total = total + jnp.mean((predicted - target) ** 2)
            return total / len(kept_targets)

        updated = _adam(objective, scaled, steps, learning_rate)
        scaled = jnp.where(free > 0, updated, scaled)
        losses.append(float(objective(scaled)))
    return start * jnp.exp(scaled), losses


def _fit_by_data_matching(
    model: "LindbladModel",
    data: Array,
    offsets: Array,
    anchor_states: Array,
    start: Array,
    *,
    stages: int,
    steps: int,
    learning_rate: float,
    coherent_periods: float = 40.0,
) -> tuple[Array, list[float]]:
    """Fit the ansatz to the *measured expectation values* themselves.

    This is the noise-robust objective.  Reconstructing propagators requires inverting the measured
    states, which amplifies shot noise; here the anchor states enter the forward model only, so the
    residual is a plain least-squares fit of what was actually measured.

    As in :func:`_fit_by_propagator_matching`, the coefficients are scaled relatively (one step size for
    coefficients spanning orders of magnitude) and the window is widened progressively.
    """
    observable_vectors = _pauli_vectors(model.num_qubits)
    start = jnp.where(jnp.abs(start) > 1e-12, start, 1e-12)
    magnitudes = jnp.abs(offsets)
    scaled = jnp.zeros_like(start)
    losses: list[float] = []

    # A coherent coefficient can only be refined safely while the accumulated phase stays unambiguous.
    # Past a few tens of periods the landscape is rugged and gradient descent locks onto a subharmonic
    # (typically half the true frequency), so beyond that horizon the coherent terms are frozen and the
    # long baselines -- which exist to give the slow rates signal -- move the dissipative terms only.
    coherent = _coherent_mask(model)
    fastest = float(jnp.max(jnp.where(coherent, jnp.abs(start), 0.0), initial=0.0))
    fastest = max(fastest, float(jnp.max(jnp.abs(model.frequencies), initial=0.0)))
    coherent_horizon = float("inf") if fastest <= 0.0 else coherent_periods * 2 * jnp.pi / fastest

    for cutoff in jnp.quantile(magnitudes, jnp.linspace(1.0 / stages, 1.0, stages)):
        keep = jnp.nonzero(magnitudes <= cutoff)[0]
        kept_offsets, kept_data = offsets[keep], data[keep]
        free = jnp.ones_like(scaled) if float(cutoff) <= coherent_horizon else (~coherent).astype(scaled.dtype)

        def objective(
            value: Array,
            kept_offsets: Array = kept_offsets,
            kept_data: Array = kept_data,
            free: Array = free,
            frozen: Array = scaled,
        ) -> Array:
            active = jnp.where(free > 0, value, frozen)
            propagators = model.propagators(start * jnp.exp(active), kept_offsets)
            predicted = jnp.real(jnp.einsum("ar,trs,ks->tka", observable_vectors.conj(), propagators, anchor_states))
            return jnp.mean((predicted - kept_data) ** 2)

        updated = _adam(objective, scaled, steps, learning_rate)
        scaled = jnp.where(free > 0, updated, scaled)
        losses.append(float(objective(scaled)))
    return start * jnp.exp(scaled), losses


def _propagator_residual(model: "LindbladModel", coefficients: Array, offsets: Array, targets: Array) -> float:
    """Mean squared difference between the model's propagators and the measured ones."""
    return float(jnp.mean(jnp.abs(model.propagators(coefficients, offsets) - targets) ** 2))


def _fit_by_propagator_matching(
    model: "LindbladModel",
    targets: Array,
    offsets: Array,
    start: Array,
    *,
    stages: int,
    steps: int,
    learning_rate: float,
) -> tuple[Array, list[float]]:
    """Least-squares fit of the measured propagators, widening the time window in stages.

    Two details make this work across the dynamic range of a real device:

    * a **relative** parameterisation ``theta = start * exp(u)``, so one step size is meaningful for
      coefficients spanning three orders of magnitude, and signs (hence positive rates) are preserved;
    * **progressive widening** -- a short window has a broad basin of convergence and locks the fast
      coherent terms, and each widening adds slower dynamics while warm-starting from the last stage.

    Unlike the generator logarithm this never inverts the data, so it degrades gracefully with noise.
    """
    start = jnp.where(jnp.abs(start) > 1e-12, start, 1e-12)
    magnitudes = jnp.abs(offsets)
    scaled = jnp.zeros_like(start)
    losses: list[float] = []

    for cutoff in jnp.quantile(magnitudes, jnp.linspace(1.0 / stages, 1.0, stages)):
        keep = jnp.nonzero(magnitudes <= cutoff)[0]
        kept_offsets, kept_targets = offsets[keep], targets[keep]

        def objective(value: Array, kept_offsets: Array = kept_offsets, kept_targets: Array = kept_targets) -> Array:
            predicted = model.propagators(start * jnp.exp(value), kept_offsets)
            return jnp.mean(jnp.abs(predicted - kept_targets) ** 2)

        scaled = _adam(objective, scaled, steps, learning_rate)
        losses.append(float(objective(scaled)))
    return start * jnp.exp(scaled), losses


def embed_operator(matrix: Array, positions: tuple[int, ...], num_qubits: int) -> Array:
    """Place a ``len(positions)``-qubit operator at ``positions`` of a ``num_qubits`` register.

    :param matrix: Operator on ``len(positions)`` qubits, in that order.
    :param positions: Target qubit indices, ascending.
    :param num_qubits: Size of the destination register.
    :return: The embedded ``(2**num_qubits, 2**num_qubits)`` operator.
    """
    count = len(positions)
    padded = jnp.kron(matrix, jnp.eye(2 ** (num_qubits - count), dtype=complex))
    order = list(positions) + [q for q in range(num_qubits) if q not in positions]
    permutation = [order.index(q) for q in range(num_qubits)]
    tensor = padded.reshape((2,) * (2 * num_qubits))
    tensor = jnp.transpose(tensor, permutation + [num_qubits + p for p in permutation])
    return tensor.reshape(2**num_qubits, 2**num_qubits)


def rebuild_term(term: "Term", positions: tuple[int, ...], num_qubits: int) -> "Term":
    """Rebuild ``term`` on a smaller register, acting at ``positions``.

    This is what lets a ``k``-local term be evaluated on a patch rather than on the whole device: the
    term is reconstructed from its stored local operators, so nothing of size ``4**n`` is ever formed.
    """
    if term.coherent:
        assert term.local_hamiltonian is not None, f"{term.name} has no local Hamiltonian to rebuild from"
        quadrature = (
            None if term.local_quadrature is None else embed_operator(term.local_quadrature, positions, num_qubits)
        )
        return coherent_term(
            term.name,
            embed_operator(term.local_hamiltonian, positions, num_qubits),
            num_qubits,
            frequency=term.frequency,
            quadrature_hamiltonian=quadrature,
            support=positions,
        )
    assert term.local_jump is not None, f"{term.name} has no local jump operator to rebuild from"
    return dissipative_term(term.name, embed_operator(term.local_jump, positions, num_qubits), num_qubits, positions)


def patch_model(model: "LindbladModel", patch: tuple[int, ...]) -> tuple["LindbladModel", list[int]]:
    """Restrict a model to a patch of qubits.

    Terms whose support lies inside ``patch`` are rebuilt on the (re-indexed) patch register; terms
    straddling the boundary are dropped, which is the light-cone truncation.

    :param model: The full-device ansatz.
    :param patch: Qubit indices forming the patch, ascending.
    :return: The patch model and the indices (into the full coefficient vector) of its terms.
    """
    patch = tuple(sorted(patch))
    relabel = {q: i for i, q in enumerate(patch)}
    terms, indices = [], []
    for index, term in enumerate(model.terms):
        if term.support and set(term.support).issubset(patch):
            positions = tuple(relabel[q] for q in term.support)
            terms.append(rebuild_term(term, positions, len(patch)))
            indices.append(index)
    return LindbladModel(num_qubits=len(patch), terms=tuple(terms)), indices


def neighbourhood(model: "LindbladModel", support: tuple[int, ...], radius: int) -> tuple[int, ...]:
    """Qubits within ``radius`` couplings of ``support``, following the model's interaction graph."""
    adjacency: dict[int, set[int]] = {q: set() for q in range(model.num_qubits)}
    for term in model.terms:
        for a in term.support:
            for b in term.support:
                if a != b:
                    adjacency[a].add(b)
    frontier = set(support)
    for _ in range(radius):
        frontier = frontier | {n for q in frontier for n in adjacency[q]}
    return tuple(sorted(frontier))


def _local_basis(num_qubits: int) -> tuple[Array, Array]:
    """Product Pauli eigenstates and Pauli-string observables on a small register, as vectors."""
    labels = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
    from ._promotion import promote_state_vector_to_density_matrix
    from .states import PAULI_STATES

    single = jnp.stack([promote_state_vector_to_density_matrix(PAULI_STATES[k]).matrix for k in labels])
    product = single
    for _ in range(num_qubits - 1):
        count, dim = product.shape[0], product.shape[1]
        product = jnp.einsum("pij,qkl->pqikjl", product, single).reshape(count * 6, dim * 2, dim * 2)
    states = product.transpose(0, 2, 1).reshape(product.shape[0], -1)
    return states, _pauli_vectors(num_qubits)


def _embedded_local_basis(positions: tuple[int, ...], num_qubits: int) -> tuple[Array, Array]:
    """States ``rho_S (x) (I/2)**env`` and observables ``O_S (x) I`` on a ``num_qubits`` register.

    This is the reduced experiment on ``positions`` with everything else maximally mixed -- exactly what
    averaging the complete data over the other qubits' Pauli-eigenstate preparations produces.
    """
    labels = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
    from ._promotion import promote_state_vector_to_density_matrix
    from .states import PAULI_STATES

    single = [promote_state_vector_to_density_matrix(PAULI_STATES[k]).matrix for k in labels]
    half = jnp.eye(2, dtype=complex) / 2
    eye = jnp.eye(2, dtype=complex)
    paulis = [eye, X.matrix, Y.matrix, Z.matrix]

    states = []
    for combination in itertools.product(range(6), repeat=len(positions)):
        assignment = dict(zip(positions, combination))
        states.append(reduce(jnp.kron, [single[assignment[q]] if q in positions else half for q in range(num_qubits)]))
    observables = []
    for combination in itertools.product(range(4), repeat=len(positions)):
        assignment = dict(zip(positions, combination))
        observables.append(
            reduce(jnp.kron, [paulis[assignment[q]] if q in positions else eye for q in range(num_qubits)])
        )
    to_vectors = lambda mats: jnp.stack(mats).transpose(0, 2, 1).reshape(len(mats), -1)
    return to_vectors(states), to_vectors(observables)


def learn_local(
    model: "LindbladModel",
    measure,
    initial_guess: Array,
    *,
    settling: float = 0.0,
    points: int = 5,
    degree: int = 2,
    derivative_step: float | None = None,
) -> LearningResult:
    """Learn the generator from **local** derivatives — scalable in the register size.

    With every qubit outside a support ``T`` maximally mixed (which is what averaging the complete data
    over their Pauli-eigenstate preparations achieves), the short-time derivative of a ``T``-local
    observable depends *only* on terms whose support lies inside ``T``:

    .. math:: \\frac{d}{dt}\\langle O_T\\rangle = \\mathrm{Tr}\\!\\left[O_T\\,\\mathcal{L}_T(\\rho_T)\\right].

    A term straddling the boundary of ``T`` contributes nothing, because its factor outside ``T`` is a
    traceless Pauli and :math:`\\mathrm{Tr}[(\\mathbb{1}/2)P] = 0`; a term disjoint from ``T`` cancels
    identically.  There is therefore **no light cone and no patch approximation** — each term is
    estimated from observables on its own support alone, and nothing larger than ``4**k`` (with ``k``
    the largest term weight) is ever built, whatever the register size.

    Because the generator is linear in the coefficients, stacking the resulting equations over all
    supports gives one convex least-squares problem for the whole coefficient vector.

    :param model: The (structurally correct) ansatz to fit.
    :param measure: Callable taking times and returning ``(n_times, 6**n, 4**n)`` expectations.
    :param initial_guess: Used only to choose the derivative step.
    :param settling: Delay before the first measurement.  Unlike :func:`learn`, this **must stay at 0**
        for the locality argument to hold: it relies on the environment being maximally mixed at the
        reference time, which is true of the freshly prepared states but not after evolution (or after
        a turn-on ramp, which this estimator therefore does not tolerate).
    :param points: Number of closely spaced times used for the derivative estimate.
    :param degree: Degree of the least-squares polynomial used to extract the slope.
    :param derivative_step: Spacing of those times; by default a small fraction of the fastest period.
    :return: The fitted coefficients.
    """
    initial_guess = jnp.asarray(initial_guess, dtype=float)
    coherent = _coherent_mask(model)
    fastest = float(jnp.max(jnp.where(coherent, jnp.abs(initial_guess), 0.0), initial=0.0))
    fastest = max(fastest, float(jnp.max(jnp.abs(model.frequencies), initial=0.0)))
    if derivative_step is None:
        # The step must be small against the fastest timescale in the generator, coherent or not:
        # the truncation error of the slope estimate scales with (rate * step)**degree.
        scale = max(fastest, float(jnp.max(jnp.abs(initial_guess), initial=1.0)))
        derivative_step = 0.002 / scale if scale > 0 else 1e-3

    offsets = derivative_step * jnp.arange(points)
    times = settling + offsets
    data = jnp.asarray(measure(times))

    supports = sorted({term.support for term in model.terms if term.support})
    vandermonde = jnp.stack([offsets**power for power in range(degree + 1)], axis=1)

    design_rows, slope_rows = [], []
    for support in supports:
        # Reconstruct the local input states at the reference time by linear inversion of the marginal.
        local = _marginalise_data(data, support, model.num_qubits)
        width = 2 ** len(support)
        reference_vectors = local[0].astype(complex) @ _pauli_vectors(len(support)) / width
        reference_states = reference_vectors.reshape(-1, width, width).transpose(0, 2, 1)

        # Every term that *overlaps* the support may move its observables, and it is reduced onto the
        # support on the small register S_j u T.  Restricting to terms with support inside T would be
        # wrong for a collective jump operator such as D[sigma^-_i + sigma^-_j], whose expansion
        # contains D[sigma^-_i] -- a piece living entirely inside a single-qubit support.
        columns, indices = [], []
        for index, term in enumerate(model.terms):
            if not term.support or not (set(term.support) & set(support)):
                continue
            register = tuple(sorted(set(support) | set(term.support)))
            relabel = {q: i for i, q in enumerate(register)}
            rebuilt = rebuild_term(term, tuple(relabel[q] for q in term.support), len(register))
            positions = tuple(relabel[q] for q in support)
            _, observables = _embedded_local_basis(positions, len(register))
            # Use the states that were actually *measured* at the reference time, not the ones we
            # intended to prepare: a turn-on ramp rotates them.  A local unitary ramp leaves the
            # environment exactly maximally mixed, so the locality identity still holds with the
            # measured local state in place of the ideal one.
            states = jnp.stack(
                [
                    embed_operator(matrix, positions, len(register)) / 2 ** (len(register) - len(support))
                    for matrix in reference_states
                ]
            )
            states = states.transpose(0, 2, 1).reshape(states.shape[0], -1)
            columns.append(
                jnp.real(jnp.einsum("ar,rs,ks->ka", observables.conj(), rebuilt.generator, states)).reshape(-1)
            )
            indices.append(index)
        if not columns:
            continue
        block = jnp.zeros((columns[0].shape[0], model.size)).at[:, jnp.array(indices)].set(jnp.stack(columns, axis=1))

        # Measured slopes of the same local observables.
        flattened = local.reshape(points, -1)
        coefficients, *_ = jnp.linalg.lstsq(vandermonde, flattened, rcond=None)
        design_rows.append(block)
        slope_rows.append(coefficients[1])

    design = jnp.concatenate(design_rows, axis=0)
    slopes = jnp.concatenate(slope_rows)
    solution, *_ = jnp.linalg.lstsq(design, slopes, rcond=None)
    solution = jnp.where(coherent, solution, jnp.maximum(solution, 0.0))
    residual = float(jnp.mean((design @ solution - slopes) ** 2))
    return LearningResult(coefficients=solution, times=times, offsets=offsets, losses=(residual,))
