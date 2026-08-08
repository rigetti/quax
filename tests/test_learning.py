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

"""Tests for the Lindbladian learning algorithm, driven by the case battery."""

import jax.numpy as jnp
import pytest

from quax.learning import learn, measurement_schedule

from .lindbladian_cases import Experiment, build_cases

CASES = {case.name: case for case in build_cases(seed=7)}
STATIC_SMALL = [c for c in CASES.values() if c.num_qubits <= 2 and not c.model.is_time_dependent]
STATIC_SMALL_IDS = [c.name for c in STATIC_SMALL]


def _worst_error(case, coefficients) -> float:
    return float(jnp.max(case.relative_error(coefficients)))


def test_schedule_spans_the_fast_and_slow_timescales():
    """The schedule resolves the fastest coherent term and still reaches the decay time."""
    case = CASES["1q-driven-detuned"]
    offsets = measurement_schedule(case.model, case.initial_guess)

    fastest = float(jnp.max(jnp.abs(case.true_coefficients[2:])))  # detuning and drive
    slowest_rate = float(jnp.min(jnp.abs(case.true_coefficients[:2])))
    positive = jnp.sort(jnp.abs(offsets)[jnp.abs(offsets) > 0])

    assert jnp.any(offsets == 0.0), "the anchor itself must be measured"
    assert jnp.any(offsets < 0), "sampling should straddle the anchor"
    assert float(positive[0]) * fastest < jnp.pi, "the core must not alias the fastest term"
    assert float(positive[-1]) * slowest_rate > 0.1, "the tail must reach the dissipative timescale"


@pytest.mark.parametrize("case", STATIC_SMALL, ids=STATIC_SMALL_IDS)
def test_recovers_static_generators_from_noiseless_data(case):
    """With exact expectation values every static case is recovered essentially exactly."""
    result = learn(case.model, Experiment(case=case).measure, case.initial_guess)
    assert _worst_error(case, result.coefficients) < 1e-6


@pytest.mark.parametrize("case", STATIC_SMALL, ids=STATIC_SMALL_IDS)
def test_unknown_turn_on_ramp_does_not_degrade_recovery(case):
    """Anchoring on measured states makes the fit immune to an unknown turn-on transient."""
    experiment = Experiment(case=case, ramp=0.2, seed=5)
    result = learn(case.model, experiment.measure, case.initial_guess)
    assert _worst_error(case, result.coefficients) < 1e-6


def test_learner_improves_on_its_starting_guess():
    """The fit must actually move: the result is far closer to the truth than the initial guess."""
    case = CASES["2q-driven-zz"]
    result = learn(case.model, Experiment(case=case).measure, case.initial_guess)
    assert _worst_error(case, result.coefficients) < 1e-3 * _worst_error(case, case.initial_guess)


@pytest.mark.slow
def test_recovers_a_four_qubit_generator():
    """The largest static case is still recovered from the complete Pauli data set."""
    case = CASES["4q-detuning-zz-ring"]
    result = learn(case.model, Experiment(case=case).measure, case.initial_guess)
    assert _worst_error(case, result.coefficients) < 1e-6


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Modulated generators need a time-ordered propagator, whose period bookkeeping requires "
    "concrete times and so cannot yet be differentiated through. Fitting time-dependent models is "
    "the outstanding gap in the learner.",
    strict=True,
)
def test_time_dependent_case_is_improved():
    """A modulated (time-dependent) generator has no static logarithm, so it uses the propagator fit."""
    case = CASES["2q-exchange-detuned"]
    experiment = Experiment(case=case)
    result = learn(case.model, experiment.measure, case.initial_guess, steps=150)
    assert _worst_error(case, result.coefficients) < _worst_error(case, case.initial_guess)


@pytest.mark.parametrize("name", ["1q-t1-t2", "1q-driven-detuned"])
def test_single_qubit_cases_survive_realistic_shot_noise(name):
    """A fast Rabi drive alongside slow decoherence is recovered to ~1% at 1e5 shots per setting.

    Two things make this work: the objective forward-models the *measured expectations* (reconstructing
    propagators would invert -- and so amplify -- the noise), and the coherent coefficients are frozen
    once the window exceeds their phase-unambiguous horizon, which otherwise pulls a fast frequency
    onto a subharmonic.
    """
    case = CASES[name]
    experiment = Experiment(case=case, shots=100_000, ramp=0.2, seed=2)
    result = learn(case.model, experiment.measure, case.initial_guess, noise_floor=experiment.noise_sigma)
    assert _worst_error(case, result.coefficients) < 0.02


@pytest.mark.parametrize("name", ["2q-driven-zz", "2q-detuning-zz", "2q-xz-crosstalk"])
def test_two_qubit_cases_improve_under_shot_noise(name):
    """Two-qubit problems reach the ~5-11% level at 1e5 shots via local-marginal fitting.

    Fitting each term against the reduced dynamics on its own support keeps a weak two-qubit coupling
    from being one small signal buried among exponentially many irrelevant ones.
    """
    case = CASES[name]
    experiment = Experiment(case=case, shots=100_000, ramp=0.2, seed=2)
    result = learn(case.model, experiment.measure, case.initial_guess, noise_floor=experiment.noise_sigma)
    assert _worst_error(case, result.coefficients) < 0.15


def test_local_marginals_reproduce_the_reduced_experiment():
    """Marginalising the complete data must equal a direct measurement of the reduced experiment.

    Averaging a qubit over its six Pauli eigenstates is exactly the maximally mixed state, so the
    marginalised data is the tomography of ``rho_S (x) (I/2)**env``.  Applying the *same* average to the
    reconstructed anchor states keeps predictions and measurements consistent even after a ramp.
    """
    from quax.learning import (
        _marginalise_data,
        _marginalise_states,
        _pauli_vectors,
        _reconstruct_states,
        _support_observable_indices,
    )

    case = CASES["2q-detuning-zz"]
    times = jnp.array([0.0, 0.37, 1.1])
    data = Experiment(case=case, ramp=0.2, seed=5).measure(times)
    propagators = case.model.propagators(case.true_coefficients, times)
    anchor = _reconstruct_states(data[0], 2)
    pauli = _pauli_vectors(2)

    for support in [(0,), (1,), (0, 1)]:
        observables = pauli[_support_observable_indices(support, 2)]
        states = _marginalise_states(anchor, support, 2)
        predicted = jnp.real(jnp.einsum("ar,trs,ks->tka", observables.conj(), propagators, states))
        measured = _marginalise_data(data, support, 2)
        assert predicted.shape == (3, 6 ** len(support), 4 ** len(support))
        assert jnp.allclose(predicted, measured, atol=1e-10)


def test_local_marginals_need_far_fewer_settings_than_full_tomography():
    """The objective's settings grow with the number of terms, not with the register size."""
    case = CASES["4q-detuning-zz-ring"]
    supports = sorted({term.support for term in case.model.terms if term.support})
    local = sum(6 ** len(s) * 4 ** len(s) for s in supports)
    full = 6**4 * 4**4
    assert local < full / 100, f"local {local} vs full {full}"


LOCAL_CASES = [c for c in CASES.values() if not c.model.is_time_dependent and c.num_qubits <= 2]
LOCAL_IDS = [c.name for c in LOCAL_CASES]


@pytest.mark.parametrize("case", LOCAL_CASES, ids=LOCAL_IDS)
def test_local_derivative_learner_is_exact(case):
    """`learn_local` recovers every static case from local observables alone."""
    from quax.learning import learn_local

    result = learn_local(case.model, Experiment(case=case).measure, case.initial_guess)
    assert _worst_error(case, result.coefficients) < 1e-3


@pytest.mark.slow
def test_local_derivative_learner_handles_four_qubits():
    """The local estimator scales to the largest cases without any global fit."""
    from quax.learning import learn_local

    case = CASES["4q-driven-ring"]
    result = learn_local(case.model, Experiment(case=case).measure, case.initial_guess)
    assert _worst_error(case, result.coefficients) < 1e-3


def test_local_derivative_is_exactly_local():
    """The derivative of a local observable depends only on the reduced generator on its support.

    With the environment maximally mixed a term straddling the boundary contributes nothing, because
    its factor outside the support is a traceless Pauli.  This holds even for a *spreading* transverse
    coupling, which is what removes the light cone from the estimator.
    """
    from quax.learning import _embedded_local_basis, _pauli_vectors, _support_observable_indices, rebuild_term

    case = CASES["2q-exchange-detuned"]  # includes a transverse XY coupling
    model, theta = case.model, case.true_coefficients
    support = (0,)

    states, observables = _embedded_local_basis(support, 2)
    exact = jnp.real(jnp.einsum("ar,rs,ks->ka", observables.conj(), model.generator_at(theta), states))

    # Rebuild every overlapping term on (support u its own support) and sum their reduced actions.
    reduced = jnp.zeros_like(exact)
    for coefficient, term in zip(theta, model.terms):
        if not (set(term.support) & set(support)):
            continue
        register = tuple(sorted(set(support) | set(term.support)))
        relabel = {q: i for i, q in enumerate(register)}
        rebuilt = rebuild_term(term, tuple(relabel[q] for q in term.support), len(register))
        local_states, local_observables = _embedded_local_basis(tuple(relabel[q] for q in support), len(register))
        reduced = reduced + coefficient * jnp.real(
            jnp.einsum("ar,rs,ks->ka", local_observables.conj(), rebuilt.generator, local_states)
        )
    assert jnp.allclose(exact, reduced, atol=1e-12), "local reduction must reproduce the global derivative"
    assert _pauli_vectors(2).shape == (16, 16)
    assert len(_support_observable_indices(support, 2)) == 4
