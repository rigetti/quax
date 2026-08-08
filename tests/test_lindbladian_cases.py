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

"""Validation of the Lindbladian-learning test battery itself.

These tests check the *problems*, not any learner: that each ansatz is structurally sound and
identifiable, that the initial guesses lie within the promised tolerance, that the generated data is
physical (CPTP), and that the shot-noise and ramp variants do what they claim.
"""

import jax.numpy as jnp
import pytest

from .lindbladian_cases import Experiment, build_cases, pauli_observables, product_states

CASES = build_cases(seed=7)
CASE_IDS = [c.name for c in CASES]
SMALL_CASES = [c for c in CASES if c.num_qubits <= 2]
SMALL_IDS = [c.name for c in SMALL_CASES]


def _column_vectors(matrices):
    return matrices.transpose(0, 2, 1).reshape(matrices.shape[0], -1)


def test_battery_covers_the_intended_progression():
    """The battery spans 1, 2 and 4 qubits and all the intended physical effects."""
    assert [c.num_qubits for c in CASES].count(1) >= 5
    assert [c.num_qubits for c in CASES].count(2) >= 6
    assert [c.num_qubits for c in CASES].count(4) >= 3
    joined = " ".join(name for c in CASES for name in c.names)
    for expected in ("amp_damp", "dephase", "excite", "detuning", "drive_x", "zz", "xz", "exchange", "joint_decay"):
        assert expected in joined, f"battery is missing any {expected} term"
    assert any(c.model.is_time_dependent for c in CASES), "battery has no time-dependent case"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_case_is_structurally_consistent(case):
    """Coefficients, guesses and term lists all line up, and labels are unique."""
    assert case.true_coefficients.shape == (case.model.size,)
    assert case.initial_guess.shape == (case.model.size,)
    assert len(set(case.names)) == case.model.size, "term labels must be unique"
    assert jnp.all(jnp.isfinite(case.true_coefficients))


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_initial_guess_is_within_tolerance(case):
    """The guess is structurally correct and every coefficient is within the promised fraction."""
    relative = case.relative_error(case.initial_guess)
    assert jnp.all(relative <= case.guess_fraction + 1e-9), f"guess outside {case.guess_fraction:.0%}"
    # A guess that is *exactly* the truth would make the battery trivial.
    assert jnp.max(relative) > 0.01, "guess is suspiciously close to the truth"
    # Rates must stay physical (non-negative) after perturbation.
    dissipative = [
        i for i, n in enumerate(case.names) if n.split("[")[0] in ("amp_damp", "dephase", "excite", "joint_decay")
    ]
    assert jnp.all(case.initial_guess[jnp.array(dissipative)] > 0.0)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_model_is_identifiable(case):
    """The unit generators are linearly independent, so the coefficients are identifiable at all."""
    flattened = case.model.generators.reshape(case.model.size, -1)
    rank = jnp.linalg.matrix_rank(flattened)
    assert int(rank) == case.model.size, f"rank {int(rank)} < {case.model.size}: ansatz is degenerate"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_hamiltonian_terms_are_much_faster_than_decoherence(case):
    """Coherent couplings sit orders of magnitude above the dissipative rates, as on real hardware."""
    coherent, dissipative = [], []
    for name, value in zip(case.names, case.true_coefficients):
        kind = name.split("[")[0]
        (coherent if kind in ("detuning", "drive_x", "drive_y", "zz", "xz", "exchange") else dissipative).append(
            float(jnp.abs(value))
        )
    assert dissipative, "every case should include decoherence"
    assert max(dissipative) < 0.2, "dissipative rates should be ~1/(tens of microseconds)"
    if coherent:
        assert max(coherent) / max(dissipative) > 5.0, "coherent terms should dominate the dynamics"
    if any(n.startswith("drive") for n in case.names):
        assert max(coherent) / max(dissipative) > 100.0, "a Rabi drive should be ~1000x the decoherence"


@pytest.mark.parametrize("case", SMALL_CASES, ids=SMALL_IDS)
def test_evolution_is_trace_preserving_and_positive(case):
    """The generated dynamics is a legitimate CPTP evolution of the prepared states."""
    times = jnp.array([0.0, 0.05, 1.0])
    propagators = case.model.propagators(case.true_coefficients, times)
    states = product_states(case.num_qubits)
    dim = states.shape[1]

    assert jnp.allclose(propagators[0], jnp.eye(dim**2), atol=1e-8), "U(0) must be the identity"
    vectors = _column_vectors(states)
    for index in range(len(times)):
        evolved = (vectors @ propagators[index].T).reshape(-1, dim, dim).transpose(0, 2, 1)
        traces = jnp.trace(evolved, axis1=-2, axis2=-1)
        assert jnp.allclose(traces.real, 1.0, atol=1e-6), "trace must be preserved"
        eigenvalues = jnp.linalg.eigvalsh((evolved + evolved.conj().transpose(0, 2, 1)) / 2)
        assert jnp.min(eigenvalues) > -1e-6, "evolved states must stay positive semidefinite"


@pytest.mark.parametrize("case", SMALL_CASES, ids=SMALL_IDS)
def test_noiseless_measurement_shape_and_initial_values(case):
    """Measured data has the complete Pauli shape, and at t=0 reproduces the prepared states."""
    experiment = Experiment(case=case)
    times = jnp.array([0.0, 0.1])
    data = experiment.measure(times)

    n_states, n_observables = 6**case.num_qubits, 4**case.num_qubits
    assert data.shape == (2, n_states, n_observables)
    assert jnp.allclose(data[:, :, 0], 1.0, atol=1e-9), "identity observable is fixed by normalisation"

    states, observables = product_states(case.num_qubits), pauli_observables(case.num_qubits)
    expected = jnp.real(jnp.einsum("oab,kba->ko", observables.conj().transpose(0, 2, 1), states))
    assert jnp.allclose(data[0], expected, atol=1e-8), "t=0 data must be the prepared states"


@pytest.mark.parametrize("shots", [100_000, 4_096])
def test_shot_noise_has_the_expected_magnitude(shots):
    """Shot noise perturbs expectations with the advertised 1/sqrt(N) standard deviation."""
    case = next(c for c in SMALL_CASES if c.num_qubits == 2)  # thousands of samples for a stable estimate
    times = jnp.array([0.1, 0.3, 0.7])
    clean = Experiment(case=case).measure(times)
    noisy = Experiment(case=case, shots=shots, seed=3).measure(times)

    residual = (noisy - clean)[:, :, 1:]  # drop the identity observable, which is never measured
    observed = float(jnp.std(residual))
    expected = 1.0 / jnp.sqrt(shots)
    assert 0.7 * expected < observed < 1.3 * expected
    assert jnp.allclose(noisy[:, :, 0], 1.0), "identity observable must stay exact"


@pytest.mark.parametrize("case", SMALL_CASES, ids=SMALL_IDS)
def test_ramp_corrupts_the_initial_state_but_keeps_it_physical(case):
    """With a turn-on ramp the t=0 data is no longer the prepared states, but is still a valid state set."""
    without = Experiment(case=case, ramp=0.0).measure(jnp.array([0.0]))
    with_ramp = Experiment(case=case, ramp=0.2, seed=5).measure(jnp.array([0.0]))

    assert not jnp.allclose(without, with_ramp, atol=1e-3), "the ramp must actually change the initial states"
    assert jnp.allclose(with_ramp[:, :, 0], 1.0, atol=1e-9), "the ramp is trace preserving"
    # The ramped states must still span the space, or nothing could be learned from them.
    states = Experiment(case=case, ramp=0.2, seed=5)._initial_states()
    assert int(jnp.linalg.matrix_rank(_column_vectors(states))) == states.shape[1] ** 2


def test_four_qubit_case_measures_the_complete_pauli_set():
    """The largest cases still expose the full 6^n x 4^n data set the learner expects."""
    case = next(c for c in CASES if c.num_qubits == 4 and not c.model.is_time_dependent)
    data = Experiment(case=case).measure(jnp.array([0.0, 0.5]))
    assert data.shape == (2, 6**4, 4**4)
    assert jnp.all(jnp.isfinite(data))
    assert jnp.max(jnp.abs(data)) <= 1.0 + 1e-8, "Pauli expectations are bounded by 1"


def test_time_dependent_case_is_periodic_and_physical():
    """The modulated exchange model has a well-defined period and integrates to a CPTP map."""
    case = next(c for c in CASES if c.model.is_time_dependent and c.num_qubits == 2)
    period = case.model.period
    assert period is not None and 0.0 < period < 0.1, "modulation period should be a few nanoseconds"

    propagators = case.model.propagators(case.true_coefficients, jnp.array([0.0, period, 2 * period, 0.5]))
    assert jnp.allclose(propagators[0], jnp.eye(16), atol=1e-8)
    # Periodicity: U(2T) = U(T)^2 for a periodic generator.
    assert jnp.allclose(propagators[2], propagators[1] @ propagators[1], atol=1e-5)
