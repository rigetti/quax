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
"""Tests for the fSim generator and the partner-conditional relaxation model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quax as qx

DIMS = ((2, 2), (2, 2))


def _same_channel(a: qx.Unitary | qx.SuperOp, b: qx.Unitary | qx.SuperOp) -> bool:
    """Equality as channels, which ignores a global phase; the Jozsa fidelity of a rank-one Choi matrix is only
    accurate to the square root of round-off, so superoperators are compared directly."""
    return bool(jnp.allclose(qx.to_superop(a).matrix, qx.to_superop(b).matrix, atol=1e-10))


def _channel_fidelity(a: qx.Unitary | qx.SuperOp, b: qx.Unitary | qx.SuperOp) -> float:
    return float(jnp.real(qx.process_fidelity(qx.to_superop(a), qx.to_superop(b))))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fsim_generates_the_fsim_gate(seed):
    theta, phi = jax.random.uniform(jax.random.key(seed), (2,), minval=-np.pi, maxval=np.pi)
    gate = qx.evolve(qx.hamiltonians.fsim(theta, phi), 1.0)
    assert isinstance(gate, qx.Unitary)
    assert _same_channel(gate, qx.gates.FSIM(theta, phi))


def test_fsim_at_zero_pi_is_cz():
    assert _same_channel(qx.evolve(qx.hamiltonians.fsim(0.0, jnp.pi), 1.0), qx.gates.CZ)


@pytest.mark.parametrize("seed", [3, 4])
def test_fsim_equal_single_qubit_phases_match_phasedfsim(seed):
    theta, phi, gamma = jax.random.uniform(jax.random.key(seed), (3,), minval=-np.pi, maxval=np.pi)
    gate = qx.evolve(qx.hamiltonians.fsim(theta, phi, -gamma, -gamma), 1.0)
    assert _same_channel(gate, qx.gates.PHASEDFSIM(theta, 0.0, 0.0, gamma, phi))


def test_fsim_general_phases_are_the_direct_exponential():
    theta, phi, phi_0, phi_1 = 0.3, 2.9, 0.4, -0.7
    X, Y, Z, I1, P1 = qx.gates.X, qx.gates.Y, qx.gates.Z, qx.gates.I, qx.gates.P1
    expected = (
        -(theta / 4) * ((X | X) + (Y | Y)).matrix
        - phi * (P1 | P1).matrix
        + (phi_0 / 2) * (Z | I1).matrix
        + (phi_1 / 2) * (I1 | Z).matrix
    )
    hamiltonian = qx.hamiltonians.fsim(theta, phi, phi_0, phi_1)
    assert jnp.allclose(hamiltonian.matrix, expected)
    assert qx.is_hermitian(hamiltonian)
    unitary = jax.scipy.linalg.expm(-1j * expected)
    assert _same_channel(qx.evolve(hamiltonian, 1.0), qx.Unitary.from_matrix(unitary, DIMS))


def test_fsim_broadcasts_jits_and_differentiates():
    thetas = jnp.linspace(0.0, 0.5, 5)
    ensemble = qx.hamiltonians.fsim(thetas, jnp.pi)
    assert ensemble.ensemble_size == (5,)
    assert jnp.allclose(jax.jit(qx.hamiltonians.fsim)(0.2, 3.0).matrix, qx.hamiltonians.fsim(0.2, 3.0).matrix)

    def distance(theta):
        gate = qx.evolve(qx.hamiltonians.fsim(theta, jnp.pi), 1.0)
        return jnp.sum(jnp.abs(qx.to_superop(gate).matrix - qx.to_superop(qx.gates.CZ).matrix) ** 2)

    gradient = jax.grad(distance)(0.05)
    assert jnp.isfinite(gradient) and gradient > 0


def test_conditional_relaxation_with_equal_columns_is_independent_relaxation():
    decay = jnp.array([[0.02, 0.02], [0.05, 0.05]])
    dephasing = jnp.array([[0.01, 0.01], [0.03, 0.03]])
    conditional = qx.lindbladians.conditional_relaxation(decay, dephasing)
    independent = qx.lindbladians.thermal_relaxation(1 / 0.02, 1 / 0.01) | qx.lindbladians.thermal_relaxation(
        1 / 0.05, 1 / 0.03
    )
    assert jnp.allclose(conditional.matrix, independent.matrix, atol=1e-12)
    assert qx.is_cptp(qx.evolve(conditional, 1.0))


@pytest.mark.parametrize("partner", [0, 1])
def test_conditional_relaxation_decays_at_the_partners_rate(partner):
    """Qubit 0 excited with the partner frozen in |partner>: its population decays at decay[0, partner]."""
    decay = jnp.array([[0.3, 0.1], [0.0, 0.0]])
    channel = qx.evolve(qx.lindbladians.conditional_relaxation(decay, jnp.zeros((2, 2))), 1.0)
    initial = qx.promote_state_vector_to_density_matrix(
        qx.states.KET1 | (qx.states.KET1 if partner else qx.states.KET0)
    )
    final = channel @ initial
    population = qx.estimate(final, qx.Observable.from_matrix((qx.gates.P1 | qx.gates.I).matrix, DIMS))
    assert population == pytest.approx(float(jnp.exp(-decay[0, partner])), abs=1e-10)
    # the partner is untouched by the jump
    partner_population = qx.estimate(final, qx.Observable.from_matrix((qx.gates.I | qx.gates.P1).matrix, DIMS))
    assert partner_population == pytest.approx(float(partner), abs=1e-10)


def test_conditional_dephasing_at_the_partners_rate():
    dephasing = jnp.array([[0.4, 0.1], [0.0, 0.0]])
    channel = qx.evolve(qx.lindbladians.conditional_relaxation(jnp.zeros((2, 2)), dephasing), 1.0)
    for partner, rate in ((0, 0.4), (1, 0.1)):
        initial = qx.promote_state_vector_to_density_matrix(
            qx.states.XPLUS | (qx.states.KET1 if partner else qx.states.KET0)
        )
        coherence = qx.estimate(channel @ initial, qx.Observable.from_matrix((qx.gates.X | qx.gates.I).matrix, DIMS))
        # dephasing(gamma) shrinks <X> by exp(-gamma t)
        assert coherence == pytest.approx(float(jnp.exp(-rate)), abs=1e-10)


def test_conditional_relaxation_broadcasts_jits_and_differentiates():
    decay = jnp.full((3, 2, 2), 0.02) * jnp.arange(1, 4)[:, None, None]
    dephasing = jnp.full((2, 2), 0.01)
    ensemble = qx.lindbladians.conditional_relaxation(decay, dephasing)
    assert ensemble.ensemble_size == (3,)
    assert jnp.allclose(jax.jit(qx.lindbladians.conditional_relaxation)(decay[0], dephasing).matrix, ensemble[0].matrix)

    def population(rates):
        channel = qx.evolve(qx.lindbladians.conditional_relaxation(rates, dephasing), 1.0)
        initial = qx.promote_state_vector_to_density_matrix(qx.states.KET1 | qx.states.KET0)
        return qx.estimate(channel @ initial, qx.Observable.from_matrix((qx.gates.P1 | qx.gates.I).matrix, DIMS))

    gradient = jax.grad(population)(decay[0])
    assert jnp.all(jnp.isfinite(gradient))
    assert gradient[0, 0] < 0 and gradient[0, 1] == pytest.approx(0.0, abs=1e-10)

    with pytest.raises(ValueError):
        qx.lindbladians.conditional_relaxation(jnp.zeros(4), jnp.zeros(4))


def test_noisy_fsim_is_the_composed_generator():
    decay = jnp.array([[0.02, 0.03], [0.04, 0.05]])
    dephasing = jnp.array([[0.01, 0.02], [0.03, 0.04]])
    model = qx.lindbladians.noisy_fsim(0.1, jnp.pi, decay, dephasing, phi_0=0.05)
    composed = qx.Lindbladian(
        hamiltonian=qx.hamiltonians.fsim(0.1, jnp.pi, 0.05),
        jump_operators=qx.lindbladians.conditional_relaxation(decay, dephasing).jump_operators,
    )
    assert jnp.allclose(model.matrix, composed.matrix, atol=1e-12)
    gate = qx.evolve(model, 1.0)
    assert qx.is_cptp(gate)
    assert 0.9 < _channel_fidelity(gate, qx.gates.CZ) < 1.0
    ensemble = qx.lindbladians.noisy_fsim(jnp.linspace(0, 0.1, 4), jnp.pi, decay, dephasing)
    assert ensemble.ensemble_size == (4,)
