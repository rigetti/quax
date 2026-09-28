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

"""Tests for :class:`quax.ErrorGenerator`, :func:`quax.error_generator` and elementary error generators."""

from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quax as qx

DIMS = [(2,), (3,), (2, 2), (2, 3)]


def _dims_id(dims):
    return "d" + "x".join(map(str, dims))


def _dim(dims):
    return reduce(mul, dims, 1)


def _random_lindbladian(dims, seed, scale=0.05):
    """A small random Lindbladian, so that its channel stays on the principal branch of the log."""
    key_h, key_l = jax.random.split(jax.random.key(seed))
    hamiltonian = qx.random_observable((dims, dims), key_h) * scale
    jumps = qx.random_operator((dims, dims), key_l, size=(2,)) * np.sqrt(scale)
    return qx.Lindbladian(hamiltonian=hamiltonian, jump_operators=jumps)


def _apply(generator, rho):
    """Apply a generator matrix to a density matrix via column-stacked vec(ρ)."""
    d = rho.shape[0]
    return (generator.matrix @ rho.T.reshape(-1)).reshape(d, d).T


# ---------------------------------------------------------------------------
# Elementary generators match the paper's definitions
# ---------------------------------------------------------------------------


def test_elementary_generators_match_definitions():
    """H, S, C and A act on a density matrix exactly as defined in arXiv:2103.01928."""
    rho = np.asarray(qx.random_density_matrix(2, (2,), jax.random.key(0)).matrix)
    p, q = np.asarray(qx.gates.X.matrix), np.asarray(qx.gates.Z.matrix)

    def anti(a, b):
        return a @ b + b @ a

    expected = {
        ("H", "X", None): -1j * (p @ rho - rho @ p),
        ("S", "X", None): p @ rho @ p - rho,
        ("C", "X", "Z"): p @ rho @ q + q @ rho @ p - 0.5 * anti(anti(p, q), rho),
        ("A", "X", "Z"): 1j * (p @ rho @ q - q @ rho @ p + 0.5 * anti(p @ q - q @ p, rho)),
    }
    for (kind, first, second), value in expected.items():
        generator = qx.elementary_error_generator(kind, first, second)
        np.testing.assert_allclose(_apply(generator, rho), value, atol=1e-12)


@pytest.mark.parametrize("dims", DIMS, ids=_dims_id)
def test_elementary_generators_are_the_coordinate_basis(dims):
    """Each elementary generator has a single unit coordinate in its own sector."""
    generator = qx.error_generator(_random_lindbladian(dims, 0))
    labels = generator.labels
    p, q = labels[0], labels[-1]

    h = qx.elementary_error_generator("H", p, dims=dims)
    np.testing.assert_allclose(h.hamiltonian_rates, np.eye(len(labels))[0], atol=1e-12)
    np.testing.assert_allclose(h.dissipator_matrix, 0, atol=1e-12)

    s = qx.elementary_error_generator("S", q, dims=dims)
    np.testing.assert_allclose(s.stochastic_rates, np.eye(len(labels))[-1], atol=1e-12)
    np.testing.assert_allclose(s.hamiltonian_rates, 0, atol=1e-12)

    c = qx.elementary_error_generator("C", p, q, dims=dims)
    assert c.correlation_rates[0, -1] == pytest.approx(1.0)
    assert c.correlation_rates[-1, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(c.stochastic_rates, 0, atol=1e-12)

    a = qx.elementary_error_generator("A", p, q, dims=dims)
    assert a.active_rates[0, -1] == pytest.approx(1.0)
    assert a.active_rates[-1, 0] == pytest.approx(-1.0)
    np.testing.assert_allclose(a.correlation_rates, 0, atol=1e-12)


def test_elementary_generator_validation():
    with pytest.raises(ValueError, match="two distinct labels"):
        qx.elementary_error_generator("C", "X")
    with pytest.raises(ValueError, match="Unknown basis label"):
        qx.elementary_error_generator("H", "Q")
    with pytest.raises(ValueError, match="Unknown error generator kind"):
        qx.elementary_error_generator("B", "X")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Known channels
# ---------------------------------------------------------------------------


def test_pauli_dissipation_is_stochastic():
    """A Pauli jump operator √r·X gives s_X = r and nothing else."""
    rate = 0.01
    jumps = qx.Operator.from_matrix(np.sqrt(rate) * qx.gates.X.matrix[None], ((2,), (2,)))
    generator = qx.error_generator(qx.Lindbladian(hamiltonian=None, jump_operators=jumps))
    np.testing.assert_allclose(generator.stochastic_rates, [rate, 0, 0], atol=1e-12)
    np.testing.assert_allclose(generator.hamiltonian_rates, 0, atol=1e-12)
    np.testing.assert_allclose(generator.correlation_rates, 0, atol=1e-12)
    np.testing.assert_allclose(generator.active_rates, 0, atol=1e-12)


def test_coherent_over_rotation_is_hamiltonian():
    """RX(ε) = exp(-iεX/2) is the Hamiltonian error ε/2·H_X."""
    epsilon = 0.1
    generator = qx.error_generator(qx.to_superop(qx.gates.RX(epsilon)))
    np.testing.assert_allclose(generator.hamiltonian_rates, [epsilon / 2, 0, 0], atol=1e-12)
    np.testing.assert_allclose(generator.dissipator_matrix, 0, atol=1e-12)
    assert generator.total_hamiltonian_error == pytest.approx(epsilon / 2)


def test_amplitude_damping_sectors():
    """Amplitude damping has equal S_X and S_Y, plus an active X-Y term, and is CP."""
    gamma = 0.02
    generator = qx.error_generator(qx.evolve(qx.lindbladians.amplitude_damping(gamma)))
    s_x, s_y, s_z = np.asarray(generator.stochastic_rates)
    assert s_x == pytest.approx(s_y)
    assert s_x > 0
    assert s_z == pytest.approx(0, abs=1e-12)
    assert abs(float(generator.active_rates[0, 1])) > 0
    np.testing.assert_allclose(generator.hamiltonian_rates, 0, atol=1e-12)
    assert bool(generator.is_completely_positive())


def test_negative_stochastic_rate_is_not_cp():
    generator = qx.elementary_error_generator("S", "X") * -0.01
    assert not bool(generator.is_completely_positive())
    with pytest.raises(ValueError, match="not completely positive"):
        generator.to_lindbladian()


# ---------------------------------------------------------------------------
# Round trips and reconstruction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dims", DIMS, ids=_dims_id)
def test_log_of_evolved_lindbladian_recovers_generator(dims):
    """error_generator(evolve(L)) == L for a small generator, and evolve inverts it."""
    lindbladian = _random_lindbladian(dims, 1)
    channel = qx.evolve(lindbladian)
    generator = qx.error_generator(channel)
    np.testing.assert_allclose(generator.matrix, lindbladian.matrix, atol=1e-10)
    np.testing.assert_allclose(qx.evolve(generator).matrix, channel.matrix, atol=1e-10)


@pytest.mark.parametrize("dims", DIMS, ids=_dims_id)
def test_other_representations_agree(dims):
    channel = qx.evolve(_random_lindbladian(dims, 2))
    expected = qx.error_generator(channel).matrix
    for representation in (qx.to_pauli_liouville(channel), qx.to_choi(channel), qx.to_kraus(channel)):
        np.testing.assert_allclose(qx.error_generator(representation).matrix, expected, atol=1e-9)


@pytest.mark.parametrize("dims", DIMS, ids=_dims_id)
def test_coordinates_reconstruct_generator(dims):
    """Summing coordinates times elementary generators reproduces the generator matrix."""
    generator = qx.error_generator(_random_lindbladian(dims, 3))
    labels = generator.labels
    total = qx.ErrorGenerator.from_matrix(jnp.zeros_like(generator.matrix), generator.dims)
    for i, p in enumerate(labels):
        total = total + qx.elementary_error_generator("H", p, dims=dims) * float(generator.hamiltonian_rates[i])
        total = total + qx.elementary_error_generator("S", p, dims=dims) * float(generator.stochastic_rates[i])
        for j in range(i + 1, len(labels)):
            q = labels[j]
            total = total + qx.elementary_error_generator("C", p, q, dims=dims) * float(
                generator.correlation_rates[i, j]
            )
            total = total + qx.elementary_error_generator("A", p, q, dims=dims) * float(generator.active_rates[i, j])
    np.testing.assert_allclose(total.matrix, generator.matrix, atol=1e-10)


@pytest.mark.parametrize("dims", DIMS, ids=_dims_id)
def test_to_lindbladian_round_trip(dims):
    lindbladian = _random_lindbladian(dims, 4)
    generator = qx.error_generator(lindbladian)
    assert bool(generator.is_completely_positive())
    np.testing.assert_allclose(generator.to_lindbladian().matrix, lindbladian.matrix, atol=1e-10)


@pytest.mark.parametrize("dims", DIMS, ids=_dims_id)
def test_generator_infidelity_approximates_process_infidelity(dims):
    """Σs + Σh² matches the process infidelity of e^L to leading order."""
    lindbladian = _random_lindbladian(dims, 5, scale=1e-4)
    generator = qx.error_generator(lindbladian)
    channel = qx.evolve(lindbladian)
    identity = qx.to_superop(qx.Unitary.from_matrix(jnp.eye(_dim(dims), dtype=complex), (dims, dims)))
    infidelity = 1 - float(qx.process_fidelity(channel, identity))
    assert float(generator.generator_infidelity) == pytest.approx(infidelity, rel=1e-2)


# ---------------------------------------------------------------------------
# Algebra, JAX and plotting
# ---------------------------------------------------------------------------


def test_vector_space_algebra():
    a = qx.error_generator(_random_lindbladian((2,), 6))
    b = qx.error_generator(_random_lindbladian((2,), 7))
    np.testing.assert_allclose((a + b).matrix, a.matrix + b.matrix, atol=1e-12)
    np.testing.assert_allclose((a - b).matrix, a.matrix - b.matrix, atol=1e-12)
    np.testing.assert_allclose((-a).stochastic_rates, -a.stochastic_rates, atol=1e-12)
    np.testing.assert_allclose((a * -2.0).hamiltonian_rates, -2.0 * a.hamiltonian_rates, atol=1e-12)
    assert isinstance(2.0 * a, qx.ErrorGenerator)
    with pytest.raises(NotImplementedError, match="complex scalar"):
        _ = a * 1j
    with pytest.raises(ValueError, match="Cannot add"):
        _ = a + qx.error_generator(_random_lindbladian((3,), 8))


def test_jit_and_ensembles():
    channels = qx.evolve(qx.lindbladians.amplitude_damping(jnp.array([0.01, 0.02, 0.03])))
    generators = jax.jit(qx.error_generator)(channels)
    assert isinstance(generators, qx.ErrorGenerator)
    assert generators.ensemble_size == (3,)
    assert generators.stochastic_rates.shape == (3, 3)
    single = qx.error_generator(qx.evolve(qx.lindbladians.amplitude_damping(0.02)))
    np.testing.assert_allclose(generators.stochastic_rates[1], single.stochastic_rates, atol=1e-12)


def test_unsupported_type():
    with pytest.raises(TypeError, match="error_generator"):
        qx.error_generator(qx.gates.X)


def test_plot():
    pytest.importorskip("plotly")
    generator = qx.error_generator(qx.evolve(qx.lindbladians.amplitude_damping(0.02)))
    traces = qx.plot(generator).to_dict()["data"]
    assert [trace["type"] for trace in traces] == ["bar", "bar", "heatmap", "heatmap"]
    np.testing.assert_allclose(traces[1]["y"], generator.stochastic_rates, atol=1e-12)
