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
"""Tests for the magic (non-stabilizerness) measures."""

from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quax as qx

DIMS = [(2,), (3,), (4,), (2, 2), (2, 3), (3, 3), (2, 2, 2)]


def _d(dims):
    return reduce(mul, dims, 1)


def _state(*amplitudes):
    vector = jnp.array(amplitudes, dtype=complex)
    return qx.StateVector.from_matrix(vector / jnp.linalg.norm(vector), (len(amplitudes),))


def test_stabilizer_norm_of_familiar_qubit_states():
    t_state = _state(1, np.exp(1j * np.pi / 4))
    maximally_mixed = qx.DensityMatrix.from_matrix(jnp.eye(2, dtype=complex) / 2, (2,))

    assert jnp.isclose(qx.stabilizer_norm(qx.states.KET0), 1)
    assert jnp.isclose(qx.stabilizer_norm(qx.states.KETPLUS), 1)
    assert jnp.isclose(qx.stabilizer_norm(qx.states.KETPLUSI), 1)
    assert jnp.isclose(qx.stabilizer_norm(t_state), (1 + np.sqrt(2)) / 2)
    assert jnp.isclose(qx.stabilizer_norm(maximally_mixed), 1 / 2)


@pytest.mark.parametrize("dims", DIMS, ids=str)
@pytest.mark.parametrize("ensemble", [(), (3,), (2, 2)], ids=str)
def test_stabilizer_norm_matches_the_definition(dims, ensemble):
    rho = qx.random_density_matrix(2, dims, key=jax.random.key(1), size=ensemble)
    traces = jnp.einsum("...ij,kji->...k", rho.matrix, qx.n_qudit_basis(dims).matrix)
    expected = jnp.sum(jnp.abs(traces), axis=-1) / _d(dims)
    result = qx.stabilizer_norm(rho)
    assert result.shape == ensemble
    assert not jnp.iscomplexobj(result)
    assert jnp.allclose(result, expected, atol=1e-12)


def test_single_qubit_stabilizer_norm_is_the_octahedron():
    bloch = jax.random.uniform(jax.random.key(2), (50, 3), minval=-1, maxval=1)
    bloch = bloch / jnp.maximum(1, jnp.linalg.norm(bloch, axis=-1, keepdims=True))
    rho = qx.DensityMatrix.from_bloch_vector(bloch)
    assert jnp.allclose(qx.stabilizer_norm(rho), (1 + jnp.sum(jnp.abs(bloch), axis=-1)) / 2, atol=1e-12)


def test_pure_stabilizer_states_have_norm_one():
    single_qubit = qx.apply_unitary_to_state_vector(qx.ensembles.CLIFFORDS_1Q, qx.states.KET0)
    assert jnp.allclose(qx.stabilizer_norm(single_qubit), 1)

    ghz = qx.apply_unitary_to_state_vector(qx.gates.CNOT, qx.states.KETPLUS | qx.states.KET0)
    assert jnp.isclose(qx.stabilizer_norm(ghz), 1)

    qutrit_plus = _state(1, 1, 1)
    assert jnp.isclose(qx.stabilizer_norm(qutrit_plus), 1)
    assert jnp.isclose(qx.stabilizer_norm(qutrit_plus | qx.zero_state_vector(dims=(3,))), 1)


def test_mixtures_of_stabilizer_states_are_at_most_one():
    stabilizer = qx.promote_state_vector_to_density_matrix(
        qx.apply_unitary_to_state_vector(qx.ensembles.CLIFFORDS_1Q, qx.states.KET0)
    ).matrix
    weights = jax.random.dirichlet(jax.random.key(3), jnp.ones(stabilizer.shape[0]), (100,))
    mixtures = qx.DensityMatrix.from_matrix(jnp.einsum("mk,kij->mij", weights, stabilizer), (2,))
    assert jnp.all(qx.stabilizer_norm(mixtures) <= 1 + 1e-12)


@pytest.mark.parametrize("dims", DIMS, ids=str)
def test_maximally_mixed_state_has_norm_one_over_d(dims):
    d = _d(dims)
    maximally_mixed = qx.DensityMatrix.from_matrix(jnp.eye(d, dtype=complex) / d, dims)
    assert jnp.isclose(qx.stabilizer_norm(maximally_mixed), 1 / d)


@pytest.mark.parametrize(("dims_a", "dims_b"), [((2,), (2,)), ((2,), (3,)), ((3,), (2, 2))], ids=str)
def test_stabilizer_norm_is_multiplicative(dims_a, dims_b):
    a = qx.random_density_matrix(2, dims_a, key=jax.random.key(4), size=(3,))
    b = qx.random_density_matrix(2, dims_b, key=jax.random.key(5), size=(3,))
    assert jnp.allclose(qx.stabilizer_norm(a | b), qx.stabilizer_norm(a) * qx.stabilizer_norm(b), atol=1e-12)


def test_state_vector_agrees_with_its_density_matrix():
    psi = qx.random_state_vector((2, 3), key=jax.random.key(6), size=(4,))
    rho = qx.promote_state_vector_to_density_matrix(psi)
    assert jnp.allclose(qx.stabilizer_norm(psi), qx.stabilizer_norm(rho), atol=1e-12)


def test_stabilizer_norm_jit_and_grad():
    rho = qx.random_density_matrix(4, (2, 2), key=jax.random.key(7))
    assert jnp.allclose(jax.jit(qx.stabilizer_norm)(rho), qx.stabilizer_norm(rho))

    def norm_of_rotated_zero(angle):
        rotation = jnp.array([[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]], dtype=complex)
        return qx.stabilizer_norm(qx.StateVector.from_matrix(rotation @ qx.states.KET0.matrix, (2,)))

    # |0> has Tr[X rho] = Tr[Y rho] = 0, where a square root of |chi|**2 would give a NaN gradient.
    assert jnp.isfinite(jax.grad(norm_of_rotated_zero)(0.0))
    # Bloch vector (sin 2a, 0, cos 2a): the norm is (1 + |sin 2a| + |cos 2a|) / 2.
    assert jnp.isclose(jax.grad(norm_of_rotated_zero)(0.3), jnp.cos(0.6) - jnp.sin(0.6))


def test_stabilizer_norm_rejects_other_types():
    with pytest.raises(TypeError):
        qx.stabilizer_norm(qx.random_observable(((2,), (2,)), key=jax.random.key(8)))
    with pytest.raises(TypeError):
        qx.stabilizer_norm(jnp.eye(2))


def _t_state():
    return _state(1, np.exp(1j * np.pi / 4))


def test_stabilizer_renyi_entropy_of_familiar_qubit_states():
    assert jnp.isclose(qx.stabilizer_renyi_entropy(qx.states.KET0), 0)
    assert jnp.isclose(qx.stabilizer_renyi_entropy(qx.states.KETPLUS), 0)
    assert jnp.isclose(qx.stabilizer_renyi_entropy(_t_state()), np.log(4 / 3))
    assert jnp.isclose(qx.stabilizer_renyi_entropy(_t_state(), base=2), np.log2(4 / 3))


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("alpha", [0.5, 2.0, 3.0, 4.5])
def test_stabilizer_renyi_entropy_of_t_states_matches_the_closed_form(n, alpha):
    t_states = reduce(lambda a, b: a | b, [_t_state()] * n)
    expected = n / (1 - alpha) * (np.log(2 ** (1 - alpha) + 1) - np.log(2))
    assert jnp.isclose(qx.stabilizer_renyi_entropy(t_states, alpha=alpha), expected)


def test_stabilizer_renyi_entropy_limits_on_t_states():
    t_states = _t_state() | _t_state() | _t_state()
    assert jnp.isclose(qx.stabilizer_renyi_entropy(t_states, alpha=1), 3 * np.log(2) / 2)
    assert jnp.isclose(qx.stabilizer_renyi_entropy(t_states, alpha=0), 3 * np.log(3 / 2))
    assert jnp.isclose(qx.stabilizer_renyi_entropy(t_states, alpha=1 + 1e-6), 3 * np.log(2) / 2, atol=1e-5)


@pytest.mark.parametrize("dims", DIMS, ids=str)
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 3.0])
def test_stabilizer_renyi_entropy_matches_the_definition(dims, alpha):
    psi = qx.random_state_vector(dims, key=jax.random.key(9), size=(3,))
    traces = jnp.einsum("...i,kij,...j->...k", jnp.conj(psi.matrix), qx.n_qudit_basis(dims).matrix, psi.matrix)
    xi = jnp.abs(traces) ** 2 / _d(dims)
    if alpha == 1:
        entropy = -jnp.sum(xi * jnp.log(xi), axis=-1)
    else:
        entropy = jnp.log(jnp.sum(xi**alpha, axis=-1)) / (1 - alpha)
    assert jnp.allclose(qx.stabilizer_renyi_entropy(psi, alpha=alpha), entropy - np.log(_d(dims)), atol=1e-10)


def test_stabilizer_renyi_entropy_is_zero_exactly_on_stabilizer_states():
    single_qubit = qx.apply_unitary_to_state_vector(qx.ensembles.CLIFFORDS_1Q, qx.states.KET0)
    ghz = qx.apply_unitary_to_state_vector(qx.gates.CNOT, qx.states.KETPLUS | qx.states.KET0)
    qutrit = _state(1, 1, 1) | qx.zero_state_vector(dims=(3,))
    for alpha in [0, 0.5, 1, 2, 3]:
        assert jnp.allclose(qx.stabilizer_renyi_entropy(single_qubit, alpha=alpha), 0, atol=1e-10)
        assert jnp.isclose(qx.stabilizer_renyi_entropy(ghz, alpha=alpha), 0, atol=1e-10)
        assert jnp.isclose(qx.stabilizer_renyi_entropy(qutrit, alpha=alpha), 0, atol=1e-10)
    random = qx.random_state_vector((2, 3), key=jax.random.key(10), size=(20,))
    assert jnp.all(qx.stabilizer_renyi_entropy(random) > 1e-3)


def test_stabilizer_renyi_entropy_is_clifford_invariant():
    psi = qx.random_state_vector((2, 2), key=jax.random.key(11), size=(4,))
    clifford = qx.gates.CNOT @ (qx.gates.H | qx.gates.S)
    rotated = qx.apply_unitary_to_state_vector(clifford, psi)
    for alpha in [0.5, 1, 2]:
        assert jnp.allclose(
            qx.stabilizer_renyi_entropy(rotated, alpha=alpha), qx.stabilizer_renyi_entropy(psi, alpha=alpha)
        )


@pytest.mark.parametrize(("dims_a", "dims_b"), [((2,), (2,)), ((2,), (3,)), ((3,), (2, 2))], ids=str)
def test_stabilizer_renyi_entropy_is_additive(dims_a, dims_b):
    a = qx.random_state_vector(dims_a, key=jax.random.key(12), size=(3,))
    b = qx.random_state_vector(dims_b, key=jax.random.key(13), size=(3,))
    for alpha in [0.5, 1, 2]:
        total = qx.stabilizer_renyi_entropy(a, alpha=alpha) + qx.stabilizer_renyi_entropy(b, alpha=alpha)
        assert jnp.allclose(qx.stabilizer_renyi_entropy(a | b, alpha=alpha), total, atol=1e-10)


def test_stabilizer_renyi_entropy_order_and_bounds():
    psi = qx.random_state_vector((2, 3), key=jax.random.key(14), size=(10,))
    entropies = jnp.stack([qx.stabilizer_renyi_entropy(psi, alpha=alpha) for alpha in [0.5, 1, 2, 3]])
    assert jnp.all(jnp.diff(entropies, axis=0) <= 1e-12)
    assert jnp.all(entropies <= np.log(6) + 1e-12)
    assert jnp.allclose(entropies[0], 2 * jnp.log(qx.stabilizer_norm(psi)))


def test_stabilizer_renyi_entropy_jit_and_grad():
    psi = qx.random_state_vector((2, 2), key=jax.random.key(15))
    sre = jax.jit(qx.stabilizer_renyi_entropy, static_argnames="alpha")
    assert jnp.allclose(sre(psi, alpha=3.0), qx.stabilizer_renyi_entropy(psi, alpha=3.0))

    def entropy_of_rotated_zero(angle, alpha):
        rotation = jnp.array([[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]], dtype=complex)
        state = qx.StateVector.from_matrix(rotation @ qx.states.KET0.matrix, (2,))
        return qx.stabilizer_renyi_entropy(state, alpha=alpha)

    for alpha in [1.0, 2.0]:
        assert jnp.isfinite(jax.grad(entropy_of_rotated_zero)(0.0, alpha))
        assert jnp.isfinite(jax.grad(entropy_of_rotated_zero)(0.3, alpha))


def test_stabilizer_renyi_entropy_rejects_mixed_states_and_bad_orders():
    with pytest.raises(TypeError):
        qx.stabilizer_renyi_entropy(qx.random_density_matrix(2, (2,), key=jax.random.key(16)))
    with pytest.raises(ValueError):
        qx.stabilizer_renyi_entropy(qx.states.KET0, alpha=-1.0)
    with pytest.raises(ValueError):
        qx.stabilizer_renyi_entropy(qx.states.KET0, alpha=float("inf"))


def _projector(psi):
    return qx.promote_state_vector_to_density_matrix(psi).matrix


def test_mixed_stabilizer_renyi_entropy_of_familiar_qubit_states():
    zero = qx.DensityMatrix.from_matrix(_projector(qx.states.KET0), (2,))
    maximally_mixed = qx.DensityMatrix.from_matrix(jnp.eye(2, dtype=complex) / 2, (2,))
    stabilizer_mixture = qx.DensityMatrix.from_matrix(
        (_projector(qx.states.KET0) + _projector(qx.states.KETPLUS)) / 2, (2,)
    )

    assert jnp.isclose(qx.mixed_stabilizer_renyi_entropy(zero), 0)
    assert jnp.isclose(qx.mixed_stabilizer_renyi_entropy(maximally_mixed), 0)
    assert jnp.isclose(qx.mixed_stabilizer_renyi_entropy(stabilizer_mixture), np.log(4 / 3))
    assert jnp.isclose(qx.mixed_stabilizer_renyi_entropy(_t_state()), np.log(4 / 3))


@pytest.mark.parametrize("dims", DIMS, ids=str)
def test_mixed_stabilizer_renyi_entropy_matches_the_definition(dims):
    rho = qx.random_density_matrix(2, dims, key=jax.random.key(17), size=(3,))
    traces = jnp.abs(jnp.einsum("...ij,kji->...k", rho.matrix, qx.n_qudit_basis(dims).matrix))
    expected = -jnp.log(jnp.sum(traces**4, axis=-1) / jnp.sum(traces**2, axis=-1))
    assert jnp.allclose(qx.mixed_stabilizer_renyi_entropy(rho), expected, atol=1e-10)


@pytest.mark.parametrize("dims", [(2,), (2, 2), (2, 3), (3, 3)], ids=str)
def test_mixed_stabilizer_renyi_entropy_reduces_to_m2_on_pure_states(dims):
    psi = qx.random_state_vector(dims, key=jax.random.key(18), size=(4,))
    m2 = qx.stabilizer_renyi_entropy(psi, alpha=2)
    assert jnp.allclose(qx.mixed_stabilizer_renyi_entropy(psi), m2, atol=1e-12)
    assert jnp.allclose(
        qx.mixed_stabilizer_renyi_entropy(qx.promote_state_vector_to_density_matrix(psi)), m2, atol=1e-12
    )


def test_mixed_stabilizer_renyi_entropy_is_zero_on_stabilizer_code_states():
    z = qx.gates.Z.matrix
    code_state = qx.DensityMatrix.from_matrix((jnp.eye(4) + jnp.kron(z, z)) / 4, (2, 2))
    assert jnp.isclose(qx.mixed_stabilizer_renyi_entropy(code_state), 0, atol=1e-12)
    noisy = qx.random_density_matrix(2, (2, 2), key=jax.random.key(19), size=(10,))
    assert jnp.all(qx.mixed_stabilizer_renyi_entropy(noisy) > 1e-3)


def test_mixed_stabilizer_renyi_entropy_is_clifford_invariant_and_additive():
    rho = qx.random_density_matrix(2, (2, 2), key=jax.random.key(20), size=(3,))
    clifford = qx.gates.CNOT @ (qx.gates.H | qx.gates.S)
    rotated = qx.apply_unitary_to_density_matrix(clifford, rho)
    assert jnp.allclose(qx.mixed_stabilizer_renyi_entropy(rotated), qx.mixed_stabilizer_renyi_entropy(rho))
    sigma = qx.random_density_matrix(2, (3,), key=jax.random.key(21), size=(3,))
    total = qx.mixed_stabilizer_renyi_entropy(rho) + qx.mixed_stabilizer_renyi_entropy(sigma)
    assert jnp.allclose(qx.mixed_stabilizer_renyi_entropy(rho | sigma), total, atol=1e-10)


def test_mixed_stabilizer_renyi_entropy_base_jit_grad_and_types():
    rho = qx.random_density_matrix(2, (2,), key=jax.random.key(22))
    assert jnp.isclose(
        qx.mixed_stabilizer_renyi_entropy(rho, base=2), qx.mixed_stabilizer_renyi_entropy(rho) / np.log(2)
    )
    assert jnp.allclose(jax.jit(qx.mixed_stabilizer_renyi_entropy)(rho), qx.mixed_stabilizer_renyi_entropy(rho))

    def entropy_of_mixed_zero(p):
        matrix = (1 - p) * _projector(qx.states.KET0) + p * jnp.eye(2) / 2
        return qx.mixed_stabilizer_renyi_entropy(qx.DensityMatrix.from_matrix(matrix, (2,)))

    assert jnp.isfinite(jax.grad(entropy_of_mixed_zero)(0.3))
    with pytest.raises(TypeError):
        qx.mixed_stabilizer_renyi_entropy(qx.random_observable(((2,), (2,)), key=jax.random.key(23)))


def test_single_qubit_stabilizer_states_are_the_six_cardinal_states():
    expected = jnp.array(
        [
            [1, 0],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1j / np.sqrt(2)],
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
            [1 / np.sqrt(2), -1j / np.sqrt(2)],
            [0, 1],
        ]
    )
    assert jnp.allclose(qx.stabilizer_states((2,)).matrix, expected)


@pytest.mark.parametrize(
    ("dims", "count"),
    [((2,), 6), ((3,), 12), ((5,), 30), ((2, 2), 60), ((2, 3), 72), ((3, 3), 360), ((2, 2, 2), 1080), ((2, 2, 3), 720)],
    ids=str,
)
def test_stabilizer_state_counts(dims, count):
    states = qx.stabilizer_states(dims)
    assert states.ensemble_size == (count,)
    assert jnp.allclose(jnp.linalg.norm(states.matrix, axis=-1), 1)
    overlaps = jnp.abs(jnp.einsum("ai,bi->ab", jnp.conj(states.matrix), states.matrix))
    assert jnp.allclose(jnp.diagonal(overlaps), 1)
    assert jnp.all(overlaps[~jnp.eye(count, dtype=bool)] < 1 - 1e-9)


@pytest.mark.parametrize("dims", [(2, 2), (2, 3), (3, 3), (2, 2, 2)], ids=str)
def test_stabilizer_states_have_no_magic(dims):
    states = qx.stabilizer_states(dims)
    assert jnp.allclose(qx.stabilizer_norm(states), 1)
    for alpha in [0, 0.5, 1, 2]:
        assert jnp.allclose(qx.stabilizer_renyi_entropy(states, alpha=alpha), 0, atol=1e-10)


def test_stabilizer_states_contain_familiar_states():
    states = qx.stabilizer_states((2, 2)).matrix
    ghz = jnp.array([1, 0, 0, 1]) / np.sqrt(2)
    assert jnp.isclose(jnp.max(jnp.abs(states @ jnp.conj(ghz))), 1)
    t_state = jnp.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
    assert jnp.max(jnp.abs(qx.stabilizer_states((2,)).matrix @ jnp.conj(t_state))) < 1 - 1e-3


@pytest.mark.parametrize("dims", [(4,), (6,), (2, 4), (1,), ()], ids=str)
def test_stabilizer_states_reject_composite_dimensions(dims):
    with pytest.raises(ValueError):
        qx.stabilizer_states(dims)
