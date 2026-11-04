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


ODD_DIMS = [(3,), (5,), (9,), (3, 3), (3, 5)]


def _gross_point_operators(d):
    """Gross Eq. (15): A(p, q) = d^-1 sum_b conj(chi([(p, q), b])) w(b)^dagger, w(p, q) = omega^(-pq/2) Z^p X^q."""
    shift = np.roll(np.eye(d), 1, axis=0)
    clock = np.diag(np.exp(2j * np.pi * np.arange(d) / d))
    omega, half = np.exp(2j * np.pi / d), (d + 1) // 2

    def weyl(p, q):
        return omega ** (-half * p * q % d) * np.linalg.matrix_power(clock, p) @ np.linalg.matrix_power(shift, q)

    points = np.empty((d, d, d, d), dtype=complex)  # [q, p, row, col]
    for p, q in np.ndindex(d, d):
        points[q, p] = sum(omega ** (-((p * b - q * a) % d)) * weyl(a, b).conj().T for a, b in np.ndindex(d, d)) / d
    return points


def _wigner_by_definition(rho, dims):
    """W[q, p] = Tr[A(p, q) rho] / D with A the tensor product of each qudit's Gross operators."""
    operators = np.ones((1, 1), dtype=complex)
    for d in dims:
        points = _gross_point_operators(d)  # [q, p, row, col]
        operators = np.einsum("...ij,qpkl->...qpikjl", operators, points)
        operators = operators.reshape(*operators.shape[:-4], operators.shape[-4] * d, operators.shape[-2] * d)
    n = len(dims)
    order = [2 * k for k in range(n)] + [2 * k + 1 for k in range(n)]  # q_0 p_0 q_1 p_1 -> q_0 q_1 p_0 p_1
    operators = np.transpose(operators, (*order, 2 * n, 2 * n + 1))
    return np.einsum("...ij,ji->...", operators, np.asarray(rho)).real / _d(dims)


def test_wigner_function_of_familiar_qutrit_states():
    zero = qx.zero_state_vector(dims=(3,))
    strange = _state(0, 1, -1)
    measure_zero = qx.Observable.from_matrix(jnp.diag(jnp.array([1, 0, 0], dtype=complex)), ((3,), (3,)))

    assert jnp.allclose(qx.wigner_function(zero), jnp.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]) / 3)
    assert jnp.allclose(qx.wigner_function(strange), jnp.array([[-2, 1, 1], [1, 1, 1], [1, 1, 1]]) / 6)
    assert jnp.allclose(qx.wigner_function(measure_zero), jnp.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]))


@pytest.mark.parametrize("dims", ODD_DIMS, ids=str)
def test_wigner_function_matches_gross_definition(dims):
    rho = qx.random_density_matrix(_d(dims), dims, key=jax.random.key(24))
    result = qx.wigner_function(rho)
    assert result.shape == dims + dims
    assert jnp.allclose(result, _wigner_by_definition(rho.matrix, dims), atol=1e-12)


@pytest.mark.parametrize("dims", ODD_DIMS, ids=str)
def test_wigner_function_normalization_and_marginals(dims):
    rho = qx.random_density_matrix(2, dims, key=jax.random.key(25), size=(3,))
    w = qx.wigner_function(rho)
    n = len(dims)
    assert jnp.allclose(jnp.sum(w, axis=tuple(range(1, 2 * n + 1))), 1)
    marginal = jnp.sum(w, axis=tuple(range(n + 1, 2 * n + 1))).reshape(3, -1)
    assert jnp.allclose(marginal, jnp.real(jnp.diagonal(rho.matrix, axis1=-2, axis2=-1)), atol=1e-12)


def test_wigner_function_overlap_with_an_observable():
    dims = (3, 5)
    rho = qx.random_density_matrix(4, dims, key=jax.random.key(26))
    observable = qx.random_observable((dims, dims), key=jax.random.key(27))
    overlap = jnp.sum(qx.wigner_function(observable) * qx.wigner_function(rho))
    assert jnp.isclose(overlap, qx.estimate(rho, observable))


@pytest.mark.parametrize("dims", [(3,), (5,), (3, 3)], ids=str)
def test_discrete_hudson_theorem(dims):
    stabilizer = qx.wigner_function(qx.stabilizer_states(dims))
    assert jnp.all(stabilizer >= -1e-12)
    magic = qx.wigner_function(qx.random_state_vector(dims, key=jax.random.key(28), size=(20,)))
    assert jnp.all(jnp.min(magic.reshape(20, -1), axis=-1) < -1e-6)


def test_wigner_function_shifts_under_weyl_operators():
    rho = qx.random_density_matrix(2, (5,), key=jax.random.key(29))
    shift = jnp.roll(jnp.eye(5), 1, axis=0)
    shifted = qx.DensityMatrix.from_matrix(shift @ rho.matrix @ shift.T, (5,))
    assert jnp.allclose(qx.wigner_function(shifted), jnp.roll(qx.wigner_function(rho), 1, axis=0))


def test_wigner_function_of_a_product_is_the_product():
    a = qx.random_density_matrix(2, (3,), key=jax.random.key(30))
    b = qx.random_density_matrix(2, (5,), key=jax.random.key(31))
    expected = jnp.einsum("qp,rs->qrps", qx.wigner_function(a), qx.wigner_function(b))
    assert jnp.allclose(qx.wigner_function(a | b), expected, atol=1e-12)


def test_wigner_function_dispatch_jit_grad_and_errors():
    psi = qx.random_state_vector((3, 3), key=jax.random.key(32), size=(2,))
    rho = qx.promote_state_vector_to_density_matrix(psi)
    assert jnp.allclose(qx.wigner_function(psi), qx.wigner_function(rho))
    assert jnp.allclose(jax.jit(qx.wigner_function)(rho), qx.wigner_function(rho))

    def origin_of_rotated_zero(angle):
        rotation = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0], [jnp.sin(angle), jnp.cos(angle), 0], [0, 0, 1]])
        state = qx.StateVector.from_matrix((rotation @ jnp.array([1, 0, 0])).astype(complex), (3,))
        return qx.wigner_function(state)[0, 0]

    assert jnp.isfinite(jax.grad(origin_of_rotated_zero)(0.4))
    with pytest.raises(ValueError):
        qx.wigner_function(qx.random_density_matrix(2, (2,), key=jax.random.key(33)))
    with pytest.raises(ValueError):
        qx.wigner_function(qx.random_density_matrix(2, (3, 4), key=jax.random.key(34)))
    with pytest.raises(TypeError):
        qx.wigner_function(jnp.eye(3))


def test_sum_negativity_of_familiar_qutrit_states():
    strange = _state(0, 1, -1)
    assert jnp.isclose(qx.sum_negativity(qx.zero_state_vector(dims=(3,))), 0)
    assert jnp.isclose(qx.sum_negativity(strange), 1 / 3)
    assert jnp.isclose(qx.sum_negativity(strange | strange), 8 / 9)


@pytest.mark.parametrize("dims", ODD_DIMS, ids=str)
def test_sum_negativity_matches_both_forms_of_the_definition(dims):
    rho = qx.random_density_matrix(2, dims, key=jax.random.key(36), size=(3,))
    w = qx.wigner_function(rho).reshape(3, -1)
    negative_entries = -jnp.sum(jnp.where(w < 0, w, 0), axis=-1)
    one_norm = (jnp.sum(jnp.abs(w), axis=-1) - 1) / 2
    assert jnp.allclose(qx.sum_negativity(rho), negative_entries, atol=1e-12)
    assert jnp.allclose(qx.sum_negativity(rho), one_norm, atol=1e-12)


@pytest.mark.parametrize("dims", [(3,), (5,), (3, 3)], ids=str)
def test_sum_negativity_is_zero_on_stabilizer_states_and_mixtures(dims):
    states = qx.stabilizer_states(dims)
    assert jnp.allclose(qx.sum_negativity(states), 0, atol=1e-12)
    projectors = qx.promote_state_vector_to_density_matrix(states).matrix
    weights = jax.random.dirichlet(jax.random.key(37), jnp.ones(projectors.shape[0]), (10,))
    mixtures = qx.DensityMatrix.from_matrix(jnp.einsum("mk,kij->mij", weights, projectors), dims)
    assert jnp.allclose(qx.sum_negativity(mixtures), 0, atol=1e-12)


@pytest.mark.parametrize("copies", [1, 2, 3])
def test_sum_negativity_composition_law(copies):
    rho = qx.random_density_matrix(2, (3,), key=jax.random.key(38))
    power = reduce(lambda a, b: a | b, [rho] * copies)
    sn = qx.sum_negativity(rho)
    assert jnp.isclose(qx.sum_negativity(power), ((2 * sn + 1) ** copies - 1) / 2)


def test_sum_negativity_is_clifford_invariant_and_drops_under_partial_trace():
    d = 3
    j = np.arange(d)
    fourier = np.exp(2j * np.pi * np.outer(j, j) / d) / np.sqrt(d)
    phase = np.diag(np.exp(2j * np.pi * j * j / d))
    clifford = jnp.asarray(np.kron(fourier @ phase, phase @ fourier))
    rho = qx.random_density_matrix(2, (3, 3), key=jax.random.key(39), size=(5,))
    rotated = qx.DensityMatrix.from_matrix(clifford @ rho.matrix @ clifford.conj().T, (3, 3))
    assert jnp.allclose(qx.sum_negativity(rotated), qx.sum_negativity(rho), atol=1e-12)
    reduced = qx.DensityMatrix.from_matrix(jnp.einsum("nabcb->nac", rho.data), (3,))
    assert jnp.all(qx.sum_negativity(reduced) <= qx.sum_negativity(rho) + 1e-12)


def test_sum_negativity_dispatch_jit_grad_and_errors():
    psi = qx.random_state_vector((3, 5), key=jax.random.key(40), size=(2,))
    rho = qx.promote_state_vector_to_density_matrix(psi)
    assert jnp.allclose(qx.sum_negativity(psi), qx.sum_negativity(rho))
    assert jnp.allclose(jax.jit(qx.sum_negativity)(rho), qx.sum_negativity(rho))

    def negativity_of_mixed_strange(p):
        strange = _projector(_state(0, 1, -1))
        return qx.sum_negativity(qx.DensityMatrix.from_matrix((1 - p) * strange + p * jnp.eye(3) / 3, (3,)))

    assert jnp.isclose(jax.grad(negativity_of_mixed_strange)(0.2), -1 / 3 - 1 / 9)
    with pytest.raises(ValueError):
        qx.sum_negativity(qx.random_density_matrix(2, (2,), key=jax.random.key(41)))
    with pytest.raises(TypeError):
        qx.sum_negativity(qx.random_observable(((3,), (3,)), key=jax.random.key(42)))


def test_mana_of_familiar_qutrit_states():
    strange = _state(0, 1, -1)
    norrell = _state(-1, 2, -1)
    assert jnp.isclose(qx.mana(qx.zero_state_vector(dims=(3,))), 0)
    assert jnp.isclose(qx.mana(strange), np.log(5 / 3))
    assert jnp.isclose(qx.mana(norrell), np.log(5 / 3))
    assert jnp.isclose(qx.mana(strange, base=2), np.log2(5 / 3))


@pytest.mark.parametrize("dims", ODD_DIMS, ids=str)
def test_mana_is_the_log_of_the_wigner_one_norm(dims):
    rho = qx.random_density_matrix(2, dims, key=jax.random.key(43), size=(3,))
    one_norm = jnp.sum(jnp.abs(qx.wigner_function(rho).reshape(3, -1)), axis=-1)
    assert jnp.allclose(qx.mana(rho), jnp.log(one_norm), atol=1e-12)
    assert jnp.allclose(qx.mana(rho), jnp.log(2 * qx.sum_negativity(rho) + 1), atol=1e-12)


@pytest.mark.parametrize("dims", [(3,), (5,), (3, 3)], ids=str)
def test_mana_is_zero_on_stabilizer_states_and_mixtures(dims):
    states = qx.stabilizer_states(dims)
    assert jnp.allclose(qx.mana(states), 0, atol=1e-12)
    projectors = qx.promote_state_vector_to_density_matrix(states).matrix
    weights = jax.random.dirichlet(jax.random.key(44), jnp.ones(projectors.shape[0]), (10,))
    mixtures = qx.DensityMatrix.from_matrix(jnp.einsum("mk,kij->mij", weights, projectors), dims)
    assert jnp.allclose(qx.mana(mixtures), 0, atol=1e-12)


def test_mana_is_additive():
    a = qx.random_density_matrix(2, (3,), key=jax.random.key(45), size=(4,))
    b = qx.random_density_matrix(2, (5,), key=jax.random.key(46), size=(4,))
    assert jnp.allclose(qx.mana(a | b), qx.mana(a) + qx.mana(b), atol=1e-12)


def test_mana_of_qutrits_is_at_most_that_of_the_strange_state():
    states = qx.random_state_vector((3,), key=jax.random.key(47), size=(200,))
    assert jnp.all(qx.mana(states) <= np.log(5 / 3) + 1e-12)


def test_mana_drops_under_partial_trace():
    rho = qx.random_density_matrix(2, (3, 3), key=jax.random.key(48), size=(5,))
    reduced = qx.DensityMatrix.from_matrix(jnp.einsum("nabcb->nac", rho.data), (3,))
    assert jnp.all(qx.mana(reduced) <= qx.mana(rho) + 1e-12)


def test_mana_dispatch_jit_grad_and_errors():
    psi = qx.random_state_vector((3, 5), key=jax.random.key(49), size=(2,))
    rho = qx.promote_state_vector_to_density_matrix(psi)
    assert jnp.allclose(qx.mana(psi), qx.mana(rho))
    assert jnp.allclose(jax.jit(qx.mana)(rho), qx.mana(rho))

    def mana_of_mixed_strange(p):
        strange = _projector(_state(0, 1, -1))
        return qx.mana(qx.DensityMatrix.from_matrix((1 - p) * strange + p * jnp.eye(3) / 3, (3,)))

    # sn = (1 - p)/3 - p/9, so the mana is log(5/3 - 8p/9).
    assert jnp.isclose(jax.grad(mana_of_mixed_strange)(0.2), (-8 / 9) / (5 / 3 - 8 * 0.2 / 9))
    with pytest.raises(ValueError):
        qx.mana(qx.random_density_matrix(2, (2, 3), key=jax.random.key(50)))
    with pytest.raises(TypeError):
        qx.mana(qx.random_observable(((3,), (3,)), key=jax.random.key(51)))


def _qutrit_cliffords():
    j = np.arange(3)
    fourier = np.exp(2j * np.pi * np.outer(j, j) / 3) / np.sqrt(3)
    phase = np.diag(np.exp(2j * np.pi * j * j / 3))
    return fourier, phase


def _unitary(matrix, dims):
    return qx.Unitary.from_matrix(jnp.asarray(matrix, dtype=complex), (dims, dims))


def _channel_wigner_by_definition(superop):
    """W(v|u) = Tr[A_v N(A_u)] / D_out, applying the superoperator to each input phase-point operator."""
    (d_out,), (d_in,) = superop.dims
    points_in = _gross_point_operators(d_in).reshape(d_in * d_in, d_in, d_in)
    points_out = _gross_point_operators(d_out).reshape(d_out * d_out, d_out, d_out)
    images = (np.asarray(superop.matrix) @ points_in.transpose(0, 2, 1).reshape(d_in * d_in, -1).T).T
    images = images.reshape(-1, d_out, d_out).transpose(0, 2, 1)  # unvec, column-stacked
    table = np.einsum("vij,uji->vu", points_out, images).real / d_out
    return table.reshape(d_out, d_out, d_in, d_in)


def test_channel_wigner_function_of_familiar_qutrit_channels():
    identity = _unitary(np.eye(3), (3,))
    shift = _unitary(np.roll(np.eye(3), 1, axis=0), (3,))

    assert jnp.allclose(qx.wigner_function(identity).reshape(9, 9), jnp.eye(9))
    assert jnp.allclose(qx.wigner_function(shift).reshape(9, 9), jnp.roll(jnp.eye(9), 3, axis=0))
    assert jnp.isclose(qx.mana(identity), 0)


def test_channel_wigner_function_matches_the_definition():
    unitary = qx.random_unitary(dims=((5,), (5,)), key=jax.random.key(52))
    superop = qx.to_superop(unitary)
    assert jnp.allclose(qx.wigner_function(unitary), _channel_wigner_by_definition(superop), atol=1e-12)
    isometry, _ = np.linalg.qr(
        np.random.default_rng(53).normal(size=(5, 3)) + 1j * np.random.default_rng(54).normal(size=(5, 3))
    )
    embedding = qx.SuperOp.from_matrix(jnp.asarray(np.kron(isometry.conj(), isometry)), ((5,), (3,)))
    table = qx.wigner_function(embedding)
    assert table.shape == (5, 5, 3, 3)
    assert jnp.allclose(table, _channel_wigner_by_definition(embedding), atol=1e-12)
    assert jnp.allclose(jnp.sum(table, axis=(0, 1)), 1)


@pytest.mark.parametrize("dims", [(3,), (5,), (3, 3)], ids=str)
def test_channel_wigner_function_propagates_wigner_functions(dims):
    channel = qx.choi_to_superop(qx.random_choi((dims, dims), rank=2, key=jax.random.key(55)))
    rho = qx.random_density_matrix(2, dims, key=jax.random.key(56))
    n = len(dims)
    propagated = jnp.tensordot(qx.wigner_function(channel), qx.wigner_function(rho), axes=2 * n)
    assert jnp.allclose(propagated, qx.wigner_function(channel @ rho), atol=1e-12)


def test_channel_wigner_function_agrees_across_representations():
    unitary = qx.random_unitary(dims=((3,), (3,)), key=jax.random.key(57))
    expected = qx.wigner_function(unitary)
    for channel in [qx.to_superop(unitary), qx.to_choi(unitary), qx.to_kraus(unitary), qx.to_pauli_liouville(unitary)]:
        assert jnp.allclose(qx.wigner_function(channel), expected, atol=1e-12)
        assert jnp.isclose(qx.mana(channel), qx.mana(unitary))


def test_channel_mana_reduces_to_the_state_for_a_replacer_channel():
    strange = _projector(_state(0, 1, -1))
    replacer = qx.SuperOp.from_matrix(jnp.outer(strange.T.reshape(-1), jnp.eye(3).T.reshape(-1)), ((3,), (3,)))
    assert jnp.isclose(qx.mana(replacer), np.log(5 / 3))


def test_channel_mana_is_zero_on_cliffords_and_positive_otherwise():
    fourier, phase = _qutrit_cliffords()
    for gate in [fourier, phase, fourier @ phase, np.kron(fourier, phase)]:
        dims = (3,) if len(gate) == 3 else (3, 3)
        assert jnp.isclose(qx.mana(_unitary(gate, dims)), 0, atol=1e-12)
    t_gate = np.diag(np.exp(2j * np.pi * np.array([0, 1, -1]) / 9))
    assert qx.mana(_unitary(t_gate, (3,))) > 1e-3


def test_channel_mana_is_additive_and_subadditive_under_composition():
    a = qx.random_unitary(dims=((3,), (3,)), key=jax.random.key(58))
    b = qx.random_unitary(dims=((3,), (3,)), key=jax.random.key(59))
    c = qx.random_unitary(dims=((5,), (5,)), key=jax.random.key(60))
    assert jnp.isclose(qx.mana(a | c), qx.mana(a) + qx.mana(c))
    assert qx.mana(b @ a) <= qx.mana(a) + qx.mana(b) + 1e-12


def test_channel_dispatch_jit_base_and_errors():
    unitary = qx.random_unitary(dims=((3,), (3,)), key=jax.random.key(61), size=(4,))
    assert qx.wigner_function(unitary).shape == (4, 3, 3, 3, 3)
    assert jnp.allclose(jax.jit(qx.mana)(unitary), qx.mana(unitary))
    assert jnp.allclose(qx.mana(unitary, base=3), qx.mana(unitary) / np.log(3))
    with pytest.raises(ValueError):
        qx.mana(qx.random_unitary(dims=((2,), (2,)), key=jax.random.key(62)))
