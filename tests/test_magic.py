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
