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
"""Tests for the Pauli-vector representation of states and observables."""

from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp
import pytest

import quax as qx

DIMS = [(2,), (2, 2), (3,), (2, 3), (2, 2, 2)]
ENSEMBLES = [(), (3,), (2, 3)]


def _d(dims):
    return reduce(mul, dims, 1)


@pytest.mark.parametrize("dims", DIMS, ids=str)
@pytest.mark.parametrize("ensemble", ENSEMBLES, ids=str)
def test_density_matrix_round_trip(dims, ensemble):
    rho = qx.random_density_matrix(_d(dims), dims, key=jax.random.key(1), size=ensemble)
    vector = qx.to_pauli_vector(rho)
    assert vector.shape == ensemble + (_d(dims) ** 2,)
    assert not jnp.iscomplexobj(vector)
    assert jnp.allclose(vector[..., 0], 1.0), "the first entry is the trace"
    back = qx.DensityMatrix.from_pauli_vector(vector, dims)
    assert jnp.allclose(back.matrix, rho.matrix, atol=1e-12)
    assert jnp.allclose(rho.pauli_vector, vector)
    assert jnp.allclose(rho.bloch_vector, vector[..., 1:])


@pytest.mark.parametrize("dims", DIMS, ids=str)
@pytest.mark.parametrize("ensemble", ENSEMBLES, ids=str)
def test_observable_round_trip(dims, ensemble):
    observable = qx.random_observable((dims, dims), key=jax.random.key(2), size=ensemble)
    vector = qx.to_pauli_vector(observable)
    assert vector.shape == ensemble + (_d(dims) ** 2,)
    assert jnp.allclose(vector[..., 0], jnp.real(jnp.trace(observable.matrix, axis1=-2, axis2=-1)))
    back = qx.Observable.from_pauli_vector(vector, dims)
    assert jnp.allclose(back.matrix, observable.matrix, atol=1e-12)
    assert jnp.allclose(observable.pauli_vector, vector)


@pytest.mark.parametrize("dims", DIMS, ids=str)
def test_matches_the_pauli_liouville_matrix(dims):
    """to_pauli_vector(S @ rho) == to_pauli_liouville(S).matrix @ to_pauli_vector(rho)."""
    superop = qx.choi_to_superop(qx.random_choi((dims, dims), rank=_d(dims), key=jax.random.key(3)))
    rho = qx.random_density_matrix(_d(dims), dims, key=jax.random.key(4), size=(5,))
    lhs = qx.to_pauli_vector(superop @ rho)
    rhs = qx.to_pauli_vector(rho) @ qx.to_pauli_liouville(superop).matrix.T
    assert jnp.allclose(lhs, rhs, atol=1e-12)


@pytest.mark.parametrize("dims", DIMS, ids=str)
def test_estimate_is_the_scaled_inner_product(dims):
    """estimate(rho, O) == to_pauli_vector(O) @ to_pauli_vector(rho) / d."""
    rho = qx.random_density_matrix(_d(dims), dims, key=jax.random.key(5), size=(4,))
    observable = qx.random_observable((dims, dims), key=jax.random.key(6), size=(4,))
    inner = jnp.sum(qx.to_pauli_vector(observable) * qx.to_pauli_vector(rho), axis=-1) / _d(dims)
    assert jnp.allclose(inner, qx.estimate(rho, observable), atol=1e-12)


def test_product_states_have_kronecker_pauli_vectors():
    a = qx.random_density_matrix(2, (2,), key=jax.random.key(7))
    b = qx.random_density_matrix(3, (3,), key=jax.random.key(8))
    assert jnp.allclose(qx.to_pauli_vector(a | b), jnp.kron(qx.to_pauli_vector(a), qx.to_pauli_vector(b)), atol=1e-12)


def test_qubit_bloch_vector():
    bloch = jnp.array([0.3, -0.4, 0.5])
    paulis = jnp.stack([qx.gates.X.matrix, qx.gates.Y.matrix, qx.gates.Z.matrix])
    rho = qx.DensityMatrix.from_matrix((jnp.eye(2) + jnp.einsum("k,kij->ij", bloch, paulis)) / 2, (2,))
    assert jnp.allclose(rho.bloch_vector, bloch, atol=1e-12)
    assert jnp.allclose(qx.DensityMatrix.from_bloch_vector(bloch).matrix, rho.matrix, atol=1e-12)
    assert jnp.allclose(qx.to_pauli_vector(qx.states.XPLUS), jnp.array([1.0, 1.0, 0.0, 0.0]), atol=1e-12)
    assert jnp.allclose(qx.to_pauli_vector(qx.states.YMINUS), jnp.array([1.0, 0.0, -1.0, 0.0]), atol=1e-12)
    assert jnp.allclose(qx.to_pauli_vector(qx.states.KET1), jnp.array([1.0, 0.0, 0.0, -1.0]), atol=1e-12)


def test_state_vector_agrees_with_its_density_matrix():
    psi = qx.random_state_vector((2, 3), key=jax.random.key(9), size=(2,))
    assert jnp.allclose(qx.to_pauli_vector(psi), qx.to_pauli_vector(qx.promote_state_vector_to_density_matrix(psi)))


def test_jit_and_grad():
    dims = (2, 2)
    rho = qx.random_density_matrix(4, dims, key=jax.random.key(10))
    assert jnp.allclose(jax.jit(qx.to_pauli_vector)(rho), qx.to_pauli_vector(rho))
    jitted_back = jax.jit(qx.DensityMatrix.from_pauli_vector, static_argnums=1)(qx.to_pauli_vector(rho), dims)
    assert jnp.allclose(jitted_back.matrix, rho.matrix, atol=1e-12)

    def purity_from_bloch(bloch):
        state = qx.DensityMatrix.from_bloch_vector(bloch, dims)
        return jnp.real(jnp.trace(state.matrix @ state.matrix))

    gradient = jax.grad(purity_from_bloch)(rho.bloch_vector)
    # Tr[rho^2] = (1 + |bloch|^2) / d, so the gradient is 2 bloch / d.
    assert jnp.allclose(gradient, 2 * rho.bloch_vector / 4, atol=1e-10)


def test_rejects_rectangular_operators():
    with pytest.raises(ValueError):
        qx.to_pauli_vector(qx.Observable.from_matrix(jnp.zeros((3, 2), dtype=complex), ((3,), (2,))))
    with pytest.raises(ValueError):
        qx.DensityMatrix.from_pauli_vector(jnp.zeros(3), (2,))
