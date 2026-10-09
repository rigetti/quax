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
"""Tests for the Weyl characteristic functions, against their definitions."""

from functools import reduce
from itertools import product
from operator import mul

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quax as qx

DIMS = [(2,), (3,), (4,), (5,), (2, 2), (2, 3), (3, 2), (3, 3), (2, 2, 2), (4, 3)]
ENSEMBLES = [(), (3,), (2, 3)]


def _d(dims):
    return reduce(mul, dims, 1)


def _purity(rho):
    return jnp.real(jnp.einsum("...ij,...ji->...", rho.matrix, rho.matrix))


def _shift_and_clock(d):
    return np.roll(np.eye(d), 1, axis=0), np.diag(np.exp(2j * np.pi * np.arange(d) / d))


def _weyl_operators(dims):
    """W_{x,z} = prod_i tau_i^{x_i z_i} X^x Z^z in n_qudit_basis order.

    Built from the public labels so the test needs no private module.
    """
    per_qudit = []
    for d in dims:
        X, Z = _shift_and_clock(d)
        tau = (-1) ** d * np.exp(1j * np.pi / d)
        ops = []
        for label in qx.weyl_basis_labels(d):
            x, z = int(label[1]), int(label[2])
            ops.append(tau ** (x * z) * np.linalg.matrix_power(X, x) @ np.linalg.matrix_power(Z, z))
        per_qudit.append(np.array(ops))
    return reduce(
        lambda a, b: np.einsum("aij,bkl->abikjl", a, b).reshape(-1, a.shape[1] * b.shape[1], a.shape[2] * b.shape[2]),
        per_qudit,
    )


def _gross_weyl_operators(d):
    """Gross Eq. (3): w(p, q) = omega^{-2^{-1} p q} Z^p X^q.

    Labelled by (x, z) = (q, p) in weyl_basis_labels order.
    """
    X, Z = _shift_and_clock(d)
    inv2 = (d + 1) // 2
    omega = np.exp(2j * np.pi / d)
    ops = []
    for label in qx.weyl_basis_labels(d):
        q, p = int(label[1]), int(label[2])
        ops.append(omega ** (-inv2 * p * q) * np.linalg.matrix_power(Z, p) @ np.linalg.matrix_power(X, q))
    return np.array(ops)


@pytest.mark.parametrize("dims", DIMS, ids=str)
@pytest.mark.parametrize("ensemble", ENSEMBLES, ids=str)
def test_matches_the_definition_elementwise(dims, ensemble):
    rho = qx.random_density_matrix(_d(dims), dims, key=jax.random.key(1), size=ensemble)
    chi = qx.characteristic_function(rho)
    assert chi.shape == ensemble + (_d(dims) ** 2,)
    assert jnp.iscomplexobj(chi)
    expected = jnp.einsum("...ij,kji->...k", rho.matrix, _weyl_operators(dims))
    assert jnp.allclose(chi, expected, atol=1e-12)


@pytest.mark.parametrize("dims", DIMS, ids=str)
def test_weyl_operators_are_the_basis_up_to_the_phase(dims):
    """The Weyl operators behind chi equal n_qudit_basis up to a phase."""
    weyl = _weyl_operators(dims)
    basis = np.asarray(qx.n_qudit_basis(dims).matrix)
    ratios = np.einsum("kji,kji->k", basis.conj(), weyl) / _d(dims)
    assert np.allclose(np.abs(ratios), 1.0)
    assert np.allclose(np.abs(np.einsum("kij,kij->k", weyl.conj(), weyl)), _d(dims))
    if set(dims) == {2}:
        assert np.allclose(ratios, 1.0)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 5])
def test_qubits_give_the_pauli_vector(n_qubits):
    rho = qx.random_density_matrix(2, (2,) * n_qubits, key=jax.random.key(2), size=(2,))
    chi = qx.characteristic_function(rho)
    assert jnp.allclose(jnp.imag(chi), 0.0, atol=1e-12)
    assert jnp.allclose(jnp.real(chi), qx.to_pauli_vector(rho), atol=1e-12)


@pytest.mark.parametrize("dims", DIMS, ids=str)
def test_reconstruction_and_parseval(dims):
    rho = qx.random_density_matrix(2, dims, key=jax.random.key(3))
    chi = qx.characteristic_function(rho)
    weyl = _weyl_operators(dims)
    back = jnp.einsum("k,kji->ij", chi, weyl.conj()) / _d(dims)
    assert jnp.allclose(back, rho.matrix, atol=1e-12)
    assert jnp.isclose(jnp.sum(jnp.abs(chi) ** 2), _d(dims) * _purity(rho))


@pytest.mark.parametrize("d", [3, 5, 7])
def test_odd_dimensions_use_gross_weyl_operators(d):
    """Our W_{x,z} are Gross Eq. (3), and conj(chi)/d is his Xi: rho = sum_a Xi(a) w(a)."""
    rho = qx.random_density_matrix(d, (d,), key=jax.random.key(4), size=(2,))
    gross = _gross_weyl_operators(d)
    assert jnp.allclose(gross, _weyl_operators((d,)), atol=1e-12)
    xi = jnp.conj(qx.characteristic_function(rho)) / d
    assert jnp.allclose(xi, jnp.einsum("...ij,kij->...k", rho.matrix, gross.conj()) / d, atol=1e-12)
    assert jnp.allclose(jnp.einsum("...k,kij->...ij", xi, gross), rho.matrix, atol=1e-12)


@pytest.mark.parametrize("dims", DIMS, ids=str)
@pytest.mark.parametrize("ensemble", ENSEMBLES, ids=str)
def test_characteristic_distribution(dims, ensemble):
    rho = qx.random_density_matrix(2, dims, key=jax.random.key(8), size=ensemble)
    p = qx.characteristic_distribution(rho)
    assert p.shape == ensemble + (_d(dims) ** 2,)
    assert not jnp.iscomplexobj(p)
    assert jnp.all(p >= 0)
    assert jnp.allclose(p.sum(axis=-1), _purity(rho))
    assert jnp.allclose(p, jnp.abs(qx.characteristic_function(rho)) ** 2 / _d(dims))
    psi = qx.random_state_vector(dims, key=jax.random.key(9), size=ensemble)
    assert jnp.allclose(qx.characteristic_distribution(psi).sum(axis=-1), 1.0)


@pytest.mark.parametrize("dims", [(2,), (2, 2, 2), (3,), (3, 3), (5,)], ids=str)
def test_stabilizer_states_are_uniform_on_their_stabilizer_group(dims):
    """GNW Eq. (3.3): p = 1/D on the D stabilizing Weyl operators, 0 elsewhere."""
    d = _d(dims)
    zero = qx.zero_state_vector(dims=dims)
    p = np.asarray(qx.characteristic_distribution(zero))
    assert np.isclose(p.max(), 1 / d)
    assert np.sum(np.isclose(p, 1 / d)) == d
    assert np.sum(np.isclose(p, 0.0)) == d * d - d
    # Z^z |0> = |0>, so exactly the (0, z) labels carry weight.
    labels = ["".join(ls) for ls in product(*(qx.weyl_basis_labels(k) for k in dims))]
    supported = {label for label, weight in zip(labels, p) if weight > 1e-12}
    assert supported == {label for label in labels if all(ch == "0" for ch in label[1::3])}


def test_qubit_characteristic_distribution_is_the_sre_distribution():
    """Leone et al.: Xi_P(psi) = <psi|P|psi>^2 / d over the Pauli strings."""
    psi = qx.random_state_vector((2, 2), key=jax.random.key(10))
    paulis = np.asarray(qx.n_qudit_basis((2, 2)).matrix)
    expectations = jnp.einsum("i,kij,j->k", jnp.conj(psi.matrix), paulis, psi.matrix).real
    assert jnp.allclose(qx.characteristic_distribution(psi), expectations**2 / 4, atol=1e-12)


def test_state_vector_and_observable_dispatch():
    psi = qx.random_state_vector((2, 3), key=jax.random.key(11), size=(2,))
    rho = qx.promote_state_vector_to_density_matrix(psi)
    assert jnp.allclose(qx.characteristic_function(psi), qx.characteristic_function(rho))
    observable = qx.random_observable(((2, 3), (2, 3)), key=jax.random.key(12))
    expected = jnp.einsum("ij,kji->k", observable.matrix, _weyl_operators((2, 3)))
    assert jnp.allclose(qx.characteristic_function(observable), expected, atol=1e-12)


def test_rejects_unsupported_inputs():
    rectangular = qx.Observable.from_matrix(jnp.zeros((3, 2), dtype=complex), ((3,), (2,)))
    with pytest.raises(ValueError):
        qx.characteristic_function(rectangular)
    with pytest.raises(TypeError):
        qx.characteristic_function(qx.random_unitary(dims=((2,), (2,)), key=jax.random.key(14)))
    with pytest.raises(TypeError):
        qx.characteristic_distribution(jnp.eye(2))


def test_jit_and_grad():
    dims = (3, 2)
    rho = qx.random_density_matrix(6, dims, key=jax.random.key(13))
    for f in (qx.characteristic_function, qx.characteristic_distribution):
        assert jnp.allclose(jax.jit(f)(rho), f(rho))

    def purity_from_bloch(bloch):
        return qx.characteristic_distribution(qx.DensityMatrix.from_bloch_vector(bloch, dims)).sum()

    gradient = jax.grad(purity_from_bloch)(rho.bloch_vector)
    # sum p = Tr[rho^2] = (1 + |bloch|^2) / d
    assert jnp.allclose(gradient, 2 * rho.bloch_vector / 6, atol=1e-10)
