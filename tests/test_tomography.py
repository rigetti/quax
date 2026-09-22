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
"""Tests for linear-inversion tomography."""

import itertools
from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quax as qx


def _d(dims):
    return reduce(mul, dims, 1)


def _pauli_eigenstates(num_qubits: int) -> qx.DensityMatrix:
    """The 6**n products of single-qubit Pauli eigenstates, an informationally complete set."""
    singles = [qx.DensityMatrix.from_bloch_vector(sign * jnp.eye(3)[k]).matrix for k in range(3) for sign in (1, -1)]
    products = [reduce(jnp.kron, choice) for choice in itertools.product(singles, repeat=num_qubits)]
    return qx.DensityMatrix.from_matrix(jnp.stack(products), (2,) * num_qubits)


def _input_states(dims) -> qx.DensityMatrix:
    if all(d == 2 for d in dims):
        return _pauli_eigenstates(len(dims))
    # d**2 random states are informationally complete with probability one.
    return qx.random_density_matrix(_d(dims), dims, key=jax.random.key(11), size=(_d(dims) ** 2,))


def _traceless_observables(dims) -> qx.Observable:
    return qx.n_qudit_herm_basis(tuple(dims))[1:]


def _expectation_table(states: qx.DensityMatrix, observables: qx.Observable) -> jnp.ndarray:
    """``E[i, j] = Tr[O_j rho_i]``: :func:`qx.estimate` with the two ensembles indexed to broadcast."""
    return qx.estimate(states[:, None], observables[None])


def _random_hermiticity_preserving_map(dims, key, trace_preserving: bool) -> qx.SuperOp:
    """A random real Pauli-Liouville matrix, so a Hermiticity-preserving but generally non-CP map."""
    d2 = _d(dims) ** 2
    matrix = jax.random.normal(key, (d2, d2)) / d2
    if trace_preserving:
        matrix = matrix.at[0].set(jnp.eye(d2)[0])
    return qx.pauli_liouville_to_superop(qx.PauliLiouville.from_matrix(matrix, (dims, dims)))


def test_broadcast_expectation_table_matches_estimate():
    """The table the linear inversions expect is ``estimate`` broadcast over the two ensembles."""
    states = _pauli_eigenstates(1)
    observables = _traceless_observables((2,))
    table = _expectation_table(states, observables)
    assert table.shape == (6, 3)
    expected = jnp.array([[qx.estimate(states[i], observables[j]) for j in range(3)] for i in range(6)])
    assert jnp.allclose(table, expected)


@pytest.mark.parametrize("dims", [(2,), (2, 2), (3,)], ids=str)
def test_recovers_random_cptp_channel(dims):
    truth = qx.choi_to_superop(qx.random_choi((dims, dims), rank=_d(dims), key=jax.random.key(12)))
    states, observables = _input_states(dims), _traceless_observables(dims)
    table = _expectation_table(truth @ states, observables)
    estimate = qx.linear_inversion_process(states, observables, table)
    assert isinstance(estimate, qx.SuperOp)
    assert jnp.allclose(estimate.matrix, truth.matrix, atol=1e-10)


@pytest.mark.parametrize("dims", [(2,), (2, 2), (3,)], ids=str)
def test_recovers_non_cp_trace_preserving_map(dims):
    truth = _random_hermiticity_preserving_map(dims, jax.random.key(13), trace_preserving=True)
    states, observables = _input_states(dims), _traceless_observables(dims)
    table = _expectation_table(truth @ states, observables)
    estimate = qx.linear_inversion_process(states, observables, table)
    assert jnp.allclose(estimate.matrix, truth.matrix, atol=1e-10)


def test_trace_preserving_row_is_exact_under_noise():
    dims = (2, 2)
    truth = qx.choi_to_superop(qx.random_choi((dims, dims), rank=4, key=jax.random.key(14)))
    states, observables = _input_states(dims), _traceless_observables(dims)
    table = _expectation_table(truth @ states, observables)
    noisy = table + 0.05 * jax.random.normal(jax.random.key(15), table.shape)
    estimate = qx.to_pauli_liouville(qx.linear_inversion_process(states, observables, noisy))
    assert jnp.allclose(estimate.matrix[0], jnp.eye(16)[0], atol=1e-14)
    assert not jnp.allclose(estimate.matrix[1:], qx.to_pauli_liouville(truth).matrix[1:], atol=1e-3)


def test_general_solve_recovers_a_non_trace_preserving_map():
    dims = (2,)
    truth = _random_hermiticity_preserving_map(dims, jax.random.key(16), trace_preserving=False)
    states = _input_states(dims)
    observables = qx.n_qudit_herm_basis(dims)  # traceful observables see the identity row
    table = _expectation_table(truth @ states, observables)
    estimate = qx.linear_inversion_process(states, observables, table, trace_preserving=False)
    assert jnp.allclose(estimate.matrix, truth.matrix, atol=1e-10)


def _dense_least_squares(states, observables, table, d):
    """The minimum-norm least-squares solution of the vectorized design, for comparison."""
    r = np.asarray(qx.to_pauli_vector(states))
    c = np.asarray(qx.to_pauli_vector(observables))[:, 1:]
    design = np.einsum("ja,ib->ijab", c, r).reshape(-1, c.shape[1] * r.shape[1]) / d
    target = (np.asarray(table) - np.outer(r[:, 0], np.asarray(qx.to_pauli_vector(observables))[:, 0]) / d).reshape(-1)
    rows = np.linalg.lstsq(design, target, rcond=None)[0].reshape(c.shape[1], r.shape[1])
    return np.concatenate([np.eye(r.shape[1])[:1], rows])


@pytest.mark.parametrize("complete", [True, False])
def test_matches_the_dense_least_squares_solution(complete):
    dims = (2,)
    truth = qx.choi_to_superop(qx.random_choi((dims, dims), rank=2, key=jax.random.key(17)))
    states = _input_states(dims)
    if not complete:
        states = states[jnp.array([0, 2, 4])]  # |+x>, |+y>, |+z> only: not informationally complete
    observables = _traceless_observables(dims)
    table = _expectation_table(truth @ states, observables)
    noisy = table + 0.02 * jax.random.normal(jax.random.key(18), table.shape)
    estimate = qx.to_pauli_liouville(qx.linear_inversion_process(states, observables, noisy))
    assert np.allclose(np.asarray(estimate.matrix), _dense_least_squares(states, observables, noisy, 2), atol=1e-10)


def test_batched_tables_give_an_ensemble():
    dims = (2,)
    truth = qx.choi_to_superop(qx.random_choi((dims, dims), rank=2, key=jax.random.key(19), size=(3,)))
    states, observables = _input_states(dims), _traceless_observables(dims)
    tables = jnp.stack([_expectation_table(truth[k] @ states, observables) for k in range(3)])
    estimate = qx.linear_inversion_process(states, observables, tables)
    assert estimate.ensemble_size == (3,)
    assert jnp.allclose(estimate.matrix, truth.matrix, atol=1e-10)


def test_jit_and_grad():
    dims = (2,)
    states, observables = _input_states(dims), _traceless_observables(dims)
    truth = qx.choi_to_superop(qx.random_choi((dims, dims), rank=2, key=jax.random.key(20)))
    table = _expectation_table(truth @ states, observables)
    # The function carries jax.jit itself; calling it inside another jitted function still traces.
    assert isinstance(qx.linear_inversion_process, jax.stages.Wrapped)
    assert isinstance(qx.linear_inversion_state, jax.stages.Wrapped)
    jitted = jax.jit(lambda t: qx.linear_inversion_process(states, observables, t))(table)
    assert jnp.allclose(jitted.matrix, truth.matrix, atol=1e-10)

    def distance(values):
        estimate = qx.linear_inversion_process(states, observables, values)
        difference = estimate.matrix - truth.matrix
        # |z|**2 as z z*, whose gradient is finite at the exactly-zero entries of the superoperator
        # (the identity row structure), where d|z| is not.
        return jnp.real(jnp.sum(difference * jnp.conj(difference)))

    noisy = table + 0.01 * jax.random.normal(jax.random.key(24), table.shape)
    gradient = jax.grad(distance)(noisy)
    assert jnp.all(jnp.isfinite(gradient))
    # Linear inversion is linear in the table, so one Newton step reaches the truth; the Hessian is
    # singular (more table entries than channel parameters), so take the minimum-norm step.
    hessian = jax.hessian(distance)(noisy).reshape(noisy.size, noisy.size)
    corrected = noisy.reshape(-1) - jnp.linalg.lstsq(hessian, gradient.reshape(-1))[0]
    assert distance(corrected.reshape(noisy.shape)) < 1e-20


@pytest.mark.parametrize("dims", [(2,), (2, 2), (3,)], ids=str)
def test_state_recovery(dims):
    rho = qx.random_density_matrix(_d(dims), dims, key=jax.random.key(21))
    observables = _traceless_observables(dims)
    values = qx.estimate(rho, observables)
    estimate = qx.linear_inversion_state(observables, values)
    assert isinstance(estimate, qx.DensityMatrix)
    assert jnp.allclose(estimate.matrix, rho.matrix, atol=1e-10)
    # Traceful observables can determine the trace, so the general solve recovers it too.
    full = qx.n_qudit_herm_basis(tuple(dims))
    general = qx.linear_inversion_state(full, qx.estimate(0.7 * rho, full), unit_trace=False)
    assert jnp.allclose(general.matrix, 0.7 * rho.matrix, atol=1e-10)


def test_state_recovery_batched():
    dims = (2,)
    rho = qx.random_density_matrix(2, dims, key=jax.random.key(22), size=(4,))
    observables = _traceless_observables(dims)
    values = jnp.stack([qx.estimate(rho[k], observables) for k in range(4)])
    estimate = qx.linear_inversion_state(observables, values)
    assert estimate.ensemble_size == (4,)
    assert jnp.allclose(estimate.matrix, rho.matrix, atol=1e-10)


def test_deprecated_observable_tables_still_agree_and_warn():
    dims = (2,)
    truth = qx.choi_to_superop(qx.random_choi((dims, dims), rank=2, key=jax.random.key(23)))
    states, observables = _input_states(dims), _traceless_observables(dims)
    table = _expectation_table(truth @ states, observables)
    with pytest.warns(DeprecationWarning):
        old = qx.compute_superop_observables_from_states(truth, states, observables)
    assert jnp.allclose(old, table)
