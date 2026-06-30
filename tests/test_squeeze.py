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

"""Tests for ``quax.squeeze`` (per-subsystem dimension reduction of states).

``squeeze`` is defined only for states (StateVector, DensityMatrix); for
operators, channels and instruments it is mathematically ill-posed and must
raise ``TypeError``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import quax as qx

H = jnp.array([[1, 1], [1, -1]], dtype=complex) / jnp.sqrt(2)


# --------------------------------------------------------------------------
# State vectors
# --------------------------------------------------------------------------


def test_state_drops_empty_qutrit_levels():
    """|00> stored on two qutrits squeezes to two qubits."""
    psi = qx.zero_state_vector(dims=(3, 3))
    sq = qx.squeeze(psi, tol=1e-9)
    assert sq.dims == (2, 2)
    assert jnp.allclose(sq.matrix, qx.zero_state_vector(dims=(2, 2)).matrix)


def test_state_keeps_populated_level():
    """A qutrit with all population in |2> is not squeezed."""
    vec = jnp.array([0.0, 0.0, 1.0], dtype=complex)
    psi = qx.StateVector.from_matrix(vec, (3,))
    assert qx.squeeze(psi, tol=1e-9).dims == (3,)


def test_state_drops_subthreshold_leakage_and_renormalizes():
    """Population below tol is dropped and the state is renormalized."""
    amp = jnp.array([np.sqrt(1 - 1e-12), 0.0, np.sqrt(1e-12)], dtype=complex)
    psi = qx.StateVector.from_matrix(amp, (3,))
    sq = qx.squeeze(psi, tol=1e-9)
    assert sq.dims == (2,)
    assert jnp.allclose(jnp.sum(jnp.abs(sq.matrix) ** 2), 1.0)


def test_state_promote_squeeze_roundtrip():
    psi2 = qx.StateVector.from_matrix(jnp.array([0.6, 0.8], dtype=complex), (2,))
    psi3 = qx.promote(psi2, (3,))
    assert jnp.allclose(qx.squeeze(psi3, tol=1e-9).matrix, psi2.matrix)


def test_state_per_subsystem_independent():
    """Only the empty subsystem is squeezed."""
    # subsystem 0 occupies |2>, subsystem 1 only |0>,|1>
    vec = np.zeros((3, 3), dtype=complex)
    vec[2, 0] = np.sqrt(0.5)
    vec[2, 1] = np.sqrt(0.5)
    psi = qx.StateVector.from_matrix(jnp.array(vec.reshape(9)), (3, 3))
    sq = qx.squeeze(psi, tol=1e-9)
    assert sq.dims == (3, 2)


# --------------------------------------------------------------------------
# Density matrices
# --------------------------------------------------------------------------


def test_density_matrix_drops_empty_qutrit_levels():
    """|00><00| on two qutrits squeezes to two qubits."""
    rho = qx.zero_state_matrix(dims=(3, 3))
    sq = qx.squeeze(rho, tol=1e-9)
    assert sq.dims == (2, 2)
    assert jnp.allclose(sq.matrix, qx.zero_state_matrix(dims=(2, 2)).matrix)


def test_density_matrix_keeps_populated_level():
    """A qutrit with population in |2> is not squeezed."""
    rho = qx.DensityMatrix.from_matrix(jnp.diag(jnp.array([0.0, 0.0, 1.0], dtype=complex)), (3,))
    assert qx.squeeze(rho, tol=1e-9).dims == (3,)


def test_density_matrix_drops_subthreshold_and_renormalizes_trace():
    """A subthreshold |2> population is dropped and the trace renormalized to 1."""
    rho = qx.DensityMatrix.from_matrix(jnp.diag(jnp.array([0.7, 0.3 - 1e-12, 1e-12], dtype=complex)), (3,))
    sq = qx.squeeze(rho, tol=1e-9)
    assert sq.dims == (2,)
    assert jnp.allclose(jnp.trace(sq.matrix), 1.0)


def test_density_matrix_promote_squeeze_roundtrip():
    rho2 = qx.DensityMatrix.from_matrix(jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex), (2,))
    rho3 = qx.promote(rho2, (3,))
    assert jnp.allclose(qx.squeeze(rho3, tol=1e-9).matrix, rho2.matrix)


# --------------------------------------------------------------------------
# Non-states: squeeze is undefined and must raise
# --------------------------------------------------------------------------


def test_squeeze_unitary_raises():
    u = qx.promote(qx.Unitary.from_matrix(H, ((2,), (2,))), (3,))
    with pytest.raises(TypeError):
        qx.squeeze(u)


def test_squeeze_operator_raises():
    op = qx.Operator.from_matrix(jnp.eye(3, dtype=complex), ((3,), (3,)))
    with pytest.raises(TypeError):
        qx.squeeze(op)


def test_squeeze_kraus_map_raises():
    k = qx.KrausMap.from_matrix(H[None], ((2,), (2,)))
    with pytest.raises(TypeError):
        qx.squeeze(k)


def test_squeeze_superop_raises():
    s = qx.to_superop(qx.Unitary.from_matrix(H, ((2,), (2,))))
    with pytest.raises(TypeError):
        qx.squeeze(s)


def test_squeeze_choi_raises():
    c = qx.to_choi(qx.Unitary.from_matrix(H, ((2,), (2,))))
    with pytest.raises(TypeError):
        qx.squeeze(c)


def test_squeeze_pauli_liouville_raises():
    pl = qx.to_pauli_liouville(qx.Unitary.from_matrix(H, ((2,), (2,))))
    with pytest.raises(TypeError):
        qx.squeeze(pl)


def test_squeeze_quantum_instrument_raises():
    inst = qx.gates.MEASURE(dim=2)
    with pytest.raises(TypeError):
        qx.squeeze(inst)
