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

"""Tests for ``quax.squeeze`` (per-subsystem dimension reduction)."""

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
# Unitaries / operators
# --------------------------------------------------------------------------


def test_unitary_drops_identity_level():
    """H embedded as identity on |2> squeezes back to the 2x2 gate."""
    u3 = qx.promote(qx.Unitary.from_matrix(H, ((2,), (2,))), (3,))
    sq = qx.squeeze(u3, tol=1e-9)
    assert sq.dims == ((2,), (2,))
    assert jnp.allclose(sq.matrix, H)


def test_unitary_keeps_coupled_level():
    """A 1<->2 swap couples level 2 and must not be squeezed."""
    m = jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    u = qx.Unitary.from_matrix(m, ((3,), (3,)))
    assert qx.squeeze(u, tol=1e-9).dims == ((3,), (3,))


# --------------------------------------------------------------------------
# Channels
# --------------------------------------------------------------------------


def test_kraus_promote_squeeze_roundtrip():
    k = qx.KrausMap.from_matrix(H[None], ((2,), (2,)))
    k3 = qx.promote(k, (3,))
    sq = qx.squeeze(k3, tol=1e-9)
    assert sq.dims == ((2,), (2,))
    assert jnp.allclose(sq.matrix[0], H)


def test_kraus_keeps_leakage_coupling():
    """An amplitude-damping-style |1><2| Kraus op keeps level 2."""
    g = 0.3
    k0 = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, np.sqrt(1 - g)]], dtype=complex)
    k1 = jnp.array([[0, 0, 0], [0, 0, np.sqrt(g)], [0, 0, 0]], dtype=complex)  # |1><2|
    k = qx.KrausMap.from_matrix(jnp.stack([k0, k1]), ((3,), (3,)))
    assert qx.squeeze(k, tol=1e-9).dims == ((3,), (3,))


def test_superop_squeezes_via_kraus():
    s = qx.to_superop(qx.Unitary.from_matrix(H, ((2,), (2,))))
    s3 = qx.promote(s, (3,))
    sq = qx.squeeze(s3, tol=1e-9)
    assert sq.dims == ((2,), (2,))
    assert jnp.allclose(sq.matrix, s.matrix, atol=1e-6)


def test_quantum_instrument_squeeze_raises():
    inst = qx.gates.MEASURE(dim=2)
    with pytest.raises(NotImplementedError):
        qx.squeeze(inst)
