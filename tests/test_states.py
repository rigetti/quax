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

"""Tests for the lazy, precision-aware :mod:`quax.states` submodule."""

import jax
import jax.numpy as jnp
import pytest

import quax as qx


def test_states_are_attributes_not_functions():
    assert isinstance(qx.states.KET0, qx.StateVector)
    assert not callable(qx.states.KET0)
    assert "KET0" in dir(qx.states)
    assert hasattr(qx.states, "XPLUS")
    with pytest.raises(AttributeError):
        _ = qx.states.NOT_A_REAL_STATE


def test_state_aliases_agree():
    """Aliases such as KET0/ZPLUS and KETPLUS/XPLUS refer to the same state."""
    assert jnp.allclose(qx.states.KET0.matrix, qx.states.ZPLUS.matrix)
    assert jnp.allclose(qx.states.KETPLUS.matrix, qx.states.XPLUS.matrix)
    assert jnp.allclose(qx.states.SIC0.matrix, qx.states.KET0.matrix)


def test_states_use_current_precision():
    """States build at the active precision (``complex128`` under ``JAX_ENABLE_X64=1``).

    Precision-*change* behaviour is covered in isolation by
    ``test_gates.py::test_lazy_constants_track_precision_change`` (a subprocess test, to avoid
    clearing JAX's compilation caches mid-suite).
    """
    expected = jax.dtypes.canonicalize_dtype(jnp.complex128)
    assert qx.states.KET0.matrix.dtype == expected
    assert qx.states.SIC1.matrix.dtype == expected  # derived from KET0/KET1
    assert qx.states.STATES["SIC"][1].matrix.dtype == expected  # dict aggregate
    assert qx.states.PAULI_STATES["X+"].matrix.dtype == expected
