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
"""Tests for the qudit operator bases."""

import jax
import numpy as np
import pytest

import quax as qx

BASES = [(qx.n_qudit_basis, qx.weyl_basis), (qx.n_qudit_herm_basis, qx.hermitian_weyl_basis)]


@pytest.mark.parametrize("dims", [(2,), (3,), (2, 2), (2, 3), (3, 2, 2), (4, 3)], ids=str)
@pytest.mark.parametrize("build, single", BASES, ids=["weyl", "hermitian"])
def test_tensor_basis_is_the_kronecker_product_of_single_qudit_bases(build, single, dims):
    # Qudit 0 is the most significant factor, matching np.kron.
    expected = [np.eye(1)]
    for d in dims:
        expected = [np.kron(a, b) for a in expected for b in np.asarray(single(d).matrix)]
    basis = np.asarray(build(dims).matrix)
    assert basis.dtype == np.complex128
    assert np.allclose(basis, np.array(expected), rtol=0, atol=1e-15)


def test_weyl_phases_of_a_qubit_and_a_qutrit():
    tau = -np.exp(1j * np.pi / 3)
    assert qx.weyl_basis_labels(2) == ["W00", "W10", "W11", "W01"]
    assert np.allclose(qx.weyl_phases((2,)), [1, 1, 1j, 1])
    assert qx.weyl_basis_labels(3) == ["W00", "W01", "W02", "W11", "W22", "W10", "W20", "W21", "W12"]
    assert np.allclose(qx.weyl_phases((3,)), [1, 1, 1, tau, tau**4, 1, 1, tau**2, tau**2])
    assert not qx.weyl_phases((3,)).flags.writeable


@pytest.mark.parametrize("dims", [(3,), (5,), (3, 5)], ids=str)
def test_weyl_phases_turn_the_basis_into_the_characteristic_function_operators(dims):
    weyl_operators = qx.weyl_phases(dims)[:, None, None] * np.asarray(qx.n_qudit_basis(dims).matrix)
    rho = qx.random_density_matrix(2, dims, key=jax.random.key(0))
    traces = np.einsum("kij,ji->k", weyl_operators, np.asarray(rho.matrix))
    assert np.allclose(traces, qx.characteristic_function(rho))
