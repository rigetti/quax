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
