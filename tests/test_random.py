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

import jax
import jax.numpy as jnp
import pytest
from jax.numpy import linalg as la

from quax import (
    is_one_design,
    is_two_design,
    is_unitary,
    random_choi,
    random_density_matrix,
    random_state_vector,
    random_unitary,
)

# =================================================================================================
# Test:  Ginibre state matrix
# =================================================================================================


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3, 4])
def test_ginibre_is_positive_operator(num_qubits, rank, qudit_dim):
    N_avg = 10
    keys = jax.random.split(jax.random.key(485), N_avg)
    dims = (qudit_dim,) * num_qubits
    d = qudit_dim ** len(dims)
    eigenvallist = []
    for k in keys:
        eigenval = la.eig(random_density_matrix(rank, dims=dims, key=k).matrix)[0]
        eigenvallist += [eigenval]
    eigenvalues = jnp.asarray(eigenvallist)
    eigenvalues = eigenvalues.reshape(1, d * N_avg)

    assert jnp.max(jnp.absolute(jnp.imag(eigenvalues))) < 1e-10
    assert jnp.min(jnp.real(eigenvalues)) >= -1e-10


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("qudit_dim", [2, 3, 4])
def test_ginibre_is_trace_one(num_qubits, rank, qudit_dim):
    N_avg = 100
    key = jax.random.key(485)
    keys = jax.random.split(key, N_avg)
    dims = (qudit_dim,) * num_qubits

    avg_trace = jnp.mean(jax.vmap(lambda k: jnp.trace(random_density_matrix(rank, dims=dims, key=k).matrix))(keys))
    assert avg_trace <= 1 + 1e-10
    assert avg_trace >= 1 - 1e-10


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_ginibre_has_correct_second_moment(num_qubits, qudit_dim):
    # Numerically calculate Eq. 3.20 from
    # Zyczkowski and Sommers, J. Phys. A: Math. Gen. 34 7111, (2001)
    #
    #  <Tr[rho^2])_{D,K} = ( D + K ) / ( D * K + 1 )
    #
    #  D is dimension of Hilbert space and K is rank of state matrix
    key = jax.random.key(485)
    N_avg = 5000

    dims = (qudit_dim,) * num_qubits
    d = qudit_dim ** len(dims)
    rank = d

    keys = jax.random.split(key, N_avg)

    rhos = jax.vmap(lambda k: random_density_matrix(rank, dims=dims, key=k))(keys)
    # rhos = jnp.asarray([ginibre_state_matrix_ref(d, rank) for _ in range(N_avg)])
    purities = jax.vmap(lambda rho: jnp.trace(jnp.matmul(rho.matrix, rho.matrix)))(rhos)
    avg_purity = jnp.mean(purities)

    ans = (d + rank) / (d * rank + 1)
    print(f"Average purity: {avg_purity}, expected: {ans}")
    assert jnp.absolute(avg_purity - ans) < 1e-2


# =================================================================================================
# Test:  Random Unitaries
# =================================================================================================


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3, 4])
def test_random_unitaries_are_unitary(num_qubits, qudit_dim):
    num_unitaries = (10, 20)
    dims = ((qudit_dim,) * num_qubits, (qudit_dim,) * num_qubits)
    d = qudit_dim**num_qubits

    unitaries = random_unitary(dims=dims, key=jax.random.key(1234), size=num_unitaries)

    assert unitaries.matrix.shape == num_unitaries + (d, d)

    assert all(jax.vmap(is_unitary)(unitaries.matrix.reshape(-1, d, d)))


@pytest.mark.parametrize("num_qubits", [1])
def test_random_unitaries_are_1_design(num_qubits):
    """This tests that random unitaries form a 1-design for 1-qubit systems."""
    num_unitaries = (12, 4096)
    qudit_dim = 2
    dims = ((qudit_dim,) * num_qubits, (qudit_dim,) * num_qubits)
    d = qudit_dim**num_qubits

    unitaries = random_unitary(dims=dims, key=jax.random.key(1234), size=num_unitaries)

    assert unitaries.matrix.shape == num_unitaries + (d, d)

    assert is_one_design(unitaries, atol=1e-2)


@pytest.mark.parametrize("num_qubits", [1])
def test_random_unitaries_are_2_design(num_qubits):
    """This tests that random unitaries form a 2-design for 1-qubit systems."""
    num_unitaries = (120, 4096)
    qudit_dim = 2
    dims = ((qudit_dim,) * num_qubits, (qudit_dim,) * num_qubits)
    d = qudit_dim**num_qubits

    unitaries = random_unitary(dims=dims, key=jax.random.key(1234), size=num_unitaries)

    assert unitaries.matrix.shape == num_unitaries + (d, d)

    assert is_two_design(unitaries, atol=1e-2)


# =================================================================================================
# Test: random Choi from BCSZ distribution
# =================================================================================================


@pytest.mark.parametrize("seed", [48573])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
def test_random_choi_BCSZ(seed, num_qubits, qudit_dim):
    """Test that the random Choi matrix from BCSZ distribution has correct shape and properties."""
    key = jax.random.key(seed)
    dims = ((qudit_dim,) * num_qubits, (qudit_dim,) * num_qubits)
    d = qudit_dim**num_qubits
    atol = 1e-8
    # Test size
    choi = random_choi(dims=dims, rank=d, key=key)
    assert choi.dims == dims
    assert choi.matrix.shape == (d * d, d * d)

    # Test positivity
    eigenvals = la.eigvalsh(choi.matrix)
    assert jnp.min(eigenvals) >= -atol

    # Test trace preservation
    J4 = choi.matrix.reshape((d, d, d, d))
    # partial trace over output: sum_k J[i,k,j,k]
    ptr = jnp.einsum("ikjk->ij", J4)
    assert jnp.allclose(ptr, jnp.eye(d, dtype=choi.matrix.dtype), atol=atol)


# =================================================================================================
# Test: random state vectors for qudits
# =================================================================================================


@pytest.mark.parametrize("qudit_dim", [2, 3, 4])
@pytest.mark.parametrize("num_qudits", [1, 2])
def test_random_state_vector_normalization(qudit_dim, num_qudits):
    """Test that random state vectors are normalized for arbitrary qudit dimensions."""
    dims = (qudit_dim,) * num_qudits
    d = qudit_dim**num_qudits
    N = 50
    keys = jax.random.split(jax.random.key(42), N)

    for k in keys:
        psi = random_state_vector(dims=dims, key=k)
        assert psi.matrix.shape == (d,)
        norm = jnp.sum(jnp.abs(psi.matrix) ** 2)
        assert jnp.allclose(norm, 1.0, atol=1e-10)


@pytest.mark.parametrize("dims", [((2, 3), (2, 3)), ((3, 4), (3, 4))])
def test_random_unitary_mixed_dims(dims):
    """Test random unitaries with mixed qudit dimensions."""
    from functools import reduce
    from operator import mul

    d = reduce(mul, dims[0])
    key = jax.random.key(999)

    U = random_unitary(dims=dims, key=key)
    assert U.matrix.shape == (d, d)
    assert U.dims == dims
    assert is_unitary(U.matrix)
