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

"""Tests for power functions of quantum objects.

These tests validate power functions for all quantum object types:
- Integer and fractional powers for states (density matrices, state vectors)
- Integer and fractional powers for unitaries
- Integer and fractional powers for superoperators (Choi, SuperOp, Pauli-Liouville, Kraus)

For superoperators, fractional powers use the Lindbladian approach (exp(α·log(Φ))),
which preserves CPTP properties for infinitely divisible channels (e.g., depolarizing).
For non-divisible channels, fractional powers may not remain CPTP.
"""

from functools import reduce
from operator import mul
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qutip as qt
from scipy.linalg import fractional_matrix_power, expm

import quax as qx


# Entirely random CPTP maps are often not infinitely divisible,
# making fractional powers invalid. Instead, we generate
# random CPTP maps via Lindbladian exponentiation, which guarantees infinite divisibility.
def _random_lindbladian(
    dims: Tuple[int, ...],
    n_jumps: int = 2,
    h_scale: float = 1.0,
    gamma_scale: float = 1.0,
    size: Tuple[int, ...] = (),
):
    """Generate a random CPTP superoperator via Lindbladian exponentiation."""

    if size == ():
        num_superops = 1
    else:
        num_superops = reduce(mul, size)
    chois = []
    d = reduce(mul, dims)
    d2 = d * d
    for _ in range(num_superops):
        # Random Hamiltonian
        H = h_scale * qt.rand_herm(d)

        # Random jump operators
        c_ops = []
        for _ in range(n_jumps):
            L = qt.rand_herm(d) + 1j * qt.rand_herm(d)
            L = gamma_scale * L / np.sqrt(d)
            c_ops.append(L)

        # Lindbladian superoperator
        L_super = qt.liouvillian(H, c_ops)  # type: ignore
        t = 0.5
        E = (t * L_super).expm()  # guaranteed CPTP
        choi = qt.to_choi(E).full()
        chois.append(choi)

    data = jnp.asarray(chois).reshape(size + (d2, d2))
    return qx.Choi.from_matrix(data, (dims, dims))


@pytest.mark.parametrize("num_qudits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("power", [1, 2, 3, 0.5, 1.5])
@pytest.mark.parametrize("seed", [583, 3357])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (3, 4)])
def test_superoperator_powers(num_qudits, qudit_dim, power, seed, ensemble_size):
    """Test superoperator power functions."""
    if num_qudits == 3 and qudit_dim == 3:
        pytest.skip("3-qutrit superoperators (729x729 matrices) with ensembles cause OOM")
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)
    random_choi = _random_lindbladian((qudit_dim,) * num_qudits, size=ensemble_size)
    random_superop = qx.choi_to_superop(random_choi)
    random_kraus = qx.choi_to_kraus(random_choi)
    random_pauli_liouville = qx.choi_to_pauli_liouville(random_choi)

    # here we use scipy.linalg.fractional_matrix_power as a reference implementation
    powered_reference_superop = qx.SuperOp.from_matrix(
        jnp.asarray(fractional_matrix_power(np.array(random_superop.matrix), power)), dims
    )
    powered_reference_choi = qx.superop_to_choi(powered_reference_superop)

    # Test SuperOp power
    powered_superop = qx.power_superop(random_superop, power)
    fid = qx.process_fidelity(powered_superop, powered_reference_superop)

    assert jnp.allclose(qx.superop_to_choi(powered_superop).matrix, powered_reference_choi.matrix, atol=1e-6)
    assert jnp.allclose(fid, 1.0, atol=1e-6)

    # Test Choi power
    powered_choi = qx.power_choi(random_choi, power)
    fid = qx.process_fidelity(powered_choi, powered_reference_choi)
    assert jnp.allclose(powered_choi.matrix, powered_reference_choi.matrix, atol=1e-6)
    assert jnp.allclose(fid, 1.0, atol=1e-6)

    # Test Kraus power
    powered_kraus = qx.power_kraus(random_kraus, power)
    fid = qx.process_fidelity(qx.kraus_to_choi(powered_kraus), powered_reference_choi)
    assert jnp.allclose(fid, 1.0, atol=1e-6)
    assert jnp.allclose(qx.kraus_to_choi(powered_kraus).matrix, powered_reference_choi.matrix, atol=1e-5)

    # Test Pauli-Liouville power
    powered_pauli_liouville = qx.power_pauli_liouville(random_pauli_liouville, power)
    fid = qx.process_fidelity(qx.pauli_liouville_to_choi(powered_pauli_liouville), powered_reference_choi)
    assert jnp.allclose(fid, 1.0, atol=1e-6)
    assert jnp.allclose(
        qx.pauli_liouville_to_choi(powered_pauli_liouville).matrix, powered_reference_choi.matrix, atol=1e-6
    )


@pytest.mark.parametrize("num_qudits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("power", [1, 2, 3, 0.5, 1.5])
@pytest.mark.parametrize("seed", [583, 3357])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (3, 4)])
def test_power_unitarys(num_qudits, qudit_dim, power, seed, ensemble_size):
    """Test that unitary matrices are correctly powered."""
    key = jax.random.key(seed)
    d = qudit_dim**num_qudits
    dims = ((qudit_dim,) * num_qudits, (qudit_dim,) * num_qudits)

    # Generate random unitary
    random_U = qx.random_unitary(dims=dims, key=key, size=ensemble_size)

    # Compute power using our implementation
    powered_U = qx.power_unitary(random_U, power)

    # Use scipy as reference
    powered_reference = qx.Unitary.from_matrix(
        jnp.asarray(fractional_matrix_power(np.array(random_U.matrix), power)), dims
    )

    # Check they match
    assert jnp.allclose(powered_U.matrix, powered_reference.matrix, atol=1e-6)

    # Check unitarity is preserved (U^† @ U = I)
    result = powered_U.h @ powered_U
    identity_shape = (d, d)
    if ensemble_size:
        identity = jnp.eye(d)
        # Expand identity to match ensemble dimensions
        for _ in range(len(ensemble_size)):
            identity = identity[None, ...]
        identity = jnp.broadcast_to(identity, ensemble_size + identity_shape)
    else:
        identity = jnp.eye(d)

    assert jnp.allclose(result.matrix, identity, atol=1e-6)


@pytest.mark.parametrize("num_qudits", [1, 2, 3])
@pytest.mark.parametrize("qudit_dim", [2, 3])
@pytest.mark.parametrize("power", [1, 2, 3, 0.5, 1.5])
@pytest.mark.parametrize("seed", [583, 3357])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (3, 4)])
def test_density_matrix_powers(num_qudits, qudit_dim, power, seed, ensemble_size):
    """Test that density matrices are correctly powered."""
    key = jax.random.key(seed)
    # For DensityMatrix, dims is a single tuple of subsystem dimensions
    dims = (qudit_dim,) * num_qudits
    rank = qudit_dim**num_qudits

    # Generate random density matrix
    random_rho = qx.random_density_matrix(rank=rank, dims=dims, key=key, size=ensemble_size)

    # Compute power using our implementation
    powered_rho = random_rho**power

    # Use scipy as reference
    powered_reference = qx.DensityMatrix.from_matrix(
        jnp.asarray(fractional_matrix_power(np.array(random_rho.matrix), power)), dims
    )

    # Check they match
    assert jnp.allclose(powered_rho.matrix, powered_reference.matrix, atol=1e-6)

    # Check Hermiticity is preserved
    assert jnp.allclose(powered_rho.matrix, jnp.conj(jnp.swapaxes(powered_rho.matrix, -2, -1)), atol=1e-6)

    # Check positive semidefiniteness (eigenvalues should be non-negative)
    eigvals = jnp.linalg.eigvalsh(powered_rho.matrix)
    assert jnp.all(eigvals >= -1e-6), f"Negative eigenvalues found: min={jnp.min(eigvals)}"


@pytest.mark.parametrize("theta", [0, jnp.pi / 8, jnp.pi / 4, jnp.pi / 2])
def test_cis(theta):
    """Test that cis function correctly computes matrix exponentials."""
    X = jnp.array([[0, 1], [1, 0]], dtype=complex)
    Y = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
    XX = jnp.kron(X, X)
    YY = jnp.kron(Y, Y)

    # 1Q operator
    expected_value = expm(1j * theta * X)
    computed_value = qx.cis((theta * qx.gates.X)).matrix
    assert jnp.allclose(computed_value, expected_value, atol=1e-6)
    computed_value = qx.exp((1j * theta * qx.gates.X)).matrix
    assert jnp.allclose(computed_value, expected_value, atol=1e-6)

    # 2Q operator
    expected_value_XX = expm(1j * theta * XX)
    computed_value_XX = qx.cis((theta * (qx.gates.X | qx.gates.X))).matrix
    assert jnp.allclose(computed_value_XX, expected_value_XX, atol=1e-6)
    computed_value_XX = qx.exp(1j * (theta * (qx.gates.X | qx.gates.X))).matrix
    assert jnp.allclose(computed_value_XX, expected_value_XX, atol=1e-6)

    # Ensemble of 2Q operators
    # Make a 2D array of thetas
    thetas = jnp.linspace(0, jnp.pi / 2, 12).reshape((3, 4))
    expected_values = jnp.asarray([expm(1j * t * (XX + YY)) for t in thetas.flatten()]).reshape((3, 4, 4, 4))
    computed_values = qx.cis((thetas * ((qx.gates.X | qx.gates.X) + (qx.gates.Y | qx.gates.Y)))).matrix
    assert jnp.allclose(computed_values, expected_values, atol=1e-6)
