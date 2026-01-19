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
from scipy.linalg import fractional_matrix_power

from quax import (
    Choi,
    DensityMatrix,
    SuperOp,
    Unitary,
    choi_to_kraus,
    choi_to_pauli_liouville,
    choi_to_superop,
    kraus_to_choi,
    pauli_liouville_to_choi,
    process_fidelity,
    random_density_matrix,
    random_unitary,
    superop_to_choi,
)
from quax._power import (
    density_matrix_power,
    power_choi,
    power_kraus,
    power_pauli_liouville,
    power_superop,
    power_unitary,
)


# Entirely random CPTP maps are often not infinitely divisible,
# making fractional powers invalid. Instead, we generate
# random CPTP maps via Lindbladian exponentiation, which guarantees infinite divisibility.
def _random_lindbladian(
    num_qubits: int,
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
    d = 2**num_qubits
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
    return Choi(data=data, dims=((2,) * num_qubits, (2,) * num_qubits))


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("power", [1, 2, 3, 0.5, 1.5])
@pytest.mark.parametrize("seed", [583, 3357])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (3, 4)])
def test_superoperator_powers(num_qubits, power, seed, ensemble_size):
    """Test superoperator power functions."""
    dims = ((2,) * num_qubits, (2,) * num_qubits)
    # random_choi = random_choi_BCSZ(dims=dims, rank=rank, key=key, size=ensemble_size)
    random_choi = _random_lindbladian(num_qubits, size=ensemble_size)
    random_superop = choi_to_superop(random_choi)
    random_kraus = choi_to_kraus(random_choi)
    random_pauli_liouville = choi_to_pauli_liouville(random_choi)

    # here we use scipy.linalg.fractional_matrix_power as a reference implementation
    powered_reference_superop = SuperOp(
        data=jnp.asarray(fractional_matrix_power(np.array(random_superop.data), power)), dims=dims
    )
    powered_reference_choi = superop_to_choi(powered_reference_superop)

    # Test SuperOp power
    powered_superop = power_superop(random_superop, power)
    fid = process_fidelity(powered_superop, powered_reference_superop)

    assert jnp.allclose(superop_to_choi(powered_superop).data, powered_reference_choi.data, atol=1e-6)
    assert jnp.allclose(fid, 1.0, atol=1e-6)

    # Test Choi power
    powered_choi = power_choi(random_choi, power)
    fid = process_fidelity(powered_choi, powered_reference_choi)
    assert jnp.allclose(powered_choi.data, powered_reference_choi.data, atol=1e-6)
    assert jnp.allclose(fid, 1.0, atol=1e-6)

    # Test Kraus power
    powered_kraus = power_kraus(random_kraus, power)
    fid = process_fidelity(kraus_to_choi(powered_kraus), powered_reference_choi)
    assert jnp.allclose(fid, 1.0, atol=1e-6)
    assert jnp.allclose(kraus_to_choi(powered_kraus).data, powered_reference_choi.data, atol=1e-5)

    # Test Pauli-Liouville power
    powered_pauli_liouville = power_pauli_liouville(random_pauli_liouville, power)
    fid = process_fidelity(pauli_liouville_to_choi(powered_pauli_liouville), powered_reference_choi)
    assert jnp.allclose(fid, 1.0, atol=1e-6)
    assert jnp.allclose(pauli_liouville_to_choi(powered_pauli_liouville).data, powered_reference_choi.data, atol=1e-6)


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("power", [1, 2, 3, 0.5, 1.5])
@pytest.mark.parametrize("seed", [583, 3357])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (3, 4)])
def test_power_unitarys(num_qubits, power, seed, ensemble_size):
    """Test that unitary matrices are correctly powered."""
    key = jax.random.key(seed)
    dims = ((2,) * num_qubits, (2,) * num_qubits)

    # Generate random unitary
    random_U = random_unitary(dims=dims, key=key, size=ensemble_size)

    # Compute power using our implementation
    powered_U = power_unitary(random_U, power)

    # Use scipy as reference
    powered_reference = Unitary(data=jnp.asarray(fractional_matrix_power(np.array(random_U.data), power)), dims=dims)

    # Check they match
    assert jnp.allclose(powered_U.data, powered_reference.data, atol=1e-6)

    # Check unitarity is preserved (U^† @ U = I)
    result = powered_U.h @ powered_U
    identity_shape = (2**num_qubits, 2**num_qubits)
    if ensemble_size:
        identity = jnp.eye(identity_shape[0])
        # Expand identity to match ensemble dimensions
        for _ in range(len(ensemble_size)):
            identity = identity[None, ...]
        identity = jnp.broadcast_to(identity, ensemble_size + identity_shape)
    else:
        identity = jnp.eye(identity_shape[0])

    assert jnp.allclose(result.data, identity, atol=1e-6)


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("power", [1, 2, 3, 0.5, 1.5])
@pytest.mark.parametrize("seed", [583, 3357])
@pytest.mark.parametrize("ensemble_size", [(), (3,), (3, 4)])
def test_density_matrix_powers(num_qubits, power, seed, ensemble_size):
    """Test that density matrices are correctly powered."""
    key = jax.random.key(seed)
    # For DensityMatrix, dims is a single tuple of subsystem dimensions
    dims = (2,) * num_qubits
    rank = 2**num_qubits

    # Generate random density matrix
    random_rho = random_density_matrix(rank=rank, dims=dims, key=key, size=ensemble_size)

    # Compute power using our implementation
    powered_rho = density_matrix_power(random_rho, power)

    # Use scipy as reference
    powered_reference = DensityMatrix(
        data=jnp.asarray(fractional_matrix_power(np.array(random_rho.data), power)), dims=dims
    )

    # Check they match
    assert jnp.allclose(powered_rho.data, powered_reference.data, atol=1e-6)

    # Check Hermiticity is preserved
    assert jnp.allclose(powered_rho.data, jnp.conj(jnp.swapaxes(powered_rho.data, -2, -1)), atol=1e-6)

    # Check positive semidefiniteness (eigenvalues should be non-negative)
    eigvals = jnp.linalg.eigvalsh(powered_rho.data)
    assert jnp.all(eigvals >= -1e-6), f"Negative eigenvalues found: min={jnp.min(eigvals)}"
