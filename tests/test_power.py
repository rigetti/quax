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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm, fractional_matrix_power

import quax as qx


# Entirely random CPTP maps are often not infinitely divisible,
# making fractional powers invalid. Instead, we generate
# random CPTP maps via Lindbladian exponentiation, which guarantees infinite divisibility.
def _random_lindbladian(
    dims: tuple[int, ...],
    n_jumps: int = 2,
    h_scale: float = 1.0,
    gamma_scale: float = 1.0,
    size: tuple[int, ...] = (),
):
    """Generate a random CPTP superoperator via Lindbladian exponentiation (native quax)."""
    rng = np.random.default_rng(seed=42)
    d = reduce(mul, dims)

    if size == ():
        num_superops = 1
    else:
        num_superops = reduce(mul, size)

    chois = []
    for _ in range(num_superops):
        # Random Hamiltonian (Hermitian)
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        H_mat = h_scale * (A + A.conj().T) / 2
        H_obs = qx.Observable.from_matrix(jnp.asarray(H_mat, dtype=complex), (dims, dims))

        # Random jump operators with rates absorbed
        L_list = []
        for _ in range(n_jumps):
            B = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            L_mat = gamma_scale * B / np.sqrt(d)
            L_list.append(L_mat)

        L_stack = jnp.asarray(np.stack(L_list), dtype=complex)  # (n_jumps, d, d)
        jump_ops = qx.Operator.from_matrix(L_stack, (dims, dims))

        gen = qx.Lindbladian(hamiltonian=H_obs, jump_operators=jump_ops)
        channel = qx.evolve(gen, 0.5)
        chois.append(qx.superop_to_choi(channel).matrix)

    d2 = d * d
    data = jnp.stack(chois).reshape(size + (d2, d2))
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
    assert jnp.allclose(qx.superop_to_choi(powered_superop).matrix, powered_reference_choi.matrix, atol=1e-6)

    # Test Choi power
    powered_choi = qx.power_choi(random_choi, power)
    assert jnp.allclose(powered_choi.matrix, powered_reference_choi.matrix, atol=1e-6)

    # Test Kraus power — only for integer powers, since fractional powers of CPTP channels are
    # generally not CPTP; choi_to_kraus clamps negative Choi eigenvalues to 0, causing large
    # errors for non-CPTP channels. Integer powers compose CPTP maps, so they stay CPTP.
    if isinstance(power, int):
        powered_kraus = qx.power_kraus(random_kraus, power)
        assert jnp.allclose(qx.kraus_to_choi(powered_kraus).matrix, powered_reference_choi.matrix, atol=1e-5)

    # Test Pauli-Liouville power
    powered_pauli_liouville = qx.power_pauli_liouville(random_pauli_liouville, power)
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
    computed_value = qx.cis(theta * qx.gates.X).matrix
    assert jnp.allclose(computed_value, expected_value, atol=1e-6)
    computed_value = qx.exp(1j * theta * qx.gates.X).matrix
    assert jnp.allclose(computed_value, expected_value, atol=1e-6)

    # 2Q operator
    expected_value_XX = expm(1j * theta * XX)
    computed_value_XX = qx.cis(theta * (qx.gates.X | qx.gates.X)).matrix
    assert jnp.allclose(computed_value_XX, expected_value_XX, atol=1e-6)
    computed_value_XX = qx.exp(1j * (theta * (qx.gates.X | qx.gates.X))).matrix
    assert jnp.allclose(computed_value_XX, expected_value_XX, atol=1e-6)

    # Ensemble of 2Q operators
    # Make a 2D array of thetas
    thetas = jnp.linspace(0, jnp.pi / 2, 12).reshape((3, 4))
    expected_values = jnp.asarray([expm(1j * t * (XX + YY)) for t in thetas.flatten()]).reshape((3, 4, 4, 4))
    computed_values = qx.cis(thetas * ((qx.gates.X | qx.gates.X) + (qx.gates.Y | qx.gates.Y))).matrix
    assert jnp.allclose(computed_values, expected_values, atol=1e-6)


# ---------------------------------------------------------------------------
# Non-integer superoperator powers via the Lindbladian generator
# ---------------------------------------------------------------------------


def _divisible_superop(seed=0):
    """An infinitely divisible channel: exp of a Lindbladian generator."""
    gen = qx.lindbladians.amplitude_damping(0.3) + qx.lindbladians.dephasing(0.2)
    return qx.evolve(gen, 1.0)


def test_superop_fractional_power_is_cptp_and_composes():
    """S**0.5 of a divisible channel is CPTP and squares back to S."""
    S = _divisible_superop()
    root = S**0.5
    assert isinstance(root, qx.SuperOp)
    assert qx.is_cptp(root)
    assert jnp.allclose((root @ root).matrix, S.matrix, atol=1e-6)


# Every common channel we offer, as an infinitely divisible generator. Rates/times are chosen so
# each channel is well-conditioned over unit time (moderate, non-degenerate decay).
_CHANNEL_GENERATORS = [
    ("depolarizing", lambda: qx.lindbladians.depolarizing(0.1)),
    ("amplitude_damping", lambda: qx.lindbladians.amplitude_damping(0.3)),
    ("dephasing", lambda: qx.lindbladians.dephasing(0.2)),
    ("bit_flip", lambda: qx.lindbladians.bit_flip(0.15)),
    ("phase_flip", lambda: qx.lindbladians.phase_flip(0.15)),
    ("leakage", lambda: qx.lindbladians.leakage(0.1)),
    ("seepage", lambda: qx.lindbladians.seepage(0.1)),
    ("thermal_relaxation", lambda: qx.lindbladians.thermal_relaxation(2.0, 3.0, p1=0.1)),
    ("amplitude_damping+dephasing", lambda: qx.lindbladians.amplitude_damping(0.3) + qx.lindbladians.dephasing(0.2)),
]


@pytest.mark.parametrize("gen_fn", [g for _, g in _CHANNEL_GENERATORS], ids=[n for n, _ in _CHANNEL_GENERATORS])
def test_superop_fractional_power_matches_evolve(gen_fn):
    """S**s equals evolving the underlying generator for a fraction of the time, for every channel."""
    gen = gen_fn()
    S = qx.evolve(gen, 1.0)
    root = S**0.5
    # The fractional power of an infinitely divisible channel stays CPTP and matches half-time evolution.
    assert qx.is_cptp(root)
    assert jnp.allclose(root.matrix, qx.evolve(gen, 0.5).matrix, atol=1e-6)


def test_superop_integer_power_is_exact_composition():
    """Integer powers are exact repeated composition (not routed through the generator)."""
    S = _divisible_superop()
    assert jnp.allclose((S**2).matrix, (S @ S).matrix, atol=1e-9)
    assert jnp.allclose((S**3).matrix, (S @ S @ S).matrix, atol=1e-9)


def test_choi_fractional_power_cptp():
    """Choi channels inherit the Lindbladian-based fractional power."""
    choi = qx.to_choi(_divisible_superop())
    root = choi**0.5
    assert qx.is_cptp(root)
    assert jnp.allclose(qx.to_superop(root @ root).matrix, qx.to_superop(choi).matrix, atol=1e-6)
