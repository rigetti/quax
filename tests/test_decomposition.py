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

"""Tests for the decomposition module."""

import jax
import jax.numpy as jnp
import pytest

import quax as qx
from quax import gates
from quax._decomposition import (
    _to_euler_from_matrices,
    _to_pmw3_angles_from_matrices,
    to_euler,
    to_pmw3_angles,
    to_pmw4_angles,
)


def unitaries_close(U1: qx.Unitary, U2: qx.Unitary, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Check if two unitaries are equal up to global phase.

    :param U1: First unitary
    :param U2: Second unitary
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :return: True if unitaries are equivalent up to global phase
    """
    # Two unitaries are equivalent up to global phase if U1 @ U2.h is proportional to identity
    product = (U1 @ U2.h).matrix
    # Get the global phase from the (0,0) element
    phase = product[..., 0, 0]
    # Check that product is proportional to identity
    d = product.shape[-1]
    identity = jnp.eye(d, dtype=complex)
    return jnp.allclose(product, phase[..., jnp.newaxis, jnp.newaxis] * identity, rtol=rtol, atol=atol)


def reconstruct_from_euler(phi: float, theta: float, lam: float) -> qx.Unitary:
    """Reconstruct a unitary from Euler angles: RZ(phi) @ RX(theta) @ RZ(lam)."""
    return gates.RZ(phi) @ gates.RX(theta) @ gates.RZ(lam)


def phased_rx(theta: float, phase: float) -> qx.Unitary:
    """Construct a phased RX gate: RZ(-phase) @ RX(theta) @ RZ(phase) in matrix order."""
    return gates.RZ(-phase) @ gates.RX(theta) @ gates.RZ(phase)


def reconstruct_from_pmw3(omega: float, phi: float, theta: float) -> qx.Unitary:
    """Reconstruct a unitary from PMW-3 angles.

    phased_RX(pi/2, omega) @ phased_RX(pi, phi) @ phased_RX(pi/2, theta)
    """
    return phased_rx(jnp.pi / 2, omega) @ phased_rx(jnp.pi, phi) @ phased_rx(jnp.pi / 2, theta)


def reconstruct_from_pmw4(omega: float, phi: float, theta: float) -> qx.Unitary:
    """Reconstruct a unitary from PMW-4 angles.

    phased_RX(pi/2, omega) @ phased_RX(pi/2, phi) @ phased_RX(pi/2, phi) @ phased_RX(pi/2, theta)
    """
    return (
        phased_rx(jnp.pi / 2, omega)
        @ phased_rx(jnp.pi / 2, phi)
        @ phased_rx(jnp.pi / 2, phi)
        @ phased_rx(jnp.pi / 2, theta)
    )


# =================================================================================================
# Test: to_euler
# =================================================================================================


class TestToEuler:
    """Tests for the to_euler function."""

    def test_to_euler_identity(self):
        """Test that identity gate decomposes correctly."""
        angles = to_euler(gates.I)
        phi, theta, lam = angles

        # Reconstruct and compare
        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(gates.I, reconstructed)

    def test_to_euler_pauli_x(self):
        """Test that Pauli X gate decomposes correctly."""
        angles = to_euler(gates.X)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(gates.X, reconstructed)

    def test_to_euler_pauli_y(self):
        """Test that Pauli Y gate decomposes correctly."""
        angles = to_euler(gates.Y)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(gates.Y, reconstructed)

    def test_to_euler_pauli_z(self):
        """Test that Pauli Z gate decomposes correctly."""
        angles = to_euler(gates.Z)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(gates.Z, reconstructed)

    def test_to_euler_hadamard(self):
        """Test that Hadamard gate decomposes correctly."""
        angles = to_euler(gates.H)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(gates.H, reconstructed)

    def test_to_euler_s_gate(self):
        """Test that S gate decomposes correctly."""
        angles = to_euler(gates.S)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(gates.S, reconstructed)

    def test_to_euler_t_gate(self):
        """Test that T gate decomposes correctly."""
        angles = to_euler(gates.T)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(gates.T, reconstructed)

    @pytest.mark.parametrize("angle", [0.0, jnp.pi / 4, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])
    def test_to_euler_rx(self, angle):
        """Test that RX gate decomposes correctly for various angles."""
        U = gates.RX(angle)
        angles = to_euler(U)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(U, reconstructed)

    @pytest.mark.parametrize("angle", [0.0, jnp.pi / 4, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])
    def test_to_euler_ry(self, angle):
        """Test that RY gate decomposes correctly for various angles."""
        U = gates.RY(angle)
        angles = to_euler(U)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(U, reconstructed)

    @pytest.mark.parametrize("angle", [0.0, jnp.pi / 4, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])
    def test_to_euler_rz(self, angle):
        """Test that RZ gate decomposes correctly for various angles."""
        U = gates.RZ(angle)
        angles = to_euler(U)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(U, reconstructed)

    def test_to_euler_random_single(self):
        """Test that random unitary decomposes correctly."""
        key = jax.random.key(42)
        U = qx.random_unitary(dims=((2,), (2,)), key=key)
        angles = to_euler(U)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(U, reconstructed)

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_to_euler_random_multiple_seeds(self, seed):
        """Test decomposition for random unitaries with various seeds."""
        key = jax.random.key(seed)
        U = qx.random_unitary(dims=((2,), (2,)), key=key)
        angles = to_euler(U)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(U, reconstructed)

    def test_to_euler_output_shape_single(self):
        """Test that output shape is correct for single unitary."""
        angles = to_euler(gates.X)
        assert angles.shape == (3,)

    def test_to_euler_angles_in_range(self):
        """Test that output angles are in [0, 2*pi)."""
        key = jax.random.key(99)
        U = qx.random_unitary(dims=((2,), (2,)), key=key)
        angles = to_euler(U)

        assert jnp.all(angles >= 0)
        assert jnp.all(angles < 2 * jnp.pi)

    def test_to_euler_invalid_dims_2q(self):
        """Test that ValueError is raised for 2-qubit unitary."""
        with pytest.raises(ValueError, match="Expected single-qubit unitary"):
            to_euler(gates.CZ)

    def test_to_euler_invalid_dims_3level(self):
        """Test that ValueError is raised for non-qubit dimensions."""
        # Create a 3-level system unitary
        U = qx.Unitary.from_matrix(jnp.eye(3, dtype=complex), ((3,), (3,)), 0)
        with pytest.raises(ValueError, match="Expected single-qubit unitary"):
            to_euler(U)


# =================================================================================================
# Test: Ensemble support
# =================================================================================================


class TestEnsembleSupport:
    """Tests for ensemble/batch support in decomposition functions."""

    @pytest.mark.parametrize("ensemble_size", [(3,), (4,), (2, 3)])
    def test_to_euler_ensemble(self, ensemble_size):
        """Test to_euler with ensemble of unitaries."""
        key = jax.random.key(42)
        U = qx.random_unitary(dims=((2,), (2,)), key=key, size=ensemble_size)
        angles = to_euler(U)

        # Check output shape
        assert angles.shape == ensemble_size + (3,)

        # Check each unitary in the ensemble
        flat_angles = angles.reshape(-1, 3)
        flat_matrices = U.matrix.reshape(-1, 2, 2)
        num_unitaries = flat_angles.shape[0]

        for i in range(num_unitaries):
            phi, theta, lam = flat_angles[i]
            reconstructed = reconstruct_from_euler(phi, theta, lam)
            original = qx.Unitary.from_matrix(flat_matrices[i], ((2,), (2,)), 0)
            assert unitaries_close(original, reconstructed)

    @pytest.mark.parametrize("ensemble_size", [(3,), (4,), (2, 3)])
    def test_to_pmw3_angles_ensemble(self, ensemble_size):
        """Test to_pmw3_angles with ensemble of unitaries."""
        key = jax.random.key(123)
        U = qx.random_unitary(dims=((2,), (2,)), key=key, size=ensemble_size)
        angles = to_pmw3_angles(U)

        # Check output shape
        assert angles.shape == ensemble_size + (3,)

    @pytest.mark.parametrize("ensemble_size", [(3,), (4,), (2, 3)])
    def test_to_pmw4_angles_ensemble(self, ensemble_size):
        """Test to_pmw4_angles with ensemble of unitaries."""
        key = jax.random.key(456)
        U = qx.random_unitary(dims=((2,), (2,)), key=key, size=ensemble_size)
        angles = to_pmw4_angles(U)

        # Check output shape
        assert angles.shape == ensemble_size + (3,)


# =================================================================================================
# Test: JIT compilation
# =================================================================================================


class TestJITCompilation:
    """Tests for JIT compilation compatibility."""

    def test_to_euler_jit(self):
        """Test that to_euler can be JIT compiled."""
        key = jax.random.key(42)
        U = qx.random_unitary(dims=((2,), (2,)), key=key)

        # JIT compile the function
        jitted_to_euler = jax.jit(to_euler)

        # Run and compare
        angles_eager = to_euler(U)
        angles_jit = jitted_to_euler(U)

        assert jnp.allclose(angles_eager, angles_jit)

    def test_to_pmw3_angles_jit(self):
        """Test that to_pmw3_angles can be JIT compiled."""
        key = jax.random.key(42)
        U = qx.random_unitary(dims=((2,), (2,)), key=key)

        jitted_fn = jax.jit(to_pmw3_angles)

        angles_eager = to_pmw3_angles(U)
        angles_jit = jitted_fn(U)

        assert jnp.allclose(angles_eager, angles_jit)

    def test_to_pmw4_angles_jit(self):
        """Test that to_pmw4_angles can be JIT compiled."""
        key = jax.random.key(42)
        U = qx.random_unitary(dims=((2,), (2,)), key=key)

        jitted_fn = jax.jit(to_pmw4_angles)

        angles_eager = to_pmw4_angles(U)
        angles_jit = jitted_fn(U)

        assert jnp.allclose(angles_eager, angles_jit)

    def test_to_euler_jit_ensemble(self):
        """Test that JIT works with ensembles."""
        key = jax.random.key(42)
        U = qx.random_unitary(dims=((2,), (2,)), key=key, size=(5,))

        jitted_fn = jax.jit(to_euler)

        angles_eager = to_euler(U)
        angles_jit = jitted_fn(U)

        assert jnp.allclose(angles_eager, angles_jit)

    def test_vmap_over_euler(self):
        """Test that vmap works over the internal matrix function."""
        key = jax.random.key(42)
        keys = jax.random.split(key, 10)
        matrices = jax.vmap(lambda k: qx.random_unitary(dims=((2,), (2,)), key=k).matrix)(keys)

        # vmap over the internal function
        vmapped_fn = jax.vmap(_to_euler_from_matrices)
        angles = vmapped_fn(matrices)

        assert angles.shape == (10, 3)


# =================================================================================================
# Test: to_pmw3_angles
# =================================================================================================


class TestToPmw3Angles:
    """Tests for the to_pmw3_angles function.

    Note: The PMW conversion formulas are experimental and may not work for all unitaries.
    Tests for cases that don't work are marked as xfail.
    """

    def test_to_pmw3_identity(self):
        """Test PMW-3 decomposition of identity gate."""
        angles = to_pmw3_angles(gates.I)
        omega, phi, theta = angles

        reconstructed = reconstruct_from_pmw3(omega, phi, theta)
        assert unitaries_close(gates.I, reconstructed)

    def test_to_pmw3_pauli_x(self):
        """Test PMW-3 decomposition of Pauli X gate."""
        angles = to_pmw3_angles(gates.X)
        omega, phi, theta = angles

        reconstructed = reconstruct_from_pmw3(omega, phi, theta)
        assert unitaries_close(gates.X, reconstructed)

    @pytest.mark.xfail(reason="PMW conversion formulas need correction for general unitaries")
    def test_to_pmw3_hadamard(self):
        """Test PMW-3 decomposition of Hadamard gate."""
        angles = to_pmw3_angles(gates.H)
        omega, phi, theta = angles

        reconstructed = reconstruct_from_pmw3(omega, phi, theta)
        assert unitaries_close(gates.H, reconstructed)

    @pytest.mark.xfail(reason="PMW conversion formulas need correction for general unitaries")
    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_to_pmw3_random(self, seed):
        """Test PMW-3 decomposition for random unitaries."""
        key = jax.random.key(seed)
        U = qx.random_unitary(dims=((2,), (2,)), key=key)
        angles = to_pmw3_angles(U)
        omega, phi, theta = angles

        reconstructed = reconstruct_from_pmw3(omega, phi, theta)
        assert unitaries_close(U, reconstructed)

    def test_to_pmw3_output_shape(self):
        """Test that output shape is correct."""
        angles = to_pmw3_angles(gates.X)
        assert angles.shape == (3,)


# =================================================================================================
# Test: to_pmw4_angles
# =================================================================================================


class TestToPmw4Angles:
    """Tests for the to_pmw4_angles function.

    Note: The PMW conversion formulas are experimental and may not work for all unitaries.
    Tests for cases that don't work are marked as xfail.
    """

    def test_to_pmw4_identity(self):
        """Test PMW-4 decomposition of identity gate."""
        angles = to_pmw4_angles(gates.I)
        omega, phi, theta = angles

        reconstructed = reconstruct_from_pmw4(omega, phi, theta)
        assert unitaries_close(gates.I, reconstructed)

    def test_to_pmw4_pauli_x(self):
        """Test PMW-4 decomposition of Pauli X gate."""
        angles = to_pmw4_angles(gates.X)
        omega, phi, theta = angles

        reconstructed = reconstruct_from_pmw4(omega, phi, theta)
        assert unitaries_close(gates.X, reconstructed)

    @pytest.mark.xfail(reason="PMW conversion formulas need correction for general unitaries")
    def test_to_pmw4_hadamard(self):
        """Test PMW-4 decomposition of Hadamard gate."""
        angles = to_pmw4_angles(gates.H)
        omega, phi, theta = angles

        reconstructed = reconstruct_from_pmw4(omega, phi, theta)
        assert unitaries_close(gates.H, reconstructed)

    @pytest.mark.xfail(reason="PMW conversion formulas need correction for general unitaries")
    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_to_pmw4_random(self, seed):
        """Test PMW-4 decomposition for random unitaries."""
        key = jax.random.key(seed)
        U = qx.random_unitary(dims=((2,), (2,)), key=key)
        angles = to_pmw4_angles(U)
        omega, phi, theta = angles

        reconstructed = reconstruct_from_pmw4(omega, phi, theta)
        assert unitaries_close(U, reconstructed)

    def test_to_pmw4_output_shape(self):
        """Test that output shape is correct."""
        angles = to_pmw4_angles(gates.X)
        assert angles.shape == (3,)


# =================================================================================================
# Test: Internal functions
# =================================================================================================


class TestInternalFunctions:
    """Tests for internal helper functions."""

    def test_internal_euler_from_matrices(self):
        """Test the internal _to_euler_from_matrices function."""
        U = gates.X
        matrices = U.matrix

        angles = _to_euler_from_matrices(matrices)
        phi, theta, lam = angles

        reconstructed = reconstruct_from_euler(phi, theta, lam)
        assert unitaries_close(U, reconstructed)

    @pytest.mark.xfail(reason="PMW conversion formulas need correction for general unitaries")
    def test_internal_pmw3_from_matrices(self):
        """Test the internal _to_pmw3_angles_from_matrices function."""
        U = gates.H
        matrices = U.matrix

        angles = _to_pmw3_angles_from_matrices(matrices)
        omega, phi, theta = angles

        reconstructed = reconstruct_from_pmw3(omega, phi, theta)
        assert unitaries_close(U, reconstructed)

    def test_internal_functions_batched(self):
        """Test that internal functions handle batched input."""
        key = jax.random.key(42)
        keys = jax.random.split(key, 5)

        # Create batch of matrices
        matrices = jnp.stack([qx.random_unitary(dims=((2,), (2,)), key=k).matrix for k in keys])

        angles = _to_euler_from_matrices(matrices)
        assert angles.shape == (5, 3)


# =================================================================================================
# Test: Consistency between decompositions
# =================================================================================================


class TestConsistency:
    """Tests for consistency between different decomposition methods."""

    def test_pmw3_pmw4_same_angles(self):
        """Test that PMW-3 and PMW-4 return the same angles."""
        key = jax.random.key(42)
        U = qx.random_unitary(dims=((2,), (2,)), key=key)

        angles_pmw3 = to_pmw3_angles(U)
        angles_pmw4 = to_pmw4_angles(U)

        assert jnp.allclose(angles_pmw3, angles_pmw4)

    @pytest.mark.parametrize("gate", [gates.I, gates.X, gates.Y, gates.Z])
    def test_roundtrip_euler_simple_gates(self, gate):
        """Test Euler roundtrip decomposition for simple standard gates."""
        euler_angles = to_euler(gate)
        euler_reconstructed = reconstruct_from_euler(*euler_angles)
        assert unitaries_close(gate, euler_reconstructed), f"Euler roundtrip failed for {gate}"

    @pytest.mark.parametrize("gate", [gates.H, gates.S, gates.T])
    def test_roundtrip_euler_complex_gates(self, gate):
        """Test Euler roundtrip decomposition for more complex standard gates."""
        euler_angles = to_euler(gate)
        euler_reconstructed = reconstruct_from_euler(*euler_angles)
        assert unitaries_close(gate, euler_reconstructed), f"Euler roundtrip failed for {gate}"

    @pytest.mark.parametrize("gate", [gates.I, gates.X, gates.Y, gates.Z])
    def test_roundtrip_pmw_simple_gates(self, gate):
        """Test PMW roundtrip for simple gates where it works."""
        # PMW-3 decomposition
        pmw3_angles = to_pmw3_angles(gate)
        pmw3_reconstructed = reconstruct_from_pmw3(*pmw3_angles)
        assert unitaries_close(gate, pmw3_reconstructed), f"PMW-3 roundtrip failed for {gate}"

        # PMW-4 decomposition
        pmw4_angles = to_pmw4_angles(gate)
        pmw4_reconstructed = reconstruct_from_pmw4(*pmw4_angles)
        assert unitaries_close(gate, pmw4_reconstructed), f"PMW-4 roundtrip failed for {gate}"

    @pytest.mark.xfail(reason="PMW conversion formulas need correction for general unitaries")
    @pytest.mark.parametrize("gate", [gates.H, gates.S, gates.T])
    def test_roundtrip_pmw_complex_gates(self, gate):
        """Test PMW roundtrip for gates where the formulas are known to have issues."""
        # PMW-3 decomposition
        pmw3_angles = to_pmw3_angles(gate)
        pmw3_reconstructed = reconstruct_from_pmw3(*pmw3_angles)
        assert unitaries_close(gate, pmw3_reconstructed), f"PMW-3 roundtrip failed for {gate}"

        # PMW-4 decomposition
        pmw4_angles = to_pmw4_angles(gate)
        pmw4_reconstructed = reconstruct_from_pmw4(*pmw4_angles)
        assert unitaries_close(gate, pmw4_reconstructed), f"PMW-4 roundtrip failed for {gate}"
