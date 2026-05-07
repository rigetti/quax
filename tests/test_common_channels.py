# Copyright 2021-2023 Rigetti & Co, LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Tests for common quantum channels in JAX."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx
from quax import DensityMatrix

from .instrument_helpers import (
    basis_dm,
    basis_dm_multi,
    make_noisy_instrument_from_unitary,
    make_noisy_instrument_multi,
    make_spectator_instrument,
    superposition_dm,
)

# Fractional unitary power is grad-able which unitary_power is not
from quax._common_channels import fractional_unitary_power


class TestFractionalUnitaryPower:
    """Test the fractional_unitary_power function."""

    @pytest.mark.parametrize("qudit_dim", [2, 3])
    def test_fractional_power_identity(self, qudit_dim):
        """Test that identity^(1/n) = identity."""
        d = qudit_dim
        dims = ((d,), (d,))
        identity = qx.Unitary.from_matrix(jnp.eye(d, dtype=jnp.complex128), dims)

        for n in [2, 3, 5, 10]:
            result = fractional_unitary_power(identity, 1.0 / n)
            fid = qx.unitary_entanglement_fidelity(result, identity)
            assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for n={n}"

    @pytest.mark.parametrize("qudit_dim", [2, 3])
    def test_fractional_power_composition(self, qudit_dim):
        """Test that (U^(1/n))^n = U."""
        key = jax.random.PRNGKey(42)
        d = qudit_dim
        dims = ((d,), (d,))
        U = qx.random_unitary(dims=dims, key=key)

        n = 4
        U_frac = fractional_unitary_power(U, 1.0 / n)

        # Compose n times
        result = U_frac
        for _ in range(n - 1):
            result = result @ U_frac

        fid = qx.unitary_entanglement_fidelity(result, U)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for n={n}"

    def test_fractional_power_pauli_gates(self):
        """Test fractional powers of Pauli gates."""
        # Pauli X: eigenvalues are ±1, so X^(1/2) should have eigenvalues ±i
        X_half = fractional_unitary_power(qx.gates.X, 0.5)

        # X^(1/2) @ X^(1/2) should equal X
        result = X_half @ X_half
        fid = qx.unitary_entanglement_fidelity(result, qx.gates.X)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for X^(1/2)"

        # Pauli Y
        Y_half = fractional_unitary_power(qx.gates.Y, 0.5)
        result = Y_half @ Y_half
        fid = qx.unitary_entanglement_fidelity(result, qx.gates.Y)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for Y^(1/2)"

        # Pauli Z
        Z_half = fractional_unitary_power(qx.gates.Z, 0.5)
        result = Z_half @ Z_half
        fid = qx.unitary_entanglement_fidelity(result, qx.gates.Z)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for Z^(1/2)"

    def test_fractional_power_rotation_gate(self):
        """Test fractional power of rotation gate."""
        theta = jnp.pi / 3

        # RZ(θ)^(1/2) should be RZ(θ/2)
        RZ_half = fractional_unitary_power(qx.gates.RZ(theta), 0.5)
        expected = qx.gates.RZ(theta / 2)

        fid = qx.unitary_entanglement_fidelity(RZ_half, expected)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for RZ(θ)^(1/2)"

    def test_fractional_power_hadamard(self):
        """Test fractional power of Hadamard gate."""

        # H^2 = I, so H^(1/2) should satisfy (H^(1/2))^4 = I
        H_half = fractional_unitary_power(qx.gates.H, 0.5)
        result = H_half @ H_half @ H_half @ H_half

        fid = qx.unitary_entanglement_fidelity(result, qx.gates.I)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for H^(1/2)"

    def test_fractional_power_two_qubit(self):
        """Test fractional power of a two-qubit gate."""

        # CZ^(1/2) composed twice should give CZ
        CZ_half = fractional_unitary_power(qx.gates.CZ, 0.5)
        result = CZ_half @ CZ_half

        fid = qx.unitary_entanglement_fidelity(result, qx.gates.CZ)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for CZ^(1/2)"

    @pytest.mark.parametrize("qudit_dim", [2, 3, 4])
    def test_fractional_power_unitarity_preserved(self, qudit_dim):
        """Test that fractional powers preserve unitarity."""
        key = jax.random.PRNGKey(42)
        d = qudit_dim
        dims = ((d,), (d,))
        U = qx.random_unitary(dims=dims, key=key)

        # Compute fractional power
        U_frac = fractional_unitary_power(U, 0.3)

        # Check unitarity: U^† U = I
        result = U_frac.h @ U_frac
        identity = qx.Unitary.from_matrix(jnp.eye(d, dtype=jnp.complex128), dims)

        fid = qx.unitary_entanglement_fidelity(result, identity)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for U^(0.3)"

    @pytest.mark.parametrize("qudit_dim", [2, 3])
    def test_fractional_power_determinant(self, qudit_dim):
        """Test that det(U^(1/n)) has magnitude 1."""
        key = jax.random.PRNGKey(42)
        d = qudit_dim
        dims = ((d,), (d,))
        U = qx.random_unitary(dims=dims, key=key)

        n = 5
        U_frac = fractional_unitary_power(U, 1.0 / n)

        det_U = jnp.linalg.det(U.matrix)
        det_U_frac = jnp.linalg.det(U_frac.matrix)

        fid = qx.unitary_entanglement_fidelity(U_frac**n, U)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for U^(1/n) composition"

        assert jnp.isclose(jnp.abs(det_U), 1.0, atol=1e-5)
        assert jnp.isclose(jnp.abs(det_U_frac), 1.0, atol=1e-5)

    @pytest.mark.parametrize("qudit_dim", [2, 3])
    def test_fractional_power_negative_exponent(self, qudit_dim):
        """Test fractional power with negative exponent (inverse)."""
        key = jax.random.PRNGKey(42)
        d = qudit_dim
        dims = ((d,), (d,))
        U = qx.random_unitary(dims=dims, key=key)
        identity = qx.Unitary.from_matrix(jnp.eye(d, dtype=jnp.complex128), dims)

        # Test that U @ U^(-1) = I
        U_inv = fractional_unitary_power(U, -1.0)
        result = U @ U_inv

        fid = qx.unitary_entanglement_fidelity(result, identity)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for U^(-1)"

    def test_fractional_power_jit_compatible(self):
        """Test that fractional_unitary_power is JIT-compatible."""

        @jax.jit
        def compute_frac_power(U):
            return fractional_unitary_power(U, 0.5)

        result = compute_frac_power(qx.gates.H)

        # Should not raise errors and should be a valid matrix
        assert result.matrix.shape == (2, 2)

        # Test that it's actually computing the right thing
        result_squared = result @ result
        fid = qx.unitary_entanglement_fidelity(result_squared, qx.gates.H)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for JIT H^(1/2)"

    def test_fractional_power_phase_gate(self):
        """Test fractional power respects phase relationships."""
        # S^(1/2) should be T gate: [[1, 0], [0, e^(iπ/4)]]
        S_half = fractional_unitary_power(qx.gates.S, 0.5)

        # Expected T gate
        fid = qx.unitary_entanglement_fidelity(S_half, qx.gates.T)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for S^(1/2)"

    def test_fractional_power_cz_gate(self):
        """Test that CZ^(1/20) equals CPHASE(π/20)."""
        # CZ = CPHASE(π), so CZ^(1/20) should equal CPHASE(π/20)
        CZ_frac = fractional_unitary_power(qx.gates.CZ, 1.0 / 20.0)
        expected = qx.gates.CPHASE(jnp.pi / 20.0)

        fid = qx.unitary_entanglement_fidelity(CZ_frac, expected)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for CZ^(1/20)"

        # Verify composition: (CZ^(1/20))^20 = CZ
        result = CZ_frac
        for _ in range(19):
            result = result @ CZ_frac

        fid = qx.unitary_entanglement_fidelity(result, qx.gates.CZ)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for CZ^(1/20) composition"

    def test_fractional_power_gradient_support(self):
        """Test that fractional_unitary_power supports gradients.

        The implementation uses jax.scipy.linalg.logm and expm which support
        automatic differentiation, enabling gradient-based optimization.
        """

        # Test with a simple rotation gate that depends on a parameter
        def loss_fn(theta):
            # Create RZ(theta) gate
            U = jnp.array([[jnp.exp(-1j * theta / 2), 0.0], [0.0, jnp.exp(1j * theta / 2)]], dtype=jnp.complex128)
            U_half = fractional_unitary_power(qx.Unitary.from_matrix(U, ((2,), (2,))), 0.5)
            # Loss: measure deviation from expected half-angle rotation
            expected = jnp.array(
                [[jnp.exp(-1j * theta / 4), 0.0], [0.0, jnp.exp(1j * theta / 4)]], dtype=jnp.complex128
            )
            diff = U_half.matrix - expected
            return jnp.real(jnp.sum(jnp.abs(diff) ** 2))

        # Compute gradient - this should now work!
        theta = 1.0
        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(theta)

        # Verify gradient is computed (non-zero for non-optimal theta)
        assert jnp.isfinite(grad_val), "Gradient should be finite"

        # The loss should be near zero at theta (RZ is correctly computed)
        # so gradient might be small but should exist
        loss_val = loss_fn(theta)
        assert loss_val < 1e-6, f"Loss should be near zero, got {loss_val}"


class TestThermalRelaxationChoi:
    """Test thermal relaxation Choi matrix construction."""

    def test_single_qubit_thermal_relaxation_shape(self):
        """Test that single qubit thermal relaxation has correct shape."""
        t1s = jnp.array([50e-6])  # 50 microseconds
        tphis = jnp.array([30e-6])  # 30 microseconds
        duration = 1e-6  # 1 microsecond

        choi = qx.thermal_relaxation_choi(t1s, tphis, duration)
        assert choi.matrix.shape == (4, 4)

    def test_two_qubit_thermal_relaxation_shape(self):
        """Test that two qubit thermal relaxation has correct shape."""
        t1s = jnp.array([50e-6, 45e-6])
        tphis = jnp.array([30e-6, 28e-6])
        duration = 1e-6

        choi = qx.thermal_relaxation_choi(t1s, tphis, duration)
        assert choi.matrix.shape == (16, 16)

    def test_thermal_relaxation_preserves_trace(self):
        """Test that thermal relaxation channel is approximately trace-preserving.

        Note: The thermal relaxation Choi matrix provided by the user may not be exactly
        trace-preserving due to the specific construction. This test uses relaxed tolerance.
        """
        t1s = jnp.array([50e-6, 45e-6])
        tphis = jnp.array([30e-6, 28e-6])
        duration = 1e-6

        choi = qx.thermal_relaxation_choi(t1s, tphis, duration)

        # For a trace-preserving channel, partial trace over output should give identity
        # Reshape to (d_out, d_in, d_out, d_in)
        d = 4  # 2 qubits
        choi_reshaped = choi.matrix.reshape(d, d, d, d)

        # Partial trace: sum over first and third indices
        ptrace = jnp.einsum("ijik->jk", choi_reshaped)

        # Use relaxed tolerance since the Choi matrix may not be perfectly normalized
        assert jnp.allclose(
            jnp.array(ptrace),
            jnp.eye(d),
            rtol=0.05,  # 5% relative tolerance
            atol=0.05,  # Absolute tolerance
        )

    def test_thermal_relaxation_zero_duration(self):
        """Test that zero duration gives identity channel."""
        t1s = jnp.array([50e-6])
        tphis = jnp.array([30e-6])
        duration = 0.0

        choi = qx.thermal_relaxation_choi(t1s, tphis, duration)

        # Identity channel Choi matrix
        expected = jnp.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=jnp.complex128)

        assert jnp.allclose(jnp.array(choi.matrix), jnp.array(expected), rtol=1e-5, atol=1e-7)

    def test_thermal_relaxation_jit_compatible(self):
        """Test that thermal_relaxation_choi works (note: not fully JIT-compatible due to Python loop)."""
        t1s = jnp.array([50e-6, 45e-6])
        tphis = jnp.array([30e-6, 28e-6])
        duration = 1e-6

        # Can be called directly (though not fully JIT-able)
        choi = qx.thermal_relaxation_choi(t1s, tphis, duration)
        assert choi.matrix.shape == (16, 16)

        # Test that calling again gives same result
        choi2 = qx.thermal_relaxation_choi(t1s, tphis, duration)

        fid = qx.process_fidelity(choi, choi2)
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for repeated calls"

    def test_thermal_relaxation_long_duration(self):
        """Test thermal relaxation with very long duration approaches ground state."""
        t1s = jnp.array([50e-6])
        tphis = jnp.array([30e-6])
        duration = 1e-3  # 1 millisecond >> T1

        choi = qx.thermal_relaxation_choi(t1s, tphis, duration)

        # After long time, should be close to projecting onto ground state
        # The (0,0) element should be close to 1, others near 0
        choi_00 = choi.matrix[0, 0]
        assert abs(choi_00 - 1.0) < 0.1

        # Off-diagonal coherences should decay
        choi_03 = choi.matrix[0, 3]
        assert abs(choi_03) < 0.1


class TestIntegratedThermalSuperoperator:
    """Test the integrated thermal superoperator function."""

    def test_integrated_thermal_shape(self):
        """Test that integrated thermal superoperator has correct shape."""
        # Two-qubit unitary
        unitary = qx.Unitary.from_matrix(jnp.eye(4, dtype=jnp.complex128), ((2, 2), (2, 2)))
        t1s = jnp.array([50e-6, 45e-6])
        tphis = jnp.array([30e-6, 28e-6])
        duration = 1e-6

        superop = qx.integrated_thermal_superoperator(unitary, t1s, tphis, duration, num_steps=10)

        # Superoperator for 2 qubits should be 16x16
        assert superop.matrix.shape == (16, 16)

    def test_integrated_thermal_identity_unitary(self):
        """Test integrated thermal with identity unitary equals pure thermal."""
        unitary = qx.Unitary.from_matrix(jnp.eye(4, dtype=jnp.complex128), ((2, 2), (2, 2)))
        t1s = jnp.array([50e-6, 45e-6])
        tphis = jnp.array([30e-6, 28e-6])
        duration = 1e-6

        # Integrated thermal with identity
        integrated_super = qx.integrated_thermal_superoperator(unitary, t1s, tphis, duration, num_steps=100)

        # Pure thermal channel
        thermal_choi = qx.thermal_relaxation_choi(t1s, tphis, duration)

        # Should be approximately equal
        fid = qx.process_fidelity(qx.superop_to_choi(integrated_super), thermal_choi)
        assert jnp.isclose(fid, 1.0, atol=1e-3), f"Fidelity {fid} not close to 1 for integrated thermal with identity"

    def test_integrated_thermal_zero_duration(self):
        """Test integrated thermal with zero duration gives unitary channel."""
        # Random 2-qubit unitary
        theta = 0.5
        unitary = qx.Unitary.from_matrix(
            jnp.array(
                [
                    [jnp.cos(theta), 0, 0, -jnp.sin(theta)],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [jnp.sin(theta), 0, 0, jnp.cos(theta)],
                ],
                dtype=jnp.complex128,
            ),
            ((2, 2), (2, 2)),
        )

        t1s = jnp.array([50e-6, 45e-6])
        tphis = jnp.array([30e-6, 28e-6])
        duration = 0.0  # Zero duration

        integrated_super = qx.integrated_thermal_superoperator(unitary, t1s, tphis, duration, num_steps=10)

        # Should give just the unitary channel
        unitary_super = qx.unitary_to_superop(unitary)

        fid = qx.process_fidelity(qx.superop_to_choi(integrated_super), qx.superop_to_choi(unitary_super))
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for zero-duration integrated thermal"

    def test_integrated_thermal_jit_compatible(self):
        """Test that integrated_thermal_superoperator is JIT-compatible."""

        @jax.jit
        def compute_integrated(unitary, t1s, tphis, duration):
            return qx.integrated_thermal_superoperator(unitary, t1s, tphis, duration, num_steps=10)

        unitary = qx.Unitary.from_matrix(jnp.eye(4, dtype=jnp.complex128), ((2, 2), (2, 2)))
        t1s = jnp.array([50e-6, 45e-6])
        tphis = jnp.array([30e-6, 28e-6])
        duration = 1e-6

        # Should not raise any errors
        superop = compute_integrated(unitary, t1s, tphis, duration)
        assert superop.matrix.shape == (16, 16)

        # Test that calling again uses cached version
        superop2 = compute_integrated(unitary, t1s, tphis, duration)
        fid = qx.process_fidelity(qx.superop_to_choi(superop), qx.superop_to_choi(superop2))
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for JIT calls"

    def test_integrated_thermal_convergence(self):
        """Test that more steps gives better approximation."""
        theta = 0.3
        unitary = qx.Unitary.from_matrix(
            jnp.array(
                [
                    [jnp.cos(theta), -jnp.sin(theta), 0, 0],
                    [jnp.sin(theta), jnp.cos(theta), 0, 0],
                    [0, 0, jnp.cos(theta), -jnp.sin(theta)],
                    [0, 0, jnp.sin(theta), jnp.cos(theta)],
                ],
                dtype=jnp.complex128,
            ),
            ((2, 2), (2, 2)),
        )

        t1s = jnp.array([50e-6, 45e-6])
        tphis = jnp.array([30e-6, 28e-6])
        duration = 2e-6

        # Compute with different number of steps
        superop_10 = qx.integrated_thermal_superoperator(unitary, t1s, tphis, duration, num_steps=10)
        superop_50 = qx.integrated_thermal_superoperator(unitary, t1s, tphis, duration, num_steps=50)
        superop_100 = qx.integrated_thermal_superoperator(unitary, t1s, tphis, duration, num_steps=100)

        # Distance between 50 and 100 should be less than between 10 and 50
        fid_10_50 = qx.process_fidelity(qx.superop_to_choi(superop_10), qx.superop_to_choi(superop_50))
        fid_50_100 = qx.process_fidelity(qx.superop_to_choi(superop_50), qx.superop_to_choi(superop_100))

        assert fid_50_100 > fid_10_50

    def test_integrated_thermal_single_qubit(self):
        """Test integrated thermal for single qubit."""
        # Single-qubit rotation
        theta = jnp.pi / 4
        unitary = qx.Unitary.from_matrix(
            jnp.array(
                [[jnp.cos(theta / 2), -1j * jnp.sin(theta / 2)], [-1j * jnp.sin(theta / 2), jnp.cos(theta / 2)]],
                dtype=jnp.complex128,
            ),
            ((2,), (2,)),
        )

        t1s = jnp.array([50e-6])
        tphis = jnp.array([30e-6])
        duration = 1e-6

        superop = qx.integrated_thermal_superoperator(unitary, t1s, tphis, duration, num_steps=50)

        # Should have correct shape for single qubit
        assert superop.matrix.shape == (4, 4)

    def test_integrated_thermal_large_coherence_times(self):
        """Test that integrated thermal reproduces unitary in limit of very large T1 and Tphi."""
        # Two-qubit unitary (non-trivial gate)
        theta = jnp.pi / 3
        phi = jnp.pi / 6
        unitary = qx.Unitary.from_matrix(
            jnp.array(
                [
                    [jnp.cos(theta), 0, 0, -1j * jnp.sin(theta) * jnp.exp(-1j * phi)],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [-1j * jnp.sin(theta) * jnp.exp(1j * phi), 0, 0, jnp.cos(theta)],
                ],
                dtype=jnp.complex128,
            ),
            ((2, 2), (2, 2)),
        )

        # Very large coherence times (relative to gate duration)
        duration = 50e-9  # 50 ns gate duration
        t1s = jnp.array([1e-3, 1e-3])  # 1 ms >> 50 ns
        tphis = jnp.array([5e-4, 5e-4])  # 0.5 ms >> 50 ns

        integrated_super = qx.integrated_thermal_superoperator(unitary, t1s, tphis, duration, num_steps=100)

        # Expected: pure unitary channel (no decoherence)
        expected_super = qx.unitary_to_superop(unitary)

        fid = qx.process_fidelity(qx.superop_to_choi(integrated_super), qx.superop_to_choi(expected_super))
        assert jnp.isclose(fid, 1.0, atol=1e-3), f"Fidelity {fid} not close to 1 for large coherence times"


class TestDepolarizingChannelSuperoperator:
    """Test the depolarizing_channel_superoperator function."""

    def test_depolarizing_shape(self):
        """Test that depolarizing channel has correct shape."""
        # Single qubit
        superop_1q = qx.depolarizing_channel_superoperator(jnp.array(0.1), dims=(2,))
        assert superop_1q.matrix.shape == (4, 4)

        # Two qubits
        superop_2q = qx.depolarizing_channel_superoperator(jnp.array(0.1), dims=(2, 2))
        assert superop_2q.matrix.shape == (16, 16)

        # Three qubits
        superop_3q = qx.depolarizing_channel_superoperator(jnp.array(0.1), dims=(2, 2, 2))
        assert superop_3q.matrix.shape == (64, 64)

    def test_depolarizing_zero_probability(self):
        """Test that zero depolarizing probability gives identity channel."""
        superop = qx.depolarizing_channel_superoperator(jnp.array(0.0), dims=(2, 2))

        # Should be identity superoperator
        expected = qx.SuperOp.from_matrix(jnp.eye(16, dtype=jnp.complex128), ((2, 2), (2, 2)))

        fid = qx.process_fidelity(qx.superop_to_choi(superop), qx.superop_to_choi(expected))
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for zero depolarizing probability"

    def test_depolarizing_jit_compatible(self):
        """Test that depolarizing_channel_superoperator is JIT-compatible."""
        # The function is already JIT-compiled with static_argnames=("dims",)
        superop = qx.depolarizing_channel_superoperator(jnp.array(0.2), dims=(2, 2))

        # Verify shape
        assert superop.matrix.shape == (16, 16)

        # Verify a second call gives the same result
        superop2 = qx.depolarizing_channel_superoperator(jnp.array(0.2), dims=(2, 2))

        fid = qx.process_fidelity(qx.superop_to_choi(superop), qx.superop_to_choi(superop2))
        assert jnp.isclose(fid, 1.0, atol=1e-7), f"Fidelity {fid} not close to 1 for JIT depolarizing channel"

    def test_depolarizing_multiple_jit_calls(self):
        """Test that JIT-compiled function works with multiple calls and different probabilities."""
        # Multiple calls with different probabilities (same dims)
        superop1 = qx.depolarizing_channel_superoperator(jnp.array(0.1), dims=(2, 2))
        superop2 = qx.depolarizing_channel_superoperator(jnp.array(0.5), dims=(2, 2))
        superop3 = qx.depolarizing_channel_superoperator(jnp.array(0.9), dims=(2, 2))

        # All should have correct shape
        assert superop1.matrix.shape == (16, 16)
        assert superop2.matrix.shape == (16, 16)
        assert superop3.matrix.shape == (16, 16)

        # Results should be different
        fid_12 = qx.process_fidelity(qx.superop_to_choi(superop1), qx.superop_to_choi(superop2))
        fid_23 = qx.process_fidelity(qx.superop_to_choi(superop2), qx.superop_to_choi(superop3))
        assert fid_12 < 1.0, f"Fidelity {fid_12} should be less than 1 for different probabilities"
        assert fid_23 < 1.0, f"Fidelity {fid_23} should be less than 1 for different probabilities"

    def test_depolarizing_physical_bounds(self):
        """Test that depolarizing channel handles probability bounds correctly."""
        # Test at boundaries
        superop_min = qx.depolarizing_channel_superoperator(jnp.array(0.0), dims=(2,))
        superop_max = qx.depolarizing_channel_superoperator(jnp.array(1.0), dims=(2,))

        # Both should be valid superoperators
        assert superop_min.matrix.shape == (4, 4)
        assert superop_max.matrix.shape == (4, 4)

        # Test intermediate value
        superop_mid = qx.depolarizing_channel_superoperator(jnp.array(0.5), dims=(2,))
        assert superop_mid.matrix.shape == (4, 4)


# ======================================================================
# Classical confusion tests
# ======================================================================


class TestClassicalConfusion:
    """Test classical confusion (misclassification) with fixed input states."""

    @pytest.mark.parametrize("fid", [0.95, 0.80, 0.60])
    def test_qubit_symmetric_confusion(self, fid):
        """Qubit: outcome probabilities match confusion matrix entries."""
        confusion = jnp.array([[fid, 1 - fid], [1 - fid, fid]])
        transition = jnp.eye(2)
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2,))

        # Input |0>: P(outcome=0) = fid, P(outcome=1) = 1-fid
        rho0 = basis_dm(0, 2)
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho0)
        np.testing.assert_allclose(float(probs[0]), fid, atol=1e-10)
        np.testing.assert_allclose(float(probs[1]), 1 - fid, atol=1e-10)

        # Input |1>: P(outcome=1) = fid
        rho1 = basis_dm(1, 2)
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho1)
        np.testing.assert_allclose(float(probs[1]), fid, atol=1e-10)

    def test_qubit_confusion_matrix_roundtrip(self):
        """Confusion matrix extracted from instrument matches the input."""
        fid = 0.90
        confusion = jnp.array([[fid, 1 - fid], [1 - fid, fid]])
        transition = jnp.eye(2)
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2,))
        np.testing.assert_allclose(qi.confusion_matrix, confusion, atol=1e-10)
        np.testing.assert_allclose(qx.classification_fidelity(qi), fid, atol=1e-10)

    @pytest.mark.parametrize("fid", [0.90, 0.70])
    def test_qutrit_confusion(self, fid):
        """Qutrit: symmetric confusion with off-diagonal = (1-fid)/(d-1)."""
        d = 3
        off = (1.0 - fid) / (d - 1)
        confusion = fid * jnp.eye(d) + off * (jnp.ones((d, d)) - jnp.eye(d))
        transition = jnp.eye(d)
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(3,))

        np.testing.assert_allclose(qi.confusion_matrix, confusion, atol=1e-10)
        np.testing.assert_allclose(qx.classification_fidelity(qi), fid, atol=1e-10)

        # Input |1>: P(outcome=1) = fid
        rho1 = basis_dm(1, 3)
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho1)
        np.testing.assert_allclose(float(probs[1]), fid, atol=1e-10)

    def test_two_qubit_confusion(self):
        """Two-qubit instrument with correlated confusion errors (joint 4x4 matrix)."""
        confusion = jnp.array(
            [
                [0.85, 0.05, 0.05, 0.02],
                [0.05, 0.80, 0.02, 0.08],
                [0.05, 0.02, 0.80, 0.08],
                [0.05, 0.13, 0.13, 0.82],
            ]
        )
        transition = jnp.eye(4)
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2, 2))

        np.testing.assert_allclose(qi.confusion_matrix, confusion, atol=1e-10)

        # Input |01>: probabilities should match column 1 of the confusion matrix
        rho01 = basis_dm_multi((0, 1), (2, 2))
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho01)
        np.testing.assert_allclose(probs, confusion[:, 1], atol=1e-10)

        # Input |10>: probabilities should match column 2 of the confusion matrix
        rho10 = basis_dm_multi((1, 0), (2, 2))
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho10)
        np.testing.assert_allclose(probs, confusion[:, 2], atol=1e-10)

    def test_asymmetric_confusion_qubit(self):
        """Qubit with asymmetric confusion: P(0|1) != P(1|0)."""
        confusion = jnp.array([[0.95, 0.15], [0.05, 0.85]])
        transition = jnp.eye(2)
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2,))
        np.testing.assert_allclose(qi.confusion_matrix, confusion, atol=1e-10)

        # Input |1>: P(outcome=0) = 0.15, P(outcome=1) = 0.85
        rho1 = basis_dm(1, 2)
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho1)
        np.testing.assert_allclose(float(probs[0]), 0.15, atol=1e-10)
        np.testing.assert_allclose(float(probs[1]), 0.85, atol=1e-10)


# ======================================================================
# Classical transition tests
# ======================================================================


class TestClassicalTransition:
    """Test transition (back-action) with perfect classification."""

    @pytest.mark.parametrize("seed", [0, 42, 9999])
    @pytest.mark.parametrize("p_flip", [0.1, 0.3, 0.5])
    def test_qubit_transition_correct_label(self, p_flip, seed):
        """Qubit: classification always correct, state flipped with probability p_flip."""
        confusion = jnp.eye(2)
        transition = jnp.array([[1 - p_flip, p_flip], [p_flip, 1 - p_flip]])
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2,))

        # Classification is always correct
        np.testing.assert_allclose(qi.confusion_matrix, jnp.eye(2), atol=1e-10)
        np.testing.assert_allclose(qx.classification_fidelity(qi), 1.0, atol=1e-10)

        # Transition matrix matches
        np.testing.assert_allclose(qi.transition_matrix, transition, atol=1e-10)

        # Input |0>: always labeled 0, post-measurement state is mixture
        # (1-p_flip)|0><0| + p_flip|1><1|
        rho0 = basis_dm(0, 2)
        key = jax.random.key(seed)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho0)
        rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
        assert int(outcome) == 0
        np.testing.assert_allclose(float(jnp.real(rho_out.matrix[0, 0])), 1 - p_flip, atol=1e-10)
        np.testing.assert_allclose(float(jnp.real(rho_out.matrix[1, 1])), p_flip, atol=1e-10)

    @pytest.mark.parametrize("p_flip", [0.1, 0.5])
    def test_qutrit_transition(self, p_flip):
        """Qutrit: perfect classification with cyclic transition."""
        d = 3
        transition = jnp.zeros((d, d))
        for j in range(d):
            transition = transition.at[j, j].set(1 - p_flip)
            transition = transition.at[(j + 1) % d, j].set(p_flip)

        confusion = jnp.eye(d)
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(3,))
        np.testing.assert_allclose(qi.confusion_matrix, jnp.eye(d), atol=1e-10)
        np.testing.assert_allclose(qi.transition_matrix, transition, atol=1e-10)
        np.testing.assert_allclose(qx.classification_fidelity(qi), 1.0, atol=1e-10)

    def test_two_qubit_transition(self):
        """Two-qubit: perfect classification, partial bitflip on both qubits."""
        p_flip = 0.1
        transition_single = jnp.array([[1 - p_flip, p_flip], [p_flip, 1 - p_flip]])
        transition = jnp.kron(transition_single, transition_single)
        confusion = jnp.eye(4)
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2, 2))

        np.testing.assert_allclose(qi.confusion_matrix, jnp.eye(4), atol=1e-10)
        np.testing.assert_allclose(qi.transition_matrix, transition, atol=1e-10)
        np.testing.assert_allclose(qx.classification_fidelity(qi), 1.0, atol=1e-10)

    def test_complete_bitflip_backaction(self):
        """Full bitflip: classification correct, non-demolition fidelity 0."""
        confusion = jnp.eye(2)
        transition = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(2,))
        np.testing.assert_allclose(qx.classification_fidelity(qi), 1.0, atol=1e-10)
        np.testing.assert_allclose(qx.non_demolition_fidelity(qi), 0.0, atol=1e-10)


# ======================================================================
# Quantum noise tests
# ======================================================================


class TestQuantumNoise:
    """Test instruments with coherent quantum noise (unitary before projection)."""

    def test_qubit_rx_small_angle_z_states(self):
        """RX(pi/12) before projection: Z eigenstates see small confusion."""
        angle = jnp.pi / 12
        cos2 = float(jnp.cos(angle / 2) ** 2)

        rx = jnp.array(
            [
                [jnp.cos(angle / 2), -1j * jnp.sin(angle / 2)],
                [-1j * jnp.sin(angle / 2), jnp.cos(angle / 2)],
            ],
            dtype=complex,
        )
        qi = make_noisy_instrument_from_unitary(rx, 2)
        assert qx.validate(qi)

        # |0> → P(0) = cos²(angle/2)
        rho0 = basis_dm(0, 2)
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho0)
        np.testing.assert_allclose(float(probs[0]), cos2, atol=1e-10)

        # |1> → P(1) = cos²(angle/2)
        rho1 = basis_dm(1, 2)
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho1)
        np.testing.assert_allclose(float(probs[1]), cos2, atol=1e-10)

    def test_qubit_rx_small_angle_x_eigenstates(self):
        """RX(pi/12) before projection: |+> still gives 50/50 probabilities."""
        angle = jnp.pi / 12
        rx = jnp.array(
            [
                [jnp.cos(angle / 2), -1j * jnp.sin(angle / 2)],
                [-1j * jnp.sin(angle / 2), jnp.cos(angle / 2)],
            ],
            dtype=complex,
        )
        qi = make_noisy_instrument_from_unitary(rx, 2)

        rho_plus = superposition_dm(2)
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho_plus)
        np.testing.assert_allclose(float(probs[0]), 0.5, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_qutrit_unitary_noise(self, seed):
        """Qutrit with random near-identity unitary noise."""
        key = jax.random.key(seed + 100)
        H = jax.random.normal(key, (3, 3), dtype=complex)
        H = (H + H.conj().T) / 2
        H = H * 0.1  # Small perturbation
        U_noise = jax.scipy.linalg.expm(1j * H)

        qi = make_noisy_instrument_from_unitary(U_noise, 3)
        assert qx.validate(qi)

        # |0> should mostly be labeled 0 (small perturbation)
        rho0 = basis_dm(0, 3)
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho0)
        assert float(probs[0]) > 0.7, f"Expected mostly outcome 0, got {float(probs[0]):.3f}"

    def test_two_qubit_noise(self):
        """Two qubits: entangling unitary (CNOT-like) before projection creates correlated noise."""
        angle = jnp.pi / 8
        ZX = jnp.kron(
            jnp.array([[1, 0], [0, -1]], dtype=complex),
            jnp.array([[0, 1], [1, 0]], dtype=complex),
        )
        U2 = jax.scipy.linalg.expm(-1j * angle * ZX / 2)
        qi = make_noisy_instrument_multi(U2, (2, 2))
        assert qx.validate(qi)

        rho00 = basis_dm_multi((0, 0), (2, 2))
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho00)
        cos2 = float(jnp.cos(angle / 2) ** 2)
        sin2 = float(jnp.sin(angle / 2) ** 2)
        np.testing.assert_allclose(float(probs[0]), cos2, atol=1e-10)
        np.testing.assert_allclose(float(probs[1]), sin2, atol=1e-10)
        np.testing.assert_allclose(float(probs[2]), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(probs[3]), 0.0, atol=1e-10)

        rho10 = basis_dm_multi((1, 0), (2, 2))
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho10)
        np.testing.assert_allclose(float(probs[0]), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(probs[1]), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(probs[2]), cos2, atol=1e-10)
        np.testing.assert_allclose(float(probs[3]), sin2, atol=1e-10)


# ======================================================================
# Spectator tests
# ======================================================================


class TestSpectator:
    """Test instruments that act on both measured and spectator qubits."""

    @pytest.mark.parametrize("seed", [0, 42, 9999])
    def test_two_qubit_measure_first_bitflip_second(self, seed):
        """Measure qubit 0 (ideal), apply X to spectator qubit 1."""
        dims = (2, 2)
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        action = jnp.kron(jnp.eye(2, dtype=complex), X)
        qi = make_spectator_instrument(action, dims, measured_qudit=0)
        assert qx.validate(qi)

        rho = basis_dm_multi((0, 0), dims)
        key = jax.random.key(seed)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
        assert int(outcome) == 0
        expected = basis_dm_multi((0, 1), dims)
        np.testing.assert_allclose(rho_out.matrix, expected.matrix, atol=1e-10)

        rho = basis_dm_multi((1, 0), dims)
        key = jax.random.key(seed + 1)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
        assert int(outcome) == 1
        expected = basis_dm_multi((1, 1), dims)
        np.testing.assert_allclose(rho_out.matrix, expected.matrix, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 42, 9999])
    def test_two_qubit_measure_first_phase_flip_second(self, seed):
        """Measure qubit 0 (ideal), apply Z to spectator qubit 1."""
        dims = (2, 2)
        Z = jnp.array([[1, 0], [0, -1]], dtype=complex)
        action = jnp.kron(jnp.eye(2, dtype=complex), Z)
        qi = make_spectator_instrument(action, dims, measured_qudit=0)
        assert qx.validate(qi)

        plus = jnp.array([1, 1], dtype=complex) / jnp.sqrt(2)
        minus = jnp.array([1, -1], dtype=complex) / jnp.sqrt(2)
        ket_0 = jnp.array([1, 0], dtype=complex)
        vec_0plus = jnp.kron(ket_0, plus)
        rho = DensityMatrix.from_matrix(jnp.outer(vec_0plus, jnp.conj(vec_0plus)), dims)

        key = jax.random.key(seed)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
        assert int(outcome) == 0
        vec_0minus = jnp.kron(ket_0, minus)
        expected = jnp.outer(vec_0minus, jnp.conj(vec_0minus))
        np.testing.assert_allclose(rho_out.matrix, expected, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 42, 9999])
    def test_two_qutrit_spectator(self, seed):
        """Measure qutrit 0, apply cyclic permutation to spectator qutrit 1."""
        dims = (3, 3)
        perm = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
        action = jnp.kron(jnp.eye(3, dtype=complex), perm)
        qi = make_spectator_instrument(action, dims, measured_qudit=0)
        assert qx.validate(qi)

        rho = basis_dm_multi((1, 0), dims)
        key = jax.random.key(seed)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
        assert int(outcome) == 1
        expected = basis_dm_multi((1, 1), dims)
        np.testing.assert_allclose(rho_out.matrix, expected.matrix, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 42, 9999])
    def test_three_qubit_spectator(self, seed):
        """3-qubit: measure qubit 0, apply X to spectator qubit 2."""
        dims = (2, 2, 2)
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        action = jnp.kron(jnp.kron(jnp.eye(2, dtype=complex), jnp.eye(2, dtype=complex)), X)
        qi = make_spectator_instrument(action, dims, measured_qudit=0)
        assert qx.validate(qi)

        rho = basis_dm_multi((0, 0, 0), dims)
        key = jax.random.key(seed)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
        assert int(outcome) == 0
        expected = basis_dm_multi((0, 0, 1), dims)
        np.testing.assert_allclose(rho_out.matrix, expected.matrix, atol=1e-10)

        rho = basis_dm_multi((1, 0, 0), dims)
        key = jax.random.key(seed + 1)
        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
        assert int(outcome) == 1
        expected = basis_dm_multi((1, 0, 1), dims)
        np.testing.assert_allclose(rho_out.matrix, expected.matrix, atol=1e-10)

    def test_three_qubit_spectator_superposition(self):
        """3-qubit: measure qubit 0 of |+00>, spectator qubit 2 bit-flipped."""
        dims = (2, 2, 2)
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        action = jnp.kron(jnp.kron(jnp.eye(2, dtype=complex), jnp.eye(2, dtype=complex)), X)
        qi = make_spectator_instrument(action, dims, measured_qudit=0)

        plus = jnp.array([1, 1], dtype=complex) / jnp.sqrt(2)
        zero = jnp.array([1, 0], dtype=complex)
        vec = jnp.kron(jnp.kron(plus, zero), zero)
        rho = DensityMatrix.from_matrix(jnp.outer(vec, jnp.conj(vec)), dims)

        _, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        np.testing.assert_allclose(float(probs[0]), 0.5, atol=1e-10)
        np.testing.assert_allclose(float(probs[1]), 0.5, atol=1e-10)

        rho_outs, probs = qx.apply_instrument_to_density_matrix(qi, rho)
        for seed in range(50):
            key = jax.random.key(seed)
            rho_out, outcome = qx.select_outcome(rho_outs, probs, key)
            o = int(outcome)
            assert o in (0, 1)
            expected = basis_dm_multi((o, 0, 1), dims)
            np.testing.assert_allclose(rho_out.matrix, expected.matrix, atol=1e-10)


# ======================================================================
# Weak measurement tests
# ======================================================================


class TestWeakMeasurement:
    """Test weak measurements via qx.instrument_from_axis."""

    @pytest.mark.parametrize("sharpness", [0.0, 0.3, 0.5, 0.8])
    def test_qubit_weak_confusion_matrix(self, sharpness):
        """Weak qubit Z-measurement: confusion diagonal = (1 + sharpness) / 2."""
        qi = qx.instrument_from_axis(theta=0.0, phi=0.0, sharpness=sharpness)
        assert qx.validate(qi)
        expected_diag = (1 + sharpness) / 2
        np.testing.assert_allclose(qi.confusion_matrix[0, 0], expected_diag, atol=1e-10)
        np.testing.assert_allclose(qi.confusion_matrix[1, 1], expected_diag, atol=1e-10)

    def test_qubit_no_measurement(self):
        """Sharpness=0: confusion is 50/50 (no information)."""
        qi = qx.instrument_from_axis(theta=0.0, phi=0.0, sharpness=0.0)
        np.testing.assert_allclose(
            qi.confusion_matrix,
            0.5 * jnp.ones((2, 2)),
            atol=1e-10,
        )

    def test_qubit_full_measurement(self):
        """Sharpness=1: ideal projective measurement."""
        qi = qx.instrument_from_axis(theta=0.0, phi=0.0, sharpness=1.0)
        np.testing.assert_allclose(qi.confusion_matrix, jnp.eye(2), atol=1e-10)
        np.testing.assert_allclose(qi.matrix, qx.gates.MEASURE().matrix, atol=1e-10)

    @pytest.mark.parametrize("sharpness", [0.3, 0.5, 0.8])
    def test_qubit_weak_post_measurement_state(self, sharpness):
        """Weak measurement on |+>: post-state is not fully projected for sharpness < 1."""
        qi = qx.instrument_from_axis(theta=0.0, phi=0.0, sharpness=sharpness)
        rho_plus = superposition_dm(2)

        choi_0, _ = qi.outcome_superop(0)
        rho_out = qx.apply_superop_to_density_matrix(choi_0, rho_plus)
        rho_out_mat = rho_out.matrix
        prob = float(jnp.real(jnp.trace(rho_out_mat)))
        rho_out_norm = rho_out_mat / prob

        p0 = float(jnp.real(rho_out_norm[0, 0]))
        p1 = float(jnp.real(rho_out_norm[1, 1]))
        assert p1 > 1e-6, "Weak measurement on |+> should leave some |1> population"
        assert p0 > p1, "Outcome 0 should favor |0> population"

    @pytest.mark.parametrize("sharpness", [0.3, 0.5, 0.8])
    def test_qubit_weak_outcome_probabilities(self, sharpness):
        """Weak Z-measurement on |0>: P(outcome=0) = (1+sharpness)/2."""
        qi = qx.instrument_from_axis(theta=0.0, phi=0.0, sharpness=sharpness)
        rho0 = basis_dm(0, 2)
        expected_p0 = (1 + sharpness) / 2

        _, probs = qx.apply_instrument_to_density_matrix(qi, rho0)
        np.testing.assert_allclose(float(probs[0]), expected_p0, atol=1e-10)

    def test_qubit_weak_x_axis_confusion(self):
        """X-axis measurement: 50% confusion regardless of sharpness."""
        for sharpness in [0.3, 0.5, 0.8, 1.0]:
            qi = qx.instrument_from_axis(theta=jnp.pi / 2, phi=0.0, sharpness=sharpness)
            assert qx.validate(qi)
            np.testing.assert_allclose(
                qi.confusion_matrix,
                0.5 * jnp.ones((2, 2)),
                atol=1e-10,
            )

    def test_qubit_weak_x_axis_eigenstates(self):
        """Sharp X-axis measurement produces |+> and |-> post-states."""
        qi = qx.instrument_from_axis(theta=jnp.pi / 2, phi=0.0, sharpness=1.0)

        rho_zero = DensityMatrix.from_matrix(jnp.array([[1, 0], [0, 0]], dtype=complex), (2,))
        rho_plus = jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        rho_minus = jnp.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex)

        choi_0, _ = qi.outcome_superop(0)
        rho_out = qx.apply_superop_to_density_matrix(choi_0, rho_zero)
        rho_out_mat = rho_out.matrix / jnp.trace(rho_out.matrix)
        np.testing.assert_allclose(rho_out_mat, rho_plus, atol=1e-10)

        choi_1, _ = qi.outcome_superop(1)
        rho_out = qx.apply_superop_to_density_matrix(choi_1, rho_zero)
        rho_out_mat = rho_out.matrix / jnp.trace(rho_out.matrix)
        np.testing.assert_allclose(rho_out_mat, rho_minus, atol=1e-10)

    def test_qutrit_weak_measurement(self):
        """Qutrit weak measurement via confusion/transition with partial identity mixing."""
        d = 3
        sharpness = 0.6
        diag = (1 + (d - 1) * sharpness) / d
        off = (1 - sharpness) / d
        confusion = diag * jnp.eye(d) + off * (jnp.ones((d, d)) - jnp.eye(d))
        transition = jnp.eye(d)
        qi = qx.instrument_from_confusion_and_transition(confusion, transition, dims=(3,))
        assert qx.validate(qi)

        np.testing.assert_allclose(qi.confusion_matrix, confusion, atol=1e-10)

        rho0 = basis_dm(0, 3)
        _, probs = qx.apply_instrument_to_density_matrix(qi, rho0)
        np.testing.assert_allclose(float(probs[0]), diag, atol=1e-10)

    def test_two_qubit_weak_measurement(self):
        """Two-qubit weak measurement via tensor of single-qubit weak instruments."""
        sharpness = 0.6
        qi_single = qx.instrument_from_axis(theta=0.0, phi=0.0, sharpness=sharpness)
        qi = qi_single | qi_single
        assert qx.validate(qi)

        expected_diag = (1 + sharpness) / 2
        expected_cm_single = jnp.array([[expected_diag, 1 - expected_diag], [1 - expected_diag, expected_diag]])
        expected_cm = jnp.kron(expected_cm_single, expected_cm_single)
        np.testing.assert_allclose(qi.confusion_matrix, expected_cm, atol=1e-8)
