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
