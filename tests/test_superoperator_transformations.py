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

"""
Superoperator conversion tests.

This test file is responsible for verifying that the jax-based superoperator
conversions are correct. It does this in the following way.

We use a text fixture of random Choi matrices generated from the BCSZ distribution.

Thus, we first test the following convertions against QuTiP:
- Choi -> Superoperator
- Choi -> Pauli-Liouville
- Choi -> KrausMap

With confidence that we can correctly convert from Choi to the other superoperator representations,
we can then trust the converted superoperators to test the inverse conversions:
- Superoperator -> Choi
- Pauli-Liouville -> Choi
- KrausMap -> Choi

Finally, we test the conversions between non-Choi representations:
- Superoperator <-> Pauli-Liouville
- Superoperator <-> KrausMap
- Pauli-Liouville <-> KrausMap

The tests compare the JAX implementations against QuTiP's implementations by checking that the resulting
matrices match (where possible) or that the process fidelities are 1.0.

Unfortuantely, QuTiP does not have a Pauli-Liouville representation natively, so we check the Pauli-Liouville conversions
against some known values.
"""

import jax
import jax.numpy as jnp

import quax as qx

# ============================================================================
# Test Choi -> Superoperator, KrausMap and Pauli-Liouville
# ============================================================================


def test_choi_to_super(random_choi_channels, random_superop_channels):
    """
    Check that choi_to_super matches QuTiP implementation.
    """
    atol = 1e-8

    # JAX transformation
    superop = qx.choi_to_superop(random_choi_channels)

    fid = qx.process_fidelity(superop, random_superop_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    assert jnp.allclose(superop.data, random_superop_channels.data, atol=atol)


def test_choi_to_pauli_liouville(random_choi_channels, random_pauli_liouville_channels):
    """
    Check that choi_to_pauli_liouville matches operator_tools implementation.

    Note: We verify via process fidelity and direct matrix comparison.
    """
    atol = 1e-8

    # JAX transformation
    pl_jax = qx.choi_to_pauli_liouville(random_choi_channels)

    fid = qx.process_fidelity(pl_jax, random_pauli_liouville_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    assert jnp.allclose(pl_jax.data, random_pauli_liouville_channels.data, atol=atol)


def test_choi_to_kraus(random_choi_channels, random_kraus_channels):
    """
    Check that choi_to_kraus matches QuTiP implementation.
    """
    atol = 1e-6

    # JAX transformation
    kraus_maps = qx.choi_to_kraus(random_choi_channels)

    # Compare using process fidelity
    fid = qx.process_fidelity(kraus_maps, random_kraus_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    # The kraus operators can differ up to a global phase
    # assert jnp.allclose(kraus_maps.data, random_kraus_channels.data, atol=atol)


# ============================================================================
# Superoperator, KrausMap and Pauli-Liouville -> Choi
# ============================================================================


def test_super_to_choi(random_choi_channels, random_superop_channels):
    """
    Check that superop_to_choi matches QuTiP implementation.
    """
    atol = 1e-8

    # JAX transformation
    choi_jax = qx.superop_to_choi(random_superop_channels)

    fid = qx.process_fidelity(choi_jax, random_choi_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    assert jnp.allclose(choi_jax.data, random_choi_channels.data, atol=atol)


def test_pauli_liouville_to_choi(random_choi_channels, random_pauli_liouville_channels):
    """
    Check that pauli_liouville_to_choi matches operator_tools implementation.
    """
    atol = 1e-8

    # JAX transformation
    choi_jax = qx.pauli_liouville_to_choi(random_pauli_liouville_channels)

    fid = qx.process_fidelity(choi_jax, random_choi_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    assert jnp.allclose(choi_jax.data, random_choi_channels.data, atol=atol)


def test_kraus_to_choi(random_choi_channels, random_kraus_channels):
    """
    Check that kraus_to_choi matches QuTiP implementation.
    """
    atol = 1e-8

    # JAX transformation
    choi_jax = qx.kraus_to_choi(random_kraus_channels)

    fid = qx.process_fidelity(choi_jax, random_choi_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    assert jnp.allclose(choi_jax.data, random_choi_channels.data, atol=atol)


# ============================================================================
# Test Superoperator <-> Pauli-Liouville
# ============================================================================


def test_superop_to_pauli_liouville(random_superop_channels, random_pauli_liouville_channels):
    """
    Check that superop_to_pauli_liouville matches operator_tools implementation.
    """
    atol = 1e-8

    # JAX transformation
    pl_jax = qx.superop_to_pauli_liouville(random_superop_channels)

    fid = qx.process_fidelity(pl_jax, random_pauli_liouville_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    assert jnp.allclose(pl_jax.data, random_pauli_liouville_channels.data, atol=atol)


def test_pauli_liouville_to_superop(random_superop_channels, random_pauli_liouville_channels):
    """
    Check that pauli_liouville_to_superop matches operator_tools implementation.
    """
    atol = 1e-8

    # JAX transformation
    super_jax = qx.pauli_liouville_to_superop(random_pauli_liouville_channels)

    fid = qx.process_fidelity(super_jax, random_superop_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    assert jnp.allclose(super_jax.data, random_superop_channels.data, atol=atol)


# ============================================================================
# Test KrausMap <-> SuperOp
# ============================================================================


def test_kraus_to_superop(random_kraus_channels, random_superop_channels):
    """
    Check that kraus_to_superop matches QuTiP implementation.
    """
    atol = 1e-8

    # JAX transformation
    super_jax = qx.kraus_to_superop(random_kraus_channels)

    fid = qx.process_fidelity(super_jax, random_superop_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    assert jnp.allclose(super_jax.data, random_superop_channels.data, atol=atol)


def test_super_to_kraus(random_superop_channels, random_kraus_channels):
    """
    Check that superop_to_kraus matches QuTiP implementation.
    """
    atol = 1e-6

    # JAX transformation
    kraus_jax = qx.superop_to_kraus(random_superop_channels)

    fid = qx.process_fidelity(kraus_jax, random_kraus_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    # assert jnp.allclose(kraus_jax.data, random_kraus_channels.data, atol=atol)


# ============================================================================
# Test KrausMap <-> PauliLiouville
# ============================================================================


def test_kraus_to_pauli_liouville(random_kraus_channels, random_pauli_liouville_channels):
    """
    Check that kraus_to_pauli_liouville matches operator_tools implementation.
    """
    atol = 1e-8

    # JAX transformation
    pl_jax = qx.kraus_to_pauli_liouville(random_kraus_channels)

    fid = qx.process_fidelity(pl_jax, random_pauli_liouville_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    assert jnp.allclose(pl_jax.data, random_pauli_liouville_channels.data, atol=atol)


def test_pauli_liouville_to_kraus(random_pauli_liouville_channels, random_kraus_channels):
    """
    Check that pauli_liouville_to_kraus matches operator_tools implementation.
    """
    atol = 1e-6

    # JAX transformation
    kraus_jax = qx.pauli_liouville_to_kraus(random_pauli_liouville_channels)

    fid = qx.process_fidelity(kraus_jax, random_kraus_channels)
    assert jnp.allclose(fid, 1.0, atol=atol)
    # assert jnp.allclose(kraus_jax.data, random_kraus_channels.data, atol=atol)


# ============================================================================
# Test Unitary -> Superoperator
# ============================================================================


def test_unitary_to_superoperator(random_unitaries, random_unitaries_superops):
    """
    Check that unitary_to_superoperator matches QuTiP implementation.
    """
    result = qx.unitary_to_superop(random_unitaries)

    fid = qx.process_fidelity(result, random_unitaries_superops)
    assert jnp.allclose(fid, 1.0, atol=1e-8)
    assert jnp.allclose(result.data, random_unitaries_superops.data, atol=1e-8)


def test_unitary_to_pauli_liouville(random_unitaries, random_unitaries_pauli_liouvilles):
    """
    Check that unitary_to_pauli_liouville matches QuTiP implementation.

    Note: JAX and QuTiP use different Pauli-Liouville basis conventions,
    so we verify via process fidelity rather than direct matrix comparison.
    """
    result = qx.unitary_to_pauli_liouville(random_unitaries)

    fid = qx.process_fidelity(result, random_unitaries_pauli_liouvilles)
    assert jnp.allclose(fid, 1.0, atol=1e-8)
    assert jnp.allclose(result.data, random_unitaries_pauli_liouvilles.data, atol=1e-8)


def test_unitary_to_choi(random_unitaries, random_unitaries_chois):
    """
    Check that unitary_to_choi matches QuTiP implementation.
    """
    result = qx.unitary_to_choi(random_unitaries)

    fid = qx.process_fidelity(result, random_unitaries_chois)
    assert jnp.allclose(fid, 1.0, atol=1e-8)
    assert jnp.allclose(result.data, random_unitaries_chois.data, atol=1e-8)


# ============================================================================
# Round-trip tests (ensuring bijections work)
# ============================================================================


def test_choi_super_roundtrip(random_choi_channels):
    """Test Choi -> Super -> Choi preserves the channel."""
    choi_ref = random_choi_channels

    choi_roundtrip = qx.superop_to_choi(qx.choi_to_superop(choi_ref))

    f = qx.process_fidelity(choi_ref, choi_roundtrip)
    assert jnp.allclose(f, 1.0, atol=1e-6), f"fidelity={f:.5f}"


def test_pauli_liouville_super_roundtrip(random_choi_channels):
    """Test PL -> Super -> PL preserves the channel."""
    choi_ref = random_choi_channels
    pl_ref = qx.choi_to_pauli_liouville(choi_ref)

    pl_roundtrip = qx.superop_to_pauli_liouville(qx.pauli_liouville_to_superop(pl_ref))
    choi_roundtrip = qx.pauli_liouville_to_choi(pl_roundtrip)

    f = qx.process_fidelity(choi_ref, choi_roundtrip)
    assert jnp.allclose(f, 1.0, atol=1e-6), f"fidelity={f:.5f}"


def test_choi_pauli_liouville_roundtrip(random_choi_channels):
    """Test Choi -> PL -> Choi preserves the channel."""
    choi_ref = random_choi_channels

    choi_roundtrip = qx.pauli_liouville_to_choi(qx.choi_to_pauli_liouville(choi_ref))

    f = qx.process_fidelity(choi_ref, choi_roundtrip)
    assert jnp.allclose(f, 1.0, atol=1e-6), f"fidelity={f:.5f}"


# ============================================================================
# Fixture generation (run with:  pytest -k generate_superop_fixtures)
# ============================================================================


def test_generate_superop_fixtures(save_random_choi_channels):
    """Trigger the save_random_choi_channels fixture to write superop .npz files."""
    from pathlib import Path

    assert Path(save_random_choi_channels).exists()


# ============================================================================
# Test truncate_kraus
# ============================================================================


class TestKrausTruncation:
    def test_ordering(self):
        """Verify that truncate_kraus sorts Kraus operators by descending Frobenius norm."""
        # Build a KrausMap with operators in ascending norm order
        k0 = 0.1 * jnp.eye(2, dtype=complex)
        k1 = 0.5 * jnp.eye(2, dtype=complex)
        k2 = 0.9 * jnp.eye(2, dtype=complex)
        data = jnp.stack([k0, k1, k2], axis=0)
        kraus = qx.KrausMap.from_matrix(data, ((2,), (2,)))

        result = qx.truncate_kraus(kraus)
        norms = jnp.linalg.norm(result.matrix, axis=(-2, -1))

        # Norms should be in descending order
        assert norms[0] >= norms[1] >= norms[2]
        assert jnp.allclose(norms[0], 0.9 * jnp.sqrt(2.0))

    def test_removes_zeros(self):
        """Verify that near-zero Kraus operators are removed from the output."""
        # Identity channel: only one significant Kraus operator
        key = jax.random.PRNGKey(0)
        U = qx.random_unitary(dims=((2,), (2,)), key=key)
        kraus = qx.to_kraus(U)

        # Should have d^2 = 4 operators before truncation
        assert kraus.matrix.shape[-3] == 4

        result = qx.truncate_kraus(kraus)

        # Should have 1 significant operator
        assert result.matrix.shape[-3] == 1
        # Channel should be preserved
        fid = qx.process_fidelity(result, kraus)
        assert jnp.allclose(fid, 1.0, atol=1e-6)

    def test_preserves_channel(self, random_kraus_channels):
        """Verify that truncation with a tight tolerance preserves the channel."""
        result = qx.truncate_kraus(random_kraus_channels, atol=1e-10)
        fid = qx.process_fidelity(result, random_kraus_channels)
        assert jnp.allclose(fid, 1.0, atol=1e-6)

    def test_depolarizing(self):
        """Depolarizing channel has exactly 4 Kraus operators (for a single qubit)."""
        kraus = qx.to_kraus(qx.channels.depolarizing(0.1, (2,)))
        result = qx.truncate_kraus(kraus)

        # All 4 operators are significant for nonzero depolarizing probability
        assert result.matrix.shape[-3] == 4
        fid = qx.process_fidelity(result, kraus)
        assert jnp.allclose(fid, 1.0, atol=1e-8)

    def test_ensemble(self):
        """Ensemble with different effective ranks uses max count across members."""
        # Member 0: unitary (1 significant op), Member 1: depolarizing (4 significant ops)
        key = jax.random.PRNGKey(42)
        unitary_kraus = qx.to_kraus(qx.random_unitary(dims=((2,), (2,)), key=key))
        depol_kraus = qx.to_kraus(qx.to_choi(qx.to_kraus(qx.channels.depolarizing(0.1, (2,)))))

        # Stack into an ensemble of shape (2, 4, 2, 2)
        ensemble_data = jnp.stack([unitary_kraus.matrix, depol_kraus.matrix], axis=0)
        ensemble_kraus = qx.KrausMap.from_matrix(ensemble_data, ((2,), (2,)))

        result = qx.truncate_kraus(ensemble_kraus)

        # Should keep 4 (max across ensemble: unitary=1, depolarizing=4)
        assert result.matrix.shape[-3] == 4
        fid = qx.process_fidelity(result, ensemble_kraus)
        assert jnp.allclose(fid, 1.0, atol=1e-6)

    def test_at_least_one(self):
        """Even with a very large tolerance, at least one Kraus operator is kept."""
        kraus = qx.to_kraus(qx.channels.depolarizing(0.001, (2,)))
        result = qx.truncate_kraus(kraus, atol=1e10)

        assert result.matrix.shape[-3] >= 1


# ============================================================================
# Test the dtype-scaled eigenvalue tolerance
# ============================================================================


class TestKrausTolerance:
    """The default clamp in :func:`quax.choi_to_kraus` scales with dtype and spectrum.

    A fixed threshold cannot serve both precisions: ``1e-6`` is the resolution of float32
    arithmetic, but nine orders of magnitude coarser than float64's, where it silently
    discards real low-weight channel components.
    """

    def test_keeps_low_weight_components(self):
        """A channel far weaker than the old fixed 1e-6 keeps its full Kraus rank."""
        # Depolarizing at rate 1e-9 has three Choi eigenvalues around 6.7e-10, well under the
        # old clamp, which would have left only the identity operator.
        channel = qx.channels.depolarizing(1e-9, (2,))
        norms = jnp.linalg.norm(qx.to_kraus(channel).matrix, axis=(-2, -1))
        assert int(jnp.sum(norms > 0)) == 4

        # The old fixed threshold, for contrast: everything but the identity is lost.
        old = qx.choi_to_kraus(qx.to_choi(channel), tol=1e-6)
        assert int(jnp.sum(jnp.linalg.norm(old.matrix, axis=(-2, -1)) > 0)) == 1

    def test_clamps_round_off_on_a_unitary(self):
        """A unitary is rank one: round-off must not become spurious Kraus operators."""
        for gate in (qx.gates.H, qx.gates.CZ):
            norms = jnp.linalg.norm(qx.to_kraus(gate).matrix, axis=(-2, -1))
            assert int(jnp.sum(norms > 0)) == 1

    def test_output_shape_is_independent_of_the_tolerance(self):
        """The Kraus axis stays d_out * d_in, which is what keeps the function jittable."""
        choi = qx.to_choi(qx.channels.depolarizing(0.01, (2,)))
        assert qx.choi_to_kraus(choi).matrix.shape == qx.choi_to_kraus(choi, tol=0.5).matrix.shape

    def test_is_jittable(self):
        """The default tolerance is computed from traced values, so it survives a trace."""
        choi = qx.to_choi(qx.channels.depolarizing(0.01, (2,)))
        jitted = jax.jit(lambda c: qx.choi_to_kraus(c).matrix)

        assert jnp.allclose(jitted(choi), qx.choi_to_kraus(choi).matrix)

    def test_scales_per_ensemble_member(self):
        """Each ensemble member gets a tolerance from its own spectrum."""
        channels = [qx.channels.depolarizing(rate, (2,)) for rate in (0.5, 0.01, 1e-9)]
        ensemble = qx.Choi.from_matrix(jnp.stack([qx.to_choi(c).matrix for c in channels]), channels[0].dims)

        norms = jnp.linalg.norm(qx.choi_to_kraus(ensemble).matrix, axis=(-2, -1))

        assert jnp.all(jnp.sum(norms > 0, axis=-1) == 4)

    def test_truncation_drops_only_round_off(self):
        """``truncate_kraus`` keeps every operator the arithmetic can resolve."""
        assert qx.truncate_kraus(qx.to_kraus(qx.gates.H)).matrix.shape[-3] == 1
        assert qx.truncate_kraus(qx.to_kraus(qx.channels.depolarizing(1e-9, (2,)))).matrix.shape[-3] == 4
