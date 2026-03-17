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

Unfortuantely, QuTiP does not have a Pauli-Liouville representation natively, so we only check the round-trip
fidelity for conversions involving Pauli-Liouville.
"""

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
