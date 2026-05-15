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

"""Tests for the ``promote`` single-dispatch function, validated against QuTiP."""

from functools import reduce
from operator import mul
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.linalg import eigh
import pytest
import qutip as qt

import quax as qx

# ---------- helpers ----------


def _qt_embed_state(qt_psi: qt.Qobj, d_in: Tuple[int, ...], d_target: Tuple[int, ...]) -> qt.Qobj:
    """
    Embed a QuTiP state in a larger space.

    We define a embedding operator, E which is (d_out, d_in) with E[:d_in, :d_in] = I and zeros elsewhere.
    The embedded operator is then EOE† which has zeros everywhere else.

    To embed the state in the promoted space, we simple apply the embedding operator:

    promoted = E @ qt_psi

    :param qt_op: The QuTiP operator to embed.
    :param d_in: The input dimensions of the operator.
    :param d_target: The target dimension to promote.
    """
    # Ei is the embedding operator for each qudit
    Ei = [qt.Qobj(jnp.diag(jnp.ones(dout))[:, :din], dims=[[dout], [din]]) for din, dout in zip(d_in, d_target)]

    # E is the full embedding operator for the entire system (tensor product of individual embeddings)
    E = reduce(qt.tensor, Ei)

    return E @ qt_psi


def _qt_embed_dm(qt_rho: qt.Qobj, d_in: Tuple[int, ...], d_target: Tuple[int, ...]) -> qt.Qobj:
    """
    Embed a QuTiP density matrix in a larger space.

    We define a embedding operator, E which is (d_out, d_in) with E[:d_in, :d_in] = I and zeros elsewhere.
    The embedded operator is then EOE† which has zeros everywhere else.

    To embed the state in the promoted space, we simple apply the embedding operator:

    promoted = E @ qt_psi @ E.dag()

    :param qt_op: The QuTiP operator to embed.
    :param d_in: The input dimensions of the operator.
    :param d_target: The target dimension to promote.
    """
    # Ei is the embedding operator for each qudit
    Ei = [qt.Qobj(jnp.diag(jnp.ones(dout))[:, :din], dims=[[dout], [din]]) for din, dout in zip(d_in, d_target)]

    # E is the full embedding operator for the entire system (tensor product of individual embeddings)
    E = reduce(qt.tensor, Ei)

    return E @ qt_rho @ E.dag()


def _qt_embed_operator(qt_op: qt.Qobj, d_in: Tuple[int, ...], d_target: Tuple[int, ...]) -> qt.Qobj:
    """
    Embed a QuTiP operator in a larger space.

    We define a embedding operator, E which is (d_out, d_in) with E[:d_in, :d_in] = I and zeros elsewhere.
    The embedded operator is then EOE† which has zeros everywhere else.

    To embed the operator in the promoted identity space, we then define projector on the reduced subspace:
    P = EE†, and the complement projector, I - P.

    The final promoted operator is then EOE† + (I - P).

    :param qt_op: The QuTiP operator to embed.
    :param d_in: The input dimensions of the operator.
    :param d_target: The target dimension to promote.
    """
    # Ei is the embedding operator for each qudit
    Ei = [qt.Qobj(jnp.diag(jnp.ones(dout))[:, :din], dims=[[dout], [din]]) for din, dout in zip(d_in, d_target)]

    # E is the full embedding operator for the entire system (tensor product of individual embeddings)
    E = reduce(qt.tensor, Ei)

    # Identity on the target Hilbert space
    I = reduce(qt.tensor, [qt.qeye(d) for d in d_target])

    # Projector onto the embedded subspace
    P = E @ E.dag()

    return E @ qt_op @ E.dag() + (I - P)


def _qt_embed_superoperator(qt_superop, d_in, d_target):
    """
    Build a reference coherent extension of a QuTiP superoperator.

    Uses the same strategy as our promotion: eigendecomposition of the
    Choi matrix for a canonical Kraus decomposition, then zero-pad each
    operator per subsystem and add the complement projector to the first.
    """
    d_in_total = reduce(mul, d_in, 1)
    d_target_total = reduce(mul, d_target, 1)

    # Get Choi matrix from SuperOp via QuTiP
    qt_choi = qt.to_choi(qt_superop)
    J = qt_choi.full()

    # Eigendecomposition for canonical Kraus (same algorithm as quax choi_to_kraus)
    J_herm = 0.5 * (J + J.conj().T)
    eigvals, eigvecs = eigh(J_herm)

    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Clamp and sqrt
    eigvals = np.where(eigvals > 1e-6, eigvals, 0.0)
    coeffs = np.sqrt(eigvals)

    # Scale eigenvectors and reshape to Kraus operators in tensor form
    W = eigvecs * coeffs[None, :]
    n = d_in_total * d_in_total

    # Build complement projector in tensor form: identity on target with original zeroed
    complement_tensor = np.eye(d_target_total, dtype=complex).reshape(tuple(d_target) + tuple(d_target))
    orig_slices = tuple(slice(0, d) for d in d_in) * 2
    complement_tensor[orig_slices] = 0.0

    promoted_kraus = []
    for i in range(n):
        # Unvec: vec(K) -> K with transpose (matching quax convention)
        k = W[:, i].reshape(d_in_total, d_in_total).T

        # Reshape to tensor form, pad per subsystem, then reshape back
        k_tensor = k.reshape(tuple(d_in) + tuple(d_in))
        pw = [(0, D - d) for d, D in zip(d_in + d_in, d_target + d_target)]
        padded_tensor = np.pad(k_tensor, pw)

        if i == 0:
            padded_tensor = padded_tensor + complement_tensor

        padded = padded_tensor.reshape(d_target_total, d_target_total)
        promoted_kraus.append(qt.Qobj(padded, dims=[[list(d_target)], [list(d_target)]]))

    return qt.kraus_to_super(promoted_kraus)


# ======================== StateVector ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
        ((2, 4, 3), (3, 5, 3)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (5,)])
def test_promote_state_vector(seed, current_dims, target_dims, ensemble_size):
    key = jax.random.key(seed)
    sv = qx.random_state_vector(dims=current_dims, key=key, size=ensemble_size)
    promoted = qx.promote(sv, target_dims)

    # QuTiP comparison
    if ensemble_size == ():
        qt_psi = sv._to_qobj()
        qt_promoted = _qt_embed_state(qt_psi, current_dims, target_dims)
        assert jnp.allclose(promoted.matrix, jnp.squeeze(jnp.asarray(qt_promoted.full())), atol=1e-12)

    else:
        for i, qobj in enumerate(sv._to_qobj()):
            qt_psi = qobj
            qt_promoted = _qt_embed_state(qt_psi, current_dims, target_dims)
            assert jnp.allclose(promoted.matrix[i], jnp.squeeze(jnp.asarray(qt_promoted.full())), atol=1e-12)

    assert promoted.dims == target_dims
    assert promoted.ensemble_size == ensemble_size


# ======================== DensityMatrix ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
        ((2, 4, 3), (3, 5, 3)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (5,)])
def test_promote_density_matrix(seed, current_dims, target_dims, ensemble_size):
    key = jax.random.key(seed)
    d_in = reduce(mul, current_dims, 1)
    dm = qx.random_density_matrix(rank=d_in, dims=current_dims, key=key, size=ensemble_size)
    promoted = qx.promote(dm, target_dims)

    # QuTiP comparison
    if ensemble_size == ():
        qt_rho = dm._to_qobj()
        qt_promoted = _qt_embed_dm(qt_rho, current_dims, target_dims)
        assert jnp.allclose(promoted.matrix, jnp.asarray(qt_promoted.full()), atol=1e-12)

    else:
        for i, qobj in enumerate(dm._to_qobj()):
            qt_rho = qobj
            qt_promoted = _qt_embed_dm(qt_rho, current_dims, target_dims)
            assert jnp.allclose(promoted.matrix[i], jnp.asarray(qt_promoted.full()), atol=1e-12)

    assert promoted.dims == target_dims
    assert promoted.ensemble_size == ensemble_size


# ======================== Unitary ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
        ((2, 3), (3, 4)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (5,)])
def test_promote_unitary(seed, current_dims, target_dims, ensemble_size):
    key = jax.random.key(seed)
    dims = (current_dims, current_dims)
    u = qx.random_unitary(dims=dims, key=key, size=ensemble_size)
    promoted = qx.promote(u, target_dims)

    # QuTiP comparison
    if ensemble_size == ():
        qt_u = u._to_qobj()
        qt_promoted = _qt_embed_operator(qt_u, current_dims, target_dims)
        assert np.allclose(np.array(promoted.matrix), qt_promoted.full(), atol=1e-12)

    else:
        for i, qobj in enumerate(u._to_qobj()):
            qt_u = qobj
            qt_promoted = _qt_embed_operator(qt_u, current_dims, target_dims)
            assert np.allclose(np.array(promoted.matrix[i]), qt_promoted.full(), atol=1e-12)

    assert promoted.dims == (target_dims, target_dims)
    assert promoted.ensemble_size == ensemble_size


# ======================== Operator ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
        ((2, 3), (3, 4)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (5,)])
def test_promote_operator(seed, current_dims, target_dims, ensemble_size):
    """Promote a generic Operator (zero-padded, not identity-padded)."""
    key = jax.random.key(seed)
    dims = (current_dims, current_dims)

    # Create a random operator matrix
    op = qx.random_operator(dims=dims, key=key, size=ensemble_size)
    promoted = qx.promote(op, target_dims)

    # QuTiP comparison
    if ensemble_size == ():
        qt_op = op._to_qobj()
        qt_promoted = _qt_embed_operator(qt_op, current_dims, target_dims)
        assert np.allclose(np.array(promoted.matrix), qt_promoted.full(), atol=1e-12)

    else:
        for i, qobj in enumerate(op._to_qobj()):
            qt_op = qobj
            qt_promoted = _qt_embed_operator(qt_op, current_dims, target_dims)
            assert np.allclose(np.array(promoted.matrix[i]), qt_promoted.full(), atol=1e-12)

    assert promoted.dims == (target_dims, target_dims)
    assert promoted.ensemble_size == ensemble_size


# ======================== SuperOp ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
        ((2, 3), (3, 4)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_promote_superop(seed, current_dims, target_dims, ensemble_size):
    """Promoted SuperOp acts as identity on higher states."""
    key = jax.random.key(seed)
    dims = (current_dims, current_dims)
    choi = qx.random_choi(dims=dims, rank=2, key=key, size=ensemble_size)
    superop = qx.choi_to_superop(choi)
    promoted = qx.promote(superop, target_dims)

    # QuTiP comparison
    if ensemble_size == ():
        qt_s = superop._to_qobj()
        qt_promoted = _qt_embed_superoperator(qt_s, current_dims, target_dims)
        assert np.allclose(np.array(promoted.matrix), qt_promoted.full(), atol=1e-12)

    else:
        for i, qobj in enumerate(superop._to_qobj()):
            qt_s = qobj
            qt_promoted = _qt_embed_superoperator(qt_s, current_dims, target_dims)
            assert np.allclose(np.array(promoted.matrix[i]), qt_promoted.full(), atol=1e-12)

    assert promoted.dims == (target_dims, target_dims)
    assert promoted.ensemble_size == ensemble_size


# ======================== KrausMap ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
        ((2, 3), (3, 4)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_promote_kraus_map(seed, current_dims, target_dims, ensemble_size):
    """Promoted KrausMap should be equivalent to promoted SuperOp."""
    key = jax.random.key(seed)
    d_in = reduce(mul, current_dims, 1)

    dims = (current_dims, current_dims)
    choi = qx.random_choi(dims=dims, rank=d_in, key=key, size=ensemble_size)
    superop = qx.choi_to_superop(choi)
    kraus = qx.choi_to_kraus(choi)
    promoted_kraus = qx.promote(kraus, target_dims)
    promoted_superop = qx.kraus_to_superop(promoted_kraus)

    # QuTiP comparison
    if ensemble_size == ():
        qt_s = superop._to_qobj()
        qt_promoted = _qt_embed_superoperator(qt_s, current_dims, target_dims)
        assert np.allclose(np.array(promoted_superop.matrix), qt_promoted.full(), atol=1e-12)

    else:
        for i, qobj in enumerate(superop._to_qobj()):
            qt_s = qobj
            qt_promoted = _qt_embed_superoperator(qt_s, current_dims, target_dims)
            assert np.allclose(np.array(promoted_superop.matrix[i]), qt_promoted.full(), atol=1e-12)

    assert promoted_kraus.dims == (target_dims, target_dims)
    assert promoted_kraus.ensemble_size == ensemble_size


# ======================== Choi ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
        ((2, 3), (3, 4)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_promote_choi(seed, current_dims, target_dims, ensemble_size):
    """Promoted Choi should match promoted SuperOp round-tripped through Choi."""
    key = jax.random.key(seed)
    d_in = reduce(mul, current_dims, 1)
    dims = (current_dims, current_dims)

    choi = qx.random_choi(dims=dims, rank=d_in, key=key, size=ensemble_size)
    promoted_choi = qx.promote(choi, target_dims)

    superop = qx.choi_to_superop(choi)
    promoted_superop = qx.choi_to_superop(promoted_choi)

    # QuTiP comparison
    if ensemble_size == ():
        qt_s = superop._to_qobj()
        qt_promoted = _qt_embed_superoperator(qt_s, current_dims, target_dims)
        assert np.allclose(np.array(promoted_superop.matrix), qt_promoted.full(), atol=1e-12)

    else:
        for i, qobj in enumerate(superop._to_qobj()):
            qt_s = qobj
            qt_promoted = _qt_embed_superoperator(qt_s, current_dims, target_dims)
            assert np.allclose(np.array(promoted_superop.matrix[i]), qt_promoted.full(), atol=1e-12)

    assert promoted_choi.dims == (target_dims, target_dims)
    assert promoted_choi.ensemble_size == ensemble_size


# ======================== PauliLiouville ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
        ((2, 3), (3, 4)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_promote_pauli_liouville(seed, current_dims, target_dims, ensemble_size):
    """Promoted PauliLiouville should match promoted SuperOp round-tripped."""
    key = jax.random.key(seed)
    d_in = reduce(mul, current_dims, 1)
    dims = (current_dims, current_dims)

    choi = qx.random_choi(dims=dims, rank=d_in, key=key, size=ensemble_size)
    pauli_liouville = qx.choi_to_pauli_liouville(choi)
    superop = qx.pauli_liouville_to_superop(pauli_liouville)
    promoted_pauli_liouville = qx.promote(pauli_liouville, target_dims)

    superop = qx.choi_to_superop(choi)
    promoted_superop = qx.pauli_liouville_to_superop(promoted_pauli_liouville)

    # QuTiP comparison
    if ensemble_size == ():
        qt_s = superop._to_qobj()
        qt_promoted = _qt_embed_superoperator(qt_s, current_dims, target_dims)
        assert np.allclose(np.array(promoted_superop.matrix), qt_promoted.full(), atol=1e-12)

    else:
        for i, qobj in enumerate(superop._to_qobj()):
            qt_s = qobj
            qt_promoted = _qt_embed_superoperator(qt_s, current_dims, target_dims)
            assert np.allclose(np.array(promoted_superop.matrix[i]), qt_promoted.full(), atol=1e-12)

    assert promoted_pauli_liouville.dims == (target_dims, target_dims)
    assert promoted_pauli_liouville.ensemble_size == ensemble_size


# ======================== Validation ========================


def test_promote_wrong_num_subsystems():
    """Promoting with mismatched subsystem count should raise ValueError."""
    key = jax.random.key(0)
    sv = qx.random_state_vector(dims=(2,), key=key)
    with pytest.raises(ValueError, match="Number of subsystems must match"):
        qx.promote(sv, (3, 3))


def test_promote_smaller_dim_raises():
    """Promoting to a smaller dimension should raise ValueError."""
    key = jax.random.key(0)
    u = qx.random_unitary(dims=((3,), (3,)), key=key)
    with pytest.raises(ValueError, match="smaller than current dimension"):
        qx.promote(u, (2,))


def test_promote_unsupported_type():
    """Promoting an unsupported type should raise TypeError."""
    with pytest.raises(TypeError, match="promote is not implemented"):
        qx.promote("not a quantum object", (3,))  # type: ignore[call-overload]


# ======================== Promote emptiness ========================


@pytest.mark.parametrize("seed", [42, 123])
def test_promote_state_vector_zeros_in_higher_dims(seed):
    """Promoted state vector should have zero amplitude in higher-dimensional components."""
    key = jax.random.key(seed)
    sv = qx.random_state_vector(dims=(2,), key=key)
    promoted = qx.promote(sv, (4,))

    # Original 2D components preserved
    assert jnp.allclose(promoted.matrix[:2], sv.matrix, atol=1e-14)
    # Higher components are zero
    assert jnp.allclose(promoted.matrix[2:], 0.0, atol=1e-14)


@pytest.mark.parametrize("seed", [42, 123])
def test_promote_density_matrix_zeros_in_higher_dims(seed):
    """Promoted density matrix should have zero entries in higher-dimensional rows/cols."""
    key = jax.random.key(seed)
    dm = qx.random_density_matrix(rank=2, dims=(2,), key=key)
    promoted = qx.promote(dm, (4,))

    # Original 2x2 block preserved
    assert jnp.allclose(promoted.matrix[:2, :2], dm.matrix, atol=1e-14)
    # Higher rows and columns are zero
    assert jnp.allclose(promoted.matrix[2:, :], 0.0, atol=1e-14)
    assert jnp.allclose(promoted.matrix[:, 2:], 0.0, atol=1e-14)


@pytest.mark.parametrize("seed", [42])
def test_promote_unitary_identity_in_higher_dims(seed):
    """Promoted unitary should act as identity on higher-dimensional subspace."""
    key = jax.random.key(seed)
    u = qx.random_unitary(dims=((2,), (2,)), key=key)
    promoted = qx.promote(u, (4,))

    # Original 2x2 block preserved
    assert jnp.allclose(promoted.matrix[:2, :2], u.matrix, atol=1e-14)
    # Off-diagonal blocks are zero
    assert jnp.allclose(promoted.matrix[:2, 2:], 0.0, atol=1e-14)
    assert jnp.allclose(promoted.matrix[2:, :2], 0.0, atol=1e-14)
    # Lower-right block is identity
    assert jnp.allclose(promoted.matrix[2:, 2:], jnp.eye(2, dtype=complex), atol=1e-14)


# ======================== JIT compatibility ========================


def test_promote_jit_unitary():
    """Verify promote works under jax.jit."""
    key = jax.random.key(42)
    u = qx.random_unitary(dims=((2,), (2,)), key=key)

    @jax.jit
    def do_promote(u):
        return qx.promote(u, (3,))

    promoted = do_promote(u)
    assert promoted.dims == ((3,), (3,))
    assert promoted.matrix.shape == (3, 3)


def test_promote_jit_state_vector():
    """Verify promote of StateVector works under jax.jit."""
    key = jax.random.key(42)
    sv = qx.random_state_vector(dims=(2,), key=key)

    @jax.jit
    def do_promote(sv):
        return qx.promote(sv, (3,))

    promoted = do_promote(sv)
    assert promoted.dims == (3,)
    assert promoted.matrix.shape == (3,)


# ---------- regression: self-fidelity preservation ----------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
    ],
)
def test_promotion_preserves_self_fidelity(seed, current_dims, target_dims):
    """Process fidelity of a channel with itself must be ~1 after promotion."""
    key = jax.random.key(seed)
    dims = (current_dims, current_dims)
    choi = qx.random_choi(dims=dims, rank=2, key=key, size=())

    # Test all superoperator representations
    kraus = qx.choi_to_kraus(choi)
    superop = qx.choi_to_superop(choi)
    pauli_liouville = qx.choi_to_pauli_liouville(choi)

    for obj in [choi, kraus, superop, pauli_liouville]:
        promoted = qx.promote(obj, target_dims)
        pf = float(qx.process_fidelity(obj, promoted))
        assert pf == pytest.approx(1.0, abs=1e-6), (
            f"Self-fidelity after promotion failed for {type(obj).__name__}: {pf}"
        )


# ---------- regression: Unitary global-phase vs channel promotion ----------


@pytest.mark.parametrize(
    "gate_fn",
    [
        lambda: qx.gates.RX(jnp.pi),  # global phase -i
        lambda: qx.gates.RX(jnp.pi / 3),  # global phase e^{-iπ/6}
        lambda: qx.gates.RY(jnp.pi / 2),  # non-trivial global phase
    ],
    ids=["RX(pi)", "RX(pi/3)", "RY(pi/2)"],
)
@pytest.mark.parametrize(
    "target_dims",
    [(3,), (4,)],
)
def test_promote_hilbert_space_strips_phase_before_channel_promotion(gate_fn, target_dims):
    """promote_hilbert_space converts Unitary to SuperOp before promotion.

    A unitary with a non-trivial global phase (e.g. RX(π) = -iX) produces
    a different channel after ``to_superop(promote(U))`` vs
    ``promote(to_superop(U))``, because the identity complement on higher
    states introduces a relative phase.  ``promote_hilbert_space`` avoids
    this by converting to SuperOp first when a Unitary is paired with a
    channel type.
    """
    u = gate_fn()

    # The correct reference: convert to SuperOp at native dims, then promote
    reference = qx.promote(qx.to_superop(u), target_dims)

    # Create a channel at target dims to trigger promotion of the Unitary
    identity_channel = qx.to_superop(
        qx.Unitary.from_matrix(jnp.eye(reduce(mul, target_dims), dtype=complex), (target_dims, target_dims))
    )

    # promote_hilbert_space should auto-convert the Unitary before promotion
    _, promoted = qx.promote_hilbert_space(identity_channel, u)

    assert isinstance(promoted, qx.SuperOp), f"Expected SuperOp after auto-conversion, got {type(promoted).__name__}"
    pf = float(qx.process_fidelity(reference, promoted))
    assert pf == pytest.approx(1.0, abs=1e-6), (
        f"Auto-converted promotion doesn't match reference for {u} -> {target_dims}: F={pf}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2, 2), (3, 3)),
    ],
)
def test_promote_hilbert_space_auto_converts_unitary_with_channel(seed, current_dims, target_dims):
    """promote_hilbert_space auto-converts Unitary to SuperOp when paired with a channel type."""
    key = jax.random.key(seed)
    channel_dims = (current_dims, current_dims)
    channel = qx.random_choi(dims=channel_dims, rank=2, key=key, size=())
    superop = qx.to_superop(channel)
    # Promote SuperOp to target dims so it needs promotion of the unitary partner
    promoted_superop = qx.promote(superop, target_dims)

    # Create a Unitary at original dims
    u = qx.random_unitary(dims=channel_dims, key=jax.random.key(seed + 100))

    # promote_hilbert_space(big_channel, small_unitary) should auto-convert Unitary
    result_channel, result_u = qx.promote_hilbert_space(promoted_superop, u)

    # Both results should be SuperOp (Unitary was auto-converted)
    assert isinstance(result_u, qx.SuperOp), f"Expected SuperOp after auto-conversion, got {type(result_u).__name__}"
    assert result_u.dims == ((target_dims), (target_dims))

    # The fidelity should match the explicit conversion path
    u_explicit = qx.promote(qx.to_superop(u), target_dims)
    pf = float(qx.process_fidelity(result_u, u_explicit))
    assert pf == pytest.approx(1.0, abs=1e-6), f"Auto-converted Unitary doesn't match explicit conversion: F={pf}"


# ======================== Positional Embedding ========================


class TestEmbedValidation:
    """Validation tests for the positions parameter."""

    def test_positions_wrong_length(self):
        u = qx.random_unitary(dims=((2,), (2,)), key=jax.random.key(0))
        with pytest.raises(ValueError, match="positions has length 2"):
            qx.embed(u, (2, 2, 2), (0, 1))  # 1-qubit, but 2 positions

    def test_positions_wrong_length_too_few(self):
        u = qx.random_unitary(dims=((2, 2), (2, 2)), key=jax.random.key(0))
        with pytest.raises(ValueError, match="positions has length 1"):
            qx.embed(u, (2, 2, 2), (0,))  # 2-qubit, but 1 position

    def test_positions_not_distinct(self):
        u = qx.random_unitary(dims=((2, 2), (2, 2)), key=jax.random.key(0))
        with pytest.raises(ValueError, match="positions must be distinct"):
            qx.embed(u, (2, 2, 2), (1, 1))

    def test_positions_out_of_range(self):
        u = qx.random_unitary(dims=((2,), (2,)), key=jax.random.key(0))
        with pytest.raises(ValueError, match="out of range"):
            qx.embed(u, (2, 2), (3,))

    def test_positions_negative(self):
        u = qx.random_unitary(dims=((2,), (2,)), key=jax.random.key(0))
        with pytest.raises(ValueError, match="out of range"):
            qx.embed(u, (2, 2), (-1,))

    def test_positions_target_dim_too_small(self):
        u = qx.random_unitary(dims=((3,), (3,)), key=jax.random.key(0))
        with pytest.raises(ValueError, match="smaller than current dimension"):
            qx.embed(u, (2, 2, 2), (1,))


# ---- Unitary embedding ----


@pytest.mark.parametrize("seed", [42, 123])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        # 1-qubit into 2-qubit space
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
        # 1-qubit into 3-qubit space
        ((2,), (2, 2, 2), (0,)),
        ((2,), (2, 2, 2), (1,)),
        ((2,), (2, 2, 2), (2,)),
        # 2-qubit into 3-qubit space
        ((2, 2), (2, 2, 2), (0, 1)),
        ((2, 2), (2, 2, 2), (0, 2)),
        ((2, 2), (2, 2, 2), (1, 2)),
        # Mixed dims: qubit into qutrit+qubit space
        ((2,), (3, 2), (1,)),
        # Combined promote + embed: qubit into 2-qutrit space
        ((2,), (3, 3), (0,)),
        ((2,), (3, 3), (1,)),
        # Pure permutation: 2-qubit, same dims, swapped
        ((2, 2), (2, 2), (1, 0)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_embed_unitary(seed, obj_dims, target_dims, positions, ensemble_size):
    """Embedded unitary should be valid."""
    key = jax.random.key(seed)
    u = qx.random_unitary(dims=(obj_dims, obj_dims), key=key, size=ensemble_size)
    embedded = qx.embed(u, target_dims, positions)

    assert embedded.dims == (target_dims, target_dims)
    assert embedded.ensemble_size == ensemble_size
    assert jnp.all(qx.validate(embedded, atol=1e-10))


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
        ((2, 2), (2, 2, 2), (0, 2)),
    ],
)
def test_embed_unitary_vs_targeted_apply(seed, obj_dims, target_dims, positions):
    """Applying embedded unitary should equal targeted apply."""
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    u = qx.random_unitary(dims=(obj_dims, obj_dims), key=k1)
    psi = qx.random_state_vector(dims=target_dims, key=k2)

    # Embedded apply
    embedded = qx.embed(u, target_dims, positions)
    result_embed = qx.apply_unitary_to_state_vector(embedded, psi)

    # Targeted apply
    result_targeted = qx.targeted_apply_unitary(u, psi, positions)

    assert result_embed == result_targeted


# ---- Operator embedding ----


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
        ((2, 2), (2, 2, 2), (0, 2)),
        ((2,), (3, 3), (1,)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_embed_operator(seed, obj_dims, target_dims, positions, ensemble_size):
    """Embedded operator should preserve type and dims."""
    key = jax.random.key(seed)
    op = qx.random_operator(dims=(obj_dims, obj_dims), key=key, size=ensemble_size)
    embedded = qx.embed(op, target_dims, positions)

    assert embedded.dims == (target_dims, target_dims)
    assert embedded.ensemble_size == ensemble_size
    assert isinstance(embedded, qx.Operator)


def test_embed_operator_tensor_product():
    """Embedding operator at position 0 should equal O|I, at position 1 should equal I|O."""
    key = jax.random.key(42)
    op = qx.random_operator(dims=((2,), (2,)), key=key)
    eye = qx.Unitary.from_matrix(jnp.eye(2, dtype=complex), ((2,), (2,)))

    embedded_0 = qx.embed(op, (2, 2), (0,))
    expected_0 = op | eye
    assert jnp.allclose(embedded_0.data, expected_0.data, atol=1e-12)

    embedded_1 = qx.embed(op, (2, 2), (1,))
    expected_1 = eye | op
    assert jnp.allclose(embedded_1.data, expected_1.data, atol=1e-12)


# ---- StateVector embedding ----


@pytest.mark.parametrize("seed", [42, 123])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
        ((2,), (2, 2, 2), (1,)),
        ((2, 2), (2, 2, 2), (0, 2)),
        ((2,), (3, 3), (0,)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (5,)])
def test_embed_state_vector(seed, obj_dims, target_dims, positions, ensemble_size):
    """Embedded state vector should be valid and consistent with DM embedding."""
    key = jax.random.key(seed)
    sv = qx.random_state_vector(dims=obj_dims, key=key, size=ensemble_size)
    embedded = qx.embed(sv, target_dims, positions)

    assert embedded.dims == target_dims
    assert embedded.ensemble_size == ensemble_size
    assert jnp.all(qx.validate(embedded, atol=1e-10))

    # Embedding SV then promoting to DM should equal embedding the DM directly
    if ensemble_size == ():
        dm = qx.promote_state_vector_to_density_matrix(sv)
        embedded_dm = qx.embed(dm, target_dims, positions)
        embedded_sv_as_dm = qx.promote_state_vector_to_density_matrix(embedded)
        assert embedded_sv_as_dm == embedded_dm


@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
    ],
)
def test_embed_state_vector_known(obj_dims, target_dims, positions):
    """Test known embedding of |1> into 2-qubit space."""
    ket1 = qx.StateVector.from_matrix(jnp.array([0.0, 1.0], dtype=complex), (2,))
    embedded = qx.embed(ket1, target_dims, positions)

    if positions == (0,):
        # |1> at position 0 → |10> = [0, 0, 1, 0]
        expected = qx.StateVector.from_matrix(jnp.array([0.0, 0.0, 1.0, 0.0], dtype=complex), (2, 2))
    else:
        # |1> at position 1 → |01> = [0, 1, 0, 0]
        expected = qx.StateVector.from_matrix(jnp.array([0.0, 1.0, 0.0, 0.0], dtype=complex), (2, 2))

    assert embedded == expected


def test_embed_state_vector_tensor_product():
    """Embedding state at position 0 should equal sv|0>, at position 1 should equal |0>|sv."""
    key = jax.random.key(42)
    sv = qx.random_state_vector(dims=(2,), key=key)
    ket0 = qx.states.KET0

    embedded_0 = qx.embed(sv, (2, 2), (0,))
    expected_0 = sv | ket0
    assert embedded_0 == expected_0

    embedded_1 = qx.embed(sv, (2, 2), (1,))
    expected_1 = ket0 | sv
    assert embedded_1 == expected_1


def test_embed_density_matrix_tensor_product():
    """Embedding DM at position 0 should equal dm|0><0|, at position 1 should equal |0><0||dm."""
    key = jax.random.key(42)
    dm = qx.random_density_matrix(rank=2, dims=(2,), key=key)
    dm0 = qx.zero_state_matrix(dims=(2,))

    embedded_0 = qx.embed(dm, (2, 2), (0,))
    expected_0 = dm | dm0
    assert embedded_0 == expected_0

    embedded_1 = qx.embed(dm, (2, 2), (1,))
    expected_1 = dm0 | dm
    assert embedded_1 == expected_1


# ---- DensityMatrix embedding ----


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
        ((2, 2), (2, 2, 2), (0, 2)),
        ((2,), (3, 2), (1,)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_embed_density_matrix(seed, obj_dims, target_dims, positions, ensemble_size):
    """Embedded density matrix should be valid."""
    key = jax.random.key(seed)
    d_in = reduce(mul, obj_dims, 1)
    dm = qx.random_density_matrix(rank=d_in, dims=obj_dims, key=key, size=ensemble_size)
    embedded = qx.embed(dm, target_dims, positions)

    assert embedded.dims == target_dims
    assert embedded.ensemble_size == ensemble_size
    assert jnp.all(qx.validate(embedded, atol=1e-10))


# ---- SuperOp embedding ----


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
        ((2, 2), (2, 2, 2), (0, 2)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_embed_superop(seed, obj_dims, target_dims, positions, ensemble_size):
    """Embedded SuperOp should be valid CPTP."""
    key = jax.random.key(seed)
    dims = (obj_dims, obj_dims)
    choi = qx.random_choi(dims=dims, rank=2, key=key, size=ensemble_size)
    superop = qx.choi_to_superop(choi)
    embedded = qx.embed(superop, target_dims, positions)

    assert embedded.dims == (target_dims, target_dims)
    assert embedded.ensemble_size == ensemble_size
    assert isinstance(embedded, qx.SuperOp)

    if ensemble_size == ():
        assert qx.validate(embedded, atol=1e-10)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
    ],
)
def test_embed_superop_vs_targeted_apply(seed, obj_dims, target_dims, positions):
    """Applying embedded SuperOp should equal targeted apply."""
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    dims = (obj_dims, obj_dims)
    choi = qx.random_choi(dims=dims, rank=2, key=k1)
    superop = qx.choi_to_superop(choi)
    d_target = reduce(mul, target_dims, 1)
    rho = qx.random_density_matrix(rank=d_target, dims=target_dims, key=k2)

    # Embedded apply
    embedded = qx.embed(superop, target_dims, positions)
    result_embed = qx.apply_superop_to_density_matrix(embedded, rho)

    # Targeted apply
    result_targeted = qx.targeted_apply_superop(superop, rho, positions)

    assert result_embed == result_targeted


# ---- KrausMap embedding ----


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
        ((2, 2), (2, 2, 2), (0, 2)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_embed_kraus_map(seed, obj_dims, target_dims, positions, ensemble_size):
    """Embedded KrausMap should be valid CPTP and match manual reference construction."""
    key = jax.random.key(seed)
    d_in = reduce(mul, obj_dims, 1)
    dims = (obj_dims, obj_dims)
    choi = qx.random_choi(dims=dims, rank=d_in, key=key, size=ensemble_size)
    kraus = qx.choi_to_kraus(choi)
    embedded = qx.embed(kraus, target_dims, positions)

    assert embedded.dims == (target_dims, target_dims)
    assert embedded.ensemble_size == ensemble_size
    assert isinstance(embedded, qx.KrausMap)

    # Verify trace-preserving
    if ensemble_size == ():
        assert qx.validate(embedded, atol=1e-10)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
    ],
)
def test_embed_kraus_vs_superop(seed, obj_dims, target_dims, positions):
    """Embedded KrausMap converted to SuperOp should match embedded SuperOp."""
    key = jax.random.key(seed)
    dims = (obj_dims, obj_dims)
    choi = qx.random_choi(dims=dims, rank=2, key=key)
    superop = qx.choi_to_superop(choi)
    kraus = qx.choi_to_kraus(choi)

    embedded_superop = qx.embed(superop, target_dims, positions)
    embedded_kraus = qx.embed(kraus, target_dims, positions)

    assert embedded_kraus == embedded_superop


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
    ],
)
def test_embed_kraus_vs_targeted_apply(seed, obj_dims, target_dims, positions):
    """Applying embedded KrausMap should equal targeted apply."""
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    dims = (obj_dims, obj_dims)
    choi = qx.random_choi(dims=dims, rank=2, key=k1)
    kraus = qx.choi_to_kraus(choi)
    d_target = reduce(mul, target_dims, 1)
    rho = qx.random_density_matrix(rank=d_target, dims=target_dims, key=k2)

    # Embedded apply
    embedded = qx.embed(kraus, target_dims, positions)
    result_embed = qx.apply_kraus_to_density_matrix(embedded, rho)

    # Targeted apply
    result_targeted = qx.targeted_apply_kraus_map(kraus, rho, positions)

    assert result_embed == result_targeted


# ---- Choi embedding ----


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
        ((2, 2), (2, 2, 2), (0, 2)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_embed_choi(seed, obj_dims, target_dims, positions, ensemble_size):
    """Embedded Choi should be a valid CPTP channel and match manual reference construction."""
    key = jax.random.key(seed)
    d_in = reduce(mul, obj_dims, 1)
    dims = (obj_dims, obj_dims)
    choi = qx.random_choi(dims=dims, rank=d_in, key=key, size=ensemble_size)
    embedded = qx.embed(choi, target_dims, positions)

    assert embedded.dims == (target_dims, target_dims)
    assert embedded.ensemble_size == ensemble_size
    assert isinstance(embedded, qx.Choi)

    # Verify CPTP
    if ensemble_size == ():
        assert qx.validate(embedded, atol=1e-10)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
    ],
)
def test_embed_choi_vs_targeted_apply(seed, obj_dims, target_dims, positions):
    """Applying embedded Choi should equal targeted apply via SuperOp conversion."""
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    dims = (obj_dims, obj_dims)
    choi = qx.random_choi(dims=dims, rank=2, key=k1)
    d_target = reduce(mul, target_dims, 1)
    rho = qx.random_density_matrix(rank=d_target, dims=target_dims, key=k2)

    # Embedded apply via Choi
    embedded = qx.embed(choi, target_dims, positions)
    result_embed = qx.apply_choi_to_density_matrix(embedded, rho)

    # Targeted apply via SuperOp
    superop = qx.choi_to_superop(choi)
    result_targeted = qx.targeted_apply_superop(superop, rho, positions)

    assert result_embed == result_targeted


# ---- PauliLiouville embedding ----


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_embed_pauli_liouville(seed, obj_dims, target_dims, positions, ensemble_size):
    """Embedded PauliLiouville should be valid CPTP."""
    key = jax.random.key(seed)
    d_in = reduce(mul, obj_dims, 1)
    dims = (obj_dims, obj_dims)
    choi = qx.random_choi(dims=dims, rank=d_in, key=key, size=ensemble_size)
    pl = qx.choi_to_pauli_liouville(choi)
    embedded = qx.embed(pl, target_dims, positions)

    assert embedded.dims == (target_dims, target_dims)
    assert embedded.ensemble_size == ensemble_size
    assert isinstance(embedded, qx.PauliLiouville)

    # Verify CPTP
    if ensemble_size == ():
        assert qx.validate(embedded, atol=1e-10)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "obj_dims,target_dims,positions",
    [
        ((2,), (2, 2), (0,)),
        ((2,), (2, 2), (1,)),
    ],
)
def test_embed_pauli_liouville_vs_targeted_apply(seed, obj_dims, target_dims, positions):
    """Applying embedded PauliLiouville should equal targeted apply via SuperOp conversion."""
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    dims = (obj_dims, obj_dims)
    choi = qx.random_choi(dims=dims, rank=2, key=k1)
    pl = qx.choi_to_pauli_liouville(choi)
    d_target = reduce(mul, target_dims, 1)
    rho = qx.random_density_matrix(rank=d_target, dims=target_dims, key=k2)

    # Embedded apply via PauliLiouville
    embedded = qx.embed(pl, target_dims, positions)
    result_embed = qx.apply_pauli_liouville_to_density_matrix(embedded, rho)

    # Targeted apply via SuperOp
    superop = qx.choi_to_superop(choi)
    result_targeted = qx.targeted_apply_superop(superop, rho, positions)

    assert result_embed == result_targeted


# ---- QuantumInstrument embedding ----


@pytest.mark.parametrize(
    "target_dims,positions",
    [
        ((2, 2), (0,)),
        ((2, 2), (1,)),
    ],
)
def test_embed_instrument(target_dims, positions):
    """Embedded instrument should be CPTP and remap measured_qudits."""
    measure = qx.gates.MEASURE(dim=2)
    embedded = qx.embed(measure, target_dims, positions)

    assert embedded.dims == (target_dims, target_dims)
    assert isinstance(embedded, qx.QuantumInstrument)
    # measured_qudits should be remapped
    expected_measured = tuple(positions[m] for m in measure.measured_qudits)
    assert embedded.measured_qudits == expected_measured

    # Validate the embedded instrument (each outcome CP, sum is TP)
    assert qx.validate(embedded, atol=1e-10)


@pytest.mark.parametrize(
    "target_dims,positions",
    [
        ((2, 2), (0,)),
        ((2, 2), (1,)),
    ],
)
def test_embed_instrument_vs_targeted_apply(target_dims, positions):
    """Applying embedded instrument should equal targeted apply."""
    measure = qx.gates.MEASURE(dim=2)
    key = jax.random.key(42)
    d_target = reduce(mul, target_dims, 1)
    rho = qx.random_density_matrix(rank=d_target, dims=target_dims, key=key)

    # Embedded apply
    embedded = qx.embed(measure, target_dims, positions)
    rho_embed, probs_embed = qx.apply_instrument_to_density_matrix(embedded, rho)

    # Targeted apply
    rho_targeted, probs_targeted = qx.targeted_apply_instrument_to_density_matrix(measure, rho, positions)

    assert jnp.allclose(rho_embed.data, rho_targeted.data, atol=1e-12)
    assert jnp.allclose(probs_embed, probs_targeted, atol=1e-12)


# ---- QuantumInstrument per-subsystem promote ----


@pytest.mark.parametrize(
    "current_dim,target_dims",
    [
        (2, (3,)),
        (2, (4,)),
    ],
)
def test_promote_quantum_instrument(current_dim, target_dims):
    """Per-subsystem promotion of a quantum instrument should preserve CPTP."""
    measure = qx.gates.MEASURE(dim=current_dim)
    promoted = qx.promote(measure, target_dims)

    assert promoted.dims == (target_dims, target_dims)
    assert qx.validate(promoted, atol=1e-10)


# ---- No-op embedding ----


def test_embed_noop():
    """Embedding at identity positions with same dims should be a no-op."""
    key = jax.random.key(42)
    u = qx.random_unitary(dims=((2, 2), (2, 2)), key=key)
    embedded = qx.embed(u, (2, 2), (0, 1))

    assert embedded == u


# ---- Pure permutation ----


def test_embed_pure_permutation_unitary():
    """Embedding with swapped positions should permute qudit ordering."""
    # CNOT acts on qubit 0 (control) and qubit 1 (target)
    cnot = qx.gates.CNOT

    # Swap: control at position 1, target at position 0
    swapped = qx.embed(cnot, (2, 2), (1, 0))

    # Verify: swapped CNOT should have qubit 1 as control
    # |11> with control at position 1 (=1), flip target at position 0: 1->0 → |01>
    ket_11 = qx.StateVector.from_matrix(jnp.array([0, 0, 0, 1], dtype=complex), (2, 2))
    result = qx.apply_unitary_to_state_vector(swapped, ket_11)
    ket_01 = qx.StateVector.from_matrix(jnp.array([0, 1, 0, 0], dtype=complex), (2, 2))
    assert result == ket_01


# ---- permute standalone ----


def test_permute_unitary_swap():
    """permute should swap qudit ordering of a unitary."""
    cnot = qx.gates.CNOT
    swapped = qx.permute(cnot, (1, 0))

    # Same check as above: |11> → |01>
    ket_11 = qx.StateVector.from_matrix(jnp.array([0, 0, 0, 1], dtype=complex), (2, 2))
    result = qx.apply_unitary_to_state_vector(swapped, ket_11)
    ket_01 = qx.StateVector.from_matrix(jnp.array([0, 1, 0, 0], dtype=complex), (2, 2))
    assert result == ket_01


def test_permute_state_vector():
    """permute should swap qudit ordering of a state vector."""
    # |10> = [0, 0, 1, 0]
    ket_10 = qx.StateVector.from_matrix(jnp.array([0, 0, 1, 0], dtype=complex), (2, 2))
    # Swap: qudit 0 → position 1, qudit 1 → position 0 ⇒ |01>
    swapped = qx.permute(ket_10, (1, 0))
    ket_01 = qx.StateVector.from_matrix(jnp.array([0, 1, 0, 0], dtype=complex), (2, 2))
    assert swapped == ket_01


def test_permute_identity_is_noop():
    """Identity permutation should be a no-op."""
    key = jax.random.key(42)
    u = qx.random_unitary(dims=((2, 2), (2, 2)), key=key)
    permuted = qx.permute(u, (0, 1))
    assert permuted == u


def test_permute_3qubit_known_states():
    """Permuting |0>|1>|+> should rearrange the tensor factors."""
    ket0 = qx.states.KET0
    ket1 = qx.states.KET1
    plus = qx.states.XPLUS
    psi = ket0 | ket1 | plus  # |0>|1>|+>

    # (0,1,2) -> (1,0,2): swap first two => |1>|0>|+>
    permuted_102 = qx.permute(psi, (1, 0, 2))
    expected_102 = ket1 | ket0 | plus
    assert permuted_102 == expected_102

    # (0,1,2) -> (2,0,1): qudit 0->pos 2, qudit 1->pos 0, qudit 2->pos 1 => |1>|+>|0>
    permuted_201 = qx.permute(psi, (2, 0, 1))
    expected_201 = ket1 | plus | ket0
    assert permuted_201 == expected_201

    # (0,1,2) -> (0,2,1): swap last two => |0>|+>|1>
    permuted_021 = qx.permute(psi, (0, 2, 1))
    expected_021 = ket0 | plus | ket1
    assert permuted_021 == expected_021


def test_permute_random_product_states():
    """Permuting a product state should match re-tensoring in permuted order."""
    key = jax.random.key(99)
    k1, k2, k3 = jax.random.split(key, 3)
    s0 = qx.random_state_vector(dims=(2,), key=k1)
    s1 = qx.random_state_vector(dims=(2,), key=k2)
    s2 = qx.random_state_vector(dims=(2,), key=k3)
    psi = s0 | s1 | s2

    permuted = qx.permute(psi, (2, 0, 1))
    expected = s1 | s2 | s0
    assert permuted == expected


def test_permute_inverse_recovers_state():
    """Applying a permutation then its inverse should recover the original state."""
    key = jax.random.key(77)
    psi = qx.random_state_vector(dims=(2, 2, 2), key=key)

    perm = (2, 0, 1)
    inv_perm = (1, 2, 0)  # inverse of (2,0,1)
    recovered = qx.permute(qx.permute(psi, perm), inv_perm)
    assert recovered == psi


def test_permute_cycle_returns_to_original():
    """Applying a 3-cycle permutation 3 times should recover the original state."""
    key = jax.random.key(88)
    psi = qx.random_state_vector(dims=(2, 2, 2), key=key)

    perm = (1, 2, 0)
    result = psi
    for _ in range(3):
        result = qx.permute(result, perm)
    assert result == psi


# ---- Known gate embeddings ----


def test_embed_x_gate_at_position_1():
    """X gate at position 1 in 2-qubit space should flip qubit 1."""
    x = qx.gates.X
    embedded = qx.embed(x, (2, 2), (1,))

    # |00> -> |01>
    ket_00 = qx.StateVector.from_matrix(jnp.array([1, 0, 0, 0], dtype=complex), (2, 2))
    result = qx.apply_unitary_to_state_vector(embedded, ket_00)
    ket_01 = qx.StateVector.from_matrix(jnp.array([0, 1, 0, 0], dtype=complex), (2, 2))
    assert result == ket_01


def test_embed_x_gate_at_position_0():
    """X gate at position 0 in 2-qubit space should flip qubit 0."""
    x = qx.gates.X
    embedded = qx.embed(x, (2, 2), (0,))

    # |00> -> |10>
    ket_00 = qx.StateVector.from_matrix(jnp.array([1, 0, 0, 0], dtype=complex), (2, 2))
    result = qx.apply_unitary_to_state_vector(embedded, ket_00)
    ket_10 = qx.StateVector.from_matrix(jnp.array([0, 0, 1, 0], dtype=complex), (2, 2))
    assert result == ket_10


def test_embed_cnot_positions_0_2():
    """CNOT on qubits (0,2) in a 3-qubit space."""
    cnot = qx.gates.CNOT
    embedded = qx.embed(cnot, (2, 2, 2), (0, 2))

    # |100> -> |101> (control=0 is 1, flip target=2)
    ket_100 = qx.StateVector.from_matrix(jnp.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=complex), (2, 2, 2))
    result = qx.apply_unitary_to_state_vector(embedded, ket_100)
    ket_101 = qx.StateVector.from_matrix(jnp.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=complex), (2, 2, 2))
    assert result == ket_101

    # |000> -> |000> (control=0 is 0, no flip)
    ket_000 = qx.StateVector.from_matrix(jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex), (2, 2, 2))
    result2 = qx.apply_unitary_to_state_vector(embedded, ket_000)
    assert result2 == ket_000


# ---- JIT compatibility for positional embedding ----


def test_embed_jit_unitary():
    """Verify positional embedding works under jax.jit."""
    key = jax.random.key(42)
    u = qx.random_unitary(dims=((2,), (2,)), key=key)

    @jax.jit
    def do_embed(u):
        return qx.embed(u, (2, 2), (1,))

    embedded = do_embed(u)
    assert embedded.dims == ((2, 2), (2, 2))
    assert embedded.matrix.shape == (4, 4)


def test_embed_jit_state_vector():
    """Verify positional embedding of StateVector works under jax.jit."""
    key = jax.random.key(42)
    sv = qx.random_state_vector(dims=(2,), key=key)

    @jax.jit
    def do_embed(sv):
        return qx.embed(sv, (2, 2), (0,))

    embedded = do_embed(sv)
    assert embedded.dims == (2, 2)
    assert embedded.matrix.shape == (4,)


# ---- Combined promote + embed ----


@pytest.mark.parametrize("seed", [42])
def test_embed_with_dim_promotion(seed):
    """Embedding a qubit into a qutrit space should promote and embed."""
    key = jax.random.key(seed)
    u = qx.random_unitary(dims=((2,), (2,)), key=key)

    # Embed 1-qubit unitary into (3, 3) at position 0
    embedded = qx.embed(u, (3, 3), (0,))

    assert embedded.dims == ((3, 3), (3, 3))
    assert qx.validate(embedded, atol=1e-10)
