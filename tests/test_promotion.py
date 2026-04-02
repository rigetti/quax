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
    Embed a QuTiP superoperator in a larger space (CPTP).

    We define a embedding operator, E which is (d_out, d_in) with E[:d_in, :d_in] = I and zeros elsewhere.
    The super-embedding operator is E* ⊗ E.

    The complement projector is Q = I - P where P = EE†.
    The CPTP promoted channel is::

        S' = E_s @ S @ E_s† + Q ⊗ Q

    where Q ⊗ Q = to_super(Q) acts as identity only within the complement
    subspace, zeroing out cross-term coherences between the original and
    complement subspaces.
    """
    # Ei is the embedding operator for each qudit
    Ei = [qt.Qobj(jnp.diag(jnp.ones(dout))[:, :din], dims=[[dout], [din]]) for din, dout in zip(d_in, d_target)]

    # E is the full embedding operator for the entire system (tensor product of individual embeddings)
    E = reduce(qt.tensor, Ei)

    # super_E is the superembedding operator for the entire system
    super_E = qt.tensor(E.conj(), E)
    super_dims = [[list(d_target), list(d_target)], [list(d_in), list(d_in)]]
    super_E = qt.Qobj(super_E.data, dims=super_dims, superrep="super")

    # Complement projector Q = I - P where P = EE†
    I_target = reduce(qt.tensor, [qt.qeye(d) for d in d_target])
    P = E @ E.dag()
    Q = I_target - P

    return super_E @ qt_superop @ super_E.dag() + qt.to_super(Q)


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
    op = qx.random_operator(dims=dims, key=key, size=ensemble_size)  # for seeding
    print(op)
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
        qx.promote("not a quantum object", (3,))


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
