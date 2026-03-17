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
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import qutip as qt  # noqa: E402

from quax import (  # noqa: E402
    DensityMatrix,
    Operator,
    choi_to_superop,
    kraus_to_superop,
    promote,
    random_choi,
    random_density_matrix,
    random_state_vector,
    random_unitary,
    superop_to_choi,
    superop_to_pauli_liouville,
)


# ---------- helpers ----------


def _qt_embed_operator(qt_op, d_in, d_target):
    """Embed a QuTiP operator in a larger space using identity padding."""
    mat = qt_op.full()
    promoted = np.eye(d_target, dtype=complex)
    promoted[:d_in, :d_in] = mat
    return qt.Qobj(promoted)


def _qt_embed_superop(qt_superop, d_in, d_target):
    """Embed a QuTiP superoperator in a larger Liouville space (direct sum with identity on complement)."""
    mat = qt_superop.full()
    # Column-stacking: vec index = j * d + i
    qubit_idxs = [j * d_target + i for j in range(d_in) for i in range(d_in)]
    comp_idxs = [j * d_target + i for j in range(d_in, d_target) for i in range(d_in, d_target)]
    promoted = np.zeros((d_target**2, d_target**2), dtype=complex)
    promoted[np.ix_(qubit_idxs, qubit_idxs)] = mat
    promoted[np.ix_(comp_idxs, comp_idxs)] = np.eye(len(comp_idxs))
    return qt.Qobj(promoted, dims=[[[d_target], [d_target]], [[d_target], [d_target]]], superrep="super")


# ======================== StateVector ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
        ((2,), (2,)),  # identity promotion (no-op)
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (5,)])
def test_promote_state_vector(seed, current_dims, target_dims, ensemble_size):
    key = jax.random.key(seed)
    sv = random_state_vector(dims=current_dims, key=key, size=ensemble_size)
    promoted = promote(sv, target_dims)

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, target_dims, 1)

    assert promoted.dims == target_dims
    assert promoted.matrix.shape == ensemble_size + (d_target,)

    # Upper block matches original
    assert jnp.allclose(promoted.matrix[..., :d_in], sv.matrix, atol=1e-12)
    # Lower block is zero
    assert jnp.allclose(promoted.matrix[..., d_in:], 0.0, atol=1e-12)

    # Compare with QuTiP for scalar case
    if ensemble_size == ():
        qt_sv = sv._to_qobj()
        qt_full = np.zeros(d_target, dtype=complex)
        qt_full[:d_in] = qt_sv.full().ravel()
        qt_promoted = qt.Qobj(qt_full, dims=[list(target_dims), [1] * len(target_dims)])
        assert np.allclose(np.array(promoted.matrix), qt_promoted.full().ravel(), atol=1e-12)


# ======================== DensityMatrix ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (5,)])
def test_promote_density_matrix(seed, current_dims, target_dims, ensemble_size):
    key = jax.random.key(seed)
    from functools import reduce
    from operator import mul

    d_in = reduce(mul, current_dims, 1)
    dm = random_density_matrix(rank=d_in, dims=current_dims, key=key, size=ensemble_size)
    promoted = promote(dm, target_dims)
    d_target = reduce(mul, target_dims, 1)

    assert promoted.dims == target_dims
    assert promoted.matrix.shape == ensemble_size + (d_target, d_target)

    # Upper-left block matches original
    assert jnp.allclose(promoted.matrix[..., :d_in, :d_in], dm.matrix, atol=1e-12)
    # Rest is zero
    assert jnp.allclose(promoted.matrix[..., d_in:, :], 0.0, atol=1e-12)
    assert jnp.allclose(promoted.matrix[..., :d_in, d_in:], 0.0, atol=1e-12)

    # Trace is preserved
    assert jnp.allclose(
        jnp.trace(promoted.matrix, axis1=-2, axis2=-1),
        jnp.trace(dm.matrix, axis1=-2, axis2=-1),
        atol=1e-12,
    )

    # QuTiP comparison
    if ensemble_size == ():
        qt_dm = dm._to_qobj()
        qt_full = np.zeros((d_target, d_target), dtype=complex)
        qt_full[:d_in, :d_in] = qt_dm.full()
        qt_promoted = qt.Qobj(qt_full, dims=[list(target_dims), list(target_dims)])
        assert np.allclose(np.array(promoted.matrix), qt_promoted.full(), atol=1e-12)


# ======================== Unitary ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
        ((2, 2), (3, 3)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (5,)])
def test_promote_unitary(seed, current_dims, target_dims, ensemble_size):
    key = jax.random.key(seed)
    dims = (current_dims, current_dims)
    u = random_unitary(dims=dims, key=key, size=ensemble_size)
    promoted = promote(u, target_dims)

    from functools import reduce
    from operator import mul

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, target_dims, 1)

    assert promoted.dims == (target_dims, target_dims)
    assert promoted.matrix.shape == ensemble_size + (d_target, d_target)

    # Upper-left block matches
    assert jnp.allclose(promoted.matrix[..., :d_in, :d_in], u.matrix, atol=1e-12)
    # Lower-right block is identity
    assert jnp.allclose(
        promoted.matrix[..., d_in:, d_in:],
        jnp.eye(d_target - d_in, dtype=complex),
        atol=1e-12,
    )
    # Off-diagonal blocks are zero
    assert jnp.allclose(promoted.matrix[..., :d_in, d_in:], 0.0, atol=1e-12)
    assert jnp.allclose(promoted.matrix[..., d_in:, :d_in], 0.0, atol=1e-12)

    # Still unitary: U†U = I
    mat = promoted.matrix
    product = jnp.einsum("...ji,...jk->...ik", mat.conj(), mat)
    assert jnp.allclose(product, jnp.eye(d_target, dtype=complex), atol=1e-10)

    # QuTiP comparison
    if ensemble_size == ():
        qt_u = u._to_qobj()
        qt_promoted = _qt_embed_operator(qt_u, d_in, d_target)
        assert np.allclose(np.array(promoted.matrix), qt_promoted.full(), atol=1e-12)


# ======================== Operator ========================


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2, 2), (3, 3)),
    ],
)
def test_promote_operator(seed, current_dims, target_dims):
    """Promote a generic Operator (zero-padded, not identity-padded)."""
    key = jax.random.key(seed)
    from functools import reduce
    from operator import mul

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, target_dims, 1)

    # Create a random operator matrix
    mat = jax.random.normal(key, (d_in, d_in)) + 1j * jax.random.normal(jax.random.fold_in(key, 1), (d_in, d_in))
    op = Operator.from_matrix(mat, (current_dims, current_dims))
    promoted = promote(op, target_dims)

    assert promoted.dims == (target_dims, target_dims)
    assert promoted.matrix.shape == (d_target, d_target)
    assert jnp.allclose(promoted.matrix[:d_in, :d_in], op.matrix, atol=1e-12)
    assert jnp.allclose(promoted.matrix[d_in:, :], 0.0, atol=1e-12)
    assert jnp.allclose(promoted.matrix[:d_in, d_in:], 0.0, atol=1e-12)


# ======================== SuperOp ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_promote_superop(seed, current_dims, target_dims, ensemble_size):
    """Promoted SuperOp acts as identity on higher states."""
    key = jax.random.key(seed)
    dims = (current_dims, current_dims)
    from functools import reduce
    from operator import mul

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, target_dims, 1)

    choi = random_choi(dims=dims, rank=d_in, key=key, size=ensemble_size)
    superop = choi_to_superop(choi)
    promoted = promote(superop, target_dims)

    assert promoted.dims == (target_dims, target_dims)
    mat = promoted.matrix
    assert mat.shape == ensemble_size + (d_target**2, d_target**2)

    # Qubit-subspace entries match original (column-stacking indices)
    qubit_idxs = np.array([j * d_target + i for j in range(d_in) for i in range(d_in)])
    assert jnp.allclose(mat[..., qubit_idxs[:, None], qubit_idxs[None, :]], superop.matrix, atol=1e-10)
    # Complement-only subspace is identity
    comp_idxs = np.array([j * d_target + i for j in range(d_in, d_target) for i in range(d_in, d_target)])
    assert jnp.allclose(
        mat[..., comp_idxs[:, None], comp_idxs[None, :]],
        jnp.eye(len(comp_idxs), dtype=complex),
        atol=1e-10,
    )

    # QuTiP comparison (scalar only)
    if ensemble_size == ():
        qt_choi = choi._to_qobj()
        qt_super = qt.to_super(qt_choi)
        qt_promoted = _qt_embed_superop(qt_super, d_in, d_target)
        assert np.allclose(np.array(promoted.matrix), qt_promoted.full(), atol=1e-10)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
    ],
)
def test_promote_superop_channel_action(seed, current_dims, target_dims):
    """Verify the promoted channel acts correctly on states in both subspaces."""
    key = jax.random.key(seed)
    from functools import reduce
    from operator import mul

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, target_dims, 1)

    dims = (current_dims, current_dims)
    choi = random_choi(dims=dims, rank=d_in, key=key)
    superop = choi_to_superop(choi)
    promoted = promote(superop, target_dims)

    # Apply to |0><0| — should match original channel
    rho_0 = np.zeros((d_target, d_target), dtype=complex)
    rho_0[0, 0] = 1.0
    rho_0_dm = DensityMatrix.from_matrix(jnp.array(rho_0), target_dims)
    result = promoted @ rho_0_dm

    rho_0_small = np.zeros((d_in, d_in), dtype=complex)
    rho_0_small[0, 0] = 1.0
    rho_0_small_dm = DensityMatrix.from_matrix(jnp.array(rho_0_small), current_dims)
    result_orig = superop @ rho_0_small_dm

    # Result in qubit subspace should match
    assert jnp.allclose(result.matrix[:d_in, :d_in], result_orig.matrix, atol=1e-10)

    # Apply to |d_target-1><d_target-1| — should be identity (unchanged)
    rho_high = np.zeros((d_target, d_target), dtype=complex)
    rho_high[-1, -1] = 1.0
    rho_high_dm = DensityMatrix.from_matrix(jnp.array(rho_high), target_dims)
    result_high = promoted @ rho_high_dm
    assert jnp.allclose(result_high.matrix, rho_high, atol=1e-10)


# ======================== KrausMap ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
    ],
)
def test_promote_kraus_map(seed, current_dims, target_dims):
    """Promoted KrausMap should be equivalent to promoted SuperOp."""
    key = jax.random.key(seed)
    from functools import reduce
    from operator import mul

    d_in = reduce(mul, current_dims, 1)

    dims = (current_dims, current_dims)
    choi = random_choi(dims=dims, rank=d_in, key=key)
    from quax import choi_to_kraus

    kraus = choi_to_kraus(choi)
    promoted_kraus = promote(kraus, target_dims)

    assert promoted_kraus.dims == (target_dims, target_dims)
    # n_kraus should be original + 1 (for the complement identity)
    orig_n_kraus = kraus.matrix.shape[-3]
    assert promoted_kraus.matrix.shape[-3] == orig_n_kraus + 1

    # Compare channel action via SuperOp
    promoted_superop_from_kraus = kraus_to_superop(promoted_kraus)
    superop = choi_to_superop(choi)
    promoted_superop_direct = promote(superop, target_dims)

    assert jnp.allclose(
        promoted_superop_from_kraus.matrix,
        promoted_superop_direct.matrix,
        atol=1e-10,
    )


# ======================== Choi ========================


@pytest.mark.parametrize("seed", [42, 999])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
        ((2,), (4,)),
    ],
)
@pytest.mark.parametrize("ensemble_size", [(), (3,)])
def test_promote_choi(seed, current_dims, target_dims, ensemble_size):
    """Promoted Choi should match promoted SuperOp round-tripped through Choi."""
    key = jax.random.key(seed)
    from functools import reduce
    from operator import mul

    d_in = reduce(mul, current_dims, 1)
    d_target = reduce(mul, target_dims, 1)

    dims = (current_dims, current_dims)
    choi = random_choi(dims=dims, rank=d_in, key=key, size=ensemble_size)
    promoted_choi = promote(choi, target_dims)

    assert promoted_choi.dims == (target_dims, target_dims)
    assert promoted_choi.matrix.shape == ensemble_size + (d_target**2, d_target**2)

    # Round-trip: promote SuperOp -> convert to Choi, compare
    superop = choi_to_superop(choi)
    promoted_superop = promote(superop, target_dims)
    choi_from_superop = superop_to_choi(promoted_superop)

    assert jnp.allclose(promoted_choi.matrix, choi_from_superop.matrix, atol=1e-10)

    # QuTiP comparison (scalar)
    if ensemble_size == ():
        qt_choi = choi._to_qobj()
        qt_super = qt.to_super(qt_choi)
        qt_promoted_super = _qt_embed_superop(qt_super, d_in, d_target)
        qt_promoted_choi = qt.to_choi(qt_promoted_super)
        assert np.allclose(np.array(promoted_choi.matrix), qt_promoted_choi.full(), atol=1e-10)


# ======================== PauliLiouville ========================


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "current_dims,target_dims",
    [
        ((2,), (3,)),
    ],
)
def test_promote_pauli_liouville(seed, current_dims, target_dims):
    """Promoted PauliLiouville should match promoted SuperOp round-tripped."""
    key = jax.random.key(seed)
    from functools import reduce
    from operator import mul

    d_in = reduce(mul, current_dims, 1)

    dims = (current_dims, current_dims)
    choi = random_choi(dims=dims, rank=d_in, key=key)
    superop = choi_to_superop(choi)
    pl = superop_to_pauli_liouville(superop)

    promoted_pl = promote(pl, target_dims)
    assert promoted_pl.dims == (target_dims, target_dims)

    # Compare via SuperOp round-trip
    promoted_superop = promote(superop, target_dims)
    pl_from_superop = superop_to_pauli_liouville(promoted_superop)

    assert jnp.allclose(promoted_pl.matrix, pl_from_superop.matrix, atol=1e-10)


# ======================== Validation ========================


def test_promote_wrong_num_subsystems():
    """Promoting with mismatched subsystem count should raise ValueError."""
    key = jax.random.key(0)
    sv = random_state_vector(dims=(2,), key=key)
    with pytest.raises(ValueError, match="Number of subsystems must match"):
        promote(sv, (3, 3))


def test_promote_smaller_dim_raises():
    """Promoting to a smaller dimension should raise ValueError."""
    key = jax.random.key(0)
    u = random_unitary(dims=((3,), (3,)), key=key)
    with pytest.raises(ValueError, match="smaller than current dimension"):
        promote(u, (2,))


def test_promote_unsupported_type():
    """Promoting an unsupported type should raise TypeError."""
    with pytest.raises(TypeError, match="promote is not implemented"):
        promote("not a quantum object", (3,))


# ======================== JIT compatibility ========================


def test_promote_jit_unitary():
    """Verify promote works under jax.jit."""
    key = jax.random.key(42)
    u = random_unitary(dims=((2,), (2,)), key=key)

    @jax.jit
    def do_promote(u):
        return promote(u, (3,))

    promoted = do_promote(u)
    assert promoted.dims == ((3,), (3,))
    assert promoted.matrix.shape == (3, 3)


def test_promote_jit_state_vector():
    """Verify promote of StateVector works under jax.jit."""
    key = jax.random.key(42)
    sv = random_state_vector(dims=(2,), key=key)

    @jax.jit
    def do_promote(sv):
        return promote(sv, (3,))

    promoted = do_promote(sv)
    assert promoted.dims == (3,)
    assert promoted.matrix.shape == (3,)
