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

"""Tests for Lindbladian generators, ``evolve()``, and the noise-channel factories.

A handful of tests, each parameterized over qudit dimensions (1 qubit, 1 qutrit, 2 qubits,
1 qubit + 1 qutrit) and ensemble shapes:

1. :func:`test_random_lindbladian_matches_qutip` — for random (Hamiltonian, jump-operator) pairs
   the generator matches QuTiP's liouvillian, ``evolve()`` is CPTP, and it matches QuTiP's
   exponentiated channel.
2. :func:`test_promote` — ``promote()`` equals rebuilding from operator-level zero-padded operators;
   checks the promoted dims and CPTP of the evolved channel.
3. :func:`test_lindbladian_factories` — every Lindbladian factory jits, grads, and broadcasts over
   an ensemble of rates.
4. :func:`test_channel_factories` — every channel (SuperOp) factory jits, grads, broadcasts, and
   matches evolving the corresponding Lindbladian factory for unit time.
5. :func:`test_addition` / :func:`test_tensor` — generator-algebra correctness, plus the
   non-CP-operation guards in :func:`test_non_cp_operations_unsupported`.

The ``Unitary`` ⊕ ``Lindbladian`` algebra (noisy gates) is covered in ``test_quantum_objects.py``.
"""

from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qutip as qt

import quax as qx

# ---------------------------------------------------------------------------
# Parameterization: qudit dimensions and ensemble (batch) shapes
# ---------------------------------------------------------------------------

# 1 qubit, 1 qutrit, 2 qubits, 1 qubit + 1 qutrit.
DIMS = [(2,), (3,), (2, 2), (2, 3)]
ENSEMBLES = [(1,), (4,), (4, 3)]
T = 0.5  # default evolution time used across tests


def _dims_id(dims):
    return "d" + "x".join(map(str, dims))


def _ens_id(ensemble):
    return "e" + "x".join(map(str, ensemble))


_DIMS_PARAMS = pytest.mark.parametrize("dims", DIMS, ids=[_dims_id(d) for d in DIMS])
_ENS_PARAMS = pytest.mark.parametrize("ensemble", ENSEMBLES, ids=[_ens_id(e) for e in ENSEMBLES])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_operators(dims, ensemble, n_jumps, salt=0):
    """Random Hermitian Hamiltonian and stacked jump operators (numpy).

    :returns: ``H`` of shape ``(*ensemble, d, d)`` and ``jumps`` of shape
        ``(*ensemble, n_jumps, d, d)`` where ``d = prod(dims)``.
    """
    d = reduce(mul, dims, 1)
    rng = np.random.default_rng(list(dims) + [salt] + list(ensemble) + [n_jumps])

    def _complex(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    A = _complex(ensemble + (d, d))
    H = (A + np.swapaxes(A.conj(), -1, -2)) / 2  # Hermitian
    jumps = _complex(ensemble + (n_jumps, d, d))
    return H, jumps


def _lindbladian(dims, H, jumps):
    """Build a quax :class:`~quax.Lindbladian` from numpy ``H`` and ``jumps``."""
    H_obj = qx.Observable.from_matrix(jnp.asarray(H, dtype=complex), (dims, dims))
    jump_ops = qx.Operator.from_matrix(jnp.asarray(jumps, dtype=complex), (dims, dims))
    return qx.Lindbladian(hamiltonian=H_obj, jump_operators=jump_ops)


def _qutip_liouvillian(dims, H_member, jumps_member):
    """QuTiP liouvillian Qobj for a single ensemble member."""
    qdims = [list(dims), list(dims)]
    H_qt = qt.Qobj(np.asarray(H_member), dims=qdims)
    c_ops = [qt.Qobj(np.asarray(L), dims=qdims) for L in jumps_member]
    return qt.liouvillian(H_qt, c_ops)  # type: ignore[call-overload]


# Lindbladian factories paired with their channel counterparts.  Each takes a single rate
# (scalar or array); factories with extra parameters have them fixed to sensible values.
_FACTORIES = [
    ("amplitude_damping", qx.lindbladians.amplitude_damping, qx.channels.amplitude_damping),
    ("dephasing", qx.lindbladians.dephasing, qx.channels.dephasing),
    ("depolarizing", qx.lindbladians.depolarizing, qx.channels.depolarizing),
    ("bit_flip", qx.lindbladians.bit_flip, qx.channels.bit_flip),
    ("phase_flip", qx.lindbladians.phase_flip, qx.channels.phase_flip),
    ("leakage", qx.lindbladians.leakage, qx.channels.leakage),
    ("seepage", qx.lindbladians.seepage, qx.channels.seepage),
    (
        "thermal_relaxation",
        lambda r: qx.lindbladians.thermal_relaxation(r, 2.0),
        lambda r: qx.channels.thermal_relaxation(r, 2.0),
    ),
]
_FACTORY_PARAMS = pytest.mark.parametrize(
    "lindblad_factory, channel_factory", [(lf, cf) for _, lf, cf in _FACTORIES], ids=[n for n, _, _ in _FACTORIES]
)


def _ensemble_rates(ensemble):
    """A spread of positive rates shaped like ``ensemble``."""
    n = reduce(mul, ensemble, 1)
    return (0.1 + 0.05 * jnp.arange(n, dtype=float)).reshape(ensemble)


# ---------------------------------------------------------------------------
# 1. Random Lindbladians vs QuTiP
# ---------------------------------------------------------------------------


@_DIMS_PARAMS
@_ENS_PARAMS
def test_random_lindbladian_matches_qutip(dims, ensemble):
    """Generator matches QuTiP's liouvillian; ``evolve()`` is CPTP and matches QuTiP's channel."""
    H, jumps = _random_operators(dims, ensemble, n_jumps=2)
    gen = _lindbladian(dims, H, jumps)

    assert gen.dims == (dims, dims)
    assert gen.ensemble_size == ensemble

    channel = qx.evolve(gen, T)
    assert isinstance(channel, qx.SuperOp)
    assert channel.dims == (dims, dims)
    assert channel.ensemble_size == ensemble

    gen_matrix = gen.matrix
    for idx in np.ndindex(ensemble):
        liouvillian = _qutip_liouvillian(dims, H[idx], jumps[idx])

        # (a) the full generator matrix matches QuTiP's liouvillian exactly (same convention).
        assert jnp.allclose(gen_matrix[idx], jnp.asarray(liouvillian.full()), atol=1e-9)

        # (b) the evolved channel is CPTP.
        member = channel[idx]
        assert qx.is_cptp(member)

        # (c) the evolved channel matches QuTiP's exponentiated liouvillian.
        qt_channel = qx.SuperOp.from_matrix(
            jnp.asarray(qt.to_super((T * liouvillian).expm()).full(), dtype=complex), (dims, dims)
        )
        fid = qx.process_fidelity(member, qt_channel)
        assert float(fid.real) > 0.9999, f"process fidelity {fid} < 0.9999 at {idx}"


# ---------------------------------------------------------------------------
# 2. promote
# ---------------------------------------------------------------------------


@_DIMS_PARAMS
@_ENS_PARAMS
def test_promote(dims, ensemble):
    """promote() equals rebuilding from zero-padded operators; check dims and CPTP of the channel."""
    H, jumps = _random_operators(dims, ensemble, n_jumps=2)
    gen = _lindbladian(dims, H, jumps)
    target = tuple(d + 1 for d in dims)  # enlarge every subsystem by one level

    promoted = qx.promote(gen, target)
    assert isinstance(promoted, qx.Lindbladian)
    assert promoted.dims == (target, target)
    assert promoted.ensemble_size == ensemble

    # Promotion acts at the operator level: it is equivalent to zero-padding the stored
    # Hamiltonian and jump operators into the larger space and rebuilding the generator.
    rebuilt = qx.Lindbladian(
        hamiltonian=qx.promote(gen.hamiltonian, target),
        jump_operators=qx.promote(gen.jump_operators, target),
    )
    assert jnp.allclose(promoted.matrix, rebuilt.matrix, atol=1e-8)

    # The evolved promoted generator is a CPTP channel on the larger space.
    channel = qx.evolve(promoted, T)
    assert channel.dims == (target, target)
    for idx in np.ndindex(ensemble):
        assert qx.is_cptp(channel[idx])


# ---------------------------------------------------------------------------
# 3. Lindbladian factories: jit, grad, ensemble
# ---------------------------------------------------------------------------


@_FACTORY_PARAMS
@_ENS_PARAMS
def test_lindbladian_factories(lindblad_factory, channel_factory, ensemble):
    """Every Lindbladian factory jits, grads w.r.t. its rate, and broadcasts over an ensemble."""
    # jit compiles and returns a single generator.
    gen = jax.jit(lindblad_factory)(0.3)
    assert isinstance(gen, qx.Lindbladian)
    assert gen.ensemble_size == ()

    # grad w.r.t. the rate is finite.
    grad = jax.grad(lambda r: lindblad_factory(r).matrix.real.sum())(0.3)
    assert jnp.isfinite(grad)

    # An array of rates yields an ensemble of generators, each equal to the scalar-rate generator.
    rates = _ensemble_rates(ensemble)
    ens_gen = lindblad_factory(rates)
    assert ens_gen.ensemble_size == ensemble
    flat = rates.reshape(-1)
    for i, idx in enumerate(np.ndindex(ensemble)):
        assert jnp.allclose(ens_gen.matrix[idx], lindblad_factory(flat[i]).matrix, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Channel factories: jit, grad, ensemble, and parity with evolve(lindbladian)
# ---------------------------------------------------------------------------


@_FACTORY_PARAMS
@_ENS_PARAMS
def test_channel_factories(lindblad_factory, channel_factory, ensemble):
    """Every channel factory jits, grads, broadcasts, and equals evolving its Lindbladian factory."""
    # jit compiles into a CPTP channel.
    channel = jax.jit(channel_factory)(0.3)
    assert isinstance(channel, qx.SuperOp)
    assert qx.is_cptp(channel)

    # grad w.r.t. the rate is finite.
    grad = jax.grad(lambda r: channel_factory(r).matrix.real.sum())(0.3)
    assert jnp.isfinite(grad)

    # An array of rates yields an ensemble of channels that matches evolving the corresponding
    # Lindbladian factory for unit time.
    rates = _ensemble_rates(ensemble)
    ens_channel = channel_factory(rates)
    assert ens_channel.ensemble_size == ensemble
    assert jnp.allclose(ens_channel.matrix, qx.evolve(lindblad_factory(rates), 1.0).matrix, atol=1e-8)


# ---------------------------------------------------------------------------
# 5. Generator algebra: addition and tensor product
# ---------------------------------------------------------------------------


@_DIMS_PARAMS
@_ENS_PARAMS
def test_addition(dims, ensemble):
    """L1 + L2 is a Lindbladian whose generator is the (linear) sum, and evolves to a CPTP channel."""
    L1 = _lindbladian(dims, *_random_operators(dims, ensemble, n_jumps=2, salt=0))
    L2 = _lindbladian(dims, *_random_operators(dims, ensemble, n_jumps=3, salt=1))

    total = L1 + L2
    assert isinstance(total, qx.Lindbladian)
    assert total.dims == (dims, dims)
    assert total.ensemble_size == ensemble

    # GKSL is linear in the Hamiltonian and dissipator, so the generators add.
    assert jnp.allclose(total.matrix, L1.matrix + L2.matrix, atol=1e-9)

    channel = qx.evolve(total, T)
    for idx in np.ndindex(ensemble):
        assert qx.is_cptp(channel[idx])


@_DIMS_PARAMS
def test_tensor(dims):
    """L_A | L_B is the Kronecker sum: evolve(L_A | L_B, t) == evolve(L_A, t) | evolve(L_B, t)."""
    A = _lindbladian(dims, *_random_operators(dims, (), n_jumps=2, salt=0))
    B = _lindbladian(dims, *_random_operators(dims, (), n_jumps=2, salt=1))

    AB = A | B
    assert isinstance(AB, qx.Lindbladian)
    assert AB.dims == (dims + dims, dims + dims)

    lhs = qx.evolve(AB, T)
    rhs = qx.evolve(A, T) | qx.evolve(B, T)  # tensor product of the single-system channels
    assert jnp.allclose(lhs.matrix, rhs.matrix, atol=1e-8)
    assert qx.is_cptp(lhs)

    # Tensor product of ensemble Lindbladians is not supported.
    ens = _lindbladian(dims, *_random_operators(dims, (4,), n_jumps=2))
    with pytest.raises(NotImplementedError):
        _ = ens | ens


def test_non_cp_operations_unsupported():
    """Operations that could yield a non-CP generator raise instead of returning an invalid object."""
    L = qx.lindbladians.amplitude_damping(0.1)
    with pytest.raises(NotImplementedError):
        _ = -L
    with pytest.raises(NotImplementedError):
        _ = L - L
    with pytest.raises(NotImplementedError):
        _ = (1.0 + 1.0j) * L
