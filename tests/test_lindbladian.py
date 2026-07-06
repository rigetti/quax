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

"""Tests for Lindbladian generators and the evolve() function.

Tests verify:
- Lindbladian construction from Hamiltonians and jump operators
- evolve() producing valid CPTP channels for t > 0
- Parity with QuTiP's liouvillian / expm (standard channels and random Lindbladians)
- Equivalence of common Lindbladian factories with Kraus-based channel factories
- JIT compilation and autodiff
- Ensemble broadcasting
- Generator algebra (__add__, __mul__, __or__)
- exact recovery of the stored Hamiltonian and jump operators via .hamiltonian/.jump_operators
- operators-only storage: introspection, cached generator, pytree/jit/grad, unsupported non-CP ops
- promote() dispatch for Lindbladian
- leakage and seepage generators
- Random Lindbladians: CPTP evolution and QuTiP parity
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qutip as qt

import quax as qx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUBIT_DIMS = ((2,), (2,))
T = 0.5  # default evolution time used across tests


def _qt_to_superop(qt_superop: "qt.Qobj", dims=_QUBIT_DIMS) -> qx.SuperOp:
    """Convert a QuTiP superoperator (type='super') to a quax SuperOp."""
    return qx.SuperOp.from_matrix(jnp.asarray(qt_superop.full(), dtype=complex), dims)


def _evolve_qutip(H_qt, c_ops, t, dims=_QUBIT_DIMS):
    """Evolve a QuTiP Liouvillian and return the resulting quax SuperOp."""
    L = qt.liouvillian(H_qt, c_ops)  # type: ignore[call-overload]
    E = (t * L).expm()
    return _qt_to_superop(qt.to_super(E), dims)


# Random-Lindbladian test cases: (qudit dim d, number of jump operators, rng seed).
_RANDOM_CASES = [(2, 1, 0), (2, 3, 1), (3, 2, 2), (3, 4, 3)]


def _random_hamiltonian_and_jumps(d, n_jumps, seed):
    """Random Hermitian ``H`` and ``n_jumps`` random complex ``d×d`` jump operators (numpy)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (A + A.conj().T) / 2
    jumps = [rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)) for _ in range(n_jumps)]
    return H, jumps


def _random_quax_lindbladian(d, n_jumps, seed):
    """Build a quax Lindbladian from a random (H, jump operators) pair."""
    H, jumps = _random_hamiltonian_and_jumps(d, n_jumps, seed)
    H_obj = qx.Observable.from_matrix(jnp.asarray(H, dtype=complex), ((d,), (d,)))
    jump_ops = qx.Operator.from_matrix(jnp.asarray(np.stack(jumps), dtype=complex), ((d,), (d,)))
    return qx.Lindbladian(hamiltonian=H_obj, jump_operators=jump_ops), H, jumps


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_lindbladian_purely_dissipative():
    """Lindbladian(hamiltonian=None, jump_operators=L) should equal a single dissipator."""
    L = jnp.sqrt(0.1) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    jump_ops = qx.Operator.from_matrix(L[jnp.newaxis], ((2,), (2,)))
    gen = qx.Lindbladian(hamiltonian=None, jump_operators=jump_ops)
    assert isinstance(gen, qx.Lindbladian)
    assert gen.dims == _QUBIT_DIMS


def test_lindbladian_two_jump_ops_additive():
    """Combined jump stack == sum of single-jump Lindbladians."""
    L1 = jnp.sqrt(0.1) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    L2 = jnp.sqrt(0.05) * jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    L_stack = jnp.stack([L1, L2])
    jump_ops_combined = qx.Operator.from_matrix(L_stack, ((2,), (2,)))
    gen_combined = qx.Lindbladian(hamiltonian=None, jump_operators=jump_ops_combined)

    jump_ops1 = qx.Operator.from_matrix(L1[jnp.newaxis], ((2,), (2,)))
    jump_ops2 = qx.Operator.from_matrix(L2[jnp.newaxis], ((2,), (2,)))
    gen_sum = qx.Lindbladian(hamiltonian=None, jump_operators=jump_ops1) + qx.Lindbladian(
        hamiltonian=None, jump_operators=jump_ops2
    )

    assert jnp.allclose(gen_combined.matrix, gen_sum.matrix, atol=1e-10)


def test_lindbladian_with_hamiltonian():
    """A Lindbladian with H and jump operators includes both coherent and dissipative terms."""
    from quax.gates import Z

    H = qx.Observable.from_matrix(0.5 * Z.matrix, ((2,), (2,)))
    L = jnp.sqrt(0.1) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    jump_ops = qx.Operator.from_matrix(L[jnp.newaxis], ((2,), (2,)))

    gen_full = qx.Lindbladian(hamiltonian=H, jump_operators=jump_ops)
    gen_no_H = qx.Lindbladian(hamiltonian=None, jump_operators=jump_ops)

    # The generators should differ by exactly the coherent term
    assert not jnp.allclose(gen_full.matrix, gen_no_H.matrix, atol=1e-10)


def test_lindbladian_generator_algebra():
    """L1 + L2 and 2.0 * L produce correct generators."""
    L = qx.lindbladians.amplitude_damping(0.1)
    doubled = 2.0 * L
    summed = L + L
    assert jnp.allclose(doubled.matrix, summed.matrix, atol=1e-12)


# ---------------------------------------------------------------------------
# __or__: tensor product (Kronecker sum)
# ---------------------------------------------------------------------------


def test_lindbladian_or_two_qubit():
    """L_A | L_B gives the combined generator for a two-qubit system."""
    L_A = qx.lindbladians.amplitude_damping(0.1)
    L_B = qx.lindbladians.dephasing(0.2)
    L_AB = L_A | L_B
    assert isinstance(L_AB, qx.Lindbladian)
    assert L_AB.dims == ((2, 2), (2, 2))


def test_lindbladian_or_channel_is_cptp():
    """evolve(L_A | L_B, t) is CPTP."""
    L_A = qx.lindbladians.amplitude_damping(0.1)
    L_B = qx.lindbladians.dephasing(0.2)
    channel = qx.evolve(L_A | L_B, 0.5)
    assert qx.is_cptp(channel)


def test_lindbladian_or_factorizes():
    """evolve(L_A | L_B, t) ≈ evolve(L_A, t) ⊗ evolve(L_B, t) for independent subsystems."""
    L_A = qx.lindbladians.amplitude_damping(0.1)
    L_B = qx.lindbladians.dephasing(0.2)
    t = 0.5

    ch_AB = qx.evolve(L_A | L_B, t)
    ch_A = qx.evolve(L_A, t)
    ch_B = qx.evolve(L_B, t)
    ch_tensor = ch_A | ch_B  # tensor product of SuperOps

    assert jnp.allclose(ch_AB.matrix, ch_tensor.matrix, atol=1e-8)


# ---------------------------------------------------------------------------
# promote: Lindbladian → larger Hilbert space
# ---------------------------------------------------------------------------


def test_promote_lindbladian_qubit_to_qutrit():
    """promote(L, (3,)) embeds a qubit Lindbladian in qutrit space."""
    L = qx.lindbladians.amplitude_damping(0.3)
    L_promoted = qx.promote(L, (3,))
    assert isinstance(L_promoted, qx.Lindbladian)
    assert L_promoted.dims == ((3,), (3,))


def test_promote_lindbladian_subspace_generator_matches():
    """promote(L_qubit, (3,)) reproduces the original qubit generator on the qubit sub-block.

    In quax's qutrit superoperator convention, qubit indices {0,1} map to
    qutrit rows/cols {0, 1, 3, 4} (bra*3+ket for bra,ket ∈ {0,1}).  Promotion reconstructs
    and re-embeds the operators, so the sub-block matches to float32 reconstruction precision.
    """
    L = qx.lindbladians.amplitude_damping(0.3)
    L_promoted = qx.promote(L, (3,))

    # Qubit subspace rows/cols in the 9x9 matrix: bra*3+ket for bra,ket in {0,1}
    qubit_idx = jnp.array([0, 1, 3, 4])
    subblock = L_promoted.matrix[jnp.ix_(qubit_idx, qubit_idx)]
    assert jnp.allclose(subblock, L.matrix, atol=1e-3)


def test_promote_lindbladian_matches_native_qutrit():
    """promote(L_qubit, (3,)) equals the Lindbladian built natively from embedded operators.

    promote reconstructs (H, jump operators) from the generator, zero-pads them, and rebuilds.
    For amplitude damping that must equal building √γ|0⟩⟨1| directly as a 3×3 jump operator —
    which correctly *damps* the |1⟩↔|2⟩ coherence (at rate γ/2 via the -½{L†L, ρ} term)
    rather than freezing it as a naive zero-pad of the generator would.
    """
    gamma = 0.3
    L_promoted = qx.promote(qx.lindbladians.amplitude_damping(gamma), (3,))

    # Native qutrit generator: the same jump operator √γ|0⟩⟨1|, embedded as a 3×3 operator.
    sigma_minus_3 = jnp.zeros((3, 3), dtype=complex).at[0, 1].set(1.0)
    jump_ops = qx.Operator.from_matrix((jnp.sqrt(gamma) * sigma_minus_3)[jnp.newaxis], ((3,), (3,)))
    L_native = qx.Lindbladian(hamiltonian=None, jump_operators=jump_ops)

    assert jnp.allclose(L_promoted.matrix, L_native.matrix, atol=1e-3)

    # The |1⟩↔|2⟩ coherence generator entry (index 1*3+2 = 5) is damped at -γ/2, not frozen.
    coh_idx = 1 * 3 + 2
    assert jnp.isclose(L_promoted.matrix[coh_idx, coh_idx], -gamma / 2.0, atol=1e-3)


@pytest.mark.parametrize(
    "factory",
    [
        qx.lindbladians.amplitude_damping,
        qx.lindbladians.dephasing,
        qx.lindbladians.depolarizing,
    ],
)
def test_promote_lindbladian_is_cptp(factory):
    """evolve(promote(L, (3,)), t) is a valid CPTP channel — the point of operator-level promotion.

    A naive zero-pad of the generator would be non-CP (frozen cross-subspace coherences give
    a negative Choi eigenvalue); reconstructing and re-embedding the operators keeps it valid.
    """
    L_promoted = qx.promote(factory(0.3), (3,))
    channel = qx.evolve(L_promoted, 0.5)

    choi = qx.superop_to_choi(channel).matrix
    choi = (choi + choi.conj().T) / 2
    min_eig = jnp.linalg.eigvalsh(choi).min()
    assert min_eig > -1e-4, f"{factory.__name__}: Choi min eigenvalue {min_eig} indicates non-CP"


# ---------------------------------------------------------------------------
# Leakage and seepage Lindbladians
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.1, 0.3, 1.0])
def test_leakage_lindbladian_cptp(gamma):
    """evolve(leakage_lindbladian(gamma), t) is CPTP (qutrit channel)."""
    L = qx.lindbladians.leakage(gamma)
    assert isinstance(L, qx.Lindbladian)
    assert L.dims == ((3,), (3,))
    channel = qx.evolve(L, 0.5)
    assert qx.is_cptp(channel)


@pytest.mark.parametrize("gamma", [0.1, 0.3, 1.0])
def test_seepage_lindbladian_cptp(gamma):
    """evolve(seepage_lindbladian(gamma), t) is CPTP (qutrit channel)."""
    L = qx.lindbladians.seepage(gamma)
    assert isinstance(L, qx.Lindbladian)
    assert L.dims == ((3,), (3,))
    channel = qx.evolve(L, 0.5)
    assert qx.is_cptp(channel)


def test_leakage_seepage_combined():
    """leakage + seepage Lindbladians combine to produce a CPTP channel."""
    L_leak = qx.lindbladians.leakage(0.1)
    L_seep = qx.lindbladians.seepage(0.1)
    L_total = L_leak + L_seep
    channel = qx.evolve(L_total, 0.5)
    assert qx.is_cptp(channel)


def test_leakage_lindbladian_drains_level1():
    """leakage_lindbladian transfers population from |1⟩ to |2⟩."""
    gamma = 2.0  # large rate so effect is visible at t=0.5
    L = qx.lindbladians.leakage(gamma)
    channel = qx.evolve(L, 0.5)

    # Start in |1⟩⟨1|
    rho1 = jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex)
    rho_vec = rho1.reshape(-1)
    evolved = (channel.matrix @ rho_vec).reshape(3, 3)
    # |1⟩ population decreases
    assert float(jnp.real(evolved[1, 1])) < 0.99


def test_seepage_lindbladian_drains_level2():
    """seepage_lindbladian transfers population from |2⟩ to |1⟩."""
    gamma = 2.0
    L = qx.lindbladians.seepage(gamma)
    channel = qx.evolve(L, 0.5)

    # Start in |2⟩⟨2|
    rho2 = jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=complex)
    rho_vec = rho2.reshape(-1)
    evolved = (channel.matrix @ rho_vec).reshape(3, 3)
    # |2⟩ population decreases
    assert float(jnp.real(evolved[2, 2])) < 0.99


# ---------------------------------------------------------------------------
# evolve: CPTP validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: qx.lindbladians.amplitude_damping(0.5),
        lambda: qx.lindbladians.dephasing(0.3),
        lambda: qx.lindbladians.depolarizing(0.4),
        lambda: qx.lindbladians.thermal_relaxation(t1=1.0, tphi=2.0),
        lambda: qx.lindbladians.bit_flip(0.2),
        lambda: qx.lindbladians.phase_flip(0.2),
    ],
    ids=["amplitude_damping", "dephasing", "depolarizing", "thermal_relaxation", "bit_flip", "phase_flip"],
)
@pytest.mark.parametrize("t", [0.1, 0.5, 1.0])
def test_common_factories_cptp(factory, t):
    """All common Lindbladian factories produce CPTP channels for t > 0."""
    gen = factory()
    channel = qx.evolve(gen, t)
    assert isinstance(channel, qx.SuperOp)
    assert qx.is_cptp(channel), f"Channel is not CPTP for t={t}"


# ---------------------------------------------------------------------------
# QuTiP parity
# ---------------------------------------------------------------------------


def test_qutip_parity_amplitude_damping():
    """evolve(amplitude_damping_lindbladian(γ), t) matches QuTiP."""
    gamma = 0.3
    sigma_minus = qt.Qobj([[0, 1], [0, 0]])
    c_ops = [math.sqrt(gamma) * sigma_minus]
    qt_channel = _evolve_qutip(None, c_ops, T)

    qx_channel = qx.evolve(qx.lindbladians.amplitude_damping(gamma), T)

    fid = qx.process_fidelity(qx_channel, qt_channel)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999"


def test_qutip_parity_dephasing():
    """evolve(dephasing_lindbladian(γ), t) matches QuTiP."""
    gamma = 0.2
    Z = qt.sigmaz()
    c_ops = [math.sqrt(gamma / 2.0) * Z]
    qt_channel = _evolve_qutip(None, c_ops, T)

    qx_channel = qx.evolve(qx.lindbladians.dephasing(gamma), T)

    fid = qx.process_fidelity(qx_channel, qt_channel)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999"


def test_qutip_parity_depolarizing():
    """evolve(depolarizing_lindbladian(γ), t) matches QuTiP."""
    gamma = 0.15
    X, Y, Z = qt.sigmax(), qt.sigmay(), qt.sigmaz()
    scale = math.sqrt(gamma / 3.0)
    c_ops = [scale * X, scale * Y, scale * Z]
    qt_channel = _evolve_qutip(None, c_ops, T)

    qx_channel = qx.evolve(qx.lindbladians.depolarizing(gamma), T)

    fid = qx.process_fidelity(qx_channel, qt_channel)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999"


def test_qutip_parity_with_hamiltonian():
    """evolve(Lindbladian(H, L_ops), t) matches QuTiP liouvillian."""
    from quax.gates import Z as Z_gate

    gamma = 0.2
    omega = 0.5

    H_qt = omega * qt.sigmaz()
    sigma_minus = qt.Qobj([[0, 1], [0, 0]])
    c_ops = [math.sqrt(gamma) * sigma_minus]
    qt_channel = _evolve_qutip(H_qt, c_ops, T)

    H_qx = qx.Observable.from_matrix(omega * Z_gate.matrix, ((2,), (2,)))
    L = jnp.sqrt(gamma) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    jump_ops = qx.Operator.from_matrix(L[jnp.newaxis], ((2,), (2,)))
    qx_channel = qx.evolve(qx.Lindbladian(hamiltonian=H_qx, jump_operators=jump_ops), T)

    fid = qx.process_fidelity(qx_channel, qt_channel)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999"


# ---------------------------------------------------------------------------
# Hamiltonian consistency
# ---------------------------------------------------------------------------


def test_hamiltonian_evolve_matches_cis():
    """evolve(H, t) = exp(-iHt) matches cis(-t * H) for an observable."""
    from quax.gates import X

    H = qx.Observable.from_matrix(0.3 * X.matrix, ((2,), (2,)))
    t = 0.7

    via_evolve = qx.evolve(H, t)
    via_cis = qx.cis(qx.Observable.from_matrix(-t * H.matrix, ((2,), (2,))))

    assert jnp.allclose(via_evolve.matrix, via_cis.matrix, atol=1e-10)


def test_evolve_observable_is_literal_propagator():
    """evolve(H, t) returns exp(-i·t·H) exactly, with no global-phase re-normalization.

    Uses H = X, t = π/2, where exp(-iπX/2) = -iX has a zero (0,0) entry — a [0,0]-anchored
    phase correction (as in cis) would silently no-op and leave a spurious phase, so evolve
    must not apply one.
    """
    from quax.gates import X

    H = qx.Observable.from_matrix(X.matrix, ((2,), (2,)))
    U = qx.evolve(H, jnp.pi / 2)

    expected = -1j * X.matrix  # exp(-iπX/2) = cos(π/2) I - i sin(π/2) X = -i X
    assert jnp.allclose(U.matrix, expected, atol=1e-6)
    assert jnp.abs(U.matrix[0, 0]) < 1e-6  # the entry a [0,0]-anchored correction would key off


def test_purely_hamiltonian_lindbladian_matches_unitary():
    """evolve(Lindbladian(H, zero_jumps), t) ≈ unitary_to_superop(U(t)).

    The GKSL equation uses -i[H,ρ], giving evolution exp(-iHt)ρexp(iHt).
    In quax's convention evolve(H, t) = exp(-iHt), so the matching unitary is evolve(H, t).
    """
    from quax.gates import Z

    omega = 0.5
    H = qx.Observable.from_matrix(omega * Z.matrix, ((2,), (2,)))

    # Lindbladian with zero dissipation — single zero jump operator
    zero_L = jnp.zeros((1, 2, 2), dtype=complex)
    jump_ops = qx.Operator.from_matrix(zero_L, ((2,), (2,)))
    gen = qx.Lindbladian(hamiltonian=H, jump_operators=jump_ops)
    channel_via_lindbladian = qx.evolve(gen, T)

    # GKSL gives exp(-iHT), matching evolve(H, T) = exp(-i*H*T)
    unitary = qx.evolve(H, T)
    channel_via_unitary = qx.unitary_to_superop(unitary)

    fid = qx.process_fidelity(channel_via_lindbladian, channel_via_unitary)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999"


# ---------------------------------------------------------------------------
# JIT compilation
# ---------------------------------------------------------------------------


def test_evolve_lindbladian_jit():
    """jax.jit(qx.evolve) compiles and runs for Lindbladian input."""
    gen = qx.lindbladians.amplitude_damping(0.3)
    channel = jax.jit(qx.evolve)(gen, 0.5)
    assert isinstance(channel, qx.SuperOp)
    assert qx.is_cptp(channel)


def test_evolve_observable_jit():
    """jax.jit(qx.evolve) compiles and runs for Observable input."""
    from quax.gates import Z

    H = qx.Observable.from_matrix(0.5 * Z.matrix, ((2,), (2,)))
    U = jax.jit(qx.evolve)(H, 0.5)
    assert isinstance(U, qx.Unitary)


def test_lindbladian_constructor_jit():
    """jax.jit over the Lindbladian constructor compiles and runs."""
    L = jnp.sqrt(0.1) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    jump_ops = qx.Operator.from_matrix(L[jnp.newaxis], ((2,), (2,)))
    gen = jax.jit(lambda h, j: qx.Lindbladian(hamiltonian=h, jump_operators=j))(None, jump_ops)
    assert isinstance(gen, qx.Lindbladian)


def test_lindbladian_factory_jit():
    """The generator factories compile under jax.jit (single- and multi-jump)."""
    gen = jax.jit(qx.lindbladians.amplitude_damping)(0.1)
    assert isinstance(gen, qx.Lindbladian)
    assert gen.ensemble_size == ()

    dep = jax.jit(qx.lindbladians.depolarizing)(0.2)
    assert isinstance(dep, qx.Lindbladian)
    assert dep.ensemble_size == ()


# ---------------------------------------------------------------------------
# Autodiff
# ---------------------------------------------------------------------------


def test_evolve_grad_through_t():
    """jax.grad of Hilbert-Schmidt overlap through evolve w.r.t. t returns finite value."""
    target = qx.evolve(qx.lindbladians.amplitude_damping(0.3), 0.5)
    gen = qx.lindbladians.amplitude_damping(0.3)

    def loss(t):
        channel = qx.evolve(gen, t)
        # Hilbert-Schmidt overlap: avoids sqrt(0) in matrix Jozsa fidelity
        return jnp.real(jnp.sum(jnp.conj(target.matrix) * channel.matrix))

    grad = jax.grad(loss)(jnp.array(0.4))
    assert jnp.isfinite(grad), f"Gradient is not finite: {grad}"


def test_evolve_grad_through_rate():
    """jax.grad of Hilbert-Schmidt overlap through evolve w.r.t. rate returns finite value."""
    target = qx.evolve(qx.lindbladians.amplitude_damping(0.3), 0.5)

    def loss(gamma):
        gen = qx.lindbladians.amplitude_damping(gamma)
        channel = qx.evolve(gen, 0.5)
        return jnp.real(jnp.sum(jnp.conj(target.matrix) * channel.matrix))

    grad = jax.grad(loss)(jnp.array(0.3))
    assert jnp.isfinite(grad), f"Gradient is not finite: {grad}"


# ---------------------------------------------------------------------------
# Ensemble broadcasting
# ---------------------------------------------------------------------------


def test_lindbladian_ensemble_jump_operators():
    """Batched jump operators produce an ensemble of generators."""
    n_batch = 4
    gammas = jnp.linspace(0.1, 0.5, n_batch)
    sigma_minus = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

    L_batch = jnp.sqrt(gammas)[:, None, None] * sigma_minus[None]  # (n_batch, 2, 2)
    L_with_ops_dim = L_batch[:, jnp.newaxis, :, :]  # (n_batch, 1, 2, 2)
    jump_ops = qx.Operator.from_matrix(L_with_ops_dim, ((2,), (2,)))
    gen = qx.Lindbladian(hamiltonian=None, jump_operators=jump_ops)

    assert gen.ensemble_size == (n_batch,)


def test_evolve_ensemble_lindbladian():
    """evolve over an ensemble Lindbladian produces an ensemble of channels."""
    n_batch = 3
    gammas = jnp.array([0.1, 0.2, 0.3])
    sigma_minus = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

    L_batch = jnp.sqrt(gammas)[:, None, None] * sigma_minus[None]
    L_with_ops_dim = L_batch[:, jnp.newaxis, :, :]
    jump_ops = qx.Operator.from_matrix(L_with_ops_dim, ((2,), (2,)))
    gen = qx.Lindbladian(hamiltonian=None, jump_operators=jump_ops)

    channels = qx.evolve(gen, 0.5)
    assert channels.ensemble_size == (n_batch,)

    for i in range(n_batch):
        assert qx.is_cptp(channels[i]), f"Channel {i} is not CPTP"


def test_lindbladian_factory_ensemble_single_jump():
    """A batched rate passed to a single-jump factory yields an ensemble of generators."""
    gammas = jnp.array([0.1, 0.2, 0.3])
    gen = qx.lindbladians.amplitude_damping(gammas)
    assert gen.ensemble_size == (3,)
    # each ensemble member equals the corresponding scalar-rate generator
    for i in range(gammas.shape[0]):
        expected = qx.lindbladians.amplitude_damping(gammas[i])
        assert jnp.allclose(gen.matrix[i], expected.matrix, atol=1e-12)


def test_lindbladian_factory_ensemble_multi_jump():
    """Batched rates broadcast through multi-jump factories (depolarizing, thermal_relaxation)."""
    gammas = jnp.array([0.1, 0.2, 0.3])
    dep = qx.lindbladians.depolarizing(gammas)
    assert dep.ensemble_size == (3,)

    t1s = jnp.array([10.0, 20.0])
    tphis = jnp.array([5.0, 8.0])
    therm = qx.lindbladians.thermal_relaxation(t1s, tphis)
    assert therm.ensemble_size == (2,)
    for i in range(t1s.shape[0]):
        expected = qx.lindbladians.thermal_relaxation(t1s[i], tphis[i])
        assert jnp.allclose(therm.matrix[i], expected.matrix, atol=1e-12)


def test_thermal_relaxation_finite_temperature():
    """Finite-temperature thermal relaxation adds an excitation jump and relaxes toward p1."""

    def equilibrium_excited(p1):
        # long-time evolution from |0><0| — steady-state excited-state population
        ch = qx.evolve(qx.lindbladians.thermal_relaxation(1.0, 10.0, p1=p1), 80.0)
        rho0 = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex).reshape(-1)
        return float((ch.matrix @ rho0).reshape(2, 2)[1, 1].real)

    # Zero temperature: relaxes to the ground state; three jump ops (down, up≈0, dephasing).
    assert qx.lindbladians.thermal_relaxation(1.0, 10.0).jump_operators.matrix.shape[-3] == 3
    assert equilibrium_excited(0.0) < 1e-6
    # Finite temperature: relaxes toward excited-state population p1.
    assert abs(equilibrium_excited(0.3) - 0.3) < 1e-3
    assert qx.is_cptp(qx.evolve(qx.lindbladians.thermal_relaxation(1.0, 10.0, p1=0.3), 1.0))


# ---------------------------------------------------------------------------
# Random Lindbladians: exact operator recovery, CPTP, QuTiP parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d, n_jumps, seed", _RANDOM_CASES)
def test_random_lindbladian_exact_operator_recovery(d, n_jumps, seed):
    """.hamiltonian/.jump_operators return the stored operators verbatim (no gauge canonicalization)."""
    gen, H, jumps = _random_quax_lindbladian(d, n_jumps, seed)
    H_r, jumps_r = gen.hamiltonian, gen.jump_operators

    # The exact operators supplied at construction come back — same count, same values.
    assert jumps_r.matrix.shape == (n_jumps, d, d)
    assert jnp.allclose(jumps_r.matrix, jnp.asarray(np.stack(jumps), dtype=complex))
    assert H_r is not None and jnp.allclose(H_r.matrix, jnp.asarray(H, dtype=complex))

    rebuilt = qx.Lindbladian(hamiltonian=H_r, jump_operators=jumps_r)
    assert jnp.allclose(rebuilt.matrix, gen.matrix, atol=1e-6)
    assert jnp.allclose(qx.evolve(rebuilt, T).matrix, qx.evolve(gen, T).matrix, atol=1e-6)


@pytest.mark.parametrize("d, n_jumps, seed", _RANDOM_CASES)
def test_random_lindbladian_evolve_is_cptp(d, n_jumps, seed):
    """evolve() of a random Lindbladian is a valid CPTP channel."""
    gen, _, _ = _random_quax_lindbladian(d, n_jumps, seed)
    assert qx.is_cptp(qx.evolve(gen, T))


@pytest.mark.parametrize("d, n_jumps, seed", _RANDOM_CASES)
def test_random_lindbladian_qutip_parity(d, n_jumps, seed):
    """evolve() of a random Lindbladian matches QuTiP's liouvillian/expm."""
    gen, H, jumps = _random_quax_lindbladian(d, n_jumps, seed)
    qx_channel = qx.evolve(gen, T)

    H_qt = qt.Qobj(np.asarray(H))
    c_ops = [qt.Qobj(np.asarray(L)) for L in jumps]
    qt_channel = _evolve_qutip(H_qt, c_ops, T, dims=((d,), (d,)))

    fid = qx.process_fidelity(qx_channel, qt_channel)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999 (d={d}, n_jumps={n_jumps})"


def test_stored_jump_operators_jit():
    """Reading .jump_operators compiles under jax.jit and returns the stored operators."""
    gen, _, _ = _random_quax_lindbladian(2, 2, 7)
    matrix = jax.jit(lambda g: g.jump_operators.matrix)(gen)
    assert matrix.shape == (2, 2, 2)


# ---------------------------------------------------------------------------
# Operators-only storage: introspection, caching, pytree, unsupported ops
# ---------------------------------------------------------------------------


def test_factory_exposes_physical_jump_operators():
    """A factory Lindbladian stores its physical jump operator (no gauge canonicalization)."""
    gamma = 0.1
    L = qx.lindbladians.amplitude_damping(gamma)
    assert L.hamiltonian is None
    assert L.jump_operators.matrix.shape == (1, 2, 2)
    expected = jnp.sqrt(gamma) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    assert jnp.allclose(L.jump_operators.matrix.squeeze(0), expected)


def test_stored_operators_are_the_construction_inputs():
    """The constructor stores the exact Observable and Operator it was given."""
    H = qx.Observable.from_matrix(0.5 * qx.gates.Z.matrix, _QUBIT_DIMS)
    L_mat = jnp.sqrt(0.1) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    jumps = qx.Operator.from_matrix(L_mat[jnp.newaxis], _QUBIT_DIMS)
    gen = qx.Lindbladian(hamiltonian=H, jump_operators=jumps)
    assert gen.hamiltonian is H
    assert gen.jump_operators is jumps
    H_r, jumps_r = gen.hamiltonian, gen.jump_operators
    assert H_r is not None
    assert jnp.allclose(H_r.matrix, H.matrix)
    assert jnp.allclose(jumps_r.matrix, jumps.matrix)


def test_matrix_is_cached():
    """The generator matrix is computed once and cached on the (frozen) instance."""
    gen, _, _ = _random_quax_lindbladian(2, 2, 5)
    assert "matrix" not in gen.__dict__
    first = gen.matrix
    assert "matrix" in gen.__dict__  # cached_property populated the instance dict
    assert gen.matrix is first  # same array object on second access


def test_lindbladian_is_pytree_over_operators():
    """A Lindbladian flattens to its Hamiltonian and jump operators (num_qubits derived)."""
    gen, _, _ = _random_quax_lindbladian(2, 2, 6)
    leaves, treedef = jax.tree_util.tree_flatten(gen)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.num_qubits == gen.num_qubits
    assert rebuilt.dims == gen.dims
    assert jnp.allclose(rebuilt.matrix, gen.matrix)


def test_grad_through_stored_operators():
    """Autodiff flows through the stored jump operators into the evolved channel."""

    def loss(gamma):
        jumps = qx.Operator.from_matrix(
            (jnp.sqrt(gamma) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex))[jnp.newaxis], _QUBIT_DIMS
        )
        return qx.evolve(qx.Lindbladian(hamiltonian=None, jump_operators=jumps), 0.5).matrix.real.sum()

    g = jax.grad(loss)(0.1)
    assert jnp.isfinite(g)


def test_negation_and_subtraction_unsupported():
    """__neg__ and __sub__ raise: they can produce non-CP generators."""
    L = qx.lindbladians.amplitude_damping(0.1)
    with pytest.raises(NotImplementedError):
        _ = -L
    with pytest.raises(NotImplementedError):
        _ = L - L


def test_complex_scalar_multiplication_unsupported():
    """Multiplying a Lindbladian by a complex scalar raises (non-CP generator)."""
    L = qx.lindbladians.amplitude_damping(0.1)
    with pytest.raises(NotImplementedError):
        _ = (1.0 + 1.0j) * L


# ---------------------------------------------------------------------------
# Gate ⊕ noise `+` API
# ---------------------------------------------------------------------------


def test_gate_plus_lindbladian_returns_superop():
    """`Unitary + Lindbladian` exponentiates to a CPTP SuperOp (gate on the left)."""
    channel = qx.gates.X + qx.lindbladians.dephasing(0.1)
    assert isinstance(channel, qx.SuperOp)
    assert channel.dims == ((2,), (2,))
    assert qx.is_cptp(channel)


def test_lindbladian_plus_gate_returns_lindbladian():
    """`Lindbladian + Unitary` folds the gate in as a coherent term, staying a Lindbladian."""
    gen = qx.lindbladians.dephasing(0.1) + qx.gates.X
    assert isinstance(gen, qx.Lindbladian)
    # Evolving the generator reproduces the gate-on-the-left channel.
    assert jnp.allclose(qx.evolve(gen, 1.0).matrix, (qx.gates.X + qx.lindbladians.dephasing(0.1)).matrix, atol=1e-6)


def test_gate_plus_noise_coherent_part_is_the_gate():
    """With zero noise rate, `gate + noise` reduces to the gate's unitary channel."""
    channel = qx.gates.RX(0.7) + qx.lindbladians.dephasing(0.0)
    assert jnp.allclose(channel.matrix, qx.unitary_to_superop(qx.gates.RX(0.7)).matrix, atol=1e-6)


def test_noisy_cz_leakage_is_cptp_on_qutrits():
    """CZ with per-qutrit leakage: gate promoted to (3,3), CPTP SuperOp."""
    noisy = qx.gates.CZ + (qx.lindbladians.leakage(0.01) | qx.lindbladians.leakage(0.01))
    assert isinstance(noisy, qx.SuperOp)
    assert noisy.dims == ((3, 3), (3, 3))
    assert qx.is_cptp(noisy)


def test_noisy_rx_leakage_seepage_depolarizing():
    """RX with leakage + seepage + qubit-subspace depolarizing → CPTP qutrit channel."""
    noise = qx.lindbladians.leakage(0.01) + qx.lindbladians.seepage(0.01) + qx.lindbladians.depolarizing(0.01, (2,))
    assert noise.dims == ((3,), (3,))  # depolarizing(2,) promoted to the qutrit space
    channel = qx.gates.RX(0.5) + noise
    assert isinstance(channel, qx.SuperOp)
    assert channel.dims == ((3,), (3,))
    assert qx.is_cptp(channel)


def test_gate_plus_lindbladian_subsystem_count_mismatch_raises():
    """A 2-qubit gate cannot combine with single-subsystem noise (tensor the noise instead)."""
    with pytest.raises(ValueError, match="subsystem counts differ"):
        _ = qx.gates.CZ + qx.lindbladians.dephasing(0.1)


def test_gate_plus_noise_differentiable():
    """Gradients flow through `gate + noise` w.r.t. both the gate angle and the noise rate."""
    grad_angle = jax.grad(lambda th: (qx.gates.RX(th) + qx.lindbladians.dephasing(0.1)).matrix.real.sum())(0.5)
    grad_rate = jax.grad(lambda g: (qx.gates.RX(0.5) + qx.lindbladians.dephasing(g)).matrix.real.sum())(0.1)
    assert jnp.isfinite(grad_angle)
    assert jnp.isfinite(grad_rate)


def test_depolarizing_dims_qubit_matches_pauli_generator():
    """depolarizing(gamma, (2,)) is the qubit {X,Y,Z} generator (unchanged default)."""
    gen = qx.lindbladians.depolarizing(0.3, (2,))
    assert gen.jump_operators.matrix.shape == (3, 2, 2)
    assert gen.dims == ((2,), (2,))


@pytest.mark.parametrize("dims, n_jumps", [((2,), 3), ((3,), 8), ((2, 2), 15)])
def test_depolarizing_dims_generalization(dims, n_jumps):
    """depolarizing generalizes to arbitrary dimension with D²−1 traceless jump operators."""
    gen = qx.lindbladians.depolarizing(0.2, dims)
    assert gen.jump_operators.matrix.shape[-3] == n_jumps
    assert qx.is_cptp(qx.evolve(gen, 1.0))
