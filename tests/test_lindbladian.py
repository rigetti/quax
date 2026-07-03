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
- to_operators() GKSL reconstruction (round-trip + gauge invariance)
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
    return qx.Lindbladian.from_operators(H_obj, jump_ops), H, jumps


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_lindbladian_purely_dissipative():
    """Lindbladian.from_operators(None, L) should equal a single dissipator."""
    L = jnp.sqrt(0.1) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    jump_ops = qx.Operator.from_matrix(L[jnp.newaxis], ((2,), (2,)))
    gen = qx.Lindbladian.from_operators(None, jump_ops)
    assert isinstance(gen, qx.Lindbladian)
    assert gen.dims == _QUBIT_DIMS


def test_lindbladian_two_jump_ops_additive():
    """Lindbladian.from_operators(None, [L1, L2]) == from_operators(None, L1) + from_operators(None, L2)."""
    L1 = jnp.sqrt(0.1) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    L2 = jnp.sqrt(0.05) * jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    L_stack = jnp.stack([L1, L2])
    jump_ops_combined = qx.Operator.from_matrix(L_stack, ((2,), (2,)))
    gen_combined = qx.Lindbladian.from_operators(None, jump_ops_combined)

    jump_ops1 = qx.Operator.from_matrix(L1[jnp.newaxis], ((2,), (2,)))
    jump_ops2 = qx.Operator.from_matrix(L2[jnp.newaxis], ((2,), (2,)))
    gen_sum = qx.Lindbladian.from_operators(None, jump_ops1) + qx.Lindbladian.from_operators(None, jump_ops2)

    assert jnp.allclose(gen_combined.matrix, gen_sum.matrix, atol=1e-10)


def test_lindbladian_with_hamiltonian():
    """Lindbladian.from_operators(H, L_ops) should include both coherent and dissipative terms."""
    from quax.gates import Z

    H = qx.Observable.from_matrix(0.5 * Z.matrix, ((2,), (2,)))
    L = jnp.sqrt(0.1) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    jump_ops = qx.Operator.from_matrix(L[jnp.newaxis], ((2,), (2,)))

    gen_full = qx.Lindbladian.from_operators(H, jump_ops)
    gen_no_H = qx.Lindbladian.from_operators(None, jump_ops)

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
    L_native = qx.Lindbladian.from_operators(None, jump_ops)

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
    """evolve(Lindbladian.from_operators(H, L_ops), t) matches QuTiP liouvillian."""
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
    qx_channel = qx.evolve(qx.Lindbladian.from_operators(H_qx, jump_ops), T)

    fid = qx.process_fidelity(qx_channel, qt_channel)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999"


# ---------------------------------------------------------------------------
# Kraus channel equivalences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma,t", [(0.3, 0.5), (0.5, 1.0), (0.1, 2.0)])
def test_amplitude_damping_matches_kraus(gamma, t):
    """evolve(amplitude_damping_lindbladian(γ), t) ≈ relaxation_operators(1 - exp(-γt))."""
    p = 1.0 - float(jnp.exp(-gamma * t))
    kraus_channel = qx.kraus_to_superop(qx.relaxation_operators(p))
    lindblad_channel = qx.evolve(qx.lindbladians.amplitude_damping(gamma), t)

    fid = qx.process_fidelity(lindblad_channel, kraus_channel)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999"


@pytest.mark.parametrize("gamma,t", [(0.3, 0.5), (0.5, 1.0)])
def test_dephasing_matches_kraus(gamma, t):
    """evolve(dephasing_lindbladian(γ), t) ≈ dephasing_operators(1 - exp(-γt))."""
    p = 1.0 - float(jnp.exp(-gamma * t))
    kraus_channel = qx.kraus_to_superop(qx.dephasing_operators(p))
    lindblad_channel = qx.evolve(qx.lindbladians.dephasing(gamma), t)

    fid = qx.process_fidelity(lindblad_channel, kraus_channel)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999"


@pytest.mark.parametrize("gamma,t", [(0.15, 0.5), (0.3, 1.0)])
def test_depolarizing_matches_kraus(gamma, t):
    """evolve(depolarizing_lindbladian(γ), t) ≈ depolarizing_operators(p) with p = ¾(1 - exp(-4γt/3))."""
    p = 0.75 * (1.0 - float(jnp.exp(-4.0 * gamma * t / 3.0)))
    kraus_channel = qx.kraus_to_superop(qx.depolarizing_operators(p))
    lindblad_channel = qx.evolve(qx.lindbladians.depolarizing(gamma), t)

    fid = qx.process_fidelity(lindblad_channel, kraus_channel)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999"


@pytest.mark.parametrize("gamma,t", [(0.3, 0.5), (0.5, 1.0)])
def test_bit_flip_matches_kraus(gamma, t):
    """evolve(bit_flip_lindbladian(γ), t) ≈ bit_flip_operators(½(1 - exp(-2γt)))."""
    p = 0.5 * (1.0 - float(jnp.exp(-2.0 * gamma * t)))
    kraus_channel = qx.kraus_to_superop(qx.bit_flip_operators(p))
    lindblad_channel = qx.evolve(qx.lindbladians.bit_flip(gamma), t)

    fid = qx.process_fidelity(lindblad_channel, kraus_channel)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999"


@pytest.mark.parametrize("gamma,t", [(0.3, 0.5), (0.5, 1.0)])
def test_phase_flip_matches_kraus(gamma, t):
    """evolve(phase_flip_lindbladian(γ), t) ≈ phase_flip_operators(½(1 - exp(-2γt)))."""
    p = 0.5 * (1.0 - float(jnp.exp(-2.0 * gamma * t)))
    kraus_channel = qx.kraus_to_superop(qx.phase_flip_operators(p))
    lindblad_channel = qx.evolve(qx.lindbladians.phase_flip(gamma), t)

    fid = qx.process_fidelity(lindblad_channel, kraus_channel)
    assert float(fid.real) > 0.9999, f"Process fidelity {fid} < 0.9999"


@pytest.mark.parametrize("t1,tphi,t", [(1.0, 2.0, 0.5), (2.0, 5.0, 1.0)])
def test_thermal_relaxation_matches_choi(t1, tphi, t):
    """evolve(thermal_relaxation_lindbladian(t1, tphi), t) ≈ thermal_relaxation_choi."""
    t1s = jnp.array([t1])
    tphis = jnp.array([tphi])
    choi_ref = qx.thermal_relaxation_choi(t1s, tphis, t)
    ref_superop = qx.choi_to_superop(choi_ref)

    lindblad_channel = qx.evolve(qx.lindbladians.thermal_relaxation(t1, tphi), t)

    fid = qx.process_fidelity(lindblad_channel, ref_superop)
    assert float(fid.real) > 0.999, f"Process fidelity {fid} < 0.999"


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
    """evolve(Lindbladian.from_operators(H, zero_jumps), t) ≈ unitary_to_superop(U(t)).

    The GKSL equation uses -i[H,ρ], giving evolution exp(-iHt)ρexp(iHt).
    In quax's convention evolve(H, t) = exp(-iHt), so the matching unitary is evolve(H, t).
    """
    from quax.gates import Z

    omega = 0.5
    H = qx.Observable.from_matrix(omega * Z.matrix, ((2,), (2,)))

    # Lindbladian with zero dissipation — single zero jump operator
    zero_L = jnp.zeros((1, 2, 2), dtype=complex)
    jump_ops = qx.Operator.from_matrix(zero_L, ((2,), (2,)))
    gen = qx.Lindbladian.from_operators(H, jump_ops)
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


def test_lindbladian_from_operators_jit():
    """jax.jit(qx.Lindbladian.from_operators) compiles and runs."""
    L = jnp.sqrt(0.1) * jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    jump_ops = qx.Operator.from_matrix(L[jnp.newaxis], ((2,), (2,)))
    gen = jax.jit(qx.Lindbladian.from_operators)(None, jump_ops)
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
    """Lindbladian.from_operators with batched jump operators produces an ensemble of generators."""
    n_batch = 4
    gammas = jnp.linspace(0.1, 0.5, n_batch)
    sigma_minus = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

    L_batch = jnp.sqrt(gammas)[:, None, None] * sigma_minus[None]  # (n_batch, 2, 2)
    L_with_ops_dim = L_batch[:, jnp.newaxis, :, :]  # (n_batch, 1, 2, 2)
    jump_ops = qx.Operator.from_matrix(L_with_ops_dim, ((2,), (2,)))
    gen = qx.Lindbladian.from_operators(None, jump_ops)

    assert gen.ensemble_size == (n_batch,)


def test_evolve_ensemble_lindbladian():
    """evolve over an ensemble Lindbladian produces an ensemble of channels."""
    n_batch = 3
    gammas = jnp.array([0.1, 0.2, 0.3])
    sigma_minus = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

    L_batch = jnp.sqrt(gammas)[:, None, None] * sigma_minus[None]
    L_with_ops_dim = L_batch[:, jnp.newaxis, :, :]
    jump_ops = qx.Operator.from_matrix(L_with_ops_dim, ((2,), (2,)))
    gen = qx.Lindbladian.from_operators(None, jump_ops)

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


# ---------------------------------------------------------------------------
# Random Lindbladians: to_operators round-trip, CPTP, QuTiP parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d, n_jumps, seed", _RANDOM_CASES)
def test_random_lindbladian_to_operators_roundtrip(d, n_jumps, seed):
    """to_operators() then from_operators() reproduces the generator (gauge-invariant round-trip)."""
    gen, _, _ = _random_quax_lindbladian(d, n_jumps, seed)
    H_r, jumps_r = gen.to_operators()

    # d² canonical jump operators are returned regardless of the input count.
    assert jumps_r.matrix.shape == (d * d, d, d)

    rebuilt = qx.Lindbladian.from_operators(H_r, jumps_r)
    assert jnp.allclose(rebuilt.matrix, gen.matrix, atol=1e-6)
    # Gauge invariance at the channel level: same dynamics despite different operators.
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


def test_to_operators_jit():
    """Lindbladian.to_operators() compiles under jax.jit."""
    gen, _, _ = _random_quax_lindbladian(2, 2, 7)
    matrix = jax.jit(lambda g: g.to_operators()[1].matrix)(gen)
    assert matrix.shape == (4, 2, 2)
