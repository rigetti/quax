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

"""Tests for the leakage and seepage rates and the computational-subspace process fidelity [WG18]."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quax as qx

QUTRITS = (3, 3)


def _noisy(gate: qx.Unitary, noise: qx.Lindbladian) -> qx.SuperOp:
    """The gate with its noise generator, the gate promoted to the generator's dims."""
    return qx.evolve(noise + gate, 1.0)


def _exchange(rate_20: float, rate_02: float, seepage: float = 0.0) -> qx.Lindbladian:
    """|11> -> |20>, |02> at the two rates, and back at the seepage rate."""
    generator = qx.lindbladians.transition(rate_20, (2, 0), (1, 1), QUTRITS) + qx.lindbladians.transition(
        rate_02, (0, 2), (1, 1), QUTRITS
    )
    if seepage:
        generator = (
            generator
            + qx.lindbladians.transition(seepage, (1, 1), (2, 0), QUTRITS)
            + qx.lindbladians.transition(seepage, (1, 1), (0, 2), QUTRITS)
        )
    return generator


# ============================================================================
# lindbladians.transition
# ============================================================================


def test_transition_jump_operator():
    generator = qx.lindbladians.transition(0.04, (2, 0), (1, 1), QUTRITS)
    expected = jnp.zeros((9, 9), dtype=complex).at[6, 4].set(0.2)
    assert generator.dims == (QUTRITS, QUTRITS)
    assert jnp.allclose(generator.jump_operators.matrix[0], expected)


def test_leakage_and_seepage_are_transitions():
    assert jnp.allclose(qx.lindbladians.leakage(0.1).matrix, qx.lindbladians.transition(0.1, (2,), (1,), (3,)).matrix)
    assert jnp.allclose(qx.lindbladians.seepage(0.1).matrix, qx.lindbladians.transition(0.1, (1,), (2,), (3,)).matrix)


def test_transition_ensemble():
    generator = qx.lindbladians.transition(jnp.array([0.1, 0.2]), (2,), (1,), (3,))
    assert generator.ensemble_size == (2,)


@pytest.mark.parametrize(
    "final, initial",
    [((3,), (1,)), ((2, 0), (1,)), ((1,), (1,))],
    ids=["level-out-of-range", "wrong-length", "same-state"],
)
def test_transition_rejects_bad_labels(final, initial):
    with pytest.raises(ValueError):
        qx.lindbladians.transition(0.1, final, initial, (3,))


# ============================================================================
# leakage_rate, seepage_rate, and process_fidelity to a unitary on fewer levels
# ============================================================================


class TestQubitChannels:
    """On qubits alone nothing leaks, and the subspace fidelity on the whole space is the usual one."""

    @pytest.mark.parametrize("seed", [0, 1])
    def test_reduce_to_the_usual_figures_of_merit(self, seed):
        channel = qx.random_choi(((2, 2), (2, 2)), rank=4, key=jax.random.key(seed))
        assert float(qx.leakage_rate(channel)) == pytest.approx(0.0, abs=1e-12)
        # Promoted to qutrits it leaks nothing, so its fidelity on the qubit subspace is the usual one. The usual
        # fidelity takes a matrix square root (Jozsa), the subspace one a trace, so they agree to ~1e-9.
        promoted = qx.promote(channel, QUTRITS)
        assert float(qx.process_fidelity(promoted, qx.gates.I | qx.gates.I)) == pytest.approx(
            float(qx.process_fidelity(channel)), abs=1e-7
        )

    def test_a_qubit_has_no_seepage_rate(self):
        with pytest.raises(ValueError, match="no leakage subspace"):
            qx.seepage_rate(qx.channels.depolarizing(0.1))


class TestCZLeakage:
    RATE_20 = 0.02
    RATE_02 = 0.01

    def test_subspace_rates_follow_the_hand_formula(self):
        """Only |11> leaks, with population 1/4 of the uniform computational state, split by the two rates."""
        noisy = _noisy(qx.gates.CZ, _exchange(self.RATE_20, self.RATE_02))
        total = self.RATE_20 + self.RATE_02
        l1 = (1 - np.exp(-total)) / 4

        assert float(qx.leakage_rate(noisy)) == pytest.approx(l1, rel=1e-6)
        assert float(qx.leakage_rate(noisy, leaked=(0,))) == pytest.approx(self.RATE_20 / total * l1, rel=1e-6)
        assert float(qx.leakage_rate(noisy, leaked=(1,))) == pytest.approx(self.RATE_02 / total * l1, rel=1e-6)
        # One exchange leaks one qutrit, never both.
        assert float(qx.leakage_rate(noisy, leaked=(0, 1))) == pytest.approx(0.0, abs=1e-12)

    def test_subspace_rates_sum_to_the_total(self):
        noisy = _noisy(qx.gates.CZ, _exchange(0.03, 0.01, 0.02) + qx.lindbladians.depolarizing(0.02, (3, 3)))
        subspaces = [(0,), (1,), (0, 1)]
        total = sum(float(qx.leakage_rate(noisy, leaked=y)) for y in subspaces)
        assert total == pytest.approx(float(qx.leakage_rate(noisy)), rel=1e-10)

    def test_seepage_returns_population_through_the_same_exchanges(self):
        """Of qutrit 0's leaked states with a computational partner only |20> seeps; of all five leaked states, two."""
        rate = 0.05
        noisy = _noisy(qx.gates.CZ, _exchange(0.0, 0.0, seepage=rate))
        returned = 1 - np.exp(-rate)

        assert float(qx.seepage_rate(noisy, leaked=(0,))) == pytest.approx(returned / 2, rel=1e-6)
        assert float(qx.seepage_rate(noisy)) == pytest.approx(2 * returned / 5, rel=1e-6)
        assert float(qx.leakage_rate(noisy)) == pytest.approx(0.0, abs=1e-12)

    def test_target_promotes_to_the_channel(self):
        """The fidelity to the CZ of the noisy gate is the fidelity to the qubit identity of its error channel."""
        noise = _exchange(self.RATE_20, self.RATE_02) + qx.lindbladians.depolarizing(0.03, (2, 2))
        noisy = _noisy(qx.gates.CZ, noise)
        error = qx.to_superop(qx.promote(qx.gates.CZ, QUTRITS).h).matrix @ qx.to_superop(noisy).matrix
        error = qx.SuperOp.from_matrix(error, (QUTRITS, QUTRITS))
        assert float(qx.process_fidelity(noisy, qx.gates.CZ)) == pytest.approx(
            float(qx.process_fidelity(error, qx.gates.I | qx.gates.I)), abs=1e-12
        )

    def test_a_promoted_target_compares_on_the_whole_space(self):
        """A target already on the channel's dims is the usual fidelity, which a leak into a dark level costs less."""
        noisy = _noisy(qx.gates.CZ, _exchange(self.RATE_20, self.RATE_02))
        whole = float(qx.process_fidelity(noisy, qx.promote(qx.gates.CZ, QUTRITS)))
        subspace = float(qx.process_fidelity(noisy, qx.gates.CZ))
        assert whole > subspace

    def test_jit(self):
        """The target's dims are static, so plain jit picks the subspace fidelity with nothing marked static."""
        noisy = _noisy(qx.gates.CZ, _exchange(self.RATE_20, self.RATE_02))
        assert float(jax.jit(qx.process_fidelity)(noisy, qx.gates.CZ)) == pytest.approx(
            float(qx.process_fidelity(noisy, qx.gates.CZ)), abs=1e-12
        )

    def test_leakage_lowers_the_fidelity_by_its_population(self):
        """Without other noise only |11> decays, its amplitude to exp(-g/2), so F = |3 + exp(-g/2)|^2 / 16."""
        noisy = _noisy(qx.gates.CZ, _exchange(self.RATE_20, self.RATE_02))
        amplitude = np.exp(-(self.RATE_20 + self.RATE_02) / 2)
        expected = (3 + amplitude) ** 2 / 16
        assert float(qx.process_fidelity(noisy, qx.gates.CZ)) == pytest.approx(expected, rel=1e-6)

    def test_a_unitary_on_fewer_levels_defines_the_subspace_on_either_side(self):
        noisy = _noisy(qx.gates.CZ, _exchange(self.RATE_20, self.RATE_02))
        assert float(qx.process_fidelity(qx.gates.CZ, noisy)) == pytest.approx(
            float(qx.process_fidelity(noisy, qx.gates.CZ)), abs=1e-12
        )

    def test_rejects_a_smaller_superoperator(self):
        """Promoting a channel to more levels is not unique, so a smaller superoperator is never promoted."""
        noisy = _noisy(qx.gates.CZ, _exchange(self.RATE_20, self.RATE_02))
        with pytest.raises(ValueError, match="cannot be compared"):
            qx.process_fidelity(noisy, qx.to_superop(qx.gates.CZ))

    def test_rejects_a_superoperator_on_fewer_levels_than_a_unitary(self):
        with pytest.raises(ValueError, match="cannot be compared"):
            qx.process_fidelity(qx.channels.depolarizing(0.1), qx.promote(qx.gates.X, (3,)))


class TestEquality:
    """``==`` compares a unitary on fewer levels on its computational subspace, as ``process_fidelity``; never a channel."""

    RZ: qx.Unitary = qx.gates.RZ(np.pi / 2)
    PHASE: qx.Unitary = qx.Unitary.from_matrix(jnp.diag(jnp.array([1.0, 1.0j])), ((2,), (2,)))
    """The same qubit channel as ``RZ``, a global phase apart."""

    def test_a_unitary_equals_its_promotion(self):
        promoted = qx.promote(qx.gates.X, (3,))
        assert qx.gates.X == promoted
        assert qx.gates.X == qx.to_superop(promoted)
        assert qx.to_superop(promoted) == qx.gates.X

    def test_superoperators_on_different_dims_are_unequal(self):
        assert qx.to_superop(qx.gates.X) != qx.to_superop(qx.promote(qx.gates.X, (3,)))

    def test_the_global_phase_of_a_promotion_does_not_matter(self):
        """``RZ`` and the phase gate are the same qubit channel, and equal either one's promotion."""
        assert qx.to_superop(self.RZ) == qx.to_superop(self.PHASE)
        assert self.RZ == qx.promote(self.PHASE, (3,))
        assert self.RZ == qx.to_superop(qx.promote(self.PHASE, (3,)))
        minus_identity = qx.Unitary.from_matrix(-qx.gates.I.matrix, ((2,), (2,)))
        assert minus_identity == qx.promote(qx.gates.I, (3,))

    def test_the_levels_above_do_not_matter(self):
        """Seepage from |2> to |0> leaves the computational subspace alone, so the channel equals the identity there."""
        seeping = qx.evolve(qx.lindbladians.transition(0.5, (0,), (2,), (3,)), 1.0)
        assert qx.gates.I == seeping
        assert seeping == qx.gates.I

    def test_a_leaking_channel_is_unequal(self):
        leaking = qx.evolve(qx.lindbladians.leakage(0.5), 1.0)
        assert qx.gates.I != leaking
        assert leaking != qx.gates.I


class TestOneQutritLeakage:
    GAMMA = 0.1

    def test_rz_leakage_from_the_uniform_state(self):
        """RZ leaves the populations alone, so |1> holds half the population throughout and L_1 = (1 - e^-g)/2."""
        noisy = _noisy(qx.gates.RZ(np.pi / 2), qx.lindbladians.leakage(self.GAMMA))
        assert noisy.dims == ((3,), (3,))
        assert float(qx.leakage_rate(noisy)) == pytest.approx((1 - np.exp(-self.GAMMA)) / 2, rel=1e-6)
        assert float(qx.seepage_rate(noisy)) == pytest.approx(0.0, abs=1e-12)

    def test_rx_leaks_a_little_more_than_the_first_order_estimate(self):
        """The drive refills |1> from |0> while it leaks, so an RX leaks more than a gate that holds the populations."""
        first_order = (1 - np.exp(-self.GAMMA)) / 2
        rate = float(qx.leakage_rate(_noisy(qx.gates.RX(np.pi / 2), qx.lindbladians.leakage(self.GAMMA))))
        assert first_order < rate < 1.01 * first_order

    def test_ensemble(self):
        noisy = qx.channels.leakage(jnp.array([0.1, 0.2]))
        rates = qx.leakage_rate(noisy)
        assert rates.shape == (2,)
        assert jnp.allclose(rates, (1 - jnp.exp(-jnp.array([0.1, 0.2]))) / 2)

    def test_jit(self):
        rate = jax.jit(lambda gamma: qx.leakage_rate(qx.channels.leakage(gamma)))(0.1)
        assert float(rate) == pytest.approx((1 - np.exp(-0.1)) / 2, rel=1e-6)

    def test_subspace_dims(self):
        """A ququart whose computational subspace is a qutrit: leakage out of |2> into |3>."""
        noisy = qx.evolve(qx.lindbladians.transition(self.GAMMA, (3,), (2,), (4,)), 1.0)
        assert float(qx.leakage_rate(noisy, subspace_dims=(3,))) == pytest.approx(
            (1 - np.exp(-self.GAMMA)) / 3, rel=1e-6
        )
        assert float(qx.leakage_rate(noisy)) == pytest.approx(0.0, abs=1e-12)
