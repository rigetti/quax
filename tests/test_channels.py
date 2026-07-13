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

"""Tests for the qx.channels SuperOp constructors."""

import jax
import jax.numpy as jnp
import pytest

import quax as qx


@pytest.mark.parametrize(
    "channel_fn, lindblad_fn",
    [
        (lambda: qx.channels.depolarizing(0.1), lambda: qx.lindbladians.depolarizing(0.1)),
        (lambda: qx.channels.amplitude_damping(0.1), lambda: qx.lindbladians.amplitude_damping(0.1)),
        (lambda: qx.channels.dephasing(0.1), lambda: qx.lindbladians.dephasing(0.1)),
        (lambda: qx.channels.bit_flip(0.1), lambda: qx.lindbladians.bit_flip(0.1)),
        (lambda: qx.channels.phase_flip(0.1), lambda: qx.lindbladians.phase_flip(0.1)),
        (lambda: qx.channels.leakage(0.1), lambda: qx.lindbladians.leakage(0.1)),
        (lambda: qx.channels.seepage(0.1), lambda: qx.lindbladians.seepage(0.1)),
        (
            lambda: qx.channels.thermal_relaxation(50e-6, 30e-6),
            lambda: qx.lindbladians.thermal_relaxation(50e-6, 30e-6),
        ),
    ],
)
def test_channel_is_evolved_lindbladian(channel_fn, lindblad_fn):
    """Each qx.channels.* is a CPTP SuperOp equal to evolve(the matching lindbladian, 1.0)."""
    channel = channel_fn()
    assert isinstance(channel, qx.SuperOp)
    assert qx.is_cptp(channel)
    assert jnp.allclose(channel.matrix, qx.evolve(lindblad_fn(), 1.0).matrix, atol=1e-10)


def test_channel_time_argument():
    """The t argument scales the evolution: channels.X(rate, t) == evolve(lindbladians.X(rate), t)."""
    assert jnp.allclose(
        qx.channels.dephasing(0.2, t=0.5).matrix,
        qx.evolve(qx.lindbladians.dephasing(0.2), 0.5).matrix,
        atol=1e-10,
    )


def test_depolarizing_channel_dims():
    """channels.depolarizing accepts a dims argument (qutrit / multi-qubit)."""
    for dims, d in [((2,), 2), ((3,), 3), ((2, 2), 4)]:
        channel = qx.channels.depolarizing(0.2, dims)
        assert channel.dims == (dims, dims)
        assert qx.is_cptp(channel)


def test_channel_jit_and_grad():
    """Channels compile under jit and are differentiable in the rate."""
    ch = jax.jit(qx.channels.amplitude_damping)(0.1)
    assert qx.is_cptp(ch)
    g = jax.grad(lambda r: qx.channels.amplitude_damping(r).matrix.real.sum())(0.1)
    assert jnp.isfinite(g)


@pytest.mark.parametrize(
    "make",
    [
        lambda: jax.jit(lambda r: qx.channels.dephasing(r).matrix)(0.1),
        lambda: jax.jit(lambda r: qx.channels.bit_flip(r).matrix)(0.1),
        lambda: jax.jit(lambda r: qx.channels.phase_flip(r).matrix)(0.1),
        lambda: jax.jit(lambda r: qx.channels.amplitude_damping(r).matrix)(0.1),
        lambda: jax.jit(lambda r: qx.channels.leakage(r).matrix)(0.1),
        lambda: jax.jit(lambda r: qx.channels.seepage(r).matrix)(0.1),
        lambda: jax.jit(lambda a, b: qx.channels.thermal_relaxation(a, b).matrix)(50e-6, 30e-6),
        lambda: jax.jit(lambda r: qx.channels.depolarizing(r, (2,)).matrix)(0.1),
        lambda: jax.jit(lambda t: qx.channels.instrument_from_axis(t).matrix)(0.3),
    ],
    ids=[
        "dephasing",
        "bit_flip",
        "phase_flip",
        "amplitude_damping",
        "leakage",
        "seepage",
        "thermal_relaxation",
        "depolarizing",
        "instrument_from_axis",
    ],
)
def test_channel_constructors_jit(make):
    """Every channels.* constructor is jit-compatible."""
    result = make()
    assert jnp.all(jnp.isfinite(result))


def test_instrument_from_confusion_jit():
    """instrument_from_confusion_and_transition is jit-compatible (dims static)."""
    from functools import partial

    conf = jnp.array([[0.9, 0.1], [0.1, 0.9]])
    trans = jnp.eye(2)
    qi = jax.jit(partial(qx.channels.instrument_from_confusion_and_transition, dims=(2,)))(conf, trans)
    assert qi.num_outcomes == 2


def test_amplitude_damping_qudit_harmonic():
    """Qudit amplitude damping uses the harmonic annihilation (rate n·gamma out of level n)."""
    # qubit reduces to sqrt(gamma)|0><1|
    q = qx.lindbladians.amplitude_damping(0.1)
    assert jnp.allclose(q.jump_operators.matrix.squeeze(0), jnp.sqrt(0.1) * jnp.array([[0, 1], [0, 0]], dtype=complex))
    # qutrit: a = sqrt(gamma)(|0><1| + sqrt(2)|1><2|)
    a = qx.lindbladians.amplitude_damping(0.1, (3,)).jump_operators.matrix.squeeze(0)
    expected = jnp.sqrt(0.1) * jnp.array([[0, 1, 0], [0, 0, jnp.sqrt(2.0)], [0, 0, 0]], dtype=complex)
    assert jnp.allclose(a, expected)
    assert qx.is_cptp(qx.evolve(qx.lindbladians.amplitude_damping(0.1, (3,)), 1.0))
