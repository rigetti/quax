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
"""Tests for the projection onto completely positive and trace-preserving maps."""

from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp
import pytest

import quax as qx

DIMS = [(2,), (2, 2), (3,)]


def _d(dims):
    return reduce(mul, dims, 1)


def _perturbed_choi(dims, key, scale=0.05, size=()) -> qx.Choi:
    """A random channel plus a random Hermitian perturbation: neither CP nor TP."""
    key_channel, key_noise = jax.random.split(key)
    choi = qx.random_choi((dims, dims), rank=_d(dims), key=key_channel, size=size)
    d2 = _d(dims) ** 2
    noise = jax.random.normal(key_noise, size + (d2, d2)) + 1j * jax.random.normal(jax.random.key(0), size + (d2, d2))
    noise = scale * (noise + jnp.conj(jnp.swapaxes(noise, -1, -2))) / 2
    return qx.Choi.from_matrix(choi.matrix + noise, choi.dims)


@pytest.mark.parametrize("dims", DIMS, ids=str)
def test_cptp_channel_is_a_fixed_point(dims):
    channel = qx.choi_to_superop(qx.random_choi((dims, dims), rank=_d(dims), key=jax.random.key(1)))
    projected = qx.project_to_cptp(channel)
    assert isinstance(projected, qx.SuperOp)
    assert jnp.allclose(projected.matrix, channel.matrix, atol=1e-10)


@pytest.mark.parametrize("dims", DIMS, ids=str)
def test_projection_is_cptp(dims):
    choi = _perturbed_choi(dims, jax.random.key(2))
    assert not qx.is_cptp(choi)
    projected = qx.project_to_cptp(choi)
    assert isinstance(projected, qx.Choi)
    assert qx.is_cptp(projected, atol=1e-8)
    assert not jnp.allclose(projected.matrix, choi.matrix, atol=1e-3)


def test_projection_is_the_closest_cptp_map():
    """The projection P of J onto a convex set satisfies <J - P, Q - P> <= 0 for every Q in it."""
    dims = (2,)
    choi = _perturbed_choi(dims, jax.random.key(3), scale=0.2)
    projected = qx.project_to_cptp(choi, tolerance=1e-16)
    others = qx.random_choi((dims, dims), rank=4, key=jax.random.key(4), size=(50,))
    inner = jnp.sum(jnp.conj(choi.matrix - projected.matrix) * (others.matrix - projected.matrix), axis=(-2, -1))
    assert jnp.all(jnp.real(inner) < 1e-7), jnp.max(jnp.real(inner))
    # and it is closer than every one of them
    distance = jnp.linalg.norm(choi.matrix - projected.matrix)
    assert jnp.all(jnp.linalg.norm(choi.matrix - others.matrix, axis=(-2, -1)) > distance)


def test_depolarizing_beyond_its_range_projects_onto_the_boundary():
    """An isotropic map projects onto an isotropic one, and the physical range of the qubit depolarizing
    parameter ends at -1/3."""
    unphysical = qx.pauli_liouville_to_superop(
        qx.PauliLiouville.from_matrix(jnp.diag(jnp.array([1.0, -0.6, -0.6, -0.6])), ((2,), (2,)))
    )
    projected = qx.to_pauli_liouville(qx.project_to_cptp(unphysical, tolerance=1e-16))
    assert jnp.allclose(projected.matrix, jnp.diag(jnp.array([1.0, -1 / 3, -1 / 3, -1 / 3])), atol=1e-6)


def test_tp_projection_fixes_the_identity_row_and_nothing_else():
    dims = (2,)
    matrix = jax.random.normal(jax.random.key(5), (4, 4)) / 4
    defective = qx.pauli_liouville_to_superop(qx.PauliLiouville.from_matrix(matrix, (dims, dims)))
    assert not qx.is_trace_preserving(defective)
    projected = qx.to_pauli_liouville(qx.project_to_tp(defective))
    assert jnp.allclose(projected.matrix[0], jnp.eye(4)[0], atol=1e-12)
    assert jnp.allclose(projected.matrix[1:], matrix[1:], atol=1e-12)
    assert qx.is_trace_preserving(qx.project_to_tp(defective))


def test_cp_projection_clips_the_choi_spectrum():
    choi = _perturbed_choi((2, 2), jax.random.key(6))
    assert jnp.min(jnp.linalg.eigvalsh(choi.matrix)) < -1e-3
    projected = qx.project_to_cp(choi)
    assert jnp.min(jnp.linalg.eigvalsh(projected.matrix)) > -1e-12
    assert qx.is_completely_positive(projected)


def test_every_representation_comes_back_in_kind():
    choi = _perturbed_choi((2,), jax.random.key(7))
    reference = qx.project_to_cptp(choi)
    for convert in (qx.choi_to_superop, qx.choi_to_pauli_liouville):
        channel = convert(choi)
        projected = qx.project_to_cptp(channel)
        assert type(projected) is type(channel)
        assert jnp.allclose(qx.to_superop(projected).matrix, qx.choi_to_superop(reference).matrix, atol=1e-8)
    # A perturbed Choi has no Kraus form; project a channel that does and check the type round trip.
    kraus = qx.choi_to_kraus(qx.random_choi(((2,), (2,)), rank=2, key=jax.random.key(8)))
    projected = qx.project_to_cptp(kraus)
    assert isinstance(projected, qx.KrausMap)
    assert jnp.allclose(qx.to_superop(projected).matrix, qx.to_superop(kraus).matrix, atol=1e-8)


def test_ensembles_are_projected_together():
    choi = _perturbed_choi((2,), jax.random.key(9), size=(3,))
    together = qx.project_to_cptp(choi)
    assert together.ensemble_size == (3,)
    for k in range(3):
        alone = qx.project_to_cptp(choi[k])
        assert jnp.allclose(together.matrix[k], alone.matrix, atol=1e-8)
        assert qx.is_cptp(together[k], atol=1e-8)


def test_the_projections_are_jitted_and_nest_inside_jit():
    """All three carry ``jax.jit``; calling one inside another jitted function still traces."""
    choi = _perturbed_choi((2, 2), jax.random.key(10))
    for project in (qx.project_to_cp, qx.project_to_tp, qx.project_to_cptp):
        assert isinstance(project, jax.stages.Wrapped)
        assert jnp.allclose(jax.jit(lambda c, f=project: f(c))(choi).matrix, project(choi).matrix, atol=1e-10)


def test_cp_and_tp_projections_are_differentiable():
    """``project_to_cptp`` stops at a data-dependent ``while_loop``; the single projections do not."""
    choi = _perturbed_choi((2,), jax.random.key(21))

    for project in (qx.project_to_cp, qx.project_to_tp):

        def norm(matrix, f=project):
            return jnp.sum(jnp.abs(f(qx.Choi.from_matrix(matrix, choi.dims)).matrix) ** 2)

        assert jnp.all(jnp.isfinite(jax.grad(norm)(choi.matrix)))


def test_iteration_cap_is_respected():
    choi = _perturbed_choi((2, 2), jax.random.key(11), scale=0.3)
    capped = qx.project_to_cptp(choi, max_iterations=2)
    converged = qx.project_to_cptp(choi)
    assert not jnp.allclose(capped.matrix, converged.matrix, atol=1e-8)
