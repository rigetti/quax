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
r"""Projection of a channel onto the completely positive, trace-preserving maps.

A linear inversion of noisy data, a truncated series or a fitted generator evaluated outside its
range routinely gives a Choi matrix with small negative eigenvalues or a trace-preservation defect.
These functions return the closest channel in Frobenius norm on the Choi matrix:

- :func:`project_to_cp` clips the negative eigenvalues of the (Hermitized) Choi matrix;
- :func:`project_to_tp` is the affine projection onto the trace-preserving maps;
- :func:`project_to_cptp` alternates the two with Dykstra's correction, which converges to the
  projection onto the intersection (plain alternating projections would converge to *a* point of the
  intersection, not the closest one), with the stopping rule of [DYKSTOP]_.

The algorithm follows [PGDB]_, equations 8 and 12.

.. [PGDB] Quantum process tomography via completely positive and trace-preserving projection.
         Knee, Bolduc, Leach & Gauger.
         Phys. Rev. A 98, 062336 (2018).
         https://arxiv.org/abs/1803.10062

.. [DYKSTOP] Dykstra's algorithm and robust stopping criteria.
         Birgin & Raydan.
         Encyclopedia of Optimization, pp. 828-833 (Springer, 2009).
         https://doi.org/10.1007/978-0-387-74759-0_143
"""

from functools import partial
from typing import TypeVar, cast

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Choi, KrausMap, PauliLiouville, SuperOp, SuperOperator, Unitary
from ._superoperator_transformations import choi_to_kraus, choi_to_pauli_liouville, choi_to_superop, to_choi

ChannelT = TypeVar("ChannelT", bound=SuperOperator)


def _hermitian_part(matrix: Array) -> Array:
    return (matrix + jnp.conj(jnp.swapaxes(matrix, -1, -2))) / 2


def _project_cp(choi: Array) -> Array:
    """Clip the negative eigenvalues of the Hermitian part of ``(..., d², d²)`` Choi matrices."""
    eigenvalues, eigenvectors = jnp.linalg.eigh(_hermitian_part(choi))
    clipped = jnp.clip(eigenvalues, 0.0, None)
    return (eigenvectors * clipped[..., None, :]) @ jnp.conj(jnp.swapaxes(eigenvectors, -1, -2))


def _project_tp(choi: Array, d: int) -> Array:
    """The orthogonal projection of ``(..., d², d²)`` Choi matrices onto the trace-preserving maps.

    In the tensor form ``J[a, x, b, y]`` the first factor is the one kept by
    :func:`~quax.is_trace_preserving`; trace preservation is ``sum_x J[a, x, b, x] == delta_ab`` and the
    correction that restores it lies in the span of ``M (x) I``.
    """
    batch = choi.shape[:-2]
    tensor = choi.reshape(batch + (d, d, d, d))
    defect = jnp.einsum("...axbx->...ab", tensor) - jnp.eye(d, dtype=choi.dtype)
    correction = jnp.einsum("...ab,xy->...axby", defect, jnp.eye(d, dtype=choi.dtype)) / d
    return (tensor - correction).reshape(choi.shape)


def _inner(a: Array, b: Array) -> Array:
    """The Frobenius inner product over the last two axes."""
    return jnp.sum(jnp.conj(a) * b, axis=(-2, -1))


@partial(jax.jit, static_argnames=("d", "max_iterations"))
def _project_cptp(choi: Array, d: int, max_iterations: int, tolerance: float) -> Array:
    """Dykstra's alternating projections onto the CP and TP sets, on ``(..., d², d²)`` Choi matrices."""
    zeros = jnp.zeros_like(choi)

    def body(carry):
        state, cp_change, tp_change, last_cp, count, _ = carry
        pre_cp = state - cp_change
        cp = _project_cp(pre_cp)
        new_cp_change = cp - pre_cp
        pre_tp = cp - tp_change
        new_state = _project_tp(pre_tp, d)
        new_tp_change = new_state - pre_tp
        # The robust stopping criterion of [DYKSTOP], per ensemble element; the loop runs until
        # every element has converged.
        residual = (
            jnp.sum(jnp.abs(new_cp_change - cp_change) ** 2, axis=(-2, -1))
            + jnp.sum(jnp.abs(new_tp_change - tp_change) ** 2, axis=(-2, -1))
            + 2 * jnp.abs(_inner(tp_change, new_state - state))
            + 2 * jnp.abs(_inner(cp_change, cp - last_cp))
        )
        return new_state, new_cp_change, new_tp_change, cp, count + 1, jnp.max(residual)

    def keep_going(carry):
        _, _, _, _, count, residual = carry
        return (count < max_iterations) & (residual > tolerance)

    initial = (choi, zeros, zeros, zeros, jnp.asarray(0), jnp.asarray(jnp.inf, dtype=float))
    state, _, _, _, _, _ = jax.lax.while_loop(keep_going, body, initial)
    return state


def _same_representation(result: Choi, like: SuperOperator | Unitary) -> SuperOperator:
    """Return ``result`` in the representation of ``like`` (a ``Unitary`` input comes back as a ``SuperOp``)."""
    match like:
        case Choi():
            return result
        case SuperOp():
            return choi_to_superop(result)
        case PauliLiouville():
            return choi_to_pauli_liouville(result)
        case KrausMap():
            return choi_to_kraus(result)
        case _:
            return choi_to_superop(result)


def project_to_cp(channel: ChannelT) -> ChannelT:
    """The closest completely positive map, by clipping the negative eigenvalues of the Choi matrix.

    Equation 8 of [PGDB]_.  Ensembles are supported; the result has the representation of the input.

    :param channel: A ``Choi``, ``SuperOp``, ``PauliLiouville`` or ``KrausMap``.
    :return: The projected channel, in the same representation.
    """
    choi = to_choi(channel)
    return cast(ChannelT, _same_representation(Choi.from_matrix(_project_cp(choi.matrix), choi.dims), channel))


def project_to_tp(channel: ChannelT) -> ChannelT:
    """The closest trace-preserving map, an affine projection of the Choi matrix.

    Equation 12 of [PGDB]_.  Ensembles are supported; the result has the representation of the input.

    :param channel: A ``Choi``, ``SuperOp``, ``PauliLiouville`` or ``KrausMap``.
    :return: The projected channel, in the same representation.
    """
    choi = to_choi(channel)
    return cast(
        ChannelT, _same_representation(Choi.from_matrix(_project_tp(choi.matrix, choi.d[0]), choi.dims), channel)
    )


def project_to_cptp(channel: ChannelT, max_iterations: int = 10_000, tolerance: float = 1e-16) -> ChannelT:
    """The closest completely positive, trace-preserving map in Frobenius norm on the Choi matrix.

    Dykstra's alternating projections between the CP and TP sets ([PGDB]_), stopped by the criterion of
    [DYKSTOP]_ or at ``max_iterations``.  A channel that is already CPTP comes back unchanged after one
    iteration.  Ensembles are supported and projected together; the whole loop runs under ``jax.jit``.

    :param channel: A ``Choi``, ``SuperOp``, ``PauliLiouville`` or ``KrausMap``.
    :param max_iterations: The cap on the number of alternating projections.
    :param tolerance: The threshold on the stopping criterion, a squared Frobenius distance, so the
        remaining violation of complete positivity is about its square root: the default leaves Choi
        eigenvalues above about ``-1e-9`` after a few dozen iterations.  Round-off puts the floor of the
        criterion near ``1e-18``; below that the loop runs to ``max_iterations``.
    :return: The projected channel, in the same representation.
    """
    choi = to_choi(channel)
    projected = _project_cptp(choi.matrix, choi.d[0], max_iterations, tolerance)
    return cast(ChannelT, _same_representation(Choi.from_matrix(projected, choi.dims), channel))
