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
r"""Linear-inversion tomography from a table of expectation values.

The forward model of process tomography is bilinear in the Pauli vectors (:mod:`._pauli_vector`)
of the input states :math:`r_i` and observables :math:`c_j`:

.. math:: E_{ij} = \mathrm{Tr}[O_j\,S(\rho_i)] = \frac{1}{d}\sum_{ab} c_{ja} R_{ab} r_{ib},

with :math:`R` the Pauli-Liouville matrix of the channel.  In matrix form
:math:`E = r R^T c^T / d`, and because the pseudo-inverse of a Kronecker product is the Kronecker
product of the pseudo-inverses, the minimum-norm least-squares channel is

.. math:: R^T = d\, r^{+} E\, (c^{+})^T

without ever forming the :math:`nm \times d^4` design matrix.  A trace-preserving channel has the
fixed first row :math:`R_{0b} = \delta_{0b}`, which traceless observables carry no information
about, so that row is fixed and only the others are fit.
"""

from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp
from jax import Array

from ._pauli_vector import to_pauli_vector
from ._quantum_objects import DensityMatrix, Observable, PauliLiouville, SuperOp
from ._superoperator_transformations import pauli_liouville_to_superop


def _square_dims(observables: Observable) -> tuple[int, ...]:
    dims_out, dims_in = observables.dims
    if tuple(dims_out) != tuple(dims_in):
        raise ValueError(f"Tomography needs square observables, got dims {observables.dims}.")
    return tuple(dims_out)


@jax.jit(static_argnames=("trace_preserving",))
def linear_inversion_process(
    states: DensityMatrix,
    observables: Observable,
    expectations: Array,
    trace_preserving: bool = True,
) -> SuperOp:
    r"""The channel whose expectation table best matches the measured one, by linear inversion.

    Solves :math:`E_{ij} = \mathrm{Tr}[O_j\,S(\rho_i)]` for :math:`S` in the least-squares sense.  When the
    settings are informationally complete the solution is unique and exact for exact data; otherwise
    it is the minimum-norm solution.  The result is not projected onto the physical channels; compose
    with :func:`~quax.project_to_cptp` for that.

    The function is under ``jax.jit`` and differentiable in ``expectations``; ``trace_preserving``
    selects the solve, so it is a static argument.

    :param states: The ``n`` input states, ``ensemble_size == (n,)``.
    :param observables: The ``m`` measured observables, ``ensemble_size == (m,)``.
    :param expectations: The measured table, shape ``(*batch, n, m)``; a batch gives an ensemble of channels.
    :param trace_preserving: Fix the identity row of the Pauli-Liouville matrix to ``(1, 0, ..., 0)``
        and fit the rest.  Traceless observables cannot determine that row, so this is the default;
        turn it off to fit a general (e.g. trace-decreasing) map from data with traceful observables.
    :return: The channel as a ``SuperOp``, with the batch shape of ``expectations`` as its ensemble.
    """
    dims = tuple(states.dims)
    if dims != _square_dims(observables):
        raise ValueError(f"States and observables have different dimensions, {states.dims} and {observables.dims}.")
    if len(states.ensemble_size) != 1 or len(observables.ensemble_size) != 1:
        raise ValueError("linear_inversion_process takes one-dimensional ensembles of states and observables.")
    d = reduce(mul, dims, 1)
    r = to_pauli_vector(states)  # (n, d²)
    c = to_pauli_vector(observables)  # (m, d²)
    measured = jnp.asarray(expectations, dtype=float)
    if measured.shape[-2:] != (r.shape[0], c.shape[0]):
        raise ValueError(
            f"Expected an expectation table of shape (..., {r.shape[0]}, {c.shape[0]}), got {measured.shape}."
        )

    if trace_preserving:
        # The identity row contributes (1/d) r_i[0] c_j[0]; the other rows are the unknowns.
        fixed = jnp.outer(r[:, 0], c[:, 0]) / d
        free_rows_t = d * jnp.linalg.pinv(r) @ (measured - fixed) @ jnp.linalg.pinv(c[:, 1:]).T  # (..., d², d²-1)
        free_rows = jnp.swapaxes(free_rows_t, -1, -2)
        identity_row = jnp.broadcast_to(jnp.eye(d * d)[:1], free_rows.shape[:-2] + (1, d * d))
        pauli_liouville = jnp.concatenate([identity_row, free_rows], axis=-2)
    else:
        pauli_liouville = jnp.swapaxes(d * jnp.linalg.pinv(r) @ measured @ jnp.linalg.pinv(c).T, -1, -2)
    return pauli_liouville_to_superop(PauliLiouville.from_matrix(pauli_liouville, (dims, dims)))


@jax.jit(static_argnames=("unit_trace",))
def linear_inversion_state(observables: Observable, expectations: Array, unit_trace: bool = True) -> DensityMatrix:
    r"""The state whose expectation values best match the measured ones, by linear inversion.

    Solves :math:`e_j = \mathrm{Tr}[O_j\,\rho]` for :math:`\rho` in the minimum-norm least-squares sense.
    The result is Hermitian but not necessarily positive.

    The function is under ``jax.jit`` and differentiable in ``expectations``; ``unit_trace`` selects
    the solve, so it is a static argument.

    :param observables: The ``m`` measured observables, ``ensemble_size == (m,)``.
    :param expectations: The measured values, shape ``(*batch, m)``; a batch gives an ensemble of states.
    :param unit_trace: Fix :math:`\mathrm{Tr}[\rho] = 1` and fit only the traceless part, which is all that
        traceless observables can determine.
    :return: The state as a ``DensityMatrix``, with the batch shape of ``expectations`` as its ensemble.
    """
    dims = _square_dims(observables)
    if len(observables.ensemble_size) != 1:
        raise ValueError("linear_inversion_state takes a one-dimensional ensemble of observables.")
    d = reduce(mul, dims, 1)
    c = to_pauli_vector(observables)  # (m, d²)
    measured = jnp.asarray(expectations, dtype=float)
    if measured.shape[-1] != c.shape[0]:
        raise ValueError(f"Expected {c.shape[0]} expectation values along the last axis, got {measured.shape}.")

    if unit_trace:
        traceless = d * (measured - c[:, 0] / d) @ jnp.linalg.pinv(c[:, 1:]).T  # (..., d²-1)
        ones = jnp.ones(traceless.shape[:-1] + (1,), dtype=traceless.dtype)
        pauli_vector = jnp.concatenate([ones, traceless], axis=-1)
    else:
        pauli_vector = d * measured @ jnp.linalg.pinv(c).T
    return DensityMatrix.from_pauli_vector(pauli_vector, dims)
