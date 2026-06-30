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

"""Squeeze a quantum *state* to its minimum per-subsystem dimension.

``squeeze`` is the inverse-in-spirit of :func:`quax.promote`: where ``promote``
*grows* a subsystem (zero-padding a state), ``squeeze`` *shrinks* each subsystem
to the smallest dimension that still captures the state to within a tolerance.
A trailing level of a subsystem is dropped when its population (the diagonal of
that subsystem's reduced density matrix) is below ``tol``; the squeezed state is
renormalized.

**``squeeze`` is only defined for states** (:class:`StateVector` and
:class:`DensityMatrix`). It is *deliberately* undefined for operators,
channels/superoperators (:class:`Unitary`, :class:`Operator`, :class:`KrausMap`,
:class:`SuperOp`, :class:`Choi`, :class:`PauliLiouville`) and measurement
instruments (:class:`QuantumInstrument`), and raises :class:`TypeError` for them.

The reason is mathematical, not an implementation gap. Dropping a subsystem level
is lossless for a state only because, by positivity of the reduced density matrix,
a level with zero population also carries zero coherence with every other level
(``|ρ_{ij}|² ≤ ρ_{ii} ρ_{jj}``). No analogous statement holds for an operator or
channel: an operator can act non-trivially (e.g. apply a phase) on a level that
the *state* never populates, so "squeezing" an operator only has meaning relative
to an assumption about the states it acts on — an assumption that lives in the
state, not the operator. Concretely, naively truncating an operator/channel block

* discards its (diagonal) action on the dropped levels,
* generally yields a non-unitary "unitary" or a trace-decreasing "channel"
  (the truncated block need not satisfy ``U†U = I`` or ``Σ K†K = I``), and
* for channels depends on the (non-unique) Kraus representation chosen.

If you want to reconcile the dimension of an operator/channel with a state, embed
or promote it up to the state's dimension at apply time (see :func:`quax.promote`
and the ``targeted_apply_*`` helpers) rather than squeezing it down.

This supports leakage-aware simulation in the low-leakage regime: most qudits
stay in the qubit subspace, so states squeeze back to dimension 2 while ideal
gates are promoted to the state's dimension when applied.

Unlike :func:`quax.promote`, ``squeeze`` produces a *data-dependent* output shape
and therefore **cannot be jitted**.
"""

from functools import singledispatch
from typing import List

import jax.numpy as jnp
import numpy as np

from ._quantum_objects import (
    DensityMatrix,
    State,
    StateVector,
)

#: Default tolerance below which a subsystem level's population is treated as
#: negligible and the level is dropped.
DEFAULT_SQUEEZE_TOL = 1e-12


@singledispatch
def squeeze(obj, tol: float = DEFAULT_SQUEEZE_TOL) -> State:
    """Reduce each subsystem of a quantum *state* to its minimum dimension.

    The number of subsystems is preserved; only per-subsystem dimensions shrink.
    A trailing level of a subsystem is dropped when its population is below *tol*
    and the state is renormalized.  See the module docstring for the (tolerance-
    based, lossy) drop criterion.

    ``squeeze`` is defined only for states (:class:`StateVector`,
    :class:`DensityMatrix`).  It is **undefined** for operators, channels and
    instruments — squeezing those is mathematically ill-posed (see the module
    docstring) — and raises :class:`TypeError`.

    :param obj: A quax state (:class:`StateVector` or :class:`DensityMatrix`).
    :param tol: Levels whose population is below this threshold are dropped.
    :return: A new state of the same type on the squeezed dimensions.
    :raises TypeError: If *obj* is not a state.
    """
    raise TypeError(
        f"squeeze is only defined for states (StateVector, DensityMatrix); got {type(obj).__name__}. "
        "Squeezing an operator, channel or instrument is ill-defined because it acts non-trivially on "
        "levels a state may never populate. Promote/embed it up to the state's dimension instead."
    )


# ---------------------------------------------------------------------------
# Keep-dimension helpers (eager / NumPy — squeeze is not jittable)
# ---------------------------------------------------------------------------


def _significant_keep(per_level: np.ndarray, tol: float, floor: int = 2) -> int:
    """Smallest dim that retains every level at or above *tol*.

    ``per_level`` is a 1-D array of per-level populations; the result is
    ``1 + (highest index >= tol)``, then floored to *floor* (the qubit dimension,
    so a depopulated qutrit squeezes to a qubit rather than to a trivial 1-D
    space) and capped at the available number of levels.
    """
    significant = np.nonzero(per_level >= tol)[0]
    raw = int(significant[-1]) + 1 if significant.size else 1
    return min(len(per_level), max(raw, floor))


def _slice_subsystems(data, n: int, group_starts, keep: List[int]):
    """Slice every per-subsystem axis group to ``keep`` dims.

    ``group_starts`` lists the first axis of each qudit-dimension group (e.g. the
    output and input groups of a density matrix); within a group, subsystem ``i``
    is at ``start + i``.
    """
    slc: List[object] = [slice(None)] * data.ndim
    for start in group_starts:
        for i in range(n):
            slc[start + i] = slice(0, keep[i])
    return data[tuple(slc)]


def _keep_from_populations(probs, ne: int, n: int, dims, tol: float) -> List[int]:
    """Per-subsystem keep-dims from a joint-population tensor.

    ``probs`` has shape ``(*ensemble, *dims)`` and holds the joint level
    populations (non-negative, real).  For each subsystem the other subsystems are
    marginalized out and the worst case over the ensemble is taken, so a level is
    kept if it is populated above *tol* for *any* ensemble member.
    """
    max_dim = max(dims)
    rows = []
    for i in range(n):
        other = tuple(ne + j for j in range(n) if j != i)
        marginal = jnp.sum(probs, axis=other) if other else probs  # (*ensemble, d_i)
        per_level = jnp.max(marginal.reshape(-1, dims[i]), axis=0)  # worst case over ensemble
        rows.append(jnp.pad(per_level, (0, max_dim - dims[i])))
    populations = np.asarray(jnp.stack(rows))  # (n, max_dim)
    return [_significant_keep(populations[i, : dims[i]], tol) for i in range(n)]


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------


@squeeze.register(StateVector)
def _squeeze_state_vector(psi: StateVector, tol: float = DEFAULT_SQUEEZE_TOL) -> StateVector:
    """Drop subsystem levels with population below *tol* and renormalize."""
    n = psi.num_qubits
    if n == 0:
        return psi
    dims = psi.dims
    ne = psi.num_ensemble_dims
    # Per-subsystem populations reduced on-device; only the (n, max_dim) table is
    # transferred to the host — never the full state — so squeeze stays cheap for
    # large registers.
    probs = jnp.abs(psi.data) ** 2  # (*ensemble, *dims)
    keep = _keep_from_populations(probs, ne, n, dims, tol)

    if all(k == d for k, d in zip(keep, dims)):
        return psi

    subsystem_axes = tuple(range(ne, ne + n))
    new_data = _slice_subsystems(psi.data, n, (ne,), keep)
    norm = jnp.sqrt(jnp.sum(jnp.abs(new_data) ** 2, axis=subsystem_axes, keepdims=True))
    return StateVector(new_data / norm, num_qubits=n)


@squeeze.register(DensityMatrix)
def _squeeze_density_matrix(rho: DensityMatrix, tol: float = DEFAULT_SQUEEZE_TOL) -> DensityMatrix:
    """Drop subsystem levels with population below *tol* and renormalize the trace.

    The DensityMatrix tensor is ``(*ensemble, *dims_out, *dims_in)``; the joint
    populations are the diagonal (``out == in`` on every subsystem).  Each retained
    block is renormalized so the squeezed state has unit trace per ensemble member.
    """
    n = rho.num_qubits
    if n == 0:
        return rho
    dims = rho.dims
    ne = rho.num_ensemble_dims
    ensemble_shape = rho.ensemble_size

    # Joint level populations P(l0, ..., l_{n-1}) = ρ[..., l, l] (out == in for all
    # subsystems); the matrix diagonal in row-major order, reshaped back to per-qudit.
    diag = jnp.diagonal(rho.matrix, axis1=-2, axis2=-1)  # (*ensemble, D)
    probs = jnp.real(diag).reshape(ensemble_shape + dims)
    keep = _keep_from_populations(probs, ne, n, dims, tol)

    if all(k == d for k, d in zip(keep, dims)):
        return rho

    new_data = _slice_subsystems(rho.data, n, (ne, ne + n), keep)
    new_d = int(np.prod(keep))
    trace = jnp.trace(new_data.reshape(ensemble_shape + (new_d, new_d)), axis1=-2, axis2=-1)
    trace = jnp.real(trace).reshape(ensemble_shape + (1,) * (2 * n))
    return DensityMatrix(new_data / trace, num_qubits=n)
