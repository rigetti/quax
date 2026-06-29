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

"""Squeeze quantum objects to their minimum per-subsystem dimension.

``squeeze`` is the inverse-in-spirit of :func:`quax.promote`: where ``promote``
*grows* a subsystem (zero-padding states, identity-extending operators),
``squeeze`` *shrinks* each subsystem to the smallest dimension that still
captures the object to within a tolerance.

It is **tolerance-based and lossy** by design:

* For a :class:`StateVector`, a trailing level of a subsystem is dropped when its
  population (the diagonal of that subsystem's reduced density matrix) is below
  ``tol``; the squeezed state is renormalized.
* For an operator/channel (:class:`Unitary`, :class:`Operator`, :class:`KrausMap`,
  and — via the Kraus representation — :class:`SuperOp`, :class:`Choi`,
  :class:`PauliLiouville`), a trailing level of a subsystem is dropped when the
  operator's *coupling* between that level and the lower levels is below ``tol``
  (i.e. the operator neither maps population into the level from below nor lets
  its lower-level output depend on input at the level). Purely-diagonal high
  levels are therefore discarded.

This supports leakage-aware simulation in the low-leakage regime: most qudits
stay in the qubit subspace, so states and ideal gates squeeze back to dimension
2 while genuine leakage operators retain their higher levels.

Unlike :func:`quax.promote`, ``squeeze`` produces a *data-dependent* output
shape and therefore **cannot be jitted**.
"""

from functools import singledispatch
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

from ._quantum_objects import (
    Choi,
    KrausMap,
    Operator,
    PauliLiouville,
    QuantumInstrument,
    StateVector,
    SuperOp,
    Unitary,
)
from ._superoperator_transformations import (
    choi_to_kraus,
    kraus_to_choi,
    kraus_to_pauli_liouville,
    kraus_to_superop,
    pauli_liouville_to_kraus,
    superop_to_kraus,
)

#: Default tolerance below which a level's population (states) or coupling
#: (operators) is treated as negligible and the level is dropped.
DEFAULT_SQUEEZE_TOL = 1e-12


@singledispatch
def squeeze(obj, tol: float = DEFAULT_SQUEEZE_TOL):
    """Reduce each subsystem of a quantum object to its minimum dimension.

    The number of subsystems is preserved; only per-subsystem dimensions shrink.
    See the module docstring for the (tolerance-based, lossy) drop criterion.

    :param obj: A quax quantum object (StateVector, Unitary, Operator, SuperOp,
        KrausMap, Choi, or PauliLiouville).
    :param tol: Levels whose population (states) or coupling (operators) is below
        this threshold are dropped.
    :return: A new object of the same type on the squeezed dimensions.
    :raises TypeError: If *obj* is not a supported type.
    """
    raise TypeError(f"squeeze is not implemented for {type(obj).__name__}.")


# ---------------------------------------------------------------------------
# Keep-dimension helpers (eager / NumPy — squeeze is not jittable)
# ---------------------------------------------------------------------------


def _significant_keep(per_level: np.ndarray, tol: float, floor: int = 2) -> int:
    """Smallest dim that retains every level at or above *tol*.

    ``per_level`` is a 1-D array of per-level magnitudes; the result is
    ``1 + (highest index >= tol)``, then floored to *floor* (the qubit dimension,
    so a depopulated qutrit squeezes to a qubit rather than to a trivial 1-D
    space) and capped at the available number of levels.
    """
    significant = np.nonzero(per_level >= tol)[0]
    raw = int(significant[-1]) + 1 if significant.size else 1
    return min(len(per_level), max(raw, floor))


def _operator_keep_dims(absdata: np.ndarray, n: int, out0: int, tol: float) -> List[int]:
    """Per-subsystem keep-dims for an operator from its (output, input) axes.

    ``absdata`` is the elementwise magnitude of the operator tensor whose output
    qudit axes start at ``out0`` and whose input qudit axes start at ``out0 + n``.
    Any leading axes (ensemble, Kraus) are reduced over implicitly (the coupling
    test maxes over the whole remaining array). A level ``ℓ`` of subsystem ``i`` is
    *coupled* if the operator has an entry > ``tol`` mapping ``ℓ`` to/from a
    different level of ``i``; the keep-dim is one past the highest coupled level.
    """
    in0 = out0 + n
    keep: List[int] = []
    for i in range(n):
        oi, ii = out0 + i, in0 + i
        d_i = absdata.shape[oi]
        per_level = np.zeros(d_i)
        for ell in range(d_i):
            # maps into output level ℓ from a different input level
            a = np.take(absdata, ell, axis=oi)  # removes axis oi (< ii)
            a_off = np.delete(a, ell, axis=ii - 1)
            cpl_in = a_off.max() if a_off.size else 0.0
            # lower-level output depends on input level ℓ
            b = np.take(absdata, ell, axis=ii)  # removes axis ii (> oi, oi unchanged)
            b_off = np.delete(b, ell, axis=oi)
            cpl_out = b_off.max() if b_off.size else 0.0
            per_level[ell] = max(cpl_in, cpl_out)
        keep.append(_significant_keep(per_level, tol))
    return keep


def _slice_subsystems(data, n: int, group_starts: Tuple[int, ...], keep: List[int]):
    """Slice every per-subsystem axis group to ``keep`` dims.

    ``group_starts`` lists the first axis of each qudit-dimension group (e.g. the
    output and input groups of an operator); within a group, subsystem ``i`` is at
    ``start + i``.
    """
    slc: List[object] = [slice(None)] * data.ndim
    for start in group_starts:
        for i in range(n):
            slc[start + i] = slice(0, keep[i])
    return data[tuple(slc)]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@squeeze.register(StateVector)
def _squeeze_state_vector(psi: StateVector, tol: float = DEFAULT_SQUEEZE_TOL) -> StateVector:
    """Drop subsystem levels with population below *tol* and renormalize."""
    n = psi.num_qubits
    if n == 0:
        return psi
    dims = psi.dims
    ne = psi.num_ensemble_dims
    probs = jnp.abs(psi.data) ** 2  # (*ensemble, *dims), kept on-device

    # Reduce to per-subsystem, per-level populations on-device and transfer only the
    # (n, max_dim) table to the host — never the full state — so squeeze stays cheap for
    # large registers.
    max_dim = max(dims)
    rows = []
    for i in range(n):
        other = tuple(ne + j for j in range(n) if j != i)
        marginal = jnp.sum(probs, axis=other) if other else probs  # (*ensemble, d_i)
        per_level = jnp.max(marginal.reshape(-1, dims[i]), axis=0)  # worst case over ensemble
        rows.append(jnp.pad(per_level, (0, max_dim - dims[i])))
    populations = np.asarray(jnp.stack(rows))  # (n, max_dim)

    keep: List[int] = [_significant_keep(populations[i, : dims[i]], tol) for i in range(n)]

    if all(k == d for k, d in zip(keep, dims)):
        return psi

    subsystem_axes = tuple(range(ne, ne + n))
    new_data = _slice_subsystems(psi.data, n, (ne,), keep)
    norm = jnp.sqrt(jnp.sum(jnp.abs(new_data) ** 2, axis=subsystem_axes, keepdims=True))
    return StateVector(new_data / norm, num_qubits=n)


# ---------------------------------------------------------------------------
# Operators (matrix-backed): Unitary, Operator
# ---------------------------------------------------------------------------


def _squeeze_matrix_operator(obj, tol: float):
    """Squeeze a single-tensor operator (output axes then input axes)."""
    n = obj.num_qubits
    if n == 0:
        return obj
    ne = obj.num_ensemble_dims
    absdata = np.asarray(jnp.abs(obj.data))
    keep = _operator_keep_dims(absdata, n, ne, tol)
    if all(k == d for k, d in zip(keep, obj.dims[0])):
        return obj
    new_data = _slice_subsystems(obj.data, n, (ne, ne + n), keep)
    return type(obj)(new_data, num_qubits=n)


@squeeze.register(Unitary)
def _squeeze_unitary(unitary: Unitary, tol: float = DEFAULT_SQUEEZE_TOL) -> Unitary:
    """Drop decoupled high levels of a unitary (identity-on-level levels)."""
    return _squeeze_matrix_operator(unitary, tol)


@squeeze.register(Operator)
def _squeeze_operator(op: Operator, tol: float = DEFAULT_SQUEEZE_TOL) -> Operator:
    """Drop decoupled high levels of an operator."""
    return _squeeze_matrix_operator(op, tol)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


@squeeze.register(KrausMap)
def _squeeze_kraus_map(kraus: KrausMap, tol: float = DEFAULT_SQUEEZE_TOL) -> KrausMap:
    """Drop subsystem levels decoupled across all Kraus operators.

    The Kraus tensor is ``(*ensemble, n_kraus, *dims_out, *dims_in)``; the leading
    ensemble and Kraus axes are reduced over by the coupling test, so a level is
    kept if *any* Kraus operator couples it to the lower levels.
    """
    n = kraus.num_qubits
    if n == 0:
        return kraus
    out0 = kraus.num_ensemble_dims + 1  # skip the Kraus axis
    absdata = np.asarray(jnp.abs(kraus.data))
    keep = _operator_keep_dims(absdata, n, out0, tol)
    if all(k == d for k, d in zip(keep, kraus.dims[0])):
        return kraus
    new_data = _slice_subsystems(kraus.data, n, (out0, out0 + n), keep)
    return KrausMap(new_data, num_qubits=n)


@squeeze.register(SuperOp)
def _squeeze_superop(superop: SuperOp, tol: float = DEFAULT_SQUEEZE_TOL) -> SuperOp:
    """Squeeze via the Kraus representation (avoids the 4-group superop layout)."""
    return kraus_to_superop(_squeeze_kraus_map(superop_to_kraus(superop), tol))


@squeeze.register(Choi)
def _squeeze_choi(choi: Choi, tol: float = DEFAULT_SQUEEZE_TOL) -> Choi:
    """Squeeze via the Kraus representation."""
    return kraus_to_choi(_squeeze_kraus_map(choi_to_kraus(choi), tol))


@squeeze.register(PauliLiouville)
def _squeeze_pauli_liouville(pl: PauliLiouville, tol: float = DEFAULT_SQUEEZE_TOL) -> PauliLiouville:
    """Squeeze via the Kraus representation."""
    return kraus_to_pauli_liouville(_squeeze_kraus_map(pauli_liouville_to_kraus(pl), tol))


@squeeze.register(QuantumInstrument)
def _squeeze_quantum_instrument(inst: QuantumInstrument, tol: float = DEFAULT_SQUEEZE_TOL):
    """Squeezing a measurement instrument is not supported.

    A measurement's outcome space is tied to its measured dimension, so dropping a
    level would silently drop an outcome. Reconcile instruments by promoting them
    up to the state's dimension at apply time (see ``targeted_apply_*``) instead.
    """
    raise NotImplementedError(
        "squeeze is not defined for QuantumInstrument: squeezing a measurement would drop "
        "outcomes. Promote the instrument to the state's dimension at apply time instead."
    )
