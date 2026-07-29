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
import itertools
from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import DensityMatrix, StateVector


def _format_state_vector_str(vec: Array, dims: tuple[int, ...], decimals: int, atol: float) -> str:
    """Format a single (non-ensembled) state vector as a Unicode Dirac-notation string."""

    def _fmt(x: float) -> str:
        rounded = round(x, decimals)
        if rounded == 0.0:
            return "0"
        return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")

    def _fmt_coeff(amp: complex) -> str | None:
        r_str = _fmt(amp.real)
        im_str = _fmt(amp.imag)
        if r_str == "0" and im_str == "0":
            return None
        elif im_str == "0":
            return r_str
        elif r_str == "0":
            return f"{im_str}i"
        else:
            sign = "+" if amp.imag > 0 else "-"
            return f"({r_str}{sign}{_fmt(abs(amp.imag))}i)"

    terms = []
    for idx, amp in zip(itertools.product(*[range(d) for d in dims]), vec.ravel()):
        if abs(amp) < atol:
            continue
        coeff = _fmt_coeff(complex(amp))
        if coeff is None:
            continue
        ket = "|" + "".join(str(i) for i in idx) + "\u27e9"
        terms.append((coeff, ket))

    if not terms:
        return "0"

    parts = []
    for i, (coeff, ket) in enumerate(terms):
        if i == 0:
            parts.append(f"{coeff}{ket}")
        else:
            if coeff.startswith("-"):
                parts.append(f" - {coeff[1:]}{ket}")
            else:
                parts.append(f" + {coeff}{ket}")

    return "".join(parts)


def _format_density_matrix_str(mat: Array, dims: tuple[int, ...], decimals: int, atol: float) -> str:
    """Format a single (non-ensembled) density matrix as a Unicode ``|i⟩⟨j|`` string."""

    def _fmt(x: float) -> str:
        rounded = round(x, decimals)
        if rounded == 0.0:
            return "0"
        return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")

    def _fmt_coeff(val: complex) -> str | None:
        r_str = _fmt(val.real)
        im_str = _fmt(val.imag)
        if r_str == "0" and im_str == "0":
            return None
        elif im_str == "0":
            return r_str
        elif r_str == "0":
            return f"{im_str}i"
        else:
            sign = "+" if val.imag > 0 else "-"
            return f"({r_str}{sign}{_fmt(abs(val.imag))}i)"

    indices = list(itertools.product(*[range(d) for d in dims]))
    terms = []
    for row_idx, bra in zip(indices, range(len(indices))):
        for col_idx, ket in zip(indices, range(len(indices))):
            val = complex(mat[bra, ket])
            if abs(val) < atol:
                continue
            coeff = _fmt_coeff(val)
            if coeff is None:
                continue
            row_label = "".join(str(i) for i in row_idx)
            col_label = "".join(str(i) for i in col_idx)
            basis = f"|{row_label}\u27e9\u27e8{col_label}|"
            terms.append((coeff, basis))

    if not terms:
        return "0"

    parts = []
    for i, (coeff, basis) in enumerate(terms):
        if i == 0:
            parts.append(f"{coeff}{basis}")
        else:
            if coeff.startswith("-"):
                parts.append(f" - {coeff[1:]}{basis}")
            else:
                parts.append(f" + {coeff}{basis}")

    return "".join(parts)


@jax.jit(static_argnames=("n_qubits", "dims", "ensemble_size"))
def zero_state_vector(
    n_qubits: int = 0,
    ensemble_size: tuple[int, ...] = (),
    dims: tuple[int, ...] | None = None,
) -> StateVector:
    """
    Construct a vector corresponding to ``|0>``.

    :param n_qubits: The number of qubits (ignored when *dims* is given).
    :param ensemble_size: The shape of the ensemble dimensions (default: no ensemble).
    :param dims: Per-subsystem dimensions, e.g. ``(3,)`` for a single qutrit.
        When supplied, *n_qubits* is ignored.
    :return: The state vector ``|000...0>`` for the given system.
    """
    if dims is None:
        dims = (2,) * n_qubits
    d = reduce(mul, dims, 1)
    state_matrix = jnp.zeros(ensemble_size + (d,), complex)
    state_matrix = state_matrix.at[..., 0].set(complex(1.0, 0))
    return StateVector.from_matrix(state_matrix, dims)


@jax.jit(static_argnames=("n_qubits", "dims", "ensemble_size"))
def zero_state_matrix(
    n_qubits: int = 0,
    ensemble_size: tuple[int, ...] = (),
    dims: tuple[int, ...] | None = None,
) -> DensityMatrix:
    """
    Construct a matrix corresponding to ``|0><0|``.

    :param n_qubits: The number of qubits (ignored when *dims* is given).
    :param ensemble_size: The shape of the ensemble dimensions (default: no ensemble).
    :param dims: Per-subsystem dimensions, e.g. ``(3,)`` for a single qutrit.
        When supplied, *n_qubits* is ignored.
    :return: The state matrix ``|000...0><000...0|`` for the given system.
    """
    if dims is None:
        dims = (2,) * n_qubits
    d = reduce(mul, dims, 1)
    state_matrix = jnp.zeros(ensemble_size + (d, d), complex)
    state_matrix = state_matrix.at[..., 0, 0].set(complex(1.0, 0))
    return DensityMatrix.from_matrix(state_matrix, dims)


@jax.jit(static_argnames=("n_qubits", "dims"))
def mixed_state_matrix(
    n_qubits: int = 0,
    dims: tuple[int, ...] | None = None,
) -> DensityMatrix:
    """
    Construct a matrix corresponding to the maximally mixed state.

    :param n_qubits: The number of qubits (ignored when *dims* is given).
    :param dims: Per-subsystem dimensions, e.g. ``(3,)`` for a single qutrit.
        When supplied, *n_qubits* is ignored.
    :return: The state matrix  ``I / d`` where ``d`` is the total dimension.
    """
    if dims is None:
        dims = (2,) * n_qubits
    d = reduce(mul, dims, 1)
    state_matrix = jnp.eye(d, dtype=complex) / d
    return DensityMatrix.from_matrix(state_matrix, dims)


@jax.jit
def tensor_state_vectors(state_a: StateVector, state_b: StateVector) -> StateVector:
    """
    Compute the tensor product of two state vectors.

    :param state_a: The first state vector.
    :param state_b: The second state vector.
    :return: The tensor product state vector.
    """
    new_data = jnp.kron(state_a.matrix, state_b.matrix)
    new_dims = state_a.dims + state_b.dims
    return StateVector.from_matrix(new_data, new_dims)


@jax.jit
def tensor_density_matrices(state_a: DensityMatrix, state_b: DensityMatrix) -> DensityMatrix:
    """
    Compute the tensor product of two density matrices.

    :param state_a: The first density matrix.
    :param state_b: The second density matrix.
    :return: The tensor product density matrix.
    """
    new_data = jnp.kron(state_a.matrix, state_b.matrix)
    new_dims = state_a.dims + state_b.dims
    return DensityMatrix.from_matrix(new_data, new_dims)
