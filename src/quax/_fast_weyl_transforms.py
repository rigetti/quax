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
"""Weyl and Hermitian coordinates of a matrix, without building the bases.

The idea: ``X`` shifts the computational basis and ``Z`` multiplies it by phases. So the trace
``Tr[X^x Z^z A]`` only touches the entries of ``A`` that sit ``x`` steps off the diagonal, and
summing them with the phases of ``Z^z`` is a discrete Fourier transform. Every Weyl coordinate of
``A`` is therefore reached by gathering one shifted diagonal (the ``X`` part) and running an FFT
along it (the ``Z`` part). Index bookkeeping holds the two together: qudits add digit by digit,
with no carry between them, and the module's first job is to build those indices once per
``dims`` and cache them.

The functions build on each other in this order.

* :func:`_shifted_flat` is the index arithmetic: where basis state ``i`` lands after a digit-wise
  shift by ``j``.
* :func:`_weyl_gather` turns that into one flat index that pulls every shifted diagonal out of a
  matrix, and the inverse index that puts them back.
* :func:`_weyl_coefficients` gathers and FFTs: all ``Tr[X^x Z^z A]`` at once.
  :func:`_from_weyl_expansion` is its inverse: the matrix with given Weyl coordinates.
* :func:`_xz_order` reorders those coordinates into the order of :func:`~quax.n_qudit_basis`.
* :func:`_hermitian_weyl_change` expresses each element of :func:`~quax.hermitian_weyl_basis` in
  the Weyl operators of one qudit, so :func:`_hermitian_coefficients` and
  :func:`_from_hermitian_coefficients` reach the Hermitian (Pauli) coordinates from the Weyl ones
  with one small contraction per qudit.

Outside this module, :func:`~quax.to_pauli_vector` and ``from_pauli_vector`` are the Hermitian
pair and :func:`~quax.characteristic_function` is the Weyl one. For ``n`` qubits the dense basis
holds ``16**n`` entries and contracting with it costs ``O(D**4)``; the FFT route costs
``O(D**2 log D)`` time and ``O(D**2)`` memory, which is what makes 8 to 11 qubits reachable.
"""

from functools import lru_cache
from math import prod
from string import ascii_letters

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from ._operator_basis import _xz_pairs, hermitian_weyl_basis


def _shifted_flat(dims: tuple[int, ...], sign: int) -> npt.NDArray[np.intp]:
    """Where basis state ``i`` lands after a digit-wise shift by ``+j`` or ``-j``.

    A computational basis state of qudits with sizes ``dims`` is a tuple of digits, one per
    qudit, with qudit 0 most significant (the ``np.kron`` order). ``X`` on one qudit adds one to
    that qudit's digit modulo its size, and never carries into the next qudit. Entry ``[i, j]`` of
    the result is the flat index of the state whose digits are ``i``'s digits plus ``sign`` times
    ``j``'s digits, each taken modulo its own size.

    For ``dims = (2, 3)`` the states ``2 = (0, 2)`` and ``1 = (0, 1)`` add to ``(0, 0) = 0``,
    not to ``3``, because the qutrit digit wraps instead of carrying into the qubit:

    >>> _shifted_flat((2, 3), 1)[2, 1]
    np.int64(0)
    >>> _shifted_flat((2, 3), -1)[0, 1]
    np.int64(2)

    The table is accumulated one qudit at a time in a single ``(D, D)`` buffer, so building it
    needs ``O(D**2)`` memory and no table of digit tuples.

    :param dims: Qudit sizes, e.g. ``(2, 3)``; ``D`` is their product.
    :param sign: ``1`` to shift forward (``i + j``), ``-1`` to shift back (``i - j``).
    :returns: Integer array of shape ``(D, D)``, entries in ``range(D)``.
    """
    total = prod(dims)
    out = np.zeros((total, total), dtype=np.intp)
    term = np.empty_like(out)
    flat = np.arange(total)
    stride = 1
    for d in reversed(dims):
        digit = flat // stride % d
        np.add.outer(digit, sign * digit, out=term)
        term %= d
        term *= stride
        out += term
        stride *= d
    return out


@lru_cache(maxsize=32)
def _weyl_gather(dims: tuple[int, ...]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Flat indices that pull the shifted diagonals out of a matrix, and put them back.

    Row ``x`` of ``A.reshape(D * D)[forward].reshape(D, D)`` is the ``x``-th shifted diagonal of
    ``A``: its entry ``b`` is ``A[b, b + x]``, with ``b + x`` the digit-wise shift of
    :func:`_shifted_flat`. Those are exactly the entries that ``Tr[X^x Z^z A]`` sums over, for every
    ``z``. Every entry of ``A`` appears in exactly one shifted diagonal, so ``forward`` is a
    permutation, and ``inverse`` undoes it: ``diagonals.reshape(D * D)[inverse].reshape(D, D)``
    has entry ``[r, c]`` equal to ``diagonals[r - c, c]``. Note the transpose: ``inverse`` places
    row ``x`` on the diagonal ``A[b + x, b]``, which is the one :func:`_from_weyl_expansion`
    produces.

    For two qubits a digit-wise shift is a bitwise XOR, so ``forward[x, b] = 4 * b + (b ^ x)`` and
    row 0 is the main diagonal:

    >>> forward, inverse = _weyl_gather((2, 2))
    >>> forward.reshape(4, 4)
    array([[ 0,  5, 10, 15],
           [ 1,  4, 11, 14],
           [ 2,  7,  8, 13],
           [ 3,  6,  9, 12]])
    >>> inverse.reshape(4, 4)
    array([[ 0,  5, 10, 15],
           [ 4,  1, 14, 11],
           [ 8, 13,  2,  7],
           [12,  9,  6,  3]])

    Cached per ``dims``, which must be a tuple. The arrays are shared between calls; do not
    modify them in place.

    :param dims: Qudit sizes; ``D`` is their product.
    :returns: ``(forward, inverse)``, integer arrays of shape ``(D**2,)``, each a permutation of
        ``range(D**2)``.
    """
    total = prod(dims)
    forward = _shifted_flat(dims, 1)  # [x, b] -> b + x
    forward += np.arange(total)[None, :] * total  # flat index of A[b, b + x]
    inverse = _shifted_flat(dims, -1)  # [r, c] -> r - c = x
    inverse *= total
    inverse += np.arange(total)[None, :]  # flat index of diagonals[x, c]
    return forward.reshape(-1), inverse.reshape(-1)


def _weyl_coefficients(matrix: Array, dims: tuple[int, ...]) -> Array:
    """All ``Tr[X^x Z^z A]`` of a matrix ``A``, with no phase on ``X^x Z^z``.

    ``X^x`` picks the ``x``-th shifted diagonal of ``A`` (gathered with :func:`_weyl_gather`) and
    ``Z^z`` weights its entries by the phases ``omega**(z . b)``, so each coordinate is one term of
    an inverse discrete Fourier transform of that diagonal. One ``ifftn`` over the ``b`` axes gives
    every ``z`` at once; its built-in ``1/D`` is undone, since a trace has none.

    The result is laid out by digits, ``[x_0, ..., x_n-1, z_0, ..., z_n-1]``, which is the FFT's
    natural order and not the order of :func:`~quax.n_qudit_basis`; :func:`_xz_order` converts.
    The operators here are the bare ``X^x Z^z``; the phase convention of
    :func:`~quax.characteristic_function` is applied by its caller.

    :param matrix: Shape ``(*ensemble, D, D)``; any square matrix, no symmetry assumed.
    :param dims: Qudit sizes; ``D`` is their product.
    :returns: Complex array of shape ``(*ensemble, *dims, *dims)``.
    :raises ValueError: If the matrix is not ``D x D``.
    """
    # JAX clamps out-of-range gather indices instead of raising, so check the shape up front.
    if matrix.shape[-2:] != (prod(dims),) * 2:
        raise ValueError(
            f"Expected a square {prod(dims)}x{prod(dims)} matrix for dims {dims}, got {matrix.shape[-2:]}."
        )
    d = prod(dims)
    ensemble = matrix.shape[:-2]
    forward, _ = _weyl_gather(dims)
    diagonals = jnp.take(matrix.reshape(*ensemble, d * d), forward, axis=-1)  # [x, b] = A[b, b + x]
    n = len(dims)
    return jnp.fft.ifftn(diagonals.reshape(*ensemble, *dims, *dims), axes=tuple(range(-n, 0))) * d


def _from_weyl_expansion(expansion: Array, dims: tuple[int, ...]) -> Array:
    """The matrix ``sum_{x,z} expansion[x, z] X^x Z^z``.

    This inverts :func:`_weyl_coefficients`. Running the same FFT over the ``z`` axes turns each row ``x`` of the expansion into the
    ``x``-th shifted diagonal of the result, and the inverse index of :func:`_weyl_gather` puts the
    diagonals back into a matrix. Given a matrix ``A``, its expansion coefficients are
    ``Tr[(X^x Z^z)^dagger A] / D``, because the Weyl operators are orthogonal with
    ``Tr[W^dagger W] = D``.

    :param expansion: Shape ``(*ensemble, *dims, *dims)``, laid out by digits like the output of
        :func:`_weyl_coefficients`.
    :param dims: Qudit sizes; ``D`` is their product.
    :returns: Complex array of shape ``(*ensemble, D, D)``.
    """
    n = len(dims)
    d = prod(dims)
    ensemble = expansion.shape[: -2 * n]
    diagonals = jnp.fft.ifftn(expansion, axes=tuple(range(-n, 0))) * d  # [x, b] = A[b + x, b]
    _, inverse = _weyl_gather(dims)
    return jnp.take(diagonals.reshape(*ensemble, d * d), inverse, axis=-1).reshape(*ensemble, d, d)


@lru_cache(maxsize=32)
def _xz_order(dims: tuple[int, ...]) -> npt.NDArray[np.intp]:
    """Where each operator of :func:`~quax.n_qudit_basis` sits in the digit layout.

    :func:`~quax.n_qudit_basis` lists each qudit's operators in :func:`_xz_pairs` order (``I, X,
    Y, Z`` for a qubit) and tensors them with qudit 0 most significant. Flattening the
    ``[x_0, ..., z_n-1]`` layout to one axis and indexing it with this permutation gives the
    coefficients in basis order.

    For one qubit the pairs ``(0,0), (1,0), (1,1), (0,1)`` sit at positions ``2 * x + z``:

    >>> _xz_order((2,))
    array([0, 2, 3, 1])

    :param dims: Qudit sizes; ``D`` is their product.
    :returns: Integer array of shape ``(D**2,)``, a permutation of ``range(D**2)``.
    """
    pairs = [np.array(_xz_pairs(d)) for d in dims]  # per qudit, (d**2, 2) in basis order
    grids = np.meshgrid(*[np.arange(len(p)) for p in pairs], indexing="ij")  # basis index per qudit
    xs = [p[g, 0] for p, g in zip(pairs, grids)]
    zs = [p[g, 1] for p, g in zip(pairs, grids)]
    return np.asarray(np.ravel_multi_index((*xs, *zs), dims + dims)).reshape(-1)


@lru_cache(maxsize=32)
def _hermitian_weyl_change(qudit_dim: int) -> npt.NDArray[np.complex128]:
    """Each element of :func:`~quax.hermitian_weyl_basis` written in ``X^x Z^z``.

    Entry ``[k, x, z]`` is the coefficient of ``X^x Z^z`` in the ``k``-th Hermitian basis element,
    ``Tr[(X^x Z^z)^dagger B_k] / d``. Each ``B_k`` is a Weyl operator plus or minus its adjoint,
    so each ``k`` has one or two non-zero entries. Reading the coefficients off
    :func:`~quax.hermitian_weyl_basis` keeps that function the single definition of the basis.

    :param qudit_dim: Size ``d`` of the qudit.
    :returns: Complex array of shape ``(d**2, d, d)``, indexed ``[k, x, z]``.
    """
    d = qudit_dim
    shift = np.roll(np.eye(d), 1, axis=0)
    clock = np.diag(np.exp(2j * np.pi * np.arange(d) / d))
    weyl = np.array(
        [np.linalg.matrix_power(shift, x) @ np.linalg.matrix_power(clock, z) for x in range(d) for z in range(d)]
    )
    with jax.ensure_compile_time_eval():
        hermitian = np.asarray(hermitian_weyl_basis(d).matrix)
    return (np.einsum("wji,kji->kw", weyl.conj(), hermitian) / d).reshape(d * d, d, d)


def _change_of_basis_spec(n: int) -> tuple[str, str, str]:
    """Einsum subscripts for changing basis one qudit at a time over ``n`` qudits.

    Returns the subscripts of the Weyl axes ``x_0 .. x_n-1 z_0 .. z_n-1``, of the ``n`` per-qudit
    change-of-basis factors ``[k_i, x_i, z_i]``, and of the Hermitian axes ``k_0 .. k_n-1``.
    """
    xs, zs, ks = ascii_letters[:n], ascii_letters[n : 2 * n], ascii_letters[2 * n : 3 * n]
    return xs + zs, ",".join(k + x + z for k, x, z in zip(ks, xs, zs)), ks


def _hermitian_coefficients(matrix: Array, dims: tuple[int, ...]) -> Array:
    """``Tr[B_k A]`` for every element of :func:`~quax.n_qudit_herm_basis`, in order.

    These are the Pauli-vector coordinates of ``A``. They come from the Weyl coordinates of
    :func:`_weyl_coefficients` by one contraction with :func:`_hermitian_weyl_change` per qudit,
    so neither the dense basis nor a ``D**2 x D**2`` change-of-basis matrix is built.

    :param matrix: Shape ``(*ensemble, D, D)``.
    :param dims: Qudit sizes; ``D`` is their product.
    :returns: Real array of shape ``(*ensemble, D**2)``.
    """
    weyl, changes, ks = _change_of_basis_spec(len(dims))
    coefficients = jnp.einsum(
        f"...{weyl},{changes}->...{ks}", _weyl_coefficients(matrix, dims), *map(_hermitian_weyl_change, dims)
    )
    return jnp.real(coefficients.reshape(*matrix.shape[:-2], -1))


def _from_hermitian_coefficients(coefficients: Array, dims: tuple[int, ...]) -> Array:
    """The matrix ``(1/D) sum_k coefficients[k] B_k``.

    This inverts :func:`_hermitian_coefficients`. Each ``B_k`` is a combination of Weyl operators, so the coefficients are first turned into a
    Weyl expansion with :func:`_hermitian_weyl_change`, one qudit at a time, and
    :func:`_from_weyl_expansion` builds the matrix.

    :param coefficients: Shape ``(*ensemble, D**2)``, in the order of :func:`~quax.n_qudit_herm_basis`.
    :param dims: Qudit sizes; ``D`` is their product.
    :returns: Complex array of shape ``(*ensemble, D, D)``.
    """
    d = prod(dims)
    weyl, changes, ks = _change_of_basis_spec(len(dims))
    per_qudit = coefficients.astype(complex).reshape(*coefficients.shape[:-1], *(k * k for k in dims))
    expansion = jnp.einsum(f"...{ks},{changes}->...{weyl}", per_qudit, *map(_hermitian_weyl_change, dims)) / d
    return _from_weyl_expansion(expansion, dims)
