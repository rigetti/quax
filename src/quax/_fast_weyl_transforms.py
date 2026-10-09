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
r"""Weyl and Hermitian coordinates by FFT, without building an operator basis.

Each :math:`\mathrm{Tr}[X^x Z^z A]` is a Fourier mode of one shifted diagonal of :math:`A`, so all
:math:`D^2` coefficients cost :math:`O(D^2 \log D)` time and :math:`O(D^2)` memory instead of the
:math:`O(D^4)` of the dense basis. Any tensor-product basis follows by one change of basis per
qudit (:func:`_hermitian_weyl_change`).
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
    r"""Flat index of :math:`i \pm j`, digit-wise, for all basis-state pairs.

    Let :math:`n` be the number of qudits, ``len(dims)``, with local dimensions
    :math:`d_0, \dots, d_{n-1}` and :math:`D = \prod_i d_i`. A basis state is a digit vector
    :math:`b \in \mathbb{Z}_{\mathbf{d}} = \mathbb{Z}_{d_0} \times \cdots \times \mathbb{Z}_{d_{n-1}}`
    with flat index :math:`\sum_i b_i s_i`, :math:`s_i = \prod_{j > i} d_j`, so qudit 0 is the most
    significant digit, as in ``np.kron`` and ``reshape(..., *dims)``. Addition in
    :math:`\mathbb{Z}_{\mathbf{d}}` is digit-wise with no carry: entry ``[i, j]`` of the result is
    the flat index of the digit vector with components :math:`(i_k + \mathrm{sign}\, j_k) \bmod d_k`.
    This is the addition under which the Weyl shift acts, :math:`X^x |b\rangle = |b + x\rangle`.

    For ``dims = (2, 3)``, :math:`i = (0, 2)` and :math:`j = (0, 1)` have flat indices 2 and 1, and
    their digit-wise sum is :math:`(0, 0)`, flat index 0, whereas adding flat indices modulo
    :math:`D` would carry into the qubit:

    >>> _shifted_flat((2, 3), 1)[2, 1]
    np.int64(0)
    >>> _shifted_flat((2, 3), -1)[0, 1]
    np.int64(2)

    The result is accumulated one qudit at a time in a single :math:`D \times D` buffer, so the
    peak memory is :math:`O(D^2)` rather than the :math:`O(D^2 n)` of a digit-vector table.

    :param dims: Local dimensions :math:`(d_0, \dots, d_{n-1})`.
    :param sign: ``1`` for :math:`i + j`, ``-1`` for :math:`i - j`.
    :returns: Integer array of shape ``(D, D)`` with values in :math:`\{0, \dots, D-1\}`.
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
    r"""Flat indices that gather a matrix's shifted diagonals, and put them back.

    Notation is as in :func:`_shifted_flat`. Since :math:`X^x |b\rangle = |b + x\rangle`, the
    Weyl coefficients :math:`\mathrm{Tr}[X^x Z^z A]` involve only the entries :math:`A_{b,\,b+x}`
    (see :func:`_weyl_coefficients`), which for fixed :math:`x` form the :math:`x`-th shifted
    diagonal of :math:`A`. With ``forward`` the flat index of :math:`(b, b + x)` laid out ``[x, b]``,
    ``A.reshape(D * D)[forward].reshape(D, D)`` has row :math:`x` equal to the :math:`x`-th shifted
    diagonal. The pairs :math:`(b, b + x)` run over every position exactly once, so the map is a
    permutation and its inverse is another gather: ``inverse`` is the flat index of
    :math:`(x, b) = (r - c, c)` laid out ``[r, c]``, and ``diagonals.reshape(D * D)[inverse]`` puts
    row :math:`x` back on the :math:`x`-th diagonal of a matrix whose entry :math:`(r, c)` is
    ``diagonals[r - c, c]``. Here :math:`r - c` is digit-wise, so the inverse pairs a shifted
    diagonal :math:`A_{b+x,\,b}` with row :math:`x`; this is the transpose of the ``forward``
    layout, which is what :func:`_from_weyl_expansion` needs.

    For two qubits, addition in :math:`\mathbb{Z}_{\mathbf{d}}` is bitwise XOR of flat indices and
    ``forward[x, b] = 4 b + (b ^ x)``, so row :math:`x = 0` is the main diagonal:

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

    Results are cached per ``dims``, which must therefore be a tuple. The returned arrays are
    shared between calls and must not be modified in place.

    :param dims: Local dimensions :math:`(d_0, \dots, d_{n-1})`.
    :returns: ``(forward, inverse)``, integer arrays of shape ``(D**2,)``, each a permutation of
        ``range(D**2)``.
    """
    total = prod(dims)
    forward = _shifted_flat(dims, 1)  # [x, b] -> b + x
    forward += np.arange(total)[None, :] * total  # flat index of (b, b + x)
    inverse = _shifted_flat(dims, -1)  # [r, c] -> r - c = x
    inverse *= total
    inverse += np.arange(total)[None, :]  # flat index of (x, c) in the [x, b] layout
    return forward.reshape(-1), inverse.reshape(-1)


@lru_cache(maxsize=32)
def _xz_order(dims: tuple[int, ...]) -> npt.NDArray[np.intp]:
    r"""Positions of the :func:`n_qudit_basis` operators in the ``[x, z]`` layout.

    :func:`n_qudit_basis` lists each qudit's Weyl operators in :func:`_xz_pairs` order
    and takes tensor products with qudit 0 most significant. :func:`_weyl_coefficients`
    instead lays out :math:`(x_0, \dots, x_{n-1}, z_0, \dots, z_{n-1})` row-major. Entry
    :math:`k` of the result is the flat position in that layout of basis element
    :math:`k`, so ``coefficients.reshape(*ensemble, -1)[..., _xz_order(dims)]`` is in
    basis order.

    For a qubit, :func:`_xz_pairs` is :math:`I, X, Y, Z = (0,0), (1,0), (1,1), (0,1)`,
    and the layout positions of those pairs are :math:`2x + z`:

    >>> _xz_order((2,))
    array([0, 2, 3, 1])

    :param dims: Local dimensions :math:`(d_0, \dots, d_{n-1})`.
    :returns: Integer array of shape ``(D**2,)``, a permutation of ``range(D**2)``.
    """
    # per qudit, (d**2, 2) in basis order
    pairs = [np.array(_xz_pairs(d)) for d in dims]
    # basis index per qudit
    grids = np.meshgrid(*[np.arange(len(p)) for p in pairs], indexing="ij")
    xs = [p[g, 0] for p, g in zip(pairs, grids)]
    zs = [p[g, 1] for p, g in zip(pairs, grids)]
    return np.asarray(np.ravel_multi_index((*xs, *zs), dims + dims)).reshape(-1)


def _weyl_coefficients(matrix: Array, dims: tuple[int, ...]) -> Array:
    r"""All Weyl coefficients :math:`\mathrm{Tr}[X^x Z^z A]` of a matrix, by FFT.

    Notation is as in :func:`_shifted_flat`. Since
    :math:`\langle b | X^x Z^z = \omega^{z \cdot (b - x)} \langle b - x |`,

    .. math::

        \mathrm{Tr}[X^x Z^z A] = \sum_{b \in \mathbb{Z}_{\mathbf{d}}} \omega^{z \cdot b}\, A_{b,\,b+x},

    an inverse DFT over :math:`b` of the :math:`x`-th shifted diagonal. All :math:`D^2`
    coefficients cost :math:`O(D^2 \log D)` time and :math:`O(D^2)` memory, against the
    :math:`O(D^4)` of contracting with an explicit basis. No symmetry of :math:`A` is
    assumed.

    :param matrix: Array of shape ``(*ensemble, D, D)``.
    :param dims: Local dimensions :math:`(d_0, \dots, d_{n-1})`.
    :returns: Complex array of shape ``(*ensemble, *dims, *dims)``, indexed ``[x, z]`` digit-wise.
        This is not the order of :func:`n_qudit_basis`; flatten the last ``2n`` axes and index with
        :func:`_xz_order` to get it.
    :raises ValueError: If ``matrix`` is not ``D x D`` for :math:`D = \prod_i d_i`.
    """
    # JAX clamps out-of-range gather indices instead of raising, so check the shape up
    # front.
    if matrix.shape[-2:] != (prod(dims),) * 2:
        raise ValueError(
            f"Expected a square {prod(dims)}x{prod(dims)} matrix for dims {dims}, got {matrix.shape[-2:]}."
        )
    d = prod(dims)
    ensemble = matrix.shape[:-2]
    forward, _ = _weyl_gather(dims)
    diagonals = jnp.take(matrix.reshape(*ensemble, d * d), forward, axis=-1)  # [x, b] = A[b, b + x]
    n = len(dims)
    # ifftn carries a 1/D that the trace does not.
    return jnp.fft.ifftn(diagonals.reshape(*ensemble, *dims, *dims), axes=tuple(range(-n, 0))) * d


def _from_weyl_expansion(expansion: Array, dims: tuple[int, ...]) -> Array:
    r"""The matrix :math:`\sum_{x,z} \gamma_{x,z}\, X^x Z^z`, by FFT.

    Notation is as in :func:`_weyl_coefficients`. Entry :math:`(b + x, b)` of the sum is
    :math:`\sum_z \gamma_{x,z}\, \omega^{z \cdot b}`, so each shifted diagonal is one
    inverse DFT over :math:`z`, put back in place by the inverse gather of
    :func:`_weyl_gather`. Since
    :math:`\mathrm{Tr}[(X^x Z^z)^\dagger X^{x'} Z^{z'}] = D\,\delta_{xx'}\delta_{zz'}`,
    the expansion of a matrix :math:`A` is
    :math:`\gamma_{x,z} = \mathrm{Tr}[(X^x Z^z)^\dagger A] / D`.

    :param expansion: Array of shape ``(*ensemble, *dims, *dims)``, indexed ``[x, z]`` digit-wise.
    :param dims: Local dimensions :math:`(d_0, \dots, d_{n-1})`.
    :returns: Complex array of shape ``(*ensemble, D, D)``.
    """
    n = len(dims)
    d = prod(dims)
    ensemble = expansion.shape[: -2 * n]
    diagonals = jnp.fft.ifftn(expansion, axes=tuple(range(-n, 0))) * d  # [x, b] = A[b + x, b]
    _, inverse = _weyl_gather(dims)
    return jnp.take(diagonals.reshape(*ensemble, d * d), inverse, axis=-1).reshape(*ensemble, d, d)


@lru_cache(maxsize=32)
def _hermitian_weyl_change(qudit_dim: int) -> npt.NDArray[np.complex128]:
    r"""Each element of :func:`hermitian_weyl_basis` in the Weyl operators
    :math:`X^x Z^z`.

    Returns :math:`a_{k,x,z}` with :math:`B_k = \sum_{x,z} a_{k,x,z} X^x Z^z`, computed
    as :math:`a_{k,x,z} = \mathrm{Tr}[(X^x Z^z)^\dagger B_k] / d`. Each :math:`B_k`
    combines a Weyl operator with its inverse, so a row has one or two non-zero entries.
    Deriving the change of basis from :func:`hermitian_weyl_basis` keeps that function
    the single definition of the basis.

    :param qudit_dim: Local qudit dimension :math:`d`.
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
    """Einsum subscripts: Weyl ``[x, z]`` axes, per-qudit changes, ``k`` axes."""
    xs, zs, ks = ascii_letters[:n], ascii_letters[n : 2 * n], ascii_letters[2 * n : 3 * n]
    return xs + zs, ",".join(k + x + z for k, x, z in zip(ks, xs, zs)), ks
