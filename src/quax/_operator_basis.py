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

"""
Generalized operator bases for qudit systems.

This module defines the Weyl operator basis for
d-dimensional quantum systems (qudits).  For d=2 this coincides with the
Pauli basis.

The basis elements are normalized so that Tr[O_i^† O_j] = d δ_{ij},
which is the standard normalization used for the Pauli-Liouville
(or Weyl-Liouville) representation of superoperators.
"""

from functools import lru_cache
from math import prod
from typing import overload

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from ._quantum_objects import Observable, Unitary
from ._typing import ArrayT

# The basis builders below are pure functions of static integer dimensions, and their results are
# cached with ``lru_cache``.  They are deliberately built with NumPy rather than ``jax.numpy``: a
# builder may first be called from inside a ``jax.jit`` trace (e.g. ``jax.jit(qx.lindbladians.
# depolarizing)`` reaches ``n_qudit_herm_basis``), and ``jnp`` operations under a trace produce
# tracers even when their inputs are static.  ``lru_cache`` would then store a tracer that escapes
# its trace and raises ``UnexpectedTracerError`` on the next (non-traced) call.  NumPy always
# evaluates eagerly to concrete host arrays, so the cached constants are trace-independent; the
# single ``jnp`` conversion inside ``from_matrix`` bakes them in as constants.


@overload
def _batched_kron(a: npt.NDArray[np.number], b: npt.NDArray[np.number]) -> npt.NDArray[np.complex128]: ...
@overload
def _batched_kron(a: Array, b: Array) -> Array: ...
def _batched_kron(a: ArrayT, b: ArrayT) -> ArrayT:
    """Batched Kronecker product of two ensembles of square matrices.

    Operates elementwise, so it stays on whatever array type it is given (the basis builders
    pass concrete NumPy arrays; see the module comment above).

    Given *a* with shape ``(n, p, p)`` and *b* with shape ``(m, q, q)``,
    returns the ``n * m`` pairwise Kronecker products with shape
    ``(n * m, p * q, p * q)``.
    """
    n, p, _ = a.shape
    m, q, _ = b.shape
    # Outer product over ensemble dims; kronecker over matrix dims
    product = a[:, None, :, None, :, None] * b[None, :, None, :, None, :]

    # (n, m, p, q, p, q) -> (n*m, p*q, p*q)
    reshaped = product.reshape(n * m, p * q, p * q)

    # The result is always complex128. ``copy=False`` returns the product itself when it already is,
    # instead of a second full-size copy, which halves peak memory; other dtypes are cast as before.
    return reshaped.astype(complex, copy=False)


@lru_cache(maxsize=32)
def _xz_pairs(d: int) -> list[tuple[int, int]]:
    """
    Return the (x, z) ordering for single-qudit Weyl operators.

    For d=2: I, X, Y, Z order → (0,0), (1,0), (1,1), (0,1).
    For d>2: identity first, then operators grouped so that inverses are
    adjacent.
    """
    if d == 2:
        return [(0, 0), (1, 0), (1, 1), (0, 1)]

    pairs: list[tuple[int, int]] = [(0, 0)]
    for i in range(1, d // 2 + 1):
        j = d - i
        if i == j:
            pairs.extend([(0, i), (i, i), (i, 0)])
        else:
            pairs.extend([(0, i), (0, j), (i, i), (j, j), (i, 0), (j, 0)])
    for x in range(1, d):
        for z in range(1, x):
            pairs.append((x, z))
            pairs.append(((d - 1) * x % d, (d - 1) * z % d))
    return pairs


@lru_cache(maxsize=32)
def weyl_basis_labels(qudit_dim: int) -> list[str]:
    """
    Return labels for the Weyl-Heisenberg operator basis of a single qudit.

    :param qudit_dim: Local qudit dimension d.
    :return: List of d² string labels of the form ``"Wxz"``.
    """
    return [f"W{x}{z}" for x, z in _xz_pairs(qudit_dim)]


@lru_cache(maxsize=32)
def weyl_basis(qudit_dim: int) -> Unitary:
    """
    Generate the Weyl-Heisenberg operator basis for a single qudit of
    dimension *qudit_dim*.

    For d=2, operators are the Hermitian Pauli matrices {I, X, Y, Z}.
    For d>2, operators are the plain Weyl unitaries W_{x,z} = X^x Z^z.

    The returned operators satisfy:

    * The first element is the d×d identity.
    * Tr[W_i^† W_j] = d δ_{ij}  (trace-orthogonality).
    * There are d² operators in total, forming a basis for all d×d matrices.

    :param qudit_dim: Local qudit dimension d.
    :return: Ensemble of d² Weyl basis operators as a :class:`Unitary`.
    """
    if qudit_dim == 2:
        # For d=2, the Pauli matrices are both unitary and Hermitian
        return Unitary.from_matrix(hermitian_weyl_basis(qudit_dim).matrix, ((qudit_dim,), (qudit_dim,)))

    # d > 2: plain Weyl unitaries W_{x,z} = X^x Z^z
    root_of_unity = np.exp(2j * np.pi / qudit_dim)
    pairs = np.array(_xz_pairs(qudit_dim))  # (d², 2)
    x_vals, z_vals = pairs[:, 0], pairs[:, 1]

    # W_{x,z}[i, j] = omega^{jz} * delta(i, (j + x) mod d)
    j = np.arange(qudit_dim)
    row_idx = (j[None, :] + x_vals[:, None]) % qudit_dim  # (d², d)
    clock_vals = root_of_unity ** (j[None, :] * z_vals[:, None])  # (d², d)

    i = np.arange(qudit_dim)
    shift_mask = i[None, :, None] == row_idx[:, None, :]  # (d², d, d)
    weyl_ops = (clock_vals[:, None, :] * shift_mask).astype(complex)

    return Unitary.from_matrix(weyl_ops, ((qudit_dim,), (qudit_dim,)))


def hermitian_weyl_basis_labels(qudit_dim: int) -> list[str]:
    """
    Return labels for the Hermitian Weyl-Heisenberg operator basis of a
    single qudit.

    :param qudit_dim: Local qudit dimension d.
    :return: List of d² string labels of the form ``"Wxz"``.
    """
    return weyl_basis_labels(qudit_dim)


@lru_cache(maxsize=32)
def hermitian_weyl_basis(qudit_dim: int) -> Observable:
    """
    Generate the Hermitian Weyl-Heisenberg operator basis for a single qudit.

    For all d (including d=2), operators are the Hermitian combinations of
    Weyl unitaries.

    :param qudit_dim: Local qudit dimension d.
    :return: Ensemble of d² Hermitian basis operators as an :class:`Observable`.
    """
    root_of_unity = np.exp(2j * np.pi / qudit_dim)
    pairs = np.array(_xz_pairs(qudit_dim))  # (d², 2)
    x_vals, z_vals = pairs[:, 0], pairs[:, 1]

    # Inverse pair indices: W_{x,z}^{-1} = W_{-x,-z}
    inv_x = (qudit_dim - 1) * x_vals % qudit_dim
    inv_z = (qudit_dim - 1) * z_vals % qudit_dim

    # Build all phased Weyl operators: phase(x,z) * W_{x,z}
    # W_{x,z}[i, j] = omega^{jz} * delta(i, (j + x) mod d)
    j = np.arange(qudit_dim)
    row_idx = (j[None, :] + x_vals[:, None]) % qudit_dim  # (d², d)
    clock_vals = root_of_unity ** (j[None, :] * z_vals[:, None])  # (d², d)
    phase_prefactor = root_of_unity ** ((x_vals * z_vals % qudit_dim) / 2)  # (d²,)

    i = np.arange(qudit_dim)
    shift_mask = i[None, :, None] == row_idx[:, None, :]  # (d², d, d)
    weyl_ops = (phase_prefactor[:, None, None] * clock_vals[:, None, :] * shift_mask).astype(complex)

    # Hermitian combinations: (W + W†) if (x,z) >= (x',z'), else i(W - W†)
    weyl_dag = weyl_ops.conj().transpose(0, 2, 1)
    is_geq = (x_vals * qudit_dim + z_vals) >= (inv_x * qudit_dim + inv_z)
    hermitian_ops = np.where(
        is_geq[:, None, None],
        weyl_ops + weyl_dag,
        1j * (weyl_ops - weyl_dag),
    )

    # Normalize so that Tr[O_i† O_j] = d δ_{ij}
    frob_sq = np.sum(np.abs(hermitian_ops) ** 2, axis=(-2, -1), keepdims=True)
    hermitian_ops = hermitian_ops * np.sqrt(qudit_dim / frob_sq)

    return Observable.from_matrix(hermitian_ops, ((qudit_dim,), (qudit_dim,)))


@lru_cache(maxsize=32)
def n_qudit_herm_basis(dims: tuple[int, ...]) -> Observable:
    """
    Construct the tensor product Hermitian operator basis for a composite
    qudit system.

    :param dims: Tuple of per-qudit dimensions.
    :return: Ensemble of d_total² Hermitian basis operators as an :class:`Observable`.
    """
    if len(dims) == 0:
        raise ValueError("dims must have at least one element.")

    first = hermitian_weyl_basis(dims[0])

    if len(dims) == 1:
        return first

    rest = n_qudit_herm_basis(dims[1:])
    ops = _batched_kron(first.matrix, rest.matrix)
    return Observable.from_matrix(ops, (dims, dims))


@lru_cache(maxsize=32)
def n_qudit_basis(dims: tuple[int, ...]) -> Unitary:
    """
    Construct the tensor product operator basis for a composite system of
    qudits with the given dimensions.

    :param dims: Tuple of per-qudit dimensions, e.g. ``(3,)`` for a single
        qutrit or ``(2, 3)`` for a qubit-qutrit system.
    :return: Ensemble of d_total² basis operators as a :class:`Unitary`.
    """
    if len(dims) == 0:
        raise ValueError("dims must have at least one element.")

    first = weyl_basis(dims[0])

    if len(dims) == 1:
        return first

    rest = n_qudit_basis(dims[1:])
    ops = _batched_kron(first.matrix, rest.matrix)
    return Unitary.from_matrix(ops, (dims, dims))


# ---------- index arithmetic and fast Weyl transforms ----------


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
