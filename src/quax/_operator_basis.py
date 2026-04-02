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
from typing import List, Tuple

import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Observable, Unitary


def _batched_kron(a: Array, b: Array) -> Array:
    """Batched Kronecker product of two ensembles of square matrices.

    Given *a* with shape ``(n, p, p)`` and *b* with shape ``(m, q, q)``,
    returns the ``n * m`` pairwise Kronecker products with shape
    ``(n * m, p * q, p * q)``.
    """
    n, p, _ = a.shape
    m, q, _ = b.shape
    # outer product over ensemble dims, kronecker over matrix dims
    # a[:, None, :, None, :, None] * b[None, :, None, :, None, :]
    # -> (n, m, p, q, p, q) -> (n*m, p*q, p*q)
    result = (a[:, None, :, None, :, None] * b[None, :, None, :, None, :]).reshape(n * m, p * q, p * q)
    return result.astype(complex)


@lru_cache(maxsize=32)
def _xz_pairs(d: int) -> List[Tuple[int, int]]:
    """
    Return the (x, z) ordering for single-qudit Weyl operators.

    For d=2: I, X, Y, Z order → (0,0), (1,0), (1,1), (0,1).
    For d>2: identity first, then operators grouped so that inverses are
    adjacent.
    """
    if d == 2:
        return [(0, 0), (1, 0), (1, 1), (0, 1)]

    pairs: List[Tuple[int, int]] = [(0, 0)]
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
def weyl_basis_labels(qudit_dim: int) -> List[str]:
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
    root_of_unity = jnp.exp(2j * jnp.pi / qudit_dim)
    pairs = jnp.array(_xz_pairs(qudit_dim))  # (d², 2)
    x_vals, z_vals = pairs[:, 0], pairs[:, 1]

    # W_{x,z}[i, j] = omega^{jz} * delta(i, (j + x) mod d)
    j = jnp.arange(qudit_dim)
    row_idx = (j[None, :] + x_vals[:, None]) % qudit_dim  # (d², d)
    clock_vals = root_of_unity ** (j[None, :] * z_vals[:, None])  # (d², d)

    i = jnp.arange(qudit_dim)
    shift_mask = i[None, :, None] == row_idx[:, None, :]  # (d², d, d)
    weyl_ops = (clock_vals[:, None, :] * shift_mask).astype(complex)

    return Unitary.from_matrix(weyl_ops, ((qudit_dim,), (qudit_dim,)))


def hermitian_weyl_basis_labels(qudit_dim: int) -> List[str]:
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
    root_of_unity = jnp.exp(2j * jnp.pi / qudit_dim)
    pairs = jnp.array(_xz_pairs(qudit_dim))  # (d², 2)
    x_vals, z_vals = pairs[:, 0], pairs[:, 1]

    # Inverse pair indices: W_{x,z}^{-1} = W_{-x,-z}
    inv_x = (qudit_dim - 1) * x_vals % qudit_dim
    inv_z = (qudit_dim - 1) * z_vals % qudit_dim

    # Build all phased Weyl operators: phase(x,z) * W_{x,z}
    # W_{x,z}[i, j] = omega^{jz} * delta(i, (j + x) mod d)
    j = jnp.arange(qudit_dim)
    row_idx = (j[None, :] + x_vals[:, None]) % qudit_dim  # (d², d)
    clock_vals = root_of_unity ** (j[None, :] * z_vals[:, None])  # (d², d)
    phase_prefactor = root_of_unity ** ((x_vals * z_vals % qudit_dim) / 2)  # (d²,)

    i = jnp.arange(qudit_dim)
    shift_mask = i[None, :, None] == row_idx[:, None, :]  # (d², d, d)
    weyl_ops = (phase_prefactor[:, None, None] * clock_vals[:, None, :] * shift_mask).astype(complex)

    # Hermitian combinations: (W + W†) if (x,z) >= (x',z'), else i(W - W†)
    weyl_dag = weyl_ops.conj().transpose(0, 2, 1)
    is_geq = (x_vals * qudit_dim + z_vals) >= (inv_x * qudit_dim + inv_z)
    hermitian_ops = jnp.where(
        is_geq[:, None, None],
        weyl_ops + weyl_dag,
        1j * (weyl_ops - weyl_dag),
    )

    # Normalize so that Tr[O_i† O_j] = d δ_{ij}
    frob_sq = jnp.sum(jnp.abs(hermitian_ops) ** 2, axis=(-2, -1), keepdims=True)
    hermitian_ops = hermitian_ops * jnp.sqrt(qudit_dim / frob_sq)

    return Observable.from_matrix(hermitian_ops, ((qudit_dim,), (qudit_dim,)))


@lru_cache(maxsize=32)
def n_qudit_herm_basis(dims: Tuple[int, ...]) -> Observable:
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
def n_qudit_basis(dims: Tuple[int, ...]) -> Unitary:
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
