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

This module defines the Weyl (generalized Gell-Mann) operator basis for
d-dimensional quantum systems (qudits).  For d=2 this coincides with the
Pauli basis.

The basis elements are normalized so that Tr[O_i^† O_j] = d δ_{ij},
which is the standard normalization used for the Pauli-Liouville
(or Weyl-Liouville) representation of superoperators.
"""

from functools import lru_cache
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


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
        pairs.extend([(0, i), (0, j), (i, i), (j, j), (i, 0), (j, 0)])
    for x in range(1, d):
        for z in range(1, x):
            pairs.append((x, z))
            pairs.append(((d - 1) * x % d, (d - 1) * z % d))
    return pairs


@lru_cache(maxsize=32)
def qudit_operator_basis(qudit_dim: int) -> Tuple[List[str], List[NDArray[np.complex128]]]:
    """
    Generate the Weyl-Heisenberg operator basis for a single qudit of
    dimension *qudit_dim*.

    For d=2, operators are the Hermitian Pauli matrices {I, X, Y, Z}
    For d>2, operators are the plain Weyl unitaries W_{x,z} = X^x Z^z.

    The returned operators satisfy:

    * The first element is the d×d identity.
    * Tr[W_i^† W_j] = d δ_{ij}  (trace-orthogonality).
    * There are d² operators in total, forming a basis for all d×d matrices.

    Returns numpy arrays so the cache is safe to use inside JAX JIT traces.

    :param qudit_dim: Local qudit dimension d.
    :return: Tuple of (labels, operators) where labels is a list of strings
        and operators is a list of d×d numpy arrays.
    """
    d = qudit_dim
    omega = np.exp(2j * np.pi / d)
    phases = omega ** np.arange(d)
    pairs = _xz_pairs(d)

    ops: List[NDArray[np.complex128]] = []
    labels: List[str] = []

    if d == 2:
        # Hermitian Weyl basis
        for x, z in pairs:
            xzi = ((d - 1) * x % d, (d - 1) * z % d)
            coeff = omega ** ((x * z % d) / 2) * np.sqrt(0.25 if (x, z) == xzi else 0.5)
            mat = np.roll(coeff * np.diag(phases**z), x, axis=0).astype(np.complex128)
            if (x, z) >= xzi:
                mat = mat + mat.T.conj()
            else:
                mat = 1j * (mat - mat.T.conj())
            ops.append(mat)
            labels.append(f"W{x}{z}")
    else:
        # Plain Weyl unitaries W_{x,z} = X^x Z^z
        for x, z in pairs:
            mat = np.roll(np.diag(phases**z), x, axis=0).astype(np.complex128)
            ops.append(mat)
            labels.append(f"W{x}{z}")

    return labels, ops


@lru_cache(maxsize=32)
def qudit_herm_operator_basis(
    qudit_dim: int,
) -> Tuple[List[str], List[NDArray[np.complex128]]]:
    """
    Generate the Hermitian Weyl-Heisenberg operator basis for a single qudit.

    For all d (including d=2), operators are the Hermitian combinations of
    Weyl unitaries following the TrueQ ``herm_mat`` convention.

    :param qudit_dim: Local qudit dimension d.
    :return: Tuple of (labels, operators).
    """
    d = qudit_dim
    omega = np.exp(2j * np.pi / d)
    phases = omega ** np.arange(d)
    pairs = _xz_pairs(d)

    ops: List[NDArray[np.complex128]] = []
    labels: List[str] = []

    for x, z in pairs:
        xzi = ((d - 1) * x % d, (d - 1) * z % d)
        coeff = omega ** ((x * z % d) / 2) * np.sqrt(0.25 if (x, z) == xzi else 0.5)
        mat = np.roll(coeff * np.diag(phases**z), x, axis=0).astype(np.complex128)
        if (x, z) >= xzi:
            mat = mat + mat.T.conj()
        else:
            mat = 1j * (mat - mat.T.conj())
        ops.append(mat)
        labels.append(f"W{x}{z}")

    return labels, ops


@lru_cache(maxsize=32)
def n_qudit_herm_basis(dims: Tuple[int, ...]) -> List[NDArray[np.complex128]]:
    """
    Construct the tensor product Hermitian operator basis for a composite
    qudit system.

    :param dims: Tuple of per-qudit dimensions.
    :return: List of d_total² Hermitian basis operators.
    """
    if len(dims) == 0:
        raise ValueError("dims must have at least one element.")

    _, first_ops = qudit_herm_operator_basis(dims[0])

    if len(dims) == 1:
        return first_ops

    rest_ops = n_qudit_herm_basis(dims[1:])
    return [np.kron(a, b).astype(np.complex128) for a in first_ops for b in rest_ops]


@lru_cache(maxsize=32)
def weyl_to_herm_basis_change(dims: Tuple[int, ...]) -> NDArray[np.complex128]:
    """
    Compute the unitary change-of-basis matrix U from the plain Weyl basis
    to the Hermitian Weyl basis.

    U_ij = Tr(W_i^† H_j) / d_total

    The Hermitian PTM is then: U^† @ WL @ U.

    :param dims: Tuple of per-qudit dimensions.
    :return: d_total² × d_total² unitary matrix.
    """
    d_total = 1
    for d in dims:
        d_total *= d

    plain_ops = n_qudit_basis(dims)
    herm_ops = n_qudit_herm_basis(dims)

    n = d_total**2
    U = np.zeros((n, n), dtype=np.complex128)
    for i, w in enumerate(plain_ops):
        for j, h in enumerate(herm_ops):
            U[i, j] = np.trace(w.conj().T @ h) / d_total

    return U


@lru_cache(maxsize=32)
def n_qudit_basis(dims: Tuple[int, ...]) -> List[NDArray[np.complex128]]:
    """
    Construct the tensor product operator basis for a composite system of
    qudits with the given dimensions.

    Returns numpy arrays so the cache is safe to use inside JAX JIT traces.

    :param dims: Tuple of per-qudit dimensions, e.g. ``(3,)`` for a single
        qutrit or ``(2, 3)`` for a qubit-qutrit system.
    :return: List of d_total² basis operators, each of shape
        (d_total, d_total), ordered by tensor product.
    """
    if len(dims) == 0:
        raise ValueError("dims must have at least one element.")

    _, first_ops = qudit_operator_basis(dims[0])

    if len(dims) == 1:
        return first_ops

    rest_ops = n_qudit_basis(dims[1:])
    return [np.kron(a, b).astype(np.complex128) for a in first_ops for b in rest_ops]
