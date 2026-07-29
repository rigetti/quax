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

# Copied from forest-benchmarking as a reference implementation of superop2pauli_liouville and choi2pauli_liouville
import itertools
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class OperatorBasis:
    def __init__(self, labels_ops):
        self.ops_by_label = OrderedDict(labels_ops)
        self.labels = list(self.ops_by_label.keys())
        self.ops = list(self.ops_by_label.values())
        self.dim = len(self.ops)

    def product(self, *bases):
        if len(bases) > 1:
            basis_rest = bases[0].product(*bases[1:])
        else:
            assert len(bases) == 1
            basis_rest = bases[0]

        labels_ops = [(b1l + b2l, np.kron(b1, b2)) for (b1l, b1), (b2l, b2) in itertools.product(self, basis_rest)]

        return OperatorBasis(labels_ops)

    def __iter__(self):
        for labels, op in zip(self.labels, self.ops):
            yield labels, op

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError("Can only accept an integer number of factors")
        if n < 1:
            raise ValueError("Need positive number of factors")
        if n == 1:
            return self
        return self.product(*([self] * (n - 1)))

    def __repr__(self):
        return "<span[{}]>".format(",".join(self.labels))


pauli_label_ops = [
    ("I", np.eye(2)),
    ("X", np.array([[0, 1], [1, 0]])),
    ("Y", np.array([[0, -1j], [1j, 0]])),
    ("Z", np.array([[1, 0], [0, -1]])),
]
PAULI_BASIS = OperatorBasis(pauli_label_ops)


def qudit_operator_basis_np(qudit_dim: int) -> OperatorBasis:
    """Generate the Gell-Mann operator basis for a single qudit of dimension d (numpy version)."""
    d = qudit_dim
    ops = [("I", np.eye(d))]

    for i, j in itertools.combinations(range(d), r=2):
        xij = np.zeros((d, d), dtype=complex)
        xij[i, j] = np.sqrt(d / 2)
        xij[j, i] = np.sqrt(d / 2)
        ops.append((f"X({i}{j})", xij))

        yij = np.zeros((d, d), dtype=complex)
        yij[i, j] = -1j * np.sqrt(d / 2)
        yij[j, i] = 1j * np.sqrt(d / 2)
        ops.append((f"Y({i}{j})", yij))

    for k in range(1, d):
        diag = np.array([1] * k + [-k] + [0] * (d - 1 - k), dtype=complex)
        diag = diag * np.sqrt(d) / np.sqrt(k * (k + 1))
        ops.append((f"Z{k}", np.diag(diag)))

    return OperatorBasis(ops)


def n_qudit_basis_np(dims) -> OperatorBasis:
    """Construct the tensor-product operator basis for a composite qudit system (numpy version)."""
    if len(dims) == 1:
        return qudit_operator_basis_np(dims[0])
    return qudit_operator_basis_np(dims[0]).product(n_qudit_basis_np(dims[1:]))


def vec(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix).T.reshape((-1, 1))


def n_qubit_pauli_basis(n) -> OperatorBasis:
    return PAULI_BASIS**n


def n_qudit_pauli_basis(dims) -> OperatorBasis:
    """Return the operator basis for a composite system of qudits with given dims.

    For all-qubit systems (all dims == 2), returns the standard Pauli basis.
    Otherwise returns the generalized Gell-Mann basis.
    """
    if all(d == 2 for d in dims):
        return PAULI_BASIS ** len(dims)
    return n_qudit_basis_np(dims)


def pauli2computational_basis_matrix(dim, dims=None) -> np.ndarray:
    if dims is not None:
        basis = n_qudit_pauli_basis(dims)
    else:
        n_qubits = int(np.log2(dim))
        basis = n_qubit_pauli_basis(n_qubits)

    conversion_mat = np.zeros((dim**2, dim**2), dtype=complex)
    for i, (_, op) in enumerate(basis):
        conversion_mat[:, i] = vec(op).reshape((-1,))
    return conversion_mat


def computational2pauli_basis_matrix(dim, dims=None) -> np.ndarray:
    return pauli2computational_basis_matrix(dim, dims=dims).conj().T / dim


def choi2superop(choi: np.ndarray) -> np.ndarray:
    dim = int(np.sqrt(np.asarray(choi).shape[0]))
    return np.reshape(choi, [dim] * 4).swapaxes(0, 3).reshape([dim**2, dim**2])


def superop2pauli_liouville(superop: np.ndarray) -> np.ndarray:
    dim = int(np.sqrt(np.asarray(superop).shape[0]))
    c2p_basis_transform = computational2pauli_basis_matrix(dim)
    return c2p_basis_transform @ superop @ c2p_basis_transform.conj().T * dim


def choi2pauli_liouville(choi: np.ndarray) -> np.ndarray:
    return superop2pauli_liouville(choi2superop(choi))


def kraus2pauli_liouville(kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
    return superop2pauli_liouville(kraus2superop(kraus_ops))


def kraus2superop(kraus_ops: Sequence[NDArray[np.complex128]] | NDArray[np.complex128]) -> np.ndarray:
    if isinstance(kraus_ops, np.ndarray):  # handle input of single kraus op
        if len(kraus_ops[0].shape) < 2:
            kraus_ops = [kraus_ops]

    rows, cols = np.asarray(kraus_ops[0]).shape

    superop = np.zeros((rows**2, cols**2), dtype=complex)

    for op in kraus_ops:
        superop += np.kron(np.asarray(op).conj(), op)
    return superop
