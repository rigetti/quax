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

"""Shared helpers for QuantumInstrument tests."""

from functools import reduce
from operator import mul

import jax.numpy as jnp

import quax as qx
from quax import DensityMatrix, StateVector


def basis_dm(index: int, dim: int) -> DensityMatrix:
    """Create density matrix |index><index| for a single qudit of dimension *dim*."""
    mat = jnp.zeros((dim, dim), dtype=complex).at[index, index].set(1.0)
    return DensityMatrix.from_matrix(mat, (dim,))


def basis_dm_multi(indices: tuple[int, ...], dims: tuple[int, ...]) -> DensityMatrix:
    """Create density matrix for a multi-qudit computational basis state."""
    d_total = reduce(mul, dims, 1)
    flat = 0
    for idx, d in zip(indices, dims):
        flat = flat * d + idx
    mat = jnp.zeros((d_total, d_total), dtype=complex).at[flat, flat].set(1.0)
    return DensityMatrix.from_matrix(mat, dims)


def basis_sv(index: int, dim: int):
    """Create state vector |index> for a single qudit of dimension *dim*."""
    vec = jnp.zeros(dim, dtype=complex).at[index].set(1.0)
    return StateVector.from_matrix(vec, (dim,))


def superposition_dm(dim: int) -> DensityMatrix:
    """Create density matrix of equal superposition (|0> + ... + |d-1>) / sqrt(d)."""
    vec = jnp.ones(dim, dtype=complex) / jnp.sqrt(dim)
    sv = StateVector.from_matrix(vec, (dim,))
    return qx.promote_state_vector_to_density_matrix(sv)


def make_noisy_instrument_from_unitary(unitary_matrix, dim):
    """Build a single-qudit instrument: unitary U applied before ideal projection.

    E_k(rho) = P_k U rho U† P_k  where P_k = |k><k|.
    """
    superop_list = []
    for k in range(dim):
        proj_k = jnp.zeros((dim, dim), dtype=complex).at[k, k].set(1.0)
        kraus = proj_k @ unitary_matrix
        superop_k = jnp.einsum("ab,cd->acbd", jnp.conj(kraus), kraus).reshape(dim * dim, dim * dim)
        superop_list.append(superop_k)
    matrices = jnp.stack(superop_list, axis=0)
    return qx.QuantumInstrument.from_matrix(matrices, ((dim,), (dim,)), (0,))


def make_noisy_instrument_multi(unitary_matrix, dims):
    """Build a multi-qudit instrument: unitary U before ideal projection on all qudits."""
    d_total = reduce(mul, dims, 1)
    n_qudits = len(dims)
    superop_list = []
    for k in range(d_total):
        proj_k = jnp.zeros((d_total, d_total), dtype=complex).at[k, k].set(1.0)
        kraus = proj_k @ unitary_matrix
        superop_k = jnp.einsum("ab,cd->acbd", jnp.conj(kraus), kraus).reshape(d_total * d_total, d_total * d_total)
        superop_list.append(superop_k)
    matrices = jnp.stack(superop_list, axis=0)
    measured_qudits = tuple(range(n_qudits))
    return qx.QuantumInstrument.from_matrix(matrices, (dims, dims), measured_qudits)


def make_spectator_instrument(action_matrix, dims, measured_qudit=0):
    """Build a multi-qudit instrument that measures one qudit (ideal projection)
    but applies action_matrix to the full system.

    E_k(rho) = (P_k_full @ action) rho (P_k_full @ action)†
    where P_k_full projects the measured qudit to |k><k| ⊗ I_rest.
    """
    d_total = reduce(mul, dims, 1)
    d_meas = dims[measured_qudit]
    superop_list = []
    for k in range(d_meas):
        proj_k_full = jnp.zeros((d_total, d_total), dtype=complex)
        for idx in range(d_total):
            indices = []
            flat = idx
            for s in reversed(dims):
                indices.append(flat % s)
                flat //= s
            indices = tuple(reversed(indices))
            if indices[measured_qudit] == k:
                proj_k_full = proj_k_full.at[idx, idx].set(1.0)
        kraus = proj_k_full @ action_matrix
        superop_k = jnp.einsum("ab,cd->acbd", jnp.conj(kraus), kraus).reshape(d_total * d_total, d_total * d_total)
        superop_list.append(superop_k)
    matrices = jnp.stack(superop_list, axis=0)
    return qx.QuantumInstrument.from_matrix(matrices, (dims, dims), (measured_qudit,))
