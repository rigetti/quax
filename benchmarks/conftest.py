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

"""Shared fixtures and helpers for performance benchmarks."""

from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp

import quax as qx

SEED = 42


def make_state_vector(dims: tuple[int, ...], ensemble_size: tuple[int, ...] = ()) -> qx.StateVector:
    """Create a random state vector for benchmarking."""
    key = jax.random.key(SEED)
    sv = qx.random_state_vector(dims=dims, key=key, size=ensemble_size)
    sv.data.block_until_ready()
    return sv


def make_density_matrix(dims: tuple[int, ...], ensemble_size: tuple[int, ...] = ()) -> qx.DensityMatrix:
    """Create a random density matrix for benchmarking."""
    d = reduce(mul, dims)
    key = jax.random.key(SEED)
    dm = qx.random_density_matrix(rank=d, dims=dims, key=key, size=ensemble_size)
    dm.data.block_until_ready()
    return dm


def make_unitary(gate_dims: tuple[int, ...]) -> qx.Unitary:
    """Create a random unitary for benchmarking."""
    key = jax.random.key(SEED + 1)
    u = qx.random_unitary(dims=(gate_dims, gate_dims), key=key)
    u.data.block_until_ready()
    return u


def make_superop(gate_dims: tuple[int, ...]) -> qx.SuperOp:
    """Create a depolarizing channel superoperator for benchmarking."""
    s = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=gate_dims)
    s.data.block_until_ready()
    return s


def make_kraus_map(gate_dims: tuple[int, ...], rank: int, truncate: bool = False) -> qx.KrausMap:
    """Create a Kraus map with a given rank for benchmarking.

    :param gate_dims: Per-qudit dimensions of the gate, e.g. ``(2,)`` or ``(3,)``.
    :param rank: Kraus rank of the channel.
    :param truncate: If True, apply truncate_kraus to remove near-zero operators.
    """
    key = jax.random.key(SEED + 2)
    choi = qx.random_choi(dims=(gate_dims, gate_dims), rank=rank, key=key)
    km = qx.choi_to_kraus(choi)
    if truncate:
        km = qx.truncate_kraus(km)
    km.data.block_until_ready()
    return km


def make_depolarizing_kraus(gate_dims: tuple[int, ...], truncate: bool = False) -> qx.KrausMap:
    """Create a Kraus map from a depolarizing channel for benchmarking."""
    s = qx.depolarizing_channel_superoperator(jnp.array(0.05), dims=gate_dims)
    km = qx.superop_to_kraus(s)
    if truncate:
        km = qx.truncate_kraus(km)
    km.data.block_until_ready()
    return km


def make_ideal_instrument(dim: int = 2) -> qx.QuantumInstrument:
    """Create an ideal projective measurement instrument."""
    inst = qx.gates.MEASURE(dim=dim)
    inst.data.block_until_ready()
    return inst


def make_noisy_instrument(dim: int = 2) -> qx.QuantumInstrument:
    """Create a noisy instrument: random unitary before ideal projection.

    E_k(rho) = P_k U rho U† P_k  where P_k = |k><k|.
    """
    key = jax.random.key(SEED + 3)
    u = qx.random_unitary(dims=((dim,), (dim,)), key=key)
    u_mat = u.matrix

    superop_list = []
    for k in range(dim):
        proj_k = jnp.zeros((dim, dim), dtype=complex).at[k, k].set(1.0)
        kraus = proj_k @ u_mat
        superop_k = jnp.einsum("ab,cd->acbd", jnp.conj(kraus), kraus).reshape(dim * dim, dim * dim)
        superop_list.append(superop_k)
    matrices = jnp.stack(superop_list, axis=0)
    inst = qx.QuantumInstrument.from_matrix(matrices, ((dim,), (dim,)), (0,))
    inst.data.block_until_ready()
    return inst


def make_keys(key_size: int) -> jax.Array:
    """Create a JAX PRNG key or array of keys.

    :param key_size: If 1, returns a scalar key. Otherwise returns an array of keys.
    """
    base = jax.random.key(SEED + 100)
    if key_size == 1:
        return base
    return jax.random.split(base, key_size)
