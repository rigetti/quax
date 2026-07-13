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

"""Module implementing addition (generator combination) for Lindbladians."""

import jax.numpy as jnp

from ._quantum_objects import Lindbladian, Operator


def add_lindbladian(a: Lindbladian, b: Lindbladian) -> Lindbladian:
    """Add two Lindbladian generators: concatenate jump operators and sum Hamiltonians.

    Operands are promoted to common per-subsystem dimensions if they differ (so mixed-dimension
    noise, e.g. qutrit leakage and qubit-subspace depolarizing, combines).  Ensemble (batch) axes
    broadcast, so a single generator adds to an ensemble of generators.
    """
    from ._promotion import promote

    a_dims, b_dims = a.dims[0], b.dims[0]
    if len(a_dims) != len(b_dims):
        raise ValueError(f"Cannot add Lindbladians on {a_dims} and {b_dims} qudits: the subsystem counts differ.")
    target = tuple(max(x, y) for x, y in zip(a_dims, b_dims))
    if a_dims != target:
        a = promote(a, target)
    if b_dims != target:
        b = promote(b, target)

    # Concatenate the jump stacks along the n_ops axis (-3), broadcasting any leading ensemble axes.
    a_jumps, b_jumps = a.jump_operators.matrix, b.jump_operators.matrix
    ensemble = jnp.broadcast_shapes(a_jumps.shape[:-3], b_jumps.shape[:-3])
    a_jumps = jnp.broadcast_to(a_jumps, ensemble + a_jumps.shape[-3:])
    b_jumps = jnp.broadcast_to(b_jumps, ensemble + b_jumps.shape[-3:])
    combined_jumps = Operator.from_matrix(
        jnp.concatenate([a_jumps, b_jumps], axis=-3),
        a.jump_operators.dims,
    )
    # None-aware Hamiltonian sum (Observable + Observable → Observable).
    if a.hamiltonian is None:
        hamiltonian = b.hamiltonian
    elif b.hamiltonian is None:
        hamiltonian = a.hamiltonian
    else:
        hamiltonian = a.hamiltonian + b.hamiltonian
    return Lindbladian(hamiltonian=hamiltonian, jump_operators=combined_jumps)
