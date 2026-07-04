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

"""Scalar multiplication of quantum objects with type-narrowing dispatch.

A single :func:`mul` function, dispatched via :func:`functools.singledispatch`
on the operator type, handles ensemble broadcasting and returns the most
specific correct type for the result.

Type narrowing is based on **static type / dtype only**, never on the scalar's
value.  Code that *knows* the result has additional structure (e.g. the gate
constructors) should use :func:`typing.cast` to express that intent.
"""

from __future__ import annotations

import functools
from typing import cast, overload

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Involution, Lindbladian, Observable, Operator, QuantumObject, Unitary

# --------------------------------------------------------------------------- #
# Broadcasting helper (JIT-compiled)
# --------------------------------------------------------------------------- #


@functools.partial(jax.jit, static_argnums=(2, 3))
def _broadcast_scalar_data(
    scalar: Array,
    data: Array,
    ensemble_size: tuple[int, ...],
    num_ensemble_dims: int,
) -> Array:
    """Broadcast *scalar* against *data*, respecting ensemble dimensions.

    Returns the new ``data`` array (already multiplied by *scalar*).
    """
    scalar_array = jnp.asarray(scalar)
    broadcast_dims = jnp.broadcast_shapes(scalar_array.shape, ensemble_size)
    broadcast_scalar = jnp.broadcast_to(scalar_array, broadcast_dims)
    tail_ndims = data.ndim - num_ensemble_dims
    broadcast_scalar = broadcast_scalar.reshape(broadcast_scalar.shape + (1,) * tail_ndims)
    padded_data = data.reshape((1,) * (len(broadcast_dims) - num_ensemble_dims) + data.shape)
    return padded_data * broadcast_scalar


def _is_real_type(scalar: complex | Array) -> bool:
    """Return ``True`` if *scalar* is statically known to be real.

    Uses ``isinstance`` for Python scalars and ``jnp.issubdtype`` for JAX
    arrays — both are JIT-safe since they inspect type/dtype, not value.
    """
    if isinstance(scalar, (int, float)):
        return True
    if isinstance(scalar, complex):
        return False
    return bool(jnp.issubdtype(jnp.asarray(scalar).dtype, jnp.floating))


# --------------------------------------------------------------------------- #
# Singledispatch scalar multiplication
# --------------------------------------------------------------------------- #


@overload
def mul(op: Involution, scalar: complex | Array) -> Observable | Operator: ...


@overload
def mul(op: Observable, scalar: complex | Array) -> Observable | Operator: ...


@overload
def mul(op: Unitary, scalar: complex | Array) -> Operator: ...


@overload
def mul(op: QuantumObject, scalar: complex | Array) -> QuantumObject: ...


@functools.singledispatch
def mul(op: QuantumObject, scalar: complex | Array) -> QuantumObject:
    """Scalar multiplication preserving the concrete type of *op*."""
    new_data = _broadcast_scalar_data(scalar, op.data, op.ensemble_size, op.num_ensemble_dims)
    return type(op)(new_data, op.num_qubits)


@mul.register(Unitary)
def _mul_unitary(op: Unitary, scalar: complex | Array) -> Operator:
    """Unitary * scalar always returns ``Operator``.

    Scalar multiplication does not, in general, preserve unitarity.
    Gate constructors that know the result is still unitary should
    use ``cast(Unitary, ...)``.
    """
    new_data = _broadcast_scalar_data(scalar, op.data, op.ensemble_size, op.num_ensemble_dims)
    return Operator(new_data, op.num_qubits)


@mul.register(Observable)
def _mul_observable(op: Observable, scalar: complex | Array) -> Observable | Operator:
    """Real scalar → ``Observable``; complex → ``Operator``."""
    new_data = _broadcast_scalar_data(scalar, op.data, op.ensemble_size, op.num_ensemble_dims)
    if _is_real_type(scalar):
        return Observable(new_data, op.num_qubits)
    return Operator(new_data, op.num_qubits)


@mul.register(Involution)
def _mul_involution(op: Involution, scalar: complex | Array) -> Observable | Operator:
    """Real scalar → ``Observable``; complex → ``Operator``.

    Even ``±1`` is demoted to ``Observable`` because the caller can always
    use :func:`~quax.Involution` or ``__neg__`` when they know the intent.
    """
    new_data = _broadcast_scalar_data(scalar, op.data, op.ensemble_size, op.num_ensemble_dims)
    if _is_real_type(scalar):
        return Observable(new_data, op.num_qubits)
    return Operator(new_data, op.num_qubits)


@mul.register(Lindbladian)
def _mul_lindbladian(op: Lindbladian, scalar: complex | Array) -> Lindbladian:
    """Scale a Lindbladian generator by a non-negative real ``scalar``.

    Scaling the generator by ``c`` scales its Hamiltonian by ``c`` and its jump operators by
    ``sqrt(c)`` (since ``D[√c·L] = c·D[L]``), keeping the result a valid GKSL generator.  Complex
    scalars are rejected because they yield a non-CP generator; a negative real ``c`` violates the
    ``c >= 0`` precondition (it produces NaN jump operators) and is the caller's responsibility —
    the same status as the ``t >= 0`` precondition of :func:`~quax.evolve`.
    """
    if not _is_real_type(scalar):
        raise NotImplementedError(
            "Scaling a Lindbladian by a complex scalar is not supported: it produces a non-CP "
            "generator that cannot be represented by jump operators."
        )
    root = jnp.sqrt(jnp.asarray(scalar))
    scaled_jumps = cast(Operator, op.jump_operators * root)
    # Real scalar ⇒ Observable * scalar stays an Observable (see :func:`_mul_observable`).
    scaled_hamiltonian = cast(Observable, op.hamiltonian * scalar) if op.hamiltonian is not None else None
    return Lindbladian(hamiltonian=scaled_hamiltonian, jump_operators=scaled_jumps)
