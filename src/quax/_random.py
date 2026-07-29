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

from functools import partial, reduce
from operator import mul

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Choi, DensityMatrix, Observable, Operator, StateVector, Unitary


@partial(jax.jit, static_argnames=("dim", "k", "size"))
def ginibre_matrix_complex(dim: int, k: int, key: Array, size: tuple[int, ...] = ()) -> Array:
    r"""
    Given a scalars dim and k, returns a dim by k matrix, drawn from the complex Ginibre
    ensemble :cite:`IM`.

    Each element is distributed

    .. math::
        \sim [N(0, 1) + i · N(0, 1)]

    Here :math:`X \sim N(0,1)` denotes a normally distributed random variable.

    :param dim: Hilbert space dimension.
    :param k: Ultimately becomes the rank of a state.
    :param key: The random number generator.
    :return: Returns a dim by k matrix, drawn from the Ginibre ensemble.
    """
    return jax.random.normal(key, size + (dim, k), dtype=complex) + 1j * jax.random.normal(
        key, size + (dim, k), dtype=complex
    )


@partial(jax.jit, static_argnames=("dims", "rank", "size"))
def random_density_matrix(rank: int, dims: tuple[int, ...], key: Array, size: tuple[int, ...] = ()) -> DensityMatrix:
    dim = reduce(mul, dims, 1)
    if rank > dim:
        raise ValueError("The rank of the state matrix cannot exceed the dimension.")

    A = ginibre_matrix_complex(dim=dim, k=rank, key=key, size=size)  # expect size + (dim, rank)

    rho = A @ jnp.swapaxes(jnp.conjugate(A), -1, -2)  # size + (dim, dim) or (dim, dim) if size==()

    tr = jnp.trace(rho, axis1=-2, axis2=-1)
    rho = rho / tr[..., None, None]

    return DensityMatrix.from_matrix(rho, dims)


@jax.jit(static_argnames=("dims", "size"))
def random_operator(dims: tuple[tuple[int, ...], tuple[int, ...]], key: Array, size: tuple[int, ...] = ()) -> Operator:
    """Given input and output Hilbert space dimensions, returns a random operator drawn from the Ginibre ensemble."""
    d_out = reduce(mul, dims[0], 1)
    d_in = reduce(mul, dims[1], 1)

    mat = ginibre_matrix_complex(dim=d_out, k=d_in, key=key, size=size)  # size + (d_out, d_in)

    return Operator.from_matrix(mat, dims)


@jax.jit(static_argnames=("dims", "size"))
def random_observable(
    dims: tuple[tuple[int, ...], tuple[int, ...]], key: Array, size: tuple[int, ...] = ()
) -> Observable:
    """Given input and output Hilbert space dimensions, returns a random Hermitian operator drawn from the Ginibre ensemble."""
    d_out = reduce(mul, dims[0], 1)
    d_in = reduce(mul, dims[1], 1)
    mat = ginibre_matrix_complex(dim=d_out, k=d_in, key=key, size=size)  # size + (d_out, d_in)
    hermitian_mat = (mat + jnp.swapaxes(jnp.conjugate(mat), -1, -2)) / 2  # size + (d_out, d_in)
    return Observable.from_matrix(hermitian_mat, dims)


@jax.jit(static_argnames=("dims", "size"))
def random_unitary(dims: tuple[tuple[int, ...], tuple[int, ...]], key: Array, size: tuple[int, ...] = ()) -> Unitary:
    """
    Given a Hilbert space dimension dim this function returns a unitary operator
    U ∈ C^(dim by dim) drawn from the Haar measure :cite:`MEZ`.

    :param dims: The Qudit dimensions.
    :param key: The random number generator key.
    :param size: The ensemble size.
    :return: Returns a dim by dim unitary operator U drawn from the Haar measure.
    """
    assert dims[0] == dims[1]
    d = reduce(mul, dims[0])

    z = ginibre_matrix_complex(dim=d, k=d, key=key, size=size)
    q, r = jnp.linalg.qr(z)
    diag = r.diagonal(offset=0, axis1=-2, axis2=-1)
    unitaries = q * (diag / jnp.abs(diag))[..., jnp.newaxis, :]

    if size == ():
        unitaries = jnp.squeeze(unitaries)

    return Unitary.from_matrix(unitaries, dims)


@jax.jit(static_argnames=("dims", "size"))
def random_state_vector(dims: tuple[int, ...], key: Array, size: tuple[int, ...] = ()) -> "StateVector":
    r"""
    Given a Hilbert space dimension dim, returns a state vector \|ψ⟩ ∈ C^dim
    drawn uniformly from the unit sphere in C^dim.

    :param dims: The Qudit dimensions.
    :param key: The random number generator.
    :param size: The number of state vectors to generate.
    :return: Returns a state vector \|ψ⟩ drawn uniformly from the unit sphere in C^dim.
    """
    dim = reduce(mul, dims)

    # Draw complex Gaussian vectors: shape size + (dim,)
    vec = ginibre_matrix_complex(dim=dim, k=1, key=key, size=size).squeeze(-1)

    # Normalize along the last axis
    norm = jnp.linalg.norm(vec, axis=-1, keepdims=True)
    data = vec / norm

    # If you really want scalar output to be (dim,) not (1, dim), this already does it
    # because size=() => vec.shape == (dim,)
    return StateVector.from_matrix(data, dims)


@jax.jit(static_argnames=("dims", "rank", "size"))
def random_choi(
    dims: tuple[tuple[int, ...], tuple[int, ...]], rank: int, key: Array, size: tuple[int, ...] = ()
) -> Choi:
    """
    Given a Hilbert space dimension dim and a Kraus rank K, returns a (d², d²) Choi
    matrix J(Λ) of a channel drawn from the BCSZ distribution with Kraus rank K :cite:`RQO`.

    :param dim: Hilbert space dimension.
    :param rank: The number of Kraus operators in the operator sum description of the channel.
    :param key: The random number generator.
    :param size: The number of Choi matrices to generate.
    :return: Choi matrix, drawn from the BCSZ distribution with Kraus rank K.
    """
    assert dims[0] == dims[1], "Random channels must have equal input and output dimensions."
    # assert size == (), "Currently only supports generating a single Choi matrix."
    atol = 1e-8
    d = reduce(mul, dims[0])
    d2 = d * d

    # X: size + (N^2, rank)
    X = ginibre_matrix_complex(dim=d**2, k=rank, key=key, size=size)

    # XX†: size + (N^2, N^2)
    XXdag = X @ jnp.swapaxes(jnp.conj(X), -2, -1)

    # We will create a TP channel
    X4 = XXdag.reshape(*XXdag.shape[:-2], d, d, d, d)  # (..., i, j, i, k) pattern in einsum
    Y = jnp.einsum("...ijik->...jk", X4)  # size + (N, N)

    # inv_sqrt_Y = sqrtm(inv(Y))  -> do inverse sqrt stably
    Yh = 0.5 * (Y + jnp.swapaxes(jnp.conj(Y), -2, -1))
    w, v = jnp.linalg.eigh(Yh)
    w = jnp.maximum(w, atol)
    inv_sqrt_Y = (v * (1.0 / jnp.sqrt(w))[..., None, :]) @ jnp.swapaxes(jnp.conj(v), -2, -1)

    # Z = _kron_I_A(N, inv_sqrt_Y)                                    # size + (N^2, N^2)
    Id = jnp.eye(d, dtype=XXdag.dtype)
    Z4 = jnp.einsum("ij,...ab->...iajb", Id, inv_sqrt_Y)  # size + (i,a,j,b)
    Z = Z4.reshape(*inv_sqrt_Y.shape[:-2], d2, d2)

    D_row = Z @ XXdag @ Z

    # Convert paper(row) -> column-stacking (QuTiP) by swapping subsystems:
    lead = D_row.shape[:-2]
    D4 = D_row.reshape(*lead, d, d, d, d)
    D4 = jnp.transpose(D4, (*range(len(lead)), len(lead) + 1, len(lead) + 0, len(lead) + 3, len(lead) + 2))
    D_col = D4.reshape(*lead, d2, d2)

    return Choi.from_matrix(D_col, dims)
