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
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from ._quantum_objects import Choi, DensityMatrix, StateVector, Unitary


@partial(jax.jit, static_argnames=("dim", "k", "size"))
def ginibre_matrix_complex(dim: int, k: int, key: Array, size: Tuple[int, ...] = ()) -> Array:
    r"""
    Given a scalars dim and k, returns a dim by k matrix, drawn from the complex Ginibre
    ensemble [IM]_.

    Each element is distributed

    .. math::
        \sim [N(0, 1) + i · N(0, 1)]

    Here :math:`X \sim N(0,1)` denotes a normally distributed random variable.

    .. [IM] Induced measures in the space of mixed quantum states.
         Zyczkowski et al.
         J. Phys A: Math. and Gen. 34, 7111 (2001).
         https://doi.org/10.1088/0305-4470/34/35/335
         https://arxiv.org/abs/quant-ph/0012101

    :param dim: Hilbert space dimension.
    :param k: Ultimately becomes the rank of a state.
    :param key: The random number generator.
    :return: Returns a dim by k matrix, drawn from the Ginibre ensemble.
    """
    return jax.random.normal(key, size + (dim, k), dtype=complex) + 1j * jax.random.normal(
        key, size + (dim, k), dtype=complex
    )


@partial(jax.jit, static_argnames=("dims", "rank", "size"))
def random_density_matrix(rank: int, dims: Tuple[int, ...], key: Array, size: Tuple[int, ...] = ()) -> DensityMatrix:
    dim = reduce(mul, dims, 1)
    if rank > dim:
        raise ValueError("The rank of the state matrix cannot exceed the dimension.")

    A = ginibre_matrix_complex(dim=dim, k=rank, key=key, size=size)  # expect size + (dim, rank)

    rho = A @ jnp.swapaxes(jnp.conjugate(A), -1, -2)  # size + (dim, dim) or (dim, dim) if size==()

    tr = jnp.trace(rho, axis1=-2, axis2=-1)
    rho = rho / tr[..., None, None]

    num_ensemble_dims = len(size)
    return DensityMatrix.from_matrix(rho, dims, num_ensemble_dims)


@jax.jit(static_argnames=("dims", "size"))
def random_unitary(dims: Tuple[Tuple[int, ...], Tuple[int, ...]], key: Array, size: Tuple[int, ...] = ()) -> Unitary:
    """
    Given a Hilbert space dimension dim this function returns a unitary operator
    U ∈ C^(dim by dim) drawn from the Haar measure [MEZ]_.

    .. [MEZ] How to generate random matrices from the classical compact groups.
          Mezzadri.
          Notices of the American Mathematical Society 54, 592 (2007).
          http://www.ams.org/notices/200705/fea-mezzadri-web.pdf
          https://arxiv.org/abs/math-ph/0609050

    :param dim: Hilbert space dimension (scalar).
    :param rs: Optional random state
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

    num_ensemble_dims = len(size)
    return Unitary.from_matrix(unitaries, dims, num_ensemble_dims)


@jax.jit(static_argnames=("dims", "size"))
def random_state_vector(dims: Tuple[int, ...], key: Array, size: Tuple[int, ...] = ()) -> "StateVector":
    """
    Given a Hilbert space dimension dim, returns a state vector |ψ⟩ ∈ C^dim
    drawn uniformly from the unit sphere in C^dim.

    :param dims: The Qudit dimensions.
    :param key: The random number generator.
    :param size: The number of state vectors to generate.
    :return: Returns a state vector |ψ⟩ drawn uniformly from the unit sphere in C^dim.
    """
    dim = reduce(mul, dims)

    # Draw complex Gaussian vectors: shape size + (dim,)
    vec = ginibre_matrix_complex(dim=dim, k=1, key=key, size=size).squeeze(-1)

    # Normalize along the last axis
    norm = jnp.linalg.norm(vec, axis=-1, keepdims=True)
    data = vec / norm

    # If you really want scalar output to be (dim,) not (1, dim), this already does it
    # because size=() => vec.shape == (dim,)
    num_ensemble_dims = len(size)
    return StateVector.from_matrix(data, dims, num_ensemble_dims)


@jax.jit(static_argnames=("dims", "rank", "size"))
def random_choi_BCSZ(
    dims: Tuple[Tuple[int, ...], Tuple[int, ...]], rank: int, key: Array, size: Tuple[int, ...] = ()
) -> Choi:
    """
    Given a Hilbert space dimension dim and a Kraus rank K, returns a (d², d²) Choi
    matrix J(Λ) of a channel drawn from the BCSZ distribution with Kraus rank K [RQO]_.

    .. [RQO] Random quantum operations.
          Bruzda et al.
          Physics Letters A 373, 320 (2009).
          https://doi.org/10.1016/j.physleta.2008.11.043
          https://arxiv.org/abs/0804.2361

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

    num_ensemble_dims = len(size)
    return Choi.from_matrix(D_col, dims, num_ensemble_dims)
