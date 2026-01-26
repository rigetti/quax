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

from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp
from jax import Array

from ._apply import partial_trace
from ._quantum_objects import SuperOperator, Unitary
from ._superoperator_transformations import to_choi
from .ensembles import PAULI_ENSEMBLE
from .gates import SWAP


def is_unitary(operator: Array, atol: float = 1e-8):
    """
    Validates whether a given operator is unitary.

    An operator U is unitary if U†U = I, where U† is the conjugate transpose of U
    and I is the identity matrix.

    :param operator: The operator to validate.
    :param atol: Absolute tolerance for numerical comparisons.
    :return: True if the operator is unitary, False otherwise.
    """
    d = operator.shape[0]
    identity = jnp.eye(d, dtype=operator.dtype)
    product = jnp.matmul(jnp.conjugate(jnp.transpose(operator)), operator)
    return jnp.allclose(product, identity, atol=atol)


@jax.jit(static_argnames=("atol",))
def is_one_design(ensemble: Unitary, atol: float = 1e-2) -> jax.Array:
    """
    Check whether a 1-qubit unitary ensemble approximately forms a 1-design
    by testing that the Pauli twirl sends X,Y,Z to ~0 on average.

    :param ensemble: Unitary ensemble with dims ((2,), (2,)).
    :param atol: Tolerance for validation.
    :return: A JAX scalar boolean.
    """
    # These asserts run at trace time; if dims are always static this is fine.
    if ensemble.dims != ((2,), (2,)):
        raise ValueError("Only supports 1-qubit unitaries with dims ((2,), (2,)).")

    unitaries = ensemble.matrix
    # Accept either (n,2,2) or (...,2,2) and flatten ensemble dims:
    unitaries = unitaries.reshape((-1, 2, 2))  # (n,2,2)

    # Conjugate: U P U† for each unitary and each Pauli
    # U: (n,a,b), P: (p,b,c), U†: (n,c,d) -> out: (n,p,a,d)
    unitaries_dag = jnp.conj(jnp.swapaxes(unitaries, -1, -2))
    # Only use X, Y, Z Paulis (ignore I)
    twirled = jnp.einsum("nab,pbc,ncd->npad", unitaries, PAULI_ENSEMBLE.matrix[1:], unitaries_dag)  # (n,3,2,2)

    # Average over ensemble: (3,2,2)
    avg = jnp.mean(twirled, axis=0)

    # Frobenius norms for each Pauli: (3,)
    norms = jnp.linalg.norm(avg.reshape((3, -1)), axis=-1)

    # Max norm should be small for a 1-design twirl (Haar expectation is 0)
    return jnp.max(norms) < atol


@jax.jit(static_argnames=("atol",))
def is_two_design(ensemble: "Unitary", atol: float = 1e-2) -> jax.Array:
    """
    Check if a 1-qubit unitary ensemble approximately forms a 2-design by comparing
    empirical vs Haar 2nd moment operator.

    Returns a JAX scalar boolean.
    """
    if ensemble.dims != ((2,), (2,)):
        raise ValueError("Only supports 1-qubit unitaries with dims ((2,), (2,)).")

    # unitaries = ensemble.matrix.reshape((-1, 2, 2))  # (N,2,2)
    ensemble_axes = ensemble.ensemble_size
    N = reduce(mul, ensemble_axes)
    if N <= 1:
        raise ValueError("Ensemble must contain multiple unitaries.")

    # ----- Analytic Haar second moment M_haar -----
    I4 = jnp.eye(4, dtype=complex)
    S = SWAP.matrix

    P_sym = 0.5 * (I4 + S)
    P_asym = 0.5 * (I4 - S)

    # Vectorize in column-major order to match kron convention
    v_sym = jnp.reshape(jnp.swapaxes(P_sym, -1, -2), (-1,))
    v_asym = jnp.reshape(jnp.swapaxes(P_asym, -1, -2), (-1,))

    M_haar = (1.0 / 3.0) * jnp.outer(v_sym, jnp.conj(v_sym)) + 1.0 * jnp.outer(v_asym, jnp.conj(v_asym))

    # ----- Empirical second moment M_ens -----
    # U2 = U ⊗ U  -> (N,4,4)
    u2 = ensemble | ensemble  # tensor product
    # U2 = _kron2_batch(U, U)

    # S(U) = (U2) ⊗ (U2*)  -> (N,16,16)
    # (your code used conj, not conjugate-transpose, which matches the standard Liouville construction)
    su = (u2 | u2.conj()).matrix
    # SU = _kron2_batch(U2, jnp.conj(U2))

    M_ens = jnp.mean(su, axis=list(range(len(ensemble_axes))))  # (16,16)

    dist = jnp.linalg.norm(M_ens - M_haar, ord="fro")
    return dist < atol


@jax.jit
def is_identity_matrix(A: Array):
    """
    Check if a matrix A is the identity matrix.

    :param A: (..., d, d)
    :return: (...) bool
    """
    d = A.shape[-1]
    Id = jnp.eye(d, dtype=A.dtype)
    # Broadcast I to match A's leading dims automatically
    return jnp.all(jnp.isclose(A, Id, atol=1e-8), axis=(-2, -1))


@jax.jit
def is_positive_semidefinite_matrix(A: Array):
    """
    PSD check via Hermitian eigenvalues.
    :param A: (..., d, d) (should be Hermitian within tolerance)
    :return: (...) bool

    For CP via Choi, A should be Hermitian PSD. Numerical noise can introduce tiny
    negative eigenvalues; we allow a tolerance.
    """
    # Symmetrize to reduce tiny non-Hermitian numerical noise (optional but helpful)
    A = 0.5 * (A + jnp.swapaxes(jnp.conj(A), -1, -2))
    evals = jnp.linalg.eigvalsh(A)  # (..., n)
    return jnp.all(evals >= -1e-8, axis=-1)


@jax.jit
def is_trace_preserving(superoperator: SuperOperator):
    """
    Check if a quantum process, specified by a Choi matrix, is trace-preserving (TP).

    :param choi: (..., d^2, d^2)
    :return: (...) bool
    """
    choi = to_choi(superoperator)
    indices = tuple(range(len(choi.dims[0])))
    id_iff_tp = partial_trace(choi, indices=indices)  # expected (..., d, d)
    return is_identity_matrix(id_iff_tp.matrix)


@jax.jit
def is_completely_positive(superoperator: SuperOperator):
    """
    Check if a quantum process, specified by a Choi matrix, is completely positive (CP).

    See equation 3.35 of [GRAPTN]_

    :param choi: (..., d^2, d^2)
    :return: (...) bool
    """
    choi = to_choi(superoperator)
    return is_positive_semidefinite_matrix(choi.matrix)


@jax.jit
def is_cptp(superoperator: SuperOperator):
    """
    Check if a quantum process, specified by a Choi matrix, is completely positive and
    trace-preserving (CPTP).

    :param choi: (..., d^2, d^2) Choi matrix of the quantum process.
    :return: (...) bool
    """
    choi = to_choi(superoperator)
    return jnp.logical_and(
        is_completely_positive(choi),
        is_trace_preserving(choi),
    )
