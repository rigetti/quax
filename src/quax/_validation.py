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
from ._quantum_objects import SuperOperator, Unitary, Operator, State
from ._superoperator_transformations import to_choi
from .ensembles import PAULI_ENSEMBLE
from .gates import SWAP


def is_hermitian(operator: SuperOperator | Operator | Unitary | State, atol: float = 1e-8):
    """
    Validates whether a given operator is Hermitian (self-adjoint).

    An operator A is Hermitian if A = A†, where A† is the conjugate transpose of A.

    :param operator: The operator to validate (quantum object or raw array).
    :param atol: Absolute tolerance for numerical comparisons.
    :return: Boolean array (scalar for single operator, batched for ensembles).
    """
    matrix = operator.matrix
    matrix_h = jnp.conjugate(jnp.swapaxes(matrix, -1, -2))
    return jnp.all(jnp.isclose(matrix, matrix_h, atol=atol), axis=(-2, -1))


def is_unitary(operator: SuperOperator | Operator | Unitary, atol: float = 1e-8):
    """
    Validates whether a given operator is unitary.

    An operator U is unitary if U†U = I, where U† is the conjugate transpose of U
    and I is the identity matrix.

    :param operator: The operator to validate (quantum object or raw array).
    :param atol: Absolute tolerance for numerical comparisons.
    :return: Boolean array (scalar for single operator, batched for ensembles).
    """
    matrix = operator.matrix
    d = matrix.shape[-1]
    identity = jnp.eye(d, dtype=matrix.dtype)
    product = jnp.matmul(jnp.conjugate(jnp.swapaxes(matrix, -1, -2)), matrix)
    return jnp.all(jnp.isclose(product, identity, atol=atol), axis=(-2, -1))


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
def is_identity_matrix(A: Array, atol: float = 1e-8):
    """
    Check if a matrix A is the identity matrix.

    :param A: (..., d, d)
    :param atol: Absolute tolerance for numerical comparisons.
    :return: (...) bool
    """
    d = A.shape[-1]
    Id = jnp.eye(d, dtype=A.dtype)
    # Broadcast I to match A's leading dims automatically
    return jnp.all(jnp.isclose(A, Id, atol=atol), axis=(-2, -1))


@jax.jit
def is_positive_semidefinite_matrix(A: Array, atol: float = 1e-8):
    """
    PSD check via Hermitian eigenvalues.
    :param A: (..., d, d) (should be Hermitian within tolerance)
    :param atol: Absolute tolerance for numerical comparisons.
    :return: (...) bool

    For CP via Choi, A should be Hermitian PSD. Numerical noise can introduce tiny
    negative eigenvalues; we allow a tolerance.
    """
    # Symmetrize to reduce tiny non-Hermitian numerical noise (optional but helpful)
    A = 0.5 * (A + jnp.swapaxes(jnp.conj(A), -1, -2))
    evals = jnp.linalg.eigvalsh(A)  # (..., n)
    return jnp.all(evals >= -atol, axis=-1)


@jax.jit
def is_trace_preserving(superoperator: SuperOperator, atol: float = 1e-8):
    """
    Check if a quantum process, specified by a Choi matrix, is trace-preserving (TP).

    :param superoperator: A superoperator (Choi, SuperOp, KrausMap, or PauliLiouville).
    :param atol: Absolute tolerance for numerical comparisons.
    :return: (...) bool
    """
    choi = to_choi(superoperator)
    indices = tuple(range(len(choi.dims[0])))
    id_iff_tp = partial_trace(choi, indices=indices)  # expected (..., d, d)
    return is_identity_matrix(id_iff_tp.matrix, atol=atol)


@jax.jit
def is_completely_positive(superoperator: SuperOperator, atol: float = 1e-8):
    """
    Check if a quantum process, specified by a Choi matrix, is completely positive (CP).

    See equation 3.35 of [GRAPTN]_

    :param superoperator: A superoperator (Choi, SuperOp, KrausMap, or PauliLiouville).
    :param atol: Absolute tolerance for numerical comparisons.
    :return: (...) bool
    """
    choi = to_choi(superoperator)
    return is_positive_semidefinite_matrix(choi.matrix, atol=atol)


@jax.jit
def is_hermicity_preserving(superoperator: SuperOperator, atol: float = 1e-8):
    """
    Check if a quantum process is hermicity-preserving (HP).

    A map is hermicity-preserving if it maps Hermitian operators to Hermitian operators.
    This is equivalent to the Choi matrix being Hermitian.

    :param superoperator: A superoperator (Choi, SuperOp, KrausMap, or PauliLiouville).
    :param atol: Absolute tolerance for numerical comparisons.
    :return: (...) bool
    """
    choi = to_choi(superoperator)
    matrix = choi.matrix
    matrix_h = jnp.conjugate(jnp.swapaxes(matrix, -1, -2))
    return jnp.all(jnp.isclose(matrix, matrix_h, atol=atol), axis=(-2, -1))


@jax.jit
def is_cptp(superoperator: SuperOperator, atol: float = 1e-8):
    """
    Check if a quantum process, specified by a Choi matrix, is completely positive and
    trace-preserving (CPTP).

    :param superoperator: A superoperator (Choi, SuperOp, KrausMap, or PauliLiouville).
    :param atol: Absolute tolerance for numerical comparisons.
    :return: (...) bool
    """
    choi = to_choi(superoperator)
    return jnp.logical_and(
        is_completely_positive(choi, atol=atol),
        is_trace_preserving(choi, atol=atol),
    )
