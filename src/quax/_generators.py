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

"""Generator engines for :class:`~quax.Lindbladian`.

Builds the d²×d² GKSL master-equation generator from a Hamiltonian and jump operators, and
provides related helpers (e.g. converting a unitary to its Hamiltonian generator).  Kept out of
``_quantum_objects`` so the class definition stays lean; imported lazily by
:attr:`quax.Lindbladian.matrix`.
"""

from functools import reduce
from operator import mul
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from ._quantum_objects import Observable, Operator, Unitary


def unitary_to_hamiltonian(unitary: "Unitary") -> "Observable":
    """Return the Hamiltonian generator :math:`H` (Hermitian) with :math:`e^{-iH} = U`.

    This is the inverse of the Schrödinger propagator ``evolve(Observable) -> Unitary`` at
    ``t = 1`` (:func:`~quax.evolve` uses ``exp(-i t H)``).  The principal branch is used, so
    ``H`` has eigenvalues in :math:`(-\\pi, \\pi]`.

    Differentiable with respect to ``U`` (uses ``jax.lax.linalg.eig`` with
    ``enable_eigvec_derivs=True``); gradients are well-defined where ``U`` has a non-degenerate
    spectrum — the usual caveat of eigenvector derivatives.

    :param unitary: The unitary ``U``.
    :return: The Hermitian generator ``H`` (an :class:`~quax.Observable`) satisfying ``exp(-iH) = U``.
    """
    from ._quantum_objects import Observable

    eigvals, eigvecs = jax.lax.linalg.eig(unitary.matrix, compute_left_eigenvectors=False, enable_eigvec_derivs=True)
    # exp(-i h) = λ = exp(i·arg λ)  ⇒  h = -arg(λ)  (real eigenvalues of H)
    h_eig = -jnp.angle(eigvals)
    hamiltonian = (eigvecs * h_eig[..., None, :]) @ jnp.linalg.inv(eigvecs)  # V diag(h) V⁻¹
    hamiltonian = 0.5 * (hamiltonian + jnp.conj(jnp.swapaxes(hamiltonian, -1, -2)))  # enforce Hermiticity
    return Observable.from_matrix(hamiltonian, unitary.dims)


def add_hamiltonians(h1: "Observable | None", h2: "Observable | None") -> "Observable | None":
    """Sum two optional Hamiltonians, treating ``None`` as the zero operator."""
    from ._quantum_objects import Observable

    if h1 is None:
        return h2
    if h2 is None:
        return h1
    return Observable.from_matrix(h1.matrix + h2.matrix, h1.dims)


@jax.jit
def gksl_generator(hamiltonian: "Observable | None", jump_operators: "Operator") -> Array:
    """GKSL generator matrix ``-i[H,ρ] + Σ_k D[L_k]`` as ``(*ensemble, d², d²)``.

    Rates are pre-absorbed into ``jump_operators``. Engine for :attr:`quax.Lindbladian.matrix`.
    """
    dims = jump_operators.dims
    d = reduce(mul, dims[1], 1)
    I = jnp.eye(d, dtype=complex)
    L = jump_operators.matrix  # (*ensemble, n_ops, d, d)
    ensemble_shape = L.shape[:-3]

    # Build the GKSL generator as a rank-4 tensor with indices
    # [out_bra a, out_ket c, in_bra b, in_ket d], reshaped at the end to a (d², d²)
    # superoperator acting on vec(ρ).
    #
    # Tensor products via einsum: an einsum whose output indices are the union of the
    # (disjoint) input indices, with none summed away, is an outer/tensor product.  E.g.
    # ``einsum("ab,cd->acbd", A, B)`` computes A ⊗ B and then interleaves the axes so that
    # (a, c) form the superoperator's row multi-index and (b, d) its column multi-index —
    # exactly the layout a (d², d²) matrix on vec(ρ) needs.  ``einsum("...kab,...kcd->...acbd")``
    # is the same tensor product but additionally summed over the jump index k.

    # No-jump rate operator  G = Σ_k L_k† L_k  (Hermitian):
    #   G_{ab} = Σ_{k,c} conj(L_k)_{ca} (L_k)_{cb}
    G = jnp.einsum("...kca,...kcb->...ab", jnp.conj(L), L)
    # Jump (dissipative gain) term  Σ_k L_k ρ L_k†  — tensor product Σ_k conj(L_k) ⊗ L_k:
    #   sandwich_{acbd} = Σ_k conj(L_k)_{ab} (L_k)_{cd}
    sandwich = jnp.einsum("...kab,...kcd->...acbd", jnp.conj(L), L)
    # Two δ-structured halves of the anticommutator  −½{G, ρ} = −½(G ρ + ρ G), each an I ⊗ G tensor product:
    #   G_rho_{acbd} = δ_{ab} G_{cd}          (the G ρ half)
    G_rho = jnp.einsum("ab,...cd->...acbd", I, G)
    #   rho_G_{acbd} = conj(G)_{ab} δ_{cd}    (the ρ G half; G Hermitian ⇒ conj(G)_{ab} = G_{ba})
    rho_G = jnp.einsum("...ab,cd->...acbd", jnp.conj(G), I)
    # Dissipator  D[ρ] = Σ_k ( L_k ρ L_k† − ½{L_k† L_k, ρ} )
    gen_data = sandwich - 0.5 * G_rho - 0.5 * rho_G

    if hamiltonian is not None:
        H = hamiltonian.matrix
        # Two δ-structured halves of the commutator  −i[H, ρ] = −i(H ρ − ρ H), each an I ⊗ H tensor product:
        #   H_rho_{acbd} = δ_{ab} H_{cd}          (the H ρ half)
        H_rho = jnp.einsum("ab,...cd->...acbd", I, H)
        #   rho_H_{acbd} = conj(H)_{ab} δ_{cd}    (the ρ H half; H Hermitian ⇒ conj(H)_{ab} = H_{ba})
        rho_H = jnp.einsum("...ab,cd->...acbd", jnp.conj(H), I)
        gen_data = gen_data + (-1j * (H_rho - rho_H))

    return gen_data.reshape(ensemble_shape + (d * d, d * d))
